# Copyright (c) 2025, Enfu Liao <efliao@cs.nycu.edu.tw>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys

import numpy as np

from .asr_base import ASRBase


class HypothesisBuffer:
    MAX_NUM_REPEAT = 5  # 最多檢查幾個 word 是否重複

    def __init__(self):
        self.comm_buffer = []

        self.prev_buffer = []  # 之前 transcribe 的結果
        self.curr_buffer = []  # 目前 transcribe 的結果

        self.last_committed_e = 0
        self.last_committed_word = None

    def insert(self, buffer, offset=0.0):
        # 絕對時間
        buffer = [(s + offset, e + offset, text) for s, e, text in buffer]

        # 過濾掉已經 commit 過的部分
        self.curr_buffer = [
            (s, e, text) for s, e, text in buffer if s > self.last_committed_e - 0.1
        ]

        # 去除重複 (comm_buffer 的尾 vs curr_buffer 的頭)
        if len(self.curr_buffer) > 0:
            s, e, text = self.curr_buffer[0]

            if abs(s - self.last_committed_e) < 1.0:
                if self.comm_buffer:
                    n_comm = len(self.comm_buffer)
                    n_curr = len(self.curr_buffer)

                    for i in range(
                        1, min(min(n_comm, n_curr), self.MAX_NUM_REPEAT) + 1
                    ):
                        comm_tail = " ".join(
                            [self.comm_buffer[-j][2] for j in range(1, i + 1)][::-1]
                        )
                        curr_head = " ".join(
                            self.curr_buffer[j - 1][2] for j in range(1, i + 1)
                        )

                        if comm_tail == curr_head:
                            for j in range(i):
                                self.curr_buffer.pop(0)
                            break

    def flush(self):
        commit = []
        while self.curr_buffer:
            if len(self.prev_buffer) == 0:
                break

            s, e, text = self.curr_buffer[0]

            if text == self.prev_buffer[0][2]:
                commit.append((s, e, text))
                self.last_committed_text = text
                self.last_committed_e = e
                self.prev_buffer.pop(0)
                self.curr_buffer.pop(0)
            else:
                break

        self.prev_buffer = self.curr_buffer
        self.curr_buffer = []
        self.comm_buffer.extend(commit)

        return commit

    def trim(self, time):
        # 早於 time 的都刪掉
        while self.comm_buffer and self.comm_buffer[0][1] <= time:
            self.comm_buffer.pop(0)

    def get_pending(self):
        return self.prev_buffer

    def get_prompt(self, size=200):
        committed_texts = [t for _, _, t in self.comm_buffer]

        # 限制 200 個字, 但是避免切到詞
        p = []
        l = 0
        while committed_texts and l < size:
            x = committed_texts.pop(-1)
            l += len(x) + 1  # +1 for space
            p.append(x)
        return "".join(p[::-1])


class OnlineASRProcessor:
    SAMPLING_RATE = 16000

    def __init__(self, asr: ASRBase):
        self.asr = asr
        self.offset = 0.0

        self.trim_method = "segment"
        self.trim_time = 15  # seconds

        # Initialize ASR model
        self.a_buffer = np.array([], dtype=np.float32)
        self.t_buffer = HypothesisBuffer()

    def insert_audio(self, audio):
        self.a_buffer = np.append(self.a_buffer, audio)

    def process(self, use_prompt=True, context_prompt=""):
        # TODO[Enfu] context_prompt

        prompt = self.t_buffer.get_prompt() if use_prompt else None

        asr_result = self.asr.transcribe(self.a_buffer, prompt=prompt)
        self.t_buffer.insert(asr_result)
        committed_result = self.t_buffer.flush()

        if len(self.a_buffer) / self.SAMPLING_RATE > self.trim_time:
            # TODO[Enfu] implement audio buffer trimming
            pass

        return self.merge(committed_result)

    def trim(self, time):  # TODO[Enfu]
        self.t_buffer.trim(time)

        interval = time - self.offset
        self.a_buffer = self.a_buffer[int(interval * self.SAMPLING_RATE) :]

        self.offset = time

    def finish(self):
        pending = self.t_buffer.get_pending()
        self.offset += len(self.a_buffer) / 16000
        return self.merge(pending)

    def merge(self, buffer, offset=0.0):
        if len(buffer) == 0:
            return (None, None, "")

        t = " ".join(text for _, _, text in buffer)
        s = offset + buffer[0][0]
        e = offset + buffer[-1][1]
        return (s, e, t)
