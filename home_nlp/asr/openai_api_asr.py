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

import wave
from io import BytesIO

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.audio import TranscriptionVerbose

from .asr_base import ASRBase


class OpenaiApiASR(ASRBase):
    """Uses OpenAI's API for audio transcription."""

    def __init__(self, model="whisper-1", language="en", temperature=0):
        super().__init__(
            model=model,
            language=language,
            temperature=temperature,
        )

        load_dotenv()
        self.client = OpenAI()
        # FIXME[Enfu]
        if self.model != "whisper-1":
            raise ValueError("Currently, only 'whisper-1' model is supported.")

    def transcribe(self, audio: np.ndarray, prompt=None):
        if audio.dtype != np.float32:
            raise ValueError("Audio data must be in float32 format.")

        buffer = BytesIO()
        buffer.name = "audio.wav"
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)  # 單聲道
            wf.setsampwidth(2)  # 改成 2 bytes = int16 = PCM 16-bit
            wf.setframerate(16000)

            # 轉換 float32 [-1, 1] 到 int16
            audio_int16 = (audio * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())

        buffer.seek(0)  # 重置緩衝區指標到開始位置

        kwargs = {}

        if prompt is not None:
            kwargs["prompt"] = prompt

        transcription = self.client.audio.transcriptions.create(
            model=self.model,
            file=buffer,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language=self.language,
            temperature=self.temperature,
            **kwargs,
        )

        results = []
        if transcription.words is not None:
            for word in transcription.words:
                results.append((word.start, word.end, word.word))

        return results
