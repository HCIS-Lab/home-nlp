# Copyright (c) 2025, Enfu Liao <efliao@cs.nycu.edu.tw>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
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

import base64
import json
import math
import os

import numpy as np
from dotenv import load_dotenv
from scipy import signal
from typing import Any

from .ws_asr_base import WebsocketASRBase


class OpenaiApiWebsocketASR(WebsocketASRBase):
    def __init__(self, model, language, mic_sample_rate, publish_fn, logger):
        super().__init__(model, language, mic_sample_rate, publish_fn, logger)

        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("Missing OPENAI_API_KEY")


    @property
    def sample_rate(self) -> int | None:
        return 24000

    @property
    def header(self) -> dict:
        return {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "OpenAI-Beta": "realtime=v1",
        }

    @property
    def url(self) -> str:
        ws_url = "wss://api.openai.com/v1/realtime?"
        ws_url += "intent=transcription"

        return ws_url

    def build_data(self, audio: np.ndarray) -> any:
        if self.sample_rate and self.mic_sample_rate != self.sample_rate:
            # resample (poly)
            input_rate = self.mic_sample_rate
            output_rate = self.sample_rate
            gcd = math.gcd(input_rate, output_rate)
            up = output_rate // gcd
            down = input_rate // gcd
            audio = signal.resample_poly(audio, up, down)

        # base64
        audio = audio.astype(np.int16).tobytes()
        audio = base64.b64encode(audio).decode("utf-8")

        # send
        return json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": audio,
            }
        )

    def on_open(self, ws):
        self.logger.info("websocket opened")

        config = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": self.model,
                    "language": self.language,
                },
            },
        }
        ws.send(json.dumps(config))

        self.connected = True

    def on_close(self, ws, status_code, msg):
        self.logger.info(f"{status_code}: {msg}")
        self.connected = False

    def on_message(self, ws, msg):
        try:
            event = json.loads(msg)
            et = event.get("type")

            if et == "transcription_session.created":
                self.logger.info("session created")

            elif et == "input_audio_buffer.speech_started":
                self.logger.debug("speech started")

            elif et == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "").strip()
                if not transcript:
                    return
                self.logger.info(f"Transcript: {transcript}")
                self.publish_fn(transcript)

            elif et == "error":
                self.logger.error(event.get("error", "unknown"))

        except Exception as e:
            self.logger.error(e)

    def on_error(self, ws, msg):
        self.logger.error(msg)
