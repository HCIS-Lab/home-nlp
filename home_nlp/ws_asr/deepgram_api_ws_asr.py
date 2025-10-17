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

import os

import json
import numpy as np
from dotenv import load_dotenv
from typing import Any

from .ws_asr_base import WebsocketASRBase


class DeepgramApiWebsocketASR(WebsocketASRBase):
    def __init__(self, model, language, mic_sample_rate, publish_fn, logger):
        super().__init__(model, language, mic_sample_rate, publish_fn, logger)

        # https://developers.deepgram.com/docs/models-languages-overview

        load_dotenv()
        if not os.getenv("DEEPGRAM_API_KEY"):
            logger.error("Missing DEEPGRAM_API_KEY")

    @property
    def sample_rate(self) -> int | None:
        """
        None indicates that the sample rate can be customized per request.
        """
        return None

    @property
    def header(self) -> dict:
        return {
            "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}",
        }

    @property
    def url(self) -> str:
        ws_url = "wss://api.deepgram.com/v1/listen?"
        ws_url += "encoding=linear16"
        ws_url += "&punctuate=true"  # TODO[Enfu]
        ws_url += "&channels=1"
        ws_url += "&multichannel=false"
        ws_url += "&interim_results=false" # 中間過程結果  
        ws_url += f"&sample_rate={self.mic_sample_rate}"
        ws_url += f"&model={self.model}"
        ws_url += f"&language={self.language}"

        return ws_url

    def build_data(self, audio: np.ndarray) -> Any:
        return audio.tobytes()

    def on_message(self, ws, msg):

        self.logger.debug(msg)
        try:
            
            data = json.loads(msg)
            
            if data.get("type") == "Results":

                channel = data.get("channel", {})
                alternatives = channel.get("alternatives", [])
                
                if alternatives:
                    transcript = alternatives[0].get("transcript", "").strip()
                    confidence = alternatives[0].get("confidence", 0.0)
                    is_final = data.get("is_final", False)
                    speech_final = data.get("speech_final", False)
                    
                    if not transcript:
                        return
                    
                    self.logger.debug(f"Confidence: {confidence}")
                    self.logger.info(f"Transcript: {transcript}")

                    if speech_final:
                        self.publish_fn(transcript)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error: {e}")
        except Exception as e:
            self.logger.error(f"Error in on_message: {e}")
            self.logger.error(f"Message content: {msg}")

    def on_error(self, ws, msg):
        self.logger.error(msg)

    def on_close(self, ws, status_code, msg):
        self.logger.info(f"{status_code}: {msg}")
        self.connected = False

    def on_open(self, ws):
        self.logger.info("websocket opened")
        self.connected = True
