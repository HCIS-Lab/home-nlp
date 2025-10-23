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

import json
import threading
from typing import List, Optional

import numpy as np
import rclpy
import websocket
from rclpy.lifecycle import (
    LifecycleNode,
    LifecycleState,
    TransitionCallbackReturn,
)
from rclpy.publisher import Publisher
from rclpy.timer import Timer
from std_msgs.msg import Int16MultiArray, String
from websocket import WebSocketApp

from .ws_asr import (
    DeepgramApiWebsocketASR,
    OpenaiApiWebsocketASR,
    WebsocketASRBase,
)


class WebsocketAutomaticSpeechRecognitionNode(LifecycleNode):
    def __init__(self):
        super().__init__("ws_asr_node")

        self.declare_parameter("provider", "deepgram")
        self.declare_parameter("model", "nova-2")
        self.declare_parameter("language", "zh-TW")
        self.declare_parameter("sample_rate", 48000)  # microphone

        self.provider: Optional[str] = None
        self.model: Optional[str] = None
        self.language: Optional[str] = None
        self.sample_rate: Optional[int] = None
        self.asr_cycle_time: Optional[float] = None

        self.pub: Optional[Publisher] = None
        self.ws: Optional[WebSocketApp] = None
        self.ws_asr: Optional[WebsocketASRBase] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.keepalive_timer: Optional[Timer] = None
        self.audio_sub: Optional[rclpy.subscription.Subscription] = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring")

        try:
            self.provider = str(self.get_parameter("provider").value)
            self.model = str(self.get_parameter("model").value)
            self.language = str(self.get_parameter("language").value)
            self.sample_rate = int(self.get_parameter("sample_rate").value)
        except Exception as e:
            self.get_logger().error(f"get_parameter() error: {e}")
            return TransitionCallbackReturn.FAILURE

        # create a websocket and run forever
        if self.provider == "openai":
            self.ws_asr = OpenaiApiWebsocketASR(
                model=self.model,
                language=self.language,
                mic_sample_rate=self.sample_rate,
                publish_fn=self.publish,
                logger=self.get_logger(),
            )
        elif self.provider == "deepgram":
            self.ws_asr = DeepgramApiWebsocketASR(
                model=self.model,
                language=self.language,
                mic_sample_rate=self.sample_rate,
                publish_fn=self.publish,
                logger=self.get_logger(),
            )
        else:
            self.get_logger().error(
                f"Unsupported API provider: {self.provider}",
            )
            return TransitionCallbackReturn.FAILURE

        self.ws = WebSocketApp(
            self.ws_asr.url,
            header=self.ws_asr.header,
            on_open=self.ws_asr.on_open,
            on_message=self.ws_asr.on_message,
            on_error=self.ws_asr.on_error,
            on_close=self.ws_asr.on_close,
        )
        self.ws_thread = threading.Thread(
            target=lambda: self.ws.run_forever(),
            daemon=True,
        )

        self.pub = self.create_publisher(String, "transcription", 10)

        self.get_logger().info(f"model={self.model}")
        self.get_logger().info(f"language={self.language}")
        self.get_logger().info("Configured")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Activating")

        if not self.ws_thread.is_alive():
            self.ws_thread.start()

        self.audio_sub = self.create_subscription(
            Int16MultiArray,
            "audio",
            self.audio_cb,  # receive audio and send to websocket
            10,
        )

        self.keepalive_timer = self.create_timer(
            4.0,
            self.keepalive_cb,
        )

        self.get_logger().info("Activated")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating")

        # Stop keepalive timer
        if self.keepalive_timer:
            self.keepalive_timer.cancel()
            self.keepalive_timer = None
            self.get_logger().info("Keepalive timer canceled")

        # Stop audio subscription
        if self.audio_sub:
            self.destroy_subscription(self.audio_sub)
            self.audio_sub = None
            self.get_logger().info("Audio subscription destroyed")

        self.get_logger().info("Deactivated")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up")
        self._release_resources()
        self.get_logger().info("Cleanup complete")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down")
        self._release_resources()
        self.get_logger().info("Shutdown complete")
        return TransitionCallbackReturn.SUCCESS

    def audio_cb(self, msg: Int16MultiArray):
        self.get_logger().debug("Received audio chunk")

        if not self.ws_asr.connected:
            self.get_logger().warn("Websocket not connected, throw audio")
            return

        audio = np.array(msg.data, dtype=np.int16)
        audio = self.ws_asr.build_data(audio)
        if isinstance(audio, bytes):
            self.ws.send(audio, opcode=websocket.ABNF.OPCODE_BINARY)
        else:
            self.ws.send(audio)

        self.get_logger().debug("audio sent")

    def publish(self, transcript):
        msg = String()
        msg.data = transcript
        self.pub.publish(msg)

    def keepalive_cb(self):
        if self.ws_asr and self.ws_asr.connected:
            self.ws.send(json.dumps({"type": "KeepAlive"}))
            self.get_logger().debug("keepalive sent")

    def _release_resources(self):
        if self.keepalive_timer:
            self.keepalive_timer.cancel()
            self.keepalive_timer = None

        if self.audio_sub:
            self.destroy_subscription(self.audio_sub)
            self.audio_sub = None

        # Stop websocket
        if self.ws:
            self.ws.close()
            self.get_logger().info("Websocket closed")
            self.ws = None

        if self.ws_thread and self.ws_thread.is_alive():
            self.get_logger().info("Joining websocket thread...")
            self.ws_thread.join(timeout=2.0)
            if self.ws_thread.is_alive():
                self.get_logger().warn("Websocket thread did not join in time.")
            else:
                self.get_logger().info("Websocket thread joined")
        self.ws_thread = None

        # Destroy publisher
        if self.pub:
            self.destroy_publisher(self.pub)
            self.get_logger().info("Publisher destroyed")
            self.pub = None

        # Clear other resources
        self.ws_asr = None
        self.provider = None
        self.model = None
        self.language = None
        self.sample_rate = None


def main(args: List[str] | None = None):
    rclpy.init(args=args)
    node = WebsocketAutomaticSpeechRecognitionNode()
    try:
        node.trigger_configure()
        node.trigger_activate()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
