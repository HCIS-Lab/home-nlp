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

from typing import List, Optional

import rclpy
import sounddevice as sd
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.lifecycle import (
    LifecycleNode,
    LifecycleState,
    TransitionCallbackReturn,
)
from rclpy.publisher import Publisher
from rclpy.timer import Timer
from std_msgs.msg import Int16MultiArray, MultiArrayDimension


class MicrophoneNode(LifecycleNode):
    def __init__(self):
        super().__init__("mic_node")

        self.declare_parameter("sample_rate", 48000)
        self.declare_parameter("block_duration", 1.0)
        self.declare_parameter(
            "device", descriptor=ParameterDescriptor(dynamic_typing=True)
        )

        self.block_duration: Optional[float] = None
        self.block_size: Optional[int] = None
        self.device: Optional[str | int] = None

        self.timer: Optional[Timer] = None
        self.pub: Optional[Publisher] = None
        self.stream: Optional[sd.InputStream] = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring")

        try:
            sample_rate = int(self.get_parameter("sample_rate").value)
            self.block_duration = float(
                self.get_parameter("block_duration").value,
            )
            self.block_size = int(sample_rate * self.block_duration)
            self.device = self.get_parameter("device").value
        except Exception as e:
            self.get_logger().error(f"get_parameter() error: {e}")
            return TransitionCallbackReturn.FAILURE

        self.device = self.get_parameter("device").value
        if self.device is not None and not isinstance(self.device, (int, str)):
            self.get_logger().error(
                "Invalid device parameter (must be int or str).",
            )
            return TransitionCallbackReturn.FAILURE

        try:
            self.stream = sd.InputStream(
                # fixed
                dtype="int16",
                channels=1,
                # parameter
                samplerate=sample_rate,
                blocksize=self.block_size,
                device=self.device,
            )
        except Exception as e:
            self.get_logger().error(f"Failed to create InputStream: {e}")
            return TransitionCallbackReturn.FAILURE

        self.pub = self.create_publisher(Int16MultiArray, "audio", 10)

        self.get_logger().info(f"sample_rate={sample_rate}")
        self.get_logger().info(f"block_size={self.block_size}")
        self.get_logger().info(f"device={self.device}")
        self.get_logger().info("Configured")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Activating")

        try:
            if self.stream and self.stream.stopped:
                self.stream.start()
        except Exception as e:
            self.get_logger().error(f"Failed to start InputStream failed: {e}")
            return TransitionCallbackReturn.FAILURE

        self.timer = self.create_timer(self.block_duration, self.timer_cb)

        self.get_logger().info("Activated")
        return TransitionCallbackReturn.SUCCESS

    def timer_cb(self):
        if not self.stream or self.stream.stopped:
            self.get_logger().debug("No sounddevice stream")
            return

        try:
            audio_chunk, overflowed = self.stream.read(self.block_size)
        except Exception as e:
            self.get_logger().error(f"Audio read failed: {e}")
            return

        if overflowed:
            self.get_logger().warn("Input stream overflowed")
            return

        self.get_logger().debug("Get one audio chunk from sounddevice stream")

        num_frames, num_channels = audio_chunk.shape

        msg = Int16MultiArray()
        msg.data = audio_chunk.flatten().tolist()
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].label = "frames"
        msg.layout.dim[0].size = num_frames
        msg.layout.dim[0].stride = num_frames
        msg.layout.data_offset = 0

        if self.pub:
            self.pub.publish(msg)
            self.get_logger().debug(
                f"Published audio chunk of size {len(audio_chunk)}",
            )

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating")

        if self.timer:
            self.timer.cancel()

        if self.stream and self.stream.active:
            self.stream.stop()

        self.get_logger().info("Deactivated")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up")
        self._close_stream()
        self.get_logger().info("Cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down")
        self._close_stream()
        self.get_logger().info("Shut down")
        return TransitionCallbackReturn.SUCCESS

    def _close_stream(self) -> None:
        if not self.stream:
            return

        try:
            if self.stream.active:
                self.stream.stop()
            self.stream.close()
        except Exception as e:
            self.get_logger().warning(f"Error closing stream: {e}")
        finally:
            self.stream = None


def list_devices():
    print(sd.query_devices())


def main(args: List[str] | None = None):
    rclpy.init(args=args)
    node = MicrophoneNode()
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
    list_devices()
    main()
