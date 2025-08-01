"""
mic_node.py

A ROS 2 node that publishes audio data from a microphone device.

This node captures audio chunks from a microphone using the sounddevice
library and publishes them as `home_interfaces/msg/Audio` messages
on the "audio" topic. The microphone stream can be toggled on/off
via the "toggle_mic" service.

Dependencies:
    - sounddevice==0.5.2

Author: Enfu Liao
Date: 2025-06-09
"""

# TODO 處理 channel > 1 的情況
# TODO QoS profile
# TODO block_duration 也許可以改用 service 設定

from typing import List, Optional

import rclpy
import sounddevice as sd
from home_interfaces.msg import Audio
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from std_msgs.msg import Float32MultiArray, Header, MultiArrayDimension


class MicrophoneNode(LifecycleNode):
    def __init__(self):
        super().__init__("mic_node")

        self.declare_parameter("sample_rate", 48000)
        self.declare_parameter("block_duration", 1.0)
        self.declare_parameter("num_channel", 1)

        # device 可以使用 device id 也可以使用 device name
        # USB Composite Device
        # HyperX SoloCast
        # DJI MIC MINI
        self.declare_parameter(
            "device", descriptor=ParameterDescriptor(dynamic_typing=True)
        )

        self.stream: Optional[sd.InputStream] = None
        self.timer = None
        self.block_duration = None
        self.block_size = None
        self.device = None
        self.pub = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring microphone...")

        try:
            sample_rate = int(self.get_parameter("sample_rate").value)  # type: ignore
            num_channel = int(self.get_parameter("num_channel").value)  # type: ignore
            self.block_duration = float(self.get_parameter("block_duration").value)  # type: ignore
            self.block_size = int(sample_rate * self.block_duration)
            self.device = self.get_parameter("device").value
        except Exception as e:
            self.get_logger().error(f"Parameter error: {e}")
            return TransitionCallbackReturn.FAILURE

        self.device = self.get_parameter("device").value
        if self.device is not None and not isinstance(self.device, (int, str)):
            self.get_logger().error("Invalid device parameter (must be int or str).")
            return TransitionCallbackReturn.FAILURE

        try:
            self.stream = sd.InputStream(
                dtype="float32",
                samplerate=sample_rate,
                channels=num_channel,
                blocksize=self.block_size,
                device=self.device,
            )
        except Exception as e:
            self.get_logger().error(f"Failed to create InputStream: {e}")
            return TransitionCallbackReturn.FAILURE

        # Setup publisher / service / timer
        self.pub = self.create_publisher(Audio, "audio", 10)

        # Log information
        self.get_logger().info(
            f"Configured microphone: sample_rate={sample_rate}, "
            f"num_channel={num_channel}, block_size={self.block_size}, device={self.device}"
        )

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Activating microphone...")
        try:
            if self.stream and self.stream.stopped:
                self.stream.start()
            self.timer = self.create_timer(self.block_duration, self.timer_cb)  # type: ignore
        except Exception as e:
            self.get_logger().error(f"Activation failed: {e}")
            return TransitionCallbackReturn.FAILURE
        self.get_logger().info("Activated")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating microphone...")
        if self.timer:
            self.timer.cancel()
        if self.stream and self.stream.active:
            self.stream.stop()
        self.get_logger().info("Deactivated")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up microphone node...")
        if self.stream:
            try:
                if self.stream.active:
                    self.stream.stop()
                self.stream.close()
            except Exception as e:
                self.get_logger().warn(f"Error while closing stream: {e}")
        self.stream = None
        self.get_logger().info("Cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down microphone node...")
        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
        self.get_logger().info("Shutten up")
        return TransitionCallbackReturn.SUCCESS

    def timer_cb(self):
        if not self.stream or self.stream.stopped:
            return

        try:
            audio_chunk, overflowed = self.stream.read(self.block_size)
        except Exception as e:
            self.get_logger().error(f"Audio read failed: {e}")
            return

        if overflowed:
            self.get_logger().warn("Input stream overflowed")
            return

        num_frame, num_channel = audio_chunk.shape

        msg = Audio()

        # Set header
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "gpu"

        # Pack audio data
        array_data = Float32MultiArray()
        array_data.data = audio_chunk.flatten().tolist()

        array_data.layout.dim.append(MultiArrayDimension())  # type: ignore
        array_data.layout.dim[0].label = "frames"  # type: ignore
        array_data.layout.dim[0].size = num_frame  # type: ignore
        array_data.layout.dim[0].stride = num_frame * num_channel  # type: ignore

        array_data.layout.dim.append(MultiArrayDimension())  # type: ignore
        array_data.layout.dim[1].label = "channels"  # type: ignore
        array_data.layout.dim[1].size = num_channel  # type: ignore
        array_data.layout.dim[1].stride = num_channel  # type: ignore

        msg.data = array_data

        if self.pub:
            self.pub.publish(msg)
            self.get_logger().debug(f"Published audio chunk of size {len(audio_chunk)}")


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
