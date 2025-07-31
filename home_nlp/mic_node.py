"""
mic_node.py

A ROS 2 node that publishes audio data from a microphone device.

This node captures audio chunks from a microphone using the sounddevice
library and publishes them as `home_interfaces/msg/Audio` messages
on the "audio" topic. The microphone stream can be toggled on/off
via the "toggle_mic" service.

Dependencies:
    - rclpy
    - sounddevice
    - numpy
    - home_interfaces.msg.Audio
    - std_srvs.srv.SetBool

Author: Enfu Liao
Date: 2025-06-09
"""

# TODO 改成 LifecycleNode
# TODO 處理 channel > 1 的情況
# TODO QoS profile
# TODO block_duration 也許可以改用 service 設定

import rclpy
from rclpy.node import Node
from typing import List
import sounddevice as sd
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Header
from std_srvs.srv import SetBool
from home_interfaces.msg import Audio
from rcl_interfaces.msg import ParameterDescriptor


class MicrophoneNode(Node):
    def __init__(self):
        super().__init__("mic_node")

        self.declare_parameter("sample_rate", 48000)
        self.declare_parameter("block_duration", 1.0)
        self.declare_parameter("num_channel", 1)

        # device 可以使用 device id 也可以使用 device name
        # USB Composite Device
        # HyperX SoloCast
        # DJI MIC MINI
        self.declare_parameter("device", descriptor=ParameterDescriptor(dynamic_typing=True))

        self.configure()
        self.activate()

    def configure(self):
        self.get_logger().info(f"Configuring...")

        sample_rate = self.get_parameter("sample_rate").value
        num_channel = self.get_parameter("num_channel").value
        block_duration = self.get_parameter("block_duration").value
        self.block_size = int(sample_rate * block_duration)

        device = self.get_parameter("device").value
        if device is not None and not isinstance(device, (int, str)):
            raise NotImplementedError

        # Initialize stream (sounddevice)
        self.stream = sd.InputStream(
            dtype="float32",
            samplerate=sample_rate,
            channels=num_channel,
            blocksize=self.block_size,
            device=device,
        )

        # Setup publisher / service / timer
        self.pub = self.create_publisher(Audio, "audio", 10)
        self.srv = self.create_service(SetBool, "toggle_mic", self.toggle_mic_cb)
        _ = self.create_timer(block_duration, self.timer_cb)

        # Log information
        self.get_logger().info(
            f"Configured: sample_rate={sample_rate}, "
            f"num_channel={num_channel}, block_size={self.block_size}, device={device}"
        )

    def activate(self):
        self.get_logger().info(f"Activating...")
        self.toggle_mic(True)
        self.get_logger().info(f"Activated")

    def toggle_mic(self, enable: bool):
        if not self.stream:
            return

        if enable and self.stream.stopped:

            self.get_logger().info("Starting microphone stream...")
            self.stream.start()

        if not enable and self.stream.active:
            self.get_logger().info("Stopping microphone stream...")
            self.stream.stop()

    def toggle_mic_cb(self, request, response):
        self.toggle_mic(request.data)
        response.success = True
        response.message = "Microphone started" if request.data else "Microphone stopped"
        return response

    def timer_cb(self):
        if not self.stream:
            return

        if self.stream.stopped:
            return

        audio_chunk, overflowed = self.stream.read(self.block_size)
        if overflowed:
            self.get_logger().warn("Input stream overflowed")
            return

        num_frame = audio_chunk.shape[0]
        num_channel = audio_chunk.shape[1]

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

        self.pub.publish(msg)
        self.get_logger().debug(f"Published audio chunk of size {len(audio_chunk)}")

    def destroy_node(self):

        if self.stream:
            self.stream.stop()
            self.stream.close()

        super().destroy_node()


def list_devices():
    print(sd.query_devices())


def main(args: List[str] | None = None):

    rclpy.init(args=args)
    node = MicrophoneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    list_devices()
    main()
