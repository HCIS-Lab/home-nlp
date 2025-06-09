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

import rclpy
from rclpy.node import Node
from typing import List
import sounddevice as sd
import numpy as np
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from std_srvs.srv import SetBool
from home_interfaces.msg import Audio


class MicrophoneNode(Node):
    def __init__(self):
        super().__init__("mic_node")

        self.sample_rate = 48000
        self.chunk_size = 1024
        self.device_id = 6
        self.channels = 1 # TODO from config

        self.pub = self.create_publisher(Audio, "audio", 10)
        self.srv = self.create_service(
            SetBool, "toggle_mic", self.toggle_mic_cb
        )

        self.timer = self.create_timer(
            self.chunk_size / self.sample_rate, self.timer_cb
        )

        self.stream = sd.InputStream(
            dtype="float32",
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.chunk_size,
            device=self.device_id,
        )
        self.toggle_mic(True)

    def toggle_mic(self, enable: bool):
        if not self.stream:
            return

        if enable and self.stream.stopped:
            self.get_logger().info("Starting mimicrophone stream...")
            # TODO show device name
            self.stream.start()

        if not enable and self.stream.active:
            self.get_logger().info("Stopping microphone stream...")
            self.stream.stop()

    def toggle_mic_cb(self, request, response):
        self.toggle_mic(request.data)
        response.success = True
        response.message = (
            "Microphone started" if request.data else "Microphone stopped"
        )
        return response

    def timer_cb(self):
        if not self.stream:
            return

        if self.stream.stopped:
            return

        audio_chunk, overflowed = self.stream.read(self.chunk_size)

        num_frame = audio_chunk.shape[0]
        num_channel = audio_chunk.shape[1]

        msg = Audio()

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
        msg.sample_rate = self.sample_rate

        self.pub.publish(msg)
        self.get_logger().debug(
            f"Published audio chunk of size {len(audio_chunk)}"
        )

    def destroy_node(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

        super().destroy_node()


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
    main()
