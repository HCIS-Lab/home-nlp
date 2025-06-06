import rclpy
from rclpy.node import Node
from typing import List
import sounddevice as sd
import numpy as np
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from std_srvs.srv import SetBool


class MicNode(Node):
    def __init__(self):
        super().__init__("mic_node")

        self.sample_rate = 48000
        self.chunk_size = 1024
        self.device_id = 1
        self.channels = 1

        self.pub = self.create_publisher(Float32MultiArray, "audio", 10)
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

        data, overflowed = self.stream.read(self.chunk_size)

        num_frame = data.shape[0]
        num_channel = data.shape[1]

        msg = Float32MultiArray()
        msg.data = data.flatten().tolist()

        msg.layout.dim.append(MultiArrayDimension())  # type: ignore
        msg.layout.dim[0].label = "frames"  # type: ignore
        msg.layout.dim[0].size = num_frame  # type: ignore
        msg.layout.dim[0].stride = num_frame * num_channel  # type: ignore

        msg.layout.dim.append(MultiArrayDimension())  # type: ignore
        msg.layout.dim[1].label = "channels"  # type: ignore
        msg.layout.dim[1].size = num_channel  # type: ignore
        msg.layout.dim[1].stride = num_channel  # type: ignore

        self.pub.publish(msg)
        self.get_logger().debug(f"Published audio chunk of size {len(data)}")

    def destroy_node(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

        super().destroy_node()


def main(args: List[str] | None = None):

    rclpy.init(args=args)
    node = MicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
