"""
asr_node.py

TODO

Author: Enfu Liao
Date: 2025-06-09
"""

import rclpy
from rclpy.node import Node
from typing import List
from home_nlp.whisper_online import FasterWhisperASR, OnlineASRProcessor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from home_interfaces.msg import Audio
from std_msgs.msg import String
import queue
import numpy as np
from scipy.signal import resample
TARGET_SR = 16000  # TODO constants.py


class AutomaticSpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__("asr_node")
        pass

        self.language = "zh"
        self.origin_sr = 48000  # TODO from topic
        self.num_channel = 1  # TODO from topic
        self.max_empty_count = 0
        self.block_duration = 1.0

        self.block_size = int(self.origin_sr * self.block_duration)

        self.audio_queue = queue.Queue()
        self.sentence_queue = queue.Queue()
        self.sentence_buffer = []
        self.empty_count = 0
        self.running = False

        self.banlist = [
            "不吝点赞",
            "转发",
            "订阅",
            "点点栏目",
            "點點欄目",
            "打赏支持明镜",
            "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目",
            "Amara.org",
            "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目",
            "社群提供",
            "点点欄目",
            "幕",
        ]

        # Automatic Speech Recognition
        self.asr = FasterWhisperASR(self.language, "large-v2")
        self.online_asr = OnlineASRProcessor(self.asr)
        self.online_asr.init()

        self.create_subscription(
            Audio,
            "audio",
            self.audio_cb,
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            ),
        )

        self.pub = self.create_publisher(String, "transcription", 10)
        self.timer = self.create_timer(0.1, self.timer_cb)  # 10Hz

        self.get_logger().info("start")


    def audio_cb(self, msg: Audio):
        # self.get_logger().info("audio_cb")
        try:
            num_frames = msg.data.layout.dim[0].size
            num_channels = msg.data.layout.dim[1].size
            audio = np.array(msg.data.data, dtype=np.float32).reshape(
                (num_frames, num_channels)
            )

            # # TODO Use only first channel (mono)
            # mono_audio = audio[:, 0]

            # Enqueue audio chunk
            self.audio_queue.put_nowait(audio[:, 0].copy())

        except queue.Full:
            self.get_logger().warn("Audio queue full, dropping chunk.")

    def timer_cb(self):
        try:
            audio_chunk = self.audio_queue.get_nowait()
        except queue.Empty:
            self.get_logger().info("queue.Empty")
            return  # No audio to process

        # Resample
        num_target_samples = int(len(audio_chunk) * TARGET_SR / 48000)
        resampled = resample(audio_chunk, num_target_samples)

        # ASR
        self.online_asr.insert_audio_chunk(resampled.astype(np.float32))
        result = self.online_asr.process_iter()

        self.get_logger().info(f"{result=}")


        # Handle result
        if result is not None and result[2].strip() != "":
            self.sentence_buffer.append(result[2])
            self.empty_count = 0
        else:
            # TODO 應該在每個小東西就判斷 ban list
            self.empty_count += 1
            if self.empty_count >= self.max_empty_count and self.sentence_buffer:
                full_sentence = "".join(self.sentence_buffer)
                if not any(banned in full_sentence for banned in self.banlist):
                    self.publish_sentence(full_sentence)
                self.sentence_buffer.clear()
                self.empty_count = 0

    def publish_sentence(self, sentence: str):
        self.get_logger().info(f"[ASR] {sentence}")
        msg = String()
        msg.data = sentence
        self.pub.publish(msg)

    def destroy_node(self):
        # Final flush
        final = self.online_asr.finish()
        if final and final[2].strip():
            self.sentence_buffer.append(final[2])
        if self.sentence_buffer:
            full_sentence = "".join(self.sentence_buffer)
            if not any(
                banned in full_sentence for banned in self.banlist
            ):  # TODO encapsulation
                self.publish_sentence(full_sentence)

        super().destroy_node()


def main(args: List[str] | None = None):

    rclpy.init(args=args)
    node = AutomaticSpeechRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
