"""
asr_node.py

A ROS 2 node that performs automatic speech recognition (ASR) using Whisper.

This node receives audio data from the "audio" topic (published as
`home_interfaces/msg/Audio` messages), resamples it to 16 kHz mono,
and transcribes it into text using a streaming ASR model based on
Faster-Whisper. The transcribed text is published to the "transcription" topic
as `std_msgs/msg/String`.

The transcription is buffered and finalized based on silence detection and
custom rules (e.g., banlist filtering). The ASR backend is initialized and
warmed up at startup to reduce initial inference latency.

Dependencies:
    - rclpy
    - numpy
    - scipy
    - queue
    - home_interfaces.msg.Audio
    - std_msgs.msg.String
    - home_nlp.whisper_online.FasterWhisperASR
    - home_nlp.whisper_online.OnlineASRProcessor

Author: Enfu Liao
Date: 2025-06-10
"""

import rclpy
import queue
import numpy as np
import math

from scipy.signal import resample
from rclpy.node import Node
from typing import List
from home_nlp.whisper_online import FasterWhisperASR, OnlineASRProcessor
from home_nlp.ban import banlist
from home_interfaces.msg import Audio
from std_msgs.msg import String

# TODO 改成 LifecycleNode
# TODO QoS profile
# TODO 處理 channel > 1 的情況 (audio_cb) => 現在是指處理單 channel
# TODO Warmup 真的有效嗎?
# TODO 是否要加上 Header? 如果要，是要用開頭還是結尾？ => 先不加好了，反正沒用到

class AutomaticSpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__("asr_node")

        self.declare_parameter("language", "zh")
        self.declare_parameter("model", "large-v2")
        self.declare_parameter("sample_rate", 48000)
        self.declare_parameter("block_duration", 1.0)

        # TODO
        # 目前　period 代表每隔多久會執行一次 timer callback
        # 可能有一點混淆
        self.declare_parameter("period", 1.0)  
        self.declare_parameter("max_empty_count", 0)

        self.audio_queue = queue.Queue()
        self.sentence_queue = queue.Queue()

        self.empty_count = 0
        self.sentence_buffer = []

        self.configure()
        self.activate()

    def configure(self):
        self.get_logger().info(f"Configuring...")

        self.language = self.get_parameter("language").value
        self.model = self.get_parameter("model").value
        self.sample_rate = self.get_parameter("sample_rate").value
        self.block_duration = self.get_parameter("block_duration").value
        self.max_empty_count = self.get_parameter("max_empty_count").value

        _ = self.create_subscription(Audio, "audio", self.audio_cb, 10)
        self.pub = self.create_publisher(String, "transcription", 10)
        _ = self.create_timer(
            self.get_parameter("period").value,
            self.timer_cb,
        )

        self.get_logger().info(f"Configured")

    def activate(self):
        self.get_logger().info(f"Activating...")

        # Automatic Speech Recognition
        asr = FasterWhisperASR(self.language, self.model)
        self.online_asr = OnlineASRProcessor(asr)
        self.online_asr.init()

        # Warmup
        dummy_audio = np.random.randn(int(5.0 * OnlineASRProcessor.SAMPLING_RATE)).astype(np.float32) * 0.01
        self.online_asr.insert_audio_chunk(dummy_audio)
        _ = self.online_asr.process_iter()

        self.get_logger().info(f"Activated")

    def audio_cb(self, msg: Audio):
        self.get_logger().debug(f"Received Audio Chunk!")
        try:
            num_frames = msg.data.layout.dim[0].size  # type: ignore
            num_channels = msg.data.layout.dim[1].size  # type: ignore
            audio = np.array(msg.data.data, dtype=np.float32).reshape((num_frames, num_channels))
            mono_audio = audio[:, 0]
            self.audio_queue.put_nowait(mono_audio.copy())
        except queue.Full:
            self.get_logger().warn("Audio queue full, dropping chunk.")

    def timer_cb(self):

        # Get audio chunks
        chunks = []

        min_qsize = math.ceil(5.0 / self.block_duration)

        if self.audio_queue.qsize() < min_qsize:
            # 如果目前 audio queue 裡面的資料太少就先跳過
            return

        # 拿取目前 audio queue 裡面的所有音訊，連接成一整塊
        size = 0
        while size < min_qsize:
            chunk = self.audio_queue.get_nowait()
            chunks.append(chunk)
            size += 1
        audio = np.concatenate(chunks)

        # Resample
        target_len = int(len(audio) * OnlineASRProcessor.SAMPLING_RATE / self.sample_rate)
        resampled_audio = resample(audio, target_len)

        # ASR
        self.online_asr.insert_audio_chunk(resampled_audio.astype(np.float32))
        result = self.online_asr.process_iter()

        # Handle result
        if result is None:
            return

        if result[2].strip() == "":

            # result=(None, None, '')
            self.empty_count += 1

            if self.sentence_buffer and self.empty_count >= self.max_empty_count:
                full_sentence = "".join(self.sentence_buffer)
                self.sentence_buffer.clear()
                self.empty_count = 0
                self.get_logger().debug(f"{full_sentence=}")

                if not any(ban in full_sentence for ban in banlist):
                    # Publish
                    msg = String()
                    msg.data = full_sentence
                    self.pub.publish(msg)

        else:
            self.sentence_buffer.append(result[2])
            self.empty_count = 0


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
