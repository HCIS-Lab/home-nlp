"""
llm_node.py

TODO

Author: Enfu Liao
Date: 2025-06-10
"""

import rclpy
from rclpy.node import Node
from typing import List
from std_msgs.msg import String
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline


# TODO 強制 GPU?


class LargeLanguageModelNode(Node):
    def __init__(self):
        super().__init__("llm_node")

        self.declare_parameter("model", "google/gemma-3-1b-it")
        # Qwen/Qwen2.5-1.5B-Instruct
        # meta-llama/Llama-3.2-1B-Instruct
        # google/gemma-3-1b-it

        self.configure()
        self.activate()

    def configure(self):
        self.get_logger().info(f"Configuring...")

        self.model_name = self.get_parameter("model").get_parameter_value().string_value

        _ = self.create_subscription(String, "transcription", self.cb, 10)
        self.pub = self.create_publisher(String, "llm_response", 10)

        self.get_logger().info(f"Configured")

    def activate(self):
        self.get_logger().info(f"Activating...")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto",
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

        self.get_logger().info(f"Activated")

    def cb(self, msg: String):
        response = self.llm.invoke(msg.data)
        self.get_logger().debug(response)

        msg = String()
        msg.data = response
        self.pub.publish(msg)


def main(args: List[str] | None = None):

    rclpy.init(args=args)
    node = LargeLanguageModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
