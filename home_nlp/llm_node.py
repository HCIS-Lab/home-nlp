"""
llm_node.py

TODO

Author: Enfu Liao
Date: 2025-06-10
"""

import os
import rclpy
from rclpy.node import Node
from typing import List
from std_msgs.msg import String
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from dotenv import load_dotenv
from home_nlp.prompt_template import template
import re


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
        self.pub = self.create_publisher(String, "llm_response", 10)  # TODO rename plan

        self.get_logger().info(f"Configured")

    def activate(self):
        self.get_logger().info(f"Activating...")
        load_dotenv()  # TODO
        hf_token = os.getenv("HF_TOKEN")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=hf_token,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto",
            use_auth_token=hf_token,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            return_full_text=False,
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

        self.prompt_template = PromptTemplate(
            input_variables=["user_input"],
            template=template,  # TODO from independent file
        )

        self.get_logger().info(f"Activated")

    def cb(self, msg: String):
        formatted_prompt = self.prompt_template.format(user_input=msg.data)

        response = self.llm.invoke(formatted_prompt)
        self.get_logger().info(f"LLM raw output:\n{response}")
        parsed_response = self.output_parser.parse(response)

        response_msg = String()
        response_msg.data = parsed_response
        self.pub.publish(response_msg)


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
