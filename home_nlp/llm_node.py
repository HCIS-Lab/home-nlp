"""
llm_node.py

TODO

Author: Enfu Liao
Date: 2025-06-10
"""

# TODO maybe subscription only store the transcription to buffer, and use another timer callback to process data is better?

import os
import rclpy
from typing import List
from std_msgs.msg import String
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from home_nlp.prompt_template import system_prompt, example_prompt, examples
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState

import torch

torch.set_float32_matmul_precision("high")


class LargeLanguageModelNode(LifecycleNode):
    def __init__(self):
        super().__init__("llm_node")

        self.declare_parameter("model", "google/gemma-3-1b-it")
        # Qwen/Qwen2.5-1.5B-Instruct
        # meta-llama/Llama-3.2-1B-Instruct
        # meta-llama/Llama-3.1-8B-Instruct
        # google/gemma-3-1b-it
        # google/gemma-3-4b-it
        # deepseek-ai/deepseek-coder-6.7b-instruct
        # microsoft/Phi-4-mini-instruct
        # mistralai/Mistral-7B-Instruct-v0.3
        # meta-llama/Llama-3.1-8B-Instruct

        self.pub = None
        self.sub = None
        self.chat_with_history = None
        self.chat_map = {}

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring LLM Node...")

        try:
            self.model_name = self.get_parameter("model").value
            self.get_logger().info(f"Loading model: {self.model_name}")

            load_dotenv()
            hf_token = os.getenv("HF_TOKEN")
            if hf_token is None:
                self.get_logger().error("HF_TOKEN not found in environment.")
                return TransitionCallbackReturn.FAILURE

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=hf_token,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype="auto",
                token=hf_token,
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

            llm = HuggingFacePipeline(pipeline=pipe)
            chat_model = ChatHuggingFace(llm=llm, model_id=self.model_name)

            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=examples,
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    few_shot_prompt,
                    MessagesPlaceholder(variable_name="history"),
                    ("user", "{user_input}"),
                ]
            )

            chain = prompt_template | chat_model | StrOutputParser()
            self.chat_with_history = RunnableWithMessageHistory(
                chain,
                get_session_history=self.get_chat_history,
                input_messages_key="user_input",
                history_messages_key="history",
            )

            self.pub = self.create_publisher(String, "llm_response", 10)

            self.get_logger().info("Configured")
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"Configuration failed: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Activating LLM Node...")

        try:
            self.sub = self.create_subscription(String, "transcription", self.cb, 10)
            self.get_logger().info("Activated")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Activation failed: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating LLM Node...")

        if self.sub:
            self.destroy_subscription(self.sub)
            self.sub = None
            self.get_logger().info("Subscription destroyed.")

        self.chat_map.clear()
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up LLM Node...")
        self.chat_with_history = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down LLM Node...")
        return TransitionCallbackReturn.SUCCESS

    def get_chat_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.chat_map:
            self.chat_map[session_id] = InMemoryChatMessageHistory()
        return self.chat_map[session_id]

    def cb(self, msg: String):
        user_input = msg.data

        response = self.chat_with_history.invoke(
            {"user_input": user_input}, config={"session_id": "default"}
        )
        self.get_logger().info(f"LLM output:\n{response}")

        response_msg = String()
        response_msg.data = response
        if self.pub:
            self.pub.publish(response_msg)


def main(args: List[str] | None = None):
    rclpy.init(args=args)
    node = LargeLanguageModelNode()
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
