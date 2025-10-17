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

import getpass
import os
import queue
from typing import List

import rclpy
import torch
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_openai import ChatOpenAI
from rclpy.lifecycle import (
    LifecycleNode,
    LifecycleState,
    TransitionCallbackReturn,
)
from std_msgs.msg import String
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .prompt_template import example_prompt, examples, system_prompt


class LargeLanguageModelNode(LifecycleNode):
    def __init__(self):
        super().__init__("llm_node")

        self.declare_parameter("model", "google/gemma-3-1b-it")
        self.declare_parameter("temperature", 0.0)

        self.pub = None
        self.sub = None
        self.timer = None
        self.chat_with_history = None
        self.chat_map = {}

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring")

        try:
            self.model_name = self.get_parameter("model").value
        except Exception as e:
            self.get_logger().error(f"get_parameter() error: {e}")
            return TransitionCallbackReturn.FAILURE

        load_dotenv()

        if self.model_name == "gpt-4o-mini" or self.model_name == "gpt-4o":
            chat_model = self._get_openai_chat_model()
        else:
            chat_model = self._get_hf_chat_model()

        if chat_model is None:
            return TransitionCallbackReturn.FAILURE

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

        self.pub = self.create_publisher(String, "llm_output", 10)

        self.get_logger().info(f"model={self.model_name}")
        self.get_logger().info("Configured")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Activating")

        self.sub = self.create_subscription(
            # TODO[Enfu]
            # String, "llm_input", self.llm_input_cb, 10
            String,
            "transcription",
            self.llm_input_cb,
            10,
        )

        self.get_logger().info("Activated")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating")

        if self.sub:
            self.destroy_subscription(self.sub)
            self.sub = None
            self.get_logger().info("Subscription destroyed.")

        if self.timer:
            self.timer.cancel()
            self.timer = None
            self.get_logger().info("Timer destroyed.")

        self.chat_map.clear()
        self.get_logger().info("Deactivated")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up")
        self.chat_with_history = None
        self.get_logger().info("Cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS

    def llm_input_cb(self, msg: String):
        input = msg.data
        self.get_logger().debug(f"User input: {input}")

        resp: str = self.chat_with_history.invoke(
            {"user_input": input},
            config={"session_id": "default"},
        )
        self.get_logger().debug(f"LLM output:\n{resp}")

        # TODO[Enfu] 更好的寫法?
        resp = resp.replace("{{", "{")
        resp = resp.replace("}}", "}")

        response_msg = String()
        response_msg.data = resp
        if self.pub:
            self.pub.publish(response_msg)

    def get_chat_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.chat_map:
            self.chat_map[session_id] = InMemoryChatMessageHistory()
        return self.chat_map[session_id]

    def _get_openai_chat_model(self):
        if not os.environ.get("OPENAI_API_KEY"):
            self.get_logger().error("Missing OPENAI_API_KEY")
            return None

        return ChatOpenAI(
            model=self.model_name,
            temperature=self.get_parameter("temperature").value,
            max_tokens=None,
            timeout=None,
            max_retries=3,
        )

    def _get_hf_chat_model(self):
        hf_token = os.getenv("HF_TOKEN")
        if hf_token is None:
            self.get_logger().error("Missing HF_TOKEN")
            return None

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
            temperature=self.get_parameter("temperature").value,
            top_p=0.95,
            repetition_penalty=1.1,
            return_full_text=False,
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        return ChatHuggingFace(llm=llm, model_id=self.model_name)


def main(args: List[str] | None = None):
    torch.set_float32_matmul_precision("high")

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
