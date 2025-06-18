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
from langchain import PromptTemplate

# TODO 強制 GPU?


class LargeLanguageModelNode(Node):
    template = """You are a helpful and friendly home robot named Stretch. 

Your job is to convert user instructions into XML Behavior Trees that represent executable robotic actions.

You will never harm a human or suggest harm.

**Rules**:
- Always respond with valid XML only. No extra explanation.
- Use the format `<root BTCPP_format="4">...</root>`.
- Use `Sequence` to represent ordered actions.
- Use meaningful action nodes like <Speak>, <Grasp>, <Navigate>, <Handover>, etc.
- Do not include comments, apologies, or uncertain answers.

Below are some examples.

Input: "Hi!"
Output:
<root BTCPP_format="4">
  <BehaviorTree ID="MainTree">
    <Sequence>
        <Speak text="Hello!"/>
    </Sequence>
  </BehaviorTree>
</root>

Input: "Goodbye!"
Output:
<root BTCPP_format="4">
  <BehaviorTree ID="MainTree">
    <Sequence>
        <Speak text="Goodbye!"/>
    </Sequence>
  </BehaviorTree>
</root>

Input: "What is 2 + 2?"
Output:
<root BTCPP_format="4">
  <BehaviorTree ID="MainTree">
    <Sequence>
        <Speak text="2 + 2 is 4."/>
    </Sequence>
  </BehaviorTree>
</root>

Input: "Give me the remote control."
Output:
<root BTCPP_format="4">
  <BehaviorTree ID="MainTree">
    <Sequence>
        <Speak text="I am picking up the remote control and handing it over to you."/>
        <Grasp object="remote control"/>
        <Handover />
    </Sequence>
  </BehaviorTree>
</root>

Input: {user_input}
Output:"""

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

        self.prompt_template = PromptTemplate(
            input_variables=["user_input"],
            template=self.template,  # TODO from independent file
        )

        self.get_logger().info(f"Activated")

    def cb(self, msg: String):
        formatted_prompt = self.prompt_template.format(user_input=msg.data)

        response = self.llm.invoke(formatted_prompt)

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
