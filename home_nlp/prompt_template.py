"""
prompt_template.py

Defines system and example prompts for the Large Language Model (LLM) node.

The system prompt instructs the LLM (acting as a home robot named Stretch) to
convert user instructions into executable XML Behavior Trees (BTs).
It enforces strict formatting rules to ensure valid and consistent XML output.

Few-shot examples are provided to guide the LLM in generating correct responses.

Author: Enfu Liao, Chinlu Chen
Date: 2025-08-01
"""

from langchain.prompts import ChatPromptTemplate

system_prompt = """You are the brain of a helpful and friendly home robot named Stretch. 

Your job is to convert user messages or instructions into XML Behavior Trees that represent executable robotic actions.

You will never harm a human or suggest harm.

**Rules**:
- Always respond with valid XML only. No extra explanation.
- Use the format `<root BTCPP_format="4">...</root>`.
- Use `Sequence` to represent ordered actions.
- Use meaningful action nodes including <Speak>, <Grasp>, <Navigate>, <Handover>
- Do not include comments, explanations, apologies, or uncertain answers.
- Generate only the raw XML content, without wrapping it in markdown or using triple backticks. No ```xml, no formatting — just the raw XML."""

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

examples = [
    {
        "input": "Hi!",
        "output": """<root BTCPP_format="4">
  <BehaviorTree ID="MainTree">
    <Sequence>
        <Speak text="Hello!"/>
    </Sequence>
  </BehaviorTree>
</root>""",
    },
    {
        "input": "Goodbye!",
        "output": """<root BTCPP_format="4">
  <BehaviorTree ID="MainTree">
    <Sequence>
        <Speak text="Goodbye!"/>
    </Sequence>
  </BehaviorTree>
</root>""",
    },
    {
        "input": "What is 2 + 2?",
        "output": """<root BTCPP_format="4">
  <BehaviorTree ID="MainTree">
    <Sequence>
        <Speak text="2 + 2 is 4."/>
    </Sequence>
  </BehaviorTree>
</root>""",
    },
    {
        "input": "Just stand still.",
        "output": """<root BTCPP_format="4">
  <BehaviorTree ID="MainTree">
    <Sequence>
    </Sequence>
  </BehaviorTree>
</root>""",
    },
    {
        "input": "Give me the remote control.",
        "output": """<root BTCPP_format="4">
  <BehaviorTree ID="MainTree">
    <Sequence>
        <Speak text="I am picking up the remote control and handing it over to you."/>
        <Navigate object="remote control"/>
        <Grasp object="remote control"/>
        <Navigate object="human"/>
        <Handover />
    </Sequence>
  </BehaviorTree>
</root>""",
    },
]
