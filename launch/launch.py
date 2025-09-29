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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    sample_rate_arg = DeclareLaunchArgument(
        "sample_rate",
        default_value="48000",
        description="Sample Rate for the microphone",
    )

    block_duration_arg = DeclareLaunchArgument(
        "block_duration", default_value="1.0", description="Block Duration in seconds"
    )

    min_processing_duration = DeclareLaunchArgument(
        "min_processing_duration",
        default_value="1.0",
        description="Minimum processing duration in seconds",
    )

    num_channel_arg = DeclareLaunchArgument(
        "num_channel", default_value="1", description="Number of audio channels"
    )

    device_arg = DeclareLaunchArgument(
        "device",
        default_value="DJI MIC MINI",
        description="Device ID or name for the microphone",
    )

    language_arg = DeclareLaunchArgument(
        "language",
        default_value="en",
        description="Language for the ASR (Whisper) model",
    )

    period_arg = DeclareLaunchArgument(
        "period",
        default_value="0.5",
        description="Period for the main loop in seconds",  # TODO rename & rewrite description
    )

    max_empty_count_arg = DeclareLaunchArgument(
        "max_empty_count",
        default_value="0",
        description="Maximum number of consecutive empty results to determine the end of a sentence",
    )

    # asr_model_arg = DeclareLaunchArgument(
    #     "asr_model", default_value="large-v2", description="ASR (Whisper) model"
    # )

    ws_asr_model_arg = DeclareLaunchArgument(
        "ws_asr_model", default_value="nova-3", description="ASR (Nova-3) model"
    )

    llm_model_arg = DeclareLaunchArgument(
        "llm_model",
        # default_value="google/gemma-3-1b-it",
        default_value="gpt-4o-mini",
        description="LLM model to use. Available options: "
        "Qwen/Qwen2.5-1.5B-Instruct, "
        "meta-llama/Llama-3.2-1B-Instruct, "
        "meta-llama/Llama-3.1-8B-Instruct, "
        "google/gemma-3-1b-it, "
        "google/gemma-3-4b-it, "
        "deepseek-ai/deepseek-coder-6.7b-instruct, "
        "microsoft/Phi-4-mini-instruct, "
        "mistralai/Mistral-7B-Instruct-v0.3, "
        "gpt-4o, "
        "gpt-4o-mini, ",
    )

    sample_rate = LaunchConfiguration("sample_rate")
    block_duration = LaunchConfiguration("block_duration")
    min_processing_duration = LaunchConfiguration("min_processing_duration")
    num_channel = LaunchConfiguration("num_channel")
    device = LaunchConfiguration("device")
    language = LaunchConfiguration("language")
    period = LaunchConfiguration("period")
    max_empty_count = LaunchConfiguration("max_empty_count")
    # asr_model = LaunchConfiguration("asr_model")
    ws_asr_model = LaunchConfiguration("ws_asr_model")
    llm_model = LaunchConfiguration("llm_model")

    mic_node = Node(
        package="home_nlp",
        executable="mic_node",
        name="mic_node",
        parameters=[
            {
                "sample_rate": sample_rate,
                "block_duration": block_duration,
                "num_channel": num_channel,
                "device": device,
            }
        ],
        output="screen",
        emulate_tty=True,
    )

    # asr_node = Node(
    #     package="home_nlp",
    #     executable="asr_node",
    #     name="asr_node",
    #     parameters=[
    #         {
    #             "language": language,
    #             "model": asr_model,
    #             "sample_rate": sample_rate,
    #             "block_duration": block_duration,
    #             "min_processing_duration": min_processing_duration,
    #             "period": period,
    #             "max_empty_count": max_empty_count,
    #         }
    #     ],
    #     output="screen",
    #     emulate_tty=True,
    # )

    ws_asr_node = Node(
        package="home_nlp",
        executable="ws_asr_node",
        name="ws_asr_node",
        parameters=[
            {
                "language": language,
                "model": ws_asr_model,
                "sample_rate": sample_rate,
            }
        ],
        output="screen",
        emulate_tty=True,
    )

    llm_node = Node(
        package="home_nlp",
        executable="llm_node",
        name="llm_node",
        parameters=[{"model": llm_model, "period": period}],
        # remappings=[
        #     ("transcription", "llm_input"),
        # ],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            # Launch arguments
            sample_rate_arg,
            block_duration_arg,
            num_channel_arg,
            device_arg,
            language_arg,
            period_arg,
            max_empty_count_arg,
            # asr_model_arg,
            ws_asr_model_arg,
            llm_model_arg,
            # Nodes
            mic_node,
            # asr_node,
            ws_asr_node,
            llm_node,
        ]
    )
