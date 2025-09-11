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
        default_value="zh",
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

    asr_model_arg = DeclareLaunchArgument(
        "asr_model", default_value="large-v2", description="ASR (Whisper) model"
    )

    llm_model_arg = DeclareLaunchArgument(
        "llm_model", default_value="google/gemma-3-1b-it", description="LLM model"
    )

    sample_rate = LaunchConfiguration("sample_rate")
    block_duration = LaunchConfiguration("block_duration")
    num_channel = LaunchConfiguration("num_channel")
    device = LaunchConfiguration("device")
    language = LaunchConfiguration("language")
    period = LaunchConfiguration("period")
    max_empty_count = LaunchConfiguration("max_empty_count")
    asr_model = LaunchConfiguration("asr_model")
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

    asr_node = Node(
        package="home_nlp",
        executable="asr_node",
        name="asr_node",
        parameters=[
            {
                "language": language,
                "model": asr_model,
                "sample_rate": sample_rate,
                "block_duration": block_duration,
                "period": period,
                "max_empty_count": max_empty_count,
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
        remappings=[
            ("transcription", "/llm_input"),
        ],
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
            asr_model_arg,
            llm_model_arg,
            # Nodes
            mic_node,
            asr_node,
            llm_node,
        ]
    )
