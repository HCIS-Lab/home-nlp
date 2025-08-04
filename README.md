# home-nlp

[![Python](https://img.shields.io/badge/Python-3.10.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4.127--1-76B900.svg?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![cuDNN](https://img.shields.io/badge/cuDNN-9.11.0.98--1-76B900.svg?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cudnn)

A ROS 2 package for **speech-driven human-robot interaction**, combining  
- **mic_node**: captures live microphone audio  
- **asr_node**: performs real-time transcription with Whisper (via faster-whisper)  
- **llm_node**: processes transcribed text with an LLM and generates robotic behavior trees  

This pipeline enables a home robot to understand spoken input and respond with context-aware actions.  

## Quickstart

Launch the **mic_node**:

```bash
ros2 run home_nlp mic_node --ros-args \
    -p sample_rate:=48000 \
    -p block_duration:=1.0 \
    -p num_channel:=1 \
    -p device:="USB Composite Device"
```

Launch the **asr_node**:

```bash
ros2 run home_nlp asr_node --ros-args \
    -p language:="zh" \
    -p model:="large-v2" \
    -p sample_rate:=48000 \
    -p block_duration:=1.0 \
    -p period:=1.0 \
    -p max_empty_count:=0
```

Launch the **llm_node**:

```bash
ros2 run home_nlp llm_node --ros-args \
    -p period:=1.0 \
    -p model:="google/gemma-3-1b-it"
```

## TODO Docker 使用說明

Build the image:

```bash
docker build -t lnfu/home_nlp
```

Run individual nodes:

```bash
docker run --rm lnfu/home_nlp ros2 run home_nlp mic_node
docker run --rm lnfu/home_nlp ros2 run home_nlp asr_node
docker run --rm -e HF_TOKEN="${HF_TOKEN}" lnfu/home_nlp ros2 run home_nlp llm_node
```

## LLM Model Comparison

*Valid XML* indicates the percentage of runs (out of 100) producing syntactically valid XML.  
Note: This does not verify semantic correctness of the behavior tree.  

| Model              | Loading Time (s) | Response Time (s) | VRAM Usage (MB) | RAM Usage (MB) | Valid XML (%) |
| ------------------ | ---------------- | ----------------- | --------------- | -------------- | ------------- |
| gemma-3-1b-it      | 4.19             | 3.22              | 2482            | 2363           | 66            |
| gemma-3-4b-it      | 7.25             | 4.34              | 9480            | 5455           | 99            |
| deepseek 6.7b-it   | 15.64            | 3.03              | 14096           | 10094          | 76            |
| Phi-4-mini-it      | 8.26             | 3.34              | 8735            | 5251           | 60            |
| Mistral-7B-it-v0.3 | 5.68             | 2.02              | 14135           | 5344           | 98            |
| LLama-3.1-8B-it    | 6.08             | 1.80              | 15623           | 5350           | 99            |

## Whisper Streaming

This project integrates ideas and components from  
[ufal/whisper_streaming](https://github.com/ufal/whisper_streaming),  
which provides the foundation for real-time Whisper transcription.  

### Reference

```
bibtex
@inproceedings{machacek-etal-2023-turning,
    title = "Turning Whisper into Real-Time Transcription System",
    author = "Mach{\'a}{\v{c}}ek, Dominik  and
      Dabre, Raj  and
      Bojar, Ond{\v{r}}ej",
    editor = "Saha, Sriparna  and
      Sujaini, Herry",
    booktitle = "Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = nov,
    year = "2023",
    address = "Bali, Indonesia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.ijcnlp-demo.3",
    pages = "17--24",
}
```
