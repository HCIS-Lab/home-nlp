# home-nlp

## Quickstart

```bash
# 啟動麥克風
ros2 run home_nlp mic_node --ros-args -p device:="USB Composite Device"
```

## whisper_streaming

https://github.com/ufal/whisper_streaming

```
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

## LLM Model comparison
| Model               | Loading Time (s) | Response Time (s) | VRAM Usage (MB) | RAM Usage (MB) | Valid XML (%) |
|---------------------|------------------|--------------------|------------------|----------------|----------------|
| gemma-3-1b-it       | 4.19             | 3.22               | 2482             | 2363           | 66             |
| gemma-3-4b-it       | 7.25             | 4.34               | 9480             | 5455           | 99             |
| deepseek 6.7b-it    | 15.64            | 3.03               | 14096            | 10094          | 76             |
| Phi-4-mini-it       | 8.26             | 3.34               | 8735             | 5251           | 60             |
| Mistral-7B-it-v0.3  | 5.68             | 2.02               | 14135            | 5344           | 98             |
| LLama-3.1-8B-it     | 6.08             | 1.80               | 15623            | 5350           | 99             |

## TODOs

```
/usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 1.26.4
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"

/usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 1.26.4
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"

Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so}
Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor
[ros2run]: Aborted
```