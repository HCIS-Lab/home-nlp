from abc import ABC, abstractmethod
import sounddevice as sd


class TextToSpeechPlayer(ABC):
    def __init__(self, device="pulse", lang: str = "zh"):
        self.device = device
        self.lang = lang

    @abstractmethod
    def speak(self, text: str): ...

    def play_audio(self, audio, sr: int):
        with sd.OutputStream(device=self.device, samplerate=sr, channels=audio.shape[1]) as stream:
            stream.write(audio)

    @staticmethod
    def ensure_audio_shape(audio):
        if audio.ndim == 1:  # (N,) -> (N, 1)
            return audio.reshape(-1, 1)
        else:
            return audio.T
