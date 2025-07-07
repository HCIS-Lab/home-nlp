from kokoro import KPipeline
from home_nlp.tts_player import TextToSpeechPlayer


class KokoroTextToSpeechPlayer(TextToSpeechPlayer):
    def __init__(self, device="pulse", lang="zh"):
        super().__init__(device, lang)

        self.pipeline = KPipeline(lang_code=lang)
        self.voice = "zm_yunxi"

    def speak(self, text):
        generator = self.pipeline(text, voice=self.voice)
        for gs, ps, audio in generator:
            audio = self.ensure_audio_shape(audio)
            super().play_audio(audio, 24000)


if __name__ == "__main__":
    tts = KokoroTextToSpeechPlayer()
    tts.speak("請叫我派大星教授加博士先生")
