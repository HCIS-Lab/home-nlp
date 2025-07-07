import librosa

from io import BytesIO
from gtts import gTTS
from home_nlp.tts_player import TextToSpeechPlayer


class GoogleTextToSpeechPlayer(TextToSpeechPlayer):
    def __init__(self, device="pulse", lang="zh"):
        super().__init__(device, lang)

    def speak(self, text: str):
        tts = gTTS(text=text, lang=self.lang)

        # 因為 gTTS 產生的資料是 mp3 格式（壓縮）
        # 因此只能先存到檔案或是 BytesIO 再用 librosa 等套件讀取播放
        fp = BytesIO()
        tts.write_to_fp(fp)

        # Read Audio from BytesIO
        fp.seek(0)
        audio, sr = librosa.load(fp)
        audio = self.ensure_audio_shape(audio)
        super().play_audio(audio, sr)


if __name__ == "__main__":
    tts = GoogleTextToSpeechPlayer()
    tts.speak("請叫我派大星教授加博士先生")
