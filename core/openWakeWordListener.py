from openwakeword.model import Model
from openwakeword.utils import download_models
import pyaudio
import numpy as np
import os, random




def wakeword_detected():
    print("Wakeword detected!")



class OpenWakeWordListener:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1280
    THRESHOLD = 0.3
    
    
    def __init__(self):
        self.word_model_dir = "./audio_data/wake_word_models"
        self.word_model_path = f"{self.word_model_dir}/hey_jarvis_v0.1.onnx"
        self.opener_audio_dir = "./audio_data/opener_audio"
        
        if not os.path.exists(self.word_model_path):
            os.makedirs(self.word_model_dir, exist_ok=True) # Create directory if it doesn't exist
            download_models(target_directory=self.word_model_dir)
            
        self.owwModel = Model(wakeword_models=[self.word_model_path], inference_framework='onnx')
        self.n_models = len(self.owwModel.models.keys())
        
        self.audio = pyaudio.PyAudio()
        self.mic_stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        
        self.listen_for_wakewords_loop()

    def load_audio_files(self):
        """Load audio files from the opener_audio directory."""
        try:
            all_files = os.listdir(self.opener_audio_dir)
            audio_files = []
            for file in all_files:
                if file.endswith(".mp3") or file.endswith(".wav"):
                    audio_files.append(file)
            return audio_files if audio_files else ["No audio files found"]
        except FileNotFoundError:
            print("Warning: opener_audio directory not found")
            return ["Directory not found"]
    
    def play_audio_file(self):
        all_greetings = self.load_audio_files()
        if all_greetings == ["No audio files found"] or all_greetings == ["Directory not found"]:
            return None
        
        greeting_to_play = random.choice(all_greetings)
        print("Playing audio file")
        os.system(f"afplay {self.opener_audio_dir}/{greeting_to_play}")


    def listen_for_wakewords_loop(self):
        print("Listening for wakewords...")
        while True:
            # Get audio
            try:
                audio = np.frombuffer(self.mic_stream.read(self.CHUNK, exception_on_overflow=False), dtype=np.int16)
            except Exception as e:
                print(f"Audio read error: {e}")
                continue

            # Feed to openWakeWord model
            prediction = self.owwModel.predict(audio)


            for mdl in self.owwModel.prediction_buffer.keys():
                scores = list(self.owwModel.prediction_buffer[mdl])
                curr_score = format(scores[-1], '.20f').replace("-", "")
                
                if scores[-1] > self.THRESHOLD:
                    print("Wakeword detected!")
                    self.play_audio_file()
                        
                        


if __name__ == "__main__":  
    OWWListener = OpenWakeWordListener()