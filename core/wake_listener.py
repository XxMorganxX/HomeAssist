import random
import pvporcupine
import pyaudio
import numpy as np
import threading
import pyttsx3
import os

class WakeWordListener():
    def __init__(self, access_key, on_detect, keyword_path=None):
        self.access_key = access_key
        self.keyword_path = keyword_path
        self.on_detect = on_detect  # Callback function
        self._running = False
        self._thread = None

        if keyword_path is None:
            self.keyword_path = os.getenv("PORCUPINE_KEYWORD_PATH", "")
            self.porcupine = pvporcupine.create(
            access_key=self.access_key,
            keywords=['jarvis']
            )
        else:
            self.keyword_path = keyword_path
            self.porcupine = pvporcupine.create(
            access_key=self.access_key,
            keyword_paths=[self.keyword_path]
            )

        

        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )

    def _listen(self):
        print("Wake word listener started.")
        while self._running:
            pcm = self.stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            pcm = np.frombuffer(pcm, dtype=np.int16)
            if self.porcupine.process(pcm) >= 0:
                print("Wake word detected!")
                self.play_audio_file()
                self.on_detect()

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._listen)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
        self.porcupine.delete()
        print("Wake word listener stopped.")
        
    def load_audio_files(self):
        """Load audio files from the opener_audio directory."""
        try:
            all_files = os.listdir("./audio_data/opener_audio/")
            audio_files = []
            for file in all_files:
                if file.endswith(".mov"):
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
        print(f"Playing {greeting_to_play}")
        
        os.system(f"afplay ./audio_data/opener_audio/{greeting_to_play}")
    



 