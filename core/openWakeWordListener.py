from openwakeword.model import Model
from openwakeword.utils import download_models
import pyaudio
import numpy as np
import os, random
import time
import subprocess
from enum import Enum
from streaming_chatbot import main_wakeword




def wakeword_detected():
    print("Wakeword detected!")



class ListenerState(Enum):
    LISTENING = 1
    COOLDOWN = 2


class OpenWakeWordListener:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1280
    THRESHOLD = 0.3
    COOLDOWN_SECONDS = 2.0  # Cooldown period after detection
    MIN_PLAYBACK_INTERVAL = 0.5  # Short interval to prevent rapid re-triggering
    VERBOSE = False  # Set to True for detailed score logging
    
    
    def __init__(self, verbose=False):
        self.VERBOSE = verbose
        self.word_model_dir = "./audio_data/wake_word_models"
        self.word_model_path = f"{self.word_model_dir}/hey_monkey.tflite"
        self.opener_audio_dir = "./audio_data/opener_audio"
        
        if not os.path.exists(self.word_model_path):
            os.makedirs(self.word_model_dir, exist_ok=True) 
            download_models(target_directory=self.word_model_dir)
            
        # Try ONNX first, fallback to TFLite
        onnx_model_path = self.word_model_path.replace('.tflite', '.onnx')
        if os.path.exists(onnx_model_path):
            self.owwModel = Model(wakeword_models=[onnx_model_path], inference_framework='onnx')
            print(f"Using ONNX model: {onnx_model_path}")
        else:
            self.owwModel = Model(wakeword_models=[self.word_model_path], inference_framework='tflite')
            print(f"Using TFLite model: {self.word_model_path}")
        self.n_models = len(self.owwModel.models.keys())
        
        # Initialize state tracking for debouncing
        self.last_activation_time = {}
        self.previous_scores = {}
        self.state = {}
        self.last_playback_time = 0
        self.audio_process = None
        
        for model_name in self.owwModel.models.keys():
            self.last_activation_time[model_name] = 0
            self.previous_scores[model_name] = 0
            self.state[model_name] = ListenerState.LISTENING
        
        self.audio = pyaudio.PyAudio()
        self.mic_stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        
        self.listen_for_wakewords_loop()

    def load_audio_files(self):
        """Load audio files from the opener_audio directory."""
        try:
            all_files = os.listdir(self.opener_audio_dir)
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
        print(f"Playing audio file: {greeting_to_play}")
        
        # Kill any existing audio process
        if self.audio_process and self.audio_process.poll() is None:
            print("Killing existing audio process")
            self.audio_process.terminate()
            self.audio_process.wait()
        
        # Start new audio process and track it
        audio_path = f"{self.opener_audio_dir}/{greeting_to_play}"
        os.system(f"afplay {audio_path}")
        main_wakeword()
        self.last_playback_time = time.time()


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
                curr_score = scores[-1]
                prev_score = self.previous_scores.get(mdl, 0)
                current_time = time.time()
                
                # Verbose logging for debugging
                if self.VERBOSE and curr_score > 0.05:
                    print(f"[{mdl}] State: {self.state[mdl].name}, Score: {curr_score:.3f}, Prev: {prev_score:.3f}")
                
                # Update state based on current conditions
                if self.state[mdl] == ListenerState.LISTENING:
                    # Check for wake word detection with rising edge and time-based cooldown
                    time_since_last_activation = current_time - self.last_activation_time.get(mdl, 0)
                    time_since_last_playback = current_time - self.last_playback_time
                    
                    if (prev_score <= self.THRESHOLD and 
                        curr_score > self.THRESHOLD and 
                        time_since_last_activation > self.COOLDOWN_SECONDS and
                        time_since_last_playback > self.MIN_PLAYBACK_INTERVAL):
                        
                        print(f"Wakeword detected! Score: {curr_score:.3f}")
                        self.state[mdl] = ListenerState.COOLDOWN
                        self.last_activation_time[mdl] = current_time
                        self.play_audio_file()
                        
                elif self.state[mdl] == ListenerState.COOLDOWN:
                    # Check if cooldown period has passed
                    time_since_activation = current_time - self.last_activation_time[mdl]
                    if time_since_activation > self.COOLDOWN_SECONDS:
                        self.state[mdl] = ListenerState.LISTENING
                        if self.VERBOSE:
                            print(f"[{mdl}] Ready to detect wake word again")
                
                # Update previous score for next iteration
                self.previous_scores[mdl] = curr_score
                        
                        


if __name__ == "__main__":
    import sys
    OWWListener = OpenWakeWordListener()