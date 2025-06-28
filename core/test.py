from wake_listener import WakeWordListener
import os
from dotenv import load_dotenv
from pathlib import Path
import sys

load_dotenv()

project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))


print(f"Key: {os.getenv('MORGAN_PORCUPINE_ACCESS_KEY')}")


wake_listener = WakeWordListener(
    access_key=os.getenv("MORGAN_PORCUPINE_ACCESS_KEY"),
    on_detect=lambda: print("Wake word detected!")
)  



wake_listener.play_audio_file()









#com.apple.voice.compact.en-AU.Karen — Karen
#com.apple.voice.compact.en-IN.Rishi — Rishi
#com.apple.voice.compact.en-US.Samantha — Samantha
#com.apple.voice.compact.zh-HK.Sinji — Sinji
#com.apple.voice.compact.en-ZA.Tessa — Tessa