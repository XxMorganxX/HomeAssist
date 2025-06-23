# RasPi Smart Home - Voice Assistant with MCP Tools

A modular smart home voice assistant built with OpenAI integration and extensible MCP (Model Context Protocol) tool architecture.

## 🚀 Quick Start

### 0. Create Venv
```bash
python -m venv venv
source venv/bin/activate
```

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key
Create a `.env` file in the project root:
```
OPENAI_KEY=your-openai-api-key-here
```


### 3. Run the Voice Assistant
From the project root directory:
```bash
python core/streaming_chatbot.py
```

The assistant will prompt you to choose:
- **Option 1**: Basic chatbot (voice conversation only)
- **Option 2**: Tool-enabled chatbot (voice + smart home tools)

### 4. Using the Voice Assistant

1. **Speak** into your microphone
2. **Pause** for 0.9 seconds to end a speech chunk
3. **Wait** 5 seconds of silence to send your complete message
4. The assistant will **respond** via text (audio TTS coming soon)

## 🎤 How It Works

- **Real-time audio capture** with Voice Activity Detection (VAD)
- **Chunk-based transcription** using OpenAI Whisper
- **Intelligent conversation** with ChatGPT
- **Optional tool integration** for smart home control

## 🔧 Available Tools

The tool-enabled mode includes smart home tools:

- **Smart Light Control**: Turn lights on/off, adjust brightness, set colors
- **System Information**: Get system stats and information

## 📁 Project Structure

```
RasPi Smart Home/
├── core/                          # Voice assistant core
│   ├── streaming_chatbot.py       # Main voice assistant (run this!)
│   ├── audio_processing.py        # Audio capture and VAD
│   └── speech_services.py         # Whisper and ChatGPT integration
│
├── mcp_server/                    # MCP tool framework
│   ├── server.py                  # Tool server and registry
│   ├── base_tool.py               # Base class for tools
│   └── tools/                     # Smart home tools
│       ├── example_light.py       # Light control tool
│       └── system_info.py         # System information tool
│
├── examples/                      # Example usage
│   └── interactive_tool_chat.py   # Text-based tool chat
│
├── config.py                      # Configuration settings
├── .env                          # Your API keys (create this)
└── README.md                     # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Audio settings
SAMPLE_RATE = 16_000        # Audio quality
VAD_MODE = 2               # Voice detection sensitivity (0-3)
SILENCE_END_SEC = 0.9      # Pause to end speech chunk
COMPLETE_SILENCE_SEC = 5.0 # Silence to send complete message

# AI settings
WHISPER_MODEL = "whisper-1"     # Speech-to-text model
RESPONSE_MODEL = "gpt-3.5-turbo" # Chat model
MAX_TOKENS = 150               # Response length
SYSTEM_PROMPT = "..."          # Assistant personality
```

## 🛠️ Creating New Tools

1. **Create a new tool file** in `mcp_server/tools/`:

```python
# mcp_server/tools/my_tool.py
from typing import Dict, Any
from mcp_server.base_tool import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Description of what this tool does"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string", 
                    "description": "What action to take"
                }
            },
            "required": ["action"]
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params["action"]
        # Your tool logic here
        return {"result": f"Executed {action}"}
```

2. **Tool is automatically discovered** - no registration needed!

3. **Test your tool**:
```bash
python examples/interactive_tool_chat.py
```

## 🎯 Example Usage

**Voice Commands:**
- "Turn on the living room lights"
- "Set bedroom lights to 50% brightness" 
- "What's my name?" (remembers previous conversation)
- "What's the system temperature?"

**Text Tool Testing:**
```bash
python examples/interactive_tool_chat.py
# Type: "Turn on the lights"
# The AI will automatically use the smart_light tool
```

## 🔊 Audio Setup

### Microphone Test
```bash
# Check if your microphone works
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Troubleshooting Audio
- **Linux**: Install `sudo apt-get install portaudio19-dev`
- **Mac**: Install with `brew install portaudio`
- **Windows**: Usually works out of the box

### Audio Quality Tips
- Use a **USB microphone** for better quality
- Reduce **background noise** for better VAD detection
- Adjust `VAD_MODE` in config.py if too sensitive/insensitive

## ⚡ Performance Tips

### For Raspberry Pi:
- Set `VAD_MODE = 1` (less CPU intensive)
- Increase `FRAME_MS = 30` (larger chunks)
- Use lightweight system (Raspberry Pi OS Lite)

### Reduce API Costs:
- Lower `MAX_TOKENS = 100` for shorter responses
- Use `gpt-3.5-turbo` instead of GPT-4
- Optimize system prompt length

## 🐛 Troubleshooting

### "Module not found" errors:
```bash
pip install openai sounddevice webrtcvad numpy scipy python-dotenv
```

### "No OPENAI_KEY" error:
- Create `.env` file with your API key
- Or set environment variable: `export OPENAI_KEY="your-key"`

### Audio not working:
- Check microphone permissions
- Test with: `python -c "import sounddevice as sd; print(sd.query_devices())"`
- Try different `VAD_MODE` values (0-3)

### Transcription freezing:
- Check internet connection
- Verify OpenAI API key is valid
- Look for timeout errors in console

### Assistant doesn't remember:
- Conversation history is maintained within session
- Restarting the program clears memory
- Check that responses are being added to conversation

## 📚 Learn More

- **OpenAI Whisper**: [Speech recognition API](https://openai.com/research/whisper)
- **ChatGPT API**: [Conversational AI](https://platform.openai.com/docs/guides/chat)
- **WebRTC VAD**: [Voice Activity Detection](https://github.com/wiseman/py-webrtcvad)
- **SoundDevice**: [Audio I/O library](https://python-sounddevice.readthedocs.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a new tool in `mcp_server/tools/`
3. Test with `interactive_tool_chat.py`
4. Submit a pull request

The modular architecture makes it easy to add new smart home integrations!

## 📄 License

Morgan and Spencer Smart Home Project

---

🎤 **Start talking to your smart home today!**

```bash
python core/streaming_chatbot.py
```