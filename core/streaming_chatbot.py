"""
Main streaming chatbot orchestrator.
Coordinates audio capture, speech detection, transcription, and chat responses.
"""

import os
import sys
import time
import queue
from pathlib import Path
from typing import List
import logging
import json

import sounddevice as sd
from dotenv import load_dotenv

# Make script act like it's run from project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from core.audio_processing import VADChunker, wav_bytes_from_frames
from core.speech_services import SpeechServices, ConversationManager
import config

load_dotenv()

VERBOSE_LOGGING = False 
if not VERBOSE_LOGGING:
    # Suppress noisy logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp_server").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("__main__").setLevel(logging.WARNING)
else:
    # Keep all logs visible
    logging.basicConfig(level=logging.INFO)

class StreamingChatbot:
    """Main chatbot class that orchestrates the streaming conversation."""
    
    def __init__(self):
        """Initialize chatbot components."""
        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            raise ValueError("OPENAI_KEY environment variable not set")
            
        self.chunker = VADChunker(
            sample_rate=config.SAMPLE_RATE,
            frame_ms=config.FRAME_MS,
            vad_mode=config.VAD_MODE,
            silence_end_sec=config.SILENCE_END_SEC,
            max_utterance_sec=config.MAX_UTTERANCE_SEC
        )
        
        self.speech_services = SpeechServices(
            api_key=api_key,
            whisper_model=config.WHISPER_MODEL,
            chat_model=config.RESPONSE_MODEL
        )
        
        self.conversation = ConversationManager(config.SYSTEM_PROMPT)
        self.audio_queue = queue.Queue()
        
        # State tracking
        self.accumulated_chunks: List[str] = []
        self.last_speech_activity: float = None
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"⚠️  Audio status: {status}")
        self.audio_queue.put(bytes(indata))
        
    def process_chunk(self, chunk: bytes) -> None:
        """Process a speech chunk through Whisper."""
        wav_io = wav_bytes_from_frames([chunk])
        print("⏳  Transcribing chunk…")
        
        user_text = self.speech_services.transcribe(wav_io)
        if user_text:
            print(f"🎤  Chunk: {user_text}")
            self.accumulated_chunks.append(user_text)
            
    def process_complete_message(self) -> None:
        """Process accumulated chunks as complete message."""
        if not self.accumulated_chunks:
            return
            
        print("⏹️  [MESSAGE END] Complete silence detected, processing full message")
        
        # Combine all chunks
        complete_message = " ".join(self.accumulated_chunks)
        print(f"\n📝  Complete message: {complete_message}")
        print("🤖  Sending to ChatGPT...")
        
        # Add to conversation and get response
        self.conversation.add_user_message(complete_message)
        
        # Debug: Show conversation history length and content
        messages = self.conversation.get_messages()
        print(f"📊 Conversation has {len(messages)} messages")
        print("📜 Recent messages:")
        for i, msg in enumerate(messages[-3:]):  # Show last 3 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', 'no content')[:100]  # First 100 chars
            print(f"   {i}: {role}: {content}")
        
        response = self.speech_services.chat_completion(messages)
        
        if response and response.get("content"):
            self.conversation.add_assistant_message(response["content"])
            print(f"🤖  GPT: {response['content']}\n")
            print("─" * 50)
        
        # Reset for next message
        self.accumulated_chunks = []
        self.last_speech_activity = None
        
    def run(self):
        """Main run loop for streaming chatbot."""
        print("\n🎤  Ready. Speak into the microphone (Ctrl-C to quit)…")
        print("💡  I'll show each chunk as I hear it, then send the complete message after a longer pause.")
        print("🚫  CHUNKS ARE NEVER SENT TO CHATGPT INDIVIDUALLY - ONLY COMBINED!")
        print("─" * 50)
        
        with sd.RawInputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=config.FRAME_SIZE,
            dtype="int16",
            channels=1,
            callback=self.audio_callback,
        ):
            try:
                while True:
                    # Get audio frame
                    frame = self.audio_queue.get()
                    current_time = time.time()
                    
                    # Check if frame contains speech
                    frame_has_speech = self.chunker.is_speech(frame)
                    
                    # Update last speech activity
                    if frame_has_speech:
                        self.last_speech_activity = current_time
                    
                    # Process frame for chunks
                    chunk = self.chunker.process(frame)
                    if chunk:
                        self.process_chunk(chunk)
                    
                    # Check for end of complete message
                    if (self.accumulated_chunks and 
                        self.last_speech_activity and 
                        current_time - self.last_speech_activity > config.COMPLETE_SILENCE_SEC):
                        self.process_complete_message()
                        
            except KeyboardInterrupt:
                print("\n\n✋ Finished. Bye!")


class ToolEnabledStreamingChatbot(StreamingChatbot):
    """Streaming chatbot with MCP tool support."""
    
    def __init__(self):
        """Initialize tool-enabled chatbot."""
        super().__init__()
        
        # Import MCP server here to avoid circular imports
        try:
            from mcp_server.server import MCPServer
            self.mcp_server = MCPServer()
            
            # Convert MCP tools to OpenAI functions
            self.functions = []
            for tool in self.mcp_server.list_tools():
                self.functions.append({
                    "name": tool['name'],
                    "description": self.mcp_server.get_tool_info(tool['name'])['description'],
                    "parameters": tool['schema']
                })
            
            print(f"🔧 Loaded {len(self.functions)} tools: {', '.join([f['name'] for f in self.functions])}")
        except Exception as e:
            print(f"⚠️  Could not load MCP tools: {e}")
            self.mcp_server = None
            self.functions = []
    
    def process_complete_message(self) -> None:
        """Process accumulated chunks with tool support."""
        if not self.accumulated_chunks:
            return
            
        print("⏹️  [MESSAGE END] Complete silence detected, processing full message")
        
        # Combine all chunks
        complete_message = " ".join(self.accumulated_chunks)
        print(f"\n📝  Complete message: {complete_message}")
        print("🤖  Sending to ChatGPT...")
        
        # Add to conversation and get response with tools
        self.conversation.add_user_message(complete_message)
        
        # Debug: Show conversation history length and content
        messages = self.conversation.get_messages()
        print(f"📊 Conversation has {len(messages)} messages")
        print("📜 Recent messages:")
        for i, msg in enumerate(messages[-3:]):  # Show last 3 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', 'no content')[:100]  # First 100 chars
            print(f"   {i}: {role}: {content}")
        
        response = self.speech_services.chat_completion(
            messages,
            functions=self.functions if self.functions else None
        )
        
        if response:
            # Handle function calling
            if response.get("function_call") and self.mcp_server:
                func_name = response["function_call"]["name"]
                func_args = json.loads(response["function_call"]["arguments"])
                
                print(f"🔧 [Using tool: {func_name}]")
                
                # Execute tool
                tool_result = self.mcp_server.execute_tool(func_name, func_args)
                
                # Add function call and result to conversation
                # Note: We manually append to messages list since ConversationManager's 
                # add_assistant_message() doesn't support function calls
                self.conversation.messages.append({
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": func_name, 
                        "arguments": response["function_call"]["arguments"]
                    }
                })
                self.conversation.messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps(tool_result)
                })
                
                # Get final response after tool execution
                final_response = self.speech_services.chat_completion(
                    self.conversation.get_messages()
                )
                
                if final_response and final_response.get("content"):
                    self.conversation.add_assistant_message(final_response["content"])
                    print(f"🤖  GPT: {final_response['content']}\n")
                    print("─" * 50)
            
            elif response.get("content"):
                # Regular response without tools
                self.conversation.add_assistant_message(response["content"])
                print(f"🤖  GPT: {response['content']}\n")
                print("─" * 50)
        
        # Reset for next message
        self.accumulated_chunks = []
        self.last_speech_activity = None


def main():
    """Entry point with tool selection."""
    print("🎤 Streaming Chatbot")
    print("Choose mode:")
    print("1. Basic chatbot (no tools)")
    print("2. Tool-enabled chatbot (with MCP tools)")
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    try:
        if choice == '1':
            print("\n🤖 Starting basic streaming chatbot...")
            chatbot = StreamingChatbot()
        else:
            print("\n🔧 Starting tool-enabled streaming chatbot...")
            chatbot = ToolEnabledStreamingChatbot()
        
        chatbot.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("1. OPENAI_KEY is set in your environment or .env file")
        print("2. You have installed all dependencies:")
        print("   pip install openai sounddevice webrtcvad numpy scipy python-dotenv")
        print("3. Your microphone is properly configured")


if __name__ == "__main__":
    main()