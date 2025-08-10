from typing import List, Dict, Union
import json
import tiktoken

class Context:
    def __init__(self, user_input: str, conversation_history: List[Dict]):
        self.user_input = user_input
        self.conversation_history = conversation_history
        
        # Initialize tokenizer for gpt-5-nano (using gpt-4 encoding as fallback)
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-4")
        except:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        self.system_prompt = "You are a smart home assistant with access to various tools. You can check notifications, control lights, manage calendar, control Spotify, and more. Be helpful, concise, and always use the appropriate tools when needed. When users ask about their home or personal information, use the available tools."
        self.initialize_conversation_history()

    def get_user_input(self):
        return self.user_input
    
    def get_conversation_history(self):
        total_tokens = self.count_conversation_tokens()
        print(f"[DEBUG] Length of conversation history: {len(self.conversation_history)} messages")
        print(f"[DEBUG] Exact tokens in conversation history: {total_tokens}")
        return self.conversation_history
    
    def count_conversation_tokens(self) -> int:
        """Count the exact number of tokens in the conversation history using OpenAI's tokenizer."""
        # Format messages as they would be sent to the API
        formatted_messages = ""
        for message in self.conversation_history:
            role = message.get('role', '')
            content = message.get('content', '')
            formatted_messages += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Use tiktoken to count exact tokens
        return len(self.encoder.encode(formatted_messages))
    
    def initialize_conversation_history(self):
        # Only initialize if conversation history is empty
        if not self.conversation_history:
            self.conversation_history = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            ]
    
    def add_to_conversation_history(self, message: Union[Dict, List[Dict]]):
        """Add one or many messages to the history.

        Accepts either a single message dict or a list of message dicts.
        """
        if isinstance(message, list):
            self.conversation_history.extend(message)
        else:
            self.conversation_history.append(message)
    
    def get_conversation_history_str(self) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
    
    def get_conversation_history_json(self) -> str:
        return json.dumps(self.conversation_history)
    