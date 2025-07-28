import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv




# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI
import config

load_dotenv()

GENRES = [
    "music/spotify control",      # Spotify playback commands (play, pause, volume, search, etc.)
    "light control",              # Individual light on/off/brightness adjustments
    "lighting scene",             # Scene-based lighting (mood, party, movie, etc.)
    "calendar/schedule inquiry",  # Checking calendar events, appointments, schedules
    "notification check",         # Checking notifications, messages, emails, news
    "system state management",    # Reading/updating system states (users, scenes, etc.)
    "weather inquiry",            # Weather-related questions
    "greeting/farewell",          # Hello, goodbye, good morning/night interactions
    "general conversation",       # General chat, questions, or discussions
    "home automation query",      # Questions about home automation capabilities
    "multiple/combined"           # Conversations that involve multiple categories
]


SYSTEM_PROMPT = f"""
You are a chat classifier for a home automation system. Analyze conversations between users and their voice assistant to classify them into appropriate genres.

Available genres:
{chr(10).join(f"- {genre}" for genre in GENRES)}

Classification guidelines:
1. "music/spotify control" - Any Spotify commands (play, pause, next, search artist/song, volume)
2. "light control" - Direct light commands (turn on/off specific lights, set brightness)
3. "lighting scene" - Scene-based lighting (mood lighting, party mode, movie mode)
4. "calendar/schedule inquiry" - Checking events, appointments, what's on the schedule
5. "notification check" - Asking about notifications, messages, emails, or news updates
6. "system state management" - Changing system settings (current user, active scenes)
7. "weather inquiry" - Weather-related questions
8. "greeting/farewell" - Hello, goodbye, good morning/night interactions
9. "general conversation" - General questions or chat not fitting other categories
10. "home automation query" - Questions about what the system can do
11. "multiple/combined" - Conversations spanning multiple categories

Response format:
- Return ONLY the genre names separated by commas
- Choose all applicable genres
- If a conversation clearly involves multiple tools/topics, include "multiple/combined"
- Default to "general conversation" only if no other genre fits
"""
class ChatClassifier:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
        self.batch_size = config.CHAT_CLASSIFICATION_BATCH_SIZE


    def classify_chat(self, chat: dict) -> List[str]:
        """
        Classify the chat into a genre.
        """
    
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chat}
            ],
            max_tokens=50,
            temperature=config.TOOL_TEMPERATURE,  # Use lower temperature for deterministic classification
            timeout=20.0
        )
        return response.choices[0].message.content

    def classify_chats_batch(self, chats_with_ids: List[tuple], batch_size: int = None) -> dict:
        """
        Classify multiple chats in batches to reduce API costs.
        
        Args:
            chats_with_ids: List of tuples (chat_id, chat_data)
            batch_size: Number of chats to process in each API call (uses config default if None)
        
        Returns:
            Dict mapping chat_id to classification result
        """
        # Use config default if batch_size not specified
        if batch_size is None:
            batch_size = self.batch_size
        
        results = {}
        
        # Process chats in batches
        for i in range(0, len(chats_with_ids), batch_size):
            batch = chats_with_ids[i:i + batch_size]
            
            # Create batch prompt
            batch_prompt = "Classify each of the following chats. Return the results in the format 'ID:CLASSIFICATION'\n\n"
            
            for chat_id, chat_data in batch:
                # Convert chat_data to string if it's a dict/json
                chat_str = str(chat_data) if not isinstance(chat_data, str) else chat_data
                batch_prompt += f"Chat ID {chat_id}:\n{chat_str}\n\n"
            
            batch_prompt += f"""
                            For each chat above, return ONLY the classification in this exact format:
                            {batch[0][0]}:genre1,genre2
                            {batch[1][0] if len(batch) > 1 else 'N/A'}:genre1,genre2
                            (etc for each chat ID)

                            Available genres: {", ".join(GENRES)}
                            Use "general conversation" if no other genre fits.
                            """
            
            try:
                print(f"🔄 Processing batch of {len(batch)} chats...")
                response = self.client.chat.completions.create(
                    model="gpt-4.1-nano",
                    temperature=config.TOOL_TEMPERATURE,  # Use lower temperature for deterministic classification
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": batch_prompt}
                    ],
                    max_tokens=config.CHAT_CLASSIFICATION_MAX_TOKENS_PER_CHAT * len(batch)  # Scale tokens with batch size
                )
                
                # Parse batch response
                response_text = response.choices[0].message.content
                print(f"📝 Batch response: {response_text}")
                
                # Parse each line of the response
                for line in response_text.strip().split('\n'):
                    if ':' in line:
                        try:
                            chat_id_str, classification = line.split(':', 1)
                            chat_id = int(chat_id_str.strip())
                            results[chat_id] = classification.strip()
                        except (ValueError, IndexError) as e:
                            print(f"⚠️ Could not parse line: {line} - {e}")
                
            except Exception as e:
                print(f"❌ Batch processing failed: {e}")
                # Fallback to individual processing for this batch
                print("🔄 Falling back to individual processing...")
                for chat_id, chat_data in batch:
                    try:
                        results[chat_id] = self.classify_chat(chat_data)
                    except Exception as individual_error:
                        print(f"❌ Failed to classify chat {chat_id}: {individual_error}")
                        results[chat_id] = "general conversation"  # Default fallback
        
        return results


# Allow running as a script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Chat Genre Classifier')
    parser.add_argument('--test', action='store_true', help='Test classification with sample conversations')
    parser.add_argument('--classify-text', type=str, help='Classify a specific text/conversation')
    parser.add_argument('--show-genres', action='store_true', help='Show all available genres')
    
    args = parser.parse_args()
    
    if args.show_genres:
        print("=== Available Genre Categories ===\n")
        for i, genre in enumerate(GENRES, 1):
            print(f"{i:2d}. {genre}")
        print(f"\nTotal: {len(GENRES)} genres")
        
    elif args.test:
        # Test with sample conversations
        test_samples = [
            "User: Play some music\nAssistant: Playing music on Spotify.",
            "User: Turn on the lights\nAssistant: Turning on the lights.",
            "User: What's on my calendar?\nAssistant: You have a meeting at 2 PM.",
            "User: Good morning, turn on lights and play music\nAssistant: Good morning! Turning on lights and playing music."
        ]
        
        classifier = ChatClassifier()
        print("=== Testing Chat Classification ===\n")
        
        for i, sample in enumerate(test_samples, 1):
            print(f"Test {i}:")
            print(f"Input: {sample[:60]}...")
            result = classifier.classify_chat(sample)
            print(f"Classification: {result}\n")
            
    elif args.classify_text:
        # Classify provided text
        classifier = ChatClassifier()
        result = classifier.classify_chat(args.classify_text)
        print(f"Classification: {result}")
        
    else:
        # Default: Run database classification
        from db.db_connect import DBConnect
        
        print("🚀 Running chat genre classification on database...")
        with DBConnect() as db:
            db.insert_chat_genre()
