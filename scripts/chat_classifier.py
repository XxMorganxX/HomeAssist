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

GENRES = ["music/spotify request", "weather inquiry", "light control", "calendar/reminder", "other/general conversation"]


SYSTEM_PROMPT = f"""
You are a chat classifier. You are given a chat between a user and a home automation assistant. 
You need to classify it into one or more genres The genres are: {", ".join(GENRES)}.

If no genre is applicable, return "other/general conversation".

Response format:
- You must return a list of genres separated by commas.
- You can choose multiple genres.
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
                            Use "other/general conversation" if no other genre fits.
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
                        results[chat_id] = "other/general conversation"  # Default fallback
        
        return results
