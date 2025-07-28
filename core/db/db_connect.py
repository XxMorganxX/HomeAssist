import psycopg2
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from datetime import datetime
from psycopg2.extras import Json
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.chat_classifier import ChatClassifier
import config

# Load environment variables from .env
load_dotenv()

# Fetch variables
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")


class DBConnect:
    TABLE_NAME="session_logs"
    TIMESTAMP_COL_NAME="session_ended_at"
    GENRE_COL_NAME="genre"
    CHAT_COL_NAME="chat"
    SYSTEM_PROMPT_COL_NAME="sys_prompt"
    ID_COL_NAME="id"
    ACCUMULATED_TRANSCRIPTION_LATENCY_COL_NAME="accumulated_transcription_latencies"
    FINAL_TRANSCRIPTION_LATENCY_COL_NAME="final_transcription_latency"

    def __init__(self):
        self.conn = None
        self.cur = None
        self._connect_with_retry()
    
    def _connect_with_retry(self, max_retries=3, retry_delay=1):
        """Establish connection with retry logic."""
        for attempt in range(max_retries):
            try:
                self.conn = self.connect_db()
                self.cur = self.conn.cursor()
                print(f"Database connected successfully on attempt {attempt + 1}")
                return
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"Failed to connect after {max_retries} attempts")
    
    def connect_db(self):
        """Establish and return a connection to the PostgreSQL database."""
        return psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
    
    def _ensure_connection(self):
        """Ensure connection is alive, reconnect if necessary."""
        try:
            # Test connection with a simple query
            self.cur.execute("SELECT 1")
            self.cur.fetchone()
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            print("Connection lost, attempting to reconnect...")
            self._connect_with_retry()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()
    
    def _format_genre_for_db(self, genre_string):
        """Convert genre classification string to PostgreSQL array format."""
        if not genre_string:
            return None
        
        # Split by comma and clean up each genre
        genres = [genre.strip() for genre in genre_string.split(',')]
        
        # Remove empty strings
        genres = [genre for genre in genres if genre]
        
        if not genres:
            return None
        
        # Return as Python list - psycopg2 will handle the conversion to PostgreSQL array
        return genres

    def insert_new_chat(self, session_time, chat_data, system_prompt=None, accumulated_latencies=None, final_latencies=None):
        """Insert a new chat session into the 'session_logs' table with latency metrics and system prompt."""
        try:
            self._ensure_connection()
            
            # SQL insert statement with system_prompt and latency columns
            query = f"""INSERT INTO {self.TABLE_NAME} 
                       ({self.TIMESTAMP_COL_NAME}, {self.CHAT_COL_NAME}, {self.SYSTEM_PROMPT_COL_NAME},
                        {self.ACCUMULATED_TRANSCRIPTION_LATENCY_COL_NAME}, 
                        {self.FINAL_TRANSCRIPTION_LATENCY_COL_NAME}) 
                       VALUES (%s, %s, %s, %s, %s);"""
            
            # Convert latency lists to PostgreSQL arrays (can be None)
            accumulated_array = accumulated_latencies if accumulated_latencies else None
            final_array = final_latencies if final_latencies else None
            
            self.cur.execute(query, (session_time, Json(chat_data), system_prompt, accumulated_array, final_array))

            self.conn.commit()
            print("Chat session logged successfully with latency metrics!")
            
            if accumulated_latencies:
                print(f"  - Accumulated latencies: {[f'{x:.3f}s' for x in accumulated_latencies]}")
            if final_latencies:
                print(f"  - Final latencies: {[f'{x:.3f}s' for x in final_latencies]}")
                
        except Exception as e:
            print(f"Error inserting chat session: {e}")
            self.conn.rollback()  # Rollback on error
    
    def get_latest_log_entry(self):
        """Retrieve the most recent log entry from the 'session_logs' table."""
        try:
            self._ensure_connection()

            query = f"SELECT {self.TIMESTAMP_COL_NAME}, {self.CHAT_COL_NAME} FROM {self.TABLE_NAME} ORDER BY {self.TIMESTAMP_COL_NAME} DESC LIMIT 1;"
            self.cur.execute(query)
            row = self.cur.fetchone()

            if row:
                created_at, metadata = row
                print("Latest Entry:")
                print("Timestamp:", created_at)
                print("Metadata:", metadata)
                return row
            else:
                print("No entries found.")
                return None

        except Exception as e:
            print(f"Error retrieving latest log entry: {e}")
            return None
        
    def get_latest_chat_id(self):
        """Retrieve the latest chat ID from the 'session_logs' table."""
        try:
            self._ensure_connection()
            
            query = f"SELECT {self.ID_COL_NAME} FROM {self.TABLE_NAME} ORDER BY {self.ID_COL_NAME} DESC LIMIT 1;"
            self.cur.execute(query)
            row = self.cur.fetchone()
            return row[0] if row else -1
        except Exception as e:
            print(f"Error retrieving latest chat ID: {e}")
            return None
    
    def insert_chat_genre(self, batch_size=None, skip_on_error=True, use_batch_processing=True):
        """Classify and update chat genres for entries without genres."""
        # Use config default if batch_size not specified
        if batch_size is None:
            batch_size = config.CHAT_CLASSIFICATION_BATCH_SIZE
            
        batch_classifier = ChatClassifier()
            
        try:
            self._ensure_connection()
            
            # Get unclassified chats
            query = f"SELECT {self.ID_COL_NAME}, {self.TIMESTAMP_COL_NAME}, {self.CHAT_COL_NAME} FROM {self.TABLE_NAME} WHERE {self.GENRE_COL_NAME} IS NULL ORDER BY {self.TIMESTAMP_COL_NAME} DESC;"
            self.cur.execute(query)
            rows = self.cur.fetchall()
            
            if not rows:
                print("No unclassified chats found.")
                return {"success": True, "processed": 0, "failed": 0}
            
            print(f"Found {len(rows)} unclassified chats.")
            
            processed = 0
            failed = 0
            
            if use_batch_processing:
                print(f"Using batch processing (batch size: {batch_size}) - This is more cost-effective! 💰")
                
                # Prepare data for batch processing
                chats_with_ids = [(row[0], row[2]) for row in rows]  # (chat_id, chat_data)
                
                # Get batch classifications
                try:
                    batch_classifier = ChatClassifier()
                    classifications = batch_classifier.classify_chats_batch(chats_with_ids, batch_size)
                    
                    # Update database with results
                    for chat_id, genre in classifications.items():
                        try:
                            # Format genre as PostgreSQL array
                            genre_array = self._format_genre_for_db(genre)
                            update_query = f"UPDATE {self.TABLE_NAME} SET {self.GENRE_COL_NAME} = %s WHERE {self.ID_COL_NAME} = %s;"
                            self.cur.execute(update_query, (genre_array, chat_id))
                            self.conn.commit()
                            processed += 1
                            print(f"✅ Chat {chat_id} classified as: {genre} (stored as array: {genre_array})")
                        except Exception as update_error:
                            failed += 1
                            print(f"❌ Failed to update chat {chat_id}: {update_error}")
                            if not skip_on_error:
                                raise update_error
                    
                    # Handle any chats that weren't classified
                    classified_ids = set(classifications.keys())
                    all_ids = {row[0] for row in rows}
                    missing_ids = all_ids - classified_ids
                    
                    if missing_ids:
                        print(f"⚠️ {len(missing_ids)} chats were not classified in batch processing")
                        failed += len(missing_ids)
                        
                except Exception as batch_error:
                    print(f"❌ Batch processing failed: {batch_error}")
                    if not skip_on_error:
                        raise batch_error
                    
                    # Fallback to individual processing
                    print("🔄 Falling back to individual processing...")
                    use_batch_processing = False
            
            # Individual processing (fallback or if batch processing is disabled)
            if not use_batch_processing:
                print(f"Using individual processing - This will be slower and more expensive 💸")
                
                for i, row in enumerate(rows):
                    chat_id, timestamp, chat = row
                    
                    try:
                        print(f"Processing chat {i+1}/{len(rows)} (ID: {chat_id})...")
                        genre = batch_classifier.classify_chat(chat)
                        
                        # Update individual row
                        update_query = f"UPDATE {self.TABLE_NAME} SET {self.GENRE_COL_NAME} = %s WHERE {self.ID_COL_NAME} = %s;"
                        self.cur.execute(update_query, (genre, chat_id))
                        self.conn.commit()
                        
                        processed += 1
                        print(f"✅ Chat {chat_id} classified as: {genre}")
                        
                        # Rate limiting between individual API calls
                        if i < len(rows) - 1:
                            time.sleep(config.CHAT_CLASSIFICATION_RATE_LIMIT_DELAY)
                            
                    except Exception as classification_error:
                        failed += 1
                        print(f"❌ Failed to classify chat {chat_id}: {classification_error}")
                        
                        if not skip_on_error:
                            raise classification_error
                        continue
            
            result = {
                "success": True,
                "processed": processed,
                "failed": failed,
                "total_found": len(rows),
                "method": "batch" if use_batch_processing else "individual"
            }
            
            print(f"\n📊 Classification Results:")
            print(f"   Total found: {len(rows)}")
            print(f"   Successfully processed: {processed}")
            print(f"   Failed: {failed}")
            print(f"   Method used: {'Batch processing 💰' if use_batch_processing else 'Individual processing 💸'}")
            
            if use_batch_processing and processed > 0:
                api_calls_saved = len(rows) - (len(rows) // batch_size + (1 if len(rows) % batch_size else 0))
                print(f"   🎉 Estimated API calls saved: ~{api_calls_saved}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in classification process: {e}")
            self.conn.rollback()
            return {"success": False, "error": str(e), "processed": 0, "failed": 0}
    

    def close_connection(self):
        """Close the connection to the PostgreSQL database."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        print("Database connection closed")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Database operations for chat logs')
    parser.add_argument('--classify', action='store_true', help='Run genre classification on unclassified chats')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for classification (default: from config)')
    parser.add_argument('--latest', action='store_true', help='Show latest log entry')
    parser.add_argument('--test-insert', action='store_true', help='Insert a test chat entry')
    
    args = parser.parse_args()
    
    # Using context manager for automatic cleanup
    with DBConnect() as db:
        if args.classify:
            print("🚀 Starting chat genre classification...")
            result = db.insert_chat_genre(batch_size=args.batch_size)
            if result['success']:
                print(f"\n✅ Classification complete!")
            else:
                print(f"\n❌ Classification failed: {result.get('error', 'Unknown error')}")
                
        elif args.latest:
            print("📋 Fetching latest log entry...")
            db.get_latest_log_entry()
            
        elif args.test_insert:
            print("🧪 Inserting test chat...")
            db.insert_new_chat(
                datetime.now(), 
                {
                    "test_id": db.get_latest_chat_id()+1, 
                    "user": "Hello, how are you?", 
                    "assistant": "I'm good, thank you!"
                },
                system_prompt="Test system prompt"
            )
        else:
            # Default action: classify chats
            print("🚀 Running default action: chat genre classification...")
            db.insert_chat_genre()
