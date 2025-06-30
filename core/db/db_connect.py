import psycopg2
from dotenv import load_dotenv
import os
from datetime import datetime
from psycopg2.extras import Json

# Load environment variables from .env
load_dotenv()

# Fetch variables
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")


class DBConnect:
    def __init__(self):
        self.conn = self.connect_db()
        self.cur = self.conn.cursor()
    
    def connect_db(self):
        """Establish and return a connection to the PostgreSQL database."""
        return psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )

    def insert_new_chat(self, session_time, chat_data):
        """Insert a new user into the 'users' table."""
        try:
            
            # SQL insert statement (adapt to your table schema)
            query = "INSERT INTO session_logs (session_started_at, chat) VALUES (%s, %s);"
            self.cur.execute(query, (session_time, Json(chat_data)))

            self.conn.commit()
            print("User inserted successfully!")
        except Exception as e:
            print(f"Error inserting user: {e}")
    
    def get_latest_log_entry(self):
        """Retrieve the most recent log entry from the 'logs' table."""
        try:

            query = "SELECT session_started_at, chat FROM session_logs ORDER BY session_started_at DESC LIMIT 1;"
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
            query = "SELECT id FROM session_logs ORDER BY id DESC LIMIT 1;"
            self.cur.execute(query)
            row = self.cur.fetchone()
            return row[0] if row else -1
        except Exception as e:
            print(f"Error retrieving latest chat ID: {e}")
            return None

    def close_connection(self):
        """Close the connection to the PostgreSQL database."""
        self.cur.close()
        self.conn.close()

# Example usage
if __name__ == "__main__":
    db = DBConnect()
    db.insert_new_chat(datetime.now(), {"test_id": db.get_latest_chat_id()+1, "user": "Hello, how are you?", "assistant": "I'm good, thank you!"})
    print(db.get_latest_log_entry())
    print(db.get_latest_chat_id())
    db.close_connection()
