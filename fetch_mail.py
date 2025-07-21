# =============================================================================
# CREDENTIALS LOCATION
# =============================================================================
# Email credentials should be stored in: .env file in the project root
# Format: assistant_email=your_email@gmail.com
# Format: assistant_pass=your_app_password
# 
# Email data will be stored in: email_processing.json in the project root
# =============================================================================

# =============================================================================
# GOOGLE API CREDENTIALS CONFIGURATION
# =============================================================================
# Configure the location of your Google API credential files
GOOGLE_TOKEN_FILE = "creds/token.json"                    # Google OAuth token file
GOOGLE_CREDENTIALS_FILE = "creds/email_credentials.json"  # Google service account credentials
# =============================================================================

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import base64

# Load environment variables
load_dotenv()

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class EmailManager:
    def __init__(self, token_file: str = GOOGLE_TOKEN_FILE, credentials_file: str = GOOGLE_CREDENTIALS_FILE, window_index: int = None):
        """Initialize email manager with Google API credentials."""
        self.token_file = token_file
        self.credentials_file = credentials_file
        self.service = None
        self.window_index = window_index
        self.last_processed_date_str, self.last_processed_timestamp = self.load_last_processed_date(window_index)
        
    def authenticate(self):
        """Authenticate using OAuth2 credentials."""
        creds = None
        
        # Token file stores the user's access and refresh tokens
        if os.path.exists(self.token_file):
            with open(self.token_file, 'r') as token:
                creds = Credentials.from_authorized_user_info(json.load(token), SCOPES)
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('gmail', 'v1', credentials=creds)
        return self.service
    
    def load_last_processed_date(self, window_index: int = None) -> tuple:
        """Load the last processed date from state.json."""
        try:
            if os.path.exists('state.json'):
                with open('state.json', 'r') as file:
                    state = json.load(file)
                    
                    # Check if we have the new format with processing_history
                    if 'processing_history' in state and state['processing_history']:
                        history = state['processing_history']
                        
                        # If window_index is specified, use that entry
                        if window_index is not None:
                            # Convert window_index (1,2,3) to array index (-1,-2,-3)
                            array_index = -window_index
                            
                            # Check if we have enough history
                            if abs(array_index) <= len(history):
                                last_run = history[array_index]
                                print(f"🕰️ Using processing window {window_index} (from {len(history)} available)")
                            else:
                                print(f"⚠️ Warning: Only {len(history)} history entries available, cannot use window {window_index}")
                                print(f"   Using most recent entry instead.")
                                last_run = history[-1]
                        else:
                            # Default to most recent entry
                            last_run = history[-1]
                        
                        last_date = last_run.get('last_processed_date')
                    else:
                        # Fallback to old format
                        if window_index is not None:
                            print("⚠️ Warning: No processing history available, ignoring window parameter")
                        last_date = state.get('last_processed_date')
                    
                    if last_date:
                        # Keep the full timestamp for precise filtering
                        dt = datetime.fromisoformat(last_date.replace('Z', '+00:00'))
                        # Return both Gmail query format and precise timestamp
                        return dt.strftime('%Y/%m/%d'), dt
        except Exception as e:
            print(f"Note: Could not load previous state: {e}")
        return None, None
    
    def fetch_emails_from_gmail(self, max_emails: int = 10) -> List[Dict[str, Any]]:
        """Fetch emails from Gmail using OAuth API."""
        emails_data = []
        
        try:
            if not self.service:
                self.authenticate()
            
            # Build query to fetch only new emails
            query = None
            if self.last_processed_date_str:
                query = f'after:{self.last_processed_date_str}'
                print(f"📅 Fetching emails after {self.last_processed_date_str} (precise: {self.last_processed_timestamp.isoformat()})")
            else:
                print("📅 No previous state found. Fetching recent emails...")
            
            # Call the Gmail API to fetch messages
            results = self.service.users().messages().list(
                userId='me',
                maxResults=max_emails * 2,  # Fetch more to account for filtering
                q=query
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                print("📭 No new emails found since last check.")
                return emails_data
            
            print(f"📬 Found {len(messages)} potential email(s) to check.")
            
            # Process messages and filter by precise timestamp
            new_email_count = 0
            for message in messages:
                try:
                    # Get the full message
                    msg = self.service.users().messages().get(
                        userId='me',
                        id=message['id']
                    ).execute()
                    
                    # Check if email is newer than our last processed timestamp
                    if self.last_processed_timestamp:
                        email_timestamp = self.get_email_timestamp(msg)
                        if email_timestamp and email_timestamp <= self.last_processed_timestamp:
                            continue  # Skip this email, it's older than our last processed
                    
                    # Extract email data
                    email_data = self.extract_email_data(msg)
                    if email_data:
                        emails_data.append(email_data)
                        new_email_count += 1
                        
                        # Stop if we've reached our limit
                        if new_email_count >= max_emails:
                            break
                        
                except Exception as e:
                    print(f"Error processing email {message['id']}: {e}")
                    continue
            
            if new_email_count > 0:
                print(f"✅ Found {new_email_count} new email(s) after timestamp filtering.")
                    
        except Exception as e:
            print(f"Error connecting to Gmail API: {e}")
            print("Make sure your OAuth credentials are properly configured.")
            
        return emails_data
    
    def extract_email_data(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data from a Gmail API message."""
        try:
            headers = msg['payload'].get('headers', [])
            
            # Extract header information
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
            
            # Get body
            body = self.get_email_body(msg['payload'])
            
            # Create snippet
            snippet = msg.get('snippet', body[:200] + "..." if len(body) > 200 else body)
            
            return {
                "id": msg['id'],
                "subject": subject,
                "sender": sender,
                "date": date,
                "snippet": snippet,
                "body": body,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error extracting email data: {e}")
            return None
    
    def get_email_body(self, payload: Dict[str, Any]) -> str:
        """Extract the text body from a Gmail API payload."""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                    break
                elif 'parts' in part:
                    # Recursively check nested parts
                    body = self.get_email_body(part)
                    if body:
                        break
        elif payload['body'].get('data'):
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
        
        return body
    
    def get_email_timestamp(self, msg: Dict[str, Any]) -> datetime:
        """Extract the timestamp from a Gmail API message."""
        try:
            # Get internal timestamp from Gmail (milliseconds since epoch)
            internal_date = msg.get('internalDate')
            if internal_date:
                # Convert milliseconds to seconds
                timestamp = int(internal_date) / 1000
                return datetime.fromtimestamp(timestamp)
        except Exception as e:
            print(f"Error extracting timestamp: {e}")
        return None
    
    def save_emails_to_json(self, emails: List[Dict[str, Any]], filename: str = "new_mail.json"):
        """Save email data to JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(emails, file, indent=2, ensure_ascii=False)
            print(f"Saved {len(emails)} emails to {filename}")
            
            # Save state information only if not using window_index (historical mode)
            if emails and self.window_index is None:
                self.save_state(emails)
            elif self.window_index is not None:
                print(f"💾 State update skipped (using historical window {self.window_index})")
            
            return True
        except Exception as e:
            print(f"Error saving emails to JSON: {e}")
            return False
    
    def save_state(self, emails: List[Dict[str, Any]]):
        """Save processing state to state.json."""
        try:
            # Load existing state to preserve history
            existing_state = {}
            if os.path.exists('state.json'):
                try:
                    with open('state.json', 'r') as file:
                        existing_state = json.load(file)
                except:
                    pass
            
            # Get the most recent email (first in the list from Gmail API)
            last_email = emails[0] if emails else None
            
            # Create new processing entry
            new_entry = {
                "last_processed_email_id": last_email['id'] if last_email else None,
                "last_processed_date": datetime.now().isoformat(),
                "total_emails_processed": len(emails)
            }
            
            # Initialize or update processing history
            if 'processing_history' not in existing_state:
                # Migrate old format if it exists
                if 'last_processed_email_id' in existing_state:
                    existing_state['processing_history'] = [{
                        "last_processed_email_id": existing_state.get('last_processed_email_id'),
                        "last_processed_date": existing_state.get('last_processed_date'),
                        "total_emails_processed": existing_state.get('total_emails_processed', 0)
                    }]
                else:
                    existing_state['processing_history'] = []
            
            # Append new entry and keep only last 3
            existing_state['processing_history'].append(new_entry)
            existing_state['processing_history'] = existing_state['processing_history'][-3:]
            
            # Also keep the latest values at root level for backwards compatibility
            existing_state.update(new_entry)
            
            with open('state.json', 'w', encoding='utf-8') as file:
                json.dump(existing_state, file, indent=2)
            
            print(f"💾 Saved processing state to state.json")
            print(f"   Processing history entries: {len(existing_state['processing_history'])}")
            
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def fetch_and_save_emails(self, max_emails: int = 10) -> bool:
        """Fetch emails from Gmail and save to JSON file."""
        print(f"📧 Fetching up to {max_emails} recent emails from Gmail...")
        
        emails = self.fetch_emails_from_gmail(max_emails)
        
        if emails:
            success = self.save_emails_to_json(emails)
            if success:
                print(f"✅ Successfully fetched and saved {len(emails)} emails")
                return True
            else:
                print("❌ Failed to save emails to JSON")
                return False
        else:
            print("ℹ️ No new emails to process.")
            if self.window_index is not None:
                print(f"💾 State update skipped (using historical window {self.window_index})")
            return True  # This is not an error condition
         
def check_oauth_credentials():
    """Check if OAuth credentials are properly set up."""
    print("🔍 Checking OAuth credentials...")
    
    # Check if credentials file exists
    if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
        print(f"❌ OAuth credentials file not found at {GOOGLE_CREDENTIALS_FILE}!")
        print("   Please ensure your Google OAuth credentials are in the specified location.")
        return False
    
    print("✅ OAuth credentials file found!")
    return True

def fetch_and_save_emails(window_index: int = None):
    """Fetch emails from Gmail and save to email_processing.json."""
    print("📧 Fetching emails from Gmail...")
    
    try:
        email_manager = EmailManager(window_index=window_index)
        success = email_manager.fetch_and_save_emails(max_emails=10)
        return success
    except Exception as e:
        print(f"❌ Error fetching emails: {e}")
        return False

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fetch emails from Gmail')
    
    # Add mutually exclusive group for window arguments
    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument('--scrape-window-1', action='store_const', const=1, dest='window',
                             help='Use the most recent processing timestamp')
    window_group.add_argument('--scrape-window-2', action='store_const', const=2, dest='window',
                             help='Use the second most recent processing timestamp')
    window_group.add_argument('--scrape-window-3', action='store_const', const=3, dest='window',
                             help='Use the third most recent processing timestamp')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📧 EMAIL FETCHER QUICKSTART")
    print("=" * 60)
    
    # Check OAuth credentials
    if not check_oauth_credentials():
        print("\n❌ OAuth credentials not found. Please fix the issues above and try again.")
        sys.exit(1)
    
    # Fetch and save emails
    print("\n" + "=" * 60)
    success = fetch_and_save_emails(window_index=args.window)
    
    if success:
        if os.path.exists('email_processing.json'):
            print("\n🎉 Email fetching completed!")
            print("📁 Check email_processing.json for the fetched email data")
            print("📊 State saved to state.json")
            print("🚀 You can now run: python openai_email_processor.py")
        else:
            print("\n✅ Email check completed!")
            print("📅 No new emails since last check.")
            print("📊 Your email processing is up to date.")
    else:
        print("\n❌ Email fetching failed. Please check the error messages above.")
        sys.exit(1)