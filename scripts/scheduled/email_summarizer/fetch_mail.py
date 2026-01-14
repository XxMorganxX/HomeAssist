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
GOOGLE_TOKEN_FILE = "creds/email_summarizer_token.json"                    # Google OAuth token file
GOOGLE_CREDENTIALS_FILE = "creds/email_summarizer_credentials.json"  # Google service account credentials
EMAIL_SUMMARIZER_DIRECTORY = "scripts/scheduled/email_summarizer/"
# =============================================================================

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build
import base64

# Load environment variables
load_dotenv()

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
STATE_FILE = os.path.join(EMAIL_SUMMARIZER_DIRECTORY, 'email_script_state.json')
MAX_EMAILS = 10

class EmailManager:
    def __init__(self, token_file: str = GOOGLE_TOKEN_FILE, credentials_file: str = GOOGLE_CREDENTIALS_FILE, window_index: int = None):
        """Initialize email manager with Google API credentials."""
        self.token_file = token_file
        self.credentials_file = credentials_file
        # Resolve env overrides and alternate default locations
        self._resolve_credential_paths()
        print(f"Initializing async email fetcher with GOOGLE_TOKEN_FILE: {os.path.abspath(self.token_file)} and GOOGLE_CREDENTIALS_FILE: {os.path.abspath(self.credentials_file)}")
        self.service = None
        self.window_index = window_index
        self.last_processed_date_str, self.last_processed_timestamp = self.load_last_processed_date(window_index)
        print(f"🕒 Loaded state: date_str='{self.last_processed_date_str}', timestamp={self.last_processed_timestamp}")
    
    def _is_headless(self) -> bool:
        """Detect if running in a headless/non-interactive environment."""
        try:
            return (
                os.getenv("EMAIL_SUMMARIZER_HEADLESS", "0") == "1"
                or os.getenv("CI") == "true"
                or not sys.stdout.isatty()
            )
        except Exception:
            return True

    def _notify_reauth_required(self, reason: str) -> None:
        """Log that re-authorization is required. Do not write to app_state."""
        try:
            print(f"⚠️  Gmail re-authorization required: {reason}")
            print("   Please run: python scripts/scheduled/email_summarizer/reauth_gmail.py")
        except Exception:
            pass

    def _resolve_credential_paths(self):
        """Resolve token and credential file paths with env overrides and fallbacks."""
        try:
            env_token = os.getenv("EMAIL_SUMMARIZER_TOKEN_FILE")
            env_creds = os.getenv("EMAIL_SUMMARIZER_CREDENTIALS_FILE")

            if env_token:
                self.token_file = env_token
            elif not os.path.exists(self.token_file):
                alt_token = "scripts/scheduled/email_summarizer/creds/email_summarizer_token.json"
                if os.path.exists(alt_token):
                    self.token_file = alt_token

            if env_creds:
                self.credentials_file = env_creds
            elif not os.path.exists(self.credentials_file):
                alt_creds = "scripts/scheduled/email_summarizer/creds/email_summarizer_credentials.json"
                if os.path.exists(alt_creds):
                    self.credentials_file = alt_creds
        except Exception:
            pass

    def _try_env_refresh_credentials(self):
        """Create Credentials from environment refresh token without browser."""
        client_id = os.getenv("GMAIL_CLIENT_ID") or os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GMAIL_CLIENT_SECRET") or os.getenv("GOOGLE_CLIENT_SECRET")
        refresh_token = os.getenv("GMAIL_REFRESH_TOKEN") or os.getenv("GOOGLE_REFRESH_TOKEN")

        if client_id and client_secret and refresh_token:
            creds = Credentials(
                None,
                refresh_token=refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=client_id,
                client_secret=client_secret,
                scopes=SCOPES
            )
            try:
                creds.refresh(Request())
                # Persist refreshed credentials to token file for reuse
                try:
                    os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
                    with open(self.token_file, 'w') as token:
                        token.write(creds.to_json())
                except Exception:
                    pass
                print("🔑 Using environment-based refresh token for Gmail API.")
                return creds
            except Exception as e:
                print(f"⚠️  Failed to refresh using environment variables: {e}")
                if self._is_headless():
                    self._notify_reauth_required("Environment refresh token failed to refresh Gmail access.")
        return None
        
    def authenticate(self):
        """Authenticate using OAuth2 credentials."""
        creds = None
        
        # Try environment-based refresh token first (headless, no browser)
        try:
            env_creds = self._try_env_refresh_credentials()
            if env_creds:
                creds = env_creds
        except Exception as e:
            print(f"⚠️  Env-based OAuth refresh failed: {e}")
        
        # Token file stores the user's access and refresh tokens
        if not creds and os.path.exists(self.token_file):
            with open(self.token_file, 'r') as token:
                creds = Credentials.from_authorized_user_info(json.load(token), SCOPES)
            # If token file lacks a refresh token, upgrade to offline access if interactive is allowed
            if creds and not getattr(creds, 'refresh_token', None):
                if self._is_headless():
                    self._notify_reauth_required("Existing Gmail credentials are missing a refresh token (offline access).")
                    raise RuntimeError("Headless mode: missing refresh token. Run the Gmail reauth script to grant offline access.")
                print("♻️  Upgrading Gmail credentials to include refresh token (offline access)...")
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0, access_type='offline', prompt='consent')
                os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
                with open(self.token_file, 'w') as token:
                    token.write(creds.to_json())
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and getattr(creds, 'refresh_token', None):
                # Try to refresh first; if that fails with invalid_grant, wipe token and re-auth
                try:
                    creds.refresh(Request())
                except RefreshError as e:
                    print(f"⚠️  OAuth refresh failed ({e}). Removing stale token and reauthorizing...")
                    if 'invalid_grant' in str(e):
                        self._notify_reauth_required("Gmail refresh token has been revoked or expired.")
                    try:
                        os.remove(self.token_file)
                    except Exception:
                        pass
                    if self._is_headless():
                        raise RuntimeError(
                            "Headless mode: cannot perform interactive OAuth. "
                            "Run: python scripts/scheduled/email_summarizer/reauth_gmail.py"
                        )
                    creds = None
            
            if not creds or not creds.valid:
                if self._is_headless():
                    raise RuntimeError(
                        "Headless mode: Gmail OAuth reauthorization required. "
                        "Run: python scripts/scheduled/email_summarizer/reauth_gmail.py"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0, access_type='offline', prompt='consent')
            
            # Save the credentials for the next run
            os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        try:
            self.service = build('gmail', 'v1', credentials=creds)
        except Exception as e:
            print(f"❌ Failed to initialize Gmail client: {e}")
            raise
        return self.service
    
    def load_last_processed_date(self, window_index: int = None) -> tuple:
        """Load the last processed date from email_script_state.json."""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as file:
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
                                print("   Using most recent entry instead.")
                                last_run = history[-1]
                        else:
                            # Default to most recent entry
                            last_run = history[-1]
                        
                        last_date = last_run.get('last_processed_email_date') or last_run.get('last_processed_date')
                    else:
                        # Fallback to old format
                        if window_index is not None:
                            print("⚠️ Warning: No processing history available, ignoring window parameter")
                        last_date = state.get('last_processed_email_date') or state.get('last_processed_date')
                    
                    if last_date:
                        # Keep the full timestamp for precise filtering
                        dt = datetime.fromisoformat(last_date.replace('Z', '+00:00'))
                        # Normalize to timezone-aware (UTC) to avoid naive/aware comparison issues
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        # Gmail's 'after:' search is inclusive of the date, but we want emails AFTER our timestamp
                        # Since we have a precise timestamp, use the same date but rely on timestamp filtering
                        date_str = dt.strftime('%Y/%m/%d')
                        print(f"🔍 Using timestamp: {last_date} -> Gmail format: {date_str}")
                        print(f"   Note: Gmail 'after:' is inclusive, will filter by precise timestamp: {dt.isoformat()}")
                        return date_str, dt
        except Exception as e:
            print(f"Note: Could not load previous state: {e}")
        return None, None
    
    def fetch_emails_from_gmail(self, max_emails: int = 100) -> List[Dict[str, Any]]:
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
                print(f"   Gmail query: '{query}'")
            else:
                print("📅 No previous state found. Fetching recent emails...")
            
            # Call the Gmail API to fetch messages
            print(f"🔍 Calling Gmail API with max_results={max_emails}...")
            results = self.service.users().messages().list(
                userId='me',
                maxResults=max_emails,  # Fetch more to account for filtering
                q=query
            ).execute()
            
            messages = results.get('messages', [])
            print(f"📨 Gmail API returned {len(messages)} message IDs")
            
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
                        # Ensure both are tz-aware; normalize email_timestamp to UTC if needed
                        if email_timestamp and email_timestamp.tzinfo is None:
                            email_timestamp = email_timestamp.replace(tzinfo=timezone.utc)
                        if email_timestamp and email_timestamp <= self.last_processed_timestamp:
                            # Extract subject for debug message
                            subject = "No subject"
                            try:
                                headers = msg.get('payload', {}).get('headers', [])
                                for header in headers:
                                    if header['name'].lower() == 'subject':
                                        subject = header['value'][:50]
                                        break
                            except:
                                pass
                            print(f"   ⏭️  Skipping already processed: {subject} (date: {email_timestamp})")
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
                # Sort emails by timestamp to ensure proper ordering (newest first)
                emails_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                    
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
            
            # Get the actual timestamp
            email_timestamp = self.get_email_timestamp(msg)
            
            return {
                "id": msg['id'],
                "subject": subject,
                "sender": sender,
                "date": date,
                "timestamp": email_timestamp.isoformat() if email_timestamp else None,
                "snippet": snippet,
                "body": body,
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            print(f"Error extracting email data: {e}")
            return None
    
    def get_email_body(self, payload: Dict[str, Any]) -> str:
        """Extract the text body from a Gmail API payload."""
        def decode_b64_to_text(b64_str: str) -> str:
            try:
                # Add padding if missing
                padding = '=' * (-len(b64_str) % 4)
                return base64.urlsafe_b64decode(b64_str + padding).decode('utf-8', errors='ignore')
            except Exception:
                return ""

        # Prefer text/plain, then text/html, search recursively as needed
        if not isinstance(payload, dict):
            return ""

        # Direct body on this payload
        body_dict = payload.get('body', {}) if isinstance(payload.get('body', {}), dict) else {}
        data_here = body_dict.get('data')
        if data_here:
            return decode_b64_to_text(data_here)

        # If there are parts, iterate
        parts = payload.get('parts', [])
        if isinstance(parts, list):
            # First pass: look for text/plain
            for part in parts:
                mime = part.get('mimeType')
                if mime == 'text/plain':
                    data = part.get('body', {}).get('data')
                    if data:
                        return decode_b64_to_text(data)
            # Second pass: allow text/html
            for part in parts:
                mime = part.get('mimeType')
                if mime == 'text/html':
                    data = part.get('body', {}).get('data')
                    if data:
                        return decode_b64_to_text(data)
            # Third pass: recurse into multipart children
            for part in parts:
                if 'parts' in part or isinstance(part.get('body'), dict):
                    nested = self.get_email_body(part)
                    if nested:
                        return nested

        return ""
    
    def get_email_timestamp(self, msg: Dict[str, Any]) -> datetime:
        """Extract the timestamp from a Gmail API message."""
        try:
            # Get internal timestamp from Gmail (milliseconds since epoch)
            internal_date = msg.get('internalDate')
            if internal_date:
                # Convert milliseconds to seconds
                timestamp = int(internal_date) / 1000
                return datetime.fromtimestamp(timestamp, timezone.utc)
        except Exception as e:
            print(f"Error extracting timestamp: {e}")
        return None
    
    def save_emails_to_json(self, emails: List[Dict[str, Any]], filename: str = (EMAIL_SUMMARIZER_DIRECTORY+"ephemeral_data/new_mail.json")):
        """Save email data to JSON file."""
        print(f"Saving {len(emails)} emails to {os.path.abspath(filename)}")
        
        # Only create the file if there are emails to save
        if not emails:
            print("📭 No new emails to save.")
            # Clean up any existing file to ensure old emails aren't processed
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    print(f"🧹 Cleaned up old {filename}")
                except Exception as e:
                    print(f"Warning: Could not remove old file: {e}")
            return True  # This is not an error condition
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
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
        """Save processing state to email_script_state.json."""
        try:
            # Load existing state to preserve history
            existing_state = {}
            if os.path.exists(STATE_FILE):
                try:
                    with open(STATE_FILE, 'r') as file:
                        existing_state = json.load(file)
                except Exception:
                    pass
            
            # Get the most recent email (first in the list from Gmail API)
            last_email = emails[0] if emails else None
            
            # Get the timestamp of the most recent email
            last_email_timestamp = None
            if last_email:
                # Use the timestamp we already extracted and stored
                last_email_timestamp = last_email.get('timestamp')
                if not last_email_timestamp:
                    # Fallback to current time if somehow missing
                    last_email_timestamp = datetime.now(timezone.utc).isoformat()
            
            # Create new processing entry
            ts = last_email_timestamp or datetime.now(timezone.utc).isoformat()
            new_entry = {
                "last_processed_email_id": last_email['id'] if last_email else None,
                "last_processed_email_date": ts,
                "last_processed_date": ts,  # backward compatibility
                "processing_run_date": datetime.now(timezone.utc).isoformat(),
                "total_emails_processed": len(emails)
            }
            
            # Initialize or update processing history
            if 'processing_history' not in existing_state:
                # Migrate old format if it exists
                if 'last_processed_email_id' in existing_state:
                    migrated_ts = existing_state.get('last_processed_email_date') or existing_state.get('last_processed_date')
                    existing_state['processing_history'] = [{
                        "last_processed_email_id": existing_state.get('last_processed_email_id'),
                        "last_processed_email_date": migrated_ts,
                        "last_processed_date": migrated_ts,
                        "total_emails_processed": existing_state.get('total_emails_processed', 0)
                    }]
                else:
                    existing_state['processing_history'] = []
            
            # Append new entry and keep only last 3
            existing_state['processing_history'].append(new_entry)
            existing_state['processing_history'] = existing_state['processing_history'][-3:]
            
            # Also keep the latest values at root level for backwards compatibility
            existing_state.update(new_entry)
            
            with open(STATE_FILE, 'w', encoding='utf-8') as file:
                json.dump(existing_state, file, indent=2)
            
            print(f"💾 Saved processing state to {os.path.abspath(STATE_FILE)}")
            print(f"   Processing history entries: {len(existing_state['processing_history'])}")
            
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def fetch_and_save_emails(self, max_emails: int = 1000) -> bool:
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
            # Proactively remove any stale file so the summarizer does not reprocess old emails
            try:
                self.save_emails_to_json([])
            except Exception:
                pass
            if self.window_index is not None:
                print(f"💾 State update skipped (using historical window {self.window_index})")
            return True  # This is not an error condition
         
def check_oauth_credentials():
    """Check if OAuth credentials are properly set up (env, token, or file)."""
    print("🔍 Checking OAuth credentials...")
    
    # 1) Environment-based refresh token (preferred for headless)
    has_env = all([
        os.getenv("GMAIL_CLIENT_ID") or os.getenv("GOOGLE_CLIENT_ID"),
        os.getenv("GMAIL_CLIENT_SECRET") or os.getenv("GOOGLE_CLIENT_SECRET"),
        os.getenv("GMAIL_REFRESH_TOKEN") or os.getenv("GOOGLE_REFRESH_TOKEN"),
    ])
    if has_env:
        print("✅ Found environment variables for client ID/secret and refresh token")
        return True
    
    # 2) Existing token file (contains refresh token) at default or alt location
    token_candidates = [
        GOOGLE_TOKEN_FILE,
        "scripts/scheduled/email_summarizer/creds/email_summarizer_token.json",
    ]
    for p in token_candidates:
        if os.path.exists(p):
            print(f"✅ Found token file at {os.path.abspath(p)}")
            return True
    
    # 3) Client credentials file (for one-time consent to obtain refresh token)
    creds_candidates = [
        GOOGLE_CREDENTIALS_FILE,
        "scripts/scheduled/email_summarizer/creds/email_summarizer_credentials.json",
    ]
    for p in creds_candidates:
        if os.path.exists(p):
            print(f"✅ OAuth client credentials file found at {os.path.abspath(p)}")
            return True
    
    # Nothing found
    print("❌ No OAuth setup found.")
    print("   Provide either:")
    print("   - Env vars: GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN")
    print("     (or GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET/GOOGLE_REFRESH_TOKEN), or")
    print("   - A token file at one of:")
    for p in token_candidates:
        print(f"     {os.path.abspath(p)}")
    print("   - A client credentials file at one of:")
    for p in creds_candidates:
        print(f"     {os.path.abspath(p)}")
    return False

def fetch_and_save_emails(window_index: int = None):
    """Fetch emails from Gmail and save to ephemeral_data/new_mail.json."""
    print("📧 Fetching emails from Gmail...")
    
    # Show current state information
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            last_date = state.get('last_processed_email_date') or state.get('last_processed_date', 'Never')
            last_count = state.get('total_emails_processed', 0)
            print(f"📊 Previous run: {last_count} emails processed, last date: {last_date}")
    
    try:
        email_manager = EmailManager(window_index=window_index)
        success = email_manager.fetch_and_save_emails(max_emails=MAX_EMAILS)
        return success
    except Exception as e:
        print(f"❌ Error fetching emails: {e}")
        return False


def main_fetch_mail():
    fetch_and_save_emails()

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
        if os.path.exists('ephemeral_data/new_mail.json'):
            print("\n🎉 Email fetching completed!")
            print("📁 Check ephemeral_data/new_mail.json for the fetched email data")
            print("📊 State saved to email_script_state.json")
            print("🚀 You can now run: python openai_email_processor.py")
        else:
            print("\n✅ Email check completed!")
            print("📅 No new emails since last check.")
            print("📊 Your email processing is up to date.")
    else:
        print("\n❌ Email fetching failed. Please check the error messages above.")
        sys.exit(1)