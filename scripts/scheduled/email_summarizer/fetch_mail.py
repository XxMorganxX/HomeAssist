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
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from pathlib import Path
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
MAX_EMAILS = 40

# Email cache settings
EMAIL_CACHE_FILE = Path(EMAIL_SUMMARIZER_DIRECTORY) / "email_cache.json"
EMAIL_CACHE_RETENTION_DAYS = 30
EMAIL_FETCH_WINDOW_HOURS = 36  # How far back to fetch emails from Gmail


class EmailCache:
    """
    Tracks processed email message IDs to avoid duplicate processing.
    
    Uses the existing Supabase `calendar_event_cache` table with email IDs
    prefixed as `email_<gmail_msg_id>` to distinguish from calendar events.
    
    Storage backends (in order of preference):
    1. Supabase `calendar_event_cache` table (for CI/GitHub Actions)
    2. Local JSON file fallback (for local development)
    """
    
    TABLE_NAME = "calendar_event_cache"
    ID_PREFIX = "email_"
    
    def __init__(self, cache_file: Path = EMAIL_CACHE_FILE, retention_days: int = EMAIL_CACHE_RETENTION_DAYS):
        """Initialize the email cache.
        
        Args:
            cache_file: Path to the JSON cache file (fallback)
            retention_days: Number of days to retain entries (default 30)
        """
        self.cache_file = cache_file
        self.retention_days = retention_days
        self._cache: Dict[str, str] = {}  # prefixed_id -> ISO timestamp when first seen
        self._supabase_client = None
        self._use_supabase = False
        
        # Try Supabase first, fall back to local file
        self._init_supabase()
        self._load()
    
    def _init_supabase(self) -> None:
        """Initialize Supabase client for cache storage."""
        try:
            from supabase import create_client
            
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            
            if url and key:
                self._supabase_client = create_client(url, key)
                self._use_supabase = True
                print("📂 EmailCache: Using Supabase storage")
            else:
                print("📂 EmailCache: Supabase not configured, using local file")
        except ImportError:
            print("📂 EmailCache: supabase package not installed, using local file")
        except Exception as e:
            print(f"⚠️  EmailCache: Supabase init failed ({e}), using local file")
    
    def _prefixed_id(self, email_id: str) -> str:
        """Add prefix to email ID for storage."""
        if email_id.startswith(self.ID_PREFIX):
            return email_id
        return f"{self.ID_PREFIX}{email_id}"
    
    def _load(self) -> None:
        """Load cache from storage."""
        if self._use_supabase:
            self._load_from_supabase()
        else:
            self._load_from_file()
    
    def _load_from_supabase(self) -> None:
        """Load cache from Supabase table (only email entries)."""
        try:
            # Get all non-expired email entries
            cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
            
            response = (
                self._supabase_client.table(self.TABLE_NAME)
                .select("event_id, first_seen")
                .like("event_id", f"{self.ID_PREFIX}%")  # Only email entries
                .gte("first_seen", cutoff)
                .execute()
            )
            
            self._cache = {
                row["event_id"]: row["first_seen"]
                for row in (response.data or [])
            }
            print(f"📂 EmailCache: Loaded {len(self._cache)} cached emails from Supabase")
            
        except Exception as e:
            print(f"⚠️  EmailCache: Error loading from Supabase ({e}), trying local file")
            self._use_supabase = False
            self._load_from_file()
    
    def _load_from_file(self) -> None:
        """Load cache from local JSON file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._cache = data.get("seen_emails", {})
                    print(f"📂 EmailCache: Loaded {len(self._cache)} cached emails from local file")
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  EmailCache: Error loading cache file - {e}")
                self._cache = {}
        else:
            self._cache = {}
            print("📂 EmailCache: No existing cache, starting fresh")
    
    def _save(self) -> None:
        """Save cache to storage."""
        if not self._use_supabase:
            self._save_to_file()
    
    def _save_to_file(self) -> None:
        """Save cache to local JSON file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "retention_days": self.retention_days,
                "seen_emails": self._cache,
            }
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"❌ EmailCache: Error saving cache file - {e}")
    
    def is_seen(self, email_id: str) -> bool:
        """Check if an email has already been processed."""
        prefixed = self._prefixed_id(email_id)
        return prefixed in self._cache
    
    def mark_seen(self, email_id: str) -> None:
        """Mark an email as processed."""
        prefixed = self._prefixed_id(email_id)
        if prefixed not in self._cache:
            now = datetime.now(timezone.utc).isoformat()
            self._cache[prefixed] = now
            
            # Write to Supabase immediately if available
            if self._use_supabase:
                try:
                    self._supabase_client.table(self.TABLE_NAME).upsert({
                        "event_id": prefixed,
                        "first_seen": now,
                    }).execute()
                except Exception as e:
                    print(f"⚠️  EmailCache: Failed to write to Supabase - {e}")
    
    def mark_seen_batch(self, email_ids: List[str]) -> None:
        """Mark multiple emails as processed."""
        now = datetime.now(timezone.utc).isoformat()
        new_ids = []
        
        for email_id in email_ids:
            prefixed = self._prefixed_id(email_id)
            if prefixed not in self._cache:
                new_ids.append(prefixed)
                self._cache[prefixed] = now
        
        if not new_ids:
            return
        
        # Batch write to Supabase if available
        if self._use_supabase:
            try:
                records = [{"event_id": eid, "first_seen": now} for eid in new_ids]
                self._supabase_client.table(self.TABLE_NAME).upsert(records).execute()
                print(f"📝 EmailCache: Saved {len(new_ids)} emails to Supabase")
            except Exception as e:
                print(f"⚠️  EmailCache: Failed to batch write to Supabase - {e}")
        else:
            self._save_to_file()
            print(f"📝 EmailCache: Saved {len(new_ids)} emails to local file")
    
    def filter_unseen_emails(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter a list of emails to only those not yet processed.
        
        Args:
            emails: List of email dicts (must have 'id' field)
            
        Returns:
            List of emails that haven't been processed yet
        """
        unseen = []
        for email in emails:
            email_id = email.get("id", "")
            if email_id and not self.is_seen(email_id):
                unseen.append(email)
        return unseen
    
    def prune_old_entries(self) -> int:
        """Remove entries older than retention period.
        
        Returns:
            Number of entries removed
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        cutoff_iso = cutoff.isoformat()
        old_count = len(self._cache)
        
        # Prune local cache
        self._cache = {
            eid: ts for eid, ts in self._cache.items()
            if datetime.fromisoformat(ts.replace("Z", "+00:00")) > cutoff
        }
        
        removed = old_count - len(self._cache)
        
        # Prune Supabase if available (only email entries)
        if self._use_supabase:
            try:
                self._supabase_client.table(self.TABLE_NAME).delete().like(
                    "event_id", f"{self.ID_PREFIX}%"
                ).lt("first_seen", cutoff_iso).execute()
                if removed > 0:
                    print(f"🧹 EmailCache: Pruned {removed} old entries from Supabase")
            except Exception as e:
                print(f"⚠️  EmailCache: Failed to prune Supabase - {e}")
        elif removed > 0:
            print(f"🧹 EmailCache: Pruned {removed} entries older than {self.retention_days} days")
        
        return removed
    
    def save_and_prune(self) -> None:
        """Prune old entries and save."""
        self.prune_old_entries()
        self._save()
    
    def is_available(self) -> bool:
        """Check if Supabase storage is available."""
        return self._use_supabase
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self._cache),
            "cache_file": str(self.cache_file),
            "retention_days": self.retention_days,
            "using_supabase": self._use_supabase,
        }

class EmailManager:
    def __init__(self, token_file: str = GOOGLE_TOKEN_FILE, credentials_file: str = GOOGLE_CREDENTIALS_FILE):
        """Initialize email manager with Google API credentials."""
        self.token_file = token_file
        self.credentials_file = credentials_file
        # Resolve env overrides and alternate default locations
        self._resolve_credential_paths()
        print(f"Initializing email fetcher with GOOGLE_TOKEN_FILE: {os.path.abspath(self.token_file)} and GOOGLE_CREDENTIALS_FILE: {os.path.abspath(self.credentials_file)}")
        self.service = None
        self.email_cache = EmailCache()
    
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
                
                # Check if Google returned a new refresh token (token rotation)
                if creds.refresh_token and creds.refresh_token != refresh_token:
                    print("🔄 Google issued a new refresh token!")
                    # Write new token to file for GitHub Actions to update the secret
                    try:
                        new_token_file = os.path.join(
                            os.path.dirname(self.token_file) or ".",
                            "new_refresh_token.txt"
                        )
                        os.makedirs(os.path.dirname(new_token_file) or ".", exist_ok=True)
                        with open(new_token_file, 'w') as f:
                            f.write(creds.refresh_token)
                        print(f"📝 New refresh token written to {new_token_file}")
                    except Exception as e:
                        print(f"⚠️  Could not save new refresh token: {e}")
                
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
    
    def fetch_emails_from_gmail(self, max_emails: int = 100) -> List[Dict[str, Any]]:
        """Fetch emails from Gmail using OAuth API.
        
        Fetches emails from the past 36 hours and filters through the EmailCache
        to identify only unprocessed emails.
        """
        emails_data = []
        
        try:
            if not self.service:
                self.authenticate()
            
            # Calculate 36-hour window for fetching
            window_start = datetime.now(timezone.utc) - timedelta(hours=EMAIL_FETCH_WINDOW_HOURS)
            query = f"after:{window_start.strftime('%Y/%m/%d')}"
            
            print(f"📅 Fetching emails from past {EMAIL_FETCH_WINDOW_HOURS} hours")
            print(f"   Window start: {window_start.isoformat()}")
            print(f"   Gmail query: '{query}'")
            
            # Call the Gmail API to fetch messages
            print(f"🔍 Calling Gmail API with max_results={max_emails}...")
            results = self.service.users().messages().list(
                userId='me',
                maxResults=max_emails,
                q=query
            ).execute()
            
            messages = results.get('messages', [])
            print(f"📨 Gmail API returned {len(messages)} message IDs")
            
            if not messages:
                print("📭 No emails found in the time window.")
                return emails_data
            
            print(f"📬 Found {len(messages)} email(s) in time window, checking cache...")
            
            # Process all messages and extract email data
            for message in messages:
                try:
                    # Get the full message
                    msg = self.service.users().messages().get(
                        userId='me',
                        id=message['id']
                    ).execute()
                    
                    # Extract email data
                    email_data = self.extract_email_data(msg)
                    if email_data:
                        emails_data.append(email_data)
                        
                except Exception as e:
                    print(f"Error processing email {message['id']}: {e}")
                    continue
            
            # Filter through EmailCache to get only unprocessed emails
            original_count = len(emails_data)
            emails_data = self.email_cache.filter_unseen_emails(emails_data)
            skipped = original_count - len(emails_data)
            
            if skipped > 0:
                print(f"   ⏭️  Skipped {skipped} already-processed emails (cached)")
            
            if emails_data:
                print(f"✅ Found {len(emails_data)} new email(s) to process")
                # Sort emails by timestamp to ensure proper ordering (newest first)
                emails_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            else:
                print("📭 No new emails to process (all were previously processed)")
                    
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
        """Save email data to JSON file for processing."""
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
            print(f"💾 Saved {len(emails)} emails to {filename}")
            return True
        except Exception as e:
            print(f"Error saving emails to JSON: {e}")
            return False
    
    def mark_emails_processed(self, emails: List[Dict[str, Any]]) -> None:
        """Mark emails as processed in the cache after successful summarization."""
        if not emails:
            return
        
        email_ids = [e.get('id') for e in emails if e.get('id')]
        if email_ids:
            self.email_cache.mark_seen_batch(email_ids)
            self.email_cache.save_and_prune()
            print(f"✅ Marked {len(email_ids)} emails as processed in cache")
    
    def fetch_and_save_emails(self, max_emails: int = 1000) -> tuple:
        """Fetch emails from Gmail and save to JSON file.
        
        Returns:
            Tuple of (success: bool, emails: list) - the emails list is needed
            to mark them as processed after summarization.
        """
        print(f"📧 Fetching emails from past {EMAIL_FETCH_WINDOW_HOURS} hours...")
        
        emails = self.fetch_emails_from_gmail(max_emails)
        
        if emails:
            success = self.save_emails_to_json(emails)
            if success:
                print(f"✅ Successfully fetched and saved {len(emails)} emails")
                return True, emails
            else:
                print("❌ Failed to save emails to JSON")
                return False, []
        else:
            print("ℹ️ No new emails to process.")
            # Proactively remove any stale file so the summarizer does not reprocess old emails
            try:
                self.save_emails_to_json([])
            except Exception:
                pass
            return True, []  # This is not an error condition
         
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

def fetch_and_save_emails() -> tuple:
    """Fetch emails from Gmail and save to ephemeral_data/new_mail.json.
    
    Uses Supabase cache for deduplication - fetches past 36 hours of emails
    and filters out already-processed ones.
    
    Returns:
        Tuple of (success: bool, emails: list, email_manager: EmailManager)
        The email_manager is needed to mark emails as processed after summarization.
    """
    print(f"📧 Fetching emails from Gmail (past {EMAIL_FETCH_WINDOW_HOURS} hours)...")
    
    try:
        email_manager = EmailManager()
        success, emails = email_manager.fetch_and_save_emails(max_emails=MAX_EMAILS)
        return success, emails, email_manager
    except Exception as e:
        print(f"❌ Error fetching emails: {e}")
        return False, [], None


def main_fetch_mail():
    success, emails, _ = fetch_and_save_emails()
    return success

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fetch emails from Gmail')
    parser.parse_args()
    
    print("=" * 60)
    print("📧 EMAIL FETCHER")
    print(f"   Fetch window: {EMAIL_FETCH_WINDOW_HOURS} hours")
    print("   Cache: Supabase (calendar_event_cache table)")
    print("=" * 60)
    
    # Check OAuth credentials
    if not check_oauth_credentials():
        print("\n❌ OAuth credentials not found. Please fix the issues above and try again.")
        sys.exit(1)
    
    # Fetch and save emails
    print("\n" + "=" * 60)
    success, emails, email_manager = fetch_and_save_emails()
    
    if success:
        if emails:
            print("\n🎉 Email fetching completed!")
            print(f"📁 Saved {len(emails)} emails to ephemeral_data/new_mail.json")
            print("🚀 Run mail_main.py to process and summarize these emails")
        else:
            print("\n✅ Email check completed!")
            print("📭 No new emails to process (all cached)")
    else:
        print("\n❌ Email fetching failed. Please check the error messages above.")
        sys.exit(1)