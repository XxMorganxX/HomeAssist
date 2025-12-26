#!/usr/bin/env python3
"""
Gmail Re-authorization Script

This script helps you:
1. Authorize Gmail access via browser (one-time)
2. Extract the refresh token for headless/automated use
3. Optionally save credentials to .env for persistent access

Run this whenever you need to re-authorize Gmail access.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv, set_key
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Load existing env
load_dotenv(PROJECT_ROOT / ".env")

# Configuration
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
TOKEN_FILE = PROJECT_ROOT / "google_creds" / "email_summarizer_token.json"
CREDENTIALS_FILE = PROJECT_ROOT / "google_creds" / "email_summarizer_credentials.json"
ENV_FILE = PROJECT_ROOT / ".env"


def main():
    print("=" * 60)
    print("📧 GMAIL RE-AUTHORIZATION")
    print("=" * 60)
    
    # Check for credentials file
    if not CREDENTIALS_FILE.exists():
        print(f"\n❌ OAuth credentials file not found: {CREDENTIALS_FILE}")
        print("\nTo set up Gmail API access:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a project or select existing one")
        print("3. Enable Gmail API")
        print("4. Create OAuth 2.0 credentials (Desktop app)")
        print("5. Download the JSON and save as:")
        print(f"   {CREDENTIALS_FILE}")
        print("\n⚠️  IMPORTANT: In the OAuth consent screen, set the app to")
        print("   'Production' mode to get long-lived refresh tokens!")
        return False
    
    print(f"\n✅ Found credentials file: {CREDENTIALS_FILE}")
    
    # Load client info for .env export
    with open(CREDENTIALS_FILE) as f:
        client_config = json.load(f)
    
    # Handle both web and installed app credentials
    if "web" in client_config:
        client_info = client_config["web"]
    elif "installed" in client_config:
        client_info = client_config["installed"]
    else:
        print("❌ Invalid credentials file format")
        return False
    
    client_id = client_info.get("client_id")
    client_secret = client_info.get("client_secret")
    
    print("\n🔐 Starting OAuth flow...")
    print("   A browser window will open for authorization.")
    print("   Grant 'Read email messages' permission.\n")
    
    try:
        # Run OAuth flow with offline access for refresh token
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_FILE),
            SCOPES,
            redirect_uri='http://localhost:8080/'
        )
        
        # Request offline access to get refresh token
        creds = flow.run_local_server(
            port=8080,
            access_type='offline',
            prompt='consent'  # Force consent to ensure refresh token is returned
        )
        
        print("\n✅ Authorization successful!")
        
        # Save token file
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_FILE, 'w') as f:
            f.write(creds.to_json())
        print(f"✅ Token saved to: {TOKEN_FILE}")
        
        # Check for refresh token
        if creds.refresh_token:
            print(f"\n🔑 Refresh token obtained!")
            
            # Offer to save to .env
            print("\n" + "=" * 60)
            print("💾 SAVE FOR HEADLESS/AUTOMATED USE")
            print("=" * 60)
            print("\nTo run the email summarizer without browser authorization,")
            print("add these to your .env file:\n")
            print(f"GMAIL_CLIENT_ID={client_id}")
            print(f"GMAIL_CLIENT_SECRET={client_secret}")
            print(f"GMAIL_REFRESH_TOKEN={creds.refresh_token}")
            
            save = input("\n\nSave to .env automatically? [y/N]: ").strip().lower()
            if save == 'y':
                # Create .env if it doesn't exist
                if not ENV_FILE.exists():
                    ENV_FILE.touch()
                
                set_key(str(ENV_FILE), "GMAIL_CLIENT_ID", client_id)
                set_key(str(ENV_FILE), "GMAIL_CLIENT_SECRET", client_secret)
                set_key(str(ENV_FILE), "GMAIL_REFRESH_TOKEN", creds.refresh_token)
                print(f"\n✅ Saved to {ENV_FILE}")
                print("   Email summarizer will now work without browser auth!")
            else:
                print("\n📋 Copy the values above to your .env file manually.")
        else:
            print("\n⚠️  No refresh token returned!")
            print("   This usually means the OAuth app is in 'Testing' mode.")
            print("   Go to Google Cloud Console → OAuth consent screen → Publish App")
        
        # Test the connection
        print("\n🧪 Testing Gmail connection...")
        from googleapiclient.discovery import build
        service = build('gmail', 'v1', credentials=creds)
        profile = service.users().getProfile(userId='me').execute()
        print(f"✅ Connected as: {profile.get('emailAddress')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Authorization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 60)
    if success:
        print("✅ Gmail authorization complete!")
        print("   You can now run the email summarizer.")
    else:
        print("❌ Gmail authorization failed.")
        print("   Please check the errors above.")
    print("=" * 60)
