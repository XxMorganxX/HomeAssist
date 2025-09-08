#!/usr/bin/env python3
"""
Email Summarizer Main Script
This script orchestrates the entire email summarization process:
1. Fetches new emails from Gmail
2. Processes them through OpenAI for summarization
3. Generates daily summaries and key points
"""

import sys
import os
import argparse
import json
from datetime import datetime, timezone

# Import the main functions from the existing modules
from fetch_mail import fetch_and_save_emails, check_oauth_credentials
from mail_summary import main_mail_summary

def print_banner():
    """Print a nice banner for the email summarizer."""
    print("=" * 60)
    print("📧 EMAIL SUMMARIZER")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def print_separator(title):
    """Print a section separator."""
    print(f"\n{'=' * 20} {title} {'=' * 20}")

# State management constants
EMAIL_SUMMARIZER_DIRECTORY = "scripts/scheduled/email_summarizer/"
STATE_FILE = os.path.join(EMAIL_SUMMARIZER_DIRECTORY, 'email_script_state.json')

def validate_state_structure(state: dict) -> bool:
    """Validate that the state has the expected structure."""
    required_fields = [
        'processing_history',
        'last_processed_email_id',
        'last_processed_date', 
        'total_emails_processed',
        'processing_run_date'
    ]
    
    # Check all required fields exist
    for field in required_fields:
        if field not in state:
            return False
    
    # Validate processing_history is a list
    if not isinstance(state['processing_history'], list):
        return False
    
    # Validate history entries have required fields
    for entry in state['processing_history']:
        if not isinstance(entry, dict):
            return False
        entry_required = ['last_processed_email_id', 'last_processed_date', 'total_emails_processed']
        for field in entry_required:
            if field not in entry:
                return False
    
    return True

def initialize_empty_state() -> dict:
    """Create an initial empty state structure."""
    return {
        "processing_history": [],
        "last_processed_email_id": None,
        "last_processed_date": None,
        "total_emails_processed": 0,
        "processing_run_date": datetime.now(timezone.utc).isoformat()
    }

def validate_and_init_state() -> bool:
    """Validate and initialize the state file if needed."""
    print("🔍 Checking email processing state...")
    
    # Check if state file exists
    if not os.path.exists(STATE_FILE):
        print(f"📝 State file not found, creating initial state: {os.path.basename(STATE_FILE)}")
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            initial_state = initialize_empty_state()
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(initial_state, f, indent=2)
            print("✅ Initial state file created successfully")
            return True
        except Exception as e:
            print(f"❌ Error creating state file: {e}")
            return False
    
    # Try to load and validate existing state
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # Validate structure
        if validate_state_structure(state):
            # Show current state info
            last_date = state.get('last_processed_date', 'Never')
            last_count = state.get('total_emails_processed', 0)
            history_count = len(state.get('processing_history', []))
            print(f"✅ Valid state found: {last_count} emails processed, {history_count} history entries")
            print(f"   Last processed: {last_date}")
            return True
        else:
            print("⚠️  State file structure is invalid, reinitializing...")
            # Backup corrupted state
            backup_file = STATE_FILE + '.backup'
            if os.path.exists(backup_file):
                os.remove(backup_file)
            os.rename(STATE_FILE, backup_file)
            print(f"📦 Corrupted state backed up to: {os.path.basename(backup_file)}")
            
            # Create fresh state
            initial_state = initialize_empty_state()
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(initial_state, f, indent=2)
            print("✅ Fresh state file created")
            return True
            
    except json.JSONDecodeError:
        print("⚠️  State file contains invalid JSON, reinitializing...")
        # Backup corrupted state
        backup_file = STATE_FILE + '.backup'
        if os.path.exists(backup_file):
            os.remove(backup_file)
        os.rename(STATE_FILE, backup_file)
        print(f"📦 Corrupted state backed up to: {os.path.basename(backup_file)}")
        
        # Create fresh state
        initial_state = initialize_empty_state()
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(initial_state, f, indent=2)
        print("✅ Fresh state file created")
        return True
        
    except Exception as e:
        print(f"❌ Error reading state file: {e}")
        return False

def main():
    """Main function to orchestrate the email summarization process."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Email Summarizer - Fetch and summarize emails from Gmail')
    
    # Add window arguments for historical processing
    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument('--scrape-window-1', action='store_const', const=1, dest='window',
                             help='Use the most recent processing timestamp')
    window_group.add_argument('--scrape-window-2', action='store_const', const=2, dest='window',
                             help='Use the second most recent processing timestamp')
    window_group.add_argument('--scrape-window-3', action='store_const', const=3, dest='window',
                             help='Use the third most recent processing timestamp')
    
    # Add skip options
    parser.add_argument('--skip-fetch', action='store_true',
                       help='Skip email fetching and only process existing emails')
    parser.add_argument('--skip-summary', action='store_true',
                       help='Skip summarization and only fetch emails')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Initialize success tracking
    fetch_success = True
    summary_success = True
    
    # Step 0: Validate and initialize state
    print_separator("STATE INITIALIZATION")
    if not validate_and_init_state():
        print("❌ State initialization failed. Please check file permissions and try again.")
        return 1
    
    # Step 1: Check OAuth credentials
    if not args.skip_fetch:
        print_separator("CHECKING CREDENTIALS")
        if not check_oauth_credentials():
            print("\n❌ OAuth credentials not found. Please fix the issues above and try again.")
            sys.exit(1)
        print("✅ Credentials check passed")
    
    # Step 2: Fetch emails from Gmail
    if not args.skip_fetch:
        print_separator("FETCHING EMAILS")
        
        if args.window:
            print(f"📅 Using historical window {args.window}")
        
        try:
            fetch_success = fetch_and_save_emails(window_index=args.window)
            
            if fetch_success:
                if os.path.exists('ephemeral_data/new_mail.json') or os.path.exists('scripts/scheduled/email_summarizer/ephemeral_data/new_mail.json'):
                    print("✅ Email fetching completed successfully")
                else:
                    print("✅ No new emails to process")
                    if not args.skip_summary:
                        print("\n🎉 Email processing is up to date!")
                        return 0
            else:
                print("❌ Email fetching failed")
                return 1
                
        except Exception as e:
            print(f"❌ Error during email fetching: {e}")
            fetch_success = False
            return 1
    else:
        print_separator("SKIPPING EMAIL FETCH")
        print("⏭️  Email fetching skipped as requested")
    
    # Step 3: Process emails with OpenAI
    if not args.skip_summary and (os.path.exists('ephemeral_data/new_mail.json') or os.path.exists('scripts/scheduled/email_summarizer/ephemeral_data/new_mail.json')):
        print_separator("PROCESSING EMAILS WITH AI")
        
        try:
            main_mail_summary()
            print("✅ Email summarization completed successfully")
            
        except Exception as e:
            print(f"❌ Error during email summarization: {e}")
            summary_success = False
            return 1
    elif args.skip_summary:
        print_separator("SKIPPING SUMMARIZATION")
        print("⏭️  Email summarization skipped as requested")
    
    # Final summary
    print_separator("PROCESS COMPLETE")
    
    if fetch_success and summary_success:
        print("🎉 All tasks completed successfully!")
        
        # List generated files
        print("\n📁 Generated files:")
        files_to_check = [
            ("ephemeral_data/new_mail.json", "Raw email data"),
            ("scripts/scheduled/email_summarizer/ephemeral_data/new_mail.json", "Raw email data (module path)"),
            ("email_script_state.json", "Processing state (root)"),
            ("scripts/scheduled/email_summarizer/email_script_state.json", "Processing state (module path)"),
            ("ephemeral_data/email_summaries.json", "Individual email summaries"),
            ("ephemeral_data/daily_summary.txt", "Daily summary report"),
            ("ephemeral_data/daily_key_points.json", "Key points data")
        ]
        
        for file_path, description in files_to_check:
            if os.path.exists(file_path):
                print(f"  ✓ {file_path} - {description}")
        
        print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0
    else:
        print("⚠️  Some tasks failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())