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
import uuid
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

# Import the main functions from the existing modules
from fetch_mail import fetch_and_save_emails, check_oauth_credentials
from mail_summary import main_mail_summary

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../HomeAssist
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def print_banner():
    """Print a nice banner for the email summarizer."""
    print("=" * 60)
    print("📧 EMAIL SUMMARIZER")
    print("=" * 60)
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

def print_separator(title):
    """Print a section separator."""
    print(f"\n{'=' * 20} {title} {'=' * 20}")

# State management constants
EMAIL_SUMMARIZER_DIRECTORY = "scripts/scheduled/email_summarizer/"
STATE_FILE = os.path.join(EMAIL_SUMMARIZER_DIRECTORY, 'email_script_state.json')

# Where to read/write email AI artifacts
EPHEMERAL_DIR = os.path.join(EMAIL_SUMMARIZER_DIRECTORY, 'ephemeral_data')
SUMMARIES_FILE = os.path.join(EPHEMERAL_DIR, 'email_summaries.json')
NOTIFICATIONS_FILE = os.path.join(EPHEMERAL_DIR, 'email_notifications.json')

load_dotenv()

def load_summaries() -> list:
    """Load generated email summaries from disk (module path preferred)."""
    paths = [SUMMARIES_FILE, os.path.join('ephemeral_data', 'email_summaries.json')]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                continue
    return []

def group_summaries_by_topic(summaries: list) -> dict:
    """Group summaries by category (case-insensitive); fallback to 'general'."""
    groups = {}
    for s in summaries:
        if not isinstance(s, dict):
            continue
        topic = (s.get('category') or 'general').strip().lower()
        groups.setdefault(topic, []).append(s)
    return groups

def build_email_briefs(summaries: list) -> list:
    """Create compact briefs for topic assignment and title/content generation."""
    briefs = []
    for s in summaries:
        if not isinstance(s, dict):
            continue
        briefs.append({
            'id': s.get('original_email_id') or s.get('id'),
            'subject': s.get('original_subject') or s.get('subject') or 'No subject',
            'sender': s.get('original_sender') or s.get('sender') or 'Unknown',
            'summary': s.get('summary') or '',
            '_ref': s,
        })
    return briefs

def ai_assign_topics_to_emails(briefs: list) -> dict:
    """Use Gemini to assign a topic label to each email. Returns {id: topic}."""
    mapping = {}
    try:
        ensure_genai_configured()
        model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        model = genai.GenerativeModel(
            model_name,
            generation_config={
                'temperature': 0.2,
                'max_output_tokens': 800,
                'response_mime_type': 'application/json',
            },
        )
        lines = []
        for b in briefs[:50]:  # cap to keep prompt bounded
            subj = b.get('subject') or 'No subject'
            sender = b.get('sender') or 'Unknown'
            gist = str(b.get('summary') or '')[:240]
            lines.append(f"- id: {b.get('id')}\n  subject: {subj}\n  from: {sender}\n  gist: {gist}")
        emails_block = "\n\n".join(lines)
        prompt = f"""
Assign a concise topic label (1-3 words) to each email below based on subject and gist.
Topics should reflect themes like: scheduling, dining, security alert, academic, job, club, admin, personal.

Emails:
{emails_block}

Return ONLY JSON object mapping id -> topic string. Example:
{{ "1991abc...": "security alert", "1991def...": "academic" }}
"""
        resp = model.generate_content(prompt)
        content = ''
        try:
            for c in getattr(resp, 'candidates', []) or []:
                cont = getattr(c, 'content', None)
                for p in (getattr(cont, 'parts', []) if cont else []):
                    t = getattr(p, 'text', None)
                    if t:
                        content += t
        except Exception:
            try:
                content = resp.text or ''
            except Exception:
                content = ''
        if content:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    for b in briefs:
                        bid = b.get('id')
                        if bid and bid in parsed:
                            topic = str(parsed[bid]).strip().lower() or 'general'
                            mapping[bid] = topic
            except Exception:
                pass
    except Exception:
        pass
    return mapping

def group_by_ai_topics_or_fallback(summaries: list) -> dict:
    """Group summaries by AI-assigned topics, fallback to category/general."""
    briefs = build_email_briefs(summaries)
    id_to_topic = ai_assign_topics_to_emails(briefs)
    groups = {}
    if id_to_topic:
        for b in briefs:
            bid = b.get('id')
            topic = id_to_topic.get(bid)
            if not topic:
                topic = (b.get('_ref', {}).get('category') or 'general').strip().lower()
            groups.setdefault(topic, []).append(b.get('_ref'))
        return groups
    # Fallback to category-based grouping
    return group_summaries_by_topic(summaries)

def ensure_genai_configured():
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
    else:
        # Allow downstream to fallback to non-AI summary
        pass

def ai_generate_title_and_content(topic: str, emails: list) -> tuple:
    """Use Gemini to generate a title and main_content for a notification group.
    Returns (title, content). Falls back to heuristic text if AI unavailable.
    """
    try:
        ensure_genai_configured()
        model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        model = genai.GenerativeModel(
            model_name,
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 800,
                'response_mime_type': 'application/json',
            },
        )

        # Build compact email list
        lines = []
        for e in emails[:20]:  # cap to keep prompt small
            subj = e.get('original_subject') or e.get('subject') or 'No subject'
            sender = e.get('original_sender') or e.get('sender') or 'Unknown'
            summary = e.get('summary') or ''
            lines.append(f"- subject: {subj}\n  from: {sender}\n  gist: {str(summary)[:240]}")
        emails_block = "\n\n".join(lines)

        prompt = f"""
        Generate a maximally concise summarization notification for recent emails grouped by topic.

        Topic: {topic}

        Emails:
        {emails_block}

        Return ONLY valid JSON with:
        {{
        "title": "<short compelling title>",
        "content": "<2-4 sentences summarizing key points concisely>"
        }}
        """
        resp = model.generate_content(prompt)
        # Robust extraction
        content = ""
        try:
            candidates = getattr(resp, 'candidates', []) or []
            for c in candidates:
                cont = getattr(c, 'content', None)
                parts = getattr(cont, 'parts', []) if cont else []
                for p in parts:
                    t = getattr(p, 'text', None)
                    if t:
                        content += t
        except Exception:
            try:
                content = resp.text or ''
            except Exception:
                content = ''

        if content:
            try:
                parsed = json.loads(content)
                title = parsed.get('title') or f"{topic.title()} Updates"
                body = parsed.get('content') or ""
                return title, body
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: heuristic
    subjects = [
        (e.get('original_subject') or e.get('subject') or 'No subject').strip()
        for e in emails[:5]
    ]
    title = f"{topic.title()} (x{len(emails)})"
    body = "Key items: " + "; ".join(subjects)
    return title, body

def build_notifications_from_summaries(summaries: list) -> list:
    groups = group_by_ai_topics_or_fallback(summaries)
    notifications = []
    for topic, emails in groups.items():
        title, content = ai_generate_title_and_content(topic, emails)
        notifications.append({
            'id': uuid.uuid4().hex[:16],
            'notification_type': 'email',
            'title': title,
            'content': content,
            'timestamp': int(time.time()),
            'topic': topic,
            'count': len(emails),
            'email_ids': [e.get('original_email_id') or e.get('id') for e in emails],
        })
    return notifications

def save_notifications_to_file(notifications: list, path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(notifications, f, indent=2, ensure_ascii=False)
        print(f"💾 Wrote {len(notifications)} email topic notification(s) to {path}")
        return True
    except Exception as e:
        print(f"❌ Failed to write notifications: {e}")
        return False

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
            print("⚠️  State file structure is invalid, reinitializing (no backup)...")
            # Create fresh state in place without creating a backup
            initial_state = initialize_empty_state()
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(initial_state, f, indent=2)
            print("✅ Fresh state file created")
            return True
            
    except json.JSONDecodeError:
        print("⚠️  State file contains invalid JSON, reinitializing (no backup)...")
        # Create fresh state in place without creating a backup
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
            # Build topic notifications JSON from summaries
            summaries = load_summaries()
            if summaries:
                notifs = build_notifications_from_summaries(summaries)
                if notifs:
                    save_notifications_to_file(notifs, NOTIFICATIONS_FILE)
                    # Append only to the 'emails' queue in app_state (no main notifications write)
                    try:
                        from core.state_management.statemanager import StateManager
                        recipient = os.getenv("EMAIL_NOTIFICATION_RECIPIENT", "Morgan")
                        state = StateManager()
                        print("Notifications:", notifs)
                        state.add_emails_to_notification_queue(notifs, recipient)
                        print(f"📣 Appended {len(notifs)} email notifications to {recipient}'s 'emails' queue only")
                    except Exception as e:
                        print(f"⚠️  Failed to append email notifications to 'emails' queue: {e}")
            
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