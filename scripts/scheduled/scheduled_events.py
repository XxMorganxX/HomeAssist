#!/usr/bin/env python3
"""
Scheduled Events Runner
Runs email summarizer and news summary scripts nightly via cron
"""

import subprocess
import sys
import os
from datetime import datetime
import logging
import json
import shutil

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'scheduled_events_{datetime.now().strftime("%Y%m%d")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def cleanup_ephemeral_data():
    """Clean up ephemeral data directories before running scripts"""
    logging.info("🧹 Cleaning up ephemeral data directories...")
    
    # Define ephemeral data directories
    ephemeral_dirs = [
        os.path.join(os.path.dirname(__file__), 'email_summarizer', 'ephemeral_data'),
        os.path.join(os.path.dirname(__file__), 'news_summary', 'ephemeral_data')
    ]
    
    for dir_path in ephemeral_dirs:
        if os.path.exists(dir_path):
            try:
                # List files before deletion for logging
                files = os.listdir(dir_path)
                if files:
                    logging.info(f"Removing {len(files)} files from {dir_path}")
                    for file in files:
                        file_path = os.path.join(dir_path, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            logging.debug(f"Removed: {file_path}")
                    logging.info(f"✅ Cleaned {dir_path}")
                else:
                    logging.info(f"📁 No files to clean in {dir_path}")
            except Exception as e:
                logging.error(f"Error cleaning {dir_path}: {e}")
        else:
            logging.info(f"📁 Directory does not exist, creating: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)

def run_email_summarizer():
    """Run the email summarizer script"""
    script_path = os.path.join(os.path.dirname(__file__), 'email_summarizer', 'main.py')
    
    logging.info("Starting email summarizer...")
    logging.info(f"Script path: {script_path}")
    
    try:
        start_time = datetime.now()
        
        # Run with real-time output
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(script_path),
            bufsize=1,
            universal_newlines=True
        )
        
        # Collect output
        stdout_lines = []
        stderr_lines = []
        
        # Read output in real-time
        for line in process.stdout:
            line = line.rstrip()
            stdout_lines.append(line)
            # Add prefix to distinguish email summarizer output
            logging.info(f"[EMAIL] {line}")
        
        # Wait for process to complete
        process.wait()
        
        # Read any stderr
        stderr = process.stderr.read()
        if stderr:
            stderr_lines = stderr.strip().split('\n')
            for line in stderr_lines:
                logging.error(f"[EMAIL ERROR] {line}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        if process.returncode == 0:
            logging.info(f"✅ Email summarizer completed successfully in {duration:.1f}s")
            
            # Check for generated files
            ephemeral_dir = os.path.join(os.path.dirname(script_path), 'ephemeral_data')
            if os.path.exists(ephemeral_dir):
                files = os.listdir(ephemeral_dir)
                if files:
                    logging.info(f"📁 Generated {len(files)} files: {', '.join(files)}")
            
            return True
        else:
            logging.error(f"❌ Email summarizer failed with code {process.returncode} after {duration:.1f}s")
            return False
            
    except Exception as e:
        logging.error(f"❌ Exception running email summarizer: {type(e).__name__}: {e}")
        return False

def run_news_summary():
    """Run the news summary script"""
    script_path = os.path.join(os.path.dirname(__file__), 'news_summary', 'main.py')
    
    logging.info("Starting news summary...")
    logging.info(f"Script path: {script_path}")
    
    try:
        start_time = datetime.now()
        
        # Run with real-time output
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(script_path),
            bufsize=1,
            universal_newlines=True
        )
        
        # Collect output
        stdout_lines = []
        stderr_lines = []
        
        # Read output in real-time
        for line in process.stdout:
            line = line.rstrip()
            stdout_lines.append(line)
            # Add prefix to distinguish news summary output
            logging.info(f"[NEWS] {line}")
        
        # Wait for process to complete
        process.wait()
        
        # Read any stderr
        stderr = process.stderr.read()
        if stderr:
            stderr_lines = stderr.strip().split('\n')
            for line in stderr_lines:
                logging.error(f"[NEWS ERROR] {line}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        if process.returncode == 0:
            logging.info(f"✅ News summary completed successfully in {duration:.1f}s")
            
            # Check for generated files
            ephemeral_dir = os.path.join(os.path.dirname(script_path), 'ephemeral_data')
            if os.path.exists(ephemeral_dir):
                files = os.listdir(ephemeral_dir)
                if files:
                    logging.info(f"📁 Generated {len(files)} files: {', '.join(files)}")
            
            return True
        else:
            logging.error(f"❌ News summary failed with code {process.returncode} after {duration:.1f}s")
            return False
            
    except Exception as e:
        logging.error(f"❌ Exception running news summary: {type(e).__name__}: {e}")
        return False

def export_notifications():
    """Export relevant notifications from email and news summaries to app_state.json"""
    logging.info("Starting notification export...")
    
    # Direct path to app_state.json
    app_state_path = os.path.join(os.path.dirname(__file__), '..', '..', 'core', 'state_management', 'app_state.json')
    
    # Load current state
    try:
        with open(app_state_path, 'r') as f:
            app_state = json.load(f)
    except Exception as e:
        logging.error(f"Error loading app_state.json: {e}")
        return 0
    
    notifications_added = 0
    
    # Process email notifications
    email_base_path = os.path.join(os.path.dirname(__file__), 'email_summarizer', 'ephemeral_data')
    
    # Read email summaries for high-priority email items
    try:
        summaries_path = os.path.join(email_base_path, 'email_summaries.json')
        if os.path.exists(summaries_path):
            with open(summaries_path, 'r') as f:
                email_summaries = json.load(f)
                
            # Extract high priority and job-related emails
            for email_data in email_summaries:
                    priority = email_data.get('priority', 'medium')
                    category = email_data.get('category', '')
                    
                    # Create notifications for high priority or job-related emails
                    original_sender = email_data.get('original_sender', '').lower()
                    if priority == 'high' or 'job' in category.lower() or 'handshake' in original_sender or email_data.get('action_required', False):
                        # Determine recipient based on content
                        recipient = "Morgan"  # Default to Morgan for job opportunities
                        
                        # Format notification content
                        subject = email_data.get('original_subject', 'No subject')
                        
                        # Extract summary from the nested JSON structure
                        raw_response = email_data.get('raw_response', '')
                        summary = email_data.get('summary', '')
                        key_points = email_data.get('key_points', [])
                        
                        # Try to parse the nested JSON if present
                        if raw_response and '```json' in raw_response:
                            try:
                                json_start = raw_response.find('{')
                                json_end = raw_response.rfind('}') + 1
                                if json_start >= 0 and json_end > json_start:
                                    parsed_data = json.loads(raw_response[json_start:json_end])
                                    summary = parsed_data.get('summary', summary)
                                    key_points = parsed_data.get('key_points', key_points)
                                    priority = parsed_data.get('priority', priority)
                            except:
                                pass
                        
                        notification_content = f"📧 Email: {subject}\n"
                        if summary:
                            notification_content += f"Summary: {summary}\n"
                        if key_points:
                            notification_content += "Key points:\n"
                            for point in key_points[:2]:  # Limit to 2 points
                                notification_content += f"• {point}\n"
                        
                        # Create and add notification
                        notification = {
                            "intended_recipient": recipient,
                            "notification_content": notification_content.strip(),
                            "relevant_when": "next_24_hours"
                        }
                        
                        # Add to notification queue
                        if recipient in app_state["autonomous_state"]["notification_queue"]:
                            app_state["autonomous_state"]["notification_queue"][recipient]["notifications"].append(notification)
                            notifications_added += 1
                            logging.info(f"Added email notification for {recipient}: {subject}")
                        
    except Exception as e:
        logging.error(f"Error processing email notifications: {e}")
    
    # Process news notifications
    news_base_path = os.path.join(os.path.dirname(__file__), 'news_summary', 'ephemeral_data')
    
    # Read full article analysis for tech/AI news
    try:
        analysis_path = os.path.join(news_base_path, 'full_article_analysis.json')
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r') as f:
                articles = json.load(f)
                
            for article_key, article_data in articles.items():
                # Look for tech innovation and AI-related content
                tech_relevance = article_data.get('innovation_score', 0)
                key_insights = article_data.get('key_insights', [])
                
                if tech_relevance >= 7:  # High tech relevance
                    # Determine recipient based on content
                    recipient = "Morgan"  # Default for tech news
                    if any(keyword in str(article_data).lower() for keyword in ['music', 'spotify', 'audio']):
                        recipient = "Spencer"
                    
                    # Format notification
                    title = article_data.get('title', 'Tech News')
                    summary = article_data.get('summary', '')
                    
                    notification_content = f"📰 Tech News: {title}\n"
                    if summary:
                        # Truncate summary to first 100 characters
                        truncated_summary = summary[:100] + "..." if len(summary) > 100 else summary
                        notification_content += f"{truncated_summary}\n"
                    if key_insights:
                        notification_content += "Key insights:\n"
                        for insight in key_insights[:2]:  # Limit to 2 insights
                            notification_content += f"• {insight}\n"
                    
                    # Create and add notification
                    notification = {
                        "intended_recipient": recipient,
                        "notification_content": notification_content.strip(),
                        "relevant_when": "next_24_hours"
                    }
                    
                    # Add to notification queue
                    if recipient in app_state["autonomous_state"]["notification_queue"]:
                        app_state["autonomous_state"]["notification_queue"][recipient]["notifications"].append(notification)
                        notifications_added += 1
                        logging.info(f"Added news notification for {recipient}: {title}")
                    
    except Exception as e:
        logging.error(f"Error processing news notifications: {e}")
    
    # Save updated state back to file
    try:
        with open(app_state_path, 'w') as f:
            json.dump(app_state, f, indent=2)
        logging.info(f"Successfully saved app_state.json with {notifications_added} new notifications")
    except Exception as e:
        logging.error(f"Error saving app_state.json: {e}")
        return 0
    
    logging.info(f"Notification export complete. Added {notifications_added} notifications.")
    return notifications_added



def main():
    """Main function to run all scheduled events"""
    logging.info("=" * 60)
    logging.info("SCHEDULED EVENTS STARTING")
    logging.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)
    
    # Clean up ephemeral data before running scripts
    logging.info("\n[SETUP] PREPARING ENVIRONMENT")
    cleanup_ephemeral_data()
    
    # Track results
    results = {
        'email': False,
        'news': False
    }
    
    # Run email summarizer
    logging.info("\n" + "-" * 60)
    logging.info("[1/2] EMAIL SUMMARIZER")
    logging.info("-" * 60)
    results['email'] = run_email_summarizer()
    
    # Run news summary
    logging.info("\n" + "-" * 60)
    logging.info("[2/2] NEWS SUMMARY")
    logging.info("-" * 60)
    results['news'] = run_news_summary()
    
    # Export notifications if both tasks succeeded
    if results['email'] and results['news']:
        logging.info("\n" + "-" * 60)
        logging.info("[3/3] EXPORTING NOTIFICATIONS")
        logging.info("-" * 60)
        notifications_count = export_notifications()
        results['notifications'] = notifications_count > 0
        if results['notifications']:
            logging.info(f"✅ Exported {notifications_count} notifications successfully")
        else:
            logging.info("⚠️  No notifications exported")
    else:
        logging.info("\n⚠️  Skipping notification export due to failed tasks")
        results['notifications'] = False
    
    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("SCHEDULED EVENTS COMPLETE")
    logging.info(f"Email Summarizer: {'✅ SUCCESS' if results['email'] else '❌ FAILED'}")
    logging.info(f"News Summary: {'✅ SUCCESS' if results['news'] else '❌ FAILED'}")
    logging.info(f"Notifications Export: {'✅ SUCCESS' if results.get('notifications', False) else '❌ FAILED/SKIPPED'}")
    logging.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)
    
    # Exit with appropriate code
    if all(results.values()):
        logging.info("All tasks completed successfully!")
        return 0
    else:
        logging.warning("Some tasks failed. Check logs for details.")
        return 1
    
if __name__ == "__main__":
    sys.exit(main())