import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

# =============================================================================
# TOGGLE GENERATION BOOLEAN
# =============================================================================
GENERATE_EMAIL_SUMMARIES = True
GENERATE_DAILY_SUMMARY = True
GENERATE_KEY_POINTS = True

# Deterministic relevance mode
RELEVANCE_DETERMINISTIC = True

# Include reason in relevance determination (increases token usage)
INCLUDE_RELEVANCE_REASON = False

# Load environment variables
load_dotenv()

class GeminiEmailProcessor:
    def __init__(self):
        """Initialize the Gemini client with API key from environment variables."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for email summarizer")
        
        genai.configure(api_key=self.api_key)
        # Default to fast model; change to 'gemini-1.5-pro' if you prefer higher quality
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        # Relaxed safety settings for processing email content - using string format for compatibility
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Summarization model (slightly creative)
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1200,
                "response_mime_type": "application/json",
            },
            safety_settings=self.safety_settings,
        )
        # Relevance model (deterministic, low variance)
        if RELEVANCE_DETERMINISTIC:
            self.relevance_model = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": 0.0,
                    "top_p": 0.0,
                    "top_k": 1,
                    "candidate_count": 1,
                    "response_mime_type": "application/json",
                },
                safety_settings=self.safety_settings,
            )
        else:
            self.relevance_model = self.model
        
        # Create ephemeral_data folder if it doesn't exist (prefer module path if exists)
        module_ephemeral = "scripts/scheduled/email_summarizer/ephemeral_data"
        root_ephemeral = "ephemeral_data"
        self.data_folder = module_ephemeral if os.path.isdir(module_ephemeral) else root_ephemeral
        os.makedirs(self.data_folder, exist_ok=True)
        
        # Relevance criteria file locations (prefer module path)
        self.criteria_paths = [
            "scripts/scheduled/email_summarizer/email_criteria.txt",
            "email_criteria.txt",
        ]

    def _extract_text_summary(self, raw_summary: Any) -> str:
        """Ensure we return a concise text summary, not a JSON blob.

        - If raw_summary looks like JSON, try to parse and extract the 'summary' field.
        - If parsing fails, attempt a simple regex extraction; otherwise return the raw string.
        """
        try:
            if isinstance(raw_summary, dict):
                return str(raw_summary.get('summary', ''))
            if isinstance(raw_summary, str):
                candidate = raw_summary.strip()
                if candidate.startswith('{') or candidate.startswith('['):
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and 'summary' in obj:
                            return str(obj.get('summary', ''))
                        return json.dumps(obj)[:500]
                    except Exception:
                        # Fallback regex to find a "summary": "..." field
                        import re
                        m = re.search(r'"summary"\s*:\s*"([^"]+)"', candidate)
                        if m:
                            return m.group(1)
                return candidate
            return str(raw_summary)
        except Exception:
            return str(raw_summary)

    def generate_concise_digest(self, summaries: List[Dict[str, Any]], max_subjects: int = 5) -> str:
        """Create a concise digest text for notifications from processed summaries."""
        valid = [s for s in summaries if 'error' not in s]
        if not valid:
            return "No relevant emails."

        priorities = {"high": 0, "medium": 0, "low": 0}
        for s in valid:
            p = (s.get('priority') or 'medium').lower()
            if p in priorities:
                priorities[p] += 1

        subjects = []
        for s in valid[:max_subjects]:
            subj = s.get('original_subject') or 'No subject'
            subjects.append(subj.strip())

        total = len(valid)
        key_line = "; ".join(subjects)
        digest = (
            f"Email digest: {total} relevant (High {priorities['high']}, "
            f"Med {priorities['medium']}, Low {priorities['low']}). "
        )
        if key_line:
            digest += f"Key: {key_line}"
        return digest[:600]

    def load_relevance_criteria(self, override_path: Optional[str] = None) -> str:
        """Load relevance criteria text from file; return default template if missing."""
        candidate_paths = [override_path] if override_path else []
        for p in self.criteria_paths:
            if p and p not in candidate_paths:
                candidate_paths.append(p)

        for path in candidate_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    print(f"Loaded relevance criteria from {path}")
                    return text
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Warning: Could not read criteria from {path}: {e}")


    def create_relevance_prompt(self, email: Dict[str, Any], criteria_text: str) -> str:
        """Create a prompt that asks Gemini to classify email relevance based on criteria text."""
        cleaned_body = self.clean_email_body(email.get('body', ''))
        # Increase body limit for better context
        body_limit = 1500
        
        # Get current datetime for time-based relevance
        current_time = datetime.now()
        current_date_str = current_time.strftime("%Y-%m-%d %H:%M")
        current_day = current_time.strftime("%A")
        
        # Build JSON schema based on whether reason is included
        if INCLUDE_RELEVANCE_REASON:
            json_schema = """{
  "relevant": true or false,
  "reason": "brief explanation"
}"""
        else:
            json_schema = """{
  "relevant": true or false
}"""
        
        prompt = f"""You are a strict and deterministic relevance classifier. Your job is to filter OUT emails, keeping only the truly important ones according to the Relevance Criteria.

CURRENT DATE/TIME: {current_date_str} ({current_day})
Use this to evaluate deadlines and time-sensitive content.

RELEVANCE CRITERIA:
{criteria_text}

EMAIL DETAILS:
Subject: {email.get('subject', 'No subject')}
From: {email.get('sender', 'Unknown sender')}
Date: {email.get('date', 'Unknown date')}
Snippet: {email.get('snippet', 'No snippet')}

Body (first {body_limit} chars):
{cleaned_body[:body_limit]}

DETERMINISTIC RULES:
- Be VERY SELECTIVE.
- If uncertain or ambiguous, ALWAYS return relevant=false.

Respond with EXACTLY this JSON object (NO extra text, no markdown):
{json_schema}"""
        return prompt

    def assess_relevance(self, email: Dict[str, Any], criteria_text: str) -> Dict[str, Any]:
        """Call Gemini to assess relevance and return a dict with relevant, reason."""
        prompt = self.create_relevance_prompt(email, criteria_text)
        print(f"    Assessing: {email.get('subject', 'No subject')[:50]}")
        
        # Use deterministic model for relevance if enabled
        model_for_relevance = getattr(self, "relevance_model", self.model)
        response = model_for_relevance.generate_content(prompt)

        # Check for safety blocks before accessing text
        if response.prompt_feedback:
            # Check if blocked
            if response.prompt_feedback.block_reason:
                print(f"    ⚠️ BLOCKED! Reason: {response.prompt_feedback.block_reason}")
                print(f"    Safety Ratings: {response.prompt_feedback.safety_ratings}")
        
        # If candidate finish reason is not STOP (1), it might be blocked or other
        if response.candidates and response.candidates[0].finish_reason != 1:
             print(f"    ⚠️ Finish Reason: {response.candidates[0].finish_reason} (not STOP)")
             if response.candidates[0].safety_ratings:
                 print(f"    Candidate Ratings: {response.candidates[0].safety_ratings}")

        # Extract response content - let errors propagate
        content = response.text
        print(f"    Response: {content[:100]}...")

        decision = json.loads(content)
        return decision

    # DEPRECATED - Keeping for reference but not used
    def create_relevance_batch_prompt(self, emails: List[Dict[str, Any]], criteria_text: str) -> str:
        """[DEPRECATED] Create a prompt asking Gemini to classify relevance for a batch of emails.
        
        Note: This method is no longer used. We process emails individually for better reliability.
        """
        # Get current datetime for time-based relevance
        current_time = datetime.now()
        current_date_str = current_time.strftime("%Y-%m-%d %H:%M")
        
        # Build compact representations to control token usage
        body_limit = 800
        lines = []
        for i, e in enumerate(emails):
            subject = e.get('subject', 'No subject')
            sender = e.get('sender', 'Unknown')
            snippet = e.get('snippet', '')
            body = self.clean_email_body(e.get('body', ''))[:body_limit]
            eid = e.get('id', '')
            lines.append(
                f"Email {i+1}:\nID: {eid}\nSubject: {subject}\nFrom: {sender}\nSnippet: {snippet}\nBody: {body}"
            )

        emails_block = "\n\n".join(lines)
        prompt = f"""You are a relevance classifier. Current date/time: {current_date_str}

CRITERIA:
{criteria_text}

{emails_block}

Return a JSON array with one object per email in the same order:
[{{"id": "<email_id>", "relevant": true/false, "reason": "brief explanation"}}]"""
        return prompt

    # DEPRECATED - Keeping for reference but not used
    def assess_relevance_batch(self, emails: List[Dict[str, Any]], criteria_text: str) -> List[Dict[str, Any]]:
        """[DEPRECATED] Classify a batch of emails for relevance and return list of decisions in order.
        
        Note: This method is no longer used. We process emails individually for better reliability.
        """
        # Just use individual assessment for each email
        decisions = []
        for e in emails:
            dec = self.assess_relevance(e, criteria_text)
            dec["id"] = e.get('id')
            decisions.append(dec)
        return decisions

    def filter_relevant_emails(self, emails: List[Dict[str, Any]], criteria_override_path: Optional[str] = None, batch_size: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Assess relevance for emails and return (relevant_emails, decisions).

        Processes emails one by one for better reliability.
        """
        criteria_text = self.load_relevance_criteria(criteria_override_path)
        relevant_emails: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []

        total = len(emails)
        current_time = datetime.now()
        print(f"Processing {total} emails individually for relevance...")
        print(f"Current date/time for relevance: {current_time.strftime('%Y-%m-%d %H:%M')}")
        
        for i, email in enumerate(emails, 1):
            # Show progress every 10 emails or for small sets
            if i % 10 == 0 or total <= 20:
                print(f"  Processing email {i}/{total}: {email.get('subject', 'No subject')[:50]}...")
            
            # Assess relevance for this single email
            decision = self.assess_relevance(email, criteria_text)
            
            # Store the decision
            email['is_relevant'] = bool(decision.get('relevant', False))
            email['relevance_reason'] = decision.get('reason', '')
            
            decisions.append({
                "id": email.get('id'),
                "subject": email.get('subject'),
                "relevant": email['is_relevant'],
                "reason": email['relevance_reason'],
            })
            
            if email['is_relevant']:
                relevant_emails.append(email)
            
            # Small delay to avoid rate limiting (can be adjusted or removed if not needed)
            if i < total:  # Don't delay after the last email
                import time
                time.sleep(0.1)  # 100ms delay between emails

        # Print summary with more detail
        print("\nRelevance filtering complete:")
        print(f"  Total emails: {len(emails)}")
        print(f"  Relevant emails: {len(relevant_emails)}")
        if len(emails) > 0:
            print(f"  Relevance rate: {len(relevant_emails)/len(emails)*100:.1f}%")
        return relevant_emails, decisions

    def save_relevance_results(self, decisions: List[Dict[str, Any]], output_file: str = "relevance_results.json") -> None:
        """Save relevance decisions to JSON under the data folder."""
        try:
            file_path = os.path.join(self.data_folder, output_file)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "generated_at": datetime.now().isoformat(),
                    "total": len(decisions),
                    "decisions": decisions
                }, f, indent=2, ensure_ascii=False)
            print(f"Saved relevance results to {file_path}")
        except Exception as e:
            print(f"Error saving relevance results: {e}")

    def load_email_data(self, file_path: str = None) -> List[Dict[str, Any]]:
        """Load email data from the JSON file, trying both module and root paths."""
        candidate_paths: List[str] = []
        if file_path:
            candidate_paths.append(file_path)
        # Prefer data_folder/new_mail.json first
        candidate_paths.append(os.path.join(self.data_folder, 'new_mail.json'))
        # Also try the opposite location for robustness
        alt_module_path = os.path.join('scripts/scheduled/email_summarizer/ephemeral_data', 'new_mail.json')
        alt_root_path = os.path.join('ephemeral_data', 'new_mail.json')
        for p in [alt_module_path, alt_root_path]:
            if p not in candidate_paths:
                candidate_paths.append(p)

        for path in candidate_paths:
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    emails = json.load(file)
                    print(f"Loaded {len(emails)} emails from {path}")
                return emails
            except FileNotFoundError:
                continue
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from {path}: {e}")
                print("Error: new_mail.json not found in expected locations")
                return []
    
    def clean_email_body(self, body: str) -> str:
        """Clean HTML and formatting from email body."""
        # Remove HTML tags and entities
        import re
        # Remove HTML tags
        body = re.sub(r'<[^>]+>', '', body)
        # Decode HTML entities
        import html
        body = html.unescape(body)
        # Remove extra whitespace
        body = re.sub(r'\s+', ' ', body).strip()
        return body
    
    def create_email_summary_prompt(self, email: Dict[str, Any]) -> str:
        """Create a prompt for summarizing an individual email."""
        cleaned_body = self.clean_email_body(email.get('body', ''))
        
        # Check if this is a Handshake email
        is_handshake = "handshake" in email.get('sender', '').lower() or "handshake" in email.get('subject', '').lower()
        
        prompt = f"""
            Please analyze and summarize the following email:

            Subject: {email.get('subject', 'No subject')}
            From: {email.get('sender', 'Unknown sender')}
            Date: {email.get('date', 'Unknown date')}
            Snippet: {email.get('snippet', 'No snippet')}

            Email Body:
            {cleaned_body[:2000]}  # Limit body length to avoid token limits

            Please provide:
            1. A concise summary (2-3 sentences)
            2. Key points or action items
            3. Email category (e.g., job opportunity, newsletter, personal, etc.)
            4. Priority level (high/medium/low)
            5. Any important dates or deadlines mentioned

            {"IMPORTANT: If this is a Handshake email with job opportunities, make sure to include specific job details in the key_points such as:" if is_handshake else ""}
            {"- Job title and company name" if is_handshake else ""}
            {"- Salary range if mentioned" if is_handshake else ""}
            {"- Location (remote/hybrid/on-site)" if is_handshake else ""}
            {"- Application deadline" if is_handshake else ""}
            {"- Key requirements or qualifications" if is_handshake else ""}

            Format your response as JSON with the following structure:
            {{
                "summary": "brief summary",
                "key_points": ["point1", "point2", "point3"],
                "category": "email category",
                "priority": "high/medium/low",
                "dates_deadlines": ["date1", "date2"],
                "action_required": true/false
            }}
        """
        return prompt
    
    def process_single_email(self, email: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single email through Gemini."""
        try:
            prompt = self.create_email_summary_prompt(email)
            # Gemini generate_content; with JSON response requested via generation_config
            response = self.model.generate_content(prompt)
            # Extract the response content safely (avoid response.text when no parts)
            content = ""
            try:
                candidates = getattr(response, "candidates", []) or []
                for cand in candidates:
                    cand_content = getattr(cand, "content", None)
                    parts = getattr(cand_content, "parts", []) if cand_content else []
                    for part in parts:
                        piece = getattr(part, "text", None)
                        if piece:
                            content += piece
                if not content and candidates:
                    # Log finish_reason for diagnostics when empty
                    try:
                        fr = getattr(candidates[0], "finish_reason", "unknown")
                        print(f"Note: Empty Gemini response (finish_reason={fr}).")
                    except Exception:
                        pass
            except Exception:
                # As a fallback, try the quick accessor; may raise if no valid Part
                try:
                    content = response.text or ""
                except Exception:
                    content = ""
            
            # Try to parse JSON response
            try:
                summary_data = json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response
                summary_data = {
                    "summary": content,
                    "key_points": [],
                    "category": "unknown",
                    "priority": "medium",
                    "dates_deadlines": [],
                    "action_required": False,
                    "raw_response": content
                }
            
            # Add original email metadata
            summary_data.update({
                "original_email_id": email.get('id'),
                "original_subject": email.get('subject'),
                "original_sender": email.get('sender'),
                "original_date": email.get('date'),
                "processed_at": datetime.now().isoformat()
            })
            
            return summary_data
            
        except Exception as e:
            print(f"Error processing email {email.get('id', 'unknown')}: {e}")
            return {
                "error": str(e),
                "original_email_id": email.get('id'),
                "original_subject": email.get('subject'),
                "processed_at": datetime.now().isoformat()
            }
    
    def process_all_emails(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all emails and return summaries."""
        summaries = []
        
        for i, email in enumerate(emails, 1):
            print(f"Processing email {i}/{len(emails)}: {email.get('subject', 'No subject')[:50]}...")
            summary = self.process_single_email(email)
            summaries.append(summary)
            
            # Add a small delay to avoid rate limiting
            import time as _time
            _time.sleep(0.5)
        
        return summaries
    
    def save_summaries(self, summaries: List[Dict[str, Any]], output_file: str = "email_summaries.json"):
        """Save email summaries to a JSON file."""
        if not GENERATE_EMAIL_SUMMARIES:
            print("Email summaries generation is disabled")
            return
            
        try:
            file_path = os.path.join(self.data_folder, output_file)
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(summaries, file, indent=2, ensure_ascii=False)
            print(f"Saved {len(summaries)} email summaries to {file_path}")
        except Exception as e:
            print(f"Error saving summaries: {e}")
    
    def create_daily_summary(self, summaries: List[Dict[str, Any]]) -> str:
        """Create a daily summary of all processed emails."""
        try:
            # Filter out errors
            valid_summaries = [s for s in summaries if 'error' not in s]
            
            if not valid_summaries:
                return "No valid emails to summarize."
            
            # Count by category
            categories = {}
            priorities = {"high": 0, "medium": 0, "low": 0}
            action_required = 0
            
            for summary in valid_summaries:
                category = summary.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
                
                priority = summary.get('priority', 'medium')
                priorities[priority] += 1
                
                if summary.get('action_required', False):
                    action_required += 1
            
            # Create summary prompt
            summary_text = f"""
Daily Email Summary:
- Total emails processed: {len(valid_summaries)}
- Emails requiring action: {action_required}
- Priority breakdown: High={priorities['high']}, Medium={priorities['medium']}, Low={priorities['low']}
- Categories: {', '.join([f'{cat}({count})' for cat, count in categories.items()])}

Key emails requiring attention:
"""
            
            # Add high priority emails
            high_priority = [s for s in valid_summaries if s.get('priority') == 'high']
            for email in high_priority[:5]:  # Top 5 high priority
                summary_text += f"- {email.get('original_subject', 'No subject')} (from {email.get('original_sender', 'Unknown')})\n"
            
            # Add designated key points section
            summary_text += self.create_designated_key_points_section(valid_summaries)
            
            # Add bullet point summaries for each email
            summary_text += "\n\nEmail Summaries:\n"
            summary_text += "=" * 50 + "\n"
            
            # First, add a special section for Handshake job opportunities
            handshake_jobs = [s for s in valid_summaries if "handshake" in s.get('original_sender', '').lower()]
            if handshake_jobs:
                summary_text += "\n🎯 HANDSHAKE JOB OPPORTUNITIES:\n"
                summary_text += "-" * 40 + "\n"
                for job in handshake_jobs:
                    subject = job.get('original_subject', 'No subject')
                    summary = self._extract_text_summary(job.get('summary', 'No summary available'))
                    key_points = job.get('key_points', [])
                    priority = job.get('priority', 'medium').upper()
                    
                    summary_text += f"\n📧 {subject}\n"
                    summary_text += f"   Priority: {priority}\n"
                    summary_text += f"   Summary: {summary}\n"
                    
                    if key_points:
                        summary_text += "   Job Details:\n"
                        for point in key_points[:3]:
                            words = point.split()[:20]
                            truncated_point = ' '.join(words)
                            if len(point.split()) > 20:
                                truncated_point += "..."
                            summary_text += f"   • {truncated_point}\n"
                    summary_text += "\n"
            
            # Then add all emails in order
            for i, email in enumerate(valid_summaries, 1):
                subject = email.get('original_subject', 'No subject')
                sender = email.get('original_sender', 'Unknown')
                summary = self._extract_text_summary(email.get('summary', 'No summary available'))
                key_points = email.get('key_points', [])
                priority = email.get('priority', 'medium').upper()
                
                # Add special indicator for Handshake emails
                handshake_indicator = "🎯 " if "handshake" in sender.lower() else ""
                
                summary_text += f"\n{i}. {handshake_indicator}{subject}\n"
                summary_text += f"   From: {sender} | Priority: {priority}\n"
                summary_text += f"   Summary: {summary}\n"
                
                # Add 2-3 bullet points (20 words each max)
                if key_points:
                    summary_text += "   Key Points:\n"
                    for point in key_points[:3]:  # Limit to 3 points
                        # Truncate to approximately 20 words
                        words = point.split()[:20]
                        truncated_point = ' '.join(words)
                        if len(point.split()) > 20:
                            truncated_point += "..."
                        summary_text += f"   • {truncated_point}\n"
                else:
                    summary_text += "   Key Points: None identified\n"
                
                summary_text += "-" * 40 + "\n"
            
            return summary_text
            
        except Exception as e:
            return f"Error creating daily summary: {e}"
    
    def create_designated_key_points_section(self, summaries: List[Dict[str, Any]]) -> str:
        """Create a designated key points section that collects relevant information."""
        key_points_section = "\n\n🎯 DESIGNATED KEY POINTS:\n"
        key_points_section += "=" * 50 + "\n"
        
        # Define what types of content to collect (easily expandable)
        relevant_categories = {
            "handshake_jobs": {
                "description": "🎯 Handshake Job Opportunities",
                "criteria": lambda email: "handshake" in email.get('original_sender', '').lower(),
                "extract_points": lambda email: self.extract_handshake_job_points(email)
            }
            # Add more categories here as needed:
            # "deadlines": {
            #     "description": "⏰ Important Deadlines",
            #     "criteria": lambda email: any(word in email.get('original_subject', '').lower() for word in ['deadline', 'due', 'expire']),
            #     "extract_points": lambda email: self.extract_deadline_points(email)
            # },
            # "events": {
            #     "description": "📅 Upcoming Events",
            #     "criteria": lambda email: any(word in email.get('original_subject', '').lower() for word in ['event', 'workshop', 'meeting']),
            #     "extract_points": lambda email: self.extract_event_points(email)
            # }
        }
        
        # Collect all key points data for JSON export
        all_key_points_data = {
            "generated_at": datetime.now().isoformat(),
            "total_emails_processed": len(summaries),
            "categories": {}
        }
        
        # Process each category
        for category_key, category_config in relevant_categories.items():
            relevant_emails = [email for email in summaries if category_config["criteria"](email)]
            
            if relevant_emails:
                key_points_section += f"\n{category_config['description']}:\n"
                key_points_section += "-" * 40 + "\n"
                
                # Initialize category data for JSON
                all_key_points_data["categories"][category_key] = {
                    "description": category_config["description"],
                    "count": len(relevant_emails),
                    "emails": []
                }
                
                for email in relevant_emails:
                    points = category_config["extract_points"](email)
                    if points:
                        subject = email.get('original_subject', 'No subject')
                        sender = email.get('original_sender', 'Unknown')
                        key_points_section += f"\n📧 {subject}\n"
                        key_points_section += f"   From: {sender}\n"
                        
                        # Prepare email data for JSON
                        email_data = {
                            "subject": subject,
                            "sender": sender,
                            "email_id": email.get('original_email_id'),
                            "date": email.get('original_date'),
                            "priority": email.get('priority'),
                            "summary": email.get('summary'),
                            "key_points": points,
                            "truncated_points": []
                        }
                        
                        for point in points:
                            # Truncate to approximately 25 words for designated points
                            words = point.split()[:25]
                            truncated_point = ' '.join(words)
                            if len(point.split()) > 25:
                                truncated_point += "..."
                            key_points_section += f"   • {truncated_point}\n"
                            email_data["truncated_points"].append(truncated_point)
                        
                        key_points_section += "\n"
                        all_key_points_data["categories"][category_key]["emails"].append(email_data)
        
        # Save key points to JSON file
        self.save_key_points_to_json(all_key_points_data)
        
        return key_points_section
    
    def extract_handshake_job_points(self, email: Dict[str, Any]) -> List[str]:
        """Extract job-specific points from Handshake emails."""
        points = []
        key_points = email.get('key_points', [])
        summary = email.get('summary', '')
        
        # Extract job details from key points
        for point in key_points:
            if any(keyword in point.lower() for keyword in ['job', 'position', 'role', 'intern', 'analyst', 'engineer', 'developer']):
                points.append(point)
        
        # If no specific job points found, use summary
        if not points and 'job' in summary.lower():
            points.append(summary)
        
        return points[:3]  # Limit to 3 most relevant points
    
    def extract_deadline_points(self, email: Dict[str, Any]) -> List[str]:
        """Extract deadline-related points from emails."""
        points = []
        key_points = email.get('key_points', [])
        dates = email.get('dates_deadlines', [])
        
        # Extract deadline information
        for point in key_points:
            if any(word in point.lower() for word in ['deadline', 'due', 'expire', 'apply by', 'submit by']):
                points.append(point)
        
        # Add specific dates
        for date in dates:
            points.append(f"Deadline: {date}")
        
        return points[:3]
    
    def extract_event_points(self, email: Dict[str, Any]) -> List[str]:
        """Extract event-related points from emails."""
        points = []
        key_points = email.get('key_points', [])
        
        # Extract event information
        for point in key_points:
            if any(word in point.lower() for word in ['event', 'workshop', 'meeting', 'webinar', 'conference']):
                points.append(point)
        
        return points[:3]
    
    def save_key_points_to_json(self, key_points_data: Dict[str, Any], output_file: str = "daily_key_points.json"):
        """Save designated key points to a JSON file."""
        if not GENERATE_KEY_POINTS:
            print("Key points generation is disabled")
            return
            
        try:
            file_path = os.path.join(self.data_folder, output_file)
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(key_points_data, file, indent=2, ensure_ascii=False)
            print(f"Saved designated key points to {file_path}")
        except Exception as e:
            print(f"Error saving key points to JSON: {e}")

def main_mail_summary():
    """Main function to process emails."""
    try:
        # Initialize processor
        processor = GeminiEmailProcessor()
        
        # Load email data
        emails = processor.load_email_data()
        
        if not emails:
            print("No emails to process.")
            return
        
        # Step 1: Relevance filtering
        relevant_emails, decisions = processor.filter_relevant_emails(emails)
        processor.save_relevance_results(decisions)
        if not relevant_emails:
            print("No relevant emails found.")
            # Overwrite summaries with empty list to prevent stale notifications
            try:
                processor.save_summaries([])
            except Exception:
                pass
            # Remove any stale notifications file from previous runs
            try:
                notifs_path = os.path.join(processor.data_folder, "email_notifications.json")
                if os.path.exists(notifs_path):
                    os.remove(notifs_path)
                    print(f"🧹 Removed stale notifications file: {notifs_path}")
            except Exception:
                pass
            return

        # Step 2: Process only relevant emails
        print(f"Starting to process {len(relevant_emails)} relevant emails...")
        summaries = processor.process_all_emails(relevant_emails)
        
        # Save summaries
        processor.save_summaries(summaries)
        
        # Create and print daily summary
        if GENERATE_DAILY_SUMMARY:
            daily_summary = processor.create_daily_summary(summaries)
            print("\n" + "="*50)
            print("DAILY EMAIL SUMMARY")
            print("="*50)
            print(daily_summary)
            
            # Save daily summary to file
            file_path = os.path.join(processor.data_folder, "daily_summary.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(daily_summary)
            print(f"\nDaily summary saved to {file_path}")
        else:
            print("Daily summary generation is disabled")

        
    except Exception as e:
        print(f"Error in main process: {e}")

if __name__ == "__main__":
    main_mail_summary()