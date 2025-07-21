# Email Summarizer with OpenAI

This project processes email data from `email_processing.json` and uses OpenAI's GPT-4o-mini model to create intelligent summaries, categorizations, and extract key points from your emails.

## 🚀 Quick Start

### Option 1: Use Quickstart (Recommended)
```bash
python quickstart.py
```

### Option 2: Manual Setup
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up OpenAI API Key**
   Create a `.env` file in the project root:
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

3. **Run the Email Processor**
   ```bash
   python openai_email_processor.py
   ```

## 📁 Project Structure

```
Email Summarizer/
├── quickstart.py               # Quick setup and run script
├── openai_email_processor.py   # Main processing script
├── email_processing.json       # Input email data
├── requirements.txt            # Python dependencies
├── .env                       # Environment variables (create this)
└── email_ephem_data/          # Output folder (auto-created)
    ├── email_summaries.json   # Detailed email analysis
    ├── daily_summary.txt      # Human-readable summary
    └── daily_key_points.json  # Structured key points
```

## 🔧 Setup Instructions

### Prerequisites
- Python 3.7 or higher
- OpenAI API account and key

### Step 1: Get OpenAI API Key
1. Sign up at [OpenAI Platform](https://platform.openai.com/)
2. Navigate to API Keys section
3. Create a new API key
4. Copy the key (starts with `sk-`)

### Step 2: Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### Step 3: Configure Environment
Create a `.env` file in the project directory:
```bash
# .env file
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### Step 3.5: Configure Google API Credentials (Optional)
If you plan to fetch emails from Gmail, configure the credential file locations in `quickstart.py`:
```python
GOOGLE_TOKEN_FILE = "token.json"                    # Google OAuth token file
GOOGLE_CREDENTIALS_FILE = "email_credentials.json"  # Google service account credentials
```

**Credentials Location:**
- **OpenAI API Key**: `.env` file in project root
- **Email Data**: `email_processing.json` in project root  
- **Output Files**: `email_ephem_data/` folder (auto-created)
- **Google API Credentials**: Configurable in `quickstart.py`
  - `token.json` - Google OAuth token file
  - `email_credentials.json` - Google service account credentials

### Step 4: Prepare Email Data
Ensure your `email_processing.json` file contains email data in this format:
```json
[
  {
    "id": "email_id",
    "subject": "Email Subject",
    "sender": "sender@example.com",
    "date": "Date string",
    "snippet": "Email preview",
    "body": "Full email content",
    "processed_at": "timestamp"
  }
]
```

## 🎯 Running the Application

### Basic Usage
```bash
python openai_email_processor.py
```

### What Happens When You Run It
1. **Loads** email data from `email_processing.json`
2. **Processes** each email through OpenAI's GPT-4o-mini model
3. **Generates** structured summaries and categorizations
4. **Creates** three output files:
   - `email_summaries.json` - Detailed analysis
   - `daily_summary.txt` - Human-readable summary
   - `daily_key_points.json` - Structured key points

### Expected Output
```
Loaded 6 emails from email_processing.json
Starting to process 6 emails...
Processing email 1/6: Summer Career Tips: Update Your Resume & Cover Letter...
Processing email 2/6: Competition Launch: MAP - Charting Student Math Misunderstandings...
...
Saved 6 email summaries to email_summaries.json
Saved designated key points to daily_key_points.json

==================================================
DAILY EMAIL SUMMARY
==================================================
[Summary content appears here]
```

## 📊 Output Files Explained

### 1. `email_summaries.json`
Detailed analysis of each email including:
- AI-generated summary
- Key points and action items
- Email category and priority
- Dates and deadlines
- Original email metadata

### 2. `daily_summary.txt`
Human-readable overview containing:
- Email count and statistics
- Priority breakdown
- Designated key points section
- Individual email summaries with bullet points

### 3. `daily_key_points.json`
Structured data for programmatic use:
- Metadata (timestamp, email count)
- Categorized key points
- Complete email data for each relevant email
- Both full and truncated versions of points

## 🎯 Key Features

### Smart Email Analysis
- **Automatic Categorization**: Job opportunities, newsletters, personal, etc.
- **Priority Assessment**: High/Medium/Low based on content
- **Action Item Extraction**: Identifies emails requiring attention
- **HTML Cleaning**: Removes formatting from email bodies

### Designated Key Points
- **Handshake Job Opportunities**: Automatically extracts job details
- **Expandable Categories**: Easy to add new types of content to track
- **Structured Output**: Both text and JSON formats

### Cost Optimization
- Uses GPT-4o-mini (cost-effective model)
- Rate limiting to avoid API limits
- Token usage optimization

### Toggle Controls
- **GENERATE_EMAIL_SUMMARIES**: Enable/disable detailed email analysis
- **GENERATE_DAILY_SUMMARY**: Enable/disable human-readable summary
- **GENERATE_KEY_POINTS**: Enable/disable structured key points extraction

## 🔧 Customization

### Adding New Categories
To track new types of content, edit `openai_email_processor.py`:

1. **Uncomment existing categories** in `relevant_categories`:
```python
"deadlines": {
    "description": "⏰ Important Deadlines",
    "criteria": lambda email: any(word in email.get('original_subject', '').lower() for word in ['deadline', 'due', 'expire']),
    "extract_points": lambda email: self.extract_deadline_points(email)
}
```

2. **Add new extraction methods**:
```python
def extract_your_category_points(self, email: Dict[str, Any]) -> List[str]:
    # Your custom logic here
    return points
```

### Changing the Model
Update the model in the `__init__` method:
```python
self.model = "gpt-4o"  # or "gpt-3.5-turbo" for different models
```

### Modifying Prompts
Edit the `create_email_summary_prompt` method to change how emails are analyzed.

## 🛠️ Troubleshooting

### Common Issues

**"OPENAI_API_KEY environment variable is required"**
- Ensure your `.env` file exists and contains the API key
- Check that the key starts with `sk-`

**"email_processing.json not found"**
- Verify the file exists in the project directory
- Check the file name spelling

**API Rate Limits**
- The script includes automatic delays between requests
- If you still hit limits, increase the delay in `process_all_emails`

**JSON Parsing Errors**
- Ensure your `email_processing.json` is valid JSON
- Check for missing commas or brackets

### Error Handling
The system includes robust error handling for:
- Invalid JSON files
- Network connectivity issues
- API authentication problems
- Malformed email data

## 💰 Cost Considerations

- **GPT-4o-mini**: ~$0.00015 per 1K input tokens, ~$0.0006 per 1K output tokens
- **Typical email**: ~500-1000 tokens per email
- **6 emails**: ~$0.01-0.02 per run

## 📈 Example Output

### Daily Summary Excerpt
```
Daily Email Summary:
- Total emails processed: 6
- Emails requiring action: 2
- Priority breakdown: High=1, Medium=4, Low=1
- Categories: career_development(2), job_opportunity(2), newsletter(2)

🎯 DESIGNATED KEY POINTS:
==================================================

🎯 Handshake Job Opportunities:
----------------------------------------

📧 Hot job at SignalFire—worth a look 🔥
   From: Handshake <handshake@notifications.joinhandshake.com>
   • Data & Research Analyst Intern position at SignalFire
   • $30/hr salary, remote work, San Francisco location
   • Application deadline: Tuesday, September 02 at 9 AM EDT
```

## 🤝 Contributing

To add new features or improve the system:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your OpenAI API key is valid
3. Ensure all dependencies are installed
4. Check that your email data format is correct 