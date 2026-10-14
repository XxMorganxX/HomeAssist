#!/usr/bin/env python3
"""
Test script for email relevance filtering.
Run this to test the relevance filtering without processing all emails.
"""
import json
import os
import sys
import random
from typing import List, Dict, Any
from mail_summary import GeminiEmailProcessor

def test_relevance_filtering(sample_size: int = 20, seed: int = None):
    """Test relevance filtering on a random sample of emails.
    
    Args:
        sample_size: Number of emails to test
        seed: Random seed for reproducible sampling (optional)
    """
    from datetime import datetime
    
    print("="*60)
    print("EMAIL RELEVANCE FILTERING TEST")
    print("="*60)
    print(f"Current Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("This date is used to evaluate deadlines and time-sensitive content")
    
    # Initialize processor
    processor = GeminiEmailProcessor()
    
    # Load all emails
    emails = processor.load_email_data()
    if not emails:
        print("No emails found to test.")
        return
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    # Take a random sample
    actual_sample_size = min(sample_size, len(emails))
    if actual_sample_size < len(emails):
        sample = random.sample(emails, actual_sample_size)
        print(f"\nTesting with {len(sample)} randomly selected emails from {len(emails)} total")
    else:
        sample = emails
        print(f"\nTesting with all {len(sample)} emails (sample size >= total emails)")
    
    print("Processing individually...")
    print("-"*60)
    
    # Test relevance filtering
    relevant_emails, decisions = processor.filter_relevant_emails(sample)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total tested: {len(sample)}")
    print(f"Marked relevant: {len(relevant_emails)}")
    print(f"Relevance rate: {len(relevant_emails)/len(sample)*100:.1f}%")
    
    # Analyze sender distribution
    sender_counts = {}
    for email in sample:
        sender = email.get('sender', 'Unknown')
        # Extract domain from sender email
        if '@' in sender:
            domain = sender.split('@')[-1].split('>')[0]
        else:
            domain = sender[:30]  # First 30 chars if not an email
        sender_counts[domain] = sender_counts.get(domain, 0) + 1
    
    # Show top senders in sample
    print("\n" + "="*60)
    print("SAMPLE COMPOSITION (Top 5 Senders)")
    print("="*60)
    sorted_senders = sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for sender, count in sorted_senders:
        print(f"  {sender}: {count} emails ({count/len(sample)*100:.1f}%)")
    
    # Show decision breakdown with sender analysis
    print("\n" + "="*60)
    print("DECISION DETAILS")
    print("="*60)
    
    # Analyze which senders are marked relevant
    relevant_senders = {}
    for dec in decisions:
        if dec.get('relevant'):
            # Find the email to get sender
            for email in sample:
                if email.get('id') == dec.get('id'):
                    sender = email.get('sender', 'Unknown')
                    if '@' in sender:
                        domain = sender.split('@')[-1].split('>')[0]
                    else:
                        domain = sender[:30]
                    relevant_senders[domain] = relevant_senders.get(domain, 0) + 1
                    break
    
    # Group by relevance
    relevant_list = []
    not_relevant_list = []
    
    for decision in decisions:
        if decision.get('relevant'):
            relevant_list.append(decision)
        else:
            not_relevant_list.append(decision)
    
    # Show relevant emails
    if relevant_list:
        print(f"\n✅ RELEVANT EMAILS ({len(relevant_list)}):")
        print("-"*40)
        for i, dec in enumerate(relevant_list[:10], 1):  # Show first 10
            subject = dec.get('subject', 'No subject')[:60]
            reason = dec.get('reason', 'No reason')[:100]
            print(f"{i}. {subject}")
            print(f"   Reason: {reason}")
            # Find and show email date if available
            for email in sample:
                if email.get('id') == dec.get('id'):
                    email_date = email.get('date', 'Unknown date')
                    print(f"   Email date: {email_date}")
                    break
    
    # Show sample of non-relevant emails
    if not_relevant_list:
        print(f"\n❌ NOT RELEVANT EMAILS (showing first 10 of {len(not_relevant_list)}):")
        print("-"*40)
        for i, dec in enumerate(not_relevant_list[:10], 1):
            subject = dec.get('subject', 'No subject')[:60]
            reason = dec.get('reason', 'No reason')[:80]
            print(f"{i}. {subject}")
            print(f"   Reason: {reason}")
    
    # Save detailed results
    output_file = os.path.join(processor.data_folder, "test_relevance_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "test_params": {
                "sample_size": sample_size,
                "actual_sample_size": len(sample),
                "seed": seed,
                "processing_mode": "individual",
                "sampling": "random"
            },
            "summary": {
                "total": len(sample),
                "relevant": len(relevant_emails),
                "relevance_rate": f"{len(relevant_emails)/len(sample)*100:.1f}%" if len(sample) > 0 else "0%"
            },
            "decisions": decisions
        }, f, indent=2)
    
    print(f"\n📝 Detailed results saved to: {output_file}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    relevance_rate = len(relevant_emails)/len(sample)*100 if len(sample) > 0 else 0
    
    if relevance_rate < 5:
        print("✅ Very selective filtering (< 5% relevant)")
        print("This is good for reducing processing costs.")
        print("Make sure you're not missing important emails.")
    elif relevance_rate < 15:
        print("✅ Good selective filtering (5-15% relevant)")
        print("This seems like an appropriate level of filtering.")
    elif relevance_rate < 30:
        print("📊 Moderate filtering (15-30% relevant)")
        print("Consider if criteria could be stricter.")
    elif relevance_rate < 50:
        print("⚠️  Low filtering (30-50% relevant)")
        print("Criteria may be too inclusive.")
        print("Review email_criteria.txt to be more selective.")
    else:
        print("❌ Very low filtering (> 50% relevant)")
        print("Criteria are too inclusive!")
        print("Most emails are being marked as relevant.")
        print("Update email_criteria.txt to be much more selective.")

if __name__ == "__main__":
    # Parse command line arguments
    sample_size = 20
    seed = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print(f"Usage: {sys.argv[0]} [sample_size] [seed]")
            print(f"")
            print(f"Arguments:")
            print(f"  sample_size  Number of emails to randomly sample (default: 20)")
            print(f"  seed         Random seed for reproducible sampling (optional)")
            print(f"")
            print(f"Examples:")
            print(f"  {sys.argv[0]} 30          # Test 30 random emails")
            print(f"  {sys.argv[0]} 30 42       # Test 30 random emails with seed 42")
            print(f"  {sys.argv[0]} 50 12345    # Test 50 random emails with seed 12345")
            sys.exit(0)
        try:
            sample_size = int(sys.argv[1])
        except ValueError:
            print(f"Invalid sample size: {sys.argv[1]}")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            seed = int(sys.argv[2])
        except ValueError:
            print(f"Invalid seed: {sys.argv[2]}")
            sys.exit(1)
    
    print(f"Usage: {sys.argv[0]} [sample_size] [seed]")
    print(f"Example: {sys.argv[0]} 30        # Random sample")
    print(f"Example: {sys.argv[0]} 30 42     # Reproducible sample with seed 42")
    print(f"Note: Emails are randomly sampled from the full dataset")
    print()
    
    test_relevance_filtering(sample_size, seed)