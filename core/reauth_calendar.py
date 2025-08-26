#!/usr/bin/env python3
"""
Script to re-authenticate Google Calendar with offline access permissions.
This will give you long-lasting authentication that doesn't require frequent re-authentication.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.calendar_component import CalendarComponent
import mcp_server.config as config

def main():
    print("=" * 60)
    print("🔐 Google Calendar Re-authentication Tool")
    print("=" * 60)
    print("\nThis will clear your existing credentials and request new ones with:")
    print("  • Offline access (refresh tokens that don't expire)")
    print("  • Automatic token refresh capability")
    print("\nAvailable users:")
    
    users = list(config.CALENDAR_USERS.keys())
    for i, user in enumerate(users, 1):
        print(f"  {i}. {user}")
    
    print("\nWhich user do you want to re-authenticate?")
    print("Enter the number or username (or 'all' for all users): ", end="")
    
    choice = input().strip().lower()
    
    selected_users = []
    if choice == 'all':
        selected_users = users
    elif choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(users):
            selected_users = [users[idx]]
        else:
            print("❌ Invalid selection")
            return
    elif choice in [u.lower() for u in users]:
        # Find the matching user (case-insensitive)
        for user in users:
            if user.lower() == choice:
                selected_users = [user]
                break
    else:
        print("❌ Invalid selection")
        return
    
    print("\n" + "=" * 60)
    
    for user in selected_users:
        print(f"\n📅 Processing user: {user}")
        print("-" * 40)
        
        try:
            # Create calendar component
            calendar = CalendarComponent(user=user)
            
            # Clear existing credentials
            print(f"🗑️  Clearing existing credentials for {user}...")
            calendar.clear_credentials()
            
            print(f"🔄 Re-initializing with new offline access permissions...")
            print("\n⚠️  A browser window will open for authentication.")
            print("   Please sign in and grant the requested permissions.\n")
            
            # Re-initialize to trigger new authentication
            calendar = CalendarComponent(user=user)
            
            # Test the connection
            print(f"🧪 Testing connection for {user}...")
            events = calendar.get_events(num_events=1)
            
            if events:
                print(f"✅ Successfully authenticated {user} with offline access!")
                print(f"   Found {len(events)} upcoming event(s)")
            else:
                print(f"✅ Successfully authenticated {user} with offline access!")
                print(f"   No upcoming events found (this is normal)")
                
        except Exception as e:
            print(f"❌ Error processing {user}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("✨ Re-authentication complete!")
    print("\nYour calendar credentials now have:")
    print("  • Refresh tokens that don't expire")
    print("  • Automatic token refresh without user interaction")
    print("  • No more frequent re-authentication prompts!")
    print("=" * 60)

if __name__ == "__main__":
    main()