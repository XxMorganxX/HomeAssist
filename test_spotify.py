#!/usr/bin/env python3
"""
Quick test to verify Spotify authentication is working.
"""
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print("Testing Spotify Tool...")
print("=" * 70)

# Import and test the tool
from mcp_server.improved_tools.improved_spotify import ImprovedSpotifyPlaybackTool

tool = ImprovedSpotifyPlaybackTool()

# Try to get current status
result = tool.execute({
    "action": "status",
    "user": "Morgan"
})

import json
print("Tool Response:")
print(json.dumps(result, indent=2))

if result.get("success"):
    print("\n✅ Spotify tool is working!")
else:
    print(f"\n❌ Error: {result.get('error')}")