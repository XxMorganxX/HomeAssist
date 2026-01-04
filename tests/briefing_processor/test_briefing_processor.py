#!/usr/bin/env python3
"""
Briefing Processor Test Suite

Tests the BriefingProcessor by:
1. Loading sample briefings from sample_briefings.json
2. Generating openers for each briefing
3. Generating a combined opener for all briefings
4. Writing results to output_openers.json

Run from project root:
    python -m tests.briefing_processor.test_briefing_processor

Or directly:
    cd tests/briefing_processor && python test_briefing_processor.py
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from assistant_framework.utils.briefing_processor import BriefingProcessor


# Paths
SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "sample_briefings.json"
OUTPUT_FILE = SCRIPT_DIR / "output_openers.json"


async def test_single_openers(processor: BriefingProcessor, briefings: list) -> list:
    """Generate individual openers for each briefing."""
    results = []
    
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL OPENERS")
    print("=" * 60)
    
    for i, briefing in enumerate(briefings, 1):
        briefing_id = briefing.get("id", f"unknown-{i}")
        content = briefing.get("content", {})
        priority = briefing.get("priority", "normal")
        
        print(f"\n[{i}/{len(briefings)}] Processing: {briefing_id} (priority: {priority})")
        print(f"    Message: {content.get('message', 'N/A')[:60]}...")
        
        opener = await processor.generate_opener(content)
        
        result = {
            "id": briefing_id,
            "priority": priority,
            "input": content,
            "generated_opener": opener,
            "success": opener is not None,
            "char_count": len(opener) if opener else 0
        }
        results.append(result)
        
        if opener:
            print(f"    ✅ Generated: \"{opener[:80]}...\"" if len(opener) > 80 else f"    ✅ Generated: \"{opener}\"")
        else:
            print(f"    ❌ Failed to generate opener")
    
    return results


async def test_combined_opener(processor: BriefingProcessor, briefings: list) -> dict:
    """Generate a single combined opener for multiple briefings."""
    print("\n" + "=" * 60)
    print("TESTING COMBINED OPENER")
    print("=" * 60)
    
    # Format briefings for combined generation
    formatted_briefings = []
    for b in briefings:
        formatted_briefings.append({
            "id": b.get("id"),
            "content": b.get("content", {})
        })
    
    print(f"\nGenerating combined opener for {len(formatted_briefings)} briefings...")
    
    opener = await processor.generate_combined_opener(formatted_briefings)
    
    result = {
        "briefing_count": len(formatted_briefings),
        "briefing_ids": [b["id"] for b in formatted_briefings],
        "generated_opener": opener,
        "success": opener is not None,
        "char_count": len(opener) if opener else 0
    }
    
    if opener:
        print(f"\n✅ Combined opener generated ({len(opener)} chars):")
        print(f"\n    \"{opener}\"\n")
    else:
        print("\n❌ Failed to generate combined opener")
    
    return result


async def run_tests():
    """Run all briefing processor tests."""
    print("\n" + "=" * 60)
    print("BRIEFING PROCESSOR TEST SUITE")
    print("=" * 60)
    print(f"\nStarted: {datetime.now().isoformat()}")
    
    # Load sample briefings
    print(f"\nLoading briefings from: {INPUT_FILE}")
    
    if not INPUT_FILE.exists():
        print(f"❌ Input file not found: {INPUT_FILE}")
        return
    
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    briefings = data.get("briefings", [])
    print(f"✅ Loaded {len(briefings)} sample briefings")
    
    # Initialize processor
    print("\nInitializing BriefingProcessor...")
    processor = BriefingProcessor()
    
    if not processor.is_available():
        print("❌ BriefingProcessor not available (check OPENAI_API_KEY)")
        print("\nTo run this test, ensure OPENAI_API_KEY is set in your environment or .env file")
        return
    
    print(f"✅ Processor initialized (model: {processor._model})")
    
    # Run tests
    single_results = await test_single_openers(processor, briefings)
    combined_result = await test_combined_opener(processor, briefings)
    
    # Compile output
    output = {
        "test_run": {
            "timestamp": datetime.now().isoformat(),
            "input_file": str(INPUT_FILE),
            "briefing_count": len(briefings),
            "model": processor._model,
            "temperature": processor._temperature
        },
        "individual_openers": single_results,
        "combined_opener": combined_result,
        "summary": {
            "total_briefings": len(briefings),
            "successful_individual": sum(1 for r in single_results if r["success"]),
            "failed_individual": sum(1 for r in single_results if not r["success"]),
            "combined_success": combined_result["success"],
            "avg_opener_length": sum(r["char_count"] for r in single_results if r["success"]) // max(1, sum(1 for r in single_results if r["success"]))
        }
    }
    
    # Write output
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"\nIndividual openers: {output['summary']['successful_individual']}/{output['summary']['total_briefings']} successful")
    print(f"Combined opener: {'✅ Success' if output['summary']['combined_success'] else '❌ Failed'}")
    print(f"Average opener length: {output['summary']['avg_opener_length']} chars")
    
    print(f"\nWriting results to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Results saved!")
    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    
    # Run tests
    asyncio.run(run_tests())

