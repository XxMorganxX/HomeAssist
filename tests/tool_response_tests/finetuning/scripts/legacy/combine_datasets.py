#!/usr/bin/env python3
"""
Combine all training datasets and import existing test cases.
Output: combined_training_data.jsonl
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Load tool schemas and system prompt
with open(SCRIPT_DIR / "tool_schemas.json") as f:
    TOOL_SCHEMAS = json.load(f)

with open(SCRIPT_DIR / "system_prompt.txt") as f:
    SYSTEM_PROMPT_TEMPLATE = f.read()

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.replace(
    "{tools}", 
    json.dumps(TOOL_SCHEMAS["tools"], indent=2)
)

def load_jsonl(path):
    """Load JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def convert_test_cases_to_training(test_cases_path):
    """Convert tool_subagent_examples.json to training format."""
    examples = []
    
    with open(test_cases_path) as f:
        data = json.load(f)
    
    for case in data.get("test_cases", []):
        prompt = case.get("prompt", "")
        expected = case.get("expected_tool_calls", [])
        
        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(expected)}
            ]
        }
        examples.append(example)
    
    return examples

def main():
    all_examples = []
    
    # Load single-turn examples
    single_turn = load_jsonl(SCRIPT_DIR / "training_data.jsonl")
    print(f"Loaded {len(single_turn)} single-turn examples")
    all_examples.extend(single_turn)
    
    # Load multi-turn examples
    multi_turn = load_jsonl(SCRIPT_DIR / "multiturn_training_data.jsonl")
    print(f"Loaded {len(multi_turn)} multi-turn examples")
    all_examples.extend(multi_turn)
    
    # Import from existing test cases if available
    test_cases_path = SCRIPT_DIR.parent / "tool_subagent_examples.json"
    if test_cases_path.exists():
        test_examples = convert_test_cases_to_training(test_cases_path)
        print(f"Imported {len(test_examples)} examples from test cases")
        all_examples.extend(test_examples)
    
    # Also check for .2.json version
    test_cases_path_2 = SCRIPT_DIR.parent / "tool_subagent_examples.2.json"
    if test_cases_path_2.exists():
        test_examples_2 = convert_test_cases_to_training(test_cases_path_2)
        print(f"Imported {len(test_examples_2)} examples from test cases v2")
        all_examples.extend(test_examples_2)
    
    # Load Supabase-sourced real conversation data if available
    supabase_path = SCRIPT_DIR / "supabase_training_data.jsonl"
    if supabase_path.exists():
        supabase_examples = load_jsonl(supabase_path)
        print(f"Loaded {len(supabase_examples)} real conversation examples from Supabase")
        # Supabase data goes FIRST so it takes priority in dedup
        all_examples = supabase_examples + all_examples
    
    # Deduplicate by user prompt (keep first occurrence)
    seen_prompts = set()
    unique_examples = []
    for ex in all_examples:
        # Find user message
        user_msg = None
        for msg in ex["messages"]:
            if msg["role"] == "user":
                user_msg = msg["content"]
                break
        
        if user_msg and user_msg not in seen_prompts:
            seen_prompts.add(user_msg)
            unique_examples.append(ex)
    
    print(f"\nAfter deduplication: {len(unique_examples)} unique examples")
    
    # Write combined output
    output_path = SCRIPT_DIR / "combined_training_data.jsonl"
    with open(output_path, "w") as f:
        for ex in unique_examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Output: {output_path}")
    
    # Stats
    tool_calls = 0
    for ex in unique_examples:
        try:
            parsed = json.loads(ex["messages"][-1]["content"])
            if parsed:
                tool_calls += 1
        except (json.JSONDecodeError, TypeError):
            tool_calls += 1  # Supabase data may have non-standard format
    
    print(f"\nFinal stats:")
    print(f"  Total examples: {len(unique_examples)}")
    print(f"  Tool calls: {tool_calls}")

if __name__ == "__main__":
    main()
