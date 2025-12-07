"""
Test harness for MCP tool usage with OpenAI.
Reads prompts from input.json and writes assistant responses to output.json and output.txt.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

# Mock audio modules BEFORE any imports that might need them
sys.modules['sounddevice'] = MagicMock()

# Mock the device_manager module
mock_device_manager = MagicMock()
mock_device_manager.get_emeet_device = MagicMock(return_value=None)
mock_device_manager.list_audio_devices = MagicMock()
sys.modules['assistant_framework.utils.device_manager'] = mock_device_manager

# Mock barge_in module to avoid audio dependencies
mock_barge_in = MagicMock()
sys.modules['assistant_framework.utils.barge_in'] = mock_barge_in

# Project root (mcp_server/tool_response_tests -> mcp_server -> project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()


async def run_tests():
    """Run test prompts through the assistant with MCP tools enabled."""
    
    # Paths
    tests_dir = Path(__file__).parent
    input_file = tests_dir / "input.json"
    output_file = tests_dir / "output.json"
    
    # Load test prompts (organized by category)
    with open(input_file, "r") as f:
        categories_data = json.load(f)
    
    # Flatten into list of (category, prompt) tuples while preserving order
    test_cases = []
    for category_obj in categories_data:
        for category, prompts_list in category_obj.items():
            for prompt in prompts_list:
                test_cases.append({"category": category, "prompt": prompt})
    
    print(f"Loaded {len(test_cases)} test prompts across {len(categories_data)} categories")
    
    # Now safe to import (audio modules are mocked)
    from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
    from assistant_framework.config import SYSTEM_PROMPT
    
    # Build config
    config = {
        "api_key": os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4o-realtime-preview-2024-12-17",
        "max_tokens": 2000,
        "temperature": 0.6,
        "system_prompt": SYSTEM_PROMPT,
        "mcp_server_path": str(project_root / "mcp_server" / "server.py"),
        "mcp_venv_python": str(project_root / "venv" / "bin" / "python"),
    }
    
    # Initialize provider
    provider = OpenAIWebSocketResponseProvider(config)
    initialized = await provider.initialize()
    
    if not initialized:
        print("Failed to initialize provider")
        return
    
    print(f"Provider initialized. Available tools: {len(await provider.get_available_tools())}")
    
    # Run tests and collect responses
    responses = []
    
    for i, test_case in enumerate(test_cases, 1):
        category = test_case["category"]
        prompt = test_case["prompt"]
        
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_cases)} [{category}]: {prompt}")
        print('='*60)
        
        try:
            full_response = ""
            tool_calls_info = []
            
            async for chunk in provider.stream_response(prompt):
                if chunk.content:
                    full_response = chunk.content  # Final chunk has full composed response
                if chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        tool_calls_info.append({
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "result": tc.result if hasattr(tc, 'result') else None
                        })
            
            responses.append({
                "category": category,
                "prompt": prompt,
                "response": full_response,
                "tools_used": tool_calls_info
            })
            
            print(f"\nResponse: {full_response[:500]}..." if len(full_response) > 500 else f"\nResponse: {full_response}")
            if tool_calls_info:
                print(f"Tools used: {[t['name'] for t in tool_calls_info]}")
                
        except Exception as e:
            print(f"Error: {e}")
            responses.append({
                "category": category,
                "prompt": prompt,
                "response": f"ERROR: {str(e)}",
                "tools_used": []
            })
    
    # Write output as JSON
    generated_at = datetime.now().isoformat()
    
    # Get unique categories in order
    categories_list = []
    for cat_obj in categories_data:
        for cat_name in cat_obj.keys():
            if cat_name not in categories_list:
                categories_list.append(cat_name)
    
    output_data = {
        "generated_at": generated_at,
        "total_tests": len(test_cases),
        "categories": categories_list,
        "results": responses
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Write human-readable text output organized by category
    output_txt_file = tests_dir / "output.txt"
    with open(output_txt_file, "w") as f:
        f.write("MCP Tool Test Results\n")
        f.write(f"Generated: {generated_at}\n")
        f.write(f"Total Tests: {len(test_cases)} across {len(categories_list)} categories\n")
        f.write("=" * 70 + "\n\n")
        
        # Group results by category
        results_by_category = {}
        for result in responses:
            cat = result["category"]
            if cat not in results_by_category:
                results_by_category[cat] = []
            results_by_category[cat].append(result)
        
        test_num = 1
        for category in categories_list:
            if category not in results_by_category:
                continue
            
            category_results = results_by_category[category]
            
            # Category header
            f.write("\n" + "#" * 70 + "\n")
            f.write(f"#  CATEGORY: {category.upper()}\n")
            f.write(f"#  Tests: {len(category_results)}\n")
            f.write("#" * 70 + "\n\n")
            
            for result in category_results:
                f.write(f"TEST {test_num}: {result['prompt']}\n")
                f.write("-" * 70 + "\n\n")
                
                f.write(f"RESPONSE:\n{result['response']}\n\n")
                
                if result['tools_used']:
                    f.write(f"TOOLS USED ({len(result['tools_used'])}):\n")
                    for tool in result['tools_used']:
                        f.write(f"\n  Tool: {tool['name']}\n")
                        f.write(f"  Arguments: {json.dumps(tool['arguments'], indent=4)}\n")
                        if tool.get('result'):
                            # Try to pretty-print JSON result if possible
                            try:
                                result_parsed = json.loads(tool['result'])
                                result_str = json.dumps(result_parsed, indent=4)
                            except (json.JSONDecodeError, TypeError):
                                result_str = str(tool['result'])
                            f.write(f"  Result:\n    {result_str.replace(chr(10), chr(10) + '    ')}\n")
                else:
                    f.write("TOOLS USED: None\n")
                
                f.write("\n" + "=" * 70 + "\n\n")
                test_num += 1
    
    print(f"\n\nResults written to:")
    print(f"  - {output_file}")
    print(f"  - {output_txt_file}")
    
    # Cleanup
    await provider.cleanup()


if __name__ == "__main__":
    asyncio.run(run_tests())
