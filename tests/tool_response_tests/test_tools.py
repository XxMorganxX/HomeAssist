"""
Test harness for MCP tool usage with OpenAI.
Reads prompts from input.json and writes assistant responses to output.json and output.txt.

Tests the tool signal flow:
1. Realtime model outputs "TOOL" signal (minimal tokens)
2. gpt-4o-mini orchestrates tool calls
3. Tool results sent back to realtime for final response

Usage:
    python test_tools.py                    # Run all tests
    python test_tools.py --single "prompt"  # Run a single prompt
"""

import asyncio
import json
import os
import sys
import io
import time
import re
import argparse
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

# Mock audio modules BEFORE any imports that might need them
sys.modules['sounddevice'] = MagicMock()

# Mock the device_manager module
mock_device_manager = MagicMock()
mock_device_manager.get_emeet_device = MagicMock(return_value=None)
mock_device_manager.list_audio_devices = MagicMock()
sys.modules['assistant_framework.utils.audio.device_manager'] = mock_device_manager

# Mock barge_in module to avoid audio dependencies
mock_barge_in = MagicMock()
sys.modules['assistant_framework.utils.audio.barge_in'] = mock_barge_in

# Mock tones module to avoid audio during tests
mock_tones = MagicMock()
mock_tones.beep_tools_complete = MagicMock()
sys.modules['assistant_framework.utils.audio.tones'] = mock_tones

# Project root (test_tools.py -> tool_response_tests -> tests -> project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()


class ToolSignalCapture:
    """Capture tool signal flow information during tests."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.tool_signal_detected = False
        self.realtime_raw_output = None
        self.orchestration_called = False
        self.orchestration_model = None
        self.realtime_compose_called = False
        self.logs = []
        # Timing from logs (if available)
        self.tool_execution_time_ms = None
        self.orchestration_time_ms = None
    
    def parse_logs(self, log_output: str):
        """Parse captured logs to extract tool signal flow info."""
        self.logs = log_output.split('\n')
        
        for line in self.logs:
            # Check for tool signal detection
            if "Tool handoff detected" in line or "Is tool signal: True" in line:
                self.tool_signal_detected = True
            
            # Capture realtime raw output
            if "Preview:" in line and "TOOL_SIGNAL_MODE" not in line:
                # Extract the preview content
                parts = line.split("Preview:", 1)
                if len(parts) > 1:
                    self.realtime_raw_output = parts[1].strip()
            
            # Check for orchestration
            if "[Tool Orchestration] Starting" in line:
                self.orchestration_called = True
            
            # Capture orchestration model
            if "[Tool Orchestration] Calling" in line and "messages" in line:
                # Extract model name
                if "gpt-4o-mini" in line:
                    self.orchestration_model = "gpt-4o-mini"
                elif "o4-mini" in line:
                    self.orchestration_model = "o4-mini"
            
            # Check for realtime compose
            if "[Realtime Compose]" in line or "Sending tool results to realtime" in line:
                self.realtime_compose_called = True
            
            # Parse tool execution timing (e.g., "⏱️ Tool 'calendar_data' executed in 234ms")
            if "executed in" in line and "ms" in line:
                match = re.search(r'executed in (\d+)ms', line)
                if match:
                    exec_time = int(match.group(1))
                    if self.tool_execution_time_ms is None:
                        self.tool_execution_time_ms = exec_time
                    else:
                        self.tool_execution_time_ms += exec_time
    
    def to_dict(self):
        """Convert capture to dictionary for output."""
        return {
            "tool_signal_detected": self.tool_signal_detected,
            "realtime_raw_output": self.realtime_raw_output,
            "orchestration_called": self.orchestration_called,
            "orchestration_model": self.orchestration_model,
            "realtime_compose_called": self.realtime_compose_called,
            "tool_execution_time_ms": self.tool_execution_time_ms,
        }


async def run_tests(single_prompt=None):
    """Run test prompts through the assistant with MCP tools enabled.
    
    Args:
        single_prompt: If provided, run only this specific prompt. Otherwise run all tests.
    """
    
    # Paths
    tests_dir = Path(__file__).parent
    input_file = tests_dir / "input.json"
    output_file = tests_dir / "output.json"
    
    # Load test prompts (organized by category)
    with open(input_file, "r") as f:
        categories_data = json.load(f)
    
    # Flatten into list of (category, prompt) tuples while preserving order
    all_test_cases = []
    for category_obj in categories_data:
        for category, prompts_list in category_obj.items():
            for prompt in prompts_list:
                all_test_cases.append({"category": category, "prompt": prompt})
    
    # Filter to single prompt if specified
    if single_prompt:
        test_cases = [tc for tc in all_test_cases if tc["prompt"] == single_prompt]
        if not test_cases:
            print(f"ERROR: Prompt not found: {single_prompt}")
            print(f"Available prompts:")
            for tc in all_test_cases:
                print(f"  - {tc['prompt']}")
            return
        print(f"Running single test: {single_prompt}")
    else:
        test_cases = all_test_cases
        print(f"Loaded {len(test_cases)} test prompts across {len(categories_data)} categories")
    
    # Now safe to import (audio modules are mocked)
    from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
    from assistant_framework.config import SYSTEM_PROMPT, TOOL_SIGNAL_MODE
    
    print(f"TOOL_SIGNAL_MODE: {TOOL_SIGNAL_MODE}")
    
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
    signal_capture = ToolSignalCapture()
    
    # Stats for tool signal mode
    stats = {
        "total_tests": len(test_cases),
        "tool_signal_detected": 0,
        "orchestration_called": 0,
        "realtime_compose_called": 0,
        "tools_used": 0,
        "no_tools_used": 0,
    }
    
    # Timing stats
    timing_stats = {
        "total_time_ms": [],
        "tool_execution_time_ms": [],
    }
    
    for i, test_case in enumerate(test_cases, 1):
        category = test_case["category"]
        prompt = test_case["prompt"]
        
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_cases)} [{category}]: {prompt}")
        print('='*60)
        
        # Reset capture for this test
        signal_capture.reset()
        
        try:
            full_response = ""
            tool_calls_info = []
            
            # Track timing
            start_time = time.perf_counter()
            
            # Capture stdout to parse tool signal flow logs
            log_capture = io.StringIO()
            
            # We need to capture print output during stream_response
            # Use a custom approach: store original stdout and capture
            import sys
            old_stdout = sys.stdout
            sys.stdout = log_capture
            
            try:
                async for chunk in provider.stream_response(prompt):
                    if chunk.content:
                        full_response = chunk.content
                    if chunk.tool_calls:
                        for tc in chunk.tool_calls:
                            tool_calls_info.append({
                                "name": tc.name,
                                "arguments": tc.arguments,
                                "result": tc.result if hasattr(tc, 'result') else None
                            })
            finally:
                sys.stdout = old_stdout
            
            # Calculate total time
            end_time = time.perf_counter()
            total_time_ms = int((end_time - start_time) * 1000)
            
            # Parse the captured logs
            captured_output = log_capture.getvalue()
            signal_capture.parse_logs(captured_output)
            
            # Print the captured logs for visibility
            if captured_output:
                print(captured_output)
            
            # Update stats
            if signal_capture.tool_signal_detected:
                stats["tool_signal_detected"] += 1
            if signal_capture.orchestration_called:
                stats["orchestration_called"] += 1
            if signal_capture.realtime_compose_called:
                stats["realtime_compose_called"] += 1
            if tool_calls_info:
                stats["tools_used"] += 1
            else:
                stats["no_tools_used"] += 1
            
            # Update timing stats
            timing_stats["total_time_ms"].append(total_time_ms)
            if signal_capture.tool_execution_time_ms is not None:
                timing_stats["tool_execution_time_ms"].append(signal_capture.tool_execution_time_ms)
            
            responses.append({
                "category": category,
                "prompt": prompt,
                "response": full_response,
                "tools_used": tool_calls_info,
                "tool_signal_flow": signal_capture.to_dict(),
                "timing": {
                    "total_time_ms": total_time_ms,
                    "tool_execution_time_ms": signal_capture.tool_execution_time_ms,
                },
            })
            
            print(f"\nResponse: {full_response[:500]}..." if len(full_response) > 500 else f"\nResponse: {full_response}")
            if tool_calls_info:
                print(f"Tools used: {[t['name'] for t in tool_calls_info]}")
            
            # Print tool signal flow info
            print("\n📊 Tool Signal Flow:")
            print(f"   Signal detected: {signal_capture.tool_signal_detected}")
            print(f"   Realtime output: {signal_capture.realtime_raw_output}")
            print(f"   Orchestration: {signal_capture.orchestration_called} (model: {signal_capture.orchestration_model})")
            print(f"   Realtime compose: {signal_capture.realtime_compose_called}")
            
            # Print timing info
            print("\n⏱️ Timing:")
            print(f"   Total time: {total_time_ms}ms")
            if signal_capture.tool_execution_time_ms:
                print(f"   Tool execution: {signal_capture.tool_execution_time_ms}ms")
                
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            traceback.print_exc()
            responses.append({
                "category": category,
                "prompt": prompt,
                "response": f"ERROR: {str(e)}",
                "tools_used": [],
                "tool_signal_flow": signal_capture.to_dict(),
            })
    
    # Write output as JSON
    generated_at = datetime.now().isoformat()
    
    # Get unique categories in order
    categories_list = []
    for cat_obj in categories_data:
        for cat_name in cat_obj.keys():
            if cat_name not in categories_list:
                categories_list.append(cat_name)
    
    # Calculate timing summary stats
    def calc_stats(values):
        if not values:
            return {"avg": None, "min": None, "max": None, "total": None}
        return {
            "avg": int(sum(values) / len(values)),
            "min": min(values),
            "max": max(values),
            "total": sum(values),
        }
    
    timing_summary = {
        "total_time": calc_stats(timing_stats["total_time_ms"]),
        "tool_execution_time": calc_stats(timing_stats["tool_execution_time_ms"]),
    }
    
    # If running single test, merge with existing results
    if single_prompt:
        # Load existing output if it exists
        existing_data = {}
        if output_file.exists():
            try:
                with open(output_file, "r") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Update or add the single result
        existing_results = existing_data.get("results", [])
        
        # Remove any existing result for this prompt
        existing_results = [r for r in existing_results if r["prompt"] != single_prompt]
        
        # Add the new result
        existing_results.extend(responses)
        
        # Recalculate stats for all results
        total_tests = len(existing_results)
        signal_detected_count = sum(1 for r in existing_results if r.get("tool_signal_flow", {}).get("tool_signal_detected"))
        orch_called_count = sum(1 for r in existing_results if r.get("tool_signal_flow", {}).get("orchestration_called"))
        realtime_compose_count = sum(1 for r in existing_results if r.get("tool_signal_flow", {}).get("realtime_compose_called"))
        tools_used_count = sum(1 for r in existing_results if r.get("tools_used"))
        no_tools_count = total_tests - tools_used_count
        
        # Recalculate timing stats
        all_total_times = [r.get("timing", {}).get("total_time_ms") for r in existing_results if r.get("timing", {}).get("total_time_ms")]
        all_tool_times = [r.get("timing", {}).get("tool_execution_time_ms") for r in existing_results if r.get("timing", {}).get("tool_execution_time_ms")]
        
        output_data = {
            "generated_at": generated_at,
            "total_tests": total_tests,
            "tool_signal_mode": TOOL_SIGNAL_MODE,
            "categories": categories_list,
            "tool_signal_stats": {
                "total_tests": total_tests,
                "tool_signal_detected": signal_detected_count,
                "orchestration_called": orch_called_count,
                "realtime_compose_called": realtime_compose_count,
                "tools_used": tools_used_count,
                "no_tools_used": no_tools_count,
            },
            "timing_summary": {
                "total_time": calc_stats(all_total_times),
                "tool_execution_time": calc_stats(all_tool_times),
            },
            "results": existing_results
        }
    else:
        # Running all tests - output complete data
        output_data = {
            "generated_at": generated_at,
            "total_tests": len(test_cases),
            "tool_signal_mode": TOOL_SIGNAL_MODE,
            "categories": categories_list,
            "tool_signal_stats": stats,
            "timing_summary": timing_summary,
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
        f.write(f"TOOL_SIGNAL_MODE: {TOOL_SIGNAL_MODE}\n")
        f.write("=" * 70 + "\n\n")
        
        # Tool Signal Stats Summary
        f.write("TOOL SIGNAL FLOW STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total tests:              {stats['total_tests']}\n")
        f.write(f"Tool signal detected:     {stats['tool_signal_detected']}\n")
        f.write(f"Orchestration called:     {stats['orchestration_called']}\n")
        f.write(f"Realtime compose called:  {stats['realtime_compose_called']}\n")
        f.write(f"Tests with tools:         {stats['tools_used']}\n")
        f.write(f"Tests without tools:      {stats['no_tools_used']}\n")
        f.write("=" * 70 + "\n\n")
        
        # Timing Stats Summary
        f.write("TIMING STATISTICS\n")
        f.write("-" * 70 + "\n")
        ts = timing_summary["total_time"]
        if ts["avg"] is not None:
            f.write("Total Response Time:\n")
            f.write(f"  Average:  {ts['avg']}ms\n")
            f.write(f"  Min:      {ts['min']}ms\n")
            f.write(f"  Max:      {ts['max']}ms\n")
        else:
            f.write("Total Response Time:  N/A\n")
        
        te = timing_summary["tool_execution_time"]
        if te["avg"] is not None:
            f.write("Tool Execution Time:\n")
            f.write(f"  Average:  {te['avg']}ms\n")
            f.write(f"  Min:      {te['min']}ms\n")
            f.write(f"  Max:      {te['max']}ms\n")
            f.write(f"  Total:    {te['total']}ms\n")
        else:
            f.write("Tool Execution Time:  N/A\n")
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
                
                # Timing
                timing = result.get('timing', {})
                f.write("TIMING:\n")
                f.write(f"  Total time:       {timing.get('total_time_ms', 'N/A')}ms\n")
                tool_exec_time = timing.get('tool_execution_time_ms')
                f.write(f"  Tool execution:   {tool_exec_time}ms\n" if tool_exec_time else "  Tool execution:   N/A\n")
                f.write("\n")
                
                # Tool Signal Flow
                flow = result.get('tool_signal_flow', {})
                f.write("TOOL SIGNAL FLOW:\n")
                f.write(f"  Signal detected:     {flow.get('tool_signal_detected', 'N/A')}\n")
                f.write(f"  Realtime output:     {flow.get('realtime_raw_output', 'N/A')}\n")
                f.write(f"  Orchestration:       {flow.get('orchestration_called', 'N/A')}\n")
                f.write(f"  Orchestration model: {flow.get('orchestration_model', 'N/A')}\n")
                f.write(f"  Realtime compose:    {flow.get('realtime_compose_called', 'N/A')}\n\n")
                
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
    
    # Print summary
    print(f"\n\n{'='*70}")
    if single_prompt:
        print("SINGLE TEST COMPLETE")
    else:
        print("TOOL SIGNAL FLOW SUMMARY")
    print('='*70)
    
    if single_prompt:
        # Show just the single test result
        if responses:
            result = responses[0]
            flow = result.get("tool_signal_flow", {})
            timing = result.get("timing", {})
            print(f"Prompt: {result['prompt']}")
            print(f"Tool signal detected:     {flow.get('tool_signal_detected')}")
            print(f"Orchestration called:     {flow.get('orchestration_called')}")
            print(f"Realtime compose called:  {flow.get('realtime_compose_called')}")
            print(f"Tools used:               {len(result.get('tools_used', []))}")
            print(f"Total time:               {timing.get('total_time_ms')}ms")
            if timing.get('tool_execution_time_ms'):
                print(f"Tool execution time:      {timing.get('tool_execution_time_ms')}ms")
    else:
        # Show full stats
        print(f"TOOL_SIGNAL_MODE:         {TOOL_SIGNAL_MODE}")
        print(f"Total tests:              {stats['total_tests']}")
        print(f"Tool signal detected:     {stats['tool_signal_detected']} ({100*stats['tool_signal_detected']//max(1,stats['total_tests'])}%)")
        print(f"Orchestration called:     {stats['orchestration_called']} ({100*stats['orchestration_called']//max(1,stats['total_tests'])}%)")
        print(f"Realtime compose called:  {stats['realtime_compose_called']} ({100*stats['realtime_compose_called']//max(1,stats['total_tests'])}%)")
        print(f"Tests with tools:         {stats['tools_used']}")
        print(f"Tests without tools:      {stats['no_tools_used']}")
    print('='*70)
    
    if not single_prompt:
        # Print timing summary for full test runs
        print("\n⏱️ TIMING SUMMARY")
        print('='*70)
        ts = timing_summary["total_time"]
        if ts["avg"] is not None:
            print("Total Response Time:")
            print(f"  Average:  {ts['avg']}ms ({ts['avg']/1000:.2f}s)")
            print(f"  Min:      {ts['min']}ms ({ts['min']/1000:.2f}s)")
            print(f"  Max:      {ts['max']}ms ({ts['max']/1000:.2f}s)")
        else:
            print("Total Response Time:  N/A")
        
        te = timing_summary["tool_execution_time"]
        if te["avg"] is not None:
            print("Tool Execution Time:")
            print(f"  Average:  {te['avg']}ms")
            print(f"  Min:      {te['min']}ms")
            print(f"  Max:      {te['max']}ms")
            print(f"  Total:    {te['total']}ms ({te['total']/1000:.2f}s)")
        else:
            print("Tool Execution Time:  N/A (no tool timing captured)")
        print('='*70)
    
    print("\nResults written to:")
    print(f"  - {output_file}")
    print(f"  - {output_txt_file}")
    
    # Cleanup
    await provider.cleanup()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MCP tool tests")
    parser.add_argument("--single", type=str, help="Run a single test prompt")
    args = parser.parse_args()
    
    asyncio.run(run_tests(single_prompt=args.single))
