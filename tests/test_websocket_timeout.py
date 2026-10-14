#!/usr/bin/env python3
"""
Minimal test for OpenAI WebSocket connection timeout scenarios.

Tests the specific error messages that appear when WebSocket connections fail:
    ❌ WebSocket connection failed: Connection timeout to host wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview
    ⚠️  WebSocket pre-connect failed (will retry on first request): Connection timeout to host wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview

This test simulates the actual error scenarios without requiring full dependencies.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from io import StringIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Simple test results
tests_passed = 0
tests_failed = 0

def test_result(test_name: str, passed: bool, error: str = None):
    global tests_passed, tests_failed
    if passed:
        tests_passed += 1
        print(f"✅ {test_name}")
    else:
        tests_failed += 1
        print(f"❌ {test_name}")
        if error:
            print(f"   Error: {error}")


async def test_websocket_preconnect_timeout_message():
    """
    Test that initialize() prints the correct warning message when WebSocket pre-connect times out.
    
    Expected output:
        ⚠️  WebSocket pre-connect failed (will retry on first request): Connection timeout to host...
    """
    try:
        # Import only what we need
        from aiohttp.client_exceptions import ServerTimeoutError
        
        # Dynamically import the provider to avoid dependency issues
        try:
            from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
        except ImportError as e:
            test_result("test_websocket_preconnect_timeout_message", False, f"Import failed: {e}")
            return
        
        config = {
            'api_key': 'test-api-key',
            'model': 'gpt-4o-mini-realtime-preview',
            'max_tokens': 2000,
            'mcp_server_path': None,  # Disable MCP to avoid additional dependencies
        }
        
        provider = OpenAIWebSocketResponseProvider(config)
        
        # Mock aiohttp.ClientSession to simulate timeout
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            # Simulate connection timeout
            async def timeout_error(*args, **kwargs):
                raise ServerTimeoutError("Connection timeout to host wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview")
            
            mock_session.ws_connect = AsyncMock(side_effect=timeout_error)
            
            # Capture stdout to check for the warning message
            captured_output = StringIO()
            original_stdout = sys.stdout
            sys.stdout = captured_output
            
            try:
                # Initialize should succeed despite WebSocket timeout (non-fatal)
                result = await provider.initialize()
                
                # Restore stdout
                sys.stdout = original_stdout
                output = captured_output.getvalue()
                
                # Check results
                if result is not True:
                    test_result("test_websocket_preconnect_timeout_message", False, "Expected initialize() to return True")
                    return
                
                # Check for the expected warning message
                if "⚠️  WebSocket pre-connect failed" in output and "Connection timeout" in output:
                    test_result("test_websocket_preconnect_timeout_message", True)
                else:
                    test_result("test_websocket_preconnect_timeout_message", False, 
                               f"Expected warning message not found in output:\n{output}")
            finally:
                sys.stdout = original_stdout
                
    except Exception as e:
        test_result("test_websocket_preconnect_timeout_message", False, str(e))
        import traceback
        traceback.print_exc()


async def test_websocket_connection_failed_message():
    """
    Test that _ensure_ws_connected() prints the correct error message when connection fails.
    
    Expected output:
        ❌ WebSocket connection failed: Connection timeout to host...
    """
    try:
        from aiohttp.client_exceptions import ServerTimeoutError
        
        try:
            from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
        except ImportError as e:
            test_result("test_websocket_connection_failed_message", False, f"Import failed: {e}")
            return
        
        config = {
            'api_key': 'test-api-key',
            'model': 'gpt-4o-mini-realtime-preview',
            'max_tokens': 2000,
            'mcp_server_path': None,
        }
        
        provider = OpenAIWebSocketResponseProvider(config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            async def timeout_error(*args, **kwargs):
                raise ServerTimeoutError("Connection timeout to host wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview")
            
            mock_session.ws_connect = AsyncMock(side_effect=timeout_error)
            
            # Capture stdout
            captured_output = StringIO()
            original_stdout = sys.stdout
            sys.stdout = captured_output
            
            try:
                # Should raise the timeout error
                try:
                    await provider._ensure_ws_connected()
                    sys.stdout = original_stdout
                    test_result("test_websocket_connection_failed_message", False, 
                               "Expected ServerTimeoutError to be raised")
                    return
                except ServerTimeoutError:
                    pass  # Expected
                
                sys.stdout = original_stdout
                output = captured_output.getvalue()
                
                # Check for the error message
                if "❌ WebSocket connection failed" in output and "Connection timeout" in output:
                    test_result("test_websocket_connection_failed_message", True)
                else:
                    test_result("test_websocket_connection_failed_message", False, 
                               f"Expected error message not found in output:\n{output}")
                    
            finally:
                sys.stdout = original_stdout
                
    except Exception as e:
        test_result("test_websocket_connection_failed_message", False, str(e))
        import traceback
        traceback.print_exc()


async def test_websocket_warm_up_failed_message():
    """
    Test that ensure_ws_warm() prints the correct warning message when warm-up fails.
    
    Expected output:
        ⚠️  WebSocket warm-up failed (will retry on request): ...
    """
    try:
        from aiohttp.client_exceptions import ServerTimeoutError
        
        try:
            from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
        except ImportError as e:
            test_result("test_websocket_warm_up_failed_message", False, f"Import failed: {e}")
            return
        
        config = {
            'api_key': 'test-api-key',
            'model': 'gpt-4o-mini-realtime-preview',
            'max_tokens': 2000,
            'mcp_server_path': None,
        }
        
        provider = OpenAIWebSocketResponseProvider(config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            async def timeout_error(*args, **kwargs):
                raise ServerTimeoutError("Connection timeout")
            
            mock_session.ws_connect = AsyncMock(side_effect=timeout_error)
            
            # Capture stdout
            captured_output = StringIO()
            original_stdout = sys.stdout
            sys.stdout = captured_output
            
            try:
                # Should return False but not raise
                result = await provider.ensure_ws_warm()
                
                sys.stdout = original_stdout
                output = captured_output.getvalue()
                
                if result is not False:
                    test_result("test_websocket_warm_up_failed_message", False, 
                               "Expected ensure_ws_warm() to return False")
                    return
                
                # Check for the warning message
                if "⚠️  WebSocket warm-up failed" in output:
                    test_result("test_websocket_warm_up_failed_message", True)
                else:
                    test_result("test_websocket_warm_up_failed_message", False, 
                               f"Expected warning message not found in output:\n{output}")
                    
            finally:
                sys.stdout = original_stdout
                
    except Exception as e:
        test_result("test_websocket_warm_up_failed_message", False, str(e))
        import traceback
        traceback.print_exc()


async def test_websocket_connection_timeout_config():
    """
    Test that WebSocket connections use the correct timeout configuration (30 seconds).
    """
    try:
        from aiohttp import ClientTimeout
        
        try:
            from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
        except ImportError as e:
            test_result("test_websocket_connection_timeout_config", False, f"Import failed: {e}")
            return
        
        config = {
            'api_key': 'test-api-key',
            'model': 'gpt-4o-mini-realtime-preview',
            'max_tokens': 2000,
            'mcp_server_path': None,
        }
        
        provider = OpenAIWebSocketResponseProvider(config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            mock_ws = AsyncMock()
            mock_ws.closed = False
            mock_session.ws_connect = AsyncMock(return_value=mock_ws)
            
            # Suppress stdout
            original_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                await provider._ensure_ws_connected()
                
                sys.stdout = original_stdout
                
                # Verify ws_connect was called with proper timeout
                call_kwargs = mock_session.ws_connect.call_args[1]
                timeout = call_kwargs.get('timeout')
                
                if isinstance(timeout, ClientTimeout) and timeout.total == 30:
                    test_result("test_websocket_connection_timeout_config", True)
                else:
                    test_result("test_websocket_connection_timeout_config", False, 
                               f"Expected ClientTimeout(total=30), got {timeout}")
                    
            finally:
                sys.stdout = original_stdout
                
    except Exception as e:
        test_result("test_websocket_connection_timeout_config", False, str(e))
        import traceback
        traceback.print_exc()


async def run_all_tests():
    print("=" * 80)
    print("OpenAI WebSocket Connection Timeout Tests")
    print("=" * 80)
    print()
    print("Testing error messages and timeout handling for:")
    print("  ❌ WebSocket connection failed: Connection timeout...")
    print("  ⚠️  WebSocket pre-connect failed (will retry on first request)...")
    print()
    
    await test_websocket_preconnect_timeout_message()
    await test_websocket_connection_failed_message()
    await test_websocket_warm_up_failed_message()
    await test_websocket_connection_timeout_config()
    
    print()
    print("=" * 80)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 80)
    
    return 0 if tests_failed == 0 else 1


if __name__ == '__main__':
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
