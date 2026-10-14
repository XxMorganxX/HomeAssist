#!/usr/bin/env python3
"""
Test runner for OpenAI WebSocket connection timeout tests.

This script can be run directly without pytest:
    python tests/run_websocket_tests.py
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test modules
try:
    # Try running with pytest if available
    import pytest
    
    print("Running WebSocket connection timeout tests with pytest...\n")
    
    # Run the tests
    exit_code = pytest.main([
        'tests/test_openai_websocket.py',
        '-v',
        '--tb=short',
        '-s'  # Show print statements
    ])
    
    sys.exit(exit_code)
    
except ImportError:
    print("pytest not found, running tests manually...\n")
    
    # Manual test execution
    from unittest.mock import AsyncMock, MagicMock, patch
    from aiohttp.client_exceptions import ServerTimeoutError, ClientConnectorError
    from aiohttp import ClientTimeout
    
    from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
    
    # Simple test counter
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
    
    # Test 1: Initialization with pre-connect timeout
    async def test_preconnect_timeout():
        try:
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
                
                # Should return True despite timeout (non-fatal)
                result = await provider.initialize()
                
                if result is True:
                    test_result("test_preconnect_timeout", True)
                else:
                    test_result("test_preconnect_timeout", False, "Expected True from initialize()")
                    
        except Exception as e:
            test_result("test_preconnect_timeout", False, str(e))
    
    # Test 2: Connection timeout on ensure_ws_connected
    async def test_ensure_ws_timeout():
        try:
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
                
                # Should raise the timeout error
                try:
                    await provider._ensure_ws_connected()
                    test_result("test_ensure_ws_timeout", False, "Expected ServerTimeoutError")
                except ServerTimeoutError:
                    test_result("test_ensure_ws_timeout", True)
                    
        except Exception as e:
            test_result("test_ensure_ws_timeout", False, str(e))
    
    # Test 3: Connection with proper timeout config
    async def test_timeout_config():
        try:
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
                
                await provider._ensure_ws_connected()
                
                # Check that ws_connect was called with proper timeout
                call_kwargs = mock_session.ws_connect.call_args[1]
                timeout = call_kwargs.get('timeout')
                
                if isinstance(timeout, ClientTimeout) and timeout.total == 30:
                    test_result("test_timeout_config", True)
                else:
                    test_result("test_timeout_config", False, f"Expected ClientTimeout(total=30), got {timeout}")
                    
        except Exception as e:
            test_result("test_timeout_config", False, str(e))
    
    # Test 4: Reconnect after timeout
    async def test_reconnect_after_timeout():
        try:
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
                
                call_count = [0]
                async def connect_with_retry(*args, **kwargs):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        raise ServerTimeoutError("Connection timeout")
                    return mock_ws
                
                mock_session.ws_connect = AsyncMock(side_effect=connect_with_retry)
                
                # First attempt should fail
                try:
                    await provider._ensure_ws_connected()
                    test_result("test_reconnect_after_timeout", False, "Expected first connection to fail")
                    return
                except ServerTimeoutError:
                    pass
                
                # Second attempt should succeed
                ws = await provider._ensure_ws_connected()
                
                if ws is mock_ws and mock_session.ws_connect.call_count == 2:
                    test_result("test_reconnect_after_timeout", True)
                else:
                    test_result("test_reconnect_after_timeout", False, f"Expected 2 calls, got {mock_session.ws_connect.call_count}")
                    
        except Exception as e:
            test_result("test_reconnect_after_timeout", False, str(e))
    
    # Test 5: ensure_ws_warm handles timeout gracefully
    async def test_ensure_ws_warm_timeout():
        try:
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
                
                # Should return False but not raise
                result = await provider.ensure_ws_warm()
                
                if result is False:
                    test_result("test_ensure_ws_warm_timeout", True)
                else:
                    test_result("test_ensure_ws_warm_timeout", False, "Expected False from ensure_ws_warm()")
                    
        except Exception as e:
            test_result("test_ensure_ws_warm_timeout", False, str(e))
    
    # Run all tests
    async def run_all_tests():
        print("=" * 60)
        print("OpenAI WebSocket Connection Timeout Tests")
        print("=" * 60)
        print()
        
        await test_preconnect_timeout()
        await test_ensure_ws_timeout()
        await test_timeout_config()
        await test_reconnect_after_timeout()
        await test_ensure_ws_warm_timeout()
        
        print()
        print("=" * 60)
        print(f"Results: {tests_passed} passed, {tests_failed} failed")
        print("=" * 60)
        
        return 0 if tests_failed == 0 else 1
    
    # Execute
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
