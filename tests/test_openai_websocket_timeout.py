"""
Tests for OpenAI WebSocket connection timeout handling.

Tests scenarios:
1. WebSocket connection timeout during pre-connect (initialization)
2. WebSocket connection timeout during ensure_ws_connected
3. WebSocket connection timeout during warm-up
4. WebSocket connection timeout during request
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp


@pytest.fixture
def mock_openai_api_key():
    """Provide a mock API key."""
    return "sk-test-mock-api-key"


@pytest.fixture
def base_config(mock_openai_api_key):
    """Provide base configuration for OpenAI WebSocket provider."""
    return {
        'api_key': mock_openai_api_key,
        'model': 'gpt-4o-mini-realtime-preview',
        'max_tokens': 2000,
        'temperature': 0.8,
        'system_prompt': 'You are a test assistant',
        'mcp_server_path': None,  # Disable MCP for pure WebSocket testing
    }


@pytest.fixture
def websocket_provider_sync(base_config):
    """Create an OpenAI WebSocket provider instance (synchronous fixture)."""
    from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
    provider = OpenAIWebSocketResponseProvider(base_config)
    return provider


class TestWebSocketConnectionTimeout:
    """Test WebSocket connection timeout scenarios."""
    
    @pytest.mark.asyncio
    async def test_preconnect_timeout_during_initialization(self, base_config):
        """Test that pre-connect timeout during initialization is handled gracefully."""
        from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
        
        provider = OpenAIWebSocketResponseProvider(base_config)
        
        # Mock ws_connect to simulate timeout
        with patch('aiohttp.ClientSession.ws_connect', side_effect=asyncio.TimeoutError()):
            # Initialize should not raise exception, just print warning
            result = await provider.initialize()
            
            # Should still return True (non-fatal pre-connect failure)
            assert result is True
            
            # WebSocket should be None after failed pre-connect
            assert provider._ws is None
        
        await provider.cleanup()
    
    @pytest.mark.asyncio
    async def test_preconnect_connection_timeout_with_message(self, base_config):
        """Test pre-connect timeout with specific timeout error message."""
        from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
        
        provider = OpenAIWebSocketResponseProvider(base_config)
        
        # Mock ws_connect to simulate the exact timeout error from the user's report
        timeout_error = Exception("Connection timeout to host wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview")
        
        with patch('aiohttp.ClientSession.ws_connect', side_effect=timeout_error):
            # Initialize should not raise exception
            result = await provider.initialize()
            
            # Should still return True (non-fatal)
            assert result is True
            
            # WebSocket should be None
            assert provider._ws is None
        
        await provider.cleanup()
    
    @pytest.mark.asyncio
    async def test_ensure_ws_connected_timeout(self, websocket_provider_sync):
        """Test that _ensure_ws_connected properly handles timeout."""
        # Mock ws_connect to raise timeout
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.ws_connect = AsyncMock(side_effect=asyncio.TimeoutError("Connection timeout"))
        websocket_provider_sync._ws_session = mock_session
        
        # Should raise the timeout exception
        with pytest.raises(asyncio.TimeoutError):
            await websocket_provider_sync._ensure_ws_connected()
        
        # WebSocket should remain None after timeout
        assert websocket_provider_sync._ws is None
        
        await websocket_provider_sync.cleanup()
    
    @pytest.mark.asyncio
    async def test_ensure_ws_warm_timeout_returns_false(self, websocket_provider_sync):
        """Test that ensure_ws_warm returns False on timeout (non-fatal)."""
        # Mock _ensure_ws_connected to raise timeout
        with patch.object(websocket_provider_sync, '_ensure_ws_connected', 
                         side_effect=asyncio.TimeoutError("Connection timeout")):
            
            # Should return False (not raise)
            result = await websocket_provider_sync.ensure_ws_warm()
            
            assert result is False
        
        await websocket_provider_sync.cleanup()
    
    @pytest.mark.asyncio
    async def test_ensure_ws_warm_generic_exception_returns_false(self, websocket_provider_sync):
        """Test that ensure_ws_warm returns False on generic connection error."""
        # Mock _ensure_ws_connected to raise connection error
        connection_error = Exception("Connection timeout to host wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview")
        
        with patch.object(websocket_provider_sync, '_ensure_ws_connected', side_effect=connection_error):
            
            # Should return False (not raise)
            result = await websocket_provider_sync.ensure_ws_warm()
            
            assert result is False
        
        await websocket_provider_sync.cleanup()
    
    @pytest.mark.asyncio
    async def test_stream_response_timeout_on_first_connect(self, websocket_provider_sync):
        """Test that stream_response handles timeout on first connection attempt."""
        # Mock ws_connect to timeout
        with patch.object(websocket_provider_sync, '_ensure_ws_connected', 
                         side_effect=asyncio.TimeoutError("Connection timeout")):
            
            # stream_response should raise exception (fatal during request)
            with pytest.raises(asyncio.TimeoutError):
                async for _ in websocket_provider_sync.stream_response("Hello"):
                    pass
        
        await websocket_provider_sync.cleanup()
    
    @pytest.mark.asyncio
    async def test_ws_connection_retry_after_preconnect_failure(self, base_config):
        """Test that WebSocket retries connection on first request after pre-connect failure."""
        from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
        
        provider = OpenAIWebSocketResponseProvider(base_config)
        
        # Simulate pre-connect failure
        with patch('aiohttp.ClientSession.ws_connect', side_effect=asyncio.TimeoutError()):
            await provider.initialize()
            assert provider._ws is None
        
        # Now mock successful connection for first actual request
        mock_ws = AsyncMock()
        mock_ws.closed = False
        
        async def mock_ws_connect(*args, **kwargs):
            return mock_ws
        
        with patch('aiohttp.ClientSession.ws_connect', side_effect=mock_ws_connect):
            # Manually trigger connection (simulating what happens in stream_response)
            ws = await provider._ensure_ws_connected()
            
            # Should successfully connect on retry
            assert ws is not None
            assert provider._ws == mock_ws
        
        await provider.cleanup()
    
    @pytest.mark.asyncio
    async def test_timeout_with_custom_timeout_duration(self, base_config):
        """Test connection timeout with custom aiohttp timeout configuration."""
        from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
        
        provider = OpenAIWebSocketResponseProvider(base_config)
        
        # Track that timeout is properly configured
        call_kwargs = {}
        
        async def mock_ws_connect(*args, **kwargs):
            call_kwargs.update(kwargs)
            raise asyncio.TimeoutError("Timeout after configured duration")
        
        with patch('aiohttp.ClientSession.ws_connect', side_effect=mock_ws_connect):
            try:
                await provider._ensure_ws_connected()
            except asyncio.TimeoutError:
                pass
            
            # Verify timeout was configured (30 seconds as per line 265)
            assert 'timeout' in call_kwargs
            timeout_obj = call_kwargs['timeout']
            assert isinstance(timeout_obj, aiohttp.ClientTimeout)
            assert timeout_obj.total == 30
        
        await provider.cleanup()
    
    @pytest.mark.asyncio
    async def test_ws_connection_marked_dead_on_timeout(self, websocket_provider_sync):
        """Test that WebSocket connection is marked as dead after timeout."""
        # Setup: establish a mock connection first
        mock_ws = MagicMock()
        mock_ws.closed = False
        websocket_provider_sync._ws = mock_ws
        
        # Simulate timeout during stream operation
        with patch.object(websocket_provider_sync, '_ensure_ws_connected', side_effect=asyncio.TimeoutError()):
            try:
                async for _ in websocket_provider_sync.stream_response("test"):
                    pass
            except asyncio.TimeoutError:
                pass
        
        # WebSocket should be marked as None (dead) after timeout
        # This happens in the exception handler of _ws_stream_roundtrip (line 705)
        assert websocket_provider_sync._ws is None
        
        await websocket_provider_sync.cleanup()
    
    @pytest.mark.asyncio
    async def test_parallel_initialization_timeout(self, base_config):
        """Test that parallel MCP + WebSocket initialization handles WS timeout gracefully."""
        from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
        
        # Enable MCP to test parallel initialization
        base_config['mcp_server_path'] = '/fake/path/to/mcp'
        provider = OpenAIWebSocketResponseProvider(base_config)
        
        # Mock both MCP and WebSocket initialization
        with patch.object(provider, '_initialize_mcp', return_value=None), \
             patch('aiohttp.ClientSession.ws_connect', side_effect=asyncio.TimeoutError()):
            
            # Should complete initialization despite WS timeout
            result = await provider.initialize()
            
            # MCP init might succeed, WS init fails - still returns True
            assert result is True
            assert provider._ws is None
        
        await provider.cleanup()


class TestWebSocketConnectionRecovery:
    """Test WebSocket connection recovery after timeout."""
    
    @pytest.mark.asyncio
    async def test_connection_recovery_after_timeout(self, base_config):
        """Test that connection can recover after a timeout."""
        from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
        
        provider = OpenAIWebSocketResponseProvider(base_config)
        
        # First attempt: timeout
        with patch('aiohttp.ClientSession.ws_connect', side_effect=asyncio.TimeoutError()):
            try:
                await provider._ensure_ws_connected()
            except asyncio.TimeoutError:
                pass
            
            assert provider._ws is None
        
        # Second attempt: success
        mock_ws = AsyncMock()
        mock_ws.closed = False
        
        with patch('aiohttp.ClientSession.ws_connect', return_value=mock_ws):
            ws = await provider._ensure_ws_connected()
            
            assert ws is not None
            assert provider._ws == mock_ws
        
        await provider.cleanup()
    
    @pytest.mark.asyncio
    async def test_heartbeat_timeout_triggers_reconnect(self, websocket_provider_sync):
        """Test that heartbeat timeout triggers reconnection."""
        import time
        
        # Setup: establish a connection
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.ping = AsyncMock(side_effect=Exception("Heartbeat timeout"))
        
        websocket_provider_sync._ws = mock_ws
        websocket_provider_sync._last_heartbeat = time.time() - 30  # 30 seconds ago
        
        # Mock new connection
        new_mock_ws = AsyncMock()
        new_mock_ws.closed = False
        
        with patch('aiohttp.ClientSession.ws_connect', return_value=new_mock_ws):
            # Should detect stale heartbeat and reconnect
            ws = await websocket_provider_sync._ensure_ws_connected()
            
            # Should return new connection
            assert ws == new_mock_ws
            assert websocket_provider_sync._ws == new_mock_ws
        
        await websocket_provider_sync.cleanup()


class TestWebSocketErrorMessages:
    """Test that error messages are properly formatted."""
    
    @pytest.mark.asyncio
    async def test_preconnect_error_message_format(self, base_config, capsys):
        """Test that pre-connect error message matches expected format."""
        from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider
        
        provider = OpenAIWebSocketResponseProvider(base_config)
        
        error_msg = "Connection timeout to host wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview"
        
        with patch('aiohttp.ClientSession.ws_connect', side_effect=Exception(error_msg)):
            await provider.initialize()
            
            # Check that error message was printed
            captured = capsys.readouterr()
            assert "⚠️  WebSocket pre-connect failed" in captured.out
            assert "(will retry on first request)" in captured.out
        
        await provider.cleanup()
    
    @pytest.mark.asyncio
    async def test_connection_failed_error_message_format(self, websocket_provider_sync, capsys):
        """Test that connection failed error message matches expected format."""
        error_msg = "Connection timeout to host wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview"
        
        with patch('aiohttp.ClientSession.ws_connect', side_effect=Exception(error_msg)):
            try:
                await websocket_provider_sync._ensure_ws_connected()
            except Exception:
                pass
            
            # Check error message format
            captured = capsys.readouterr()
            assert "❌ WebSocket connection failed:" in captured.out
        
        await websocket_provider_sync.cleanup()
    
    @pytest.mark.asyncio
    async def test_warm_up_error_message_format(self, websocket_provider_sync, capsys):
        """Test that warm-up error message matches expected format."""
        error_msg = "Connection timeout to host wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview"
        
        with patch.object(websocket_provider_sync, '_ensure_ws_connected', side_effect=Exception(error_msg)):
            await websocket_provider_sync.ensure_ws_warm()
            
            # Check error message format
            captured = capsys.readouterr()
            assert "⚠️  WebSocket warm-up failed" in captured.out
            assert "(will retry on request)" in captured.out
        
        await websocket_provider_sync.cleanup()
