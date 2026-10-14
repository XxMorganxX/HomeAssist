"""
Tests for OpenAI WebSocket response provider connection handling.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from aiohttp import ClientTimeout, WSMsgType, ClientWebSocketResponse
from aiohttp.client_exceptions import ClientConnectorError, ServerTimeoutError

from assistant_framework.providers.response.openai_websocket import OpenAIWebSocketResponseProvider


@pytest.fixture
def websocket_config():
    """Configuration for WebSocket provider."""
    return {
        'api_key': 'test-api-key',
        'model': 'gpt-4o-mini-realtime-preview',
        'max_tokens': 2000,
        'temperature': 0.8,
        'system_prompt': 'You are a helpful assistant.',
        'mcp_server_path': None,  # Disable MCP for WebSocket-only tests
    }


@pytest.fixture
def provider(websocket_config):
    """Create a WebSocket provider instance."""
    return OpenAIWebSocketResponseProvider(websocket_config)


class TestWebSocketConnectionTimeout:
    """Tests for WebSocket connection timeout scenarios."""
    
    @pytest.mark.asyncio
    async def test_ws_preconnect_timeout_on_initialization(self, provider, capsys):
        """Test that initialization handles WebSocket pre-connect timeout gracefully."""
        # Mock the WebSocket connection to timeout
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            # Simulate connection timeout
            async def timeout_on_connect(*args, **kwargs):
                raise ServerTimeoutError("Connection timeout to host wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview")
            
            mock_session.ws_connect = AsyncMock(side_effect=timeout_on_connect)
            
            # Initialize should succeed despite WebSocket timeout
            result = await provider.initialize()
            
            # Initialization should still return True (non-fatal)
            assert result is True
            
            # Check for warning message in output
            captured = capsys.readouterr()
            assert "⚠️  WebSocket pre-connect failed" in captured.out
            assert "Connection timeout" in captured.out
    
    @pytest.mark.asyncio
    async def test_ws_connection_timeout_on_first_request(self, provider, capsys):
        """Test that first request handles WebSocket connection timeout."""
        # Skip initialization
        provider.openai_client = MagicMock()
        
        # Mock the WebSocket connection to timeout on first request
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            # Simulate connection timeout
            async def timeout_on_connect(*args, **kwargs):
                raise ServerTimeoutError("Connection timeout to host wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview")
            
            mock_session.ws_connect = AsyncMock(side_effect=timeout_on_connect)
            
            # Try to get a response (should fail with timeout)
            with pytest.raises(ServerTimeoutError) as exc_info:
                async for _ in provider.stream_response("Hello"):
                    pass
            
            assert "Connection timeout" in str(exc_info.value)
            
            # Check for error message in output
            captured = capsys.readouterr()
            assert "❌ WebSocket connection failed" in captured.out
    
    @pytest.mark.asyncio
    async def test_ws_connection_with_client_timeout_config(self, provider):
        """Test that WebSocket uses correct timeout configuration."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            # Create a mock WebSocket
            mock_ws = AsyncMock(spec=ClientWebSocketResponse)
            mock_ws.closed = False
            mock_session.ws_connect = AsyncMock(return_value=mock_ws)
            
            # Ensure connection
            await provider._ensure_ws_connected()
            
            # Verify ws_connect was called with proper timeout
            mock_session.ws_connect.assert_called_once()
            call_kwargs = mock_session.ws_connect.call_args[1]
            
            # Check timeout configuration
            assert 'timeout' in call_kwargs
            timeout = call_kwargs['timeout']
            assert isinstance(timeout, ClientTimeout)
            assert timeout.total == 30  # Should be 30 seconds
    
    @pytest.mark.asyncio
    async def test_ws_reconnect_after_timeout(self, provider, capsys):
        """Test that WebSocket reconnects after a timeout."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            # First call: timeout, second call: success
            timeout_error = ServerTimeoutError("Connection timeout")
            mock_ws = AsyncMock(spec=ClientWebSocketResponse)
            mock_ws.closed = False
            
            call_count = [0]
            async def connect_with_retry(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise timeout_error
                return mock_ws
            
            mock_session.ws_connect = AsyncMock(side_effect=connect_with_retry)
            
            # First attempt should fail
            with pytest.raises(ServerTimeoutError):
                await provider._ensure_ws_connected()
            
            # WebSocket should be marked as dead
            assert provider._ws is None
            
            # Second attempt should succeed
            ws = await provider._ensure_ws_connected()
            assert ws is mock_ws
            assert provider._ws is mock_ws
            
            # Verify ws_connect was called twice
            assert mock_session.ws_connect.call_count == 2


class TestWebSocketConnectionError:
    """Tests for WebSocket connection errors."""
    
    @pytest.mark.asyncio
    async def test_ws_connection_network_error(self, provider, capsys):
        """Test handling of network connection errors."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            # Simulate network error
            async def network_error(*args, **kwargs):
                raise ClientConnectorError(
                    connection_key=None,
                    os_error=OSError("Network unreachable")
                )
            
            mock_session.ws_connect = AsyncMock(side_effect=network_error)
            
            # Connection should fail with proper error
            with pytest.raises(ClientConnectorError):
                await provider._ensure_ws_connected()
            
            # Check for error message
            captured = capsys.readouterr()
            assert "❌ WebSocket connection failed" in captured.out
    
    @pytest.mark.asyncio
    async def test_ws_connection_invalid_api_key(self, provider, capsys):
        """Test handling of authentication errors (simulated by connection failure)."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            # Simulate auth error (OpenAI returns 401, manifests as connection error)
            async def auth_error(*args, **kwargs):
                raise ClientConnectorError(
                    connection_key=None,
                    os_error=OSError("401 Unauthorized")
                )
            
            mock_session.ws_connect = AsyncMock(side_effect=auth_error)
            
            # Connection should fail
            with pytest.raises(ClientConnectorError):
                await provider._ensure_ws_connected()
            
            # Check for error message
            captured = capsys.readouterr()
            assert "❌ WebSocket connection failed" in captured.out


class TestWebSocketPersistentConnection:
    """Tests for persistent WebSocket connection management."""
    
    @pytest.mark.asyncio
    async def test_ws_connection_reused(self, provider):
        """Test that WebSocket connection is reused across requests."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            # Create mock WebSocket
            mock_ws = AsyncMock(spec=ClientWebSocketResponse)
            mock_ws.closed = False
            mock_session.ws_connect = AsyncMock(return_value=mock_ws)
            
            # First connection
            ws1 = await provider._ensure_ws_connected()
            assert ws1 is mock_ws
            
            # Second call should reuse same connection
            ws2 = await provider._ensure_ws_connected()
            assert ws2 is ws1
            
            # ws_connect should only be called once
            assert mock_session.ws_connect.call_count == 1
    
    @pytest.mark.asyncio
    async def test_ws_reconnect_after_close(self, provider):
        """Test that WebSocket reconnects if connection is closed."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            # Create two mock WebSockets (first will be closed)
            mock_ws1 = AsyncMock(spec=ClientWebSocketResponse)
            mock_ws2 = AsyncMock(spec=ClientWebSocketResponse)
            mock_ws2.closed = False
            
            call_count = [0]
            async def connect_sequence(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return mock_ws1
                return mock_ws2
            
            mock_session.ws_connect = AsyncMock(side_effect=connect_sequence)
            
            # First connection
            ws1 = await provider._ensure_ws_connected()
            assert ws1 is mock_ws1
            
            # Simulate connection closure
            mock_ws1.closed = True
            
            # Should reconnect
            ws2 = await provider._ensure_ws_connected()
            assert ws2 is mock_ws2
            assert ws2 is not ws1
            
            # ws_connect should be called twice
            assert mock_session.ws_connect.call_count == 2
    
    @pytest.mark.asyncio
    async def test_ws_heartbeat_check(self, provider):
        """Test that heartbeat ping is sent after timeout period."""
        import time
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            # Create mock WebSocket
            mock_ws = AsyncMock(spec=ClientWebSocketResponse)
            mock_ws.closed = False
            mock_ws.ping = AsyncMock()
            mock_session.ws_connect = AsyncMock(return_value=mock_ws)
            
            # Connect
            await provider._ensure_ws_connected()
            
            # Set last heartbeat to >25 seconds ago
            provider._last_heartbeat = time.time() - 30
            
            # Next ensure_ws_connected should ping
            await provider._ensure_ws_connected()
            
            # Verify ping was called
            mock_ws.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ensure_ws_warm_success(self, provider):
        """Test ensure_ws_warm succeeds when connection is healthy."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            # Create mock WebSocket
            mock_ws = AsyncMock(spec=ClientWebSocketResponse)
            mock_ws.closed = False
            mock_session.ws_connect = AsyncMock(return_value=mock_ws)
            
            # Warm up connection
            result = await provider.ensure_ws_warm()
            
            assert result is True
            assert provider._ws is mock_ws
    
    @pytest.mark.asyncio
    async def test_ensure_ws_warm_failure(self, provider, capsys):
        """Test ensure_ws_warm handles connection failure gracefully."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            # Simulate timeout
            async def timeout_error(*args, **kwargs):
                raise ServerTimeoutError("Connection timeout")
            
            mock_session.ws_connect = AsyncMock(side_effect=timeout_error)
            
            # Warm-up should return False but not raise
            result = await provider.ensure_ws_warm()
            
            assert result is False
            
            # Check for warning message
            captured = capsys.readouterr()
            assert "⚠️  WebSocket warm-up failed" in captured.out


class TestWebSocketCleanup:
    """Tests for WebSocket cleanup and resource management."""
    
    @pytest.mark.asyncio
    async def test_cleanup_closes_websocket(self, provider):
        """Test that cleanup properly closes WebSocket connection."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            mock_session.close = AsyncMock()
            
            # Create mock WebSocket
            mock_ws = AsyncMock(spec=ClientWebSocketResponse)
            mock_ws.closed = False
            mock_ws.close = AsyncMock()
            mock_session.ws_connect = AsyncMock(return_value=mock_ws)
            
            # Connect
            await provider._ensure_ws_connected()
            assert provider._ws is mock_ws
            
            # Cleanup
            await provider.cleanup()
            
            # Verify close was called
            mock_ws.close.assert_called_once()
            mock_session.close.assert_called_once()
            
            # Verify state is cleared
            assert provider._ws is None
            assert provider._ws_session is None
    
    @pytest.mark.asyncio
    async def test_cleanup_handles_already_closed(self, provider):
        """Test that cleanup handles already-closed connections gracefully."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = True
            mock_session.close = AsyncMock()
            
            # Create mock WebSocket that's already closed
            mock_ws = AsyncMock(spec=ClientWebSocketResponse)
            mock_ws.closed = True
            mock_ws.close = AsyncMock()
            mock_session.ws_connect = AsyncMock(return_value=mock_ws)
            
            # Connect
            await provider._ensure_ws_connected()
            
            # Cleanup should succeed even if already closed
            await provider.cleanup()
            
            # close should not be called since already closed
            mock_ws.close.assert_not_called()
