# MCP Server with Terminal Client - Minimal Setup

This is a minimal setup for running the MCP (Model Context Protocol) server with a terminal client interface.

## Quick Start

### 1. Install Minimal Dependencies
```bash
pip install -r requirements_mcp_minimal.txt
```

### 2. Set Environment Variables
Create a `.env` file in the project root:
```
OPENAI_KEY=your-openai-api-key-here
MORGAN_SPOTIFY_CLIENT_ID=your-spotify-client-id
MORGAN_SPOTIFY_CLIENT_SECRET=your-spotify-client-secret
SPENCER_SPOTIFY_CLIENT_ID=your-spotify-client-id
SPENCER_SPOTIFY_CLIENT_SECRET=your-spotify-client-secret
```

### 3. Run the MCP Server
```bash
python mcp_standalone.py
```

## Available Tools

The MCP server includes these smart home tools:

- **batch_light_control**: Control smart lights by name or room
- **lighting_scene**: Set predefined lighting scenes (movie, party, mood, etc.)
- **spotify_playback**: Control Spotify playback (play, pause, next, volume, etc.)
- **calendar_data**: Access calendar information
- **state_manager**: Manage system state

## Usage Examples

Once the server is running, you can interact with it via text commands:

```
💬 You: turn on the living room lights
🤖 Assistant: I'll turn on the living room lights for you.

💬 You: play some music on spotify
🤖 Assistant: I'll start playing music on Spotify for you.

💬 You: set the lighting scene to movie
🤖 Assistant: I'll set the lighting scene to movie mode.

💬 You: pause the music
🤖 Assistant: I'll pause the music for you.
```

## Special Commands

- `tools` - Show available tools
- `status` - Show server status
- `quit`, `exit`, or `q` - Exit the terminal client
- `Ctrl+C` - Force exit

## Essential Files

The minimal setup requires these files:

```
├── mcp_standalone.py              # Standalone MCP server script
├── requirements_mcp_minimal.txt    # Minimal dependencies
├── .env                           # Environment variables
├── config.py                      # Configuration settings
├── mcp_server/                    # MCP server implementation
│   ├── server.py
│   ├── tool_registry.py
│   ├── base_tool.py
│   └── tools/
│       ├── batch_light_control.py
│       ├── lighting_scene.py
│       ├── spotify_playback.py
│       ├── calendar_data.py
│       └── state_tool.py
└── core/
    └── streaming_chatbot.py       # Tool-enabled chatbot
```

## Troubleshooting

1. **Import errors**: Make sure you're running from the project root directory
2. **Missing API keys**: Check your `.env` file has the required API keys
3. **Tool failures**: Some tools require specific hardware (smart lights, Spotify account)
4. **Dependencies**: Install the minimal requirements with `pip install -r requirements_mcp_minimal.txt` 