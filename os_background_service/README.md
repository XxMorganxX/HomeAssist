# Background Service Setup

Run HomeAssist as a macOS background service with persistent operation, automatic recovery, and Bluetooth management.

## Quick Start

### Install

```bash
cd os_background_service
./install.sh
```

The installer will:
1. Create Python virtual environment (if needed)
2. Install all required dependencies
3. Install the main assistant service
4. Install the Bluetooth connector service
5. Install the persistent MCP server service
6. Install the Terminal log viewer (opens on login)
7. Create the `homeassist` command for easy control

### Control Commands

After installation, control HomeAssist from anywhere:

```bash
# Check if running
homeassist status

# Restart the assistant
homeassist restart

# View live logs
homeassist logs

# Stop all services
homeassist stop

# Start all services
homeassist start

# Run in foreground with Bluetooth auto-connect
homeassist run

# Manage persistent MCP server
homeassist mcp start    # Start MCP server in background
homeassist mcp stop     # Stop MCP server
homeassist mcp status   # Check MCP server status
homeassist mcp logs     # View MCP server logs
```

**Make command globally accessible:**
```bash
sudo ln -sf $HOME/Desktop/HomeAssistV3/homeassist /usr/local/bin/homeassist
```

Or use without sudo from project directory:
```bash
cd $HOME/Desktop/HomeAssistV3
./homeassist status
```

### Uninstall

```bash
./uninstall.sh
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          launchd                                │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │  Bluetooth Service   │    │  Persistent MCP Server       │  │
│  │  - Auto-reconnect    │    │  - Tool server (SSE)         │  │
│  │  - 2s poll interval  │    │  - Always running            │  │
│  │  - KeepAlive: true   │    │  - Port 3000                 │  │
│  └──────────────────────┘    └──────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Assistant Launcher (on boot)                            │  │
│  │  - Opens Terminal window                                 │  │
│  │  - Runs: homeassist run                                  │  │
│  │  - One-shot startup                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               homeassist run (in Terminal)                      │
│  - Waits for Bluetooth connection                               │
│  - Verifies Meta glasses are active audio I/O device            │
│  - Alternating WiFi/Bluetooth monitoring (2.5s intervals)       │
│  - Monitors for PaMacCore errors                                │
│  - Auto-restarts on disconnect                                  │
│  - Connects to persistent MCP server                            │
│  - Wake word Bluetooth verification                             │
└─────────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `homeassist` | Main control command (installed to `/usr/local/bin/`) |
| `configure_power.sh` | System power settings utility (optional) |
| `install.sh` | Installs all services |
| `uninstall.sh` | Removes all services |
| `show_logs.command` | Log viewer executable |
| `test_bluetooth_reconnect.sh` | Test Bluetooth reconnection |
| `test_bluetooth_simple.sh` | Simple Bluetooth test |
| `com.homeassist.assistant.plist` | Assistant launcher (opens Terminal with `homeassist run`) |
| `com.homeassist.bluetooth.plist` | Bluetooth connector config |
| `com.homeassist.mcp.plist` | Persistent MCP server config |
| `com.homeassist.terminal.plist` | Terminal log viewer config |

## Useful Commands

### Check Status

```bash
# See all services
launchctl list | grep homeassist

# Expected output when running:
# PID   Status  Label
# 1234  0       com.homeassist.bluetooth
# 5678  0       com.homeassist.mcp
# 9012  0       com.homeassist.terminal
```

### View Logs

```bash
# Bluetooth connector
tail -f $HOME/Desktop/HomeAssistV3/logs/bluetooth.log

# MCP server
tail -f $HOME/Desktop/HomeAssistV3/logs/mcp_server.log

# Or use the command
homeassist logs
```

### Restart Services

```bash
# Restart Bluetooth connector
launchctl kickstart -k gui/$(id -u)/com.homeassist.bluetooth

# Restart MCP server
homeassist mcp restart
```

### Stop Services Temporarily

```bash
# Stop Bluetooth connector
launchctl unload ~/Library/LaunchAgents/com.homeassist.bluetooth.plist

# Stop MCP server
homeassist mcp stop
```

### Start Services

```bash
# Start Bluetooth connector
launchctl load ~/Library/LaunchAgents/com.homeassist.bluetooth.plist

# Start MCP server
homeassist mcp start
```

## Bluetooth Management

### Aggressive Auto-Reconnection

The Bluetooth service (`com.homeassist.bluetooth.plist`) provides:
- **Ultra-aggressive polling**: Checks connection every 2 seconds when connected, 0.5 seconds when disconnected
- **Target reconnection time**: 3-5 seconds
- **Automatic power-on**: Ensures Bluetooth is enabled
- **Persistent connection**: Always tries to maintain connection to Meta Glasses

### Manual Bluetooth Testing

```bash
# Test reconnection (automated)
cd os_background_service
./test_bluetooth_reconnect.sh

# Simple connection test
./test_bluetooth_simple.sh
```

## Persistent MCP Server

The MCP (Model Context Protocol) server runs persistently in the background, providing:
- **Fast boot times**: ~0.04s connection vs ~2s spawn time
- **Tool availability**: Calendar, Gmail, Google Search, System Info, etc.
- **SSE transport**: Server-Sent Events on port 3000
- **Auto-restart**: Managed by launchd

The assistant connects to this persistent server instead of spawning it each time.

## Boot Behavior

When you log in to macOS:

1. **`com.homeassist.bluetooth.plist`** starts immediately, begins aggressive Bluetooth connection
2. **`com.homeassist.mcp.plist`** starts the persistent MCP server on port 3000
3. **`com.homeassist.terminal.plist`** opens a Terminal window with logs (optional)
4. **`com.homeassist.assistant.plist`** (after 5 second delay) opens a NEW Terminal window and runs `homeassist run`

The assistant Terminal window will show all Bluetooth connection attempts, boot progress, and runtime logs. You can close it anytime with Ctrl+C.

**Why this design?**
- ✅ Single source of truth: All logic lives in `homeassist run`
- ✅ No duplication: Updates to `homeassist run` automatically apply to boot service
- ✅ Visibility: Full console output for debugging
- ✅ Control: Easy to stop/restart from the Terminal

## Power Management

### Caffeinate (Automatic)

The `homeassist run` command automatically uses `caffeinate` with these flags:
- `-i` : Prevent idle sleep
- `-s` : Prevent system sleep (effective on AC power)
- `-d` : Prevent display sleep
- `-m` : Prevent disk sleep

### System Power Settings (Optional)

For more aggressive sleep prevention, configure system-wide settings:

```bash
# Enable always-on settings (requires sudo)
sudo ./configure_power.sh enable

# Check current settings
./configure_power.sh status

# Restore defaults
sudo ./configure_power.sh disable
```

**What `enable` does:**
- Disables sleep on AC power
- Sets 30-minute sleep timeout on battery
- Disables hibernation (RAM stays powered)
- Disables standby and auto power off
- Disables Power Nap

## Running with Lid Closed (Clamshell Mode)

### On AC Power

With the software configuration above, the assistant should run reliably with the lid closed when connected to external power.

### On Battery (Headless)

**This is the most challenging configuration.** macOS is designed to sleep when the lid closes without an external display, regardless of software settings. The system may still enforce sleep due to thermal management.

**Recommendations for battery operation:**
1. Keep the lid slightly open (even 1cm helps)
2. Use the software sleep prevention (caffeinate + pmset)
3. Accept that macOS may still force sleep after extended periods

### Guaranteed Headless Operation

For truly reliable headless operation with the lid closed, use a **Headless Display Adapter**:

- **What it is:** A small HDMI/DisplayPort dongle that emulates a display
- **Why it works:** macOS thinks a monitor is connected, enabling full clamshell mode
- **Cost:** ~$10-15 on Amazon
- **Search for:** "HDMI headless display adapter" or "HDMI dummy plug 4K"

This is the gold standard for headless Mac servers and the most reliable solution.

## Bluetooth Audio

For Bluetooth audio devices to work with the lid closed:

1. **Pair the device** before closing the lid
2. **Ensure sleep is prevented** (the assistant handles this via caffeinate)
3. **Connect power** for reliable operation
4. **Consider a headless adapter** for guaranteed uptime

Bluetooth connections persist through sleep on modern macOS, but audio processing requires the system to be awake.

## Troubleshooting

### Service won't start

```bash
# Check for errors in logs
cat $HOME/Desktop/HomeAssistV3/logs/bluetooth.log | tail -20

# Verify .env file exists
ls -la $HOME/Desktop/HomeAssistV3/.env

# Verify Python venv
$HOME/Desktop/HomeAssistV3/venv/bin/python --version
```

### Bluetooth disconnects frequently

```bash
# Check Bluetooth service log
tail -f $HOME/Desktop/HomeAssistV3/logs/bluetooth.log

# Verify blueutil is installed
/opt/homebrew/bin/blueutil --version

# Check if device is paired
/opt/homebrew/bin/blueutil --paired
```

### MCP server not responding

```bash
# Check MCP server status
homeassist mcp status

# View MCP server logs
homeassist mcp logs

# Restart MCP server
homeassist mcp restart
```

### System still sleeps

```bash
# Check current assertions
pmset -g assertions

# Verify caffeinate is running
pgrep -fl caffeinate

# Check power settings
pmset -g
```

### PaMacCore error not detected

The `homeassist run` command monitors for `PaMacCore.*err='-50'` errors which indicate audio device disconnection. If this isn't working:

```bash
# Run in foreground to see all output
homeassist run

# Check if error appears in output when glasses disconnect
```

## Code Updates

The service always runs the latest code from your project directory. After making code changes, restart:

```bash
# If using homeassist run (foreground)
# Just Ctrl+C and restart

# If using background services
launchctl kickstart -k gui/$(id -u)/com.homeassist.assistant
```

## Service Behavior Summary

| Scenario | Behavior |
|----------|----------|
| Login | Bluetooth & MCP services auto-start |
| Logout | Services stop gracefully |
| Crash | Auto-restart within 1-2 seconds |
| Bluetooth disconnect | Auto-reconnect within 2.5-5 seconds (alternating monitor) |
| WiFi disconnect | Detected within 2.5-5 seconds (alternating monitor) |
| Audio error (PaMacCore) | Assistant restarts, returns to BT search |
| Glasses off during boot | Waits for connection before starting |
| Code changes | Requires manual restart |

**Note on Monitoring**: The assistant uses an alternating monitor that checks WiFi and Bluetooth every 2.5 seconds (staggered). This provides faster detection than checking one every 5 seconds, with a maximum detection latency of 2.5 seconds for either type of disconnection.
