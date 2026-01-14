# Background Service Setup

Run HomeAssist as a macOS background service with persistent operation, automatic recovery, and sleep prevention.

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
5. Install the watchdog monitoring service
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

# Run in foreground (for testing)
homeassist run
```

**Make command globally accessible:**
```bash
sudo ln -sf ~/Desktop/HomeAssistV3/homeassist /usr/local/bin/homeassist
```

Or use without sudo from project directory:
```bash
cd ~/Desktop/HomeAssistV3
./homeassist status
```

### Uninstall

```bash
./uninstall.sh
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        launchd                              │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │  Main Assistant      │    │  Watchdog Service        │  │
│  │  - Runs assistant    │◄───│  - Monitors main service │  │
│  │  - caffeinate -isdm  │    │  - Restarts if needed    │  │
│  │  - KeepAlive: true   │    │  - Own caffeinate        │  │
│  └──────────────────────┘    └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Power Management                          │
│  - caffeinate prevents idle/system/display sleep            │
│  - pmset configures system-wide sleep behavior              │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `run_assistant.sh` | Main runner with caffeinate integration |
| `watchdog.sh` | Independent monitoring service |
| `configure_power.sh` | System power settings utility |
| `install.sh` | Installs all services |
| `uninstall.sh` | Removes all services |
| `com.homeassist.assistant.plist` | Main service launchd config |
| `com.homeassist.watchdog.plist` | Watchdog launchd config |

## Useful Commands

### Check Status

```bash
# See both services
launchctl list | grep homeassist

# Expected output when running:
# PID   Status  Label
# 1234  0       com.homeassist.assistant
# 5678  0       com.homeassist.watchdog
```

### View Logs

```bash
# Main assistant output
tail -f ~/Desktop/HomeAssistV3/logs/assistant_stdout.log

# Main assistant errors
tail -f ~/Desktop/HomeAssistV3/logs/assistant_stderr.log

# Watchdog activity
tail -f ~/Desktop/HomeAssistV3/logs/watchdog.log
```

### Restart Services

```bash
# Restart main assistant (recommended method)
launchctl kickstart -k gui/$(id -u)/com.homeassist.assistant

# Restart watchdog
launchctl kickstart -k gui/$(id -u)/com.homeassist.watchdog
```

### Stop Services Temporarily

```bash
# Stop main assistant
launchctl unload ~/Library/LaunchAgents/com.homeassist.assistant.plist

# Stop watchdog
launchctl unload ~/Library/LaunchAgents/com.homeassist.watchdog.plist
```

### Start Services

```bash
# Start main assistant
launchctl load ~/Library/LaunchAgents/com.homeassist.assistant.plist

# Start watchdog
launchctl load ~/Library/LaunchAgents/com.homeassist.watchdog.plist
```

## Power Management

### Caffeinate (Automatic)

The runner script automatically uses `caffeinate` with these flags:
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
# Check for errors
cat ~/Desktop/HomeAssistV3/logs/assistant_stderr.log | tail -20

# Verify .env file exists
ls -la ~/Desktop/HomeAssistV3/.env

# Verify Python venv
~/Desktop/HomeAssistV3/venv/bin/python --version
```

### Service keeps restarting

```bash
# Check exit codes
launchctl list | grep homeassist
# Non-zero status means the process crashed

# Look for Python errors
grep -i error ~/Desktop/HomeAssistV3/logs/assistant_stderr.log | tail -20
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

### Bluetooth disconnects

1. Ensure the system isn't sleeping (check caffeinate)
2. Keep the Mac on AC power
3. Consider a headless display adapter
4. Check System Preferences > Bluetooth for connection status

### Watchdog not restarting assistant

```bash
# Check watchdog log
cat ~/Desktop/HomeAssistV3/logs/watchdog.log | tail -30

# Manually trigger restart
launchctl kickstart -k gui/$(id -u)/com.homeassist.assistant
```

## Code Updates

The service always runs the latest code from your project directory. After making code changes, restart the service:

```bash
launchctl kickstart -k gui/$(id -u)/com.homeassist.assistant
```

## Service Behavior Summary

| Scenario | Behavior |
|----------|----------|
| Login | Services auto-start |
| Logout | Services stop gracefully |
| Crash | Auto-restart within 5 seconds |
| System sleep attempt | Blocked by caffeinate |
| Lid close (AC power) | Continues running |
| Lid close (battery) | May sleep after timeout |
| Code changes | Requires manual restart |
