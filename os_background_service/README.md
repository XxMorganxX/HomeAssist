# Background Service Setup

Run HomeAssistV2 as a macOS background service that starts automatically and runs discretely.

## Quick Start

### Install

```bash
cd os_background_service
./install.sh
```

The assistant will now start automatically on login and restart if it crashes.

### Uninstall

```bash
./uninstall.sh
```

## Useful Commands

### Check Status

```bash
launchctl list | grep homeassist
```

### View Logs

```bash
# Live stdout
tail -f ~/Desktop/HomeAssistV2/logs/assistant_stdout.log

# Live stderr
tail -f ~/Desktop/HomeAssistV2/logs/assistant_stderr.log
```

### Stop Service

```bash
launchctl unload ~/Library/LaunchAgents/com.homeassistv2.assistant.plist
```

### Start Service

```bash
launchctl load ~/Library/LaunchAgents/com.homeassistv2.assistant.plist
```

### Restart Service (after code changes)

```bash
launchctl kickstart -k gui/$(id -u)/com.homeassistv2.assistant
```

## How It Works

**launchd** is macOS's native service manager. This setup:

1. **`run_assistant.sh`** — Wrapper script that loads `.env` and runs the assistant
2. **`com.homeassistv2.assistant.plist`** — launchd configuration file
3. **`install.sh`** — Copies the plist to `~/Library/LaunchAgents` and loads it
4. **Logs** — Output written to `logs/assistant_stdout.log` and `logs/assistant_stderr.log`

The service will:
- Auto-start on login
- Auto-restart if it crashes (after 10 second throttle)
- Stop gracefully on logout
- Run in the background (no terminal window)

## Code Updates

**Yes, the service always runs the latest code** from your project directory. Python loads modules at runtime from `/Users/morgannstuart/Desktop/HomeAssistV2`.

However, you must **restart the service** after making code changes:

```bash
launchctl kickstart -k gui/$(id -u)/com.homeassistv2.assistant
```

This kills the current process and immediately starts a new one with your updated code.
