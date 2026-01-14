#!/bin/bash
# ==============================================
# Uninstall HomeAssist launchd services
# ==============================================
# This script removes both the main assistant service and
# the watchdog service, with optional power settings restoration.

set -e

PROJECT_DIR="$HOME/Desktop/HomeAssistV3"
SERVICE_DIR="${PROJECT_DIR}/os_background_service"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

# Service files
MAIN_PLIST="com.homeassist.assistant.plist"
WATCHDOG_PLIST="com.homeassist.watchdog.plist"
BLUETOOTH_PLIST="com.homeassist.bluetooth.plist"
TERMINAL_PLIST="com.homeassist.terminal.plist"
MCP_PLIST="com.homeassist.mcp.plist"

echo "🏠 HomeAssist Service Uninstaller"
echo "===================================="
echo ""

# ==============================================
# Stop and remove MCP server service
# ==============================================

echo "🔧 Removing MCP server service..."

if launchctl list | grep -q "com.homeassist.mcp"; then
    launchctl unload "${LAUNCH_AGENTS_DIR}/${MCP_PLIST}" 2>/dev/null || true
    echo "   ✓ Stopped MCP server service"
else
    echo "   ℹ️  MCP server service was not running"
fi

if [ -f "${LAUNCH_AGENTS_DIR}/${MCP_PLIST}" ]; then
    rm "${LAUNCH_AGENTS_DIR}/${MCP_PLIST}"
    echo "   ✓ Removed MCP server plist"
else
    echo "   ℹ️  MCP server plist not found"
fi

# Also kill any running MCP processes
pkill -f "mcp_server.server" 2>/dev/null || true

# ==============================================
# Stop and remove Bluetooth connector service
# ==============================================

echo ""
echo "🔵 Removing Bluetooth connector service..."

if launchctl list | grep -q "com.homeassist.bluetooth"; then
    launchctl unload "${LAUNCH_AGENTS_DIR}/${BLUETOOTH_PLIST}" 2>/dev/null || true
    echo "   ✓ Stopped Bluetooth connector service"
else
    echo "   ℹ️  Bluetooth connector service was not running"
fi

if [ -f "${LAUNCH_AGENTS_DIR}/${BLUETOOTH_PLIST}" ]; then
    rm "${LAUNCH_AGENTS_DIR}/${BLUETOOTH_PLIST}"
    echo "   ✓ Removed Bluetooth connector plist"
else
    echo "   ℹ️  Bluetooth connector plist not found"
fi

# ==============================================
# Stop and remove watchdog service
# ==============================================

echo ""
echo "🐕 Removing watchdog service..."

if launchctl list | grep -q "com.homeassist.watchdog"; then
    launchctl unload "${LAUNCH_AGENTS_DIR}/${WATCHDOG_PLIST}" 2>/dev/null || true
    echo "   ✓ Stopped watchdog service"
else
    echo "   ℹ️  Watchdog service was not running"
fi

if [ -f "${LAUNCH_AGENTS_DIR}/${WATCHDOG_PLIST}" ]; then
    rm "${LAUNCH_AGENTS_DIR}/${WATCHDOG_PLIST}"
    echo "   ✓ Removed watchdog plist"
else
    echo "   ℹ️  Watchdog plist not found"
fi

# ==============================================
# Stop and remove main assistant service
# ==============================================

echo ""
echo "🤖 Removing main assistant service..."

if launchctl list | grep -q "com.homeassist.assistant"; then
    launchctl unload "${LAUNCH_AGENTS_DIR}/${MAIN_PLIST}" 2>/dev/null || true
    echo "   ✓ Stopped main assistant service"
else
    echo "   ℹ️  Main assistant service was not running"
fi

if [ -f "${LAUNCH_AGENTS_DIR}/${MAIN_PLIST}" ]; then
    rm "${LAUNCH_AGENTS_DIR}/${MAIN_PLIST}"
    echo "   ✓ Removed main assistant plist"
else
    echo "   ℹ️  Main assistant plist not found"
fi

# ==============================================
# Stop and remove Terminal log viewer service
# ==============================================

echo ""
echo "🖥️  Removing Terminal log viewer..."

if launchctl list | grep -q "com.homeassist.terminal"; then
    launchctl unload "${LAUNCH_AGENTS_DIR}/${TERMINAL_PLIST}" 2>/dev/null || true
    echo "   ✓ Stopped Terminal log viewer"
else
    echo "   ℹ️  Terminal log viewer was not running"
fi

if [ -f "${LAUNCH_AGENTS_DIR}/${TERMINAL_PLIST}" ]; then
    rm "${LAUNCH_AGENTS_DIR}/${TERMINAL_PLIST}"
    echo "   ✓ Removed Terminal log viewer plist"
else
    echo "   ℹ️  Terminal log viewer plist not found"
fi

# ==============================================
# Kill any lingering caffeinate processes
# ==============================================

echo ""
echo "☕ Cleaning up caffeinate processes..."

# Find caffeinate processes started by our scripts
CAFF_PIDS=$(pgrep -f "caffeinate.*-i.*-s.*-d.*-m" 2>/dev/null || true)
if [ -n "$CAFF_PIDS" ]; then
    echo "$CAFF_PIDS" | xargs kill 2>/dev/null || true
    echo "   ✓ Killed lingering caffeinate processes"
else
    echo "   ℹ️  No caffeinate processes found"
fi

# ==============================================
# Power settings restoration (optional)
# ==============================================

echo ""
echo "⚡ Power Settings"
echo "-----------------"
echo "Would you like to restore default power settings?"
echo "(This is recommended if you used configure_power.sh enable)"
echo ""
read -p "Restore default power settings? (y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Restoring power settings (will prompt for password)..."
    if [ -f "${SERVICE_DIR}/configure_power.sh" ]; then
        sudo "${SERVICE_DIR}/configure_power.sh" disable
    else
        echo "   ⚠️  configure_power.sh not found, skipping"
    fi
else
    echo ""
    echo "ℹ️  Skipped power settings restoration."
    echo "   You can restore later with: sudo ${SERVICE_DIR}/configure_power.sh disable"
fi

# ==============================================
# Cleanup backup file
# ==============================================

BACKUP_FILE="$HOME/.homeassist_power_backup.txt"
if [ -f "$BACKUP_FILE" ]; then
    echo ""
    read -p "Remove power settings backup file? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "$BACKUP_FILE"
        echo "   ✓ Removed backup file"
    fi
fi

# ==============================================
# Summary
# ==============================================

echo ""
echo "=========================================="
echo "🎉 Uninstallation complete!"
echo "=========================================="
echo ""
echo "The assistant services have been removed."
echo "Your Mac will no longer run HomeAssist in the background."
echo ""
echo "Note: Log files were preserved at ${PROJECT_DIR}/logs/"
echo "      You can delete them manually if desired."
echo ""
echo "To reinstall, run: ${SERVICE_DIR}/install.sh"
echo ""
