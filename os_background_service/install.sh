#!/bin/bash
# ==============================================
# Install HomeAssistV2 as a launchd service
# ==============================================
# This script installs both the main assistant service and
# the watchdog service, with optional power management configuration.

set -e

PROJECT_DIR="/Users/morgannstuart/Desktop/HomeAssistV2"
SERVICE_DIR="${PROJECT_DIR}/os_background_service"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

# Service files
MAIN_PLIST="com.homeassistv2.assistant.plist"
WATCHDOG_PLIST="com.homeassistv2.watchdog.plist"

echo "🏠 HomeAssistV2 Service Installer"
echo "=================================="
echo ""

# ==============================================
# Pre-flight checks
# ==============================================

# Check for .env file
if [ ! -f "${PROJECT_DIR}/.env" ]; then
    echo "⚠️  Warning: .env file not found at ${PROJECT_DIR}/.env"
    echo "   The assistant will fail to start without it."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for Python venv
if [ ! -f "${PROJECT_DIR}/venv/bin/python" ]; then
    echo "❌ Python venv not found at ${PROJECT_DIR}/venv"
    echo "   Please create the virtual environment first:"
    echo "   python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# ==============================================
# Setup directories and permissions
# ==============================================

echo "📁 Setting up directories..."

# Create logs directory
mkdir -p "${PROJECT_DIR}/logs"
echo "   ✓ Created logs directory"

# Make scripts executable
chmod +x "${SERVICE_DIR}/run_assistant.sh"
chmod +x "${SERVICE_DIR}/watchdog.sh"
chmod +x "${SERVICE_DIR}/configure_power.sh"
chmod +x "${SERVICE_DIR}/uninstall.sh"
echo "   ✓ Made scripts executable"

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCH_AGENTS_DIR"
echo "   ✓ Ensured LaunchAgents directory exists"

# ==============================================
# Install main assistant service
# ==============================================

echo ""
echo "🤖 Installing main assistant service..."

# Unload if already loaded (ignore errors)
launchctl unload "${LAUNCH_AGENTS_DIR}/${MAIN_PLIST}" 2>/dev/null || true

# Copy plist to LaunchAgents
cp "${SERVICE_DIR}/${MAIN_PLIST}" "${LAUNCH_AGENTS_DIR}/"
echo "   ✓ Copied ${MAIN_PLIST}"

# Load the service
launchctl load "${LAUNCH_AGENTS_DIR}/${MAIN_PLIST}"
echo "   ✓ Loaded main assistant service"

# ==============================================
# Install watchdog service
# ==============================================

echo ""
echo "🐕 Installing watchdog service..."

# Unload if already loaded (ignore errors)
launchctl unload "${LAUNCH_AGENTS_DIR}/${WATCHDOG_PLIST}" 2>/dev/null || true

# Copy plist to LaunchAgents
cp "${SERVICE_DIR}/${WATCHDOG_PLIST}" "${LAUNCH_AGENTS_DIR}/"
echo "   ✓ Copied ${WATCHDOG_PLIST}"

# Load the service
launchctl load "${LAUNCH_AGENTS_DIR}/${WATCHDOG_PLIST}"
echo "   ✓ Loaded watchdog service"

# ==============================================
# Power management configuration (optional)
# ==============================================

echo ""
echo "⚡ Power Management Configuration"
echo "---------------------------------"
echo "For best results running with the lid closed, you can configure"
echo "macOS power settings to prevent sleep."
echo ""
echo "This requires sudo and modifies system-wide power settings."
echo ""
read -p "Configure power settings for always-on operation? (y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running power configuration (will prompt for password)..."
    sudo "${SERVICE_DIR}/configure_power.sh" enable
else
    echo ""
    echo "ℹ️  Skipped power configuration."
    echo "   You can run it later with: sudo ${SERVICE_DIR}/configure_power.sh enable"
fi

# ==============================================
# Verify installation
# ==============================================

echo ""
echo "🔍 Verifying installation..."

sleep 2  # Give services time to start

MAIN_STATUS=$(launchctl list | grep "com.homeassistv2.assistant" || echo "NOT FOUND")
WATCHDOG_STATUS=$(launchctl list | grep "com.homeassistv2.watchdog" || echo "NOT FOUND")

echo "   Main service:     $MAIN_STATUS"
echo "   Watchdog service: $WATCHDOG_STATUS"

# ==============================================
# Summary
# ==============================================

echo ""
echo "=========================================="
echo "🎉 Installation complete!"
echo "=========================================="
echo ""
echo "The assistant will now:"
echo "  • Start automatically when you log in"
echo "  • Restart automatically if it crashes"
echo "  • Be monitored by the watchdog service"
echo "  • Prevent system sleep via caffeinate"
echo ""
echo "📋 Useful commands:"
echo ""
echo "  Check status:"
echo "    launchctl list | grep homeassist"
echo ""
echo "  View logs:"
echo "    tail -f ${PROJECT_DIR}/logs/assistant_stdout.log"
echo "    tail -f ${PROJECT_DIR}/logs/watchdog.log"
echo ""
echo "  Restart services:"
echo "    launchctl kickstart -k gui/\$(id -u)/com.homeassistv2.assistant"
echo ""
echo "  Configure power (if skipped):"
echo "    sudo ${SERVICE_DIR}/configure_power.sh enable"
echo ""
echo "  Uninstall:"
echo "    ${SERVICE_DIR}/uninstall.sh"
echo ""

# ==============================================
# Hardware recommendation
# ==============================================

echo "💡 For true headless operation with lid closed:"
echo "   Consider using an HDMI Headless Display Adapter (~\$10-15)"
echo "   This tricks macOS into thinking a display is connected,"
echo "   enabling reliable clamshell mode without software workarounds."
echo ""
