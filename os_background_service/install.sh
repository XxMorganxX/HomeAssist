#!/bin/bash
# ==============================================
# Install HomeAssistV2 as a launchd service
# ==============================================

set -e

PROJECT_DIR="/Users/morgannstuart/Desktop/HomeAssistV2"
SERVICE_DIR="${PROJECT_DIR}/os_background_service"
PLIST_NAME="com.homeassistv2.assistant.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "🏠 HomeAssistV2 Service Installer"
echo "=================================="

# Create logs directory
mkdir -p "${PROJECT_DIR}/logs"
echo "✅ Created logs directory"

# Make runner script executable
chmod +x "${SERVICE_DIR}/run_assistant.sh"
echo "✅ Made runner script executable"

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCH_AGENTS_DIR"

# Copy plist to LaunchAgents
cp "${SERVICE_DIR}/${PLIST_NAME}" "${LAUNCH_AGENTS_DIR}/"
echo "✅ Copied plist to ${LAUNCH_AGENTS_DIR}"

# Unload if already loaded (ignore errors)
launchctl unload "${LAUNCH_AGENTS_DIR}/${PLIST_NAME}" 2>/dev/null || true

# Load the service
launchctl load "${LAUNCH_AGENTS_DIR}/${PLIST_NAME}"
echo "✅ Loaded service"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "The assistant will now:"
echo "  • Start automatically when you log in"
echo "  • Restart automatically if it crashes"
echo "  • Run in the background"
echo ""
echo "See README.md for useful commands."
echo ""
