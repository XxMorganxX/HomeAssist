#!/bin/bash
# ==============================================
# Uninstall HomeAssistV2 launchd service
# ==============================================

PLIST_NAME="com.homeassistv2.assistant.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "🏠 HomeAssistV2 Service Uninstaller"
echo "===================================="

# Unload the service
if launchctl list | grep -q "com.homeassistv2.assistant"; then
    launchctl unload "${LAUNCH_AGENTS_DIR}/${PLIST_NAME}" 2>/dev/null || true
    echo "✅ Stopped service"
else
    echo "ℹ️  Service was not running"
fi

# Remove plist
if [ -f "${LAUNCH_AGENTS_DIR}/${PLIST_NAME}" ]; then
    rm "${LAUNCH_AGENTS_DIR}/${PLIST_NAME}"
    echo "✅ Removed plist file"
else
    echo "ℹ️  Plist file not found"
fi

echo ""
echo "🎉 Uninstallation complete!"
echo "The assistant will no longer run in the background."
echo ""
