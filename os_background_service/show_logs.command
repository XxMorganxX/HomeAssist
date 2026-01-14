#!/bin/bash
# ==============================================
# HomeAssist Log Viewer
# Double-click this file to open logs in Terminal
# ==============================================

PROJECT_DIR="$HOME/Desktop/HomeAssistV3"
RUNNER_LOG="${PROJECT_DIR}/logs/assistant_runner.log"
OUTPUT_LOG="${PROJECT_DIR}/logs/assistant_output.log"

clear
echo "🏠 HomeAssist Live Log Viewer"
echo "=============================="
echo ""
echo "Service status:"
launchctl list | grep homeassist
echo ""
echo "Press Ctrl+C to stop viewing logs"
echo "-------------------------------------------"
echo ""

# Show both logs combined with color coding
tail -f "$RUNNER_LOG" "$OUTPUT_LOG" 2>/dev/null | while read -r line; do
    if [[ "$line" =~ "==>" ]]; then
        echo -e "\033[0;36m$line\033[0m"  # Cyan for file headers
    elif [[ "$line" =~ "Error" ]] || [[ "$line" =~ "error" ]] || [[ "$line" =~ "⚠️" ]]; then
        echo -e "\033[0;31m$line\033[0m"  # Red for errors
    elif [[ "$line" =~ "✅" ]] || [[ "$line" =~ "Starting" ]]; then
        echo -e "\033[0;32m$line\033[0m"  # Green for success
    elif [[ "$line" =~ "⏳" ]] || [[ "$line" =~ "Waiting" ]]; then
        echo -e "\033[0;33m$line\033[0m"  # Yellow for waiting
    else
        echo "$line"
    fi
done
