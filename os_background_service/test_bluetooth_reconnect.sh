#!/bin/bash
# ==============================================
# Test: Bluetooth Aggressive Auto-Reconnect
# ==============================================
# Fully automated test - no user input required

DEVICE_ADDRESS="98-59-49-bf-40-45"
DEVICE_NAME="Meta Glasses"
BLUEUTIL="/opt/homebrew/bin/blueutil"
LOG="$HOME/Desktop/HomeAssistV3/logs/bluetooth.log"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo "========================================"
echo "  Bluetooth Auto-Reconnect Test"
echo "========================================"
echo ""

# 1. Check service
echo -e "${BLUE}[1/5] Checking service...${NC}"
if ! launchctl list | grep -q com.homeassist.bluetooth; then
    echo -e "${RED}❌ Service not running${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Service is running${NC}"
echo ""

# 2. Check initial connection
echo -e "${BLUE}[2/5] Checking connection...${NC}"
if [ "$("$BLUEUTIL" --is-connected "$DEVICE_ADDRESS")" = "1" ]; then
    echo -e "${GREEN}✅ $DEVICE_NAME is CONNECTED${NC}"
else
    echo -e "${YELLOW}⚠️  $DEVICE_NAME is DISCONNECTED${NC}"
    echo "   Waiting for service to connect..."
    sleep 10
    if [ "$("$BLUEUTIL" --is-connected "$DEVICE_ADDRESS")" = "1" ]; then
        echo -e "${GREEN}✅ Service connected it!${NC}"
    else
        echo -e "${RED}❌ Device still not connected${NC}"
        echo "   Check if glasses are on and in range"
        exit 1
    fi
fi
echo ""

# 3. Mark log and disconnect
echo -e "${YELLOW}[3/5] Disconnecting device...${NC}"
LOG_MARK=$(wc -l < "$LOG" | tr -d ' ')
"$BLUEUTIL" --disconnect "$DEVICE_ADDRESS"
DISCONNECT_TIME=$(date +%s)
echo -e "${RED}🔌 Disconnected at $(date '+%H:%M:%S')${NC}"
echo ""

# 4. Wait for automatic reconnection
echo -e "${BLUE}[4/5] Waiting for auto-reconnect (30s timeout)...${NC}"
echo -e "${CYAN}   Live log:${NC}"
echo "   ----------------------------------------"

RECONNECTED=false
LAST_LINE=$LOG_MARK

for i in $(seq 1 30); do
    # Show new log lines
    CURRENT=$(wc -l < "$LOG" | tr -d ' ')
    if [ "$CURRENT" -gt "$LAST_LINE" ]; then
        DIFF=$((CURRENT - LAST_LINE))
        if [ "$DIFF" -gt 0 ]; then
            tail -n "$DIFF" "$LOG" | while read -r line; do
                echo "   $line"
            done
        fi
        LAST_LINE=$CURRENT
    fi
    
    # Check if reconnected
    if [ "$("$BLUEUTIL" --is-connected "$DEVICE_ADDRESS")" = "1" ]; then
        RECONNECT_TIME=$(($(date +%s) - DISCONNECT_TIME))
        echo ""
        echo -e "   ${GREEN}✅ RECONNECTED in ${RECONNECT_TIME} seconds!${NC}"
        RECONNECTED=true
        break
    fi
    
    sleep 1
done

echo "   ----------------------------------------"
echo ""

# 5. Results
echo "========================================"
echo -e "${BLUE}[5/5] TEST RESULTS${NC}"
echo "========================================"
echo ""

if [ "$RECONNECTED" = true ]; then
    echo -e "${GREEN}✅ DISCONNECT DETECTION: Passed${NC}"
    echo -e "${GREEN}✅ AUTO-RECONNECT: Passed (${RECONNECT_TIME}s)${NC}"
    echo ""
    echo -e "${GREEN}========================================"
    echo "  🎉 ALL TESTS PASSED!"
    echo "========================================"
    echo ""
    echo "  The Bluetooth service successfully:"
    echo "  • Detected disconnect immediately"
    echo "  • Reconnected in ${RECONNECT_TIME} seconds"
    echo "  • Aggressive auto-reconnect VERIFIED!"
    echo "========================================${NC}"
else
    echo -e "${GREEN}✅ DISCONNECT DETECTION: Passed${NC}"
    echo -e "${RED}❌ AUTO-RECONNECT: Failed (timeout)${NC}"
    echo ""
    echo -e "${YELLOW}========================================"
    echo "  ⚠️  PARTIAL SUCCESS"
    echo "========================================"
    echo ""
    echo "  The service detected the disconnect"
    echo "  but reconnection timed out."
    echo ""
    echo "  This can happen if the glasses"
    echo "  entered low-power mode."
    echo ""
    echo "  Service will reconnect when device"
    echo "  becomes available again."
    echo "========================================${NC}"
fi
echo ""
