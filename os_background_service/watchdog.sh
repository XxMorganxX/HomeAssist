#!/bin/bash
# ==============================================
# HomeAssist Watchdog Service
# ==============================================
# Independent monitoring service that ensures the main assistant
# stays running. Provides an additional layer of resilience.
#
# Features:
# - Monitors main assistant process
# - Restarts assistant if unresponsive
# - Maintains its own caffeinate assertion
# - Logs watchdog activity

set -e

PROJECT_DIR="/Users/morgannstuart/Desktop/HomeAssistV3"
LOG_DIR="${PROJECT_DIR}/logs"
WATCHDOG_LOG="${LOG_DIR}/watchdog.log"
SERVICE_LABEL="com.homeassist.assistant"
CHECK_INTERVAL=30  # seconds between checks

# PID for our caffeinate process
CAFFEINE_PID=""

# ==============================================
# Logging
# ==============================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$WATCHDOG_LOG"
}

# ==============================================
# Cleanup
# ==============================================
cleanup() {
    log "Watchdog shutting down..."
    if [ -n "$CAFFEINE_PID" ] && kill -0 "$CAFFEINE_PID" 2>/dev/null; then
        kill "$CAFFEINE_PID" 2>/dev/null || true
        log "Stopped caffeinate (PID: $CAFFEINE_PID)"
    fi
    exit 0
}

trap cleanup SIGTERM SIGINT SIGHUP

# ==============================================
# Service Management
# ==============================================

is_service_running() {
    # Check if the main assistant service is running
    if launchctl list | grep -q "$SERVICE_LABEL"; then
        # Get the PID from launchctl
        local pid=$(launchctl list | grep "$SERVICE_LABEL" | awk '{print $1}')
        if [ "$pid" != "-" ] && [ -n "$pid" ]; then
            return 0  # Running
        fi
    fi
    return 1  # Not running
}

restart_service() {
    log "⚠️  Attempting to restart assistant service..."
    
    # Try to kickstart the service (restart if running, start if not)
    if launchctl kickstart -k "gui/$(id -u)/$SERVICE_LABEL" 2>/dev/null; then
        log "✅ Service kickstarted successfully"
        return 0
    fi
    
    # Fallback: try load if kickstart fails
    local plist="$HOME/Library/LaunchAgents/${SERVICE_LABEL}.plist"
    if [ -f "$plist" ]; then
        launchctl unload "$plist" 2>/dev/null || true
        sleep 1
        if launchctl load "$plist" 2>/dev/null; then
            log "✅ Service loaded successfully via fallback"
            return 0
        fi
    fi
    
    log "❌ Failed to restart service"
    return 1
}

check_process_health() {
    # Get the assistant's PID
    local pid=$(launchctl list | grep "$SERVICE_LABEL" | awk '{print $1}')
    
    if [ "$pid" = "-" ] || [ -z "$pid" ]; then
        return 1  # No PID means not healthy
    fi
    
    # Check if process exists and is not a zombie
    if ps -p "$pid" -o state= 2>/dev/null | grep -q "Z"; then
        log "⚠️  Assistant process is a zombie"
        return 1
    fi
    
    # Process exists and is healthy
    return 0
}

# ==============================================
# Main Watchdog Loop
# ==============================================

main() {
    mkdir -p "$LOG_DIR"
    
    log "=========================================="
    log "HomeAssist Watchdog Starting"
    log "=========================================="
    log "Monitoring service: $SERVICE_LABEL"
    log "Check interval: ${CHECK_INTERVAL}s"
    
    # Start caffeinate for watchdog's own sleep prevention
    log "Starting caffeinate for watchdog..."
    caffeinate -i -s -d -m &
    CAFFEINE_PID=$!
    log "Caffeinate started (PID: $CAFFEINE_PID)"
    
    local consecutive_failures=0
    local max_failures=3
    
    while true; do
        sleep "$CHECK_INTERVAL"
        
        # Check if main service is running
        if is_service_running; then
            # Service is registered, check process health
            if check_process_health; then
                # All good
                consecutive_failures=0
            else
                # Process unhealthy
                log "⚠️  Service registered but process unhealthy"
                ((consecutive_failures++))
            fi
        else
            log "⚠️  Main assistant service not running"
            ((consecutive_failures++))
        fi
        
        # Attempt restart if we've had failures
        if [ $consecutive_failures -ge 1 ]; then
            log "Failures: $consecutive_failures / $max_failures before forced restart"
            
            if [ $consecutive_failures -ge $max_failures ]; then
                log "🔄 Max failures reached, forcing restart..."
                restart_service
                consecutive_failures=0
                sleep 5  # Give service time to start
            fi
        fi
        
        # Verify our caffeinate is still running
        if [ -n "$CAFFEINE_PID" ] && ! kill -0 "$CAFFEINE_PID" 2>/dev/null; then
            log "⚠️  Caffeinate died, restarting..."
            caffeinate -i -s -d -m &
            CAFFEINE_PID=$!
            log "Caffeinate restarted (PID: $CAFFEINE_PID)"
        fi
    done
}

# ==============================================
# Entry Point
# ==============================================
main "$@"

