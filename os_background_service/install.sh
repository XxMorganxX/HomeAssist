#!/bin/bash
# ==============================================
# Install HomeAssist as a launchd service
# ==============================================
# This script installs both the main assistant service and
# the watchdog service, with optional power management configuration.

set -e

PROJECT_DIR="$HOME/Desktop/HomeAssistV3"
SERVICE_DIR="${PROJECT_DIR}/os_background_service"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

# Service files
MAIN_PLIST="com.homeassist.assistant.plist"
BLUETOOTH_PLIST="com.homeassist.bluetooth.plist"
TERMINAL_PLIST="com.homeassist.terminal.plist"
MCP_PLIST="com.homeassist.mcp.plist"

echo "🏠 HomeAssist Service Installer"
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

# Check for Python venv - create if needed
if [ ! -f "${PROJECT_DIR}/venv/bin/python" ]; then
    echo "📦 Python venv not found, creating it..."
    
    # Check if python3 is available
    if ! command -v python3 &> /dev/null; then
        echo "❌ python3 not found. Please install Python 3 first."
        exit 1
    fi
    
    # Create venv
    cd "${PROJECT_DIR}"
    python3 -m venv venv
    
    # Install requirements
    if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
        echo "📥 Installing requirements..."
        "${PROJECT_DIR}/venv/bin/pip" install --upgrade pip
        "${PROJECT_DIR}/venv/bin/pip" install -r "${PROJECT_DIR}/requirements.txt"
        echo "   ✓ Requirements installed"
    else
        echo "⚠️  Warning: requirements.txt not found"
        echo "   You may need to install dependencies manually"
    fi
    
    echo "   ✓ Virtual environment created"
else
    echo "✓ Python venv exists"
fi

# ==============================================
# Setup directories and permissions
# ==============================================

echo "📁 Setting up directories..."

# Create logs directory
mkdir -p "${PROJECT_DIR}/logs"
echo "   ✓ Created logs directory"

# Make scripts executable
chmod +x "${SERVICE_DIR}/configure_power.sh"
chmod +x "${SERVICE_DIR}/uninstall.sh"
chmod +x "${SERVICE_DIR}/show_logs.command"
chmod +x "${SERVICE_DIR}/test_bluetooth_reconnect.sh"
chmod +x "${SERVICE_DIR}/test_bluetooth_simple.sh"
chmod +x "${PROJECT_DIR}/homeassist"
echo "   ✓ Made scripts executable"

# Create symlink for global command
if [ -d "/usr/local/bin" ]; then
    sudo ln -sf "${PROJECT_DIR}/homeassist" /usr/local/bin/homeassist 2>/dev/null || {
        echo "   ⚠️  Could not create /usr/local/bin/homeassist symlink (no sudo)"
        echo "      You can still run: ${PROJECT_DIR}/homeassist"
    }
    if [ -L "/usr/local/bin/homeassist" ]; then
        echo "   ✓ Created 'homeassist' command (accessible from anywhere)"
    fi
else
    echo "   ⚠️  /usr/local/bin not found, skipping global command setup"
fi

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCH_AGENTS_DIR"
echo "   ✓ Ensured LaunchAgents directory exists"

# ==============================================
# Install MCP server service (persistent tool server)
# ==============================================

echo ""
echo "🔧 Installing MCP server service (persistent tool server)..."

# Unload if already loaded (ignore errors)
launchctl unload "${LAUNCH_AGENTS_DIR}/${MCP_PLIST}" 2>/dev/null || true

# Copy plist to LaunchAgents
cp "${SERVICE_DIR}/${MCP_PLIST}" "${LAUNCH_AGENTS_DIR}/"
echo "   ✓ Copied ${MCP_PLIST}"

# Load the service
launchctl load "${LAUNCH_AGENTS_DIR}/${MCP_PLIST}"
echo "   ✓ Loaded MCP server service"
echo "   ✓ MCP server runs persistently for ~2s faster boot"

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
# Install Bluetooth connector service
# ==============================================

echo ""
echo "🔵 Installing Bluetooth connector service..."

# Check for blueutil dependency
if ! command -v blueutil &> /dev/null; then
    echo "   ⚠️  blueutil not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install blueutil
        echo "   ✓ blueutil installed"
    else
        echo "   ⚠️  Homebrew not found. Please install blueutil manually: brew install blueutil"
        echo "      Bluetooth service will attempt to install blueutil on first run."
    fi
else
    echo "   ✓ blueutil already installed"
fi

# Unload if already loaded (ignore errors)
launchctl unload "${LAUNCH_AGENTS_DIR}/${BLUETOOTH_PLIST}" 2>/dev/null || true

# Copy plist to LaunchAgents
cp "${SERVICE_DIR}/${BLUETOOTH_PLIST}" "${LAUNCH_AGENTS_DIR}/"
echo "   ✓ Copied ${BLUETOOTH_PLIST}"

# Load the service
launchctl load "${LAUNCH_AGENTS_DIR}/${BLUETOOTH_PLIST}"
echo "   ✓ Loaded Bluetooth connector service"

# ==============================================
# Install Terminal log viewer (opens on login)
# ==============================================

echo ""
echo "🖥️  Installing Terminal log viewer..."

# Unload if already loaded (ignore errors)
launchctl unload "${LAUNCH_AGENTS_DIR}/${TERMINAL_PLIST}" 2>/dev/null || true

# Copy plist to LaunchAgents
cp "${SERVICE_DIR}/${TERMINAL_PLIST}" "${LAUNCH_AGENTS_DIR}/"
echo "   ✓ Copied ${TERMINAL_PLIST}"

# Load the service
launchctl load "${LAUNCH_AGENTS_DIR}/${TERMINAL_PLIST}"
echo "   ✓ Loaded Terminal log viewer (opens on login)"

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

MCP_STATUS=$(launchctl list | grep "com.homeassist.mcp" || echo "NOT FOUND")
MAIN_STATUS=$(launchctl list | grep "com.homeassist.assistant" || echo "NOT FOUND")
BLUETOOTH_STATUS=$(launchctl list | grep "com.homeassist.bluetooth" || echo "NOT FOUND")
TERMINAL_STATUS=$(launchctl list | grep "com.homeassist.terminal" || echo "NOT FOUND")

echo "   MCP server:        $MCP_STATUS"
echo "   Main launcher:     $MAIN_STATUS"
echo "   Bluetooth service: $BLUETOOTH_STATUS"
echo "   Terminal viewer:   $TERMINAL_STATUS"

# ==============================================
# Summary
# ==============================================

echo ""
echo "=========================================="
echo "🎉 Installation complete!"
echo "=========================================="
echo ""
echo "The assistant will now:"
echo "  • Start automatically when you log in (Terminal window opens)"
echo "  • Run 'homeassist run' with full Bluetooth management"
echo "  • Restart automatically on Bluetooth disconnect or PaMacCore errors"
echo "  • Aggressively maintain Bluetooth connection to Meta Glasses"
echo "  • Use persistent MCP server for ~2s faster boot"
echo ""
echo "📋 Useful commands:"
echo ""
echo "  Quick command (from anywhere):"
echo "    homeassist status    - Check if running"
echo "    homeassist restart   - Restart assistant"
echo "    homeassist logs      - View live logs"
echo "    homeassist stop      - Stop all services"
echo "    homeassist start     - Start all services"
echo "    homeassist mcp status - Check MCP server"
echo ""
echo "  Or use launchctl directly:"
echo "    launchctl list | grep homeassist"
echo ""
echo "  View logs:"
echo "    homeassist logs      - Live log viewer"
echo "    tail -f ${PROJECT_DIR}/logs/bluetooth.log"
echo "    tail -f ${PROJECT_DIR}/logs/mcp_server.log"
echo ""
echo "  Restart services:"
echo "    homeassist restart"
echo "    launchctl kickstart -k gui/\$(id -u)/com.homeassist.assistant"
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
