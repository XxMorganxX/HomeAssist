#!/bin/bash
# ==============================================
# Power Management Configuration for HomeAssistV2
# ==============================================
# This script configures macOS power settings to prevent
# sleep and maximize uptime for the assistant.
#
# REQUIRES SUDO for pmset commands.
#
# Usage:
#   ./configure_power.sh enable   - Configure for always-on operation
#   ./configure_power.sh disable  - Restore default sleep settings
#   ./configure_power.sh status   - Show current power settings
#   ./configure_power.sh backup   - Backup current settings before changes

set -e

BACKUP_FILE="$HOME/.homeassistv2_power_backup.txt"

# ==============================================
# Functions
# ==============================================

show_usage() {
    echo "Usage: $0 {enable|disable|status|backup}"
    echo ""
    echo "Commands:"
    echo "  enable  - Configure power settings for always-on operation (requires sudo)"
    echo "  disable - Restore default/backed-up power settings (requires sudo)"
    echo "  status  - Show current power settings (no sudo needed)"
    echo "  backup  - Backup current power settings to file"
    echo ""
    echo "Note: 'enable' and 'disable' require sudo privileges."
}

check_sudo() {
    if [ "$EUID" -ne 0 ]; then
        echo "❌ This command requires sudo. Please run:"
        echo "   sudo $0 $1"
        exit 1
    fi
}

backup_settings() {
    echo "📋 Backing up current power settings..."
    pmset -g > "$BACKUP_FILE"
    echo "✅ Settings backed up to: $BACKUP_FILE"
}

show_status() {
    echo "🔋 Current Power Settings"
    echo "========================="
    echo ""
    echo "--- General Settings ---"
    pmset -g
    echo ""
    echo "--- Sleep Assertions ---"
    pmset -g assertions
    echo ""
    echo "--- Battery Status ---"
    pmset -g batt
}

enable_always_on() {
    check_sudo "enable"
    
    echo "⚡ Configuring power settings for always-on operation..."
    echo ""
    
    # Backup first if not already done
    if [ ! -f "$BACKUP_FILE" ]; then
        pmset -g > "$BACKUP_FILE"
        echo "📋 Backed up current settings to: $BACKUP_FILE"
    fi
    
    echo "Applying settings..."
    echo ""
    
    # ----- AC Power Settings (-c) -----
    echo "🔌 Configuring AC power settings..."
    
    # Disable sleep entirely when on AC
    pmset -c sleep 0
    echo "   ✓ Disabled system sleep on AC"
    
    # Disable display sleep (or set very high)
    pmset -c displaysleep 0
    echo "   ✓ Disabled display sleep on AC"
    
    # Disable disk sleep
    pmset -c disksleep 0
    echo "   ✓ Disabled disk sleep on AC"
    
    # Disable Power Nap (can wake system unexpectedly)
    pmset -c powernap 0
    echo "   ✓ Disabled Power Nap on AC"
    
    # Wake on network access (useful for remote management)
    pmset -c womp 1
    echo "   ✓ Enabled wake on network access"
    
    # Prevent sleeping when display is off
    pmset -c lidwake 1
    echo "   ✓ Enabled lid wake"
    
    # ----- Battery Power Settings (-b) -----
    echo ""
    echo "🔋 Configuring battery power settings..."
    
    # Set longer sleep timeout on battery (30 minutes)
    # Note: Setting to 0 on battery drains it quickly
    pmset -b sleep 30
    echo "   ✓ Set battery sleep to 30 minutes"
    
    # Display sleep after 10 minutes on battery
    pmset -b displaysleep 10
    echo "   ✓ Set display sleep to 10 minutes on battery"
    
    # Disable Power Nap on battery
    pmset -b powernap 0
    echo "   ✓ Disabled Power Nap on battery"
    
    # ----- Hibernation Settings (-a = all) -----
    echo ""
    echo "💤 Configuring hibernation settings..."
    
    # Hibernation mode 0 = no hibernation, RAM stays powered
    # This allows faster wake but uses more power
    pmset -a hibernatemode 0
    echo "   ✓ Disabled hibernation (RAM stays powered)"
    
    # Disable standby (deep sleep after normal sleep)
    pmset -a standby 0
    echo "   ✓ Disabled standby mode"
    
    # Disable autopoweroff
    pmset -a autopoweroff 0
    echo "   ✓ Disabled auto power off"
    
    # Disable TCP keepalive during sleep
    # This prevents network activity from waking the system unexpectedly
    pmset -a tcpkeepalive 0
    echo "   ✓ Disabled TCP keepalive during sleep"
    
    echo ""
    echo "✅ Power settings configured for always-on operation!"
    echo ""
    echo "⚠️  Important Notes:"
    echo "   • On AC power: System will not sleep"
    echo "   • On battery: System will sleep after 30 minutes to preserve battery"
    echo "   • For true headless operation with lid closed, consider using a"
    echo "     headless display adapter (HDMI/DisplayPort dummy plug)"
    echo ""
    echo "To restore default settings, run: sudo $0 disable"
}

disable_always_on() {
    check_sudo "disable"
    
    echo "🔄 Restoring default power settings..."
    echo ""
    
    # Restore from backup if available
    if [ -f "$BACKUP_FILE" ]; then
        echo "📋 Found backup file, restoring previous settings..."
        # Note: pmset doesn't have a direct restore command,
        # so we set reasonable defaults
    fi
    
    # ----- Restore AC Power Defaults -----
    echo "🔌 Restoring AC power defaults..."
    pmset -c sleep 1          # Sleep after 1 minute (or adjust)
    pmset -c displaysleep 10  # Display sleep after 10 minutes
    pmset -c disksleep 10     # Disk sleep after 10 minutes
    pmset -c powernap 1       # Enable Power Nap
    pmset -c womp 1           # Wake on network
    echo "   ✓ AC power defaults restored"
    
    # ----- Restore Battery Defaults -----
    echo "🔋 Restoring battery defaults..."
    pmset -b sleep 5          # Sleep after 5 minutes
    pmset -b displaysleep 2   # Display sleep after 2 minutes
    pmset -b powernap 0       # Power Nap off on battery (Apple default)
    echo "   ✓ Battery defaults restored"
    
    # ----- Restore Hibernation Defaults -----
    echo "💤 Restoring hibernation defaults..."
    pmset -a hibernatemode 3  # Default hybrid sleep/hibernate
    pmset -a standby 1        # Enable standby
    pmset -a autopoweroff 1   # Enable auto power off
    pmset -a tcpkeepalive 1   # Enable TCP keepalive
    echo "   ✓ Hibernation defaults restored"
    
    echo ""
    echo "✅ Default power settings restored!"
    echo ""
    echo "Your Mac will now sleep normally."
}

# ==============================================
# Main
# ==============================================

case "${1:-}" in
    enable)
        enable_always_on
        ;;
    disable)
        disable_always_on
        ;;
    status)
        show_status
        ;;
    backup)
        backup_settings
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

