"""
Audio device management utilities using sounddevice.
"""

import sounddevice as sd


def get_emeet_device():
    """Find EMEET audio device index."""
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            # Check if device has input channels and name contains "emeet"
            if device.get('max_input_channels', 0) > 0 and "emeet" in device.get('name', '').lower():
                print(f"Found EMEET device at index #{i}: {device['name']}")
                return i
        return None
    except Exception as e:
        try:
            print(f"Error enumerating audio devices: {e}")
        except Exception:
            pass
        return None


def list_audio_devices():
    """List all available audio devices."""
    try:
        print("\n" + "=" * 60)
        print("Available Audio Devices:")
        print("=" * 60)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            device_type = []
            if device.get('max_input_channels', 0) > 0:
                device_type.append("INPUT")
            if device.get('max_output_channels', 0) > 0:
                device_type.append("OUTPUT")
            
            type_str = "/".join(device_type) if device_type else "NONE"
            default_marker = " [DEFAULT]" if i == sd.default.device[0] else ""
            
            print(f"{i:2d}. [{type_str:12}] {device['name']}{default_marker}")
            print(f"     Sample Rate: {device.get('default_samplerate', 'N/A')} Hz")
            print(f"     Channels: In={device.get('max_input_channels', 0)}, Out={device.get('max_output_channels', 0)}")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"Error listing audio devices: {e}")


def get_default_input_device():
    """Get the default input device index."""
    try:
        return sd.default.device[0]  # Input device is first element
    except Exception as e:
        print(f"Error getting default input device: {e}")
        return None


def get_device_info(device_index):
    """Get detailed information about a specific device."""
    try:
        device = sd.query_devices(device_index)
        return device
    except Exception as e:
        print(f"Error getting device info for index {device_index}: {e}")
        return None
