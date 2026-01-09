"""
Audio device management utilities using sounddevice.

Handles detection and configuration for various audio devices:
- Meta Ray-Ban glasses (Bluetooth HFP)
- EMEET conference speakers
- Default system devices
"""

import sounddevice as sd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class AudioDeviceConfig:
    """Configuration for an audio device."""
    device_index: Optional[int]
    device_name: str
    sample_rate: int
    channels: int
    dtype: str
    blocksize: int  # Frames per callback (blocksize in sd.InputStream)
    latency: str    # 'low', 'high', or float seconds
    is_bluetooth: bool
    
    def get_stream_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for sd.InputStream creation."""
        return {
            'device': self.device_index,
            'samplerate': self.sample_rate,
            'channels': self.channels,
            'dtype': self.dtype,
            'blocksize': self.blocksize,
            'latency': self.latency,
        }


# ============================================================================
# Meta Ray-Ban Glasses (Bluetooth HFP)
# ============================================================================
# Hardware: 5 microphones (spatial capture), records locally at high quality
# Via Bluetooth: Limited by HFP (Hands-Free Profile)
#   - Sample Rate: 16kHz (sometimes 8kHz)
#   - Channels: Mono
#   - Bit Depth: 16-bit
#   - Codec: SCO/mSBC
# 
# Bluetooth audio arrives in bursts (packet-based), so we need:
#   - Moderate blocksize (1024 = 64ms at 16kHz is good balance)
#   - High latency (larger internal buffers to absorb bursts)
# ============================================================================

def detect_meta_raybans() -> Optional[Tuple[int, dict]]:
    """
    Detect if Meta Ray-Bans are connected.
    
    Returns:
        Tuple of (device_index, device_info) if found, None otherwise
    """
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            name = device.get('name', '').lower()
            # Ray-Bans can appear as various names:
            # - "Ray-Ban Stories"
            # - "Morgan's Meta BANS"
            # - "Ray-Ban Meta"
            # - etc.
            is_raybans = (
                device.get('max_input_channels', 0) > 0 and
                ('ray-ban' in name or 'raybans' in name or 'ray ban' in name or 'meta bans' in name)
            )
            if is_raybans:
                print(f"🕶️  Meta Ray-Bans detected at index #{i}: {device['name']}")
                print(f"   Native sample rate: {device.get('default_samplerate', 'N/A')} Hz")
                print(f"   Max input channels: {device.get('max_input_channels', 0)}")
                return (i, device)
        return None
    except Exception as e:
        print(f"Error detecting Ray-Bans: {e}")
        return None


def get_raybans_config(device_index: int, device_info: dict) -> AudioDeviceConfig:
    """
    Get optimized audio config for Meta Ray-Ban glasses.
    
    Bluetooth HFP characteristics:
    - 16kHz sample rate (mono, 16-bit)
    - Audio arrives in bursts due to packet-based BT transmission
    - mSBC codec delivers ~7.5ms packets (~120 samples each)
    """
    return AudioDeviceConfig(
        device_index=device_index,
        device_name=device_info.get('name', 'Ray-Ban'),
        sample_rate=16000,       # Bluetooth HFP native rate
        channels=1,              # Mono (BT mics are mono)
        dtype='int16',           # 16-bit
        blocksize=1024,          # 64ms at 16kHz - good balance for BT
        latency='high',          # Larger internal buffers for bursty BT audio
        is_bluetooth=True
    )


# ============================================================================
# EMEET Conference Speaker
# ============================================================================

def get_emeet_device() -> Optional[int]:
    """Find EMEET audio device index."""
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            # Check if device has input channels and name contains "emeet"
            if device.get('max_input_channels', 0) > 0 and "emeet" in device.get('name', '').lower():
                print(f"🎤 Found EMEET device at index #{i}: {device['name']}")
                return i
        return None
    except Exception as e:
        try:
            print(f"Error enumerating audio devices: {e}")
        except Exception:
            pass
        return None


def get_emeet_config(device_index: int) -> AudioDeviceConfig:
    """Get audio config for EMEET device."""
    device_info = sd.query_devices(device_index)
    return AudioDeviceConfig(
        device_index=device_index,
        device_name=device_info.get('name', 'EMEET'),
        sample_rate=16000,
        channels=1,
        dtype='int16',
        blocksize=3200,          # 200ms at 16kHz (USB is reliable)
        latency='low',           # USB is stable, use low latency
        is_bluetooth=False
    )


# ============================================================================
# Default Device Config
# ============================================================================

def get_default_config() -> AudioDeviceConfig:
    """Get default audio config when no specific device detected."""
    return AudioDeviceConfig(
        device_index=None,       # Use system default
        device_name="default",
        sample_rate=16000,
        channels=1,
        dtype='int16',
        blocksize=3200,          # 200ms at 16kHz
        latency='low',
        is_bluetooth=False
    )


# ============================================================================
# Main Detection Function
# ============================================================================

def get_audio_device_config() -> AudioDeviceConfig:
    """
    Detect connected audio device and return optimized configuration.
    
    Checks for (in order of priority):
    1. Meta Ray-Bans (Bluetooth - needs special handling)
    2. EMEET device (USB conference speaker)
    3. Default system device
    
    Returns:
        AudioDeviceConfig with optimized settings for detected device
    """
    # Check for Ray-Bans first (Bluetooth needs special handling)
    raybans = detect_meta_raybans()
    if raybans:
        device_index, device_info = raybans
        config = get_raybans_config(device_index, device_info)
        print(f"   Using: blocksize={config.blocksize}, latency='{config.latency}'")
        return config
    
    # Check for EMEET
    emeet_index = get_emeet_device()
    if emeet_index is not None:
        return get_emeet_config(emeet_index)
    
    # Use default device
    return get_default_config()


# ============================================================================
# Utility Functions
# ============================================================================

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
