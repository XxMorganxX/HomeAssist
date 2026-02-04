#!/usr/bin/env python3
"""
Check if Meta Ray-Ban glasses are the active audio input/output device.
Exit code 0 = active, exit code 1 = not active.
"""

import sys
import subprocess


def is_meta_glasses_active_audio_device() -> bool:
    """
    Check if Meta Ray-Ban glasses are both:
    1. Connected via Bluetooth
    2. Set as the default input AND output audio device
    
    Uses system_profiler to query macOS audio system state.
    
    Returns:
        True if Meta glasses are connected AND set as default I/O, False otherwise
    """
    try:
        # Query macOS audio system
        result = subprocess.run(
            ['system_profiler', 'SPAudioDataType'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return False
        
        output = result.stdout.lower()
        
        # Look for Meta glasses device names (various possible names)
        meta_device_indicators = [
            "meta bans",
            "ray-ban",
            "raybans", 
            "ray ban"
        ]
        
        # Check if any Meta device indicator appears in output
        has_meta_device = any(indicator in output for indicator in meta_device_indicators)
        if not has_meta_device:
            return False
        
        # Parse the output to check for "Default Input Device: Yes" and "Default Output Device: Yes"
        # Note: macOS system_profiler shows Meta glasses as TWO entries (one for input, one for output)
        # We need to accumulate both flags across all Meta device sections
        
        lines = result.stdout.split('\n')
        is_meta_device_section = False
        found_default_input = False
        found_default_output = False
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if this line contains a device name with Meta/Ray-Ban
            if any(indicator in line_lower for indicator in meta_device_indicators):
                is_meta_device_section = True
            # If we hit a new device section (non-indented line with ':'), end current section
            elif line and not line[0].isspace() and ':' in line:
                is_meta_device_section = False
            
            # Check for default status markers in Meta device sections
            if is_meta_device_section:
                if 'default input device: yes' in line_lower:
                    found_default_input = True
                elif 'default output device: yes' in line_lower:
                    found_default_output = True
        
        # Return true only if we found both input AND output
        return found_default_input and found_default_output
        
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


if __name__ == "__main__":
    is_active = is_meta_glasses_active_audio_device()
    sys.exit(0 if is_active else 1)
