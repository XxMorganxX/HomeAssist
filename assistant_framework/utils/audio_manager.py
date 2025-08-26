"""
Shared audio resource manager to prevent conflicts between components.
"""

import asyncio
import threading
import time
from typing import Optional
import pyaudio

# Best-effort import of tone helper; never let this fail
try:
    from .tones import beep_ready_to_listen  # type: ignore
except Exception:  # pragma: no cover
    try:
        from assistant_framework.utils.tones import beep_ready_to_listen  # type: ignore
    except Exception:  # pragma: no cover
        def beep_ready_to_listen() -> None:  # type: ignore
            return


class SharedAudioManager:
    """
    Manages shared PyAudio instance to prevent resource conflicts.
    Ensures only one component can access audio at a time.
    """
    
    def __init__(self):
        self._audio: Optional[pyaudio.PyAudio] = None
        self._lock = threading.Lock()
        self._current_owner: Optional[str] = None
        self._owner_count = 0
        self._cleanup_delay = 0.5  # Seconds to wait before cleanup
        
    def acquire_audio(self, owner_name: str, force_cleanup: bool = True) -> Optional[pyaudio.PyAudio]:
        """
        Acquire PyAudio instance for the given owner.
        
        Args:
            owner_name: Name of the component requesting audio
            force_cleanup: Whether to force cleanup before acquisition
            
        Returns:
            PyAudio instance or None if failed
        """
        with self._lock:
            try:
                # If there's a different owner, clean up first
                if self._current_owner and self._current_owner != owner_name:
                    if force_cleanup:
                        self._cleanup_audio_unsafe()
                        # Small delay to let audio system settle
                        time.sleep(self._cleanup_delay)
                    else:
                        print(f"⚠️  Audio busy with {self._current_owner}, cannot acquire for {owner_name}")
                        return None
                
                # Create new PyAudio instance if needed
                if not self._audio:
                    print(f"🎤 Initializing audio for {owner_name}")
                    # Audible cue specifically when preparing to listen for wakeword
                    if owner_name == "wakeword":
                        try:
                            beep_ready_to_listen()
                        except Exception:
                            pass
                    self._audio = pyaudio.PyAudio()
                    self._current_owner = owner_name
                    self._owner_count = 1
                elif self._current_owner == owner_name:
                    # Same owner requesting again
                    self._owner_count += 1
                else:
                    # This shouldn't happen with force_cleanup=True
                    print(f"⚠️  Unexpected audio state: current={self._current_owner}, requested={owner_name}")
                    return None
                
                print(f"✅ Audio acquired by {owner_name} (count: {self._owner_count})")
                if owner_name == "wakeword":
                    try:
                        beep_ready_to_listen()
                    except Exception:
                        pass
                return self._audio
                
            except Exception as e:
                print(f"❌ Failed to acquire audio for {owner_name}: {e}")
                self._cleanup_audio_unsafe()
                return None
    
    def release_audio(self, owner_name: str, force_cleanup: bool = False):
        """
        Release PyAudio instance from the given owner.
        
        Args:
            owner_name: Name of the component releasing audio
            force_cleanup: Whether to force immediate cleanup
        """
        with self._lock:
            # Handle case where audio was already cleaned up
            if self._current_owner is None and self._audio is None:
                print(f"📤 Audio already released for {owner_name}")
                return
                
            if self._current_owner != owner_name:
                print(f"⚠️  {owner_name} tried to release audio, but {self._current_owner} owns it")
                return
            
            self._owner_count = max(0, self._owner_count - 1)
            print(f"📤 Audio released by {owner_name} (count: {self._owner_count})")
            
            if self._owner_count <= 0 or force_cleanup:
                self._cleanup_audio_unsafe()
    
    def _cleanup_audio_unsafe(self):
        """
        Clean up PyAudio instance. Must be called with lock held.
        """
        if self._audio:
            try:
                print(f"🧹 Cleaning up audio (owner: {self._current_owner})")
                # Store reference and clear it immediately to prevent double cleanup
                audio_to_cleanup = self._audio
                self._audio = None
                self._current_owner = None
                self._owner_count = 0
                
                # Now terminate the stored reference
                audio_to_cleanup.terminate()
                print(f"✅ Audio cleanup completed")
            except Exception as e:
                print(f"⚠️  Error terminating PyAudio: {e}")
                # Ensure state is cleared even on error
                self._audio = None
                self._current_owner = None
                self._owner_count = 0
    
    def force_cleanup(self):
        """Force cleanup of all audio resources."""
        with self._lock:
            self._cleanup_audio_unsafe()
    
    def get_status(self) -> dict:
        """Get current audio manager status."""
        with self._lock:
            return {
                'has_audio': self._audio is not None,
                'current_owner': self._current_owner,
                'owner_count': self._owner_count
            }


# Global shared instance
_audio_manager = SharedAudioManager()


def get_audio_manager() -> SharedAudioManager:
    """Get the global audio manager instance."""
    return _audio_manager


async def safe_audio_transition(from_owner: str, to_owner: str, delay: float = 0.3):
    """
    Safely transition audio ownership between components.
    This handles force cleanup and settling time - the new component
    will acquire audio when it starts.
    
    Args:
        from_owner: Current owner to release
        to_owner: New owner that will acquire (for logging only)
        delay: Seconds to wait for audio system to settle
    """
    manager = get_audio_manager()
    
    # Force cleanup to ensure clean state for transition
    print(f"🔄 Forcing audio cleanup for transition: {from_owner} → {to_owner}")
    manager.force_cleanup()
    
    # Wait for audio system to settle
    await asyncio.sleep(delay)
    
    print(f"🔄 Audio transition: {from_owner} → {to_owner} ready")