"""
Shared audio resource manager to prevent conflicts between components.
Includes optional verbose debug logging controlled by the AUDIO_DEBUG env var.

Note: This has been simplified for sounddevice which doesn't require
instance management like PyAudio did. We now just track ownership for coordination.
"""

import asyncio
import threading
import time
import os
from datetime import datetime
from typing import Optional

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
    Manages shared audio resource ownership to prevent conflicts between components.
    With sounddevice, we don't need to manage an instance, just coordinate ownership.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._current_owner: Optional[str] = None
        self._owner_count = 0
        self._cleanup_delay = 0.1  # Reduced from 0.5 for faster transitions
        # Debug controls
        self._debug_enabled = str(os.getenv("AUDIO_DEBUG", "")).lower() in ("1", "true", "yes", "on")
        self._debug_file_path = os.getenv("AUDIO_DEBUG_FILE")

    def _log(self, message: str) -> None:
        if not self._debug_enabled:
            return
        ts = datetime.now().strftime("%H:%M:%S.%f")
        thread_name = threading.current_thread().name
        prefix = f"[AUDIO][{ts}][{thread_name}] "
        try:
            print(prefix + message)
        except Exception:
            pass
        if self._debug_file_path:
            try:
                with open(self._debug_file_path, "a", encoding="utf-8") as f:
                    f.write(prefix + message + "\n")
            except Exception:
                pass
        
    def acquire_audio(self, owner_name: str, force_cleanup: bool = True) -> bool:
        """
        Acquire audio ownership for the given component.
        
        Args:
            owner_name: Name of the component requesting audio
            force_cleanup: Whether to force cleanup before acquisition
            
        Returns:
            True if acquired successfully, False otherwise
        """
        with self._lock:
            self._log(
                f"acquire_audio(owner={owner_name}, force_cleanup={force_cleanup}) "
                f"state={{owner:{self._current_owner}, count:{self._owner_count}}}"
            )
            try:
                # If there's a different owner, clean up first
                if self._current_owner and self._current_owner != owner_name:
                    if force_cleanup:
                        self._log(f"Different owner detected ({self._current_owner}); forcing cleanup before acquiring for {owner_name}")
                        self._cleanup_audio_unsafe()
                        # Small delay to let audio system settle
                        time.sleep(self._cleanup_delay)
                    else:
                        print(f"⚠️  Audio busy with {self._current_owner}, cannot acquire for {owner_name}")
                        return False
                
                # Assign ownership
                if self._current_owner == owner_name:
                    # Same owner requesting again
                    self._owner_count += 1
                else:
                    self._current_owner = owner_name
                    self._owner_count = 1
                
                print(f"✅ Audio acquired by {owner_name} (count: {self._owner_count})")
                self._log(
                    f"acquire_audio success for {owner_name}; state={{owner:{self._current_owner}, count:{self._owner_count}}}"
                )
                return True
                
            except Exception as e:
                print(f"❌ Failed to acquire audio for {owner_name}: {e}")
                self._cleanup_audio_unsafe()
                return False
    
    def release_audio(self, owner_name: str, force_cleanup: bool = False):
        """
        Release audio ownership from the given component.

        Args:
            owner_name: Name of the component releasing audio
            force_cleanup: Whether to force immediate cleanup
        """
        with self._lock:
            self._log(
                f"release_audio(owner={owner_name}, force_cleanup={force_cleanup}) "
                f"state(before)={{owner:{self._current_owner}, count:{self._owner_count}}}"
            )
            # Handle case where audio was already cleaned up
            if self._current_owner is None:
                self._log("release_audio: no current owner; nothing to do")
                return

            if self._current_owner != owner_name:
                # Only warn if there's an actual owner mismatch
                if self._current_owner is not None:
                    print(f"⚠️  {owner_name} tried to release audio owned by {self._current_owner}")
                self._log(f"release_audio: owner mismatch; current_owner={self._current_owner}")
                return

            self._owner_count = max(0, self._owner_count - 1)
            print(f"📤 Audio released by {owner_name} (count: {self._owner_count})")

            # Release ownership when count reaches zero
            if self._owner_count <= 0:
                self._current_owner = None
            if force_cleanup:
                self._cleanup_audio_unsafe()
            self._log(
                f"release_audio complete; state(after)={{owner:{self._current_owner}, count:{self._owner_count}}}"
            )
    
    def _cleanup_audio_unsafe(self):
        """
        Clean up audio state. Must be called with lock held.
        """
        try:
            if self._current_owner:
                print(f"🧹 Cleaning up audio (owner: {self._current_owner})")
                self._log("_cleanup_audio_unsafe: clearing ownership")
            self._current_owner = None
            self._owner_count = 0
            print("✅ Audio cleanup completed")
        except Exception as e:
            print(f"⚠️  Error cleaning up audio: {e}")
            self._current_owner = None
            self._owner_count = 0
    
    def force_cleanup(self):
        """Force cleanup of all audio resources."""
        with self._lock:
            self._log("force_cleanup called")
            self._cleanup_audio_unsafe()
    
    def get_status(self) -> dict:
        """Get current audio manager status."""
        with self._lock:
            return {
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
    # Debug helper
    def _dbg(msg: str) -> None:
        try:
            # Access manager's internal logger if available
            getattr(manager, "_log", lambda m: None)(f"safe_audio_transition: {msg}")
        except Exception:
            pass
    
    # Soft release to avoid tearing down audio on every handoff
    print(f"🔄 Releasing audio for transition: {from_owner} → {to_owner}")
    _dbg(f"begin release; status(before)={manager.get_status()}")
    try:
        manager.release_audio(from_owner, force_cleanup=False)
    except Exception as e:
        print(f"⚠️  Audio release warning during transition: {e}")
    _dbg(f"released; status(now)={manager.get_status()}")
    
    # Wait for audio system to settle
    await asyncio.sleep(delay)
    
    print(f"🔄 Audio transition: {from_owner} → {to_owner} ready")
    _dbg(f"settled after {delay}s; status={manager.get_status()}")
