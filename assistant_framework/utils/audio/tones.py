"""
Tiny, resilient notification tones.

Design goals:
- Non-blocking: fire-and-forget on a background thread
- No dependency on PyAudio or pygame (avoids device conflicts)
- Cross-platform best-effort with safe fallbacks
- Never raise exceptions to callers
"""

import os
import sys
import threading
import subprocess
from shutil import which


# Public event-specific helpers
def beep_wake_detected() -> None:
    _play_async(_beep_impl_wake)


def beep_agent_message() -> None:
    _play_async(_beep_impl_agent)


def beep_transcription_end() -> None:
    _play_async(_beep_impl_end)


def beep_ready_to_listen() -> None:
    _play_async(_beep_impl_ready)


def beep_send_detected() -> None:
    _play_async(_beep_impl_send)


def _beep_impl_wake() -> None:
    """Distinct beep for wakeword detected."""
    _beep_platform(
        mac_sounds=["Ping", "Glass"],
        linux_ids=["message-new-instant", "bell"],
        win_tone=(1200, 120),
    )


def _beep_impl_agent() -> None:
    """Distinct beep for first agent response chunk."""
    _beep_platform(
        mac_sounds=["Pop", "Submarine"],
        linux_ids=["complete", "dialog-information"],
        win_tone=(900, 120),
    )


def _beep_impl_end() -> None:
    """Distinct beep for transcription end."""
    _beep_platform(
        mac_sounds=["Sosumi", "Basso"],
        linux_ids=["dialog-warning", "bell"],
        win_tone=(600, 150),
    )


def _beep_impl_ready() -> None:
    """Distinct beep indicating ready to listen again."""
    _beep_platform(
        # Use a very distinct sound to avoid confusion with listening_start ("Tink")
        mac_sounds=["Frog", "Bottle"],
        linux_ids=["audio-volume-change", "bell"],
        win_tone=(1400, 100),
    )


def _beep_platform(mac_sounds, linux_ids, win_tone) -> None:
    """Best-effort short beep for the current platform."""
    try:
        # macOS: prefer system beep via AppleScript (does not touch audio devices)
        if sys.platform == "darwin":
            try:
                afplay = which("afplay")
                if afplay:
                    for sound in mac_sounds:
                        name = f"{sound}.aiff"
                        candidate = f"/System/Library/Sounds/{name}"
                        if os.path.exists(candidate):
                            subprocess.run(
                                [afplay, candidate],
                                check=False,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            return
            except Exception:
                pass

        # Windows: try winsound, otherwise fall back to bell char
        if sys.platform.startswith("win"):
            try:
                import winsound  # type: ignore
                freq, dur = win_tone
                winsound.Beep(int(freq), int(dur))
                return
            except Exception:
                pass

        # Linux/other: try canberra-gtk-play or paplay if present
        linux_candidates = []
        # First, event ids
        for event_id in linux_ids:
            linux_candidates.append(("canberra-gtk-play", ["--id", event_id]))
        # Then, common files
        linux_candidates.extend(
            [
                ("paplay", ["/usr/share/sounds/freedesktop/stereo/bell.oga"]),
                ("aplay", ["/usr/share/sounds/alsa/Front_Center.wav"]),
            ]
        )

        for player, args in linux_candidates:
            try:
                if which(player):
                    subprocess.run(
                        [player, *args],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
            except Exception:
                pass

        # Final fallback: ASCII bell to stdout
        _fallback_bell()
    except Exception:
        # Never propagate exceptions
        pass


def play_short_beep() -> None:
    """Fire-and-forget short beep with robust fallbacks."""
    _play_async(_beep_impl_ready)


def _play_async(fn) -> None:
    try:
        t = threading.Thread(target=fn, daemon=True)
        t.start()
    except Exception:
        _fallback_bell()


def _fallback_bell() -> None:
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


def _beep_impl_send() -> None:
    """Distinct beep for send phrase detected."""
    _beep_platform(
        mac_sounds=["Hero", "Funk"],
        linux_ids=["message-sent", "mail-sent", "dialog-information"],
        win_tone=(1050, 180),
    )


def beep_system_ready() -> None:
    """Play sound when system is fully initialized and ready."""
    _play_async(_beep_impl_system_ready)


def beep_listening_start() -> None:
    """Play sound when transcription/listening begins."""
    _play_async(_beep_impl_listening)


def beep_response_start() -> None:
    """Play sound when assistant starts generating response."""
    _play_async(_beep_impl_response)


def beep_error() -> None:
    """Play sound on error."""
    _play_async(_beep_impl_error)


def _beep_impl_system_ready() -> None:
    """Single tone for system ready (simplified for faster boot)."""
    _beep_platform(
        mac_sounds=["Glass"],
        linux_ids=["service-login", "device-added"],
        win_tone=(1000, 120),
    )


def _beep_impl_listening() -> None:
    """Short high beep for listening start."""
    _beep_platform(
        # Keep this crisp; distinct from ready_to_listen ("Frog")
        mac_sounds=["Tink"],
        linux_ids=["audio-channel-front-center"],
        win_tone=(1500, 80),
    )


def _beep_impl_response() -> None:
    """Soft tone for response starting."""
    _beep_platform(
        mac_sounds=["Morse"],
        linux_ids=["message"],
        win_tone=(700, 100),
    )


def _beep_impl_error() -> None:
    """Low tone for errors."""
    _beep_platform(
        mac_sounds=["Basso", "Sosumi"],
        linux_ids=["dialog-error", "dialog-warning"],
        win_tone=(400, 200),
    )


def beep_shutdown() -> None:
    """Play sound when assistant is shutting down / session ending."""
    _play_async(_beep_impl_shutdown)


def beep_wake_model_ready() -> None:
    """Play sound when wake word model is initialized and ready."""
    _play_async(_beep_impl_wake_model_ready)


def beep_tool_success() -> None:
    """Play sound when a tool executes successfully."""
    _play_async(_beep_impl_tool_success)


def beep_tool_failure() -> None:
    """Play sound when a tool execution fails."""
    _play_async(_beep_impl_tool_failure)


def beep_tool_call() -> None:
    """Play sound when a tool is being called/started."""
    _play_async(_beep_impl_tool_call)


def beep_tools_complete() -> None:
    """Play sound when all tool calls are complete (last tool finished)."""
    _play_async(_beep_impl_tools_complete)


def _beep_impl_shutdown() -> None:
    """Single tone for shutdown/goodbye (simplified for faster transition)."""
    _beep_platform(
        mac_sounds=["Blow"],
        linux_ids=["service-logout", "bell"],
        win_tone=(700, 150),
    )


def _beep_impl_wake_model_ready() -> None:
    """Single beep for wake word model ready (user can now activate wake word)."""
    _beep_platform(
        # Clear, distinct sound to indicate system is ready for wake word
        mac_sounds=["Glass"],
        linux_ids=["device-added", "bell"],
        win_tone=(1100, 100),
    )


def _beep_impl_tool_success() -> None:
    """Upward chime for successful tool execution."""
    _beep_platform(
        # Pleasant, confirmatory sounds
        mac_sounds=["Pop", "Tink", "Glass"],
        linux_ids=["complete", "message-sent-instant", "dialog-information"],
        win_tone=(1200, 100),
    )


def _beep_impl_tool_failure() -> None:
    """Downward tone for failed tool execution."""
    _beep_platform(
        # Warning/error sounds
        mac_sounds=["Funk", "Basso"],
        linux_ids=["dialog-warning", "message-attention"],
        win_tone=(500, 150),
    )


def _beep_impl_tool_call() -> None:
    """Short click for tool being called."""
    _beep_platform(
        # Quick, subtle click to indicate action started
        mac_sounds=["Tink"],
        linux_ids=["button-pressed", "bell"],
        win_tone=(1000, 50),
    )


def _beep_impl_tools_complete() -> None:
    """Completion chime when all tools are done."""
    _beep_platform(
        # Satisfying completion sound
        mac_sounds=["Bottle"],
        linux_ids=["complete", "bell"],
        win_tone=(1100, 120),
    )


# =============================================================================
# STATE TRANSITION BEEPS (Config-Driven)
# Each unique state transition gets a distinct sound for audio feedback.
# Sounds are configured in assistant_framework/config.py under TRANSITION_SOUNDS.
# =============================================================================

def _get_transition_config():
    """Lazy-load transition sound config to avoid circular imports."""
    try:
        from assistant_framework.config import (
            TRANSITION_SOUNDS,
            TRANSITION_SOUNDS_LINUX,
            TRANSITION_SOUNDS_WINDOWS,
        )
        return TRANSITION_SOUNDS, TRANSITION_SOUNDS_LINUX, TRANSITION_SOUNDS_WINDOWS
    except ImportError:
        # Fallback defaults if config not available
        return {}, {}, {}


def _play_transition_sound(transition_key: str) -> None:
    """
    Play the configured sound for a state transition.
    
    Only plays a sound if the transition is explicitly configured.
    Unlisted transitions are silent.
    
    Args:
        transition_key: Key in TRANSITION_SOUNDS config (e.g., "idle_to_wakeword")
    """
    mac_sounds, linux_sounds, win_sounds = _get_transition_config()
    
    # Only play if transition is explicitly listed in config
    # Unlisted transitions = silent (no sound)
    if transition_key not in mac_sounds:
        return
    
    mac_sound = mac_sounds.get(transition_key)
    linux_id = linux_sounds.get(transition_key)
    win_tone = win_sounds.get(transition_key)
    
    # Skip if sound is explicitly disabled (set to None)
    if mac_sound is None:
        return
    
    _beep_platform(
        mac_sounds=[mac_sound],
        linux_ids=[linux_id, "bell"] if linux_id else ["bell"],
        win_tone=win_tone or (800, 100),
    )


def beep_transition_idle_to_wakeword() -> None:
    """IDLE → WAKE_WORD_LISTENING: System entering wake word detection mode."""
    _play_async(lambda: _play_transition_sound("idle_to_wakeword"))


def beep_transition_idle_to_synthesizing() -> None:
    """IDLE → SYNTHESIZING: Starting TTS directly from idle (dev mode)."""
    _play_async(lambda: _play_transition_sound("idle_to_synthesizing"))


def beep_transition_idle_to_transcribing() -> None:
    """IDLE → TRANSCRIBING: Starting transcription directly from idle (dev mode)."""
    _play_async(lambda: _play_transition_sound("idle_to_transcribing"))


def beep_transition_wakeword_to_transcribing() -> None:
    """WAKE_WORD_LISTENING → TRANSCRIBING: Wake word detected, starting transcription."""
    _play_async(lambda: _play_transition_sound("wakeword_to_transcribing"))


def beep_transition_wakeword_to_processing() -> None:
    """WAKE_WORD_LISTENING → PROCESSING_RESPONSE: Proactive response triggered."""
    _play_async(lambda: _play_transition_sound("wakeword_to_processing"))


def beep_transition_wakeword_to_synthesizing() -> None:
    """WAKE_WORD_LISTENING → SYNTHESIZING: Pre-generated briefing starting."""
    _play_async(lambda: _play_transition_sound("wakeword_to_synthesizing"))


def beep_transition_wakeword_to_idle() -> None:
    """WAKE_WORD_LISTENING → IDLE: Wake word detection cancelled."""
    _play_async(lambda: _play_transition_sound("wakeword_to_idle"))


def beep_transition_transcribing_to_processing() -> None:
    """TRANSCRIBING → PROCESSING_RESPONSE: Transcription complete, processing."""
    _play_async(lambda: _play_transition_sound("transcribing_to_processing"))


def beep_transition_transcribing_to_idle() -> None:
    """TRANSCRIBING → IDLE: Transcription cancelled/timeout."""
    _play_async(lambda: _play_transition_sound("transcribing_to_idle"))


def beep_transition_processing_to_synthesizing() -> None:
    """PROCESSING_RESPONSE → SYNTHESIZING: Response ready, speaking."""
    _play_async(lambda: _play_transition_sound("processing_to_synthesizing"))


def beep_transition_processing_to_transcribing() -> None:
    """PROCESSING_RESPONSE → TRANSCRIBING: Barge-in during processing."""
    _play_async(lambda: _play_transition_sound("processing_to_transcribing"))


def beep_transition_processing_to_idle() -> None:
    """PROCESSING_RESPONSE → IDLE: Processing cancelled."""
    _play_async(lambda: _play_transition_sound("processing_to_idle"))


def beep_transition_synthesizing_to_idle() -> None:
    """SYNTHESIZING → IDLE: TTS complete."""
    _play_async(lambda: _play_transition_sound("synthesizing_to_idle"))


def beep_transition_synthesizing_to_wakeword() -> None:
    """SYNTHESIZING → WAKE_WORD_LISTENING: TTS complete, resuming wake word."""
    _play_async(lambda: _play_transition_sound("synthesizing_to_wakeword"))


def beep_transition_synthesizing_to_transcribing() -> None:
    """SYNTHESIZING → TRANSCRIBING: Barge-in during TTS."""
    _play_async(lambda: _play_transition_sound("synthesizing_to_transcribing"))


def beep_transition_error_to_idle() -> None:
    """ERROR → IDLE: Error recovery."""
    _play_async(lambda: _play_transition_sound("error_to_idle"))


def beep_transition_to_error() -> None:
    """Any → ERROR: System entering error state."""
    # All error transitions use the same sound
    _play_async(lambda: _play_transition_sound("wakeword_to_error"))

