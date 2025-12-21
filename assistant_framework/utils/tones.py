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
    """Ascending two-tone chime for system ready."""
    _beep_platform(
        mac_sounds=["Blow", "Glass"],
        linux_ids=["service-login", "device-added"],
        win_tone=(800, 100),
    )
    # Small delay then second tone
    import time
    time.sleep(0.1)
    _beep_platform(
        mac_sounds=["Glass"],
        linux_ids=["bell"],
        win_tone=(1200, 100),
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


def _beep_impl_shutdown() -> None:
    """Descending two-tone for shutdown/goodbye."""
    _beep_platform(
        mac_sounds=["Glass"],
        linux_ids=["service-logout", "bell"],
        win_tone=(1000, 100),
    )
    import time
    time.sleep(0.1)
    _beep_platform(
        mac_sounds=["Blow"],
        linux_ids=["bell"],
        win_tone=(600, 150),
    )


def _beep_impl_wake_model_ready() -> None:
    """Quick double-beep for wake word model ready."""
    _beep_platform(
        # Avoid "Pop" because it's also used by other events; "Purr" is unique/distinct.
        mac_sounds=["Purr"],
        linux_ids=["device-added", "bell"],
        win_tone=(1100, 60),
    )
    import time
    time.sleep(0.05)
    _beep_platform(
        mac_sounds=["Purr"],
        linux_ids=["bell"],
        win_tone=(1100, 60),
    )

