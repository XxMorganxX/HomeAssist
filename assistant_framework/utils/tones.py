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
        mac_sounds=["Tink", "Glass"],
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


