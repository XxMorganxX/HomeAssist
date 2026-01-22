"""
TTS announcements for phase transitions.

Provides instant audio playback for state transition announcements.
Audio files are pre-generated and cached at boot for zero-latency playback.

Design goals:
- Instant: pre-cached audio plays immediately with no TTS delay
- Non-blocking: playback runs in separate thread
- Graceful failure: never crashes the main flow
- Boot optimization: all audio generated once during startup
- Cache validation: regenerates if TTS provider/voice changes
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from assistant_framework.interfaces.text_to_speech import TextToSpeechInterface


# Global cache for pre-generated audio files
_audio_cache: Dict[str, Path] = {}
_cache_dir: Optional[Path] = None
_cache_initialized = False

# Metadata filename for cache validation
_CACHE_METADATA_FILE = "cache_metadata.json"


def get_cache_dir() -> Path:
    """Get or create the cache directory for audio files."""
    global _cache_dir
    if _cache_dir is None:
        # Use a persistent cache in the project's audio_data folder
        project_root = Path(__file__).parent.parent.parent.parent
        _cache_dir = project_root / "audio_data" / "announcement_cache"
        _cache_dir.mkdir(parents=True, exist_ok=True)
    return _cache_dir


def _get_tts_signature(tts_provider: "TextToSpeechInterface") -> Dict[str, str]:
    """
    Get a signature dict identifying the TTS provider and voice settings.
    
    Used to validate cache - if signature changes, cache is invalidated.
    
    Args:
        tts_provider: TTS provider instance
        
    Returns:
        Dict with provider name, voice, and other identifying settings
    """
    signature = {
        "provider": type(tts_provider).__name__,
    }
    
    # Extract voice setting (different providers store it differently)
    if hasattr(tts_provider, 'voice'):
        signature["voice"] = str(tts_provider.voice)
    elif hasattr(tts_provider, 'voice_name'):
        signature["voice"] = str(tts_provider.voice_name)
    elif hasattr(tts_provider, 'config') and isinstance(tts_provider.config, dict):
        signature["voice"] = str(tts_provider.config.get('voice', 'default'))
    else:
        signature["voice"] = "default"
    
    # Extract model if available (e.g., for OpenAI TTS)
    if hasattr(tts_provider, 'model'):
        signature["model"] = str(tts_provider.model)
    
    # Extract speed if available
    if hasattr(tts_provider, 'speed'):
        signature["speed"] = str(tts_provider.speed)
    
    return signature


def _load_cache_metadata() -> Optional[Dict[str, str]]:
    """Load cache metadata from file."""
    cache_dir = get_cache_dir()
    metadata_path = cache_dir / _CACHE_METADATA_FILE
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_cache_metadata(signature: Dict[str, str]) -> None:
    """Save cache metadata to file."""
    cache_dir = get_cache_dir()
    metadata_path = cache_dir / _CACHE_METADATA_FILE
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(signature, f, indent=2)
    except Exception as e:
        print(f"⚠️  Failed to save cache metadata: {e}")


def _validate_cache_metadata(tts_provider: "TextToSpeechInterface") -> bool:
    """
    Check if cached files match current TTS settings.
    
    Args:
        tts_provider: Current TTS provider instance
        
    Returns:
        True if cache is valid, False if it needs regeneration
    """
    current_signature = _get_tts_signature(tts_provider)
    cached_signature = _load_cache_metadata()
    
    if cached_signature is None:
        # No metadata - cache may be from old version, regenerate
        return False
    
    # Compare signatures
    if current_signature != cached_signature:
        print(f"🔄 TTS settings changed - regenerating announcement cache")
        print(f"   Cached: {cached_signature.get('provider', '?')}/{cached_signature.get('voice', '?')}")
        print(f"   Current: {current_signature.get('provider', '?')}/{current_signature.get('voice', '?')}")
        return False
    
    return True


async def precache_announcements(tts_provider: "TextToSpeechInterface", tool_names: Optional[List[str]] = None) -> None:
    """
    Pre-generate and cache all announcement audio files at boot.
    
    Call this during startup to eliminate TTS latency during runtime.
    Validates existing cached files and TTS settings - regenerates if provider/voice changed.
    
    Args:
        tts_provider: Initialized TTS provider instance
        tool_names: Optional list of tool names to pre-cache (for tool success/failure)
    """
    global _cache_initialized, _audio_cache
    
    if not tts_provider:
        print("⚠️  Cannot precache announcements: no TTS provider")
        return
    
    cache_dir = get_cache_dir()
    
    # Check if TTS settings have changed (provider, voice, model)
    cache_valid = _validate_cache_metadata(tts_provider)
    
    if not cache_valid:
        # TTS settings changed - clear entire cache and regenerate
        _clear_cache_files(cache_dir)
        _audio_cache.clear()
    
    # Track stats
    loaded_from_cache = 0
    newly_generated = 0
    
    # Static announcements to pre-cache
    static_phrases = {
        "termination": "Conversation ended",
        "conversation_start": "Listening",
    }
    
    # Pre-cache static phrases
    for key, phrase in static_phrases.items():
        cache_path = cache_dir / f"{key}.wav"
        if cache_valid and _is_valid_cache_file(cache_path):
            # Valid cache exists - use it
            _audio_cache[key] = cache_path
            loaded_from_cache += 1
        else:
            # Generate new cache file
            if await _generate_and_cache(tts_provider, phrase, cache_path):
                _audio_cache[key] = cache_path
                newly_generated += 1
    
    # Pre-cache tool announcements if tool names provided
    if tool_names:
        for tool_name in tool_names:
            spoken_name = _format_tool_name_for_speech(tool_name)
            
            # Cache success announcement
            success_key = f"tool_{tool_name}_success"
            success_path = cache_dir / f"{success_key}.wav"
            if cache_valid and _is_valid_cache_file(success_path):
                _audio_cache[success_key] = success_path
                loaded_from_cache += 1
            else:
                if await _generate_and_cache(tts_provider, f"{spoken_name} success", success_path):
                    _audio_cache[success_key] = success_path
                    newly_generated += 1
            
            # Cache failure announcement
            failure_key = f"tool_{tool_name}_failed"
            failure_path = cache_dir / f"{failure_key}.wav"
            if cache_valid and _is_valid_cache_file(failure_path):
                _audio_cache[failure_key] = failure_path
                loaded_from_cache += 1
            else:
                if await _generate_and_cache(tts_provider, f"{spoken_name} failed", failure_path):
                    _audio_cache[failure_key] = failure_path
                    newly_generated += 1
    
    # Save current TTS signature as cache metadata
    if newly_generated > 0:
        _save_cache_metadata(_get_tts_signature(tts_provider))
    
    _cache_initialized = True
    
    # Log summary with TTS info
    total = len(_audio_cache)
    tts_sig = _get_tts_signature(tts_provider)
    tts_info = f"{tts_sig.get('provider', '?')}/{tts_sig.get('voice', '?')}"
    
    if newly_generated == 0 and loaded_from_cache > 0:
        print(f"🔊 Loaded {loaded_from_cache} cached announcements ({tts_info})")
    elif loaded_from_cache == 0 and newly_generated > 0:
        print(f"🔊 Generated {newly_generated} announcements ({tts_info})")
    else:
        print(f"🔊 Announcements: {loaded_from_cache} cached, {newly_generated} new ({tts_info})")


def _clear_cache_files(cache_dir: Path) -> None:
    """
    Clear all cached audio files (but keep the directory).
    
    Called when TTS settings change and cache needs regeneration.
    
    Args:
        cache_dir: Path to the cache directory
    """
    if not cache_dir.exists():
        return
    
    for audio_file in cache_dir.glob("*.wav"):
        try:
            audio_file.unlink()
        except Exception:
            pass
    
    # Also remove the metadata file
    metadata_path = cache_dir / _CACHE_METADATA_FILE
    if metadata_path.exists():
        try:
            metadata_path.unlink()
        except Exception:
            pass


def _is_valid_cache_file(cache_path: Path) -> bool:
    """
    Check if a cached audio file exists and is valid.
    
    Validates:
    - File exists
    - File is not empty
    - File has minimum size (likely valid audio)
    
    Args:
        cache_path: Path to the cache file
        
    Returns:
        True if file is valid, False if missing or invalid
    """
    if not cache_path.exists():
        return False
    
    try:
        file_size = cache_path.stat().st_size
        # Minimum size check: valid audio should be at least 1KB
        # (even short phrases produce several KB of audio)
        if file_size < 1024:
            print(f"⚠️  Cache file too small, will regenerate: {cache_path.name}")
            return False
        return True
    except Exception:
        return False


async def _generate_and_cache(tts_provider: "TextToSpeechInterface", text: str, cache_path: Path) -> bool:
    """
    Generate audio for a phrase and save to cache file.
    
    Args:
        tts_provider: TTS provider instance
        text: Text to synthesize
        cache_path: Path to save the audio file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        audio = await tts_provider.synthesize(text)
        if audio and audio.audio_data:
            # Save audio data to file
            with open(cache_path, 'wb') as f:
                f.write(audio.audio_data)
            return True
    except Exception as e:
        print(f"⚠️  Failed to cache '{text}': {e}")
    return False


def announce_termination(tts_provider: "TextToSpeechInterface" = None) -> None:
    """
    Play 'Conversation ended' announcement when termination phrase detected.
    
    Uses pre-cached audio for instant playback.
    Falls back to live TTS if cache not available.
    
    Args:
        tts_provider: TTS provider (only needed if cache miss)
    """
    _play_cached_or_generate("termination", "Conversation ended", tts_provider)


def announce_conversation_start(tts_provider: "TextToSpeechInterface" = None) -> None:
    """
    Play brief acknowledgment when wake word detected and conversation starts.
    
    Uses pre-cached audio for instant playback.
    Falls back to live TTS if cache not available.
    
    Args:
        tts_provider: TTS provider (only needed if cache miss)
    """
    _play_cached_or_generate("conversation_start", "Listening", tts_provider)


def announce_tool_call(
    tts_provider: "TextToSpeechInterface",
    tool_name: str,
    success: bool
) -> None:
    """
    Play '{tool_name} success' or '{tool_name} failed' after tool execution.
    
    Uses pre-cached audio for instant playback.
    Falls back to live TTS if cache not available.
    
    Args:
        tts_provider: TTS provider (only needed if cache miss)
        tool_name: Name of the tool that was executed
        success: Whether the tool execution succeeded
    """
    status = "success" if success else "failed"
    cache_key = f"tool_{tool_name}_{status}"
    spoken_name = _format_tool_name_for_speech(tool_name)
    fallback_text = f"{spoken_name} {status}"
    
    _play_cached_or_generate(cache_key, fallback_text, tts_provider)


def _play_cached_or_generate(cache_key: str, fallback_text: str, tts_provider: "TextToSpeechInterface" = None) -> None:
    """
    Play from cache if available, otherwise generate live.
    
    Args:
        cache_key: Key to look up in audio cache
        fallback_text: Text to synthesize if cache miss
        tts_provider: TTS provider for fallback generation
    """
    cache_path = _audio_cache.get(cache_key)
    
    if cache_path and cache_path.exists():
        # Play from cache - instant!
        _play_audio_file(cache_path)
    elif tts_provider:
        # Cache miss - fall back to live TTS (slower)
        _speak_in_thread(tts_provider, fallback_text)


def _play_audio_file(audio_path: Path) -> None:
    """
    Play an audio file instantly in a separate thread.
    
    Uses platform-native playback for minimal latency.
    
    Args:
        audio_path: Path to the audio file
    """
    def _play():
        try:
            if sys.platform == "darwin":
                # macOS: use afplay (fast, native)
                subprocess.run(
                    ["afplay", str(audio_path)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            elif sys.platform.startswith("linux"):
                # Linux: use aplay
                subprocess.run(
                    ["aplay", "-q", str(audio_path)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                # Windows or other: try ffplay
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
        except Exception as e:
            print(f"⚠️  Audio playback failed: {e}")
    
    # Play in daemon thread for non-blocking execution
    thread = threading.Thread(target=_play, daemon=True)
    thread.start()


def _format_tool_name_for_speech(tool_name: str) -> str:
    """
    Format a tool name for natural speech.
    
    Examples:
        "calendar_data" -> "Calendar"
        "get_weather" -> "Weather"
        "spotify_control" -> "Spotify"
        "send_sms" -> "SMS"
    
    Args:
        tool_name: Raw tool name (e.g., "calendar_data")
        
    Returns:
        Formatted name suitable for TTS (e.g., "Calendar")
    """
    # Common prefixes to strip
    prefixes_to_strip = ["get_", "set_", "send_", "fetch_", "update_", "create_", "delete_"]
    
    # Common suffixes to strip
    suffixes_to_strip = ["_data", "_info", "_control", "_tool"]
    
    name = tool_name.lower()
    
    # Strip common prefixes
    for prefix in prefixes_to_strip:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    
    # Strip common suffixes
    for suffix in suffixes_to_strip:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    
    # Handle special abbreviations that should stay uppercase
    uppercase_words = {"sms", "api", "url", "id", "ai"}
    
    # Replace underscores with spaces and title case
    words = name.replace("_", " ").split()
    formatted_words = []
    for word in words:
        if word.lower() in uppercase_words:
            formatted_words.append(word.upper())
        else:
            formatted_words.append(word.capitalize())
    
    return " ".join(formatted_words) if formatted_words else tool_name


def _speak_in_thread(
    tts_provider: "TextToSpeechInterface",
    text: str
) -> None:
    """
    Fallback: Fire-and-forget TTS announcement in a dedicated thread.
    
    Used when cache is not available. Generates audio live.
    
    Args:
        tts_provider: Initialized TTS provider instance
        text: Text to speak
    """
    if not tts_provider:
        return
    
    def _run_tts():
        """Thread target: run async TTS in new event loop."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_speak_async(tts_provider, text))
            finally:
                loop.close()
        except Exception as e:
            # Log but don't propagate - announcements should never crash
            print(f"⚠️  TTS announcement failed: {e}")
    
    # Start in daemon thread (won't block program exit)
    thread = threading.Thread(target=_run_tts, daemon=True)
    thread.start()


async def _speak_async(
    tts_provider: "TextToSpeechInterface",
    text: str
) -> None:
    """
    Internal async helper to synthesize and play announcement.
    
    Args:
        tts_provider: Initialized TTS provider instance
        text: Text to speak
    """
    try:
        # Synthesize the announcement
        audio = await tts_provider.synthesize(text)
        
        if audio and audio.audio_data:
            # Play the audio (use async method if available)
            if hasattr(tts_provider, 'play_audio_async'):
                await tts_provider.play_audio_async(audio)
            else:
                # Fallback to sync method
                tts_provider.play_audio(audio)
                
    except asyncio.CancelledError:
        # Task was cancelled - exit silently
        pass
    except Exception as e:
        # Log but don't propagate
        print(f"⚠️  TTS synthesis/playback failed: {e}")


def clear_announcement_cache() -> None:
    """
    Clear the announcement cache (delete all cached audio files and metadata).
    
    Useful when TTS voice/settings change and cache needs regenerating.
    Call this manually or it will auto-detect settings changes on next boot.
    """
    global _audio_cache, _cache_initialized
    
    cache_dir = get_cache_dir()
    _clear_cache_files(cache_dir)
    
    _audio_cache.clear()
    _cache_initialized = False
    print("🗑️  Announcement cache cleared")
