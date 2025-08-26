"""
CLI entrypoint to run assistant framework components in isolation.

Usage examples:
  - Full pipeline (mic → response → TTS):
      python -m assistant_framework.main pipeline

  - Transcription only:
      python -m assistant_framework.main transcribe --final-only

  - Response only:
      python -m assistant_framework.main respond --message "Hello" --no-context

  - TTS only:
      python -m assistant_framework.main tts --text "Hello Mr. Stuart" --no-play --save speech_audio/hello.mp3

  - Single message through pipeline:
      python -m assistant_framework.main single --message "Turn on the living room lights"

  - Wake word only:
      python -m assistant_framework.main wakeword

  - Status and discovery:
      python -m assistant_framework.main status
      python -m assistant_framework.main list-providers
      python -m assistant_framework.main tools
"""

import asyncio
import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional


async def _build_config(env: str,
                        transcription: Optional[str],
                        response: Optional[str],
                        tts: Optional[str],
                        context: Optional[str]) -> Dict[str, Any]:
    """Create configuration for the orchestrator with optional overrides."""
    # Import configuration lazily so we could, in the future, control side effects
    try:
        from . import config as framework_config  # type: ignore
    except ImportError:
        # Support running this file directly: python assistant_framework/main.py
        import sys
        from pathlib import Path
        pkg_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(pkg_root))
        from assistant_framework import config as framework_config  # type: ignore

    # Apply provider overrides before constructing config dict
    if any([transcription, response, tts, context]):
        framework_config.set_providers(
            transcription=transcription,
            response=response,
            tts=tts,
            context=context,
        )

    # Select config profile
    if env and env != "default":
        # If user specified a preset, set it statically and use it
        framework_config.set_active_preset(env)
    cfg = framework_config.get_config_for_preset()

    return cfg


async def _create_orchestrator(config: Dict[str, Any]):
    try:
        from .orchestrator import create_orchestrator  # type: ignore
    except ImportError:
        import sys
        from pathlib import Path
        pkg_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(pkg_root))
        from assistant_framework.orchestrator import create_orchestrator  # type: ignore
    return await create_orchestrator(config)


async def cmd_pipeline(args) -> int:
    config = await _build_config(
        env=args.env,
        transcription=args.transcription,
        response=args.response,
        tts=args.tts,
        context=args.context,
    )
    orchestrator = await _create_orchestrator(config)
    try:
        speech_dir = Path(args.dir) if args.dir else None
        await orchestrator.run_full_pipeline(
            auto_save_audio=not args.no_save,
            speech_audio_dir=speech_dir,
            play_responses=not args.no_play,
        )
        return 0
    finally:
        await orchestrator.cleanup()


async def cmd_transcribe(args) -> int:
    config = await _build_config(
        env=args.env,
        transcription=args.transcription,
        response=args.response,
        tts=args.tts,
        context=args.context,
    )
    orchestrator = await _create_orchestrator(config)
    try:
        final_seen = 0
        async for item in orchestrator.run_transcription_only():
            if args.final_only and not item.is_final:
                continue
            prefix = "[FINAL]" if item.is_final else "[PARTIAL]"
            print(f"{prefix} {item.text}")
            if item.is_final:
                final_seen += 1
                if args.max_utterances and final_seen >= args.max_utterances:
                    break
        return 0
    finally:
        await orchestrator.cleanup()


async def cmd_respond(args) -> int:
    interactive_mode = not args.message

    config = await _build_config(
        env=args.env,
        transcription=args.transcription,
        response=args.response,
        tts=args.tts,
        context=args.context,
    )
    orchestrator = await _create_orchestrator(config)
    try:
        async def _stream_and_print(user_message: str) -> None:
            printed_stream = False
            async for chunk in orchestrator.run_response_only(
                user_message, use_context=not args.no_context
            ):
                if chunk.is_complete:
                    if not printed_stream and chunk.content:
                        print(chunk.content, end="", flush=True)
                    print()
                    break
                if chunk.content:
                    printed_stream = True
                    print(chunk.content, end="", flush=True)

        if interactive_mode:
            print("Interactive respond mode. Type your message and press Enter. Type /exit to quit.")
            loop = asyncio.get_event_loop()
            while True:
                try:
                    line = await loop.run_in_executor(None, input, ">> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                line = (line or "").strip()
                if not line:
                    continue
                if line in ("/exit", "/quit", ":q", "\u0004"):
                    break
                await _stream_and_print(line)
        else:
            await _stream_and_print(args.message)
        return 0
    finally:
        await orchestrator.cleanup()


async def cmd_tts(args) -> int:
    if not args.text:
        print("Error: --text is required for tts")
        return 2

    config = await _build_config(
        env=args.env,
        transcription=args.transcription,
        response=args.response,
        tts=args.tts,
        context=args.context,
    )
    orchestrator = await _create_orchestrator(config)
    try:
        save_path = Path(args.save) if args.save else None
        audio = await orchestrator.run_tts_only(
            args.text,
            voice=args.voice,
            speed=args.speed,
            pitch=args.pitch,
            save_path=save_path,
            play_audio=not args.no_play,
        )
        print(json.dumps({
            "format": audio.format.value,
            "sample_rate": audio.sample_rate,
            "size_mb": round(audio.get_size_mb(), 4),
            "voice": audio.voice,
            "language": audio.language,
        }, indent=2))
        return 0
    finally:
        await orchestrator.cleanup()


async def cmd_single(args) -> int:
    if not args.message:
        print("Error: --message is required for single")
        return 2

    config = await _build_config(
        env=args.env,
        transcription=args.transcription,
        response=args.response,
        tts=args.tts,
        context=args.context,
    )
    orchestrator = await _create_orchestrator(config)
    try:
        save_path = Path(args.save) if args.save else None
        result = await orchestrator.process_single_message(
            args.message,
            use_context=not args.no_context,
            generate_tts=not args.no_tts,
            play_audio=not args.no_play,
            save_path=save_path,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    finally:
        await orchestrator.cleanup()


async def cmd_status(args) -> int:
    config = await _build_config(
        env=args.env,
        transcription=args.transcription,
        response=args.response,
        tts=args.tts,
        context=args.context,
    )
    orchestrator = await _create_orchestrator(config)
    try:
        status = orchestrator.get_status()
        print(json.dumps(status, indent=2, default=str))
        return 0
    finally:
        await orchestrator.cleanup()


async def cmd_wakeword(args) -> int:
    config = await _build_config(
        env=args.env,
        transcription=args.transcription,
        response=args.response,
        tts=args.tts,
        context=args.context,
    )
    orchestrator = await _create_orchestrator(config)
    try:
        print("👂 Listening for wake word... Press Ctrl+C to stop.")
        async for event in orchestrator.run_wakeword_only():
            print(f"🔔 Wakeword: {event['model_name']} score={event['score']:.3f}")
            if args.once:
                break
        return 0
    finally:
        await orchestrator.cleanup()


async def cmd_wakeword_pipeline(args) -> int:
    config = await _build_config(
        env=args.env,
        transcription=args.transcription,
        response=args.response,
        tts=args.tts,
        context=args.context,
    )
    orchestrator = await _create_orchestrator(config)
    try:
        greeting_dir = Path(args.greeting_dir) if args.greeting_dir else None
        await orchestrator.run_wakeword_then_pipeline(
            greeting_dir=greeting_dir,
            auto_save_audio=not args.no_save,
            speech_audio_dir=Path(args.dir) if args.dir else None,
            play_responses=not args.no_play,
        )
        return 0
    finally:
        await orchestrator.cleanup()

async def cmd_list_providers(args) -> int:
    config = await _build_config(
        env=args.env,
        transcription=args.transcription,
        response=args.response,
        tts=args.tts,
        context=args.context,
    )
    orchestrator = await _create_orchestrator(config)
    try:
        providers = orchestrator.get_available_providers()
        print(json.dumps(providers, indent=2))
        return 0
    finally:
        await orchestrator.cleanup()


async def cmd_tools(args) -> int:
    config = await _build_config(
        env=args.env,
        transcription=args.transcription,
        response=args.response,
        tts=args.tts,
        context=args.context,
    )
    orchestrator = await _create_orchestrator(config)
    try:
        if not orchestrator.response:
            print("Response provider not available")
            return 1
        tools = await orchestrator.response.get_available_tools()
        print(json.dumps(tools, indent=2, default=str))
        return 0
    finally:
        await orchestrator.cleanup()


def _add_common_args(p):
    p.add_argument("--env", choices=["dev", "prod", "test", "default"], default="default",
                   help="Configuration profile to use (default comes from assistant_framework.config.CONFIG_PRESET)")
    p.add_argument("--transcription", help="Override transcription provider")
    p.add_argument("--response", help="Override response provider")
    p.add_argument("--tts", help="Override TTS provider")
    p.add_argument("--context", help="Override context provider")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="assistant_framework")
    sub = parser.add_subparsers(dest="command", required=True)

    # pipeline
    p_pipeline = sub.add_parser("pipeline", help="Run full pipeline (mic → response → TTS)")
    _add_common_args(p_pipeline)
    p_pipeline.add_argument("--no-play", action="store_true", help="Do not play TTS audio")
    p_pipeline.add_argument("--no-save", action="store_true", help="Do not auto-save audio files")
    p_pipeline.add_argument("--dir", help="Directory to save audio files (defaults to speech_audio)")
    p_pipeline.set_defaults(func=cmd_pipeline)

    # transcribe
    p_transcribe = sub.add_parser("transcribe", help="Run transcription only")
    _add_common_args(p_transcribe)
    p_transcribe.add_argument("--final-only", action="store_true", help="Only print final utterances")
    p_transcribe.add_argument("--max-utterances", type=int, help="Stop after N final utterances")
    p_transcribe.set_defaults(func=cmd_transcribe)

    # respond
    p_respond = sub.add_parser("respond", help="Run response only for a message")
    _add_common_args(p_respond)
    p_respond.add_argument("--message", help="Message to send to the assistant")
    p_respond.add_argument("--no-context", action="store_true", help="Disable context usage")
    p_respond.set_defaults(func=cmd_respond)

    # tts
    p_tts = sub.add_parser("tts", help="Synthesize speech for provided text")
    _add_common_args(p_tts)
    p_tts.add_argument("--text", help="Text to synthesize")
    p_tts.add_argument("--voice", help="Voice override")
    p_tts.add_argument("--speed", type=float, help="Speed override")
    p_tts.add_argument("--pitch", type=float, help="Pitch override (semitones)")
    p_tts.add_argument("--save", help="Path to save audio (e.g., speech_audio/out.mp3)")
    p_tts.add_argument("--no-play", action="store_true", help="Do not play TTS audio")
    p_tts.set_defaults(func=cmd_tts)

    # single
    p_single = sub.add_parser("single", help="Process a single message through the pipeline")
    _add_common_args(p_single)
    p_single.add_argument("--message", help="Message to process")
    p_single.add_argument("--no-context", action="store_true", help="Disable context usage")
    p_single.add_argument("--no-tts", action="store_true", help="Do not generate TTS for the response")
    p_single.add_argument("--no-play", action="store_true", help="Do not play TTS audio")
    p_single.add_argument("--save", help="Path to save audio output")
    p_single.set_defaults(func=cmd_single)

    # status
    p_status = sub.add_parser("status", help="Show orchestrator/provider status")
    _add_common_args(p_status)
    p_status.set_defaults(func=cmd_status)

    # list-providers
    p_list = sub.add_parser("list-providers", help="List available providers")
    _add_common_args(p_list)
    p_list.set_defaults(func=cmd_list_providers)

    # tools
    p_tools = sub.add_parser("tools", help="List available MCP tools (if configured)")
    _add_common_args(p_tools)
    p_tools.set_defaults(func=cmd_tools)

    # wakeword
    p_wake = sub.add_parser("wakeword", help="Run wake word detection only")
    _add_common_args(p_wake)
    p_wake.add_argument("--once", action="store_true", help="Exit after first detection")
    p_wake.set_defaults(func=cmd_wakeword)

    # wakeword+pipeline
    p_wake_pipe = sub.add_parser("wake-pipeline", help="Wait for wake word, play greeting, then start full pipeline")
    _add_common_args(p_wake_pipe)
    p_wake_pipe.add_argument("--greeting-dir", help="Directory of greeting audio (.mov or .mp3)")
    p_wake_pipe.add_argument("--no-play", action="store_true", help="Do not play TTS audio in pipeline")
    p_wake_pipe.add_argument("--no-save", action="store_true", help="Do not auto-save TTS audio in pipeline")
    p_wake_pipe.add_argument("--dir", help="Directory to save audio files (defaults to speech_audio)")
    p_wake_pipe.set_defaults(func=cmd_wakeword_pipeline)

    return parser


def main() -> int:
    # Optional: allow suppressing config summary via env
    if os.getenv("AF_QUIET_IMPORT") == "1":
        os.environ["QUIET_IMPORT"] = "1"

    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())


