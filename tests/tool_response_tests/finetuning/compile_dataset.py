#!/usr/bin/env python3
"""
Compile tool-calling fine-tuning data into one validated JSONL dataset.

This script treats the source JSONL files as natural-language -> tool-schema
training pairs, normalizes assistant outputs down to pure JSON arrays, validates
them against the current tool schemas, deduplicates examples, and writes a
single clean dataset for fine-tuning.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).parent
ASSETS_DIR = ROOT_DIR / "assets"
SOURCE_DIR = ROOT_DIR / "datasets" / "source"
BUILD_DIR = ROOT_DIR / "build"
DEFAULT_OUTPUT = BUILD_DIR / "tool_schema_finetuning.jsonl"


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


TOOL_SCHEMAS = load_json(ASSETS_DIR / "tool_schemas.json")
SYSTEM_PROMPT_TEMPLATE = (ASSETS_DIR / "system_prompt.txt").read_text()
SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.replace(
    "{tools}",
    json.dumps(TOOL_SCHEMAS["tools"], indent=2),
)
TOOL_SCHEMAS_BY_NAME = {tool["name"]: tool for tool in TOOL_SCHEMAS["tools"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile validated tool-calling fine-tuning data."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_DIR,
        help="Directory containing source JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path.",
    )
    return parser.parse_args()


def strip_thinking_trace(content: str) -> str:
    """Return the pure tool-call JSON payload from an assistant message."""
    marker = "</think>"
    if marker not in content:
        return content.strip()
    return content.split(marker, 1)[1].strip()


def validate_value_against_schema(value, prop_schema: dict, path: str) -> list[str]:
    errors: list[str] = []
    expected_type = prop_schema.get("type")

    if expected_type == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string, got {type(value).__name__}")
        elif "enum" in prop_schema and value not in prop_schema["enum"]:
            errors.append(f"{path}: '{value}' not in enum {prop_schema['enum']}")
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path}: expected integer, got {type(value).__name__}")
        else:
            minimum = prop_schema.get("minimum")
            maximum = prop_schema.get("maximum")
            if minimum is not None and value < minimum:
                errors.append(f"{path}: {value} < minimum {minimum}")
            if maximum is not None and value > maximum:
                errors.append(f"{path}: {value} > maximum {maximum}")
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: expected boolean, got {type(value).__name__}")
    elif expected_type == "array":
        if not isinstance(value, list):
            errors.append(f"{path}: expected array, got {type(value).__name__}")
        elif "items" in prop_schema:
            item_schema = prop_schema["items"]
            for index, item in enumerate(value):
                errors.extend(
                    validate_object_against_schema(item, item_schema, f"{path}[{index}]")
                )

    return errors


def validate_object_against_schema(
    obj: dict, schema: dict, path: str = ""
) -> list[str]:
    errors: list[str] = []

    if schema.get("type") == "object" and not isinstance(obj, dict):
        return [f"{path}: expected object, got {type(obj).__name__}"]

    if not isinstance(obj, dict):
        return errors

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for required_key in required:
        if required_key not in obj:
            errors.append(f"{path}: missing required param '{required_key}'")

    for key, value in obj.items():
        if key not in properties:
            errors.append(f"{path}: unknown param '{key}'")
            continue
        errors.extend(
            validate_value_against_schema(value, properties[key], f"{path}.{key}")
        )

    return errors


def validate_tool_call(tool_call: dict) -> list[str]:
    errors: list[str] = []

    if not isinstance(tool_call, dict):
        return [f"tool call must be an object, got {type(tool_call).__name__}"]

    name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})

    if not isinstance(name, str):
        return ["tool call missing string 'name'"]
    if not isinstance(arguments, dict):
        return [f"[{name}] arguments must be an object"]

    schema = TOOL_SCHEMAS_BY_NAME.get(name)
    if not schema:
        return [f"unknown tool '{name}'"]

    params_schema = schema.get("parameters", {})
    required = params_schema.get("required", [])
    properties = params_schema.get("properties", {})

    for required_key in required:
        if required_key not in arguments:
            errors.append(f"[{name}] missing required param '{required_key}'")

    for key, value in arguments.items():
        if key not in properties:
            errors.append(f"[{name}] unknown param '{key}'")
            continue
        errors.extend(
            validate_value_against_schema(value, properties[key], f"[{name}].{key}")
        )

    one_of = params_schema.get("oneOf", [])
    if one_of:
        matches = 0
        for constraint in one_of:
            required_group = constraint.get("required", [])
            if all(item in arguments for item in required_group):
                matches += 1
        if matches == 0:
            groups = [constraint.get("required", []) for constraint in one_of]
            errors.append(f"[{name}] must satisfy one of: {groups}")
        elif matches > 1:
            errors.append(
                f"[{name}] satisfies multiple oneOf constraints (should be exactly one)"
            )

    return errors


def _has_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return True


def validate_tool_semantics(tool_call: dict) -> list[str]:
    """Apply action-specific validation that JSON schema alone does not capture."""
    errors: list[str] = []
    name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})

    if not isinstance(arguments, dict):
        return errors

    if name == "spotify_playback":
        action = arguments.get("action")
        if action in {"search_track", "search_artist"} and not _has_value(
            arguments.get("query")
        ):
            errors.append(f"[{name}] action '{action}' requires a non-empty query")
        if action == "volume" and "volume_level" not in arguments:
            errors.append("[spotify_playback] action 'volume' requires volume_level")
        if action == "play" and "search_type" in arguments and not _has_value(
            arguments.get("query")
        ):
            errors.append(
                "[spotify_playback] search_type should only be set when play includes a query"
            )

    elif name == "kasa_lighting":
        interaction = arguments.get("interaction")
        if interaction == "direct":
            missing = [
                field for field in ("action", "light_name") if not _has_value(arguments.get(field))
            ]
            if missing:
                errors.append(
                    "[kasa_lighting] direct interaction requires "
                    + ", ".join(missing)
                )
        elif interaction == "scene":
            if not _has_value(arguments.get("scene_name")):
                errors.append("[kasa_lighting] scene interaction requires scene_name")
            has_room = _has_value(arguments.get("room"))
            has_light_names = _has_value(arguments.get("light_names"))
            if not has_room and not has_light_names:
                errors.append(
                    "[kasa_lighting] scene interaction requires either room or light_names"
                )

    elif name == "calendar_data":
        commands = arguments.get("commands", [])
        if not isinstance(commands, list):
            return errors
        for command in commands:
            if not isinstance(command, dict):
                continue
            read_or_write = command.get("read_or_write")
            if not _has_value(read_or_write):
                errors.append("[calendar_data] command requires read_or_write")
                continue
            if read_or_write == "read":
                if not _has_value(command.get("calendar")):
                    errors.append("[calendar_data] read commands require calendar")
                if not _has_value(command.get("read_type")):
                    errors.append("[calendar_data] read commands require read_type")
                if command.get("read_type") == "specific_date" and not _has_value(
                    command.get("date")
                ):
                    errors.append(
                        "[calendar_data] specific_date reads require a non-empty date"
                    )
            elif read_or_write == "create_event":
                has_calendar = _has_value(command.get("calendar"))
                has_calendars = _has_value(command.get("calendars"))
                if not has_calendar and not has_calendars:
                    errors.append(
                        "[calendar_data] create_event requires calendar or calendars"
                    )
                if has_calendar and has_calendars:
                    errors.append(
                        "[calendar_data] create_event should use either calendar or calendars, not both"
                    )

    elif name == "todos":
        action = arguments.get("action")
        if action == "create":
            if not any(_has_value(arguments.get(field)) for field in ("title", "message")):
                errors.append("[todos] create requires title or message")
            has_event_time = _has_value(arguments.get("event_time"))
            has_remind_before = arguments.get("remind_before_minutes") is not None
            if has_event_time and not has_remind_before:
                errors.append(
                    "[todos] event_time reminders require remind_before_minutes"
                )
            if has_remind_before and not has_event_time:
                errors.append(
                    "[todos] remind_before_minutes requires event_time"
                )
        elif action in {"complete", "reopen", "update", "delete"}:
            if not any(_has_value(arguments.get(field)) for field in ("todo_id", "match")):
                errors.append(f"[todos] action '{action}' requires todo_id or match")
            if action == "update":
                has_field_change = any(
                    _has_value(arguments.get(field))
                    for field in ("title", "message", "details", "due_at")
                ) or bool(arguments.get("clear_due_at"))
                if not has_field_change:
                    errors.append(
                        "[todos] update requires at least one field change"
                    )

    elif name == "briefing":
        if arguments.get("action") == "dismiss" and not _has_value(
            arguments.get("briefing_id")
        ):
            errors.append("[briefing] dismiss requires briefing_id")

    elif name == "read_clipboard":
        max_length = arguments.get("max_length")
        if max_length is not None and max_length <= 0:
            errors.append("[read_clipboard] max_length must be positive")

    elif name == "get_notifications":
        limit = arguments.get("limit")
        if limit is not None and limit <= 0:
            errors.append("[get_notifications] limit must be positive")

    return errors


def validate_tool_sequence(tool_calls: list[dict]) -> list[str]:
    """Apply runtime-aware validation across a full assistant tool payload."""
    errors: list[str] = []
    seen_signatures: set[str] = set()
    google_search_calls = 0

    for tool_call in tool_calls:
        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        signature = json.dumps(tool_call, separators=(",", ":"), sort_keys=True)

        if signature in seen_signatures:
            errors.append(f"[{name}] duplicate tool call in same assistant turn")
        else:
            seen_signatures.add(signature)

        if name == "google_search":
            google_search_calls += 1
            if google_search_calls > 1:
                errors.append("[google_search] only one google_search call is allowed per assistant turn")

        if name != "calendar_data":
            continue

        commands = arguments.get("commands", [])
        if not isinstance(commands, list):
            continue

        if len(commands) > 1:
            errors.append("[calendar_data] only one command is allowed per tool call")

        for command in commands:
            if not isinstance(command, dict):
                continue
            if command.get("read_or_write") != "create_event":
                continue

            missing_fields = [
                field
                for field in ("event_title", "date", "start_time", "end_time")
                if not command.get(field)
            ]
            if missing_fields:
                errors.append(
                    "[calendar_data] create_event examples must include "
                    + ", ".join(missing_fields)
                )

            if command.get("calendar") == "all":
                errors.append("[calendar_data] create_event cannot use calendar='all'")

    return errors


def canonicalize_assistant_content(content: str) -> tuple[str | None, list[str]]:
    raw_payload = strip_thinking_trace(content)

    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        return None, [f"assistant content is not valid JSON: {exc.msg}"]

    if not isinstance(parsed, list):
        return None, ["assistant content must be a JSON array of tool calls"]

    errors: list[str] = []
    for tool_call in parsed:
        errors.extend(validate_tool_call(tool_call))
        errors.extend(validate_tool_semantics(tool_call))
    errors.extend(validate_tool_sequence(parsed))

    if errors:
        return None, errors

    return json.dumps(parsed, separators=(",", ":")), []


def canonicalize_example(example: dict) -> tuple[dict | None, list[str]]:
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, ["example is missing a messages list"]

    normalized_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    errors: list[str] = []
    saw_user = False

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if not isinstance(role, str) or not isinstance(content, str):
            errors.append("message role/content must both be strings")
            continue

        if role == "system":
            continue
        if role == "user":
            saw_user = True
            normalized_messages.append({"role": "user", "content": content.strip()})
            continue
        if role == "assistant":
            normalized_content, message_errors = canonicalize_assistant_content(content)
            if message_errors:
                errors.extend(message_errors)
                continue
            normalized_messages.append(
                {"role": "assistant", "content": normalized_content}
            )
            continue

        errors.append(f"unsupported role '{role}'")

    if errors:
        return None, errors
    if not saw_user:
        return None, ["example has no user messages"]

    return {"messages": normalized_messages}, []


def example_signature(example: dict) -> str:
    return json.dumps(example["messages"], separators=(",", ":"), sort_keys=True)


def iter_source_files(source_dir: Path) -> list[Path]:
    return sorted(
        path for path in source_dir.rglob("*.jsonl") if path.is_file()
    )


def read_examples(path: Path) -> list[dict]:
    examples: list[dict] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                examples.append(json.loads(raw_line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} line {line_number}: invalid JSON ({exc.msg})"
                ) from exc
    return examples


def compile_dataset(source_dir: Path) -> tuple[list[dict], dict]:
    source_files = iter_source_files(source_dir)
    if not source_files:
        raise FileNotFoundError(f"No JSONL source files found in {source_dir}")

    compiled_examples: list[dict] = []
    seen_signatures: set[str] = set()
    invalid_details: list[tuple[Path, int, list[str]]] = []
    duplicate_count = 0
    tool_counter: Counter[str] = Counter()

    for source_path in source_files:
        raw_examples = read_examples(source_path)
        print(f"Reading {source_path.relative_to(ROOT_DIR)}: {len(raw_examples)} examples")

        for index, example in enumerate(raw_examples, start=1):
            normalized_example, errors = canonicalize_example(example)
            if errors:
                invalid_details.append((source_path, index, errors))
                continue

            signature = example_signature(normalized_example)
            if signature in seen_signatures:
                duplicate_count += 1
                continue

            seen_signatures.add(signature)
            compiled_examples.append(normalized_example)

            for message in normalized_example["messages"]:
                if message["role"] != "assistant":
                    continue
                for tool_call in json.loads(message["content"]):
                    tool_counter[tool_call["name"]] += 1

    stats = {
        "source_files": len(source_files),
        "valid_examples": len(compiled_examples),
        "invalid_examples": len(invalid_details),
        "duplicate_examples": duplicate_count,
        "tool_counts": dict(tool_counter),
        "invalid_details": invalid_details,
    }
    return compiled_examples, stats


def write_output(examples: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example) + "\n")


def print_summary(output_path: Path, stats: dict) -> None:
    print("\nCompilation summary:")
    print(f"  Source files: {stats['source_files']}")
    print(f"  Valid examples: {stats['valid_examples']}")
    print(f"  Skipped invalid: {stats['invalid_examples']}")
    print(f"  Skipped duplicates: {stats['duplicate_examples']}")
    print(f"  Output: {output_path}")

    if stats["tool_counts"]:
        print("\nTool distribution:")
        for name, count in sorted(
            stats["tool_counts"].items(), key=lambda item: (-item[1], item[0])
        ):
            print(f"  {name}: {count}")

    if stats["invalid_details"]:
        print("\nInvalid examples:")
        for source_path, index, errors in stats["invalid_details"][:20]:
            joined_errors = "; ".join(errors)
            print(f"  {source_path.relative_to(ROOT_DIR)} #{index}: {joined_errors}")
        remaining = len(stats["invalid_details"]) - 20
        if remaining > 0:
            print(f"  ... and {remaining} more")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_path = args.output.resolve()

    compiled_examples, stats = compile_dataset(source_dir)
    write_output(compiled_examples, output_path)
    print_summary(output_path, stats)


if __name__ == "__main__":
    main()
