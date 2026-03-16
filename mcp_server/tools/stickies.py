"""
Stickies Note Tool using BaseTool.

Safely reads and edits the user's single macOS Stickies note with defense-in-depth
security: only files within the Stickies sandbox can be touched; edits are validated
and rolled back on failure.

The sticky note ID is cached in app_state for fast subsequent lookups.
"""

import os
import re
import subprocess
import tempfile
import plistlib
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from mcp_server.base_tool import BaseTool
from mcp_server.config import LOG_TOOLS
from mcp_server.state_manager import StateManager


# -----------------------------------------------------------------------------
# Security constants (Layer 1: hardcoded sandbox path)
# -----------------------------------------------------------------------------
STICKIES_DIR = Path.home() / "Library/Containers/com.apple.Stickies/Data/Library/Stickies"
UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
MAX_CONTENT_SIZE = 100_000  # 100KB limit
RTF_HEADER = "{\\rtf1"
RTF_FOOTER = "}"


def _has_dangerous_chars(s: str) -> bool:
    """Layer 3: Path traversal prevention - reject .. / \\ null."""
    if "\x00" in s or ".." in s or "/" in s or "\\" in s:
        return True
    return False


def _validate_rtf(content: str) -> bool:
    """Layer 7: RTF content validation."""
    if not content or len(content) > MAX_CONTENT_SIZE:
        return False
    stripped = content.strip()
    if not stripped.startswith(RTF_HEADER):
        return False
    if not stripped.endswith(RTF_FOOTER):
        return False
    return True


def _rtf_escape(text: str) -> str:
    """Escape RTF special characters in a text fragment."""
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


# ---- RTF building blocks (structure is code-controlled, not LLM-controlled) ----

_RTF_HEADER = (
    "{\\rtf1\\ansi\\ansicpg1252\\cocoartf2639\n"
    "{\\fonttbl{\\f0\\fswiss\\fcharset0 Helvetica;}}\n"
    "{\\colortbl;\\red0\\green0\\blue0;}\n"
    "\\pard\\f0\\fs24\\cf1\n"
)
_RTF_FOOTER = "}"


def _rtf_heading(text: str) -> str:
    """Bold 16pt heading line."""
    return f"\\b\\fs32 {_rtf_escape(text)}\\b0\\fs24\\par\n"


def _rtf_subheading(text: str) -> str:
    """Bold 13pt subheading line."""
    return f"\\b\\fs26 {_rtf_escape(text)}\\b0\\fs24\\par\n"


def _rtf_body(text: str) -> str:
    """Normal 12pt body line."""
    return f"{_rtf_escape(text)}\\par\n"


def _rtf_bullet(text: str) -> str:
    """Indented dash-bullet line."""
    return f"    - {_rtf_escape(text)}\\par\n"


def _rtf_blank() -> str:
    """Empty paragraph (vertical space)."""
    return "\\par\n"


def _build_rtf(notes_lines: list, todo_items: list) -> str:
    """Build a complete RTF document from structured note data.

    All headings, spacing, and bullet formatting are handled here —
    the LLM only provides raw text content.

    Layout:
        **My Notes**          (bold heading, always present)
        <blank line>
        **Subheading**        (bold subhead, per entry)
        Body text...          (normal)
        <blank line>
        **To Do**             (bold heading, always present)
        - item 1
        - item 2
    """
    parts: list[str] = [_RTF_HEADER]

    # ── Notes section ──
    parts.append(_rtf_heading("My Notes"))
    parts.append(_rtf_blank())

    for line in notes_lines:
        stripped = line.strip()
        if not stripped:
            parts.append(_rtf_blank())
        elif stripped.startswith("## "):
            # Subheading stored from previous edits
            parts.append(_rtf_subheading(stripped[3:].strip()))
        elif stripped.startswith("# "):
            # Skip any stale top-level headings (we already printed "My Notes")
            heading_text = stripped[2:].strip().lower()
            if heading_text in ("my notes", "notes"):
                continue
            parts.append(_rtf_subheading(stripped[2:].strip()))
        else:
            parts.append(_rtf_body(stripped))

    # ── Divider ──
    parts.append(_rtf_blank())

    # ── To Do section ──
    parts.append(_rtf_heading("To Do List"))
    if todo_items:
        for item in todo_items:
            parts.append(_rtf_bullet(item))
    else:
        parts.append(_rtf_body("(no items)"))

    parts.append(_RTF_FOOTER)
    return "".join(parts)


class StickiesTool(BaseTool):
    """Tool for reading and writing the user's desktop sticky note (macOS Stickies app)."""

    name = "stickies"
    description = (
        "Manage the user's desktop sticky note (macOS Stickies app). "
        "Use this tool when the user asks about their notes, to-do list, sticky note, "
        "desktop notes, or anything they've written down.\n"
        "NOTE STRUCTURE (validated by tool):\n"
        "  H1: My Notes\n"
        "  H2: Topic/Subheading lines (notes section)\n"
        "  Paragraph body under each H2\n"
        "  H1: To Do List\n"
        "  Paragraph lines for each to-do item (dash list)\n"
        "ACTIONS:\n"
        "  'read' — get note content. Use section param: 'notes', 'todo', or 'both' (default).\n"
        "  'write' — provide full RTF content in 'rtf_content'. The tool validates structure and writes it.\n"
        "    (Legacy) You can also provide structured edits via 'edits' array.\n"
        "This is NOT for calendar events - use calendar_data for that."
    )
    version = "4.2.0"

    def __init__(self):
        """Initialize the Stickies tool."""
        super().__init__()
        self._is_macos = platform.system() == "Darwin"
        self._state_manager = StateManager()

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write"],
                    "description": "'read' returns note content; 'write' applies structured edits (never wipes the note).",
                },
                "section": {
                    "type": "string",
                    "enum": ["notes", "todo", "both"],
                    "description": "For read: which section to return. 'notes' = top half, 'todo' = bottom half, 'both' = full note (default).",
                    "default": "both",
                },
                "rtf_content": {
                    "type": "string",
                    "description": (
                        "For write: full RTF content of the sticky note. The tool validates structure.\n"
                        "Must include H1 headings for 'My Notes' and 'To Do List', H2 headings for note topics,\n"
                        "and paragraph content under each H2."
                    ),
                },
                "edits": {
                    "type": "array",
                    "description": (
                        "For write (legacy): array of edit objects. EACH object MUST have 'op' field.\n"
                        "Format: [{\"op\":\"remove_todo\",\"match\":\"text\"}] NOT [{\"remove_todo\":{\"match\":\"text\"}}]\n"
                        "Ops: add_todo, remove_todo, edit_todo, add_note, remove_note, edit_note."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": ["add_todo", "remove_todo", "edit_todo", "add_note", "remove_note", "edit_note"],
                                "description": "The edit operation type.",
                            },
                            "item": {
                                "type": "string",
                                "description": "For add_todo: the task text. Just plain text, no formatting.",
                            },
                            "due": {
                                "type": "string",
                                "description": "For add_todo: optional due date (e.g., 'Feb 12', 'Monday').",
                            },
                            "match": {
                                "type": "string",
                                "description": "For remove_todo/remove_note: text to match (case-insensitive, partial match).",
                            },
                            "old": {
                                "type": "string",
                                "description": "For edit_todo/edit_note: existing text to find (case-insensitive).",
                            },
                            "new": {
                                "type": "string",
                                "description": "For edit_todo/edit_note: replacement text.",
                            },
                            "subheading": {
                                "type": "string",
                                "description": "For add_note: REQUIRED subheading/topic for the note. Notes must follow subheading → content structure. Plain text only.",
                            },
                            "content": {
                                "type": "string",
                                "description": "For add_note: REQUIRED paragraph content under the subheading. Plain text only.",
                            },
                        },
                        "required": ["op"],
                    },
                },
            },
            "required": ["action"],
        }

    # -------------------------------------------------------------------------
    # Find the single sticky note (with app_state caching)
    # -------------------------------------------------------------------------
    def _find_sticky_id(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the single sticky note UUID.
        
        First checks app_state for cached sticky_id. If found and valid, uses it.
        Otherwise searches .SavedStickiesState and caches the result in app_state.
        
        Returns (sticky_id, error_code). Returns the first valid sticky found.
        """
        # Check app_state cache first
        cached_id = self._state_manager.get("sticky_note_id")
        if cached_id:
            # Verify it still exists on disk
            rtfd_path = STICKIES_DIR / (cached_id + ".rtfd")
            if rtfd_path.is_dir():
                self.logger.debug(f"Using cached sticky_id from app_state: {cached_id}")
                return cached_id, None
            else:
                # Cached ID is stale, clear it
                self.logger.info(f"Cached sticky_id {cached_id} no longer exists, searching...")
                self._state_manager.set("sticky_note_id", None)

        # Search for sticky note in .SavedStickiesState
        state_file = STICKIES_DIR / ".SavedStickiesState"
        if not state_file.exists() or not state_file.is_file():
            return None, "STICKIES_NOT_FOUND"

        try:
            with open(state_file, "rb") as f:
                state = plistlib.load(f)
        except (plistlib.InvalidFileException, OSError):
            return None, "STICKIES_NOT_FOUND"

        if not isinstance(state, list):
            return None, "STICKIES_NOT_FOUND"

        for item in state:
            if not isinstance(item, dict):
                continue
            uuid_val = item.get("UUID")
            if not uuid_val or not isinstance(uuid_val, str):
                continue
            if _has_dangerous_chars(uuid_val):
                continue
            if not UUID_PATTERN.match(uuid_val.strip()):
                continue
            # Verify the .rtfd directory exists
            rtfd_path = STICKIES_DIR / (uuid_val + ".rtfd")
            if rtfd_path.is_dir():
                # Cache in app_state for future lookups
                self._state_manager.set("sticky_note_id", uuid_val)
                self.logger.info(f"Found and cached sticky_id in app_state: {uuid_val}")
                return uuid_val, None

        return None, "STICKIES_NOT_FOUND"

    # -------------------------------------------------------------------------
    # Security: Layer 4-5 - Canonical path and structure validation
    # -------------------------------------------------------------------------
    def _get_safe_path(self, sticky_id: str) -> Tuple[Optional[Path], Optional[str]]:
        """
        Resolve sticky_id to RTF path only if it lies inside Stickies sandbox.
        Returns (rtf_path, error_code). rtf_path is the path to TXT.rtf inside the .rtfd bundle.
        """
        # Build path: STICKIES_DIR / "{uuid}.rtfd" / "TXT.rtf"
        rtfd_name = sticky_id.strip() + ".rtfd"
        rtfd_path = STICKIES_DIR / rtfd_name
        rtf_path = rtfd_path / "TXT.rtf"

        try:
            # Resolve to canonical path (follows symlinks - we check containment next)
            canon_stickies = STICKIES_DIR.resolve()
            if not canon_stickies.exists():
                return None, "STICKIES_NOT_FOUND"

            canon_rtfd = rtfd_path.resolve()
            # Must be inside Stickies directory (catches symlinks pointing out)
            try:
                canon_rtfd.relative_to(canon_stickies)
            except ValueError:
                return None, "SECURITY_OUTSIDE_SANDBOX"

            # Layer 5: must be a directory, not a symlink we shouldn't follow
            if not canon_rtfd.is_dir():
                return None, "SECURITY_INVALID_STRUCTURE"
            # TXT.rtf must exist and be a file
            if not rtf_path.exists() or not rtf_path.is_file():
                return None, "SECURITY_INVALID_STRUCTURE"

            return rtf_path, None
        except OSError:
            return None, "STICKIES_NOT_FOUND"

    # -------------------------------------------------------------------------
    # Security: Layer 6 - Backup and atomic write
    # -------------------------------------------------------------------------
    def _create_backup(self, rtf_path: Path) -> Optional[Path]:
        """Create a backup of the RTF file. Returns backup path or None on failure."""
        try:
            content = rtf_path.read_bytes()
            fd, backup_path = tempfile.mkstemp(suffix=".stickies_backup", prefix="stickies_")
            os.close(fd)
            Path(backup_path).write_bytes(content)
            return Path(backup_path)
        except OSError:
            return None

    def _atomic_write(
        self, rtf_path: Path, content: str, backup_path: Optional[Path]
    ) -> Tuple[bool, Optional[str]]:
        """Write content via temp file and atomic rename. Restore from backup on failure."""
        if not _validate_rtf(content):
            return False, "SECURITY_RTF_INVALID"

        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".rtf", dir=rtf_path.parent)
            tmp_file = Path(tmp_path)
            try:
                os.write(tmp_fd, content.encode("utf-8"))
                os.close(tmp_fd)
                tmp_fd = -1
                # Atomic replace
                os.replace(tmp_file, rtf_path)
                return True, None
            finally:
                if tmp_fd >= 0:
                    try:
                        os.close(tmp_fd)
                    except OSError:
                        pass
                if tmp_file.exists():
                    try:
                        tmp_file.unlink()
                    except OSError:
                        pass
        except OSError as e:
            self.logger.warning("Stickies write failed, restoring backup: %s", e)
            if backup_path and backup_path.exists():
                try:
                    rtf_path.write_bytes(backup_path.read_bytes())
                except OSError:
                    pass
            return False, "STICKIES_WRITE_FAILED"
        finally:
            if backup_path and backup_path.exists():
                try:
                    backup_path.unlink()
                except OSError:
                    pass

    # -------------------------------------------------------------------------
    # Actions: read, write
    # -------------------------------------------------------------------------
    def _action_read(self, sticky_id: str, section: str = "both") -> Dict[str, Any]:
        """Read content of the sticky note, optionally filtering by section.
        
        Args:
            sticky_id: The sticky note UUID
            section: 'notes' (top half), 'todo' (bottom half), or 'both' (full note)
        """
        rtf_path, err = self._get_safe_path(sticky_id)
        if err:
            return _error_response(err)
        assert rtf_path is not None

        try:
            raw = rtf_path.read_text(encoding="utf-8", errors="replace")
            # Strip RTF for a simple plain-text approximation (full RTF parse would be heavier)
            plain = self._rtf_to_plain_approx(raw)
            
            # Filter by section if requested
            if section in ("notes", "todo"):
                notes_content, todo_content = self._split_sections(plain)
                if section == "notes":
                    return {
                        "success": True,
                        "section": "notes",
                        "content": notes_content or "(no notes section found)",
                    }
                else:  # todo
                    return {
                        "success": True,
                        "section": "todo",
                        "content": todo_content or "(no to-do section found)",
                    }
            
            # Default: return both sections
            return {
                "success": True,
                "section": "both",
                "content": plain,
            }
        except OSError as e:
            self.logger.warning("Stickies read failed: %s", e)
            return _error_response("STICKIES_NOT_FOUND")

    def _split_sections(self, content: str) -> Tuple[str, str]:
        """Split note content into notes section (top) and to-do section (bottom).
        
        Looks for 'To Do', 'To Do List', or markdown-prefixed versions as the section divider.
        Returns (notes_content, todo_content).
        """
        # Try to find the To Do section marker (case-insensitive)
        import re
        # Match various forms: "To Do", "To Do List", "# To Do", "## To Do List", etc.
        match = re.search(r"(?m)^#{0,3}\s*To\s*Do(\s*List)?\s*$", content, re.IGNORECASE)
        
        if match:
            notes_part = content[:match.start()].strip()
            todo_part = content[match.start():].strip()
            return notes_part, todo_part
        
        # No To Do section found — treat entire content as notes
        return content.strip(), ""

    def _rtf_to_plain_approx(self, rtf: str) -> str:
        """Convert RTF to plain text with subheading markers for internal use.
        
        Bold text (fs26) on its own line becomes "## Subheading" for internal parsing.
        Main headings (My Notes, To Do List) are not marked with ##.
        """
        _SKIP_GROUPS = {"fonttbl", "colortbl", "stylesheet", "info", "expandedcolortbl"}
        _MAIN_HEADINGS = {"my notes", "notes", "to do", "todo", "to do list"}
        
        # First pass: extract text with bold markers
        out: list[str] = []
        i = 0
        skip_depth = 0
        in_bold = False
        
        while i < len(rtf):
            ch = rtf[i]
            
            if ch == "{":
                if skip_depth > 0:
                    skip_depth += 1
                    i += 1
                    continue
                j = i + 1
                if j < len(rtf) and rtf[j] == "\\":
                    j += 1
                    word_start = j
                    while j < len(rtf) and rtf[j].isalpha():
                        j += 1
                    word = rtf[word_start:j]
                    if word in _SKIP_GROUPS:
                        skip_depth = 1
                        i = j
                        continue
                i += 1
                continue
            
            if ch == "}":
                if skip_depth > 0:
                    skip_depth -= 1
                i += 1
                continue
            
            if skip_depth > 0:
                i += 1
                continue
            
            if ch == "\\":
                i += 1
                if i >= len(rtf):
                    break
                if rtf[i] in "{}":
                    out.append(rtf[i])
                    i += 1
                    continue
                if rtf[i] == "\\":
                    out.append("\\")
                    i += 1
                    continue
                word_start = i
                while i < len(rtf) and rtf[i].isalpha():
                    i += 1
                word = rtf[word_start:i]
                # Capture numeric parameter
                param_start = i
                while i < len(rtf) and (rtf[i].isdigit() or rtf[i] == "-"):
                    i += 1
                param = rtf[param_start:i] if param_start < i else ""
                if i < len(rtf) and rtf[i] == " ":
                    i += 1
                # Track bold state - \b turns on, \b0 turns off
                if word == "b":
                    if param == "0":
                        # \b0 = bold off
                        in_bold = False
                        out.append("\x03")  # Bold end marker
                    else:
                        # \b = bold on
                        in_bold = True
                        out.append("\x02")  # Bold start marker
                elif word in ("line", "par"):
                    out.append("\n")
                elif word == "tab":
                    out.append("\t")
                continue
            
            out.append(ch)
            i += 1
        
        # Second pass: process lines and mark subheadings
        raw_text = "".join(out)
        lines = raw_text.split("\n")
        result: list[str] = []
        
        for line in lines:
            stripped = line.strip()
            # Check if line has bold marker at start
            if stripped.startswith("\x02"):
                end_idx = stripped.find("\x03")
                if end_idx > 0:
                    bold_text = stripped[1:end_idx].strip()
                    rest = stripped[end_idx+1:].strip()
                    # If bold-only line (no text after) and not a main heading, mark as subheading
                    if bold_text.lower() not in _MAIN_HEADINGS and not rest:
                        result.append(f"## {bold_text}")
                    else:
                        result.append(bold_text + (" " + rest if rest else ""))
                else:
                    bold_text = stripped.replace("\x02", "").replace("\x03", "").strip()
                    if bold_text.lower() not in _MAIN_HEADINGS:
                        result.append(f"## {bold_text}")
                    else:
                        result.append(bold_text)
            else:
                clean = stripped.replace("\x02", "").replace("\x03", "")
                result.append(clean)
        
        # Collapse multiple blank lines and filter artifacts
        cleaned: list[str] = []
        blank_count = 0
        # Known artifacts to filter out (exact line matches)
        artifact_lines = {"*;;", ";;", "*;", ";*", "Helvetica;", "Helvetica; ;;"}
        
        for line in result:
            # Skip known artifact lines
            if line.strip() in artifact_lines:
                continue
            if not line:
                blank_count += 1
                if blank_count <= 1:
                    cleaned.append("")
            else:
                blank_count = 0
                cleaned.append(line)
        
        # Final cleanup: remove any remaining artifacts from the text
        output = "\n".join(cleaned).strip()
        # Remove artifact patterns that might appear inline or with whitespace
        for artifact in ["*;;", ";;", "*;", ";*"]:
            output = output.replace(artifact, "")
        # Clean up any resulting double newlines from removal
        while "\n\n\n" in output:
            output = output.replace("\n\n\n", "\n\n")
        
        return output.strip()

    # -------------------------------------------------------------------------
    # Structured edit engine
    # -------------------------------------------------------------------------
    def _action_write(self, sticky_id: str, edits: list) -> Dict[str, Any]:
        """Apply structured edits to the sticky note without wiping it.

        1. Reads current content and splits into notes / todo sections.
        2. Applies each edit operation in order.
        3. Reconstructs the full note and writes it back.
        """
        rtf_path, err = self._get_safe_path(sticky_id)
        if err:
            return _error_response(err)
        assert rtf_path is not None

        # Read current content
        try:
            raw = rtf_path.read_text(encoding="utf-8", errors="replace")
            plain = self._rtf_to_plain_approx(raw)
        except OSError:
            plain = ""

        # Split into sections
        notes_lines, todo_items = self._parse_note(plain)

        # Apply each edit
        applied: list[str] = []
        errors: list[str] = []

        for i, edit in enumerate(edits):
            if not isinstance(edit, dict):
                errors.append(f"Edit {i}: not a valid object")
                continue
            op = (edit.get("op") or "").strip().lower()
            try:
                result = self._apply_edit(op, edit, notes_lines, todo_items)
                applied.append(result)
            except ValueError as e:
                errors.append(f"Edit {i} ({op}): {e}")

        # Build RTF directly from structured data (code controls all formatting)
        to_write = _build_rtf(notes_lines, todo_items)

        # Plain-text summary for the API response
        new_content = self._reconstruct_note_plain(notes_lines, todo_items)
        if len(to_write) > MAX_CONTENT_SIZE:
            return _error_response("SECURITY_RTF_INVALID")

        backup_path = self._create_backup(rtf_path)
        ok, write_err = self._atomic_write(rtf_path, to_write, backup_path)
        if not ok:
            return _error_response(write_err or "STICKIES_WRITE_FAILED")

        # Refresh Stickies app
        self._refresh_stickies_app()

        response: Dict[str, Any] = {
            "success": True,
            "edits_applied": applied,
            "updated_content": new_content,
        }
        if errors:
            response["edit_errors"] = errors
        return response

    def _validate_rtf_structure(self, rtf: str) -> Optional[str]:
        """Validate that the RTF follows the required structure."""
        if "\\rtf1" not in rtf:
            return "RTF header missing"

        my_notes_match = re.search(r"\\b\\fs32\\s+My\\s+Notes\\b0\\fs24\\par", rtf, re.IGNORECASE)
        todo_match = re.search(r"\\b\\fs32\\s+To\\s+Do(\\s+List)?\\b0\\fs24\\par", rtf, re.IGNORECASE)
        if not my_notes_match:
            return "Missing H1 heading: My Notes"
        if not todo_match:
            return "Missing H1 heading: To Do List"

        notes_section = rtf[my_notes_match.end():todo_match.start()]
        todo_section = rtf[todo_match.end():]

        # Require at least one H2 subheading
        subheadings = list(re.finditer(r"\\b\\fs26\\s+(.+?)\\b0\\fs24\\par", notes_section, re.IGNORECASE))
        if not subheadings:
            return "Notes section must include at least one H2 subheading"

        # Each subheading must have a non-bold body line after it
        for idx, match in enumerate(subheadings):
            start = match.end()
            end = subheadings[idx + 1].start() if idx + 1 < len(subheadings) else len(notes_section)
            segment = notes_section[start:end]
            body_line = re.search(r"(?m)^(?!.*\\b\\fs26)(?!.*\\b\\fs32).+?\\par", segment)
            if not body_line:
                return "Each H2 subheading must have paragraph content under it"

        # Require at least one to-do item
        if not re.search(r"\\n\\s*-\\s+.+?\\par", todo_section):
            return "To Do List must include at least one dash item"

        return None

    def _action_write_rtf(self, sticky_id: str, rtf_content: str) -> Dict[str, Any]:
        """Write full RTF content after validating structure."""
        rtf_path, err = self._get_safe_path(sticky_id)
        if err:
            return _error_response(err)
        assert rtf_path is not None

        validation_error = self._validate_rtf_structure(rtf_content)
        if validation_error:
            return {
                "success": False,
                "error": validation_error,
                "error_code": "INVALID_RTF_STRUCTURE",
            }

        if len(rtf_content) > MAX_CONTENT_SIZE:
            return _error_response("SECURITY_RTF_INVALID")

        backup_path = self._create_backup(rtf_path)
        ok, write_err = self._atomic_write(rtf_path, rtf_content, backup_path)
        if not ok:
            return _error_response(write_err or "STICKIES_WRITE_FAILED")

        self._refresh_stickies_app()
        return {
            "success": True,
            "edits_applied": [],
            "updated_content": self._rtf_to_plain_approx(rtf_content),
        }

    def _parse_note(self, content: str) -> Tuple[list, list]:
        """Parse note content into notes_lines and todo_items.

        notes_lines: raw content lines from the notes section (subheadings kept
                     as '## Title' so _build_rtf can style them, but the top-level
                     '# My Notes' / 'My Notes' heading is stripped since code adds it).
        todo_items:  just the item text (no '- ' prefix, no heading).
        """
        if not content.strip():
            return [], []

        notes_section, todo_section = self._split_sections(content)

        # Parse notes lines — strip the top-level heading (code adds it)
        notes_lines: list[str] = []
        if notes_section:
            for line in notes_section.split("\n"):
                stripped = line.strip()
                # Skip top-level "My Notes" / "Notes" heading in any form
                if re.match(r"^#{0,3}\s*(My\s+)?Notes\s*$", stripped, re.IGNORECASE):
                    continue
                # Skip any "To Do" markers that ended up in notes (from corruption)
                if re.match(r"^#{0,3}\s*To\s*Do(\s*List)?\s*$", stripped, re.IGNORECASE):
                    continue
                notes_lines.append(line)
            # Strip leading/trailing blank lines left by heading removal
            while notes_lines and not notes_lines[0].strip():
                notes_lines.pop(0)
            while notes_lines and not notes_lines[-1].strip():
                notes_lines.pop()

        # Parse to-do items (strip prefix)
        todo_items: list[str] = []
        if todo_section:
            for line in todo_section.split("\n"):
                stripped = line.strip()
                # Skip the "To Do" / "To Do List" heading itself (any markdown prefix)
                if re.match(r"^#{0,3}\s*To\s*Do(\s*List)?\s*$", stripped, re.IGNORECASE):
                    continue
                # Skip placeholder text from corrupted files
                if stripped.lower() in ("(no items)", "no items"):
                    continue
                if stripped.startswith("- "):
                    todo_items.append(stripped[2:].strip())
                elif stripped.startswith("\u2022 "):
                    todo_items.append(stripped[2:].strip())
                elif stripped:
                    # Only include non-empty lines that look like actual items
                    # Skip lines that are just leftover noise
                    todo_items.append(stripped)

        return notes_lines, todo_items

    def _normalize_for_compare(self, text: str) -> str:
        """Normalize text for fuzzy duplicate comparison.
        
        Strips punctuation, extra whitespace, common filler words, and lowercases.
        """
        import re
        if not text:
            return ""
        # Lowercase
        text = text.lower()
        # Remove common prefixes that don't change meaning
        for prefix in ["received ", "got ", "have ", "i have ", "there are ", "there is "]:
            if text.startswith(prefix):
                text = text[len(prefix):]
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Collapse whitespace
        text = ' '.join(text.split())
        return text

    def _semantic_match(self, description: str, item_text: str) -> bool:
        """Check if a semantic description matches an item.
        
        Handles cases where the user says "remove the greeting" but the item is "hello world".
        Maps common descriptions to patterns.
        """
        # Semantic mappings: description -> list of patterns that match it
        semantic_patterns = {
            "greeting": ["hello", "hi ", "hey", "greetings", "howdy", "hello world", "hi there"],
            "test": ["test", "testing", "hello world", "foo", "bar", "example"],
            "placeholder": ["placeholder", "todo", "tbd", "xxx", "sample", "example", "hello world"],
            "sample": ["sample", "example", "test", "hello world", "placeholder"],
            "default": ["default", "example", "sample", "hello world", "placeholder"],
        }
        
        # Check if the description maps to any patterns
        if description in semantic_patterns:
            patterns = semantic_patterns[description]
            for pattern in patterns:
                if pattern in item_text:
                    return True
        
        return False

    def _apply_edit(self, op: str, edit: dict, notes_lines: list, todo_items: list) -> str:
        """Apply a single edit operation. Modifies notes_lines/todo_items in place.

        Returns a human-readable description of what was done.
        Raises ValueError on invalid input.
        """
        if op == "add_todo":
            item = edit.get("item", "").strip()
            if not item:
                raise ValueError("'item' is required for add_todo")
            due = edit.get("due", "").strip()
            entry = f"{item} (due: {due})" if due else item
            
            # Fuzzy duplicate check: normalize and check if new item is contained in existing or vice versa
            item_norm = self._normalize_for_compare(item)
            for i, existing in enumerate(todo_items):
                # Extract just the item text (before any "(due:" suffix)
                existing_text = existing.split(" (due:")[0].strip()
                existing_norm = self._normalize_for_compare(existing_text)
                # Check: exact match, or one contains the other (semantic duplicate)
                if item_norm == existing_norm or item_norm in existing_norm or existing_norm in item_norm:
                    # If due date provided and different, update the existing item instead of skipping
                    if due and f"(due: {due})" not in existing:
                        todo_items[i] = entry
                        return f"Updated existing to-do with new due date: {entry}"
                    return f"Already exists: '{existing_text}' (no change needed)"
            todo_items.append(entry)
            return f"Added to-do: {entry}"

        elif op == "remove_todo":
            match = edit.get("match", "").strip().lower()
            if not match:
                raise ValueError("'match' is required for remove_todo")
            removed = []
            remaining = []
            for item in todo_items:
                item_lower = item.lower()
                # Direct substring match
                if match in item_lower:
                    removed.append(item)
                # Semantic matching for common descriptions
                elif self._semantic_match(match, item_lower):
                    removed.append(item)
                else:
                    remaining.append(item)
            if not removed:
                raise ValueError(f"No to-do item matching '{match}' found")
            todo_items.clear()
            todo_items.extend(remaining)
            return f"Removed to-do: {', '.join(removed)}"

        elif op == "edit_todo":
            old = edit.get("old", "").strip().lower()
            new = edit.get("new", "").strip()
            if not old or not new:
                raise ValueError("'old' and 'new' are required for edit_todo")
            found = False
            for i, item in enumerate(todo_items):
                if old in item.lower():
                    todo_items[i] = new
                    found = True
                    break
            if not found:
                raise ValueError(f"No to-do item matching '{old}' found")
            return f"Edited to-do: '{old}' → '{new}'"

        elif op == "add_note":
            content = edit.get("content", "").strip()
            subheading = edit.get("subheading", "").strip()
            if not subheading:
                raise ValueError("'subheading' is required for add_note (notes must follow subheading → content structure)")
            if not content:
                raise ValueError("'content' is required for add_note")
            
            # Fuzzy duplicate check for content under same/similar subheading
            content_norm = self._normalize_for_compare(content)
            subheading_norm = self._normalize_for_compare(subheading)
            
            # Find if this subheading already exists
            existing_subheading_idx = None
            for idx, line in enumerate(notes_lines):
                line_stripped = line.strip()
                # Match "## Subheading" marker
                if line_stripped.startswith("## "):
                    existing_sub = self._normalize_for_compare(line_stripped[3:])
                    if subheading_norm == existing_sub or subheading_norm in existing_sub or existing_sub in subheading_norm:
                        existing_subheading_idx = idx
                        break
            
            # Check for duplicate content in the note
            for i, line in enumerate(notes_lines):
                line_norm = self._normalize_for_compare(line)
                if not line_norm:
                    continue
                # Similar content already exists
                if content_norm == line_norm or content_norm in line_norm or line_norm in content_norm:
                    if content != line.strip():
                        notes_lines[i] = content
                        return f"Updated existing note: '{line[:30]}' → '{content[:30]}'"
                    return f"Already exists: '{line[:40]}' (no change needed)"
            
            # Subheading exists - insert content right after it
            if existing_subheading_idx is not None:
                notes_lines.insert(existing_subheading_idx + 1, content)
                return f"Added note under existing '{subheading}': {content[:40]}"
            
            # New subheading + content block
            notes_lines.append("")
            notes_lines.append(f"## {subheading}")
            notes_lines.append(content)
            return f"Added note under '{subheading}': {content[:60]}"

        elif op == "remove_note":
            match = edit.get("match", "").strip().lower()
            if not match:
                raise ValueError("'match' is required for remove_note")
            # Find and remove matching subheading + its content block
            found = False
            new_lines: list[str] = []
            skip_until_next_heading = False
            for line in notes_lines:
                stripped = line.strip().lower()
                # Check if this line is a subheading that matches (direct or semantic)
                is_heading_match = stripped.startswith("## ") and (match in stripped or self._semantic_match(match, stripped))
                if is_heading_match:
                    skip_until_next_heading = True
                    found = True
                    continue
                # Check if this is a content line matching (not a heading) - direct or semantic
                is_content_match = not skip_until_next_heading and not stripped.startswith("#") and (match in stripped or self._semantic_match(match, stripped))
                if is_content_match:
                    found = True
                    continue
                # Stop skipping when we hit the next heading
                if skip_until_next_heading and stripped.startswith("#"):
                    skip_until_next_heading = False
                if not skip_until_next_heading:
                    new_lines.append(line)
            if not found:
                raise ValueError(f"No note content matching '{match}' found")
            notes_lines.clear()
            notes_lines.extend(new_lines)
            return f"Removed note matching: {match}"

        elif op == "edit_note":
            old = edit.get("old", "").strip()
            new = edit.get("new", "").strip()
            if not old or not new:
                raise ValueError("'old' and 'new' are required for edit_note")
            found = False
            old_lower = old.lower()
            for i, line in enumerate(notes_lines):
                if old_lower in line.lower():
                    # Replace the matching portion, preserving the rest of the line
                    idx = line.lower().index(old_lower)
                    notes_lines[i] = line[:idx] + new + line[idx + len(old):]
                    found = True
                    break
            if not found:
                raise ValueError(f"No note text matching '{old}' found")
            return f"Edited note: '{old}' → '{new}'"

        else:
            raise ValueError(f"Unknown operation: '{op}'. Use: add_todo, remove_todo, edit_todo, add_note, remove_note, edit_note")

    def _reconstruct_note_plain(self, notes_lines: list, todo_items: list) -> str:
        """Reconstruct a plain-text summary of the note (for API response).
        
        Strips internal ## markers and filters artifacts for clean output.
        """
        # Known artifacts to filter out
        artifacts = {"*;;", ";;", "*;", ";*", "Helvetica;", "Helvetica; ;;"}
        
        parts: list[str] = ["My Notes"]
        
        for line in notes_lines:
            stripped = line.strip()
            # Skip artifacts
            if stripped in artifacts:
                continue
            if stripped:
                # Strip internal ## markers for display
                if stripped.startswith("## "):
                    parts.append(stripped[3:])
                else:
                    parts.append(stripped)
        
        parts.append("")
        parts.append("To Do List")
        if todo_items:
            for item in todo_items:
                parts.append(f"    - {item}")
        else:
            parts.append("(no items)")

        # Final cleanup: remove any remaining artifacts
        output = "\n".join(parts)
        for artifact in ["*;;", ";;", "*;", ";*"]:
            output = output.replace(artifact, "")
        while "\n\n\n" in output:
            output = output.replace("\n\n\n", "\n\n")
        
        return output.strip()
    
    def _refresh_stickies_app(self) -> None:
        """
        Refresh the Stickies app to show updated content.
        
        The Stickies app doesn't have AppleScript support for refreshing,
        so we quit and relaunch it. This is fast and ensures the note shows
        the latest content.
        """
        try:
            # Check if Stickies is running
            result = subprocess.run(
                ["pgrep", "-x", "Stickies"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode != 0:
                # Stickies not running, nothing to refresh
                return
            
            # Quit Stickies gracefully (it auto-saves)
            subprocess.run(
                ["osascript", "-e", 'tell application "Stickies" to quit'],
                capture_output=True,
                timeout=3
            )
            
            # Brief pause to ensure it fully quits
            import time
            time.sleep(0.3)
            
            # Relaunch Stickies
            subprocess.run(
                ["open", "-a", "Stickies"],
                capture_output=True,
                timeout=3
            )
            
            self.logger.debug("Refreshed Stickies app to show updated content")
            
        except subprocess.TimeoutExpired:
            self.logger.warning("Timeout refreshing Stickies app")
        except Exception as e:
            # Don't fail the write operation if refresh fails
            self.logger.warning(f"Could not refresh Stickies app: {e}")

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute read or write action with full validation."""
        try:
            if LOG_TOOLS:
                self.logger.info("Executing Tool: Stickies -- %s", params)

            if not self._is_macos:
                return {
                    "success": False,
                    "error": "Stickies tool is only supported on macOS",
                    "error_code": "STICKIES_NOT_FOUND",
                }

            # Find the single sticky note
            sticky_id, find_err = self._find_sticky_id()
            if find_err or not sticky_id:
                return _error_response(find_err or "STICKIES_NOT_FOUND")

            action = (params.get("action") or "").strip().lower()
            if action not in ("read", "write"):
                return {
                    "success": False,
                    "error": "Invalid action; use 'read' or 'write'",
                    "error_code": "INVALID_ACTION",
                }

            if action == "read":
                section = (params.get("section") or "both").strip().lower()
                if section not in ("notes", "todo", "both"):
                    section = "both"
                return self._action_read(sticky_id, section)

            if action == "write":
                rtf_content = params.get("rtf_content")
                if isinstance(rtf_content, str) and rtf_content.strip():
                    return self._action_write_rtf(sticky_id, rtf_content)

                edits = params.get("edits")
                if not edits or not isinstance(edits, list):
                    return {
                        "success": False,
                        "error": "Either 'rtf_content' or 'edits' array is required for write action.",
                        "error_code": "MISSING_WRITE_INPUT",
                    }
                return self._action_write(sticky_id, edits)

            return _error_response("INVALID_ACTION")

        except Exception as e:
            self.logger.error("Stickies tool failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "error_code": "STICKIES_WRITE_FAILED",
            }


def _error_response(error_code: str) -> Dict[str, Any]:
    """Build a standard error response with human-readable message."""
    messages = {
        "SECURITY_OUTSIDE_SANDBOX": "Resolved path is outside Stickies directory.",
        "SECURITY_INVALID_STRUCTURE": "Sticky note bundle structure is invalid.",
        "SECURITY_RTF_INVALID": "Content is not valid RTF or exceeds size limit.",
        "INVALID_RTF_STRUCTURE": "RTF structure is invalid. Expected H1 My Notes, H2 subheadings with paragraphs, and H1 To Do List with dash items.",
        "STICKIES_NOT_FOUND": "No sticky note found. Please create one in the Stickies app first.",
        "STICKIES_WRITE_FAILED": "Write failed; original content was restored from backup.",
        "INVALID_ACTION": "Invalid action; use 'read' or 'write'.",
        "MISSING_EDITS": "'edits' array is required for write. Use ops: add_todo, remove_todo, edit_todo, add_note, remove_note, edit_note.",
        "MISSING_WRITE_INPUT": "Provide either 'rtf_content' or 'edits' for write.",
    }
    return {
        "success": False,
        "error": messages.get(error_code, "Operation failed."),
        "error_code": error_code,
    }
