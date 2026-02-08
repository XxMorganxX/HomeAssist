"""
Stickies Note Tool using BaseTool.

Safely reads and edits macOS Stickies notes with defense-in-depth security:
only files within the Stickies sandbox can be touched; edits are validated
and rolled back on failure.
"""

import os
import re
import tempfile
import plistlib
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from mcp_server.base_tool import BaseTool
from mcp_server.config import LOG_TOOLS


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


def _plain_to_rtf(text: str) -> str:
    """Convert plain text to minimal valid RTF (escape backslash and braces)."""
    escaped = (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", "\\line\n")
    )
    return (
        "{\\rtf1\\ansi\\ansicpg1252\\cocoartf2639\n"
        "{\\fonttbl\\f0\\fswiss\\fcharset0 Helvetica;}\n"
        "{\\colortbl;\\red255\\green255\\blue255;}\n"
        "\\pard\\tx560\\tx1120\\tx1680\\tx2240\\tx2800\\tx3360\\tx3920\\tx4480\\tx5040\\tx5600\\tx6160\\tx6720\\partightenfactor0\n"
        "\\f0\\fs24 \\cf0 "
        + escaped
        + "\n}"
    )


class StickiesTool(BaseTool):
    """Tool for listing, reading, and writing the user's desktop notes/to-do list (macOS Stickies app)."""

    name = "stickies"
    description = (
        "Manage the user's desktop notes and to-do list (macOS Stickies app). "
        "Use this tool when the user asks about their notes, to-do list, sticky notes, "
        "desktop notes, or anything they've written down. "
        "Actions: 'list' returns all note IDs; 'read' gets a note's content; "
        "'write' updates a note. This is NOT for calendar events - use calendar_data for that."
    )
    version = "1.0.0"

    def __init__(self):
        """Initialize the Stickies tool."""
        super().__init__()
        self._is_macos = platform.system() == "Darwin"

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "read", "write"],
                    "description": "Action: 'list' all stickies, 'read' one by ID, or 'write' content to one.",
                },
                "sticky_id": {
                    "type": "string",
                    "description": "UUID of the sticky note (required for read and write).",
                },
                "content": {
                    "type": "string",
                    "description": "Plain text content to write (required for write). Converted to RTF.",
                },
                "format": {
                    "type": "string",
                    "enum": ["plain", "rtf"],
                    "description": "For write: 'plain' (default) or 'rtf'. Plain is auto-converted to RTF.",
                    "default": "plain",
                },
            },
            "required": ["action"],
        }

    # -------------------------------------------------------------------------
    # Security: Layer 2-3 - UUID and path traversal validation
    # -------------------------------------------------------------------------
    def _validate_sticky_id(self, sticky_id: str) -> Tuple[bool, Optional[str]]:
        """Validate sticky_id format and reject path traversal. Returns (ok, error_code)."""
        if not isinstance(sticky_id, str) or not sticky_id.strip():
            return False, "SECURITY_UUID_INVALID"
        if _has_dangerous_chars(sticky_id):
            return False, "SECURITY_PATH_TRAVERSAL"
        if not UUID_PATTERN.match(sticky_id.strip()):
            return False, "SECURITY_UUID_INVALID"
        return True, None

    # -------------------------------------------------------------------------
    # Security: Layer 4-5 - Canonical path and structure validation
    # -------------------------------------------------------------------------
    def _get_safe_path(self, sticky_id: str) -> Tuple[Optional[Path], Optional[str]]:
        """
        Resolve sticky_id to RTF path only if it lies inside Stickies sandbox.
        Returns (rtf_path, error_code). rtf_path is the path to TXT.rtf inside the .rtfd bundle.
        """
        ok, err = self._validate_sticky_id(sticky_id)
        if not ok:
            return None, err

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
    # Actions: list, read, write
    # -------------------------------------------------------------------------
    def _action_list(self) -> Dict[str, Any]:
        """List all sticky note UUIDs and metadata from .SavedStickiesState."""
        state_file = STICKIES_DIR / ".SavedStickiesState"
        if not state_file.exists() or not state_file.is_file():
            return {
                "success": True,
                "stickies": [],
                "message": "No Stickies state file found or no stickies.",
            }

        try:
            with open(state_file, "rb") as f:
                state = plistlib.load(f)
        except (plistlib.InvalidFileException, OSError) as e:
            self.logger.warning("Could not read Stickies state: %s", e)
            return {
                "success": True,
                "stickies": [],
                "message": "Could not parse Stickies state file.",
            }

        if not isinstance(state, list):
            return {"success": True, "stickies": [], "message": "Invalid state format."}

        stickies = []
        for item in state:
            if not isinstance(item, dict):
                continue
            uuid_val = item.get("UUID")
            if not uuid_val or not isinstance(uuid_val, str):
                continue
            ok, _ = self._validate_sticky_id(uuid_val)
            if not ok:
                continue
            # Only include stickies that actually have an .rtfd directory
            rtfd_path = STICKIES_DIR / (uuid_val + ".rtfd")
            if not rtfd_path.is_dir():
                continue
            stickies.append({
                "sticky_id": uuid_val,
                "frame": item.get("Frame"),
                "sticky_color": item.get("StickyColor"),
            })

        return {"success": True, "stickies": stickies}

    def _action_read(self, sticky_id: str) -> Dict[str, Any]:
        """Read content of a single sticky note."""
        rtf_path, err = self._get_safe_path(sticky_id)
        if err:
            return _error_response(err)
        assert rtf_path is not None

        try:
            raw = rtf_path.read_text(encoding="utf-8", errors="replace")
            # Strip RTF for a simple plain-text approximation (full RTF parse would be heavier)
            plain = self._rtf_to_plain_approx(raw)
            return {
                "success": True,
                "sticky_id": sticky_id,
                "content": plain,
                "raw_rtf_length": len(raw),
            }
        except OSError as e:
            self.logger.warning("Stickies read failed: %s", e)
            return _error_response("STICKIES_NOT_FOUND")

    def _rtf_to_plain_approx(self, rtf: str) -> str:
        """Approximate RTF to plain text by removing control words and keeping text."""
        out = []
        i = 0
        while i < len(rtf):
            if rtf[i] == "\\":
                i += 1
                if i >= len(rtf):
                    break
                if rtf[i] in "{}":
                    out.append(rtf[i])
                    i += 1
                    continue
                # Skip control word (e.g. \rtf1, \line)
                while i < len(rtf) and rtf[i].isalpha():
                    i += 1
                if i < len(rtf) and rtf[i] == " ":
                    i += 1
                # Skip numeric parameter
                while i < len(rtf) and (rtf[i].isdigit() or rtf[i] == "-"):
                    i += 1
                if i < len(rtf) and rtf[i] == " ":
                    i += 1
                continue
            if rtf[i] == "{":
                i += 1
                continue
            if rtf[i] == "}":
                i += 1
                continue
            out.append(rtf[i])
            i += 1
        text = "".join(out)
        return text.replace("\\line", "\n").replace("\n\n", "\n").strip()

    def _action_write(self, sticky_id: str, content: str, fmt: str) -> Dict[str, Any]:
        """Write content to a sticky note (plain text converted to RTF by default)."""
        rtf_path, err = self._get_safe_path(sticky_id)
        if err:
            return _error_response(err)
        assert rtf_path is not None

        if fmt == "rtf":
            to_write = content
        else:
            to_write = _plain_to_rtf(content)

        if len(to_write) > MAX_CONTENT_SIZE:
            return _error_response("SECURITY_RTF_INVALID")

        backup_path = self._create_backup(rtf_path)
        ok, write_err = self._atomic_write(rtf_path, to_write, backup_path)
        if not ok:
            return _error_response(write_err or "STICKIES_WRITE_FAILED")

        return {
            "success": True,
            "sticky_id": sticky_id,
            "message": "Sticky note updated. If Stickies app is open, you may need to click the note or restart Stickies to see changes.",
            "warning": "STICKIES_APP_RUNNING",
        }

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute list, read, or write action with full validation."""
        try:
            if LOG_TOOLS:
                self.logger.info("Executing Tool: Stickies -- %s", params)

            if not self._is_macos:
                return {
                    "success": False,
                    "error": "Stickies tool is only supported on macOS",
                    "error_code": "STICKIES_NOT_FOUND",
                }

            action = (params.get("action") or "").strip().lower()
            if action not in ("list", "read", "write"):
                return {
                    "success": False,
                    "error": "Invalid action; use 'list', 'read', or 'write'",
                    "error_code": "SECURITY_UUID_INVALID",
                }

            if action == "list":
                return self._action_list()

            sticky_id = params.get("sticky_id")
            if not sticky_id:
                return {
                    "success": False,
                    "error": "sticky_id is required for read and write",
                    "error_code": "SECURITY_UUID_INVALID",
                }

            if action == "read":
                return self._action_read(sticky_id)

            if action == "write":
                content = params.get("content")
                if content is None:
                    return {
                        "success": False,
                        "error": "content is required for write",
                        "error_code": "SECURITY_RTF_INVALID",
                    }
                fmt = (params.get("format") or "plain").strip().lower()
                if fmt not in ("plain", "rtf"):
                    fmt = "plain"
                return self._action_write(sticky_id, str(content), fmt)

            return _error_response("SECURITY_UUID_INVALID")

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
        "SECURITY_UUID_INVALID": "Invalid sticky ID format; must be a UUID.",
        "SECURITY_PATH_TRAVERSAL": "Path traversal or invalid characters in sticky ID.",
        "SECURITY_OUTSIDE_SANDBOX": "Resolved path is outside Stickies directory.",
        "SECURITY_INVALID_STRUCTURE": "Sticky note bundle structure is invalid.",
        "SECURITY_RTF_INVALID": "Content is not valid RTF or exceeds size limit.",
        "STICKIES_NOT_FOUND": "Sticky note not found or Stickies data directory missing.",
        "STICKIES_WRITE_FAILED": "Write failed; original content was restored from backup.",
    }
    return {
        "success": False,
        "error": messages.get(error_code, "Operation failed."),
        "error_code": error_code,
    }
