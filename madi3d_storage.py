"""Canonical writable storage locations and atomic small-file writes for MADI3D."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


APP_DIR_NAME = "MADI3D"


@dataclass(frozen=True)
class JsonObjectReadResult:
    """Status-aware result for one small JSON-object settings file."""

    status: str
    value: dict
    error: str = ""


def _normalized_platform(platform_name=None) -> str:
    value = str(platform_name or sys.platform).strip().lower()
    if value.startswith("win"):
        return "windows"
    if value in {"darwin", "mac", "macos"}:
        return "macos"
    if value.startswith("linux"):
        return "linux"
    return value


def _environment(environ=None):
    return os.environ if environ is None else environ


def _home(home=None) -> Path:
    return Path.home() if home is None else Path(home)


def config_dir(*, platform_name=None, environ=None, home=None) -> Path:
    """Return MADI3D's per-user configuration directory without creating it."""
    platform = _normalized_platform(platform_name)
    env = _environment(environ)
    home_path = _home(home)

    if platform == "windows":
        root = Path(env.get("APPDATA") or home_path / "AppData" / "Roaming")
    elif platform == "macos":
        root = home_path / "Library" / "Application Support"
    else:
        root = Path(env.get("XDG_CONFIG_HOME") or home_path / ".config")
    return root / APP_DIR_NAME


def data_dir(*, platform_name=None, environ=None, home=None) -> Path:
    """Return MADI3D's per-user persistent data directory without creating it."""
    platform = _normalized_platform(platform_name)
    env = _environment(environ)
    home_path = _home(home)

    if platform == "windows":
        root = Path(env.get("LOCALAPPDATA") or home_path / "AppData" / "Local")
    elif platform == "macos":
        root = home_path / "Library" / "Application Support"
    else:
        root = Path(env.get("XDG_DATA_HOME") or home_path / ".local" / "share")
    return root / APP_DIR_NAME


def cache_dir(*, platform_name=None, environ=None, home=None) -> Path:
    """Return MADI3D's per-user cache directory without creating it."""
    platform = _normalized_platform(platform_name)
    env = _environment(environ)
    home_path = _home(home)

    if platform == "windows":
        root = Path(env.get("LOCALAPPDATA") or home_path / "AppData" / "Local")
        return root / APP_DIR_NAME / "Cache"
    if platform == "macos":
        return home_path / "Library" / "Caches" / APP_DIR_NAME
    root = Path(env.get("XDG_CACHE_HOME") or home_path / ".cache")
    return root / APP_DIR_NAME


def config_file(name, *, platform_name=None, environ=None, home=None) -> Path:
    return config_dir(
        platform_name=platform_name,
        environ=environ,
        home=home,
    ) / Path(name)


def read_json_object(path) -> dict:
    """Read a UTF-8 JSON object, returning an empty object for missing/invalid data."""
    return read_json_object_with_status(path).value


def read_json_object_with_status(path) -> JsonObjectReadResult:
    """Read a UTF-8 JSON object while preserving missing versus invalid state."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        return JsonObjectReadResult("missing", {})
    except UnicodeError:
        return JsonObjectReadResult(
            "invalid", {}, "The settings file is not valid UTF-8 text."
        )
    except OSError as exc:
        return JsonObjectReadResult(
            "invalid", {}, f"The settings file could not be read: {exc}"
        )
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return JsonObjectReadResult(
            "invalid", {}, "The settings file does not contain valid JSON."
        )
    if not isinstance(value, dict):
        return JsonObjectReadResult(
            "invalid", {}, "The settings file must contain one JSON object."
        )
    return JsonObjectReadResult("valid", value)


def atomic_write_text(path, text, *, encoding="utf-8") -> Path:
    """Atomically replace one small text file using a unique sibling temp file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(str(text))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        return destination
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def atomic_write_json(path, value) -> Path:
    payload = json.dumps(value, indent=2)
    return atomic_write_text(path, payload, encoding="utf-8")


__all__ = [
    "APP_DIR_NAME",
    "JsonObjectReadResult",
    "atomic_write_json",
    "atomic_write_text",
    "cache_dir",
    "config_dir",
    "config_file",
    "data_dir",
    "read_json_object",
    "read_json_object_with_status",
]
