# -*- coding: utf-8 -*-
"""Lazy, platform-neutral CMTK execution backend for MADI3D.

Importing this module never probes WSL, reads CMTK, or launches a subprocess.
Detection happens only when a CMTK feature explicitly calls ``detect()``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from madi3d_storage import atomic_write_json, config_file, read_json_object


MANAGED_WSL_DISTRO = "MADI3D-CMTK"
CMTK_CORE_TOOLS = ("mat2dof", "dof2mat", "registration", "warp", "reformatx", "describe")
CMTK_STATE_READY = "ready"
CMTK_STATE_UNAVAILABLE = "unavailable"
CMTK_STATE_UNCHECKED = "unchecked"


class CMTKError(RuntimeError):
    pass


class CMTKUnavailableError(CMTKError):
    pass


class CMTKSetupCancelled(CMTKError):
    pass


class CMTKSetupPending(CMTKError):
    """Setup cannot continue until Windows has been restarted."""
    pass


@dataclass
class CMTKStatus:
    ready: bool = False
    platform: str = ""
    backend: str = ""
    distro: str = ""
    validated_tools: tuple[str, ...] = ()
    version: str = ""
    state: str = CMTK_STATE_UNCHECKED
    last_validated: str = ""
    summary: str = ""
    details: list[str] = field(default_factory=list)

    def diagnostic_text(self) -> str:
        lines = [self.summary] if self.summary else []
        if self.platform:
            lines.append(f"Platform: {self.platform}")
        if self.backend:
            lines.append(f"Backend: {self.backend}")
        if self.distro:
            lines.append(f"WSL distribution: {self.distro}")
        if self.version:
            lines.append(f"Version: {self.version}")
        if self.validated_tools:
            lines.append("Validated tools: " + ", ".join(self.validated_tools))
        if self.last_validated:
            lines.append(f"Last validated: {self.last_validated}")
        lines.extend(str(v) for v in self.details if str(v).strip())
        return "\n".join(lines).strip()


def decode_console(value) -> str:
    """Decode console bytes from WSL/Windows without assuming UTF-16 endianness."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    data = bytes(value)
    if not data:
        return ""

    # Respect an explicit UTF-16 BOM first. WSL output is not consistent across
    # Windows builds/commands: some streams are UTF-8, while others have been
    # observed as UTF-16LE or UTF-16BE.
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16", errors="replace")
        except Exception:
            pass

    # BOM-less UTF-16 emitted by Windows can be distinguished for ASCII-heavy
    # command output by which byte lane contains the NUL padding. LE has NULs
    # primarily at odd byte positions; BE primarily at even positions.
    if data.count(b"\x00") > max(2, len(data) // 8):
        even_nuls = data[0::2].count(b"\x00")
        odd_nuls = data[1::2].count(b"\x00")
        encoding = "utf-16be" if even_nuls > odd_nuls else "utf-16le"
        try:
            return data.decode(encoding, errors="replace")
        except Exception:
            pass

    return data.decode("utf-8", errors="replace")


def process_error(proc) -> str:
    return (
        decode_console(getattr(proc, "stderr", "")).strip()
        or decode_console(getattr(proc, "stdout", "")).strip()
    )


def _missing_command(text: str) -> bool:
    low = str(text or "").lower()
    return any(marker in low for marker in (
        "command not found", "not recognized as an internal", "no such file or directory",
        "unknown command", "invalid command", "could not find command",
        "cannot execute", "executable file not found", "cmtk: not found",
    ))


def _missing_runtime_command(text: str) -> bool:
    low = str(text or "").lower()
    return any(marker in low for marker in (
        "command not found", "not recognized as an internal",
        "unknown command", "invalid command", "could not find command",
        "executable file not found", "cmtk: not found",
    ))


def is_missing_cmtk_command_error(value) -> bool:
    """Return True only when the configured CMTK executable/tool itself is missing."""
    if isinstance(value, FileNotFoundError):
        return True
    result = getattr(value, "result", None)
    if result is not None:
        text = (
            str(getattr(result, "stderr", "") or "").strip()
            or str(getattr(result, "stdout", "") or "").strip()
        )
        if _missing_runtime_command(text):
            return True
    if hasattr(value, "returncode"):
        return _missing_runtime_command(process_error(value))
    return _missing_runtime_command(str(value or ""))


def windows_path_fallback(path: str) -> Optional[str]:
    raw = os.path.abspath(os.fspath(path))
    if len(raw) >= 3 and raw[1] == ":" and raw[2] in ("\\", "/"):
        return f"/mnt/{raw[0].lower()}{raw[2:].replace(chr(92), '/')}"
    return None


class CMTKBackend:
    kind = "base"

    def __init__(self, runner=subprocess.run, cmtk_command="cmtk"):
        self.runner = runner
        self.cmtk_command = str(cmtk_command or "cmtk")

    @property
    def label(self):
        return self.kind

    def command(self, tool: Optional[str], args: Sequence[str]) -> list[str]:
        raise NotImplementedError

    def execution_command(self, tool: Optional[str], args: Sequence[str]) -> list[str]:
        """Return argv as seen by the backend-native operating system."""
        cmd = [self.cmtk_command]
        if tool:
            cmd.append(str(tool))
        return cmd + [str(v) for v in args]

    def _notify_missing_tool(self, tool, detail):
        callback = getattr(self, "_madi_missing_tool_callback", None)
        if callable(callback):
            try:
                callback(str(tool or self.cmtk_command or "cmtk"), str(detail or ""))
            except Exception:
                pass

    def run(self, tool: Optional[str], args=(), **kwargs):
        command = self.command(tool, [os.fspath(v) for v in args])
        try:
            proc = self.runner(command, **kwargs)
        except FileNotFoundError as exc:
            self._notify_missing_tool(tool, exc)
            raise
        if getattr(proc, "returncode", 0) != 0 and is_missing_cmtk_command_error(proc):
            self._notify_missing_tool(tool, process_error(proc))
        return proc

    def probe(self, tools: Iterable[str] = ()) -> tuple[bool, list[str]]:
        requested_tools = tuple(
            dict.fromkeys(str(tool) for tool in tools if str(tool).strip())
        )
        # The Ubuntu/Debian `cmtk` launcher routes --help through `man`, which
        # is not guaranteed to exist in minimal Linux or WSL installations.
        # Probe actual CMTK executables with their toolkit-level --version
        # option instead.  A no-tool probe uses one core executable that is
        # present in supported CMTK distributions.
        probe_tools = requested_tools or ("dof2mat",)
        details = []
        for tool in probe_tools:
            try:
                proc = self.run(
                    tool,
                    ["--version"],
                    capture_output=True,
                    check=False,
                    timeout=20,
                )
            except Exception as exc:
                return False, details + [f"{tool}: {exc}"]
            text = process_error(proc)
            if getattr(proc, "returncode", 1) != 0 or _missing_command(text):
                return False, details + [f"{tool}: {text or 'validation failed'}"]
            version = next(
                (line.strip() for line in text.splitlines() if line.strip()),
                "",
            )
            details.append(f"{tool}: {version or 'OK'}")
        return True, details

    def translate_path(self, path: str) -> str:
        return os.path.abspath(os.fspath(path))

    def create_execution_temp_dir(self, prefix: str = "madi3d-cmtk-") -> Optional[str]:
        """Return a backend-native temporary directory, or ``None`` when unnecessary."""
        return None

    def copy_execution_tree_to_local(self, source_path: str, destination_path: str) -> None:
        raise CMTKError(f"{self.label} does not expose a remote execution filesystem.")

    def remove_execution_temp_dir(self, path: str) -> None:
        return None


class NativeCMTKBackend(CMTKBackend):
    kind = "native"

    @property
    def label(self):
        return f"Native ({self.cmtk_command})"

    def command(self, tool, args):
        return self.execution_command(tool, args)


class WSLCMTKBackend(CMTKBackend):
    kind = "wsl"

    def __init__(self, distro, runner=subprocess.run, cmtk_command="cmtk", user=None):
        super().__init__(runner=runner, cmtk_command=cmtk_command)
        self.distro = str(distro)
        self.user = str(user).strip() if user else ""

    @property
    def label(self):
        suffix = f", user {self.user}" if self.user else ""
        return f"WSL ({self.distro}{suffix})"

    def _prefix(self):
        cmd = ["wsl.exe", "-d", self.distro]
        if self.user:
            cmd += ["-u", self.user]
        # Use WSL direct execution so argv reaches the target unchanged.
        return cmd + ["--exec"]

    def command(self, tool, args):
        return self._prefix() + self.execution_command(tool, args)

    def translate_path(self, path: str) -> str:
        raw = os.path.abspath(os.fspath(path))
        try:
            proc = self.runner(
                self._prefix() + ["wslpath", "-a", "-u", raw],
                capture_output=True, check=False, timeout=15,
            )
            if getattr(proc, "returncode", 1) == 0:
                out = decode_console(getattr(proc, "stdout", b"")).strip()
                if out:
                    return out
        except Exception:
            pass
        fallback = windows_path_fallback(raw)
        if fallback:
            return fallback
        raise CMTKError(f"Could not translate Windows path for WSL: {raw}")

    @staticmethod
    def _validated_execution_temp_path(path: str, prefix: str = "madi3d-cmtk-") -> str:
        value = str(path or "").strip()
        candidate = Path(value)
        if (
            not value.startswith("/tmp/")
            or candidate.parent.as_posix() != "/tmp"
            or not candidate.name.startswith(prefix)
            or candidate.name in ("", ".", "..")
        ):
            raise CMTKError(f"Refusing unsafe WSL temporary path: {value!r}")
        return value

    def create_execution_temp_dir(self, prefix: str = "madi3d-cmtk-") -> str:
        prefix = str(prefix or "")
        if not prefix.startswith("madi3d-cmtk-") or not all(
            ch.isalnum() or ch in "._-" for ch in prefix
        ):
            raise CMTKError(f"Invalid CMTK WSL temporary prefix: {prefix!r}")
        template = f"/tmp/{prefix}XXXXXX"
        proc = self.runner(
            self._prefix() + ["/usr/bin/mktemp", "-d", template],
            capture_output=True, check=False, timeout=15,
        )
        if getattr(proc, "returncode", 1) != 0:
            raise CMTKError(
                "Could not create native WSL CMTK workspace: "
                + (process_error(proc) or f"exit {getattr(proc, 'returncode', None)}")
            )
        path = decode_console(getattr(proc, "stdout", b"")).strip()
        return self._validated_execution_temp_path(path, prefix=prefix)

    def copy_execution_tree_to_local(self, source_path: str, destination_path: str) -> None:
        source = self._validated_execution_temp_path(source_path)
        destination = os.path.abspath(os.fspath(destination_path))
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        translated_destination = self.translate_path(destination)
        proc = self.runner(
            self._prefix() + ["/bin/cp", "-R", "--", source, translated_destination],
            capture_output=True, check=False, timeout=120,
        )
        if getattr(proc, "returncode", 1) != 0:
            raise CMTKError(
                f"Could not copy CMTK artifact from WSL to {destination}: "
                + (process_error(proc) or f"exit {getattr(proc, 'returncode', None)}")
            )
        if not Path(destination).is_dir():
            raise CMTKError(
                f"WSL CMTK artifact copy reported success but destination is missing: {destination}"
            )

    def remove_execution_temp_dir(self, path: str) -> None:
        value = self._validated_execution_temp_path(path)
        proc = self.runner(
            self._prefix() + ["/bin/rm", "-rf", "--", value],
            capture_output=True, check=False, timeout=30,
        )
        if getattr(proc, "returncode", 1) != 0:
            raise CMTKError(
                "Could not remove native WSL CMTK workspace: "
                + (process_error(proc) or f"exit {getattr(proc, 'returncode', None)}")
            )


class CMTKManager:
    """Lazy detector/cache. Construction deliberately has no I/O side effects."""

    def __init__(self, *, runner=subprocess.run, which=shutil.which,
                 platform_name=None, config_path=None):
        self.runner = runner
        self.which = which
        self.platform_override = platform_name
        self.config_override = Path(config_path) if config_path else None
        self.backend: Optional[CMTKBackend] = None
        self.validated_tools: set[str] = set()
        self.last_status = CMTKStatus(summary="CMTK has not been checked yet.")

    def platform_name(self):
        if self.platform_override:
            return self.platform_override
        if sys.platform.startswith("win"):
            return "windows"
        if sys.platform.startswith("linux"):
            return "linux"
        if sys.platform == "darwin":
            return "macos"
        return sys.platform

    def config_path(self):
        if self.config_override:
            return self.config_override
        return config_file(
            "cmtk_backend.json",
            platform_name=self.platform_name(),
        )

    def load_config(self):
        return read_json_object(self.config_path())

    def write_config(self, data):
        try:
            atomic_write_json(self.config_path(), dict(data or {}))
            return True
        except Exception:
            return False

    def _require_config_write(self, data):
        if self.write_config(data):
            return
        raise CMTKError(
            f"Could not save CMTK settings atomically to: {self.config_path()}\n"
            "Check that the MADI3D configuration folder is writable, then retry."
        )

    def update_config(self, **changes):
        data = self.load_config()
        for key, value in changes.items():
            if value is None:
                data.pop(key, None)
            else:
                data[key] = value
        self._require_config_write(data)
        return data

    @staticmethod
    def _trusted_backend_config(config):
        config = dict(config or {})
        kind = str(config.get("backend") or "").lower()
        if kind == "wsl":
            distro = str(config.get("distro") or "").strip()
            source = str(config.get("selection_source") or "").lower()
            return bool(distro and (config.get("managed_distro") or source == "existing"))
        if kind == "native":
            return bool(str(config.get("cmtk_command") or "").strip())
        return False

    @staticmethod
    def _version_from_details(details):
        for raw in details or ():
            text = str(raw or "").strip()
            low = text.lower()
            marker = low.find("cmtk ")
            if marker >= 0:
                return text[marker:].strip()
        return ""

    def persisted_status(self) -> CMTKStatus:
        """Read saved CMTK readiness without starting WSL or executing CMTK."""
        config = self.load_config()
        state = str(config.get("readiness_state") or "").lower()
        if state not in {CMTK_STATE_READY, CMTK_STATE_UNAVAILABLE, CMTK_STATE_UNCHECKED}:
            state = CMTK_STATE_READY if self._trusted_backend_config(config) else CMTK_STATE_UNCHECKED
        tools = tuple(dict.fromkeys(str(v) for v in (config.get("validated_tools") or ()) if str(v).strip()))
        if state == CMTK_STATE_READY and not tools:
            # Migration from the pre-persistence backend config: the selected backend
            # had already passed a real CMTK validation. Trust that state and let a
            # genuine command-not-found failure invalidate it later.
            tools = tuple(CMTK_CORE_TOOLS)
        kind = str(config.get("backend") or "").lower()
        backend_label = ""
        distro = str(config.get("distro") or "")
        command = str(config.get("cmtk_command") or "cmtk")
        if kind == "wsl" and distro:
            user = str(config.get("user") or "").strip()
            backend_label = f"WSL ({distro}{', user ' + user if user else ''})"
        elif kind == "native" and command:
            backend_label = f"Native ({command})"
        summary = str(config.get("readiness_summary") or "").strip()
        if not summary:
            summary = (
                "CMTK is ready." if state == CMTK_STATE_READY else
                "CMTK is not installed or needs setup." if state == CMTK_STATE_UNAVAILABLE else
                "CMTK has not been checked yet."
            )
        details = [str(v) for v in (config.get("readiness_details") or ()) if str(v).strip()]
        status = CMTKStatus(
            ready=state == CMTK_STATE_READY,
            platform=self.platform_name(),
            backend=backend_label,
            distro=distro,
            validated_tools=tools,
            version=str(config.get("cmtk_version") or ""),
            state=state,
            last_validated=str(config.get("last_successful_validation") or ""),
            summary=summary,
            details=details,
        )
        self.last_status = status
        return status

    def _backend_from_config(self, config):
        config = dict(config or {})
        kind = str(config.get("backend") or "").lower()
        command = str(config.get("cmtk_command") or "cmtk")
        if kind == "wsl" and self._trusted_backend_config(config):
            return WSLCMTKBackend(
                str(config.get("distro") or ""), runner=self.runner,
                cmtk_command=command, user=str(config.get("user") or ""),
            )
        if kind == "native" and self._trusted_backend_config(config):
            return NativeCMTKBackend(runner=self.runner, cmtk_command=command)
        return None

    def _attach_runtime_state(self, backend, *, tools=(), version="", persisted_ready=False):
        backend.validated_tools = set(str(v) for v in tools if str(v).strip())
        backend.cmtk_version = str(version or "")
        backend._madi_persisted_ready = bool(persisted_ready)
        backend._madi_missing_tool_callback = self.mark_command_missing
        return backend

    def restore_persisted_backend(self) -> Optional[CMTKBackend]:
        """Restore a previously validated backend without probing it."""
        if self.backend is not None:
            return self.backend
        status = self.persisted_status()
        if not status.ready:
            return None
        config = self.load_config()
        if "readiness_state" not in config:
            # Persist the migration itself, still without launching CMTK/WSL.
            try:
                self.update_config(
                    readiness_state=CMTK_STATE_READY,
                    readiness_summary="CMTK is ready.",
                    validated_tools=list(status.validated_tools or CMTK_CORE_TOOLS),
                    cmtk_version=status.version,
                )
                config = self.load_config()
            except Exception:
                pass
        backend = self._backend_from_config(config)
        if backend is None:
            self.mark_unavailable("Saved CMTK backend settings are incomplete.")
            return None
        tools = status.validated_tools or tuple(CMTK_CORE_TOOLS)
        self.backend = self._attach_runtime_state(
            backend, tools=tools, version=status.version, persisted_ready=True
        )
        self.validated_tools = set(tools)
        return self.backend

    def mark_unavailable(self, summary="CMTK is not installed or needs setup.", details=()):
        """Persist dependency failure while preserving backend identity for repair/reinstall."""
        self.backend = None
        self.validated_tools.clear()
        detail_list = [str(v) for v in details if str(v).strip()]
        self.last_status = CMTKStatus(
            ready=False, platform=self.platform_name(), state=CMTK_STATE_UNAVAILABLE,
            summary=str(summary or "CMTK is not installed or needs setup."),
            details=detail_list,
        )
        try:
            self.update_config(
                readiness_state=CMTK_STATE_UNAVAILABLE,
                readiness_summary=self.last_status.summary,
                readiness_details=detail_list,
                last_failure=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        except Exception as exc:
            self.last_status.details.append(f"Could not persist CMTK failure state: {exc}")
        return self.last_status

    def mark_command_missing(self, tool, detail=""):
        message = f"Configured CMTK tool '{tool}' was not found."
        details = [str(detail)] if str(detail or "").strip() else []
        return self.mark_unavailable(message, details)

    def save_backend(self, backend, *, managed=None, validated_tools=(), version="", details=()):
        data = self.load_config()
        data.update({"backend": backend.kind, "cmtk_command": backend.cmtk_command})
        if isinstance(backend, WSLCMTKBackend):
            data["distro"] = backend.distro
            if backend.user:
                data["user"] = backend.user
            else:
                data.pop("user", None)
        if managed is True:
            data["managed_distro"] = True
            data["selection_source"] = "managed"
        elif managed is False:
            data.pop("managed_distro", None)
            data.pop("managed_source", None)
            data.pop("managed_install_path", None)
            data.pop("cmtk_package_version", None)
            data["selection_source"] = "existing"
        data.pop("setup_pending", None)
        data.pop("setup_mode", None)
        data.pop("wsl2_feature_repair_attempted", None)
        data.update({
            "readiness_state": CMTK_STATE_READY,
            "readiness_summary": "CMTK is ready.",
            "readiness_details": [str(v) for v in details if str(v).strip()],
            "validated_tools": sorted(dict.fromkeys(str(v) for v in validated_tools if str(v).strip())),
            "cmtk_version": str(version or ""),
            "last_successful_validation": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        data.pop("last_failure", None)
        self._require_config_write(data)

    def capture(self, cmd, timeout=20):
        try:
            return self.runner(list(cmd), capture_output=True, check=False, timeout=timeout)
        except Exception:
            return None

    def wsl_available(self):
        """True only when the WSL runtime reports a successful ready status."""
        if self.which("wsl.exe") is None and self.which("wsl") is None:
            return False
        proc = self.capture(["wsl.exe", "--status"])
        if proc is None:
            proc = self.capture(["wsl", "--status"])
        return proc is not None and getattr(proc, "returncode", 1) == 0

    def list_wsl_distros(self):
        proc = self.capture(["wsl.exe", "-l", "-q"])
        if proc is None:
            return []
        names = []
        for line in decode_console(getattr(proc, "stdout", b"")).replace("\x00", "").splitlines():
            name = line.strip().lstrip("* ").strip()
            if name and name not in names:
                names.append(name)
        return names

    @staticmethod
    def rank_distros(distros, configured=""):
        configured = str(configured or "").lower()

        def key(name):
            low = name.lower()
            rank = (
                0 if configured and low == configured else
                1 if low == MANAGED_WSL_DISTRO.lower() else
                2 if low.startswith("ubuntu") else
                3 if low.startswith("debian") else
                10
            )
            return rank, low

        return sorted(dict.fromkeys(distros), key=key)

    def accept_backend(self, backend, tools, details=None, *, managed=None):
        prospective_tools = set(self.validated_tools)
        prospective_tools.update(str(v) for v in tools if str(v).strip())
        detail_list = list(details or [])
        version = self._version_from_details(detail_list)
        if not version:
            version = str(getattr(backend, "cmtk_version", "") or "")
        self.save_backend(
            backend, managed=managed, validated_tools=prospective_tools,
            version=version, details=detail_list,
        )
        self.backend = self._attach_runtime_state(
            backend, tools=prospective_tools, version=version, persisted_ready=True
        )
        self.validated_tools = prospective_tools
        self.last_status = CMTKStatus(
            ready=True,
            platform=self.platform_name(),
            backend=backend.label,
            distro=getattr(backend, "distro", ""),
            validated_tools=tuple(sorted(self.validated_tools)),
            version=version,
            state=CMTK_STATE_READY,
            last_validated=self.load_config().get("last_successful_validation", ""),
            summary="CMTK is ready.",
            details=detail_list,
        )
        return self.backend

    def validate_backend(self, backend, tools, *, managed=None):
        ok, details = backend.probe(tools)
        if ok:
            return self.accept_backend(backend, tools, details, managed=managed)
        self.mark_unavailable(
            "The selected CMTK backend failed validation.", details
        )
        return None

    def invalidate(self):
        self.backend = None
        self.validated_tools.clear()
        self.last_status = CMTKStatus(
            state=CMTK_STATE_UNCHECKED, summary="CMTK will be checked on next use."
        )

    def detect(self, tools: Iterable[str] = ()) -> Optional[CMTKBackend]:
        """Probe now; call only from an actual CMTK-dependent action."""
        tools = tuple(dict.fromkeys(str(v) for v in tools if str(v).strip()))
        if self.backend is not None:
            missing = tuple(v for v in tools if v not in self.validated_tools)
            if not missing:
                return self.backend
            if self.validate_backend(self.backend, missing):
                return self.backend
            self.invalidate()

        config = self.load_config()
        platform_name = self.platform_name()
        details = []

        if platform_name == "windows":
            if not self.wsl_available():
                self.mark_unavailable(
                    "Windows Subsystem for Linux is not ready.",
                    ["This check was triggered by a CMTK action, not application startup."],
                )
                return None

            distros = self.list_wsl_distros()
            if not distros:
                self.mark_unavailable("WSL is installed, but no Linux distribution is installed.")
                return None

            configured_distro = str(config.get("distro") or "")
            selection_source = str(config.get("selection_source") or "")
            trusted_config = bool(config.get("managed_distro")) or selection_source == "existing"
            candidates = []
            if trusted_config and configured_distro and any(
                d.lower() == configured_distro.lower() for d in distros
            ):
                candidates.append((configured_distro, str(config.get("user") or "")))

            # Do not probe arbitrary personal WSL distributions. Starting a distro
            # can trigger its first-run/OOBE path. Users can explicitly select one
            # from the setup dialog when they want MADI3D to use an existing CMTK.
            for distro, user in candidates:
                backend = WSLCMTKBackend(
                    distro,
                    runner=self.runner,
                    cmtk_command=config.get("cmtk_command", "cmtk"),
                    user=user,
                )
                ok, probe_details = backend.probe(tools)
                if ok:
                    return self.accept_backend(backend, tools, probe_details)
                details.append(
                    f"{distro}: " + (probe_details[-1] if probe_details else "CMTK not found")
                )

            if candidates:
                summary = "The configured WSL CMTK backend is not usable."
            else:
                summary = "WSL is available, but no CMTK backend is configured yet."
                details.append("Installed WSL distributions: " + ", ".join(distros))
                details.append(
                    "Automatic setup will use an isolated MADI3D environment; choose an existing "
                    "installation explicitly to use a personal WSL distribution."
                )
            self.mark_unavailable(summary, details)
            return None

        configured = str(config.get("cmtk_command") or "").strip()
        candidates = [v for v in (configured, self.which("cmtk"), "cmtk") if v]
        for command in dict.fromkeys(candidates):
            backend = NativeCMTKBackend(runner=self.runner, cmtk_command=command)
            ok, probe_details = backend.probe(tools)
            if ok:
                return self.accept_backend(backend, tools, probe_details)
            details.extend(probe_details[-1:])

        self.mark_unavailable(
            "A usable native CMTK installation was not found.", details[-3:]
        )
        return None


_MANAGER: Optional[CMTKManager] = None


def get_cmtk_manager():
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = CMTKManager()
    return _MANAGER


__all__ = [
    "CMTKBackend", "CMTK_CORE_TOOLS", "CMTKError", "CMTKManager",
    "CMTKSetupCancelled", "CMTKSetupPending", "CMTKStatus",
    "CMTKUnavailableError", "CMTK_STATE_READY", "CMTK_STATE_UNAVAILABLE",
    "CMTK_STATE_UNCHECKED", "MANAGED_WSL_DISTRO", "NativeCMTKBackend",
    "WSLCMTKBackend", "decode_console", "get_cmtk_manager",
    "is_missing_cmtk_command_error", "process_error", "windows_path_fallback",
]
