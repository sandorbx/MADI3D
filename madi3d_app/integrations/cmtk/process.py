# -*- coding: utf-8 -*-
"""Long-running process execution for CMTK backends.

This module owns streamed output, cancellation, timeout handling, bounded
capture, and process termination for compute jobs such as ``warp`` and
``reformatx``.

For WSL jobs MADI3D launches CMTK in its own Linux session/process group and
receives the Linux PID through a fixed control marker before accepting the job
as managed. Cancellation signals that Linux process group; it never terminates
the whole WSL distribution.
"""
from __future__ import annotations

import codecs
import os
import queue
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from madi3d_app.integrations.cmtk.backend import CMTKError, is_missing_cmtk_command_error


TextCallback = Optional[Callable[[str], None]]
CancelCheck = Optional[Callable[[], bool]]

DEFAULT_CAPTURE_LIMIT = 1024 * 1024
WSL_PID_PREFIX = "__MADI3D_CMTK_PID__="
WSL_HANDSHAKE_TIMEOUT = 5.0
_WSL_SESSION_SCRIPT = (
    'printf "__MADI3D_CMTK_PID__=%s\\n" "$$"; exec "$@"'
)


@dataclass(frozen=True)
class CMTKProcessResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    cancelled: bool = False
    timed_out: bool = False

    def error_text(self) -> str:
        return (self.stderr or self.stdout).strip()


class CMTKProcessError(CMTKError):
    def __init__(self, message: str, result: Optional[CMTKProcessResult] = None):
        super().__init__(message)
        self.result = result


class _TailBuffer:
    """Bounded text tail for result/error summaries."""

    def __init__(self, limit: Optional[int]):
        self.limit = None if limit is None else max(0, int(limit))
        self.parts = deque()
        self.length = 0

    def append(self, text: str) -> None:
        text = str(text)
        if not text or self.limit == 0:
            return
        if self.limit is None:
            self.parts.append(text)
            self.length += len(text)
            return
        if len(text) >= self.limit:
            self.parts.clear()
            self.parts.append(text[-self.limit:])
            self.length = self.limit
            return
        self.parts.append(text)
        self.length += len(text)
        while self.length > self.limit and self.parts:
            excess = self.length - self.limit
            first = self.parts[0]
            if len(first) <= excess:
                self.parts.popleft()
                self.length -= len(first)
            else:
                self.parts[0] = first[excess:]
                self.length -= excess

    def text(self) -> str:
        return "".join(self.parts)


def _stream_encoding(first: bytes) -> str:
    data = bytes(first or b"")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    if data.count(b"\x00") > max(2, len(data) // 8):
        even_nuls = data[0::2].count(b"\x00")
        odd_nuls = data[1::2].count(b"\x00")
        return "utf-16be" if even_nuls > odd_nuls else "utf-16le"
    return "utf-8"


def _decoded_chunks(stream, chunk_size: int = 4096):
    """Decode pipe bytes as soon as they are available, without waiting for EOF."""
    try:
        fd = stream.fileno()
    except Exception:
        fd = None
    if fd is not None:
        read = lambda size: os.read(fd, size)
    else:
        read = getattr(stream, "read1", None) or stream.read

    first = read(chunk_size)
    if not first:
        return
    decoder = codecs.getincrementaldecoder(_stream_encoding(first))(errors="replace")
    text = decoder.decode(first)
    if text:
        yield text
    while True:
        chunk = read(chunk_size)
        if not chunk:
            break
        text = decoder.decode(chunk)
        if text:
            yield text
    final = decoder.decode(b"", final=True)
    if final:
        yield final


def _is_wsl_backend(backend) -> bool:
    return str(getattr(backend, "kind", "")).lower() == "wsl"


def _wsl_prefix(backend) -> list[str]:
    distro = str(getattr(backend, "distro", "") or "").strip()
    if not distro:
        raise CMTKProcessError("WSL CMTK backend has no distribution name.")
    cmd = ["wsl.exe", "-d", distro]
    user = str(getattr(backend, "user", "") or "").strip()
    if user:
        cmd += ["-u", user]
    return cmd + ["--exec"]


def _launch_command(backend, tool: Optional[str], args: Sequence[object]) -> tuple[str, ...]:
    values = [os.fspath(v) for v in args]
    if not _is_wsl_backend(backend):
        return tuple(backend.command(tool, values))

    builder = getattr(backend, "execution_command", None)
    if not callable(builder):
        raise CMTKProcessError("WSL CMTK backend does not provide an execution command.")
    inner = [str(v) for v in builder(tool, values)]
    if not inner or not inner[0].strip():
        raise CMTKProcessError("WSL CMTK backend produced an empty execution command.")

    # Force a child session and wait for it. The constant shell snippet emits
    # the Linux session PID, then exec() replaces that same process with CMTK.
    # This preserves process-group cancellation while keeping wsl.exe attached
    # until the actual CMTK command terminates.
    return tuple(
        _wsl_prefix(backend)
        + [
            "/usr/bin/setsid",
            "--fork",
            "--wait",
            "/bin/sh",
            "-c",
            _WSL_SESSION_SCRIPT,
            "madi3d-cmtk",
            *inner,
        ]
    )


def _send_wsl_group_signal(backend, linux_pid: int, signal_name: str) -> bool:
    runner = getattr(backend, "runner", subprocess.run)
    cmd = (
        _wsl_prefix(backend)
        + ["/usr/bin/kill", f"-{signal_name}", "--", f"-{int(linux_pid)}"]
    )
    try:
        proc = runner(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
        )
        return int(getattr(proc, "returncode", 1)) == 0
    except Exception:
        return False


def _wait_process(proc, timeout: float) -> bool:
    try:
        proc.wait(timeout=max(0.0, float(timeout)))
        return True
    except Exception:
        return proc.poll() is not None


def _terminate_managed_process(
    backend,
    proc,
    *,
    linux_pid: Optional[int],
    grace_seconds: float,
) -> bool:
    """Terminate the managed compute process and its descendants."""
    if proc.poll() is not None:
        return True

    grace = max(0.0, float(grace_seconds))
    if _is_wsl_backend(backend) and linux_pid:
        _send_wsl_group_signal(backend, linux_pid, "TERM")
        if _wait_process(proc, grace):
            return True
        _send_wsl_group_signal(backend, linux_pid, "KILL")
        if _wait_process(proc, max(1.0, min(5.0, grace or 1.0))):
            return True
        # Linux received SIGKILL already; clean up the Windows proxy last.
        try:
            proc.kill()
        except Exception:
            pass
        return _wait_process(proc, 2.0)

    if os.name != "nt":
        pid = getattr(proc, "pid", None)
        if pid:
            try:
                os.killpg(int(pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
        else:
            try:
                proc.terminate()
            except Exception:
                pass
        if _wait_process(proc, grace):
            return True
        if pid:
            try:
                os.killpg(int(pid), signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        else:
            try:
                proc.kill()
            except Exception:
                pass
        return _wait_process(proc, max(1.0, min(5.0, grace or 1.0)))

    try:
        proc.terminate()
    except Exception:
        pass
    if _wait_process(proc, grace):
        return True
    try:
        proc.kill()
    except Exception:
        pass
    return _wait_process(proc, max(1.0, min(5.0, grace or 1.0)))


def _reader_events(name, stream, events, sentinel, *, wsl_control=False):
    try:
        if stream is None:
            return
        if not wsl_control:
            for text in _decoded_chunks(stream):
                events.put((name, text))
            return

        pending = ""
        for text in _decoded_chunks(stream):
            pending += text
            lines = pending.splitlines(keepends=True)
            if lines and not lines[-1].endswith(("\n", "\r")):
                pending = lines.pop()
            else:
                pending = ""
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(WSL_PID_PREFIX):
                    raw_pid = stripped[len(WSL_PID_PREFIX):].strip()
                    try:
                        events.put(("control_pid", int(raw_pid)))
                    except Exception:
                        events.put(("control_error", f"Invalid WSL PID marker: {stripped}"))
                else:
                    events.put((name, line))
        if pending:
            stripped = pending.strip()
            if stripped.startswith(WSL_PID_PREFIX):
                raw_pid = stripped[len(WSL_PID_PREFIX):].strip()
                try:
                    events.put(("control_pid", int(raw_pid)))
                except Exception:
                    events.put(("control_error", f"Invalid WSL PID marker: {stripped}"))
            else:
                events.put((name, pending))
    finally:
        events.put((name, sentinel))


def run_cmtk_streaming(
    backend,
    tool: Optional[str],
    args: Sequence[object] = (),
    *,
    on_stdout: TextCallback = None,
    on_stderr: TextCallback = None,
    cancel_check: CancelCheck = None,
    timeout: Optional[float] = None,
    terminate_grace: float = 5.0,
    capture_limit: Optional[int] = DEFAULT_CAPTURE_LIMIT,
    check: bool = True,
    popen_factory=subprocess.Popen,
) -> CMTKProcessResult:
    """Run one CMTK command with owned process lifetime and streamed output."""
    command = _launch_command(backend, tool, args)
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "bufsize": 0,
    }
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    try:
        proc = popen_factory(list(command), **kwargs)
    except Exception as exc:
        if is_missing_cmtk_command_error(exc):
            notifier = getattr(backend, "_notify_missing_tool", None)
            if callable(notifier):
                notifier(tool, exc)
        raise CMTKProcessError(
            f"Could not start CMTK command {tool or getattr(backend, 'cmtk_command', 'cmtk')!r}: {exc}"
        ) from exc

    events = queue.Queue()
    sentinel = object()
    is_wsl = _is_wsl_backend(backend)
    threading.Thread(
        target=_reader_events,
        args=("stdout", proc.stdout, events, sentinel),
        kwargs={"wsl_control": is_wsl},
        daemon=True,
    ).start()
    threading.Thread(
        target=_reader_events,
        args=("stderr", proc.stderr, events, sentinel),
        daemon=True,
    ).start()

    captured = {
        "stdout": _TailBuffer(capture_limit),
        "stderr": _TailBuffer(capture_limit),
    }
    finished = set()
    linux_pid = None
    cancelled = False
    timed_out = False
    started = time.monotonic()

    def consume(name, payload):
        nonlocal linux_pid
        if name == "control_pid":
            pid = int(payload)
            if pid <= 0:
                raise CMTKProcessError(f"Invalid WSL CMTK process id: {pid}")
            if linux_pid is not None and linux_pid != pid:
                raise CMTKProcessError("WSL CMTK job reported more than one process id.")
            linux_pid = pid
            return
        if name == "control_error":
            raise CMTKProcessError(str(payload))
        if payload is sentinel:
            finished.add(name)
            return
        text = str(payload)
        captured[name].append(text)
        callback = on_stdout if name == "stdout" else on_stderr
        if callback is not None:
            callback(text)

    try:
        while len(finished) < 2 or proc.poll() is None:
            try:
                name, payload = events.get(timeout=0.05)
                consume(name, payload)
            except queue.Empty:
                pass

            now = time.monotonic()
            if (
                is_wsl
                and linux_pid is None
                and proc.poll() is None
                and now - started >= WSL_HANDSHAKE_TIMEOUT
            ):
                raise CMTKProcessError(
                    "WSL CMTK job did not report its Linux process id; refusing "
                    "to continue with an unmanaged compute process."
                )

            if proc.poll() is None and cancel_check is not None and not cancelled:
                cancelled = bool(cancel_check())

            if (
                proc.poll() is None
                and timeout is not None
                and not timed_out
                and now - started >= max(0.0, float(timeout))
            ):
                timed_out = True

            if proc.poll() is None and (cancelled or timed_out):
                if is_wsl and linux_pid is None:
                    # A cold WSL distribution can take more than one second to
                    # emit the ownership marker.  Keep waiting until the normal
                    # handshake deadline instead of abandoning a live job.
                    continue
                if not _terminate_managed_process(
                    backend,
                    proc,
                    linux_pid=linux_pid,
                    grace_seconds=terminate_grace,
                ):
                    raise CMTKProcessError(
                        "CMTK process did not terminate after TERM/KILL cleanup."
                    )

        while True:
            try:
                name, payload = events.get_nowait()
            except queue.Empty:
                break
            consume(name, payload)

    except BaseException:
        if proc.poll() is None:
            _terminate_managed_process(
                backend,
                proc,
                linux_pid=linux_pid,
                grace_seconds=min(max(0.0, float(terminate_grace)), 1.0),
            )
        raise
    finally:
        for stream in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    result = CMTKProcessResult(
        command=command,
        returncode=int(proc.returncode if proc.returncode is not None else -1),
        stdout=captured["stdout"].text(),
        stderr=captured["stderr"].text(),
        cancelled=cancelled,
        timed_out=timed_out,
    )

    if cancelled:
        raise CMTKProcessError("CMTK command was cancelled.", result)
    if timed_out:
        raise CMTKProcessError("CMTK command timed out.", result)
    if check and result.returncode != 0:
        detail = result.error_text()
        if is_missing_cmtk_command_error(result):
            notifier = getattr(backend, "_notify_missing_tool", None)
            if callable(notifier):
                notifier(tool, detail)
        message = f"CMTK {tool or 'command'} failed with exit code {result.returncode}."
        if detail:
            message += f"\n{detail}"
        raise CMTKProcessError(message, result)
    return result


__all__ = [
    "CMTKProcessError",
    "CMTKProcessResult",
    "DEFAULT_CAPTURE_LIMIT",
    "WSL_PID_PREFIX",
    "run_cmtk_streaming",
]
