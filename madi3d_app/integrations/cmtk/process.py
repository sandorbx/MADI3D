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

from madi3d_app.diagnostics import report_diagnostic
from madi3d_app.integrations.cmtk.backend import (
    CMTKError, host_wsl_version, is_missing_cmtk_command_error, wsl_runtime_guidance,
)


TextCallback = Optional[Callable[[str], None]]
CancelCheck = Optional[Callable[[], bool]]

DEFAULT_CAPTURE_LIMIT = 1024 * 1024
STREAM_EVENT_LIMIT = 64  # At most 64 decoded 4 KiB chunks, plus reader-local chunks.
POST_EXIT_DRAIN_TIMEOUT = 5.0
WSL_PID_PREFIX = "__MADI3D_CMTK_PID__="
# Includes distribution cold boot, not just shell execution. One minute gives
# cold starts headroom while bounding an unresponsive launch; never retry it.
WSL_HANDSHAKE_TIMEOUT = 60.0
WSL_STARTUP_NOTICE = 5.0
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


def _runtime_failure_message(backend, message):
    if _is_wsl_backend(backend):
        return message + "\n" + wsl_runtime_guidance(host_wsl_version(backend.runner))
    return message


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


def _posix_group_absent(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


@dataclass(frozen=True)
class _WSLProbe:
    state: str
    returncode: Optional[int]
    stdout: str = ""
    stderr: str = ""


def _diagnostic_tail(value, limit=2048):
    if isinstance(value, bytes):
        value = value.decode(_stream_encoding(value), errors="replace")
    return str(value or "")[-limit:].strip()


def _wsl_group_probe(backend, linux_pid):
    """Distinguish an absent group from a failed attempt to query it."""
    try:
        result = backend.runner(
            _wsl_prefix(backend) + ["/usr/bin/env", "LC_ALL=C", "/usr/bin/kill", "-0", "--", f"-{int(linux_pid)}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=5,
        )
        rc = int(result.returncode)
        stdout = _diagnostic_tail(getattr(result, "stdout", b""))
        stderr = _diagnostic_tail(getattr(result, "stderr", b""))
        if rc == 0 and not stdout and not stderr:
            state = "present"
        elif rc != 0 and "no such process" in stderr.lower():
            state = "absent"
        else:
            state = "probe-failed"
        return _WSLProbe(state, rc, stdout, stderr)
    except Exception as exc:
        return _WSLProbe("probe-failed", None, stderr=_diagnostic_tail(str(exc)))


def _wsl_teardown_message(backend, proc, linux_pid, probe, finished):
    """Capture bounded evidence for this session and its owned process group."""
    rows = "unavailable"
    if linux_pid:
        # setsid makes the marker both SID and PGID. Restrict ps to that
        # session, then retain only the owned group; bound output in the guest.
        script = "ps -g \"$1\" -o pid=,ppid=,pgid=,sid=,stat=,args= | awk -v pgid=\"$1\" '$3 == pgid' | head -c 8192"
        try:
            result = backend.runner(
                _wsl_prefix(backend) + ["/bin/sh", "-c", script, "madi3d-diagnostic", str(linux_pid)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=5,
            )
            if int(result.returncode) == 0 and not _diagnostic_tail(getattr(result, "stderr", b"")):
                rows = _diagnostic_tail(getattr(result, "stdout", b""), 8192) or "<none>"
            else:
                rows = f"unavailable (rc={result.returncode}, stderr={_diagnostic_tail(getattr(result, 'stderr', b'')) or '<empty>'})"
        except Exception as exc:
            rows = f"unavailable ({_diagnostic_tail(str(exc))})"
    proxy_rc = proc.poll()
    probe_text = (f"{probe.state}, rc={probe.returncode}, stderr={probe.stderr or '<empty>'}, "
                  f"stdout={probe.stdout or '<empty>'}") if probe else "not performed"
    return ("CMTK owned-process teardown could not be confirmed. "
            f"WSL distro={getattr(backend, 'distro', '')}, marker PID / expected PGID={linux_pid}; "
            f"probe: {probe_text}; proxy exited={proxy_rc is not None}, rc={proxy_rc}; "
            f"output EOF: stdout={'stdout' in finished}, stderr={'stderr' in finished}.\n"
            f"Owned group processes (PID PPID PGID SID STAT CMD): {rows}")


def _terminate_managed_process(
    backend,
    proc,
    *,
    linux_pid: Optional[int],
    grace_seconds: float,
    windows_job=None,
    wsl_probe=None,
) -> bool:
    """Terminate the managed compute process and its descendants."""
    grace = max(0.0, float(grace_seconds))
    if _is_wsl_backend(backend) and linux_pid:
        _send_wsl_group_signal(backend, linux_pid, "TERM")
        # Launcher exit is not group exit: descendants may still own the pipes.
        time.sleep(grace)
        _send_wsl_group_signal(backend, linux_pid, "KILL")
        exited = _wait_process(proc, max(1.0, min(5.0, grace or 1.0)))
        if not exited:
            # Linux received SIGKILL already; clean up the Windows proxy last.
            try:
                proc.kill()
            except Exception:
                pass
            exited = _wait_process(proc, 2.0)
        probe = _wsl_group_probe(backend, linux_pid)
        if wsl_probe is not None:
            wsl_probe.append(probe)
        return exited and probe.state == "absent"

    if _is_wsl_backend(backend):
        # A proxy without its Linux ownership marker cannot certify guest exit.
        if windows_job is not None:
            windows_job.terminate()
        else:
            try:
                proc.kill()
            except Exception:
                pass
        _wait_process(proc, 2.0)
        return False

    if windows_job is not None:
        windows_job.terminate()
        return _wait_process(proc, max(1.0, min(5.0, grace or 1.0)))

    if os.name != "nt":
        pid = getattr(proc, "pid", None)
        if pid:
            pid = int(pid)
            try:
                os.killpg(pid, signal.SIGTERM)
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
        deadline = time.monotonic() + grace
        while True:
            # Reap the leader, but require absence of the whole process group.
            if proc.poll() is not None and pid and _posix_group_absent(pid):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))
        if pid:
            try:
                os.killpg(pid, signal.SIGKILL)
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
        exited = _wait_process(proc, max(1.0, min(5.0, grace or 1.0)))
        return bool(exited and pid and _posix_group_absent(pid))

    raise CMTKProcessError("Native Windows CMTK cleanup requires its owned Job Object.")


def _reader_events(name, stream, events, sentinel, stop, *, wsl_control=False,
                   ownership=None):
    def send(kind, payload):
        while not stop.is_set():
            try:
                events.put((kind, payload), timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    try:
        if stream is None:
            return
        pending = ""
        skipping_line = False

        def control_line(line):
            stripped = line.strip()
            if not stripped.startswith(WSL_PID_PREFIX):
                return False
            raw_pid = stripped[len(WSL_PID_PREFIX):]
            if (not raw_pid.isascii() or not raw_pid.isdecimal()
                    or not 1 < int(raw_pid) <= 2147483647):
                send("control_error", "Malformed WSL PID marker.")
                return False
            pid = int(raw_pid)
            # Keep evidence even if a callback fails or the owner stops draining
            # the queue. Only this stdout reader writes the single PID.
            if ownership is not None:
                ownership.append(pid)
            send("control_pid", pid)
            return True

        for text in _decoded_chunks(stream):
            if stop.is_set():
                break
            if not wsl_control:
                if not send(name, text):
                    break
                continue
            # WSL may print startup diagnostics before invoking our script.
            # Scan line starts until ownership arrives, buffering at most 128
            # characters plus one decoded chunk, even for unterminated lines.
            pending += text
            while "\n" in pending:
                line, _, pending = pending.partition("\n")
                if not skipping_line and control_line(line):
                    wsl_control = False
                    break
                send(name, line + "\n")
                skipping_line = False
            if not wsl_control:
                if pending:
                    send(name, pending)
                pending = ""
            elif skipping_line or len(pending) > 128:
                send(name, pending)
                pending = ""
                skipping_line = True
        if pending:
            if skipping_line or not control_line(pending):
                send(name, pending)
    except Exception as exc:
        if not stop.is_set():
            send("reader_error", exc)
    finally:
        send(name, sentinel)


class _ProcessRetirement:
    """Retain exact native ownership evidence after failed termination."""

    def __init__(self, backend, proc, linux_pid, windows_job):
        self.backend, self.proc = backend, proc
        self.linux_pid, self.windows_job = linux_pid, windows_job
        self.probe_thread = None
        self.probe = None
        self.confirmed = False
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self._check()

    def _check(self):
        if self.confirmed:
            return True
        if self.proc.poll() is None:
            return False
        if _is_wsl_backend(self.backend):
            if self.linux_pid is None:
                return False
            if self.probe_thread is not None and self.probe_thread.is_alive():
                return False
            if self.probe is not None and self.probe.state == "absent":
                self.confirmed = True
            else:
                # Only query ownership; never repeat termination to check exit.
                def query():
                    self.probe = _wsl_group_probe(self.backend, self.linux_pid)
                self.probe_thread = threading.Thread(target=query, daemon=True,
                                                     name="cmtk-retirement-probe")
                self.probe_thread.start()
                return False
        elif self.windows_job is not None:
            self.confirmed = self.windows_job.retirement_confirmed()
        elif os.name != "nt":
            self.confirmed = _posix_group_absent(int(self.proc.pid))
        if self.confirmed and self.windows_job is not None:
            self.windows_job.close()
        return self.confirmed


def run_cmtk_streaming(
    backend,
    tool: Optional[str],
    args: Sequence[object] = (),
    *,
    on_command: Optional[Callable[[tuple[str, ...]], None]] = None,
    on_stdout: TextCallback = None,
    on_stderr: TextCallback = None,
    cancel_check: CancelCheck = None,
    timeout: Optional[float] = None,
    terminate_grace: float = 5.0,
    capture_limit: Optional[int] = DEFAULT_CAPTURE_LIMIT,
    post_exit_timeout: float = POST_EXIT_DRAIN_TIMEOUT,
    check: bool = True,
    stdin=None,
    stdout_capture_limit: Optional[int] = None,
    on_retirement_pending: Optional[Callable[[Callable[[], bool]], None]] = None,
    popen_factory=subprocess.Popen,
) -> CMTKProcessResult:
    """Run one CMTK command with owned process lifetime and streamed output."""
    command = _launch_command(backend, tool, args)
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "bufsize": 0,
    }
    if stdin is not None:
        # The caller retains ownership of the stream. This is used for bounded
        # file-backed input such as large point clouds; it does not change the
        # managed process or output-reader lifecycle.
        kwargs["stdin"] = stdin
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    windows_job = None
    try:
        if os.name == "nt":
            from madi3d_app.integrations.cmtk._windows_job import WindowsJob

            windows_job = WindowsJob()
            kwargs["creationflags"] |= 0x00000004  # CREATE_SUSPENDED
        if on_command is not None:
            on_command(command)
        proc = popen_factory(list(command), **kwargs)
        if windows_job is not None:
            try:
                windows_job.attach_and_resume(proc)
            except BaseException:
                proc.kill()
                _wait_process(proc, 2.0)
                for stream in (proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()
                raise
    except Exception as exc:
        if windows_job is not None:
            windows_job.close()
        if is_missing_cmtk_command_error(exc):
            notifier = getattr(backend, "_notify_missing_tool", None)
            if callable(notifier):
                notifier(tool, exc)
        raise CMTKProcessError(
            _runtime_failure_message(
                backend,
                f"Could not start CMTK command {tool or getattr(backend, 'cmtk_command', 'cmtk')!r}: {exc}",
            )
        ) from exc

    retain_process = getattr(backend, "_retain_managed_process", None)
    owned_process = retain_process(proc) if callable(retain_process) else None
    events = queue.Queue(maxsize=STREAM_EVENT_LIMIT)
    sentinel = object()
    stop = threading.Event()
    is_wsl = _is_wsl_backend(backend)
    ownership = []
    readers = [
        threading.Thread(
            target=_reader_events,
            args=(name, stream, events, sentinel, stop),
            kwargs={"wsl_control": is_wsl and name == "stdout",
                    "ownership": ownership},
            name=f"cmtk-{name}-{proc.pid}",
            daemon=True,
        )
        for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr))
    ]
    started_readers = []

    captured = {
        "stdout": _TailBuffer(
            capture_limit if stdout_capture_limit is None else stdout_capture_limit
        ),
        "stderr": _TailBuffer(capture_limit),
    }
    # Streams the workflow does not consume would otherwise vanish into
    # the pipe; forward them to the application log under the tool name.
    forward_labels = {
        name: f"CMTK {tool or 'command'} [{name}]" for name in ("stdout", "stderr")
    }
    finished = set()
    linux_pid = None
    startup_error = None
    startup_notice = False
    cancelled = False
    timed_out = False
    started = time.monotonic()
    exited_at = None
    callback_seconds = 0.0
    callback_seconds_at_exit = 0.0
    terminated = False
    wsl_probes = []
    terminated_at = None
    callback_seconds_at_termination = 0.0

    def consume(name, payload):
        nonlocal linux_pid, callback_seconds, startup_error
        if name == "control_pid":
            pid = int(payload)
            if pid <= 0:
                raise CMTKProcessError(f"Invalid WSL CMTK process id: {pid}")
            if linux_pid is not None and linux_pid != pid:
                raise CMTKProcessError("WSL CMTK job reported more than one process id.")
            linux_pid = pid
            return
        if name == "control_error":
            # Continue collecting diagnostics and ownership until the marker,
            # EOF, or deadline. A bad line must not hide a later valid PID.
            startup_error = str(payload)
            return
        if name == "reader_error":
            raise CMTKProcessError(f"Could not read CMTK output: {payload}") from payload
        if payload is sentinel:
            finished.add(name)
            return
        text = str(payload)
        captured[name].append(text)
        callback = on_stdout if name == "stdout" else on_stderr
        if callback is not None:
            callback_started = time.monotonic()
            try:
                callback(text)
            finally:
                callback_seconds += time.monotonic() - callback_started
        else:
            # The workflow did not attach a consumer for this stream, so the
            # text is surfaced as console-attributed output with the tool
            # name. Consumed streams are the workflow owner's logging
            # responsibility, so the same bytes are never logged twice.
            report_diagnostic(
                f"{forward_labels[name]}: {text}",
                severity="Warning" if name == "stderr" else "Info",
            )

    try:
        for reader in readers:
            reader.start()
            started_readers.append(reader)
        while len(finished) < 2 or proc.poll() is None:
            try:
                name, payload = events.get(timeout=0.05)
                consume(name, payload)
            except queue.Empty:
                pass

            if linux_pid is None and ownership:
                linux_pid = ownership[0]
            now = time.monotonic()
            if proc.poll() is not None and exited_at is None:
                exited_at = now
                callback_seconds_at_exit = callback_seconds
            if cancel_check is not None and not cancelled:
                cancelled = bool(cancel_check())

            if (
                timeout is not None
                and not timed_out
                and now - started >= max(0.0, float(timeout))
            ):
                timed_out = True

            if is_wsl:
                if linux_pid is None and not startup_notice and now - started >= WSL_STARTUP_NOTICE:
                    startup_notice = True
                    report_diagnostic(
                        f"CMTK WSL startup is delayed (distro={getattr(backend, 'distro', '')}); "
                        f"waiting up to {WSL_HANDSHAKE_TIMEOUT:g}s for Linux ownership.",
                        severity="Warning",
                    )
                startup_finished = exited_at is not None and len(finished) == 2
                if startup_error and (linux_pid is not None or startup_finished
                                      or now - started >= WSL_HANDSHAKE_TIMEOUT):
                    raise CMTKProcessError(startup_error)
                if linux_pid is None and (startup_finished or now - started >= WSL_HANDSHAKE_TIMEOUT):
                    timed_out = timed_out or not startup_finished
                    reason = (f"WSL launcher exited early (exit code {proc.poll()}) without a Linux PID marker."
                              if startup_finished else
                              f"WSL startup unresponsive: no Linux PID marker within {WSL_HANDSHAKE_TIMEOUT:g}s.")
                    raise CMTKProcessError(
                        f"{reason} distro={getattr(backend, 'distro', '')}. "
                        "Linux ownership is unconfirmed; check WSL startup and the diagnostics below."
                    )

            if exited_at is not None and len(finished) < 2:
                # Slow durable/UI sinks may still have a bounded transport tail
                # to consume after process exit. Budget pipe waiting, not time
                # spent delivering that ordinary final output to the caller.
                drain_seconds = now - exited_at - (callback_seconds - callback_seconds_at_exit)
                if drain_seconds >= max(0.0, float(post_exit_timeout)):
                    timed_out = True

            if (cancelled or timed_out) and not terminated:
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
                    windows_job=windows_job,
                    wsl_probe=wsl_probes,
                ):
                    detail = (_wsl_teardown_message(backend, proc, linux_pid, wsl_probes[-1] if wsl_probes else None, finished)
                              if is_wsl else "CMTK process did not terminate after TERM/KILL cleanup.")
                    raise CMTKProcessError(detail)
                terminated = True
                terminated_at = time.monotonic()
                callback_seconds_at_termination = callback_seconds
            if terminated_at is not None and len(finished) < 2:
                drain_seconds = now - terminated_at - (callback_seconds - callback_seconds_at_termination)
                if drain_seconds >= 2.0:
                    raise CMTKProcessError("CMTK pipes remained open after group termination.")

        while True:
            try:
                name, payload = events.get_nowait()
            except queue.Empty:
                break
            consume(name, payload)

        # Retire owned descendants which closed their pipes before exiting.
        if not terminated:
            terminated = _terminate_managed_process(
                backend, proc, linux_pid=linux_pid, grace_seconds=0.0,
                windows_job=windows_job, wsl_probe=wsl_probes,
            )
            if not terminated:
                detail = (_wsl_teardown_message(backend, proc, linux_pid, wsl_probes[-1] if wsl_probes else None, finished)
                          if is_wsl else "CMTK descendants did not terminate after command completion.")
                raise CMTKProcessError(detail)

    except BaseException as exc:
        # The reader may have found ownership before a callback/control failure
        # prevented its queue event from being consumed.
        if linux_pid is None and ownership:
            linux_pid = ownership[0]
        if not terminated:
            terminated = _terminate_managed_process(
                backend,
                proc,
                linux_pid=linux_pid,
                grace_seconds=min(max(0.0, float(terminate_grace)), 1.0),
                windows_job=windows_job,
            )
        # Drain after termination before stopping readers: stderr may arrive
        # after the stdout control failure. Consume without invoking a failed
        # callback again, releasing queue backpressure with a bounded deadline.
        drain_deadline = time.monotonic() + 2.0
        while time.monotonic() < drain_deadline:
            try:
                name, payload = events.get(timeout=0.05)
            except queue.Empty:
                if not any(reader.is_alive() for reader in started_readers):
                    break
                continue
            if name in captured and payload is not sentinel:
                captured[name].append(str(payload))
        stop.set()
        if linux_pid is None and ownership:
            linux_pid = ownership[0]
            terminated = _terminate_managed_process(
                backend, proc, linux_pid=linux_pid,
                grace_seconds=min(max(0.0, float(terminate_grace)), 1.0),
                windows_job=windows_job,
            )
        if isinstance(exc, CMTKProcessError):
            result = CMTKProcessResult(
                command, int(proc.returncode if proc.returncode is not None else -1),
                captured["stdout"].text(), captured["stderr"].text(), cancelled, timed_out,
            )
            exc.result = result
            detail = result.error_text()
            if detail and detail not in str(exc):
                exc.args = (f"{exc}\n{detail}",)
            exc.args = (_runtime_failure_message(backend, str(exc)),)
        raise
    finally:
        stop.set()
        if windows_job is not None and (terminated or (
                owned_process is None and on_retirement_pending is None)):
            windows_job.close()
        for reader in started_readers:
            reader.join(timeout=2.0)
        if linux_pid is None and ownership:
            linux_pid = ownership[0]
        for stream in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        retirement = None
        if not terminated and (owned_process is not None or on_retirement_pending is not None):
            retirement = _ProcessRetirement(backend, proc, linux_pid, windows_job)
        if owned_process is not None:
            backend._settle_managed_process(
                owned_process, confirmed=bool(terminated),
                retirement_check=retirement)
        if retirement is not None and on_retirement_pending is not None:
            on_retirement_pending(retirement)
        if any(reader.is_alive() for reader in started_readers):
            raise CMTKProcessError(_runtime_failure_message(
                backend, "CMTK output readers did not finish after process cleanup."))

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
        raise CMTKProcessError(_runtime_failure_message(backend, "CMTK command timed out."), result)
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
