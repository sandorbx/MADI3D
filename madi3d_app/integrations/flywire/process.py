"""Supervise opaque network/native work outside the application process."""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time


class WorkerCleanupError(RuntimeError):
    """Termination was not confirmed; the caller must retain its reservation."""

    def __init__(self, message="FlyWire worker termination could not be confirmed.", *, retirement_check=None):
        super().__init__(message)
        self.retirement_check = retirement_check


class _WorkerRetirement:
    """The supervisor's retained handles, queried without killing or waiting."""

    def __init__(self, process, observed):
        self.process = process
        self.observed = tuple(observed.values())

    def __call__(self):
        import psutil
        if self.process.poll() is None:
            return False
        for child in self.observed:
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    return False
            except psutil.NoSuchProcess:
                pass
            except psutil.Error:
                return False
        return True


def _darwin_address_space_bytes():
    """Read raw top-level Mach mappings, including reserved GPU ranges.

    task_info/psutil virtual sizes can exclude reserved regions, whereas XNU's
    RLIMIT_AS setter and enforcement both use the unadjusted map size.
    """
    import ctypes

    system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    system.mach_task_self.restype = ctypes.c_uint32
    system.mach_task_self.argtypes = ()
    region = system.mach_vm_region_recurse
    region.restype = ctypes.c_int
    region.argtypes = (ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64),
                       ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
                       ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint32))
    task = system.mach_task_self()
    address, size = ctypes.c_uint64(), ctypes.c_uint64()
    # VM_REGION_SUBMAP_SHORT_INFO_64: 48 bytes, packed to 4-byte alignment on
    # both supported architectures. Only address/size are needed here.
    info = (ctypes.c_int32 * 12)()
    total = 0
    while True:
        start = address.value
        depth, count = ctypes.c_uint32(0), ctypes.c_uint32(len(info))
        error = region(task, ctypes.byref(address), ctypes.byref(size),
                       ctypes.byref(depth), info, ctypes.byref(count))
        if error == 1:  # KERN_INVALID_ADDRESS: no more mappings
            break
        if error:
            raise OSError(f"Cannot read worker address space (Mach error {error}).")
        end = address.value + size.value
        if address.value < start or size.value == 0 or end >= 2**64:
            raise OSError("Invalid worker address-space mapping.")
        total += size.value
        address.value = end
    if not total:
        raise OSError("Worker address-space map is empty.")
    return total


def _allocation_limit(max_bytes):
    """Install an explicit allocation ceiling before importing decoders.

    Windows accounts committed memory across the job; Linux limits absolute
    virtual space. Darwin bounds additional virtual mappings above the measured
    bootstrap map, so reserved mappings do not consume decoder headroom. This
    is not a physical-memory cap; admission and RSS supervision remain separate.
    Return the Windows job handle so it stays alive until worker process exit.
    """
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("Worker allocation ceiling must be a positive integer or None.")
    if os.name != "nt":
        if max_bytes is None:
            return None
        import resource
        address_bytes = max_bytes
        if sys.platform == "darwin":
            address_bytes += _darwin_address_space_bytes()
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        ceiling = address_bytes if hard == resource.RLIM_INFINITY else min(address_bytes, hard)
        if ceiling >= 2**63 - 1:
            raise ValueError("Worker address-space ceiling must remain finite.")
        resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
        if resource.getrlimit(resource.RLIMIT_AS) != (ceiling, ceiling):
            raise OSError("Worker address-space limit was not installed.")
        return None

    import ctypes
    from ctypes import wintypes

    class BasicLimit(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimit), ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimit()
    limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
    if max_bytes is not None:
        limits.BasicLimitInformation.LimitFlags |= 0x200  # JOB_MEMORY
        limits.JobMemoryLimit = max_bytes
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        error = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(error)
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        error = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(error)
    return job


def worker_entry(request_path, max_bytes):
    """Private early executable dispatch, shared by source and frozen entrypoints."""
    # Native libraries must not reserve a host-sized thread pool/address space
    # before the explicit worker budget can contain them.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    phase = "allocation limit"
    try:
        if type(max_bytes) is int and max_bytes == 0:
            max_bytes = None  # CLI sentinel; configured ceilings must be positive.
        # Leave the Windows job handle open until OS process teardown. Closing
        # it here would terminate this worker before returning its exit code.
        _job_handle = _allocation_limit(max_bytes)
        phase = "decoder import"
        from .backend import main
        return main(request_path)
    except Exception as exc:
        import json
        from pathlib import Path
        reason = f"FlyWire worker {phase} failed ({type(exc).__name__})."
        if isinstance(exc, ImportError):
            reason = f"Install the optional FlyWire decoder dependencies: {str(exc)[:300]}"
        elif isinstance(exc, OSError):
            reason += f" OS error {getattr(exc, 'winerror', None) or exc.errno}."
        Path(request_path).with_name("result.json").write_text(json.dumps({
            "status": "failed", "error_type": type(exc).__name__, "reason": reason}), encoding="utf-8")
        return 1


def run_bounded(command, *, deadline_seconds, max_rss_bytes, cancelled=lambda: False):
    """Reap the worker and observed descendants on every exit path.

    Includes Windows interpreter redirectors. RSS is a sampled emergency stop,
    not an allocation bound or a replacement for admission before spawning.
    The backend runs serially and starts no independent process pools.
    """
    import psutil

    if (deadline_seconds is not None
            and (not math.isfinite(deadline_seconds) or deadline_seconds <= 0)
            or (max_rss_bytes is not None and
                (type(max_rss_bytes) is not int or max_rss_bytes <= 0))):
        raise ValueError("Any worker deadline must be finite and positive, and any RSS ceiling must be a positive integer.")
    started = time.monotonic()
    peak = 0
    reason = None
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    observed = {}
    try:
        observer = psutil.Process(process.pid)
        observed[observer.pid] = observer
        while process.poll() is None:
            try:
                observed.update((p.pid, p) for p in observer.children(recursive=True))
            except psutil.NoSuchProcess:
                pass
            total = 0
            for child in observed.values():
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, total)
            if cancelled():
                reason = "cancelled"
            elif deadline_seconds is not None and time.monotonic() - started >= deadline_seconds:
                reason = "deadline"
            elif max_rss_bytes is not None and total > max_rss_bytes:
                reason = "memory_limit"
            if reason:
                break
            time.sleep(0.05)
    except psutil.NoSuchProcess:
        # A short-lived child may have completed before its first observation.
        pass
    finally:
        try:
            for parent in tuple(observed.values()):
                try:
                    observed.update((p.pid, p) for p in parent.children(recursive=True))
                except psutil.NoSuchProcess:
                    pass
            for child in reversed(tuple(observed.values())):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            _, alive = psutil.wait_procs(list(observed.values()), timeout=5)
            if alive:
                raise WorkerCleanupError("FlyWire worker descendants did not terminate.",
                                         retirement_check=_WorkerRetirement(process, observed))
        except (psutil.Error, OSError, subprocess.TimeoutExpired) as exc:
            raise WorkerCleanupError("FlyWire worker termination could not be confirmed.",
                                     retirement_check=_WorkerRetirement(process, observed)) from exc
    return {"status": reason or ("ok" if process.returncode == 0 else "failed"),
            "exit_code": process.returncode, "elapsed_seconds": round(time.monotonic() - started, 3),
            "peak_rss_bytes": peak}


if __name__ == "__main__":
    raise SystemExit(worker_entry(sys.argv[1], int(sys.argv[2])))
