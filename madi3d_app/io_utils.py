"""Qt-free cancellation, resource checks and atomic file publication."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
import time

CHUNK_BYTES = 1024 * 1024
MEMORY_BUDGET_FRACTION = 0.90
_WINDOWS_SHARING_VIOLATIONS = {32, 33}
_PUBLICATION_RETRY_DELAYS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.4, 0.4, 0.4)


def file_revision(path):
    """Cheap change detection, not proof of byte identity."""
    stat = os.stat(path)
    return (os.path.normcase(os.path.abspath(path)), stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns)


def check_cancelled(cancel_check=None):
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Operation cancelled before publication.")


def available_memory_bytes():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except (ImportError, OSError):
        return 0


def check_resources(*, memory_bytes=0, disk_bytes=0, directory=None):
    available = available_memory_bytes()
    budget = int(available * MEMORY_BUDGET_FRACTION) if available else 512 * 1024**2
    if int(memory_bytes) > budget:
        basis = (
            f"{MEMORY_BUDGET_FRACTION:.0%} of available RAM"
            if available else "available RAM could not be determined"
        )
        raise MemoryError(
            f"This operation needs approximately {memory_bytes / 1024**3:.2f} GiB of additional memory; "
            f"the current memory limit is {budget / 1024**3:.2f} GiB ({basis}). "
            "Close other applications, unload data, or reduce the selection or output size."
        )
    if disk_bytes:
        folder = Path(directory or tempfile.gettempdir()).absolute()
        while not folder.exists() and folder != folder.parent:
            folder = folder.parent
        free = shutil.disk_usage(folder).free
        if int(disk_bytes) + 64 * 1024**2 > free:
            raise OSError("There is not enough free space for the output and its temporary files.")


def copy_file(source, destination, *, cancel_check=None):
    with open(source, "rb") as incoming, open(destination, "wb") as outgoing:
        while True:
            check_cancelled(cancel_check)
            chunk = incoming.read(CHUNK_BYTES)
            if not chunk:
                break
            outgoing.write(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    shutil.copystat(source, destination)


def _retry_windows_sharing_violation(operation):
    """Retry transient Windows sharing violations from sync/AV file watchers."""
    for delay in (*_PUBLICATION_RETRY_DELAYS, None):
        try:
            return operation()
        except OSError as exc:
            if getattr(exc, "winerror", None) not in _WINDOWS_SHARING_VIOLATIONS or delay is None:
                raise
            time.sleep(delay)


@contextmanager
def staged_file(destination, *, cancel_check=None, publish=True):
    """Publish a nonempty sibling file only after the writer succeeds."""
    target = Path(destination)
    suffix = "".join(target.suffixes) or ".tmp"
    fd, name = tempfile.mkstemp(prefix=".madi3d-", suffix=suffix, dir=target.parent)
    os.close(fd)
    stage = Path(name)
    failure = None
    try:
        check_cancelled(cancel_check)
        yield str(stage)
        if not publish:
            return
        check_cancelled(cancel_check)
        if not stage.is_file() or stage.stat().st_size <= 0:
            raise OSError("The writer did not produce a complete output file.")
        with stage.open("rb+") as stream:
            os.fsync(stream.fileno())
        check_cancelled(cancel_check)
        _retry_windows_sharing_violation(lambda: os.replace(stage, target))
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            _retry_windows_sharing_violation(lambda: stage.unlink(missing_ok=True))
        except OSError:
            if failure is None:
                raise


def publish_file_pair(stage, destination, companion_stage, companion_destination, *, cancel_check=None, replace_func=None,
                      optional_history=False, recovery_path=None, on_commit=None):
    """Publish under an import barrier; restore prior files on ordinary failure.

    Two renames are not atomic. The barrier survives an interrupted process or
    failed rollback, so consumers cannot accept a partial pair. Recovery copies
    are retained only when restoration itself fails. ``on_commit`` reports the
    guarded pair's commit before any backup cleanup can raise.
    """
    target, companion = Path(destination), Path(companion_destination)
    replace = replace_func or os.replace
    if optional_history:
        # Retire old metadata before replacing independently interpretable data.
        # Failure to make it harmless aborts before touching the primary file.
        previous = None
        check_cancelled(cancel_check)
        if companion.exists():
            fd, name = tempfile.mkstemp(prefix=".madi3d-history-", dir=companion.parent)
            os.close(fd)
            previous = Path(name)
            try:
                os.replace(companion, previous)
            except BaseException:
                previous.unlink(missing_ok=True)
                raise
        try:
            try:
                check_cancelled(cancel_check)
                replace(stage, target)
            except BaseException:
                if previous is not None:
                    os.replace(previous, companion)
                raise
            # Commit point: cancellation must not report the primary as unwritten.
            if companion_stage is not None:
                try:
                    replace(companion_stage, companion)
                except OSError as exc:
                    return f"Data saved; provenance companion failed: {exc}"
            return None
        finally:
            if previous is not None:
                previous.unlink(missing_ok=True)
    barrier = Path(str(companion) + ".publishing")
    backups = {}
    published = []
    restored = False
    with barrier.open("x", encoding="utf-8") as stream:
        stream.write("MADI3D export publication in progress\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        for staged in (stage, companion_stage):
            if staged is not None:
                output = Path(staged)
                if not output.is_file() or output.stat().st_size <= 0:
                    raise OSError("The writer did not produce a complete output file.")
                with output.open("rb+") as stream:
                    os.fsync(stream.fileno())
        for path in (target, companion):
            if path.is_file():
                if path == target and recovery_path is not None:
                    backups[path] = Path(recovery_path)
                    continue
                fd, backup = tempfile.mkstemp(prefix=".madi3d-recovery-", dir=path.parent)
                os.close(fd)
                backups[path] = Path(backup)
                copy_file(path, backup, cancel_check=cancel_check)
        check_cancelled(cancel_check)
        # No cancellation boundary after the first visible replacement.
        published.append(target)
        replace(stage, target)
        published.append(companion)
        if companion_stage is None:
            companion.unlink(missing_ok=True)
        else:
            replace(companion_stage, companion)
        restored = True
        if on_commit is not None:
            on_commit()
    except BaseException as failure:
        try:
            for path in reversed(published):
                if path in backups:
                    if path == target and recovery_path is not None:
                        with staged_file(path) as restored_stage:
                            copy_file(recovery_path, restored_stage)
                    else:
                        os.replace(backups[path], path)
                else:
                    path.unlink(missing_ok=True)
            restored = True
        except OSError as recovery_failure:
            raise OSError(f"Export failed and prior output could not be restored. Import is blocked by {barrier}; "
                          f"recovery copies: {list(backups.values())}") from recovery_failure
        raise failure
    finally:
        if restored or not published:
            for backup in backups.values():
                if recovery_path is None or backup != Path(recovery_path):
                    backup.unlink(missing_ok=True)
            barrier.unlink(missing_ok=True)
