"""Qt-free cancellation, resource checks and atomic file publication."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import logging
import ntpath
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

CHUNK_BYTES = 1024 * 1024
PUBLICATION_MARKER_SUFFIX = ".publishing"
NUMBERED_FILENAME_RESERVE = "_4294967295"
_WINDOWS_FILE_LOCK_ERRORS = {5, 32, 33}
_PUBLICATION_RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.6)
_LOG = logging.getLogger(__name__)


def filename_unit_budget(directory=None):
    """Return the filename's UTF-16 budget for native Windows callers."""
    if directory is not None and os.name == "nt":
        parent = os.path.join(os.path.abspath(directory), "")
        if not parent.startswith("\\\\?\\"):
            return min(255, 259 - len(parent.encode("utf-16-le")) // 2)
    return 255


def bounded_filename(stem, suffix, *, reserve="", directory=None, hash_source=None):
    """Fit an automatic filename and its reserved suffix within portable limits."""
    original = str(stem)
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f\ud800-\udfff]', "_", original).rstrip(" .") or "output"
    if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³]",
                    stem.split(".")[0], re.IGNORECASE):
        stem = "_" + stem
    tail = suffix + reserve
    byte_budget = 255 - len(tail.encode("utf-8"))
    unit_budget = filename_unit_budget(directory) - len(tail.encode("utf-16-le")) // 2
    if len(stem.encode("utf-8")) <= byte_budget and len(stem.encode("utf-16-le")) // 2 <= unit_budget:
        return stem + suffix
    source = original if hash_source is None else str(hash_source)
    digest = "-" + hashlib.sha256(source.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    byte_budget -= len(digest)
    unit_budget -= len(digest)
    if min(byte_budget, unit_budget) < 0:
        raise OSError("The output folder is too deep for the filename and its required companion files. Choose a shorter folder.")
    prefix = stem.encode("utf-8")[:byte_budget].decode("utf-8", errors="ignore")
    prefix = prefix.encode("utf-16-le")[:2 * unit_budget].decode("utf-16-le", errors="ignore")
    return prefix.rstrip(" ._") + digest + suffix


def resolve_file_locator(value, base_directory):
    """Resolve a stored locator in the directory of the file that declared it."""
    raw = os.fspath(value)
    if not raw or "://" in raw:
        return raw
    drive, _ = ntpath.splitdrive(raw)
    if drive and (os.sep == "/" or not os.path.isabs(raw)):
        # A drive-relative spelling can also be an exact POSIX filename.
        if (
            os.sep != "/"
            or ntpath.isabs(raw)
            or not os.path.lexists(os.path.join(base_directory, raw))
        ):
            return raw
    if not os.path.isabs(raw):
        native = os.path.join(base_directory, raw)
        if os.sep == "/" and "\\" in raw and not os.path.lexists(native):
            # Backslashes are legal POSIX filename characters. An existing
            # literal reference takes precedence over Windows relative syntax.
            if raw.startswith("\\"):
                return raw  # Windows drive-rooted path with no drive specified.
            native = os.path.join(base_directory, raw.replace("\\", "/"))
        raw = native
    return os.path.abspath(raw)


def file_revision(path):
    """Cheap change detection, not proof of byte identity."""
    stat = os.stat(path)
    return (os.path.normcase(os.path.abspath(path)), stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns)


def check_cancelled(cancel_check=None):
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Operation cancelled before publication.")


def available_memory_bytes():
    """Return measured available RAM, including zero, or None if unavailable."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except (ImportError, OSError):
        return None


def check_resources(*, memory_bytes=0, disk_bytes=0, directory=None):
    """Check resources and return the RAM budget snapshot, or None if unknown."""
    available = available_memory_bytes()
    budget = available
    if budget is not None and int(memory_bytes) > budget:
        raise MemoryError(
            f"This operation needs approximately {memory_bytes / 1024**3:.2f} GiB of additional memory; "
            f"the currently available RAM is {budget / 1024**3:.2f} GiB. "
            "Close other applications, unload data, or reduce the selection or output size."
        )
    if disk_bytes:
        folder = Path(directory or tempfile.gettempdir()).absolute()
        while not folder.exists() and folder != folder.parent:
            folder = folder.parent
        free = shutil.disk_usage(folder).free
        if int(disk_bytes) + 64 * 1024**2 > free:
            raise OSError("There is not enough free space for the output and its temporary files.")
    return budget


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


def _retry_windows_file_lock(operation, *, operation_name, destination):
    """Bound retries for Windows locks; persistent access denial still fails."""
    started = time.monotonic()
    for retries, delay in enumerate((*_PUBLICATION_RETRY_DELAYS, None)):
        try:
            return operation()
        except OSError as exc:
            if getattr(exc, "winerror", None) not in _WINDOWS_FILE_LOCK_ERRORS or delay is None:
                diagnostic = (
                    f"File operation failed: operation={operation_name}, destination={destination}, "
                    f"winerror={getattr(exc, 'winerror', None)}, errno={exc.errno}, "
                    f"retries={retries}, elapsed_retry_seconds={time.monotonic() - started:.3f}"
                )
                _LOG.error("%s: %s", diagnostic, exc)
                exc.add_note(diagnostic)
                raise
            time.sleep(delay)


def atomic_replace(source, destination, *, cancel_check=None):
    """Replace atomically, allowing up to 3.1 seconds for Windows locks to clear.

    Replacing an open destination without delete sharing can report access
    denied (5), as well as sharing/lock violations (32/33). Retrying the rename
    preserves the old destination; it never falls back to deleting or copying.
    """
    def replace():
        check_cancelled(cancel_check)
        os.replace(source, destination)

    _retry_windows_file_lock(replace, operation_name="replace", destination=destination)


@contextmanager
def staged_file(destination, *, cancel_check=None, publish=True):
    """Publish a nonempty sibling file only after the writer succeeds."""
    target = Path(destination)
    suffix = next(
        (extension for extension in (".nii.gz", ".ome.tiff", ".ome.tif")
         if target.name.lower().endswith(extension)),
        target.suffix or ".tmp",
    )
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
        atomic_replace(stage, target, cancel_check=cancel_check)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            _retry_windows_file_lock(
                lambda: stage.unlink(missing_ok=True), operation_name="unlink", destination=stage,
            )
        except OSError:
            if failure is None:
                raise


def _path_too_long(error, path):
    if error.errno == errno.ENAMETOOLONG or getattr(error, "winerror", None) == 206:
        return True
    # CRT open and MoveFileEx can report invalid names for overlong NTFS
    # components. Classify those errors only when the component proves it.
    return (
        os.name == "nt" and error.errno == errno.EINVAL
        and len(path.name.encode("utf-16-le", "surrogatepass")) // 2 > 255
    )


def _publication_path_error(path):
    return OSError(
        errno.ENAMETOOLONG,
        "The filename or folder is too long for export provenance. "
        "Choose a shorter filename or a folder closer to the drive root",
        str(path),
    )


def publish_file_pair(stage, destination, companion_stage, companion_destination, *, cancel_check=None, replace_func=None,
                      optional_history=False, recovery_path=None, on_commit=None,
                      fail_if_exists=False):
    """Publish under an import barrier; restore prior files on ordinary failure.

    Two renames are not atomic. The barrier survives an interrupted process or
    failed rollback, so consumers cannot accept a partial pair. Recovery copies
    are retained only when restoration itself fails. ``on_commit`` reports the
    guarded pair's commit before any backup cleanup can raise. Reformat uses
    ``fail_if_exists`` to reject an occupied destination after staging.
    """
    target, companion = Path(destination), Path(companion_destination)
    replace = replace_func or atomic_replace
    if optional_history:
        # Retire old metadata before replacing independently interpretable data.
        # Failure to make it harmless aborts before touching the primary file.
        previous = None
        check_cancelled(cancel_check)
        try:
            companion_exists = companion.exists()
        except OSError as exc:
            if not _path_too_long(exc, companion):
                raise
            companion_exists = False
        if companion_exists:
            fd, name = tempfile.mkstemp(prefix=".madi3d-history-", dir=companion.parent)
            os.close(fd)
            previous = Path(name)
            try:
                atomic_replace(companion, previous, cancel_check=cancel_check)
            except BaseException:
                previous.unlink(missing_ok=True)
                raise
        try:
            try:
                check_cancelled(cancel_check)
                if replace_func is None:
                    atomic_replace(stage, target, cancel_check=cancel_check)
                else:
                    replace(stage, target)
            except BaseException:
                if previous is not None:
                    atomic_replace(previous, companion)
                raise
            # Commit point: cancellation must not report the primary as unwritten.
            if companion_stage is not None:
                try:
                    replace(companion_stage, companion)
                except OSError as exc:
                    if _path_too_long(exc, companion):
                        exc = _publication_path_error(companion)
                    return f"Data saved; provenance companion failed: {exc}"
            return None
        finally:
            if previous is not None:
                previous.unlink(missing_ok=True)
    barrier = Path(str(companion) + PUBLICATION_MARKER_SUFFIX)
    backups = {}
    published = []
    restored = False
    try:
        marker = barrier.open("x", encoding="utf-8")
    except OSError as exc:
        if _path_too_long(exc, barrier):
            raise _publication_path_error(barrier) from exc
        raise
    with marker as stream:
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
        if fail_if_exists and (
            target.exists() or target.is_symlink()
            or companion.exists() or companion.is_symlink()
        ):
            raise FileExistsError(f"Output already exists: {target}")
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
        if replace_func is None:
            atomic_replace(stage, target, cancel_check=cancel_check)
        else:
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
                        atomic_replace(backups[path], path)
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
