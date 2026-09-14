"""Qt-free cancellation, resource checks and atomic file publication."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile

CHUNK_BYTES = 1024 * 1024
MEMORY_BUDGET_FRACTION = 0.90


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


@contextmanager
def staged_file(destination, *, cancel_check=None):
    """Publish a nonempty sibling file only after the writer succeeds."""
    target = Path(destination)
    suffix = "".join(target.suffixes) or ".tmp"
    fd, name = tempfile.mkstemp(prefix=".madi3d-", suffix=suffix, dir=target.parent)
    os.close(fd)
    stage = Path(name)
    try:
        check_cancelled(cancel_check)
        yield str(stage)
        check_cancelled(cancel_check)
        if not stage.is_file() or stage.stat().st_size <= 0:
            raise OSError("The writer did not produce a complete output file.")
        with stage.open("rb+") as stream:
            os.fsync(stream.fileno())
        check_cancelled(cancel_check)
        os.replace(stage, target)
    finally:
        stage.unlink(missing_ok=True)
