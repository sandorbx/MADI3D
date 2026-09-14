"""Safe local caching for files downloaded from NeuronBridge."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests
from tqdm import tqdm

from madi3d_storage import cache_dir


DOWNLOAD_CHUNK_SIZE = 1 << 20
DOWNLOAD_TIMEOUT = (5.0, 10.0)
MAX_CACHE_STEM_LENGTH = 180

# Policy rationale and measured EM/LM sample: docs/madi3d/files/neuronbridge_cache.md.
MAX_ASSET_BYTES = 2 * 1024**3
MIN_FREE_DISK_BYTES = 1024**3


@dataclass(frozen=True)
class DownloadResourcePolicy:
    """Bound each staged asset and leave space for other application writes."""

    max_asset_bytes: int = MAX_ASSET_BYTES
    reserve_bytes: int = MIN_FREE_DISK_BYTES

    def __post_init__(self):
        for name in ("max_asset_bytes", "reserve_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer byte count.")


class DownloadResourceError(OSError):
    """A transfer would exceed its asset ceiling or destination disk budget."""


_WINDOWS_INVALID = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def neuronbridge_cache_root(*, platform_name=None, environ=None, home=None) -> Path:
    """Return the per-user NeuronBridge cache root without creating it."""
    env = os.environ if environ is None else environ
    override = env.get("NB_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return cache_dir(
        platform_name=platform_name,
        environ=env,
        home=home,
    ) / "NeuronBridge"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def portable_cache_filename(raw_stem, suffix) -> str:
    """Return a stable filename safe on Windows, macOS, and Linux.

    Ordinary names retain MADI3D's historical spelling: spaces become
    underscores. Names needing additional sanitization receive a short hash so
    distinct source identifiers cannot collapse onto one cache entry.
    """
    raw = str(raw_stem or "download")
    suffix = str(suffix or "")
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix
    if not re.fullmatch(r"(?:\.[A-Za-z0-9]{1,12})?", suffix):
        raise ValueError(f"Unsafe cache filename suffix: {suffix!r}")

    historical = raw.replace(" ", "_")
    cleaned = "".join(
        "_" if ord(char) < 32 or char in _WINDOWS_INVALID else char
        for char in historical
    ).rstrip(" .")
    if not cleaned:
        cleaned = "download"

    changed = cleaned != historical
    if cleaned.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        cleaned = "_" + cleaned
        changed = True

    if len(cleaned) > MAX_CACHE_STEM_LENGTH:
        cleaned = cleaned[:MAX_CACHE_STEM_LENGTH].rstrip(" .")
        changed = True

    if changed:
        digest = _short_hash(raw)
        budget = max(1, MAX_CACHE_STEM_LENGTH - len(digest) - 1)
        cleaned = (cleaned[:budget].rstrip(" .") or "download") + "-" + digest

    return cleaned + suffix


def cached_file_ready(path, *, expected_sha256=None, cancel_check=None) -> bool:
    """Require a nonempty file and verify a checksum when the source supplies one."""
    candidate = Path(path)
    try:
        if not candidate.is_file() or candidate.stat().st_size == 0:
            return False
        if expected_sha256:
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_SIZE), b""):
                    if cancel_check and cancel_check():
                        raise InterruptedError("NeuronBridge download cancelled.")
                    digest.update(chunk)
            return digest.hexdigest() == expected_sha256
        return True
    except InterruptedError:
        raise
    except OSError:
        return False


def asset_cache_filename(*, library, release, data_version, alignment, asset_type, url, asset_id=None, checksum=None):
    """Key the complete versioned asset, independently of result/channel identity."""
    suffix = {"AlignedBodyOBJ": ".obj", "AlignedBodySWC": ".swc",
              "VisuallyLosslessStack": ".h5j"}[asset_type]
    identity = [library, release, data_version, alignment, asset_type, url, asset_id, checksum]
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return "nb-" + digest + suffix


def stream_download(
    url,
    destination,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    request_get=requests.get,
    progress_factory=tqdm,
    expected_sha256=None,
    deadline_seconds=1800,
    resource_policy=DownloadResourcePolicy(),
    disk_usage=shutil.disk_usage,
) -> Path:
    """Bound staging on the destination filesystem; atomically publish on success.

    ``disk_usage(path)`` supplies the current free bytes via its ``free`` field.
    The initial free-space budget never grows during a transfer, and live probes
    before each write account for space consumed by other processes.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if cancel_check is not None and cancel_check():
        raise InterruptedError("NeuronBridge download cancelled.")

    temp_path = None
    written = 0
    digest = hashlib.sha256()
    deadline = time.monotonic() + deadline_seconds
    def checkpoint():
        if cancel_check is not None and cancel_check():
            raise InterruptedError("NeuronBridge download cancelled.")
        if time.monotonic() >= deadline:
            raise TimeoutError("NeuronBridge asset transfer exceeded its time limit.")
    try:
        response = request_get(
            str(url),
            stream=True,
            timeout=DOWNLOAD_TIMEOUT,
            headers={"Accept-Encoding": "identity"},
        )
        with response:
            response.raise_for_status()
            checkpoint()
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise IOError("Unexpected compressed asset transfer; bounded identity transfer required.")
            raw_total = response.headers.get("Content-Length", 0)
            try:
                total = max(0, int(raw_total or 0))
            except (TypeError, ValueError):
                total = 0

            initial_budget = None

            def check_resources(received, pending):
                nonlocal initial_budget
                required = max(total, received)
                context = f"requested {total or 'unknown'}, received {received} bytes"
                if required > resource_policy.max_asset_bytes:
                    raise DownloadResourceError(
                        f"NeuronBridge download {context}; per-asset limit is "
                        f"{resource_policy.max_asset_bytes} bytes."
                    )
                free = disk_usage(destination.parent).free
                available = max(0, free - resource_policy.reserve_bytes)
                if initial_budget is None:
                    initial_budget = available
                # Previously staged bytes already occupy disk. Reserve room for
                # this chunk and, when declared, the rest of the response.
                remaining = max(pending, total - written)
                if (free < resource_policy.reserve_bytes
                        or required > initial_budget or remaining > available):
                    raise DownloadResourceError(
                        f"NeuronBridge download {context}; destination {destination.parent} "
                        f"has {free} free bytes, requires a {resource_policy.reserve_bytes}-byte "
                        f"reserve, and allows {available} more bytes "
                        f"(initial budget {initial_budget} bytes). "
                        "Free disk space or choose another NeuronBridge cache location."
                    )

            check_resources(0, 0)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                buffering=0,
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                show_bar = sys.stderr is not None and hasattr(sys.stderr, "write")
                with progress_factory(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=f"Download {destination.name}",
                    leave=False,
                    disable=not show_bar,
                    file=sys.stderr if show_bar else None,
                ) as bar:
                    while True:
                        checkpoint()
                        chunk = response.raw.read1(DOWNLOAD_CHUNK_SIZE, decode_content=False)
                        checkpoint()
                        if not chunk:
                            break
                        check_resources(written + len(chunk), len(chunk))
                        if total and written + len(chunk) > total:
                            raise IOError(
                                "NeuronBridge download exceeded its declared length: "
                                f"received {written + len(chunk)} of {total} bytes."
                            )
                        if handle.write(chunk) != len(chunk):
                            raise IOError("NeuronBridge cache staging write was incomplete.")
                        digest.update(chunk)
                        written += len(chunk)
                        bar.update(len(chunk))
                handle.flush()
                os.fsync(handle.fileno())

        if total and written != total:
            raise IOError(
                "NeuronBridge download was incomplete: "
                f"received {written} of {total} bytes."
            )

        if not written:
            raise IOError("NeuronBridge returned an empty download.")
        if expected_sha256 and digest.hexdigest() != expected_sha256:
            raise IOError("NeuronBridge asset checksum does not match the supplied evidence.")
        if cancel_check is not None and cancel_check():
            raise InterruptedError("NeuronBridge download cancelled.")
        os.replace(temp_path, destination)
        temp_path = None
        return destination
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = [
    "DOWNLOAD_CHUNK_SIZE",
    "DOWNLOAD_TIMEOUT",
    "MAX_CACHE_STEM_LENGTH",
    "MAX_ASSET_BYTES",
    "MIN_FREE_DISK_BYTES",
    "DownloadResourcePolicy",
    "DownloadResourceError",
    "asset_cache_filename",
    "cached_file_ready",
    "neuronbridge_cache_root",
    "portable_cache_filename",
    "stream_download",
]
