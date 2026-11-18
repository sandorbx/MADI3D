"""Safe local caching for NeuronBridge downloads and derived candidate data."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

import numpy as np
import requests
from PIL import Image, PngImagePlugin
from tqdm import tqdm

from madi3d_storage import cache_dir, downloaded_assets_dir
from madi3d_app.io_utils import file_revision, filename_unit_budget


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


class AssetChecksumError(IOError):
    """A complete remote response contradicts retained checksum evidence."""
    status = "unresolved"

    def __init__(self, url, expected, observed):
        super().__init__("Downloaded asset checksum does not match the supplied evidence.")
        self.source_discrepancy = {"kind": "checksum-mismatch", "locator": url,
                                   "expected_sha256": expected, "observed_sha256": observed}


_WINDOWS_INVALID = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"{device}{digit}" for device in ("COM", "LPT") for digit in "123456789¹²³"}
)


def neuronbridge_cache_root(
    *, configured_root=None, platform_name=None, environ=None, home=None
) -> Path:
    """Return the durable asset root, retaining explicit historical overrides."""
    env = os.environ if environ is None else environ
    override = env.get("NB_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if configured_root:
        return Path(configured_root).expanduser()
    return downloaded_assets_dir(
        platform_name=platform_name,
        environ=env,
        home=home,
    ) / "NeuronBridge"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()[:10]


def portable_cache_filename(raw_stem, suffix, *, preserve_spaces=False,
                            marker="", reserve="", directory=None) -> str:
    """Return a stable filename safe on Windows, macOS, and Linux.

    Spaces become underscores unless a scene label must be preserved. Names
    needing additional sanitization receive a short hash so
    distinct source identifiers cannot collapse onto one cache entry.
    The coordinate marker, format extension and reserved companion tail all
    count toward the physical filename budget.
    """
    raw = str(raw_stem or "download")
    suffix = str(suffix or "")
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix
    if not re.fullmatch(r"(?:\.[A-Za-z0-9]{1,12})?", suffix):
        raise ValueError(f"Unsafe cache filename suffix: {suffix!r}")

    historical = raw if preserve_spaces else raw.replace(" ", "_")
    cleaned = "".join(
        "_" if ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF or char in _WINDOWS_INVALID else char
        for char in historical
    ).rstrip(" .")
    if not cleaned:
        cleaned = "download"

    changed = cleaned != historical
    if cleaned.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        cleaned = "_" + cleaned
        changed = True

    tail = marker + suffix
    reserved_tail = tail + reserve
    byte_budget = min(MAX_CACHE_STEM_LENGTH, 255 - len(reserved_tail.encode("utf-8")))
    unit_budget = filename_unit_budget(directory) - len(reserved_tail.encode("utf-16-le")) // 2
    encoded = cleaned.encode("utf-8")
    if not changed and len(encoded) <= byte_budget and len(cleaned.encode("utf-16-le")) // 2 <= unit_budget:
        return cleaned + tail

    digest = "-" + _short_hash(raw)
    if min(byte_budget, unit_budget) < len(digest):
        raise OSError("The cache folder is too deep for the geometry and its provenance. Choose a shorter cache folder.")
    prefix = encoded[:byte_budget - len(digest)].decode("utf-8", errors="ignore")
    prefix = prefix.encode("utf-16-le")[:2 * (unit_budget - len(digest))].decode(
        "utf-16-le", errors="ignore",
    )
    return prefix.rstrip(" .") + digest + tail


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
                        raise InterruptedError("Asset download cancelled.")
                    digest.update(chunk)
            return digest.hexdigest() == expected_sha256
        return True
    except InterruptedError:
        raise
    except OSError:
        return False


def _asset_cache_suffix(asset_type):
    return {"AlignedBodyOBJ": ".obj", "AlignedBodySWC": ".swc",
            "VisuallyLosslessStack": ".h5j"}[asset_type]


def asset_cache_filename(*, library, release, data_version, alignment, asset_type, url=None,
                         asset_id=None, checksum=None):
    """Key a versioned asset by stable ID when supplied, otherwise by locator."""
    locator = ["asset_id", asset_id] if asset_id else ["url", url]
    identity = [library, release, data_version, alignment, asset_type, locator, checksum]
    digest = hashlib.sha256(json.dumps(
        identity, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return "nb-" + digest + _asset_cache_suffix(asset_type)


def legacy_asset_cache_filename(*, library, release, data_version, alignment, asset_type,
                                url, asset_id=None, checksum=None):
    """Return the previous URL-based key so existing downloads remain reusable."""
    identity = [library, release, data_version, alignment, asset_type, url, asset_id, checksum]
    digest = hashlib.sha256(json.dumps(
        identity, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return "nb-" + digest + _asset_cache_suffix(asset_type)


def compact_asset_cache_path(root, filename):
    """Physical spelling of a versioned asset key for deep cache folders."""
    stem, suffix = os.path.splitext(filename)
    return Path(root) / (stem[:len("nb-") + 24] + suffix)


def stream_download(
    url,
    destination,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    request_get=None,
    provider_only=False,
    progress_factory=tqdm,
    expected_sha256=None,
    deadline_seconds=1800,
    resource_policy=DownloadResourcePolicy(),
    disk_usage=shutil.disk_usage,
    on_accepted=None,
) -> Path:
    """Bound staging on the destination filesystem; atomically publish on success.

    ``disk_usage(path)`` supplies the current free bytes via its ``free`` field.
    The initial free-space budget never grows during a transfer, and live probes
    before each write account for space consumed by other processes.
    ``on_accepted(sha256, revision)`` receives the transfer's byte evidence after
    publication; the returned path remains the normal download result.
    Set ``provider_only`` for URLs carried by NeuronBridge records or metadata.
    Other callers supply an explicitly selected or pinned catalog URL.
    """
    url = str(url)
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        raise OSError("Malformed asset URL. Select a valid download source.") from None
    if (parsed.scheme not in {"http", "https"} or not host
            or parsed.username is not None or "\\" in url
            or any(ord(char) <= 32 or ord(char) == 127 for char in url)):
        raise OSError("Invalid HTTP(S) asset URL. Select a valid download source.")
    if provider_only:
        buckets = ("janelia-flylight-color-depth", "janelia-flylight-imagery")
        provider = (
            host == "s3.amazonaws.com"
            and any(parsed.path.startswith(f"/{bucket}/") for bucket in buckets)
        ) or any(host in {f"{bucket}.s3.amazonaws.com", f"{bucket}.s3.us-east-1.amazonaws.com"}
                 for bucket in buckets)
        if parsed.scheme != "https" or port is not None or not provider:
            raise OSError(
                "This asset URL is outside the supported NeuronBridge download stores. "
                "Resolve the asset through NeuronBridge or select a local copy."
            )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if cancel_check is not None and cancel_check():
        raise InterruptedError("Asset download cancelled.")

    temp_path = None
    written = 0
    digest = hashlib.sha256()
    deadline = time.monotonic() + deadline_seconds
    def checkpoint():
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Asset download cancelled.")
        if time.monotonic() >= deadline:
            raise TimeoutError("Asset transfer exceeded its time limit.")
    session = None
    try:
        if request_get is None:
            if provider_only:
                session = requests.Session()
                session.trust_env = False
                request_get = session.get
            else:
                request_get = requests.get
        response = request_get(
            url,
            stream=True,
            allow_redirects=not provider_only,
            timeout=DOWNLOAD_TIMEOUT,
            headers={"Accept-Encoding": "identity"},
        )
        with response:
            if provider_only and 300 <= response.status_code < 400:
                raise OSError("NeuronBridge redirected the asset download. Resolve it again or select a local copy.")
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
                        f"Asset download {context}; per-asset limit is "
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
                        f"Asset download {context}; destination {destination.parent} "
                        f"has {free} free bytes, requires a {resource_policy.reserve_bytes}-byte "
                        f"reserve, and allows {available} more bytes "
                        f"(initial budget {initial_budget} bytes). "
                        "Free disk space or choose another download location."
                    )

            check_resources(0, 0)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                buffering=0,
                dir=destination.parent,
                prefix=".nb-download-",
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
                                "Asset download exceeded its declared length: "
                                f"received {written + len(chunk)} of {total} bytes."
                            )
                        if handle.write(chunk) != len(chunk):
                            raise IOError("Asset staging write was incomplete.")
                        digest.update(chunk)
                        written += len(chunk)
                        bar.update(len(chunk))
                handle.flush()
                os.fsync(handle.fileno())
                staged_stat = os.fstat(handle.fileno())

        if total and written != total:
            raise IOError(
                "Asset download was incomplete: "
                f"received {written} of {total} bytes."
            )

        if not written:
            raise IOError("The server returned an empty download.")
        if expected_sha256 and digest.hexdigest() != expected_sha256:
            raise AssetChecksumError(url, expected_sha256, digest.hexdigest())
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Asset download cancelled.")
        os.replace(temp_path, destination)
        temp_path = None
        revision = file_revision(destination)
        if revision[1:4] != (staged_stat.st_size, staged_stat.st_mtime_ns, staged_stat.st_ino):
            raise IOError("The downloaded source changed during publication.")
        if on_accepted is not None:
            on_accepted(digest.hexdigest(), revision)
        return destination
    finally:
        if session is not None:
            session.close()
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


# --------------------------------------------------------------------- gradients

GRADIENT_CACHE_DIRECTORY = "gradient-derivatives"
# The one route into the derived-gradient cache. A stored gradient is
# interchangeable with a freshly produced one only while this key derivation and
# the producer that filled it both agree, so both are versioned.
GRADIENT_CACHE_FORMAT = "nb-derived-gradient-v2"
# A gradient is a 16-bit distance map and compresses to a few kilobytes; these
# ceilings bound the disposable cache rather than describe an expected entry.
MAX_GRADIENT_ENTRY_BYTES = 8 * 1024 * 1024
MAX_GRADIENT_CACHE_BYTES = 256 * 1024 * 1024
MAX_GRADIENT_CACHE_FILES = 16_384


def gradient_cache_root() -> Path:
    """Return the derived-gradient cache root without creating it."""
    return cache_dir() / "NeuronBridge" / GRADIENT_CACHE_DIRECTORY


def derived_gradient_key(*, library, candidate, canvas_yx, producer_revision,
                         implementation_revision, gradient_format) -> str:
    """Key one generated candidate gradient by archive, candidate and producer.

    ``library`` is the installed distribution's own identity mapping, which
    carries the installed archive SHA-256 whenever the installation record has
    one; nothing invents a digest that the installation never observed.
    ``candidate`` is the candidate's identity within that distribution and
    ``producer_revision``, ``implementation_revision`` and ``gradient_format``
    independently identify the upstream reference, local implementation and pixel format. Any change to those facts produces a different key,
    so a cache hit can only be a gradient whose determining inputs are the same.
    """
    identity = {"cache_format": GRADIENT_CACHE_FORMAT,
                "library": dict(library),
                "candidate": dict(candidate),
                "canvas_yx": [int(value) for value in canvas_yx],
                "producer_revision": producer_revision,
                "implementation_revision": implementation_revision,
                "gradient_format": gradient_format}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


class GradientCache:
    """Disposable uint16 candidate-gradient cache, independent of project state.

    Entries are the producer's own PNG encoding, published atomically and pruned
    to a byte/file ceiling. A cache miss, an unwritable cache and an unreadable
    entry all mean "generate it again": they never invalidate the candidate and
    nothing outside this directory learns that the cache existed.
    """

    def __init__(self, root=None, *, max_bytes=MAX_GRADIENT_CACHE_BYTES,
                 max_files=MAX_GRADIENT_CACHE_FILES):
        self.root = Path(root).expanduser() if root is not None else gradient_cache_root()
        self.max_bytes = int(max_bytes)
        self.max_files = int(max_files)
        self.lock = threading.Lock()

    def path_for(self, key):
        # Entries are addressed only by a derived key, never by a caller-built path,
        # so a cache can never reach outside its own directory.
        if not re.fullmatch(r"[0-9a-f]{64}", str(key)):
            raise ValueError("A derived gradient key requires 64 lowercase hexadecimal digits.")
        return self.root / (key + ".png")

    def load(self, key, *, canvas_yx):
        """Return cached gradient pixels, or ``None`` when nothing is usable."""
        from .gradient_preparation import gradient_pixel_sha256

        path = self.path_for(key)
        try:
            if path.stat().st_size > MAX_GRADIENT_ENTRY_BYTES:
                return None
            with Image.open(path) as image:
                if image.format != "PNG" or image.n_frames != 1:
                    return None
                pixels = np.asarray(image)
                if pixels.dtype != np.uint16 or pixels.shape != tuple(canvas_yx):
                    return None
                if (image.text.get("MADI3D-Derived-Key") != key
                        or image.text.get("MADI3D-Pixel-SHA256") != gradient_pixel_sha256(pixels)):
                    return None
                detached = pixels.copy()
                detached.setflags(write=False)
                return detached
        except (OSError, ValueError, SyntaxError):
            return None

    def publish(self, key, gradient):
        """Store one generated gradient; return the path, or ``None`` if unwritable."""
        path = self.path_for(key)
        try:
            content = _gradient_png(gradient, key)
            with self.lock:
                self.root.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=self.root, suffix=".tmp", delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(content)
                    os.replace(temporary, path)
                    temporary = None
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                try:
                    self._prune()
                except OSError:
                    pass  # a failed prune never costs a published gradient
        except OSError:
            return None
        return path

    def _prune(self):
        # Publication holds the lock across staging and replacement, so any
        # remaining ``.tmp`` entry is a leftover from an interrupted write.
        for leftover in self.root.glob("*.tmp"):
            leftover.unlink(missing_ok=True)
        files = sorted(((path.stat().st_mtime_ns, path.stat().st_size, path)
                        for path in self.root.glob("*.png")), reverse=True)
        size = 0
        for position, (_, length, path) in enumerate(files):
            size += length
            if position >= self.max_files or size > self.max_bytes:
                path.unlink(missing_ok=True)


def _gradient_png(gradient, key):
    """Encode generated gradient pixels exactly, in the archive's own encoding."""
    if (not isinstance(gradient, np.ndarray) or gradient.dtype != np.uint16
            or gradient.ndim != 2):
        raise ValueError("A derived gradient cache entry requires 2-D uint16 pixels.")
    from .gradient_preparation import gradient_pixel_sha256

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("MADI3D-Derived-Key", key)
    metadata.add_text("MADI3D-Pixel-SHA256", gradient_pixel_sha256(gradient))
    image = Image.fromarray(gradient)
    try:
        if image.mode != "I;16":
            raise ValueError("A derived gradient cache entry requires 16-bit unsigned pixels.")
        stream = io.BytesIO()
        image.save(stream, format="PNG", pnginfo=metadata)
    finally:
        image.close()
    content = stream.getvalue()
    if not content:
        raise ValueError("Derived gradient encoding produced no PNG bytes.")
    return content


__all__ = [
    "DOWNLOAD_CHUNK_SIZE",
    "DOWNLOAD_TIMEOUT",
    "GRADIENT_CACHE_DIRECTORY",
    "GRADIENT_CACHE_FORMAT",
    "MAX_CACHE_STEM_LENGTH",
    "MAX_ASSET_BYTES",
    "MAX_GRADIENT_CACHE_BYTES",
    "MAX_GRADIENT_CACHE_FILES",
    "MAX_GRADIENT_ENTRY_BYTES",
    "MIN_FREE_DISK_BYTES",
    "DownloadResourcePolicy",
    "DownloadResourceError",
    "GradientCache",
    "asset_cache_filename",
    "cached_file_ready",
    "derived_gradient_key",
    "gradient_cache_root",
    "neuronbridge_cache_root",
    "portable_cache_filename",
    "stream_download",
]
