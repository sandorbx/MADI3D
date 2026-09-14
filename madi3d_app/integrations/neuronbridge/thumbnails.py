"""Optional preview evidence and bounded, disposable thumbnail storage.

The official client uses image.files.CDMThumbnail for CDM and
match.files.CDMBestThumbnail for PPPM. Only those supplied file references (or
explicit small raster previews) are consumed; no filenames become identities.
"""
from dataclasses import dataclass
from functools import cached_property
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
import time
from urllib.parse import urlsplit

from PIL import Image
import requests

from madi3d_storage import cache_dir
from .public_api import file_url
from .records import SourceIdentity

MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 16 * 1024 * 1024
MAX_CACHE_BYTES = 128 * 1024 * 1024
MAX_CACHE_FILES = 2048
THUMBNAIL_SIZE = (320, 240)
MAX_PREVIEW_BYTES = THUMBNAIL_SIZE[0] * THUMBNAIL_SIZE[1] * 4


@dataclass(frozen=True)
class ThumbnailReference:
    url: str
    asset_type: str

    @property
    def key(self):
        return self.url


@dataclass(frozen=True)
class ThumbnailLookup:
    source: SourceIdentity
    data_version: str | None

    @cached_property
    def key(self):
        evidence = json.dumps((self.source.to_dict(), self.data_version), sort_keys=True, separators=(",", ":"))
        return "metadata:" + hashlib.sha256(evidence.encode("utf-8")).hexdigest()


def _reference(image, keys, config):
    for key in keys:
        url = file_url(image, key, config)
        if not url:
            continue
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            continue
        # Never fall back to searchable TIFFs, volume stacks or 3-D geometry.
        if Path(parsed.path).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            continue
        return ThumbnailReference(url, key)
    return None


def thumbnail_reference(occurrence, session):
    """Project evidence only. Missing/invalid previews never invalidate a hit."""
    if session is None:
        return None
    try:
        parameters = {p.name: p.value for p in session.parameters}
        public = parameters.get("public_api_evidence")
        if public:
            config = public["lookup_snapshot"]["config"]["payload"]
            row = public["result_response"]["payload"]["results"][occurrence.row_index]
            image = row.get("image", {})
            if str(image.get("id")) != occurrence.target.image.image_id:
                return None
            if parameters.get("requested_method") == "PPPM":
                return _reference(row, ("CDMBestThumbnail", "CDMBest"), config)
            return _reference(image, ("CDMThumbnail", "CDM"), config)
        local = parameters.get("local_search")
        if local:
            field = next((f for f in occurrence.fields if f.header == "Identity metadata evidence"), None)
            evidence = json.loads(field.value) if field and isinstance(field.value, str) else {}
            image = evidence.get("resolved_image", {})
            if str(image.get("id")) != occurrence.target.image.image_id:
                return None
            # Local snapshots retain their catalog/config evidence. A fully
            # qualified file URL also works when an old snapshot lacks config.
            catalog = local.get("catalog_evidence", {})
            config = catalog.get("config", {}).get("payload", {})
            return _reference(image, ("CDMThumbnail", "CDM"), config)
        for asset in occurrence.target.assets:
            if asset.asset_type in {"CDMThumbnail", "CDMBestThumbnail"} and asset.url:
                return _reference({"files": {asset.asset_type: asset.url}}, (asset.asset_type,), {})
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        pass
    return None


def thumbnail_request(occurrence, session):
    """Prefer retained preview URLs; CSV may request exact metadata resolution."""
    reference = thumbnail_reference(occurrence, session)
    if reference is not None or session is None or session.source_kind != "imported_csv":
        return reference
    source = occurrence.target
    if source.kind not in {"em", "lm"} or not source.library or not source.image.alignment_space:
        return None
    name = source.biological.neuron_id if source.kind == "em" else source.biological.line_name
    if not name or any(d.code in {"ambiguous_identity_field", "ambiguous_target_identity", "missing_target_identity"}
                       for d in occurrence.diagnostics):
        return None
    return ThumbnailLookup(source, session.neuronbridge_data_version)


def resolve_thumbnail(request, cancel):
    """Resolve only a unique compatible image, without selecting a 3-D asset."""
    from .assets import AssetMetadataClient, image_mismatches
    _checkpoint(cancel)
    source = request.source
    name = source.biological.neuron_id if source.kind == "em" else source.biological.line_name
    client = AssetMetadataClient(request.data_version, cancel=cancel)
    try:
        images = (client.get_em_images(name.rsplit(":", 1)[-1]) if source.kind == "em"
                  else client.get_lm_images(name))
        _checkpoint(cancel)
        matches = [image for image in images if not image_mismatches(source, image)]
        if len(matches) != 1:
            raise ValueError(f"{len(matches)} published images match this CSV identity. "
                             "Use Public lookup to select an exact image; no preview was guessed.")
        reference = _reference(matches[0], ("CDMThumbnail", "CDM"), client.config)
        if reference is None:
            raise ValueError("The matching published image has no supported preview.")
        return reference
    finally:
        client.close()


def thumbnail_error(error):
    if isinstance(error, requests.Timeout):
        return "Preview request timed out. Check your connection and retry."
    if isinstance(error, requests.HTTPError):
        status = error.response.status_code if error.response is not None else "error"
        return f"NeuronBridge could not supply this preview (HTTP {status})."
    if isinstance(error, requests.RequestException):
        return "Cannot connect to the preview service. Check your connection and retry."
    return str(error)[:500]


def _checkpoint(cancel):
    if cancel.is_set():
        raise InterruptedError("Thumbnail loading cancelled.")


def fetch_thumbnail(url, cancel):
    """One small raster request; no redirects, retries, or asset resolution."""
    _checkpoint(cancel)
    began = time.monotonic()
    with requests.Session() as session:
        with session.get(url, stream=True, timeout=(3, 8), allow_redirects=False) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("Thumbnail is unavailable at its supplied URL.")
            if int(response.headers.get("Content-Length", "0")) > MAX_DOWNLOAD_BYTES:
                raise ValueError("Thumbnail exceeds the preview download limit.")
            content = bytearray()
            for block in response.iter_content(64 * 1024):
                _checkpoint(cancel)
                if time.monotonic() - began > 15 or len(content) + len(block) > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Thumbnail exceeds the preview transfer limit.")
                content.extend(block)
    _checkpoint(cancel)
    return bytes(content)


class ThumbnailCache:
    """128 MiB / 2048 files, independent of authoritative project metadata."""
    def __init__(self, root=None, fetch=fetch_thumbnail, resolve=resolve_thumbnail):
        self.root = Path(root) if root is not None else cache_dir() / "NeuronBridge" / "thumbnails"
        self.fetch = fetch
        self.resolve = resolve
        self.lock = Lock()

    def read(self, reference, cancel, *, network=False):
        _checkpoint(cancel)
        # Resolution/size belongs to this disposable cache, never the project.
        key = hashlib.sha256((f"{THUMBNAIL_SIZE[0]}x{THUMBNAIL_SIZE[1]}:" + reference.key).encode("utf-8")).hexdigest()
        path = self.root / (key + ".png")
        try:
            if path.stat().st_size <= MAX_PREVIEW_BYTES:
                content = path.read_bytes()
                with Image.open(io.BytesIO(content)) as image:
                    if image.size[0] <= THUMBNAIL_SIZE[0] and image.size[1] <= THUMBNAIL_SIZE[1]:
                        image.load()
                        _checkpoint(cancel)
                        return content
        except (OSError, ValueError):
            pass
        if not network:
            return None
        resolved = self.resolve(reference, cancel) if isinstance(reference, ThumbnailLookup) else reference
        _checkpoint(cancel)
        content = self.fetch(resolved.url, cancel)
        if len(content) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Thumbnail exceeds the preview download limit.")
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Thumbnail exceeds the preview pixel limit.")
            image.thumbnail(THUMBNAIL_SIZE)
            output = io.BytesIO()
            image.convert("RGB").save(output, format="PNG")
        content = output.getvalue()
        _checkpoint(cancel)
        # A read-only/full cache does not prevent displaying a fetched preview.
        try:
            with self.lock:
                self.root.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=self.root, suffix=".tmp", delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(content)
                    _checkpoint(cancel)
                    os.replace(temporary, path)
                    self._prune()
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return content

    def _prune(self):
        files = sorted(((p.stat().st_mtime, p.stat().st_size, p) for p in self.root.glob("*.png")), reverse=True)
        size = 0
        for position, (_, length, path) in enumerate(files):
            size += length
            if position >= MAX_CACHE_FILES or size > MAX_CACHE_BYTES:
                path.unlink(missing_ok=True)
