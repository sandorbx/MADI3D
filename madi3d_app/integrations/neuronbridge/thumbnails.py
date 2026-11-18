"""Optional preview evidence and bounded, disposable thumbnail storage.

The official client uses image.files.CDMThumbnail for CDM and
match.files.CDMBestThumbnail for PPPM. Only those supplied file references (or
explicit small raster previews) are consumed; no filenames become identities.

Results of a Local Search preview from the installed distribution itself:
``LibraryPreviewReference`` carries the library and candidate identity, and
``render_library_preview`` reconstructs the candidate pixels through the
read-only installed-library readers. No NeuronBridge request participates in a
local preview; public lookup routes are untouched.
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

import numpy as np
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
class LibraryPreviewReference:
    """Preview identity of one installed-distribution candidate.

    The installed library (ID plus the archive SHA-256 observed during its
    installation) and the candidate (published member key plus ordinal) are the
    whole identity; the key never embeds a machine-local path, so a relocated
    library keeps its previews and a replaced package never reuses them.
    """
    library_id: str
    archive_sha256: str
    member_key: str
    candidate_ordinal: int

    def __post_init__(self):
        if not isinstance(self.library_id, str) or not self.library_id.strip():
            raise ValueError("A local preview requires a library ID.")
        if (not isinstance(self.archive_sha256, str) or len(self.archive_sha256) != 64 or
                any(c not in "0123456789abcdef" for c in self.archive_sha256)):
            raise ValueError("A local preview requires the searched archive SHA-256.")
        if not isinstance(self.member_key, str) or not self.member_key.strip():
            raise ValueError("A local preview requires a candidate member key.")
        if type(self.candidate_ordinal) is not int or self.candidate_ordinal < 0:
            raise ValueError("A local preview requires a nonnegative candidate ordinal.")

    @cached_property
    def key(self):
        evidence = json.dumps(("madi3d-library-preview-v2", self.library_id, self.archive_sha256,
                               self.member_key, self.candidate_ordinal), sort_keys=True, separators=(",", ":"))
        return "library:" + hashlib.sha256(evidence.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ThumbnailLookup:
    source: SourceIdentity
    data_version: str | None
    library_release_from_catalog: bool = False

    @cached_property
    def key(self):
        identity = (self.source.to_dict(), self.data_version)
        if self.library_release_from_catalog:
            identity += ("library_release_from_catalog",)
        evidence = json.dumps(identity, sort_keys=True, separators=(",", ":"))
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
        retained = (session.evidence_for(occurrence) if occurrence.evidence_index is not None
                    else session.external_evidence.get(occurrence.evidence_ref, {}))
        preview = _reference({"files": retained.get("preview_reference", {})},
                             ("CDMBestThumbnail", "CDMBest", "CDMThumbnail", "CDM"), {})
        parameters = {p.name: p.value for p in session.parameters}
        public = session.external_evidence if session.source_kind == "precomputed" else None
        if public:
            config = public.get("lookup_snapshot", {}).get("config", {}).get("payload", {})
            row = session.evidence_for(occurrence)
            image = row.get("image", {})
            if str(image.get("id")) != occurrence.target.image.image_id:
                return None
            if parameters.get("requested_method") == "PPPM":
                return preview or _reference(row, ("CDMBestThumbnail", "CDMBest"), config)
            return preview or _reference(image, ("CDMThumbnail", "CDM"), config)
        local = session.context if session.context.get("backend") == "madi3d-local" else None
        if local:
            evidence = session.evidence_for(occurrence).get("identity", {})
            image = (evidence["observed_response"]["results"][evidence["resolved_image_index"]]
                     if "resolved_image_index" in evidence else evidence.get("resolved_image", {}))
            if str(image.get("id")) != occurrence.target.image.image_id:
                return None
            # Local snapshots retain their catalog/config evidence. A fully
            # qualified file URL also works when an old snapshot lacks config.
            catalog = session.external_evidence.get("catalog", {})
            config = catalog.get("config", {}).get("payload", {})
            return preview or _reference(image, ("CDMThumbnail", "CDM"), config)
        for asset in occurrence.target.assets:
            if asset.asset_type in {"CDMThumbnail", "CDMBestThumbnail"} and asset.url:
                return _reference({"files": {asset.asset_type: asset.url}}, (asset.asset_type,), {})
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        pass
    return None


def library_preview_reference(occurrence, session):
    """Local-search results preview from the installed library, never from HTTP.

    Schema-3 local evidence records the searched distribution's identity, and
    every occurrence retains its candidate identity. Older local snapshots
    retain their original metadata routes. Schema-3 results with incomplete
    identity have no preview; a missing library location is discovered at
    render time and never falls through to a public URL.
    """
    if session is None or session.source_kind != "custom_query":
        return None
    context = session.context
    if (context.get("backend") != "madi3d-local" or context.get("schema_version") != 3
            or not context.get("library_id")):
        return None
    try:
        return LibraryPreviewReference(context["library_id"], context.get("archive_sha256"),
                                       occurrence.member_id, occurrence.candidate_ordinal)
    except ValueError:
        return None


def thumbnail_request(occurrence, session):
    """Use installed pixels exclusively for schema 3; preserve older preview routes."""
    if (session is not None and session.source_kind == "custom_query" and
            session.context.get("backend") == "madi3d-local" and session.context.get("schema_version") == 3):
        return library_preview_reference(occurrence, session)
    reference = thumbnail_reference(occurrence, session)
    if reference is not None or session is None:
        return reference
    local = session.source_kind == "custom_query" and session.context.get("backend") == "madi3d-local"
    if session.source_kind != "imported_csv" and not local:
        return None
    source = occurrence.target
    if local and not source.image.image_id:
        return None
    if source.kind not in {"em", "lm"} or not source.library or not source.image.alignment_space:
        return None
    name = source.biological.neuron_id if source.kind == "em" else source.biological.line_name
    if not name or any(d.code in {"ambiguous_identity_field", "ambiguous_target_identity", "missing_target_identity"}
                       for d in occurrence.diagnostics):
        return None
    return ThumbnailLookup(source, session.neuronbridge_data_version, library_release_from_catalog=local)


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
        matches = []
        for image in images:
            mismatches = image_mismatches(source, image)
            # Local libraries declare their release in the catalog. The public
            # image response may omit that field; an explicit conflict still fails.
            if request.library_release_from_catalog and image.get("libraryRelease") is None:
                mismatches.pop("libraryRelease", None)
            if not mismatches:
                matches.append(image)
        if len(matches) != 1:
            raise ValueError(f"{len(matches)} published images match the recorded identity. "
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


class LocalPreviewUnavailable(ValueError):
    """The installed library cannot supply this candidate's pixels right now."""


def _encoded(image):
    """One resize/encode path for every raster source, bounded before thumbnailing."""
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("Thumbnail exceeds the preview pixel limit.")
    image.thumbnail(THUMBNAIL_SIZE)
    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def render_library_preview(reference, cancel, *, locate=None):
    """Render one local-search preview from installed candidate pixels only.

    A sparse candidate becomes its exact dense RGB reconstruction (omitted
    pixels ``[0, 0, 0]``); a dense-archive candidate goes through the strict
    TIFF decoder. Both share one resize/encode path. Missing, removed or
    unreadable libraries raise ``LocalPreviewUnavailable``; the public network
    routes are never consulted for a local result.
    """
    from .library_store import installed_library_directory, open_library
    from .local_scorer import checkpoint
    checkpoint(cancel)
    directory = (locate or installed_library_directory)(reference.library_id)
    if directory is None:
        raise LocalPreviewUnavailable(
            f"The installed library {reference.library_id!r} is not available at a known location.")
    try:
        with open_library(directory, cancel=cancel) as reader:
            identity = reader.identity
            if (identity.get("library_id") != reference.library_id or
                    identity.get("archive_sha256") != reference.archive_sha256):
                raise LocalPreviewUnavailable(
                    "Local preview unavailable: the installed library is a different package "
                    "from the one used by this saved result.")
            canvas = reader.dense_canvas({"member_key": reference.member_key, "ordinal": reference.candidate_ordinal})
    except InterruptedError:
        raise
    except (OSError, ValueError) as exc:
        raise LocalPreviewUnavailable(str(exc)) from exc
    checkpoint(cancel)
    if not isinstance(canvas, np.ndarray) or canvas.ndim != 3 or canvas.shape[2] != 3:
        raise LocalPreviewUnavailable("The installed candidate pixels are not an RGB canvas.")
    with Image.fromarray(canvas, mode="RGB") as image:
        content = _encoded(image)
    checkpoint(cancel)
    return content


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
    def __init__(self, root=None, fetch=fetch_thumbnail, resolve=resolve_thumbnail,
                 render=render_library_preview):
        self.root = Path(root) if root is not None else cache_dir() / "NeuronBridge" / "thumbnails"
        self.fetch = fetch
        self.resolve = resolve
        self.render = render
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
        if isinstance(reference, LibraryPreviewReference):
            # Local pixels are rendered from the installed library, with or
            # without "Load thumbnails"; no network route participates.
            content = self.render(reference, cancel)
            if content is None:
                return None
            return self._store(path, content, cancel)
        if not network:
            return None
        resolved = self.resolve(reference, cancel) if isinstance(reference, ThumbnailLookup) else reference
        _checkpoint(cancel)
        content = self.fetch(resolved.url, cancel)
        if len(content) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Thumbnail exceeds the preview download limit.")
        with Image.open(io.BytesIO(content)) as image:
            content = _encoded(image)
        _checkpoint(cancel)
        return self._store(path, content, cancel)

    def _store(self, path, content, cancel):
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
