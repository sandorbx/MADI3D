"""Read-only access to installed local-search distributions.

A distribution is searchable as soon as its package is present under
``ColorDepthLibraries/<library-id>/``: this module opens the published files
directly and never writes, re-indexes, repairs or snapshots them. The
``archive_catalog`` module owns distribution identity, ``archive_install``
owns the download/extract/publish transition, and this module owns reading:

* ``SparseLibraryReader`` — the published sparse package (``inventory.sqlite``
  plus ``index/<shard_id:08d>.npy`` pixel spans). ``position`` is a
  little-endian uint32 linear pixel position; ``rgb`` is three uint8 channels.
  At most the currently required shard is memory-mapped, and shard files and
  the inventory are never hashed before a search.
* ``DenseArchiveReader`` — the extracted official candidate directory
  (``BulkArchive``). Candidates are enumerated in normalized archive-relative
  POSIX path order with bounded threaded prefetch through the strict TIFF
  decoder; the archive's own same-stem gradient PNG is exposed for the same
  candidate identity.

Both readers own the database/file/mmap handles they open and release them on
success, error and cancellation (``close()``, or use them as context
managers). While a reader is open the same application process can prevent
removal of that library directory; there is deliberately no cross-process
lease — the installer's publication and removal boundaries do not coordinate
with an in-flight reader beyond normal OS file semantics.

``decode_rgb``, ``decode_gradient`` and ``decode_shape_boundary`` strictly read
the candidate TIFF, supplied uint16 gradient PNG and binary boundary PNG.
None rescales or resamples.

Validation is bounded to immediate consumption safety: supported package
schema/version, a candidate span inside its shard, the correct pixel dtype,
and the declared canvas/profile. Installation establishes sparse pixel
integrity; searches trust the immutable published package. Provenance columns (source identity, evidence and
diagnostics) are read verbatim from the published inventory; nothing here
re-derives them.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path, PurePosixPath
import sqlite3

import numpy as np

from .archive_catalog import (CATALOG_DATA_VERSION, INVENTORY_FILENAME, PACKAGE_FORMAT,
                              PACKAGE_INVENTORY_TABLES, PACKAGE_MANIFEST_FILENAME,
                              PACKAGE_VERSION, PROVIDER_MADI3D, SHARD_DIRECTORY, SHARD_EXTENSION,
                              SPARSE_RGB_U32_FORMAT, BulkArchive, SparseRelease, distribution)
from .archive_install import INSTALLATION_FILENAME, InstalledArchive, read_installation
from .local_catalog import parse_json
from .local_scorer import checkpoint
from .search_profiles import BRAIN, search_profile

PIXEL_DTYPE = np.dtype([("position", "<u4"), ("rgb", "u1", (3,))])
# The published shard ceiling; a larger mapped span would exceed the pinned
# package layout and the search memory model that budgets one shard.
SHARD_BYTES = 16 * 1024 * 1024
# Identity metadata responses were produced under this bounded read limit.
MAX_METADATA_BYTES = 16 * 1024 * 1024
# ``archive_catalog.PACKAGE_INVENTORY_TABLES`` names the identity/span columns;
# reading for search also passes through the published provenance columns.
READER_MEMBER_COLUMNS = tuple(PACKAGE_INVENTORY_TABLES["members"]) + (
    "pixel_sha256", "image_evidence", "metadata_evidence", "source", "diagnostics")
METADATA_DIRECTORY = "metadata"
# A supplied gradient is one 16-bit canvas; the limit bounds a read before decode.
GRADIENT_DECODE_LIMIT_BYTES = 16 * 1024 * 1024
CANONICAL_MEMBERS_SQL = ("SELECT ordinal, member_key, shard, pixel_offset, pixel_count "
                         "FROM members ORDER BY ordinal")
CITATIONS = (
    "NeuronBridge: Clements et al. (2024), https://doi.org/10.1186/s12859-024-05732-7",
    "Public search images: Janelia FlyLight / FlyEM, https://neuronbridge.janelia.org/",
)


class UnsupportedInstalledLibrary(ValueError):
    """The selected directory is not a supported installed search library."""


def decode_rgb(path, profile=BRAIN):
    """Bound before decoding; do not convert palettes, alpha, scalars or axes."""
    import tifffile
    with tifffile.TiffFile(path) as image:
        if len(image.pages) != 1 or len(image.series) != 1:
            raise ValueError("Search TIFF must contain exactly one RGB plane.")
        page = image.pages[0]
        if (page.shape != (*profile.search_canvas_yx, 3) or page.dtype != np.uint8 or
                page.photometric != tifffile.PHOTOMETRIC.RGB or page.planarconfig != 1 or
                int(page.tags[274].value if 274 in page.tags else 1) != 1):
            raise ValueError("Search TIFF has an unsupported canvas, pixel type, orientation or RGB layout.")
        return page.asarray(maxworkers=1)


def decode_gradient(path, profile=BRAIN):
    """Read one supplied uint16 gradient PNG without converting or rescaling it.

    A supplied gradient is the archive's own producer output, so it must arrive
    as a single-frame 16-bit unsigned PNG on the declared search canvas. Palette,
    signed, 8-bit or resized images are rejected rather than silently turned into
    a gradient with a different meaning; ``gradient_supply`` falls back to the
    pinned producer instead.
    """
    from PIL import Image
    target = Path(path)
    if target.stat().st_size > GRADIENT_DECODE_LIMIT_BYTES:
        raise ValueError("Supplied candidate gradient exceeds the gradient read limit.")
    with Image.open(target) as image:
        if image.format != "PNG" or image.n_frames != 1:
            raise ValueError("A supplied candidate gradient must be a single-frame PNG.")
        pixels = np.asarray(image)
        if pixels.dtype != np.uint16 or pixels.shape != tuple(profile.search_canvas_yx):
            raise ValueError("Supplied candidate gradient is not uint16 pixels on the declared canvas.")
        return pixels.astype(np.uint16, copy=True)


def decode_shape_boundary(path, profile=BRAIN):
    """Read an exact binary uint8 PNG as a detached immutable keep mask."""
    from PIL import Image
    target = Path(path)
    if target.stat().st_size > GRADIENT_DECODE_LIMIT_BYTES:
        raise ValueError("Shape boundary exceeds the PNG read limit.")
    with Image.open(target) as image:
        if image.format != "PNG" or image.n_frames != 1 or image.mode != "L":
            raise ValueError("Shape boundary must be a single-frame uint8 grayscale PNG.")
        pixels = np.asarray(image)
        if pixels.dtype != np.uint8 or pixels.shape != tuple(profile.search_canvas_yx):
            raise ValueError("Shape boundary must be uint8 on the declared canvas.")
        for start in range(0, len(pixels), 32):
            block = pixels[start:start + 32]
            if np.any((block != 0) & (block != 255)):
                raise ValueError("Shape boundary pixels must be exactly 0 or 255.")
        keep = pixels.astype(np.bool_)
        keep.setflags(write=False)
        return keep


def decoder_check():
    """Small real TIFF/PNG round trips, also invoked by the frozen smoke path.

    Each sample re-reads its own compression tag so a missing native codec
    cannot pass by silently being written with another one. PackBits is pinned
    because the official Janelia archive distribution stores its searchable
    candidates that way; LZW and deflate are what MADI3D writes itself.
    """
    import io
    from PIL import Image
    import tifffile
    rgb = np.array([[[19, 117, 250], [0, 0, 0]], [[2, 255, 40], [234, 52, 12]]], dtype=np.uint8)
    pinned = {"lzw": tifffile.COMPRESSION.LZW, "deflate": tifffile.COMPRESSION.DEFLATE,
              "packbits": tifffile.COMPRESSION.PACKBITS}
    for compression, tag in pinned.items():
        stream = io.BytesIO()
        tifffile.imwrite(stream, rgb, photometric="rgb", compression=compression)
        stream.seek(0)
        with tifffile.TiffFile(stream) as image:
            page = image.pages[0]
            if page.compression != tag:
                raise ValueError(f"TIFF {compression} sample did not pin its compression tag.")
            if not np.array_equal(page.asarray(maxworkers=1), rgb):
                raise ValueError(f"TIFF {compression} decoder changed RGB pixels.")
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    stream.seek(0)
    with Image.open(stream) as image:
        if not np.array_equal(np.asarray(image), rgb):
            raise ValueError("PNG decoder changed RGB pixels.")


def sparse_pixels(rgb):
    flat = rgb.reshape(-1, 3)
    positions = np.flatnonzero(np.any(flat != 0, axis=1))
    result = np.empty(len(positions), PIXEL_DTYPE)
    result["position"], result["rgb"] = positions, flat[positions]
    return result


def read_sparse(data, positions):
    result = np.zeros((len(positions), 3), np.uint8)
    if not len(data):
        return result
    indexes = np.searchsorted(data["position"], positions)
    valid = indexes < len(data)
    slots = np.flatnonzero(valid)
    slots = slots[data["position"][indexes[valid]] == positions[valid]]
    result[slots] = data["rgb"][indexes[slots]]
    return result


def citations_for(profile_id, library_name):
    """Attribution retained with every search over one collection."""
    if profile_id == "nb-vnc-prepared-signal-v1":
        collection = "MANC: https://www.janelia.org/project-team/flyem/manc-connectome"
        terms = "MANC searchable CDMs via NeuronBridge: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/"
    elif str(library_name).startswith("FlyEM_Hemibrain"):
        collection = "Hemibrain: https://doi.org/10.7554/eLife.57443"
        terms = "Hemibrain data terms: https://www.janelia.org/node/65250"
    else:
        collection = "FlyWire FAFB: https://flywire.ai/"
        terms = ("NeuronBridge-hosted FlyWire CDMs; redistribution terms under qualification. "
                 "FlyWire public release guidelines: https://flywire.ai/guidelines")
    return (*CITATIONS, collection, terms)


def _close_map(array):
    if array is not None:
        array._mmap.close()


class SparseLibraryReader:
    """Read-only view of one published sparse RGB u32 distribution package."""

    def __init__(self, directory, *, cancel=None):
        self.directory = Path(directory).expanduser().absolute()
        self._cancel = cancel
        checkpoint(cancel)
        self.manifest = self._read_package_manifest()
        try:
            profile = search_profile(self.manifest["search_profile"])
        except (KeyError, TypeError, ValueError) as exc:
            raise UnsupportedInstalledLibrary(
                f"The published package declares search profile {self.manifest.get('search_profile')!r}, "
                "which this MADI3D version does not support.") from exc
        self.profile_id = profile.profile_id
        self.canvas_yx = tuple(self.manifest["search_canvas_yx"])
        if (len(self.canvas_yx) != 2 or any(type(v) is not int or v <= 0 for v in self.canvas_yx)
                or self.canvas_yx != profile.search_canvas_yx):
            raise UnsupportedInstalledLibrary(
                "The published package search canvas disagrees with its pinned search profile.")
        self.candidate_count = self.manifest["candidate_count"]
        self._installation = read_installation(self.directory)
        if (self.directory / INSTALLATION_FILENAME).exists() and self._installation is None:
            raise UnsupportedInstalledLibrary("The managed library installation record is invalid.")
        if self._installation is not None:
            descriptor = distribution(self._installation["library_id"])
            if (not isinstance(descriptor, SparseRelease) or
                    self.candidate_count != self._installation["candidate_count"] or
                    self.candidate_count != descriptor.candidate_count):
                raise UnsupportedInstalledLibrary(
                    "The published package candidate count disagrees with the managed installation.")
        self._mapped = None
        self._mapped_id = None
        self._closed = False
        inventory = self.directory / INVENTORY_FILENAME
        if not inventory.is_file():
            raise UnsupportedInstalledLibrary(f"The published {INVENTORY_FILENAME} is missing.")
        try:
            self._db = sqlite3.connect(inventory.as_uri() + "?mode=ro", uri=True, timeout=30.0)
        except sqlite3.Error as exc:
            raise UnsupportedInstalledLibrary(
                f"The published {INVENTORY_FILENAME} is not a readable SQLite inventory.") from exc
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute("PRAGMA query_only=ON")
            present = {row[1] for row in self._db.execute("PRAGMA table_info(members)")}
            missing = [column for column in READER_MEMBER_COLUMNS if column not in present]
            if missing:
                raise UnsupportedInstalledLibrary(
                    f"The published {INVENTORY_FILENAME} table 'members' lacks the "
                    f"{', '.join(missing)} columns this package format requires.")
            for table, columns in PACKAGE_INVENTORY_TABLES.items():
                present = {row[1] for row in self._db.execute(f"PRAGMA table_info({table})")}
                if [column for column in columns if column not in present]:
                    raise UnsupportedInstalledLibrary(
                        f"The published {INVENTORY_FILENAME} table {table!r} lacks its required columns.")
            stored = int(self._db.execute("SELECT count(*) FROM members").fetchone()[0])
            if stored != self.candidate_count:
                raise UnsupportedInstalledLibrary(
                    f"The published inventory holds {stored} candidates, but its package manifest "
                    f"declares {self.candidate_count}.")
        except sqlite3.Error as exc:
            self._db.close()
            raise UnsupportedInstalledLibrary(
                f"The published {INVENTORY_FILENAME} is not a readable SQLite inventory.") from exc
        except Exception:
            self._db.close()
            raise

    # ------------------------------------------------------------------ identity

    def _read_package_manifest(self):
        path = self.directory / PACKAGE_MANIFEST_FILENAME
        if not path.is_file():
            raise UnsupportedInstalledLibrary(f"The published package manifest {PACKAGE_MANIFEST_FILENAME!r} is missing.")
        try:
            value = parse_json(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise UnsupportedInstalledLibrary(
                f"The published package manifest {PACKAGE_MANIFEST_FILENAME!r} is not readable JSON.") from exc
        if not isinstance(value, dict):
            raise UnsupportedInstalledLibrary(
                f"The published package manifest {PACKAGE_MANIFEST_FILENAME!r} is not a JSON object.")
        for name, expected in (("package_format", PACKAGE_FORMAT), ("package_version", PACKAGE_VERSION),
                               ("index_format", SPARSE_RGB_U32_FORMAT)):
            if value.get(name) != expected:
                raise UnsupportedInstalledLibrary(
                    f"The installed package declares {name}={value.get(name)!r}; this reader supports {expected!r}.")
        if not isinstance(value.get("candidate_count"), int) or value["candidate_count"] <= 0:
            raise UnsupportedInstalledLibrary("The published package manifest has no positive candidate count.")
        return value

    @property
    def representation(self):
        return self.manifest["index_format"]

    @property
    def citations(self):
        return citations_for(self.profile_id, self.manifest["library_name"])

    @property
    def identity(self):
        """Installed distribution facts retained with every search over this library."""
        installation = self._installation or {}
        values = {"library_id": installation.get("library_id"), "library": self.manifest.get("library_name"),
                  "library_release": self.manifest.get("release"), "data_version": self.manifest.get("data_version"),
                  # Sparse release assets are MADI3D-published by definition; an
                  # installation record may restate the provider explicitly.
                  "provider": installation.get("provider") or PROVIDER_MADI3D, "representation": self.representation,
                  # The archive SHA-256 observed during installation is the installed
                  # package identity. A directly published package directory carries
                  # no transferred-archive digest, and none is invented for it.
                  "archive_sha256": installation.get("archive_sha256")}
        return {key: value for key, value in values.items() if value is not None}

    @property
    def searched_collection(self):
        return self.manifest["library_name"]

    # ---------------------------------------------------------------- candidates

    def candidates(self):
        """Yield ``(row, pixels)`` in ordinal order, mapping at most one shard."""
        self._require_open()

        def iterate():
            cursor = self._db.execute(CANONICAL_MEMBERS_SQL)
            try:
                for row in cursor:
                    checkpoint(self._cancel)
                    pixels = self._span(row["shard"], row["pixel_offset"], row["pixel_count"])
                    yield dict(row), pixels
            finally:
                cursor.close()

        return iterate()

    def result_row(self, row):
        """Hydrate one retained scoring row from the trusted inventory."""
        self._require_open()
        retained = self._db.execute("SELECT * FROM members WHERE ordinal=?", (row["ordinal"],)).fetchone()
        if retained is None or any(key in row and row[key] != retained[key] for key in (
                "ordinal", "member_key", "object_id", "pixel_sha256", "shard", "pixel_offset", "pixel_count")):
            raise ValueError("Candidate identity differs from the retained library member.")
        return dict(retained)

    def dense_canvas(self, row):
        """Reconstruct exact published pixels; installation owns their integrity."""
        retained = self.result_row(row)
        pixels = self._span(retained["shard"], retained["pixel_offset"], retained["pixel_count"])
        if retained["pixel_count"] != retained["nonzero_pixels"]:
            raise ValueError("Invalid sparse candidate pixels.")
        checkpoint(self._cancel)
        canvas = np.zeros((*self.canvas_yx, 3), dtype=np.uint8)
        canvas.reshape(-1, 3)[pixels["position"]] = pixels["rgb"]
        checkpoint(self._cancel)
        return canvas

    def shape_boundary(self):
        """Sparse releases declare no shape boundary."""
        self._require_open()
        return None

    def gradient_path(self, row):
        """Sparse packages publish no gradient files, so none is supplied here.

        The published package carries RGB spans only; ``gradient_supply`` runs the
        pinned producer on the exact dense reconstruction instead. The method
        exists so both installed-library readers answer the same question and the
        supply policy has one routing point.
        """
        self._require_open()
        if not isinstance(row.get("member_key"), str):
            raise ValueError("Candidate row does not identify a library member.")
        return None

    def _span(self, shard_id, offset, count):
        mapped = self._map(shard_id)
        if (type(offset) is not int or type(count) is not int or offset < 0 or count < 0
                or offset + count > len(mapped)):
            raise ValueError("Invalid candidate index span.")
        return mapped[offset:offset + count]

    def _map(self, shard_id):
        self._require_open()
        if self._mapped_id == shard_id:
            return self._mapped
        self._unmap()
        if type(shard_id) is not int:
            raise ValueError("Candidate has no search shard span.")
        declared = self._db.execute("SELECT * FROM shards WHERE id=?", (shard_id,)).fetchone()
        target = self.directory / SHARD_DIRECTORY / f"{shard_id:08d}{SHARD_EXTENSION}"
        if declared is None or not target.is_file():
            raise ValueError(f"Declared search shard {shard_id} is missing from the installed package.")
        if target.stat().st_size != declared["size"]:
            raise ValueError(f"Search shard {shard_id} is incomplete on disk.")
        mapped = np.load(target, mmap_mode="r", allow_pickle=False)
        if mapped.dtype != PIXEL_DTYPE or mapped.ndim != 1 or mapped.nbytes > SHARD_BYTES:
            _close_map(mapped)
            raise ValueError("Invalid or oversized search shard.")
        self._mapped, self._mapped_id = mapped, shard_id
        return mapped

    def _unmap(self):
        _close_map(self._mapped)
        self._mapped = None
        self._mapped_id = None

    # ---------------------------------------------------------------- metadata

    def identity_observation(self, evidence, *, cancel=None, check_storage=None):
        """Read exact retained identity evidence published inside this package.

        ``check_storage`` observes parsed storage while its raw bytes still
        live. Evidence without ``response_ref`` was embedded by the producing
        store and is returned verbatim; nothing is fetched or refreshed.
        """
        if not isinstance(evidence, dict):
            raise ValueError("Retained identity evidence is missing.")
        if "response_ref" not in evidence:
            if check_storage is not None:
                check_storage(evidence, 0)
            return evidence
        path = self.directory / METADATA_DIRECTORY / (hashlib.sha256(evidence["url"].encode()).hexdigest() + ".json")
        checkpoint(cancel)
        with path.open("rb") as stream:
            raw = stream.read(MAX_METADATA_BYTES + 1)
        checkpoint(cancel)
        if (len(raw) > MAX_METADATA_BYTES or len(raw) != evidence["size"] or
                evidence["response_ref"] != evidence["sha256"] or hashlib.sha256(raw).hexdigest() != evidence["sha256"]):
            raise ValueError("Retained identity metadata changed. Reinstall this distribution.")
        observation = dict(evidence, observed_response=parse_json(raw))
        if check_storage is not None:
            check_storage(observation, len(raw))
        checkpoint(cancel)
        return observation

    # ----------------------------------------------------------------- lifetime

    def _require_open(self):
        if self._closed:
            raise ValueError("Library reader is closed.")

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._unmap()
        finally:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()
        return False


class DenseArchiveReader:
    """Read-only view of one extracted official candidate directory."""

    def __init__(self, directory, *, cancel=None):
        self.directory = Path(directory).expanduser().absolute()
        self._cancel = cancel
        self._closed = False
        checkpoint(cancel)
        self._installation = read_installation(self.directory)
        if self._installation is None:
            raise UnsupportedInstalledLibrary("A dense library requires a valid managed installation.")
        descriptor = distribution(self._installation["library_id"])
        if not isinstance(descriptor, BulkArchive):
            raise UnsupportedInstalledLibrary("That distribution is not a dense candidate-directory library.")
        self.descriptor = descriptor
        self.profile_id = descriptor.profile_id
        self.canvas_yx = tuple(descriptor.candidate_canvas_yx)
        self._candidate_root = self.directory.joinpath(*PurePosixPath(descriptor.candidate_directory).parts)
        if not self._candidate_root.is_dir():
            raise UnsupportedInstalledLibrary(
                f"The extracted candidate directory {descriptor.candidate_directory!r} is missing.")
        self._candidate_stream = None
        self.candidate_paths = self._scan_candidates()
        if not (len(self.candidate_paths) == self._installation["candidate_count"]
                == descriptor.candidate_count):
            raise UnsupportedInstalledLibrary(
                "The dense library candidate count disagrees with the managed installation.")

    def _scan_candidates(self):
        collected = []
        for path in self._candidate_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.directory).as_posix()
            if self.descriptor.accepts_member(relative):
                collected.append(relative)
        return tuple(sorted(collected))

    # ------------------------------------------------------------------ identity

    @property
    def representation(self):
        return self.descriptor.representation

    @property
    def candidate_count(self):
        return len(self.candidate_paths)

    @property
    def citations(self):
        return citations_for(self.profile_id, self.descriptor.geometry_source_library)

    @property
    def identity(self):
        installation = self._installation
        values = {"library_id": installation["library_id"],
                  "library": self.descriptor.geometry_source_library,
                  "library_release": self.descriptor.geometry_release,
                  # The archive's geometry source collection is the one declared
                  # in the pinned NeuronBridge catalog survey; this route reads
                  # no live catalog.
                  "data_version": CATALOG_DATA_VERSION,
                  "provider": installation.get("provider") or self.descriptor.provider,
                  "representation": self.representation,
                  "archive_id": self.descriptor.archive_id,
                  "archive_sha256": installation.get("archive_sha256")}
        return {key: value for key, value in values.items() if value is not None}

    @property
    def searched_collection(self):
        return self.descriptor.archive_id

    # ---------------------------------------------------------------- candidates

    @property
    def decode_workers(self):
        return min(4, os.cpu_count() or 1)

    def candidates(self):
        """Ordered TIFF prefetch with at most decode_workers in-flight canvases."""
        self._require_open()
        if self._candidate_stream is not None:
            self._candidate_stream.close()
        self._candidate_stream = self._candidates()
        return self._candidate_stream

    def _candidates(self):
        pending = deque()
        paths = iter(enumerate(self.candidate_paths))

        def consume(relative):
            checkpoint(self._cancel)
            rgb = decode_rgb(self._path(relative), self.descriptor.profile)
            checkpoint(self._cancel)
            return rgb

        executor = ThreadPoolExecutor(max_workers=self.decode_workers, thread_name_prefix="nb-cdm")
        try:
            def submit():
                checkpoint(self._cancel)
                item = next(paths, None)
                if item is not None:
                    ordinal, relative = item
                    pending.append(({"ordinal": ordinal, "member_key": relative},
                                    executor.submit(consume, relative)))
            for _ in range(self.decode_workers):
                submit()
            while pending:
                checkpoint(self._cancel)
                row, future = pending.popleft()
                value = future.result()
                del future
                checkpoint(self._cancel)
                yield row, value
                del value
                submit()
        finally:
            for _, future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    def _path(self, relative):
        return self.directory.joinpath(*PurePosixPath(relative).parts)

    def result_row(self, row):
        """Construct retained identity/evidence without decoding rejected candidates."""
        self._require_open()
        ordinal, relative = row["ordinal"], row["member_key"]
        if not 0 <= ordinal < len(self.candidate_paths) or self.candidate_paths[ordinal] != relative:
            raise ValueError("Candidate identity differs from the installed archive directory.")
        identity = self.descriptor.parse_member(relative)
        target = self._path(relative)
        source = {"kind": "em", "provider": "NeuronBridge",
                  "library": self.descriptor.geometry_source_library,
                  "library_release": self.descriptor.geometry_release, "published_name": None,
                  "biological": {"neuron_id": identity.body_id, "line_name": None, "neuron_type": None,
                                 "neuron_instance": None},
                  "image": {"image_id": None, "slide_code": None, "alignment_space": self.descriptor.alignment_space,
                            "sex": None, "magnification": None, "anatomical_area": self.descriptor.profile.anatomical_area,
                            "mounting_protocol": None},
                  "assets": [], "channel": None}
        metadata_evidence = {"identity_contract": self.descriptor.filename_parser,
                             "archive_id": self.descriptor.archive_id, "member_path": relative,
                             "body_id": identity.body_id, "variant": identity.variant or None,
                             "resolution": "official-archive-filename-grammar"}
        image_evidence = {"path": relative, "size": target.stat().st_size}
        return {"ordinal": ordinal, "member_key": relative,
                "object_id": hashlib.sha256(relative.encode("utf-8")).hexdigest(),
                "image_evidence": image_evidence, "metadata_evidence": metadata_evidence,
                "source": source, "diagnostics": [], "shard": None, "pixel_offset": None, "pixel_count": None}

    def dense_canvas(self, row):
        """Decode one retained candidate through the strict TIFF decoder."""
        self._require_open()
        relative = row.get("member_key")
        if not isinstance(relative, str):
            raise ValueError("Candidate identity differs from the installed archive directory.")
        paths = self.candidate_paths
        ordinal = row.get("ordinal")
        if not (type(ordinal) is int and 0 <= ordinal < len(paths) and paths[ordinal] == relative):
            # Dense membership has always accepted a member key without a matching ordinal.
            index = bisect_left(paths, relative)
            if index == len(paths) or paths[index] != relative:
                raise ValueError("Candidate identity differs from the installed archive directory.")
        canvas = decode_rgb(self._path(relative), self.descriptor.profile)
        return canvas

    def shape_boundary(self):
        """Read the declared shape boundary; a required boundary never falls back."""
        self._require_open()
        member = self.descriptor.shape_boundary_member
        if member is None:
            return None
        try:
            return decode_shape_boundary(self._path(member), self.descriptor.profile)
        except (OSError, ValueError, SyntaxError) as exc:
            raise ValueError("The Hemibrain shape-search boundary is unavailable; reinstall the library.") from exc

    def gradient_path(self, row):
        """Official supplied-gradient PNG for the same candidate identity, if any."""
        self._require_open()
        relative = row.get("member_key")
        if not isinstance(relative, str):
            raise ValueError("Candidate row does not identify a library member.")
        member = self.descriptor.supplied_gradient_member(relative)
        if member is None:
            return None
        target = self.directory.joinpath(*PurePosixPath(member).parts)
        return target if target.is_file() else None

    def identity_observation(self, evidence, *, cancel=None, check_storage=None):
        """The archive's grammar-resolved evidence is complete as published."""
        if not isinstance(evidence, dict):
            raise ValueError("Retained identity evidence is missing.")
        checkpoint(cancel)
        if check_storage is not None:
            check_storage(evidence, 0)
        return evidence

    # ----------------------------------------------------------------- lifetime

    def _require_open(self):
        if self._closed:
            raise ValueError("Library reader is closed.")

    def close(self):
        if self._candidate_stream is not None:
            self._candidate_stream.close()
            self._candidate_stream = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()
        return False


def installed_library_directory(library_id, *, root=None):
    """Locate one installed distribution's directory, or ``None`` when it is absent.

    Discovery uses the same NeuronBridge downloaded-assets destination as
    installation and search. This reads only the compact installation record at that
    managed path; nothing is fetched, and an unlocatable library is an
    ordinary absence rather than an error. Preview consumers must treat
    ``None`` as unavailable local pixels, never as a reason to consult a
    public route.
    """
    if not isinstance(library_id, str) or not library_id:
        return None
    from .archive_install import ArchiveInstaller
    try:
        archive = ArchiveInstaller(root).installation(library_id)
    except (ValueError, OSError):
        return None
    return archive.path if archive is not None else None


def open_library(location, *, cancel=None):
    """Open one installed local-search distribution for reading.

    ``location`` is an ``InstalledArchive`` or the library directory itself;
    a directory containing the published sparse package opens without any
    installation step, and an installed dense archive opens through its pinned
    descriptor.
    """
    if isinstance(location, InstalledArchive):
        descriptor = location.descriptor
        return _reader_for(descriptor, location.path, cancel)
    directory = Path(location).expanduser().absolute()
    metadata = read_installation(directory)
    if metadata is not None:
        descriptor = distribution(metadata["library_id"])
        if isinstance(descriptor, BulkArchive):
            return DenseArchiveReader(directory, cancel=cancel)
    if (directory / PACKAGE_MANIFEST_FILENAME).is_file():
        return SparseLibraryReader(directory, cancel=cancel)
    raise UnsupportedInstalledLibrary(
        "Select an installed local-search library: the directory must hold the published "
        "sparse package or an installed official archive distribution.")


def _reader_for(descriptor, directory, cancel):
    if isinstance(descriptor, BulkArchive):
        return DenseArchiveReader(directory, cancel=cancel)
    if isinstance(descriptor, SparseRelease):
        return SparseLibraryReader(directory, cancel=cancel)
    raise UnsupportedInstalledLibrary("That distribution has no supported installed-library reader.")
