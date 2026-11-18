"""One-download, one-extraction installation of a pinned local-search distribution.

Each distribution in ``archive_catalog`` is a complete search library as published:
the official Janelia ``Color_Depth_MIPs_For_Download`` archive holds candidate TIFFs
that are readable straight after extraction, and each MADI3D sparse release asset
already carries its own ``package-manifest.json``, ``inventory.sqlite`` and
``index/<shard:08d>.npy`` pixel spans. One download and one extraction therefore
produce one managed directory plus one compact ``installation.json``; search needs
no follow-up indexing phase, no repair state and no generated index.

This module owns the installation transition only: staged download with proven
Range resume, central-directory inspection and path-safety rejection, real
disk-space preflight, extraction, package verification against the pinned
descriptor, atomic publication and managed removal. ``archive_catalog`` remains
the owner of distribution identity, package layout and the candidate contract,
``local_catalog`` the owner of public-object URL rules, ``library_store`` the
owner of read-only access to an installed distribution, and ``preferences`` the
owner of the user-facing provider destination. Search execution, scoring, mesh
or geometry retrieval are deliberately not here.

Integrity comes from the provider, not from a re-derived index. A MADI3D-published
asset carries an authoritative SHA-256 that the received bytes must equal; the
Janelia archive publishes no whole-object digest (its ETag is a multipart marker),
so the SHA-256 observed while downloading becomes the installed-package identity
and a changed ETag or Last-Modified header never refuses an installation.

Crash discipline: every interruption leaves either the previous installation
intact or the new installation complete. An interrupted download may retain its
``.part`` file plus the small resume record needed to continue it; an interrupted
extraction never makes a library appear installed; a replacement moves the working
copy aside only after the new extraction has fully verified, and ``recover()``
heals that window.

Network access is confined to the injectable transport, so installation, resume
and every failure mode are testable offline against synthetic archives. One
install runs at a time per destination; the calling layer serialises requests.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import zipfile
import zlib
from threading import Event, Lock
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

import requests
from urllib3.exceptions import HTTPError as Urllib3HTTPError

from madi3d_storage import atomic_write_json, read_json_object_with_status
from .archive_catalog import (ARCHIVE_LIBRARY_DIRECTORY, BulkArchive, GRADIENT_INCLUDED, INVENTORY_FILENAME,
                             PACKAGE_FORMAT, PACKAGE_INVENTORY_TABLES, PACKAGE_MANIFEST_FILENAME,
                             PACKAGE_VERSION, SHARD_DIRECTORY, SHARD_EXTENSION, SparseRelease,
                             archive_library_directory, distribution, distributions, portable_member_path)
from .cache import MIN_FREE_DISK_BYTES, neuronbridge_cache_root
from .local_catalog import bulk_url, file_hash, now, parse_json
from .local_scorer import SearchCancelled, checkpoint

INSTALLATION_SCHEMA_VERSION = "madi3d-local-search-installation-v2"
INSTALLATION_FILENAME = "installation.json"
RESUME_SCHEMA_VERSION = "madi3d-local-search-resume-v2"
# Staged download and extraction live beside the installed libraries so every byte
# of a transition stays on the destination filesystem and publication is a
# directory rename rather than a copy. The dot-prefixed name can never collide
# with a distribution library identifier.
STAGING_DIRECTORY = ".staging"
EXTRACTION_SUFFIX = ".extract-"
REPLACED_SUFFIX = ".replacing-"
DOWNLOAD_CHUNK_BYTES = 1 << 20
# The resume record is rewritten at this interval, so a killed process loses at
# most one interval of transfer instead of the whole download.
SIDECAR_INTERVAL_BYTES = 64 << 20
TRANSFER_ATTEMPTS = 3
CLEAN_RESTART_LIMIT = 3
RETRY_DELAY_SECONDS = 2.0
EXTRACTION_MARGIN_RATIO = 0.05
EXTRACTION_MARGIN_FLOOR_BYTES = 1 << 30
CONNECT_TIMEOUT, READ_TIMEOUT = 10, 60
SQLITE_TIMEOUT_SECONDS = 30.0
# A Janelia S3 object is served directly; a GitHub release asset is served through
# one signed CDN hop. Redirects are followed only inside these hosts and only over
# HTTPS, so a pinned HTTPS source can never be downgraded or moved to a host that
# is not the pinned provider.
GITHUB_RELEASE_HOST = "github.com"
RELEASE_ASSET_HOSTS = frozenset({GITHUB_RELEASE_HOST, "release-assets.githubusercontent.com",
                                 "objects.githubusercontent.com"})
RELEASE_PATH_PATTERN = r"/[^/?#\\]+/[^/?#\\]+/releases/download/[^/?#\\]+/[^/?#\\]+"
REDIRECT_STATUSES = (301, 302, 303, 307, 308)
MAX_REDIRECT_HOPS = 5
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class InvalidArchive(ValueError):
    """The ZIP is structurally unsound or holds a member that is unsafe to extract."""


class ArchiveContractMismatch(ValueError):
    """The observed object or extracted package disagrees with its pinned descriptor."""


class ArchiveResourceError(OSError):
    """The transfer or extraction would exceed the destination's explicit disk budget."""


class _CleanRestart(Exception):
    """A partial transfer cannot be proven safe to append to, so it starts over."""


def extraction_margin(expansion_bytes):
    """Return the bounded slack added to a real uncompressed central-directory total."""
    return max(EXTRACTION_MARGIN_FLOOR_BYTES, int(int(expansion_bytes) * EXTRACTION_MARGIN_RATIO))


@dataclass(frozen=True)
class ArchiveResourcePolicy:
    """Explicit disk budget for one pinned distribution archive.

    The generic per-asset ceiling used for cached NeuronBridge assets cannot apply
    here: the sparse release assets are ~1.1-1.2 GB compressed and expand to
    several gigabytes of pixel shards, and the official Janelia archive expands
    from ~0.9 GB to ~4.8 GB. The download requirement comes from the selected
    descriptor and the extraction requirement from the real central-directory total.
    """

    descriptor: object
    reserve_bytes: int = MIN_FREE_DISK_BYTES

    def __post_init__(self):
        if not isinstance(self.descriptor, (BulkArchive, SparseRelease)):
            raise ValueError("An archive resource policy requires one distribution descriptor.")
        if type(self.reserve_bytes) is not int or self.reserve_bytes <= 0:
            raise ValueError("The archive disk reserve must be a positive byte count.")

    @classmethod
    def for_archive(cls, descriptor, *, reserve_bytes=MIN_FREE_DISK_BYTES):
        return cls(descriptor, reserve_bytes)

    @property
    def download_bytes(self):
        return self.descriptor.expected_size_bytes + self.reserve_bytes

    def extraction_bytes(self, expansion_bytes):
        return int(expansion_bytes) + extraction_margin(expansion_bytes) + self.reserve_bytes


def official_library_root(root=None):
    """Return the managed ``ColorDepthLibraries`` directory for one destination.

    Without an explicit root this uses the default NeuronBridge downloaded-assets destination.
    """
    base = Path(root).expanduser() if root is not None else neuronbridge_cache_root()
    return Path(base) / ARCHIVE_LIBRARY_DIRECTORY


def distribution_url(url):
    """Return the pinned HTTPS source of one distribution, refusing anything else.

    Janelia objects keep ``local_catalog``'s public-bucket rule. MADI3D assets are
    GitHub release-asset URLs, which are not S3 objects, so they are accepted only
    in their canonical release form: HTTPS, no credentials, port or query, and the
    ``<owner>/<repo>/releases/download/<tag>/<asset>`` path.
    """
    parsed = urlsplit(str(url))
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port
            or parsed.fragment or parsed.query or "\\" in str(url)):
        raise ValueError("A distribution source must be a direct HTTPS object URL.")
    if parsed.hostname == GITHUB_RELEASE_HOST:
        if not re.fullmatch(RELEASE_PATH_PATTERN, parsed.path):
            raise ValueError("A MADI3D distribution source must be a pinned GitHub release asset URL.")
        return url
    return bulk_url(url)


def _redirect_target(current, location):
    """Resolve one redirect hop, refusing any move away from the pinned HTTPS provider."""
    if not location:
        raise ValueError("The distribution source redirected without a target; the download was refused.")
    target = urljoin(current, location)
    parsed = urlsplit(target)
    if parsed.scheme != "https":
        raise ValueError("The distribution source redirected away from HTTPS; the download was refused.")
    if parsed.hostname not in RELEASE_ASSET_HOSTS:
        raise ValueError("The distribution source redirected outside the pinned release hosts; "
                         "the download was refused.")
    return target


def _validator(headers):
    observed = {"etag": str(headers.get("ETag") or "").strip(),
                "last_modified": str(headers.get("Last-Modified") or "").strip()}
    return {key: value for key, value in observed.items() if value} or None


def _verify_pinned_size(descriptor, total):
    """Require the served object to still be the pinned archive in length.

    Length is a preflight, not an identity claim: identity is the SHA-256 of the
    received bytes. Provider ETag and Last-Modified headers are observations, not
    contracts, and a changed one never refuses an installation.
    """
    if total is not None and int(total) != descriptor.expected_size_bytes:
        raise ArchiveContractMismatch(
            f"{descriptor.library_id} is served as {total} bytes, but its pinned contract declares "
            f"{descriptor.expected_size_bytes}. The object was replaced upstream; re-establish its "
            "pinned descriptor before installing it.")


def _confirmed_identity(descriptor, evidence):
    """Bind the observed digest to the pinned descriptor before a byte is extracted."""
    observed = evidence["sha256"]
    pinned = descriptor.provider_sha256
    if pinned and observed != pinned:
        raise ArchiveContractMismatch(
            f"{descriptor.library_id} downloaded as SHA-256 {observed}, not the pinned {pinned}. The "
            "served object is not the published package; nothing was installed.")
    return evidence


def _content_range(header, requested_start):
    """Return ``(start, total)`` from a ``bytes`` Content-Range header, or ``None``."""
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", str(header or "").strip())
    if match is None:
        return None
    start, _, total = (int(part) for part in match.groups())
    if start != int(requested_start):
        raise _CleanRestart("The server returned a different byte range than requested; the download "
                            "restarts from the beginning.")
    return start, total


def _digest_prefix(path, length, expected, cancel):
    """Return a SHA-256 object holding exactly the first ``length`` verified bytes."""
    digest = hashlib.sha256()
    remaining = int(length)
    with Path(path).open("rb") as stream:
        while remaining > 0:
            checkpoint(cancel)
            chunk = stream.read(min(DOWNLOAD_CHUNK_BYTES, remaining))
            if not chunk:
                raise _CleanRestart("The partial archive is shorter than its resume record; the download "
                                    "restarts from the beginning.")
            digest.update(chunk)
            remaining -= len(chunk)
    if digest.hexdigest() != expected:
        raise _CleanRestart("The partial archive no longer matches its resume record; the download "
                            "restarts from the beginning.")
    return digest


def _staged_files(staging_root, descriptor):
    """Return the staged archive, its resumable part and its bounded resume record."""
    prefix = str(Path(staging_root) / descriptor.library_id)
    return Path(prefix + ".zip"), Path(prefix + ".zip.part"), Path(prefix + ".resume.json")


def _write_resume(sidecar, descriptor, *, offset, sha256, validator, complete, observed_at):
    """Persist only what is required to prove a later transfer may append to this file."""
    atomic_write_json(sidecar, {"schema_version": RESUME_SCHEMA_VERSION,
                                "library_id": descriptor.library_id, "url": descriptor.url,
                                "archive_size_bytes": descriptor.expected_size_bytes,
                                "bytes": int(offset), "sha256": sha256, "validator": validator,
                                "complete": bool(complete), "observed_at": observed_at})


@dataclass(frozen=True)
class _ResumeRecord:
    offset: int
    sha256: str
    validator: dict | None
    complete: bool
    observed_at: str


@dataclass
class _ResumePoint:
    """A prefix whose bytes are already absorbed into ``digest``."""
    offset: int
    validator: dict | None
    digest: object


def _read_resume(sidecar, descriptor):
    """Return the recorded resume state, or ``None`` when it cannot describe this object."""
    result = read_json_object_with_status(sidecar)
    if result.status != "valid":
        return None
    value = result.value
    if (value.get("schema_version") != RESUME_SCHEMA_VERSION
            or value.get("library_id") != descriptor.library_id
            or value.get("url") != descriptor.url
            or int(value.get("archive_size_bytes") or 0) != descriptor.expected_size_bytes):
        return None
    offset, digest = value.get("bytes"), value.get("sha256")
    if type(offset) is not int or offset < 0 or not _SHA256_HEX.fullmatch(str(digest or "")):
        return None
    validator = value.get("validator")
    if validator is not None and not isinstance(validator, dict):
        return None
    return _ResumeRecord(offset, digest, validator, bool(value.get("complete")),
                         str(value.get("observed_at") or ""))


def _member_is_unsafe(info):
    """Return why one central-directory entry cannot be extracted safely, else ``""``.

    Only regular files and plain directories are accepted: absolute paths, ``..``,
    backslashes, unusable characters, Windows device names, symlinks, devices,
    encrypted members and unsupported compression are rejected before anything
    touches the destination filesystem.
    """
    name = info.filename
    mode = info.external_attr >> 16
    # Only the type bits decide here: DOS-authored archives carry permission or
    # attribute bits with no type field at all, which is an ordinary file.
    file_type = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode):
        return "is a symbolic link"
    if name.endswith("/"):
        if file_type not in (0, stat.S_IFDIR, stat.S_IFREG):
            return "is not a regular directory entry"
        try:
            portable_member_path(name[:-1])
        except ValueError as exc:
            return f"is an unsafe directory path: {exc}"
        return ""
    if file_type not in (0, stat.S_IFREG):
        return "is not a regular file"
    if info.flag_bits & 0x1:
        return "is encrypted"
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2,
                                 zipfile.ZIP_LZMA):
        return f"uses unsupported compression method {info.compress_type}"
    try:
        portable_member_path(name)
    except ValueError as exc:
        return f"is an unsafe path: {exc}"
    if PurePosixPath(name).parts == (INSTALLATION_FILENAME,):
        return f"would displace the reserved {INSTALLATION_FILENAME} record"
    return ""


def _plan_members(reader, descriptor):
    """Inspect the central directory once: reject unsafe members and sum real sizes.

    Returns ``(entries, expansion_bytes)``, where ``entries`` holds
    ``(info, member_path, is_directory)`` in central-directory order.
    """
    entries, seen, expansion = [], set(), 0
    for info in reader.infolist():
        reason = _member_is_unsafe(info)
        if reason:
            raise InvalidArchive(f"Archive member {info.filename!r} {reason}.")
        directory = info.filename.endswith("/")
        member = info.filename[:-1] if directory else info.filename
        if member in seen:
            raise InvalidArchive(f"Archive member {member!r} is declared more than once.")
        seen.add(member)
        if not directory:
            expansion += int(info.file_size)
        entries.append((info, member, directory))
    if not entries:
        raise InvalidArchive("The archive central directory declares no members.")
    return tuple(entries), expansion


def _extracted_sizes(entries):
    return {member: int(info.file_size) for info, member, is_dir in entries if not is_dir}


def _require_complete_on_disk(files, sizes, destination, cancel):
    """Prove every declared member reached the disk whole, without hashing or decoding it."""
    for member in files:
        checkpoint(cancel)
        target = destination.joinpath(*PurePosixPath(member).parts)
        if not target.is_file() or target.stat().st_size != sizes[member]:
            raise InvalidArchive(f"Extracted member {member!r} is incomplete on disk.")


def _verify_supplied_gradients(descriptor, candidates, files):
    """Require the archive's own gradient directory to be structurally usable.

    A supplied gradient is one PNG beside each candidate, named by the candidate's
    own stem. The directory's presence, its size and the pinned candidates' stems are
    checked; no image is decoded.
    """
    if descriptor.gradient_policy != GRADIENT_INCLUDED:
        return

    if descriptor.shape_boundary_member is not None and descriptor.shape_boundary_member not in files:
        raise ArchiveContractMismatch("The Hemibrain shape-search boundary is unavailable; reinstall the library.")

    def stem(member):
        return member.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()

    prefix = descriptor.gradient_directory + "/"
    images = {stem(member) for member in files
              if member.startswith(prefix) and member.lower().endswith(".png")
              and member != descriptor.shape_boundary_member}
    if len(images) < descriptor.candidate_count:
        raise ArchiveContractMismatch(
            f"{descriptor.gradient_directory}/ holds {len(images)} gradient images, but "
            f"{descriptor.library_id} supplies a gradient for each of its "
            f"{descriptor.candidate_count} candidates.")
    unmatched = [member for member in candidates if stem(member) not in images]
    if unmatched:
        raise ArchiveContractMismatch(
            f"{descriptor.gradient_directory}/ has no gradient image for {len(unmatched)} candidates, "
            f"for example {unmatched[0]!r}.")


def _verify_dense_package(descriptor, entries, destination, cancel):
    """Apply the bounded candidate and gradient checks established with the descriptor.

    This proves the extraction really is the pinned search library. It never hashes
    or decodes an individual candidate and never asks the metadata API anything.
    """
    sizes = _extracted_sizes(entries)
    files = tuple(sizes)
    if len(files) != descriptor.member_count:
        raise ArchiveContractMismatch(
            f"The archive extracted {len(files)} members, but {descriptor.library_id} declares "
            f"{descriptor.member_count}.")
    candidates = tuple(member for member in files if descriptor.accepts_member(member))
    if len(candidates) != descriptor.candidate_count:
        raise ArchiveContractMismatch(
            f"The archive holds {len(candidates)} searchable candidates, but {descriptor.library_id} "
            f"requires {descriptor.candidate_count}.")
    try:
        identities = tuple(descriptor.parse_member(member) for member in candidates)
    except ValueError as exc:
        raise ArchiveContractMismatch(
            f"An archive candidate is outside the {descriptor.filename_parser} filename grammar pinned "
            f"for {descriptor.library_id}.") from exc
    bodies = {identity.body_id for identity in identities}
    if len(bodies) != descriptor.distinct_body_count:
        raise ArchiveContractMismatch("Archive candidate body IDs disagree with the pinned descriptor.")
    if sum(1 for identity in identities if identity.is_variant) != descriptor.candidate_variant_count:
        raise ArchiveContractMismatch("Archive candidate variants disagree with the pinned descriptor.")
    for member in descriptor.verified_member_paths:
        if member not in sizes:
            raise ArchiveContractMismatch(f"Pinned candidate {member!r} is missing from the archive.")
    _verify_supplied_gradients(descriptor, candidates, files)
    _require_complete_on_disk(files, sizes, destination, cancel)
    return {"member_count": len(files), "candidate_count": len(candidates),
            "extracted_bytes": sum(sizes.values())}


def _read_package_manifest(destination):
    """Return the published package's own manifest, or refuse the package."""
    path = destination / PACKAGE_MANIFEST_FILENAME
    if not path.is_file():
        raise ArchiveContractMismatch(
            f"The published package manifest {PACKAGE_MANIFEST_FILENAME!r} is missing.")
    try:
        value = parse_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ArchiveContractMismatch(
            f"The published package manifest {PACKAGE_MANIFEST_FILENAME!r} is not readable JSON.") from exc
    if not isinstance(value, dict):
        raise ArchiveContractMismatch(
            f"The published package manifest {PACKAGE_MANIFEST_FILENAME!r} is not a JSON object.")
    return value


def _verify_package_declaration(descriptor, manifest):
    """Require the published manifest to describe this pinned distribution."""
    declared = {"package_format": PACKAGE_FORMAT, "package_version": PACKAGE_VERSION,
                "index_format": descriptor.representation,
                "library_name": descriptor.geometry_source_library, "release": descriptor.geometry_release,
                "search_profile": descriptor.profile_id, "alignment_space": descriptor.alignment_space,
                "candidate_count": descriptor.candidate_count}
    for name, expected in declared.items():
        if manifest.get(name) != expected:
            raise ArchiveContractMismatch(
                f"The published package declares {name}={manifest.get(name)!r}, but {descriptor.library_id} "
                f"is pinned to {expected!r}. The release asset is not this distribution.")
    canvas = manifest.get("search_canvas_yx")
    if (not isinstance(canvas, list) or len(canvas) != 2
            or any(type(value) is not int for value in canvas)
            or tuple(canvas) != descriptor.profile.search_canvas_yx):
        raise ArchiveContractMismatch(
            f"The published package search canvas {canvas!r} disagrees with the {descriptor.profile_id} "
            f"search grid pinned for {descriptor.library_id}.")


def _verify_payload_coverage(manifest, files, sizes):
    """Require the published manifest to name exactly the payload the archive carries.

    The manifest is the package's own file list, so it also proves nothing extra was
    published alongside the library and no declared file was left behind. Its
    per-file SHA-256 values are not recomputed here: the whole archive was already
    verified against the pinned digest, and ZIP CRC checking covers extraction.
    """
    listed = manifest.get("files")
    if not isinstance(listed, list) or not listed:
        raise ArchiveContractMismatch("The published package manifest declares no payload files.")
    declared = {}
    for entry in listed:
        if not isinstance(entry, dict):
            raise ArchiveContractMismatch("A published package file entry is not an object.")
        path = str(entry.get("path") or "")
        try:
            portable_member_path(path)
        except ValueError as exc:
            raise ArchiveContractMismatch(f"The published package declares an unsafe file path {path!r}.") from exc
        if not _SHA256_HEX.fullmatch(str(entry.get("sha256") or "")):
            raise ArchiveContractMismatch(f"The published package entry {path!r} has no SHA-256.")
        size = entry.get("size")
        if type(size) is not int or size < 0:
            raise ArchiveContractMismatch(f"The published package entry {path!r} has no usable size.")
        declared[path] = int(size)
    if set(declared) | {PACKAGE_MANIFEST_FILENAME} != set(files):
        raise ArchiveContractMismatch(
            f"The published package manifest names {len(declared)} payload files, but the archive "
            f"carries {len(files)}. The package is incomplete or carries undeclared files.")
    for path, size in declared.items():
        if sizes[path] != size:
            raise ArchiveContractMismatch(
                f"The published package declares {path!r} as {size} bytes, but the archive holds "
                f"{sizes[path]}.")


def _verify_sparse_inventory(descriptor, destination, cancel):
    """Return the candidate count proven by the package's own inventory.

    The installer reads the published ``inventory.sqlite`` the way any consumer
    would: it requires the tables and columns that locate a candidate span, one row
    per pinned candidate, a span for every row, and a shard file for every span.
    Nothing is written, rebuilt or re-indexed.
    """
    inventory = destination / INVENTORY_FILENAME
    if not inventory.is_file():
        raise ArchiveContractMismatch(f"The published {INVENTORY_FILENAME} is missing from the package.")
    try:
        connection = sqlite3.connect(inventory.as_uri() + "?mode=ro", uri=True,
                                     timeout=SQLITE_TIMEOUT_SECONDS)
    except sqlite3.Error as exc:
        raise InvalidArchive(f"The published {INVENTORY_FILENAME} is not a readable SQLite inventory.") from exc
    try:
        try:
            connection.execute("PRAGMA query_only=ON")
            for table, columns in PACKAGE_INVENTORY_TABLES.items():
                present = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                missing = [column for column in columns if column not in present]
                if missing:
                    raise ArchiveContractMismatch(
                        f"The published {INVENTORY_FILENAME} table {table!r} lacks the "
                        f"{', '.join(missing)} columns this package format requires.")
            candidates = int(connection.execute("SELECT count(*) FROM members").fetchone()[0])
            if candidates != descriptor.candidate_count:
                raise ArchiveContractMismatch(
                    f"The published inventory holds {candidates} candidates, but {descriptor.library_id} "
                    f"requires {descriptor.candidate_count}.")
            unspanned = int(connection.execute(
                "SELECT count(*) FROM members WHERE shard IS NULL OR pixel_offset IS NULL OR pixel_count IS NULL"
            ).fetchone()[0])
            if unspanned:
                raise ArchiveContractMismatch(
                    f"{unspanned} published candidates have no search shard span.")
            orphaned = int(connection.execute(
                "SELECT count(*) FROM members m LEFT JOIN shards s ON s.id = m.shard WHERE s.id IS NULL"
            ).fetchone()[0])
            if orphaned:
                raise ArchiveContractMismatch(
                    f"{orphaned} published candidates point at a search shard the inventory does not declare.")
            for shard_id, size in connection.execute("SELECT id, size FROM shards ORDER BY id"):
                checkpoint(cancel)
                member = f"{SHARD_DIRECTORY}/{int(shard_id):08d}{SHARD_EXTENSION}"
                target = destination.joinpath(*PurePosixPath(member).parts)
                if not target.is_file() or target.stat().st_size != int(size):
                    raise ArchiveContractMismatch(
                        f"Declared search shard {member!r} is missing or incomplete on disk.")
        except sqlite3.Error as exc:
            raise InvalidArchive(
                f"The published {INVENTORY_FILENAME} is not a readable SQLite inventory.") from exc
        return candidates
    finally:
        connection.close()


def _verify_sparse_package(descriptor, entries, destination, cancel):
    """Prove the extraction is the pinned MADI3D sparse package exactly as published.

    The package owns its own manifest, inventory and pixel shards, so the installer
    verifies that published contract instead of building a second one, and publishes
    the package files a search reads without generating another index.
    """
    sizes = _extracted_sizes(entries)
    files = tuple(sizes)
    manifest = _read_package_manifest(destination)
    _verify_package_declaration(descriptor, manifest)
    _verify_payload_coverage(manifest, files, sizes)
    candidates = _verify_sparse_inventory(descriptor, destination, cancel)
    _require_complete_on_disk(files, sizes, destination, cancel)
    return {"member_count": len(files), "candidate_count": candidates,
            "extracted_bytes": sum(sizes.values())}


def _verify_package(descriptor, entries, destination, cancel):
    """Apply the checks this distribution's published package shape requires."""
    if isinstance(descriptor, SparseRelease):
        return _verify_sparse_package(descriptor, entries, destination, cancel)
    return _verify_dense_package(descriptor, entries, destination, cancel)


def _installation_metadata(descriptor, evidence, summary):
    """Return the compact installation record; the ZIP member list is never retained."""
    record = {"schema_version": INSTALLATION_SCHEMA_VERSION, "library_id": descriptor.library_id,
              "provider": descriptor.provider, "source_url": descriptor.url,
              "source_release": descriptor.source_release, "representation": descriptor.representation,
              "search_profile": descriptor.profile_id, "archive_sha256": evidence["sha256"],
              "candidate_count": summary["candidate_count"], "installed_at": now()}
    if descriptor.provider_sha256:
        record["provider_sha256"] = descriptor.provider_sha256
    return record


def _package_is_present(descriptor, directory):
    """Whether the published files a search reads are still where they were installed."""
    if isinstance(descriptor, SparseRelease):
        return ((Path(directory) / INVENTORY_FILENAME).is_file()
                and (Path(directory) / SHARD_DIRECTORY).is_dir())
    return (Path(directory) / descriptor.candidate_directory).is_dir()


def _installation_is_valid(metadata, directory):
    """Return whether one record still describes a usable managed installation."""
    if not isinstance(metadata, dict) or metadata.get("schema_version") != INSTALLATION_SCHEMA_VERSION:
        return False
    library_id = metadata.get("library_id")
    if not isinstance(library_id, str) or library_id != Path(directory).name:
        return False
    try:
        descriptor = distribution(library_id)
    except (TypeError, ValueError):
        return False
    expected = {"provider": descriptor.provider, "source_url": descriptor.url,
                "source_release": descriptor.source_release, "representation": descriptor.representation,
                "search_profile": descriptor.profile_id, "candidate_count": descriptor.candidate_count}
    if any(metadata.get(key) != value for key, value in expected.items()):
        return False
    if metadata.get("provider_sha256") != (descriptor.provider_sha256 or None):
        return False
    if not _SHA256_HEX.fullmatch(str(metadata.get("archive_sha256") or "")):
        return False
    if not str(metadata.get("installed_at") or "").strip():
        return False
    return _package_is_present(descriptor, directory)


def read_installation(directory):
    """Return validated metadata for one candidate directory, or ``None``."""
    result = read_json_object_with_status(Path(directory) / INSTALLATION_FILENAME)
    if result.status != "valid" or not _installation_is_valid(result.value, directory):
        return None
    return result.value


@dataclass(frozen=True)
class InstalledArchive:
    """One published distribution: its extracted package plus its compact record."""

    library_id: str
    path: Path
    metadata: dict = field(compare=False)

    @property
    def descriptor(self):
        return distribution(self.library_id)

    @property
    def search_root(self):
        """The directory search reads straight away; no indexing step is required."""
        descriptor = self.descriptor
        return self.path / descriptor.candidate_directory if isinstance(descriptor, BulkArchive) else self.path

    @property
    def archive_sha256(self):
        """The SHA-256 observed for the installed package during its transfer."""
        return self.metadata["archive_sha256"]

    @property
    def provider_sha256(self):
        return self.metadata.get("provider_sha256")

    @property
    def candidate_count(self):
        return self.metadata["candidate_count"]

    @property
    def provider(self):
        return self.metadata["provider"]

    @property
    def representation(self):
        return self.metadata["representation"]

    @property
    def search_profile(self):
        return self.metadata["search_profile"]

    @property
    def source_url(self):
        return self.metadata["source_url"]

    @property
    def source_release(self):
        return self.metadata["source_release"]

    @property
    def installed_at(self):
        return self.metadata["installed_at"]


class ArchiveTransport:
    """Range-capable GETs for one pinned distribution object.

    A byte-range resume is required to move a multi-gigabyte archive, so this
    transport adds the range request over the pinned-source discipline it
    shares with the rest of the NeuronBridge routes: pinned public sources
    only, no ambient credentials, no compressed transfer, no interpretation of
    an ETag as a checksum, and redirects followed only inside the pinned GitHub
    release hosts and only over HTTPS.
    """

    def __init__(self, cancel=None, *, session=None):
        self.cancel = cancel or Event()
        self.session = session or requests.Session()
        self.session.trust_env = False
        self._response_lock = Lock()
        self._active_response = None
        add_callback = getattr(self.cancel, "add_callback", None)
        if add_callback is not None:
            add_callback(self._abort_active_request)

    def close(self):
        remove_callback = getattr(self.cancel, "remove_callback", None)
        if remove_callback is not None:
            remove_callback(self._abort_active_request)
        self._abort_active_request()
        self.session.close()

    def _abort_active_request(self):
        with self._response_lock:
            response = self._active_response
        if response is None:
            return
        shutdown = getattr(response.raw, "shutdown", None)
        try:
            shutdown() if shutdown is not None else response.close()
        except (OSError, RuntimeError, ValueError):
            pass  # The worker may have released the connection concurrently.

    def _activate(self, response):
        with self._response_lock:
            self._active_response = response
        if self.cancel.is_set():
            self._abort_active_request()

    def _request(self, url, headers):
        response = self.session.get(url, stream=True, allow_redirects=False,
                                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), headers=headers)
        hops = 0
        while response.status_code in REDIRECT_STATUSES:
            location = response.headers.get("Location")
            response.close()
            if hops >= MAX_REDIRECT_HOPS:
                raise ValueError("The distribution source redirected repeatedly; the download was refused.")
            url = _redirect_target(url, location)
            hops += 1
            response = self.session.get(url, stream=True, allow_redirects=False,
                                        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), headers=headers)
        return response

    @contextmanager
    def open(self, url, *, start=0):
        """Yield the streaming response for ``url``, optionally from byte ``start``."""
        distribution_url(url)
        checkpoint(self.cancel)
        headers = {"Accept-Encoding": "identity"}
        if start:
            headers["Range"] = f"bytes={int(start)}-"
        response = self._request(url, headers)
        self._activate(response)
        try:
            yield response
        finally:
            with self._response_lock:
                if self._active_response is response:
                    self._active_response = None
            response.close()


class ArchiveInstaller:
    """Install, rediscover and remove managed local-search distributions."""

    def __init__(self, root=None, *, disk_usage=shutil.disk_usage):
        base = Path(root).expanduser() if root is not None else neuronbridge_cache_root()
        self.destination = Path(base)
        self.root = official_library_root(base)
        self._disk_usage = disk_usage

    # ---------------------------------------------------------------- discovery

    def path(self, library_id):
        """Return the managed directory for one distribution, named by its library ID."""
        return archive_library_directory(library_id, root=self.destination)

    @property
    def staging_root(self):
        return self.root / STAGING_DIRECTORY

    def installation(self, library_id):
        """Return the installed library for one distribution, or ``None`` when absent."""
        distribution(library_id)
        path = self.path(library_id)
        metadata = read_installation(path)
        return None if metadata is None else InstalledArchive(library_id, path, metadata)

    def installed(self):
        """Return every installed distribution, keyed by MADI3D library identifier.

        The scan is read-only and trusts the compact record rather than walking the
        tree: a directory counts as installed only when its ``installation.json``
        still agrees with its pinned descriptor and its published package is present.
        """
        found = {}
        if not self.root.is_dir():
            return found
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir() or entry.is_symlink():
                continue
            metadata = read_installation(entry)
            if metadata is not None:
                found[metadata["library_id"]] = InstalledArchive(metadata["library_id"], entry, metadata)
        return found

    def required_bytes(self, library_id):
        """Return the download budget for one distribution before anything is fetched."""
        policy = ArchiveResourcePolicy.for_archive(distribution(library_id))
        return {"download_bytes": policy.download_bytes, "reserve_bytes": policy.reserve_bytes,
                "basis": ("Pinned compressed archive size plus the destination reserve. Extraction is "
                          "re-checked against the real central-directory total before unpacking.")}

    # ----------------------------------------------------------------- recovery

    def recover(self):
        """Heal an interrupted replacement; return the healed library identifiers.

        A crash between moving the previous installation aside and moving the new
        one into place leaves the working copy under a ``.replacing-`` name, which
        discovery never reports as installed. Restoring it here means the library is
        installed again rather than missing; a superseded copy left behind after a
        successful publish, and any abandoned extraction directory, are discarded.
        """
        healed = []
        if not self.root.is_dir():
            return healed
        for descriptor in distributions():
            key = descriptor.library_id
            final = self.path(key)
            asides = [entry for entry in sorted(self.root.iterdir())
                      if entry.name.startswith(key + REPLACED_SUFFIX)
                      and entry.is_dir() and not entry.is_symlink()]
            if not final.exists() and len(asides) == 1:
                os.rename(asides[0], final)
                healed.append(key)
                continue
            for aside in asides:
                shutil.rmtree(aside, ignore_errors=True)
            self._clear_extraction(descriptor)
        return healed

    # ------------------------------------------------------------------- install

    def install(self, library_id, *, cancel=None, progress=None, transport=None):
        """Download, extract, verify and publish one pinned distribution library.

        An existing installation is replaced only after the new extraction has
        completed and passed every check, so a failed reinstall never destroys a
        working library.
        """
        descriptor = distribution(library_id)
        report = progress or (lambda phase, done, total: None)
        events = cancel if cancel is not None else Event()
        self.root.mkdir(parents=True, exist_ok=True)
        self.recover()
        policy = ArchiveResourcePolicy.for_archive(descriptor)
        own_transport = transport is None
        transport = transport or ArchiveTransport(events)
        zip_path, _, _ = _staged_files(self.staging_root, descriptor)
        extraction = self.staging_root / (descriptor.library_id + EXTRACTION_SUFFIX + uuid4().hex[:8])
        destination = self.path(descriptor.library_id)
        try:
            self.staging_root.mkdir(parents=True, exist_ok=True)
            downloaded = self._download(descriptor, transport, events, report, policy)
            evidence = _confirmed_identity(descriptor, downloaded)
            extraction.mkdir(parents=True)
            try:
                summary = self._extract(descriptor, zip_path, extraction, events, report, policy)
                metadata = _installation_metadata(descriptor, evidence, summary)
                atomic_write_json(extraction / INSTALLATION_FILENAME, metadata)
                checkpoint(events)
                report("Publishing", 0, 1)
                self._publish(extraction, destination)
            except BaseException:
                shutil.rmtree(extraction, ignore_errors=True)
                raise
            # Publication succeeded, so the archive file that produced this library
            # is never retained beside it.
            for leftover in _staged_files(self.staging_root, descriptor):
                leftover.unlink(missing_ok=True)
            report("Publishing", 1, 1)
            return InstalledArchive(descriptor.library_id, destination, metadata)
        except (SearchCancelled, ArchiveResourceError):
            # The archive itself was never shown to be wrong, so a verified staged
            # download survives: the retry continues from it instead of fetching it
            # a second time.
            raise
        except BaseException:
            self._clear_staging(descriptor)
            raise
        finally:
            if own_transport:
                transport.close()

    # ------------------------------------------------------------------ download

    def _download(self, descriptor, transport, cancel, report, policy):
        zip_path, part, sidecar = _staged_files(self.staging_root, descriptor)
        total = descriptor.expected_size_bytes
        self._check_space(self.staging_root, policy.download_bytes, "download the distribution archive")
        reused = self._reuse_staged_archive(descriptor, zip_path, sidecar, cancel, report)
        if reused is not None:
            return reused
        restarts = attempts = 0
        while True:
            checkpoint(cancel)
            try:
                resume = self._resume_point(descriptor, part, sidecar, cancel)
                evidence = self._transfer(descriptor, part, sidecar, transport, cancel, report, resume)
            except SearchCancelled:
                raise
            except _CleanRestart:
                self._clear_download(descriptor)
                restarts += 1
                if restarts > CLEAN_RESTART_LIMIT:
                    raise OSError("The distribution archive could not be resumed safely; install it again.")
                continue
            except (requests.RequestException, Urllib3HTTPError, TimeoutError) as exc:
                # The verified prefix is retained, so a retry continues the transfer.
                attempts += 1
                if attempts >= TRANSFER_ATTEMPTS:
                    raise OSError("The distribution archive transfer failed repeatedly. The partial download "
                                  "was retained; install again to resume it.") from exc
                cancel.wait(RETRY_DELAY_SECONDS)
                continue
            except (ArchiveContractMismatch, InvalidArchive, ValueError, OSError):
                self._clear_download(descriptor)
                raise
            os.replace(part, zip_path)
            _write_resume(sidecar, descriptor, offset=total, sha256=evidence["sha256"],
                          validator=evidence["validator"], complete=True, observed_at=evidence["observed_at"])
            return evidence

    def _reuse_staged_archive(self, descriptor, zip_path, sidecar, cancel, report):
        """Reuse a completed download from an interrupted attempt, re-verifying its bytes."""
        record = _read_resume(sidecar, descriptor)
        if record is None or not record.complete:
            return None
        total = descriptor.expected_size_bytes
        if record.offset != total or not zip_path.is_file() or zip_path.stat().st_size != total:
            self._clear_download(descriptor)
            zip_path.unlink(missing_ok=True)
            return None
        report("Downloading", 0, total)
        digest = file_hash(zip_path, cancel)
        if digest != record.sha256:
            self._clear_download(descriptor)
            zip_path.unlink(missing_ok=True)
            return None
        report("Downloading", total, total)
        return {"sha256": digest, "size": total, "validator": record.validator,
                "observed_at": record.observed_at or now()}

    def _resume_point(self, descriptor, part, sidecar, cancel):
        """Return a verified resume point, or ``None`` when the transfer must start over."""
        record = _read_resume(sidecar, descriptor)
        if record is None or record.complete or record.offset <= 0:
            return None
        if not part.is_file():
            sidecar.unlink(missing_ok=True)
            return None
        actual = part.stat().st_size
        if actual < record.offset:
            sidecar.unlink(missing_ok=True)
            return None
        if actual > record.offset:
            with part.open("r+b") as stream:
                stream.truncate(record.offset)
        digest = _digest_prefix(part, record.offset, record.sha256, cancel)
        return _ResumePoint(record.offset, record.validator, digest)

    def _transfer(self, descriptor, part, sidecar, transport, cancel, report, resume):
        """Stream one attempt from the resume offset and return observed archive evidence."""
        total = descriptor.expected_size_bytes
        offset = resume.offset if resume is not None else 0
        digest = resume.digest if resume is not None else hashlib.sha256()
        validator = resume.validator if resume is not None else None
        observed_at = now()
        written = offset
        try:
            with transport.open(descriptor.url, start=offset) as response:
                status, headers = response.status_code, response.headers
                if status == 404:
                    raise FileNotFoundError(f"The distribution archive object is missing: {descriptor.url}")
                if status in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"The distribution archive returned HTTP {status}.", response=response)
                if offset and status != 206:
                    raise _CleanRestart("The server ignored the resume request; the download restarts "
                                        "from the beginning.")
                if not offset and status not in (200, 206):
                    raise OSError(f"The distribution archive returned HTTP {status}: {descriptor.url}")
                if str(headers.get("Content-Encoding", "identity")).lower() != "identity":
                    raise ValueError("Compressed HTTP transfer is unsupported for distribution archives.")
                declared = None
                if status == 206:
                    span = _content_range(headers.get("Content-Range"), offset)
                    if span is None:
                        raise _CleanRestart("The resume response carries no usable Content-Range; the "
                                            "download restarts from the beginning.")
                    declared = span[1]
                else:
                    length = str(headers.get("Content-Length") or "")
                    # Content-Length drives progress and the disk preflight; the
                    # received bytes themselves prove what was actually published.
                    declared = int(length) if length.isdigit() else None
                _verify_pinned_size(descriptor, declared)
                validator = _validator(headers)
                if resume is not None and not validator:
                    raise _CleanRestart("The object supplies no validator, so the partial download cannot "
                                        "be proven safe; it restarts from the beginning.")
                if resume is not None and resume.validator != validator:
                    raise _CleanRestart("The upstream validator changed during the interruption; the "
                                        "download restarts from the beginning.")
                with part.open("ab" if offset else "wb", buffering=0) as stream:
                    persisted = offset
                    report("Downloading", offset, total)
                    while True:
                        checkpoint(cancel)
                        chunk = response.raw.read1(DOWNLOAD_CHUNK_BYTES, decode_content=False)
                        checkpoint(cancel)
                        if not chunk:
                            break
                        if written + len(chunk) > total:
                            raise ArchiveContractMismatch(
                                f"{descriptor.library_id} sent more than its pinned {total} bytes.")
                        if stream.write(chunk) != len(chunk):
                            raise OSError("Archive staging write was incomplete.")
                        written += len(chunk)
                        digest.update(chunk)
                        report("Downloading", written, total)
                        if written - persisted >= SIDECAR_INTERVAL_BYTES:
                            _write_resume(sidecar, descriptor, offset=written, sha256=digest.hexdigest(),
                                          validator=validator, complete=False, observed_at=observed_at)
                            persisted = written
                    stream.flush()
                    os.fsync(stream.fileno())
                if written != total:
                    raise TimeoutError(f"The distribution archive transfer stopped at {written} of {total} "
                                       "bytes; the partial download was retained.")
        except Exception:
            if written > 0:
                # Keep exactly the bytes that reached disk so the next attempt can
                # resume from them, or prove that it must not.
                _write_resume(sidecar, descriptor, offset=written, sha256=digest.hexdigest(),
                              validator=validator, complete=False, observed_at=observed_at)
            raise
        return {"sha256": digest.hexdigest(), "size": written, "validator": validator,
                "observed_at": observed_at}

    # --------------------------------------------------------------- extraction

    def _extract(self, descriptor, zip_path, destination, cancel, report, policy):
        """Extract every safe member after a real-space preflight, verifying CRC."""
        try:
            with zipfile.ZipFile(zip_path) as reader:
                entries, expansion = _plan_members(reader, descriptor)
                self._check_space(self.staging_root, policy.extraction_bytes(expansion),
                                  "extract the distribution library")
                report("Extracting", 0, len(entries))
                for index, (info, member, directory) in enumerate(entries, 1):
                    checkpoint(cancel)
                    target = destination.joinpath(*PurePosixPath(member).parts)
                    if directory:
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            with reader.open(info) as source, target.open("wb", buffering=0) as sink:
                                while True:
                                    checkpoint(cancel)
                                    chunk = source.read(DOWNLOAD_CHUNK_BYTES)
                                    if not chunk:
                                        break
                                    if sink.write(chunk) != len(chunk):
                                        raise InvalidArchive(f"Extraction wrote an incomplete copy of {member!r}.")
                        except (zipfile.BadZipFile, zlib.error) as exc:
                            raise InvalidArchive(f"Archive member {member!r} is corrupt.") from exc
                    report("Extracting", index, len(entries))
        except zipfile.BadZipFile as exc:
            raise InvalidArchive("The downloaded archive has no readable ZIP central directory.") from exc
        checkpoint(cancel)
        return _verify_package(descriptor, entries, destination, cancel)

    # ------------------------------------------------------------------ publish

    def _publish(self, staging_dir, destination):
        """Move the verified library into place, replacing any working copy last.

        Windows cannot rename a directory onto an existing directory, so the working
        copy is moved aside first and the new library renamed into place. A failed
        rename restores the working copy immediately; a process death inside that
        window is healed by :meth:`recover`.
        """
        if destination.exists() and not destination.is_dir():
            raise ArchiveContractMismatch(
                f"{destination} is not a managed library directory; remove it to install here.")
        aside = None
        if destination.exists():
            aside = destination.with_name(destination.name + REPLACED_SUFFIX + uuid4().hex[:8])
            os.rename(destination, aside)
        try:
            os.rename(staging_dir, destination)
        except OSError:
            if aside is not None and not destination.exists():
                os.rename(aside, destination)
            raise
        if aside is not None:
            shutil.rmtree(aside, ignore_errors=True)

    # ------------------------------------------------------------------- removal

    def remove(self, library_id, *, confirmed=False):
        """Delete the selected managed distribution library after explicit confirmation."""
        if confirmed is not True:
            raise ValueError("Removing an installed distribution library requires explicit confirmation.")
        descriptor = distribution(library_id)
        path = self.path(library_id)
        if read_installation(path) is None:
            raise FileNotFoundError(f"No installed distribution library for {library_id} at {path}.")
        if path.is_symlink():
            raise ArchiveContractMismatch(f"{path} is a symbolic link and was not removed.")
        shutil.rmtree(path)
        self._clear_staging(descriptor)
        return True

    # ------------------------------------------------------------------- staging

    def _clear_staging(self, descriptor):
        for path in _staged_files(self.staging_root, descriptor):
            path.unlink(missing_ok=True)
        self._clear_extraction(descriptor)

    def _clear_download(self, descriptor):
        _, part, sidecar = _staged_files(self.staging_root, descriptor)
        part.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)

    def _clear_extraction(self, descriptor):
        if not self.staging_root.is_dir():
            return
        prefix = descriptor.library_id + EXTRACTION_SUFFIX
        for entry in self.staging_root.iterdir():
            if entry.name.startswith(prefix) and entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)

    def _check_space(self, directory, required, context):
        free = int(self._disk_usage(directory).free)
        if free < int(required):
            raise ArchiveResourceError(
                f"Insufficient disk space to {context}: {directory} has {free} free bytes and needs "
                f"{int(required)}. Free space or choose another NeuronBridge download location, then "
                "install again.")
