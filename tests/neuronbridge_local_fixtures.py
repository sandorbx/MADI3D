"""Deterministic synthetic installed-distribution fixtures; never fetch a collection.

Publishes packages in the exact ``library_store`` layout (package manifest,
inventory, pixel shards, published metadata) so tests exercise the real
read-only readers and the real ``archive_install`` discovery contract.
"""
from dataclasses import replace
from unittest import mock
import pytest

import base64
import copy
import hashlib
import io
import json
import shutil
from pathlib import Path
from urllib.parse import quote
import sqlite3

import numpy as np
import tifffile

from madi3d_app.integrations.neuronbridge.archive_catalog import (
    INVENTORY_FILENAME, PACKAGE_FORMAT, PACKAGE_MANIFEST_FILENAME, PACKAGE_VERSION,
    SHARD_DIRECTORY, SPARSE_RGB_U32_FORMAT, BulkArchive, distribution)
from madi3d_app.integrations.neuronbridge.archive_install import (
    INSTALLATION_FILENAME, INSTALLATION_SCHEMA_VERSION, official_library_root)
from madi3d_app.integrations.neuronbridge.cdm_export import export_brain_cdm
from madi3d_app.integrations.neuronbridge.local_catalog import IMAGE_BUCKET, canonical, now
from madi3d_app.integrations.neuronbridge.library_store import sparse_pixels
from madi3d_app.integrations.neuronbridge.public_api import BASE_URL
from madi3d_app.integrations.neuronbridge.search_profiles import search_profile
from generate_neuronbridge_local_oracle import generated_query

FIXTURES = Path(__file__).parent / "fixtures" / "neuronbridge_local"


_population_counts = {}


@pytest.fixture(scope="module", autouse=True)
def synthetic_populations():
    """Scope tiny descriptors to tests, keeping every managed identity owner aligned."""
    from madi3d_app.integrations.neuronbridge import archive_install, library_store
    original = distribution

    def selected(library_id):
        descriptor = original(library_id)
        count = _population_counts.get(library_id)
        if count is None:
            return descriptor
        changes = {"candidate_count": count}
        if isinstance(descriptor, BulkArchive):
            variants = min(descriptor.candidate_variant_count, count - 1)
            changes.update(candidate_original_count=count - variants,
                           candidate_variant_count=variants, distinct_body_count=count,
                           variant_marker=descriptor.variant_marker if variants else "")
        return replace(descriptor, **changes)

    with mock.patch.dict(_population_counts, clear=True), \
            mock.patch.object(archive_install, "distribution", selected), \
            mock.patch.object(library_store, "distribution", selected), \
            mock.patch(__name__ + ".distribution", selected):
        yield


def tiff_bytes(rgb):
    stream = io.BytesIO()
    tifffile.imwrite(stream, rgb, photometric="rgb", compression="lzw", metadata=None)
    return stream.getvalue()


def retained_query(path, result=None):
    result = result or generated_query()
    artifact = export_brain_cdm(result, path)
    record = {"version": 1, "query_id": "local-fixture-query", "label": "Exploratory brain fixture",
              "artifact": artifact.to_dict(), "dependencies": [{"selection": artifact.selection.to_dict(),
              "geometry": artifact.geometry.to_dict(), "pixels": artifact.source_pixel_sha256}],
              "export_directory": str(path), "png_base64": base64.b64encode((path / "cdm.png").read_bytes()).decode(),
              "out_of_date": []}
    return result.rgb, record


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _publish_sparse_package(directory, *, library_name, release, data_version, profile_id,
                            canvas_yx, members, members_per_shard=None):
    """Write one published sparse package: shards, inventory, metadata and manifest."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    shard_directory = directory / SHARD_DIRECTORY
    shard_directory.mkdir()
    metadata_directory = directory / "metadata"
    per_shard = members_per_shard or max(1, len(members))
    spans = {}  # ordinal -> (shard, offset, count)
    current, shard_id, offset = [], 0, 0

    def flush_shard():
        nonlocal current, shard_id, offset
        array = np.concatenate([pixels for _, pixels in current])
        np.save(shard_directory / f"{shard_id:08d}.npy", array, allow_pickle=False)
        shard_id += 1
        offset = 0
        current = []

    for ordinal, member in enumerate(members):
        pixels = sparse_pixels(member["rgb"])
        if current and (member.get("new_shard") or len(current) >= per_shard):
            flush_shard()
        spans[ordinal] = (shard_id, offset, len(pixels))
        current.append((ordinal, pixels))
        offset += len(pixels)
    if current:
        flush_shard()

    rows = []
    for ordinal, member in enumerate(members):
        metadata_evidence = dict(member.get("metadata_evidence_default") or {})
        metadata = member.get("metadata")
        if metadata is not None:
            payload = metadata["payload"] if isinstance(metadata["payload"], bytes) else \
                canonical(metadata["payload"]).encode("utf-8")
            metadata_directory.mkdir(exist_ok=True)
            target = metadata_directory / (hashlib.sha256(metadata["url"].encode()).hexdigest() + ".json")
            target.write_bytes(payload)
            metadata_evidence.update(url=metadata["url"], size=len(payload),
                                     sha256=hashlib.sha256(payload).hexdigest(),
                                     response_ref=hashlib.sha256(payload).hexdigest())
        image_evidence = member.get("image_evidence") or {
            "url": f"https://s3.amazonaws.com/{IMAGE_BUCKET}/{member['key']}",
            "size": len(member["rgb"].tobytes()),
            "sha256": hashlib.sha256(tiff_bytes(member["rgb"])).hexdigest(),
            "hash_authority": "synthetic_fixture"}
        shard, span_offset, count = spans[ordinal]
        rows.append((ordinal, member["key"], hashlib.sha256(member["key"].encode("utf-8")).hexdigest(),
                     json.dumps(image_evidence), hashlib.sha256(member["rgb"].tobytes()).hexdigest(),
                     int(count), json.dumps(metadata_evidence), json.dumps(member.get("source")),
                     json.dumps(list(member.get("diagnostics") or [])), shard, int(span_offset), int(count)))

    inventory = directory / INVENTORY_FILENAME
    connection = sqlite3.connect(inventory)
    try:
        connection.executescript("""
            CREATE TABLE members (
                ordinal INTEGER PRIMARY KEY, member_key TEXT NOT NULL, object_id TEXT NOT NULL,
                image_evidence TEXT, pixel_sha256 TEXT, nonzero_pixels INTEGER,
                metadata_evidence TEXT, source TEXT, diagnostics TEXT,
                shard INTEGER, pixel_offset INTEGER, pixel_count INTEGER);
            CREATE INDEX member_key_index ON members(member_key);
            CREATE INDEX pending_index ON members(shard, ordinal);
            CREATE TABLE metadata_cache (url TEXT PRIMARY KEY, evidence TEXT NOT NULL);
            CREATE TABLE shards (id INTEGER PRIMARY KEY, sha256 TEXT NOT NULL, size INTEGER NOT NULL);
            CREATE TABLE repair_observations (evidence TEXT NOT NULL);
        """)
        connection.executemany("INSERT INTO members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        for path in sorted(shard_directory.iterdir()):
            declared = int(path.stem, 10)
            connection.execute("INSERT INTO shards VALUES (?, ?, ?)",
                               (declared, _digest(path), path.stat().st_size))
        connection.commit()
    finally:
        connection.close()

    files = [{"path": path.relative_to(directory).as_posix(), "sha256": _digest(path),
              "size": path.stat().st_size}
             for path in sorted(directory.rglob("*"))
             if path.is_file() and path.name != PACKAGE_MANIFEST_FILENAME]
    payload = {"package_format": PACKAGE_FORMAT, "package_version": PACKAGE_VERSION,
               "index_format": SPARSE_RGB_U32_FORMAT, "library_name": library_name, "release": release,
               "data_version": data_version, "search_profile": profile_id,
               "alignment_space": search_profile(profile_id).alignment_space,
               "search_canvas_yx": list(canvas_yx), "candidate_count": len(members), "files": files,
               "source_snapshot_id": "synthetic-snapshot-" + hashlib.sha256(
                   str(len(members)).encode("utf-8")).hexdigest()[:16],
               "source_inventory_sha256": _digest(inventory)}
    (directory / PACKAGE_MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    return directory


def write_installation(directory, *, library_id, candidate_count=None):
    """Write a descriptor-consistent installation record for managed discovery."""
    if candidate_count is not None:
        _population_counts[library_id] = candidate_count
    descriptor = distribution(library_id)
    record = {"schema_version": INSTALLATION_SCHEMA_VERSION, "library_id": library_id,
              "provider": descriptor.provider, "source_url": descriptor.url,
              "source_release": descriptor.source_release, "representation": descriptor.representation,
              "search_profile": descriptor.profile_id,
              "archive_sha256": hashlib.sha256(("synthetic-installation:" + library_id).encode()).hexdigest(),
              "candidate_count": descriptor.candidate_count,
              "installed_at": now()}
    if descriptor.provider_sha256:
        record["provider_sha256"] = descriptor.provider_sha256
    (Path(directory) / INSTALLATION_FILENAME).write_text(json.dumps(record), encoding="utf-8")
    return record


def build_installed_sparse_library(base, *, library_id="flywire-fafb-v783", members,
                                   data_version="v3_10_0", members_per_shard=None,
                                   with_installation=True):
    """Publish a synthetic installed sparse library under the managed root."""
    members = list(members)
    descriptor = distribution(library_id)
    directory = _publish_sparse_package(
        official_library_root(base) / library_id, library_name=descriptor.geometry_source_library,
        release=descriptor.geometry_release, data_version=data_version, profile_id=descriptor.profile_id,
        canvas_yx=descriptor.profile.search_canvas_yx, members=members,
        members_per_shard=members_per_shard)
    if with_installation:
        write_installation(directory, library_id=library_id, candidate_count=len(members))
    return directory


def member_identity(descriptor, token, prototype):
    """Published metadata response plus matching source/evidence for one token."""
    image = copy.deepcopy(prototype)
    published = descriptor.geometry_published_name_prefix
    image.update(libraryName=descriptor.geometry_source_library, publishedName=f"{published}:{token}",
                 id=f"image-{token}", alignmentSpace=descriptor.alignment_space,
                 anatomicalArea=descriptor.profile.anatomical_area)
    base = f"{descriptor.alignment_space}/{descriptor.geometry_source_library}"
    cdm = f"https://s3.amazonaws.com/{IMAGE_BUCKET}/{base}/{token}-{descriptor.alignment_space}-CDM.png"
    swc = f"https://s3.amazonaws.com/{IMAGE_BUCKET}/{base}/SWC/{token}.swc"
    image["files"] = {"store": image.get("store") or "store", "CDM": cdm, "CDMThumbnail": cdm,
                      "AlignedBodySWC": swc}
    url = f"{BASE_URL}/v3_10_0/metadata/by_body/{quote(token, safe='')}.json"
    source = {"kind": "em", "provider": "NeuronBridge", "library": descriptor.geometry_source_library,
              "library_release": descriptor.geometry_release, "published_name": f"{published}:{token}",
              "biological": {"neuron_id": token},
              "image": {"image_id": f"image-{token}", "alignment_space": descriptor.alignment_space,
                        "anatomical_area": descriptor.profile.anatomical_area},
              "assets": [{"asset_type": "AlignedBodySWC", "url": swc}], "channel": None}
    evidence = {"resolved_image_index": 0, "lookup_token": token,
                "resolution": "exact_metadata_image_and_published_name"}
    return url, {"payload": {"results": [image]}}, source, evidence


def unknown_identity(descriptor, token, reason="Public metadata is missing for this synthetic member."):
    source = {"kind": "em", "provider": "NeuronBridge", "library": descriptor.geometry_source_library,
              "library_release": descriptor.geometry_release,
              "image": {"alignment_space": descriptor.alignment_space,
                        "anatomical_area": descriptor.profile.anatomical_area}}
    return source, [reason], {"status": "missing", "resolution": "unresolved"}


def library_fixture(root, count=3, *, library_id="flywire-fafb-v783", tokens=None, result=None,
                    candidate_pixels=None, members_per_shard=None, flipped_members=()):
    """Publish a deterministic installed sparse library plus a retained Brain query.

    Returns ``(directory, library_id, keys, query)``. Member 0 matches the
    query exactly, member 1 misses one pixel, member 2 is mirrored (repeating
    every three), matching the historical local-search fixture sequence.
    """
    rgb, query = retained_query(root / "saved query Ω", result)
    descriptor = distribution(library_id)
    tokens = tokens or [str(100000 + i) for i in range(count)]
    prototype = json.loads((FIXTURES / "hemibrain-metadata.json").read_text(encoding="utf-8"))["results"][0]
    members, keys = [], []
    for index in range(count):
        token = tokens[index]
        variant = "-FL" if index in flipped_members else ""
        key = (f"{descriptor.alignment_space}/{descriptor.geometry_source_library}"
               f"/searchable_neurons/{index}/{token}-{descriptor.alignment_space}-CDM{variant}.tif")
        keys.append(key)
        if candidate_pixels is not None:
            candidate = candidate_pixels[index % len(candidate_pixels)]
        elif index % 3 == 2:
            candidate = rgb[:, ::-1].copy()
        else:
            candidate = rgb.copy()
            if index % 3 == 1:
                y, x = np.nonzero(np.any(candidate != 0, axis=2))
                candidate[y[-1], x[-1]] = 0
        url, metadata, source, evidence = member_identity(descriptor, token, prototype)
        members.append({"key": key, "rgb": candidate, "metadata": {"url": url, **metadata},
                        "metadata_evidence_default": evidence, "source": source})
    directory = build_installed_sparse_library(root, library_id=library_id, members=members,
                                               members_per_shard=members_per_shard)
    return directory, library_id, keys, query
