"""Pinned CloudVolume decoding at release-qualified public locators, worker only."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import tempfile

from madi3d_app.io_utils import check_cancelled

from .geometry import GeometryLimits, GeometryUnavailable, _limits_check, normalize_swc, write_mesh, write_skeleton
from .records import FlyWireRemoteDescriptor


BACKEND_VERSION = "12.15.2"
_RADIUS = [{"id": "radius", "data_type": "float32", "num_components": 1}]
_INFO_LIMIT = 1024**2


def read_response(session, url, limit=None, *, byte_range=None, destination=None, cancel_check=None):
    """Read checked bytes, or stream to a file and return its raw byte evidence."""
    check_cancelled(cancel_check)
    headers = {"Accept-Encoding": "identity"}
    if byte_range is not None:
        start, end = byte_range
        expected = end - start
        if start < 0 or end <= start or (limit is not None and expected > limit):
            raise ValueError("Mesh range exceeds transfer limits.")
        headers["Range"] = f"bytes={start}-{end - 1}"
        limit = expected
    with session.get(url, headers=headers, timeout=(10, 30), stream=True) as response:
        if response.status_code == 404:
            raise GeometryUnavailable("The selected geometry is unavailable.")
        response.raise_for_status()
        if byte_range is not None:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(?:\d+|\*)", response.headers.get("Content-Range", ""))
            if response.status_code != 206 or match is None or tuple(map(int, match.groups())) != (start, end - 1):
                raise ValueError("Server did not honor the requested mesh byte range.")
        length = response.headers.get("Content-Length")
        if length is not None and limit is not None and int(length) > limit:
            raise ValueError("Response exceeds the transfer limit.")
        result = bytearray() if destination is None else None
        digest = hashlib.sha256() if destination is not None else None
        received = 0
        for chunk in response.iter_content(64 * 1024):
            check_cancelled(cancel_check)
            if limit is not None and received + len(chunk) > limit:
                raise ValueError("Response exceeds the transfer limit.")
            received += len(chunk)
            if destination is None:
                result.extend(chunk)
            else:
                destination.write(chunk)
                digest.update(chunk)
        if (length is not None and received != int(length)) or (byte_range is not None and received != limit):
            raise ValueError("Truncated geometry response.")
        check_cancelled(cancel_check)
        if destination is not None:
            return {"payload_bytes": received, "source_checksum_sha256": digest.hexdigest()}
        return bytes(result)


class _ShardRanges:
    """CloudVolume ShardReader's range-cache protocol, without an opaque HTTP read.

    Shard addressing, hashing and index decoding remain CloudVolume's. This
    concrete adapter only supplies streamed, size-checked anonymous HTTPS bytes.
    """

    def __init__(self, session, limit, profile):
        self.session = session
        self.limit = limit
        self.transferred = 0
        self.directory = profile.mesh_directory
        self.volume = profile.geometry_source.rsplit("/", 1)[0]

    def download_as(self, requests, progress=None):
        result = {}
        for request in requests:
            path, start, end = request["path"], int(request["start"]), int(request["end"])
            if not re.fullmatch(re.escape(self.directory) + r"/[0-9a-f]+\.shard", path):
                raise ValueError("Unexpected fixed-release shard locator.")
            remaining = self.limit - self.transferred if self.limit is not None else None
            payload = read_response(self.session, self.volume + "/" + path,
                                    remaining, byte_range=(start, end))
            self.transferred += len(payload)
            result[(path, start, end)] = payload
        return result


def retrieve(descriptor, path, limits, *, cancel_check=None):
    """Produce one normalized artifact and bounded source evidence, never scene state."""
    profile = descriptor.context.profile
    from importlib.metadata import version
    import numpy as np
    import requests
    from cloudvolume import Skeleton

    if version("cloud-volume") != BACKEND_VERSION:
        raise RuntimeError(f"FlyWire retrieval requires cloud-volume=={BACKEND_VERSION}.")
    with requests.Session() as session:
        # Public source access has no account/credential lookup or authentication.
        session.trust_env = False
        if descriptor.representation == "swc":
            if profile.skeleton_format == "swc":
                locator = profile.skeleton_source + "/" + descriptor.root_id + ".swc"
                # Raw bytes live only in worker-owned scratch, never in the cache.
                with tempfile.TemporaryFile(dir=Path(path).parent) as source:
                    evidence = read_response(session, locator, limits.max_payload_bytes,
                                             destination=source, cancel_check=cancel_check)
                    source.seek(0)
                    summary = normalize_swc(path, source, profile.skeleton_unit_nm, limits,
                                            cancel_check=cancel_check)
                return {"source_locator": locator, "source_format": "swc",
                        "source_unit_nm": profile.skeleton_unit_nm, **evidence, **summary}
            info = json.loads(read_response(session, profile.skeleton_source + "/info", _INFO_LIMIT))
            declaration = info.get("vertex_attributes")
            attributes = [declaration] if isinstance(declaration, dict) else declaration
            if attributes != _RADIUS:
                raise GeometryUnavailable("The skeleton source has no supported measured-radius declaration.")
            payload = read_response(session, profile.skeleton_source + "/" + descriptor.root_id, limits.max_payload_bytes)
            if len(payload) < 8:
                raise ValueError("Truncated skeleton header.")
            vertices, edges = struct.unpack_from("<II", payload)
            _limits_check(vertices, edges, limits)
            if len(payload) != 8 + vertices * 16 + edges * 8:
                if len(payload) >= 8 + vertices * 12 + edges * 8:
                    raise GeometryUnavailable("Missing or truncated measured-radius data.")
                raise ValueError("Truncated or unexpected skeleton attribute bytes.")
            skeleton = Skeleton.from_precomputed(payload, segid=int(descriptor.root_id), vertex_attributes=attributes)
            summary = write_skeleton(path, skeleton.vertices, skeleton.edges, skeleton.radius,
                                     np.asarray(info["transform"]).reshape(3, 4), limits)
            return {"source_locator": profile.skeleton_source + "/" + descriptor.root_id,
                    "source_info": info, "payload_bytes": len(payload), **summary}

        from cloudvolume.datasource.precomputed.sharding import ShardingSpecification, ShardReader
        from cloudvolume.datasource.precomputed.mesh.multilod import MultiLevelPrecomputedMeshManifest, extract_lod_meshes
        from cloudvolume import Mesh

        info = json.loads(read_response(session, profile.geometry_source + "/info", _INFO_LIMIT))
        if info.get("@type") != profile.mesh_format:
            raise ValueError("Unexpected mesh format for the selected release.")
        spec = ShardingSpecification.from_dict(info["sharding"])
        if profile.mesh_sharding is not None:
            if (any(info["sharding"].get(key) != value for key, value in profile.mesh_sharding)
                    or tuple(info["transform"]) != profile.mesh_transform
                    or info["vertex_quantization_bits"] != profile.mesh_vertex_quantization_bits):
                raise ValueError("Mesh declaration conflicts with the qualified source product.")
        elif spec.data_encoding != "raw":
            raise ValueError("Unexpected mesh manifest compression.")
        volume = profile.geometry_source.rsplit("/", 1)[0]
        ranges = _ShardRanges(session, limits.max_payload_bytes, profile)
        reader = ShardReader(volume, ranges, spec, shard_index_cache_size=1, minishard_index_cache_size=1)
        location = reader.exists(int(descriptor.root_id), profile.mesh_directory, return_byte_range=True)
        if location is None:
            raise GeometryUnavailable("The selected mesh is unavailable.")
        shard, offset, size = location
        binary = next(iter(ranges.download_as([{"path": shard, "start": offset, "end": offset + size}]).values()))
        if spec.data_encoding == "gzip":
            # Only the manifest is gzip encoded. Draco fragments precede it in
            # the shard and their offsets remain relative to its encoded start.
            with gzip.GzipFile(fileobj=io.BytesIO(binary)) as compressed:
                binary = (compressed.read(limits.max_payload_bytes + 1)
                          if limits.max_payload_bytes is not None else compressed.read())
            if (limits.max_payload_bytes is not None
                    and len(binary) > limits.max_payload_bytes):
                raise ValueError("Decoded mesh manifest exceeds the payload limit.")
        manifest = MultiLevelPrecomputedMeshManifest.from_binary(binary, int(descriptor.root_id), shard_offset=offset, path=shard)
        lod = descriptor.mesh_lod
        sizes = [sum(map(int, group)) for group in manifest.fragment_offsets]
        available = [level for level, size in enumerate(sizes) if size > 0
                     and (profile.mesh_lods is None or level in profile.mesh_lods)]
        if not available:
            raise GeometryUnavailable("This neuron has no available mesh LOD.")
        # Prefer the requested level, then the closest finer level, then coarser.
        finer = [level for level in available if level <= lod]
        lod = max(finer) if finer else min(available)
        start = int(offset) - sum(sizes) + sum(sizes[:lod])
        payload = next(iter(ranges.download_as([{"path": shard, "start": start, "end": start + sizes[lod]}]).values()))
        transform = np.asarray(info["transform"] + [0, 0, 0, 1], dtype=float).reshape(4, 4)
        if not np.isfinite(transform).all() or np.linalg.det(transform[:3, :3]) == 0:
            raise ValueError("Invalid mesh model transform.")
        fragments = extract_lod_meshes(manifest, lod, payload, info["vertex_quantization_bits"], transform)[int(descriptor.root_id)]
        _limits_check(sum(len(m.vertices) for m in fragments),
                      sum(len(m.faces) for m in fragments), limits)
        mesh = Mesh.concatenate(*fragments)
        summary = write_mesh(path, mesh.vertices, mesh.faces, limits)
        return {"source_locator": volume, "source_info": info, "mesh_lod": lod,
                "requested_mesh_lod": descriptor.mesh_lod,
                "payload_bytes": ranges.transferred, **summary}


def main(request_path):
    """Private executable dispatch; parent supervision owns cancellation/cleanup."""
    request_path = Path(request_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result_path = request_path.with_name("result.json")
    try:
        descriptor = FlyWireRemoteDescriptor.from_dict(request["descriptor"])
        evidence = retrieve(descriptor, request_path.with_name("geometry." + ("vtp" if descriptor.representation == "mesh" else "swc")),
                            GeometryLimits(**request["limits"]))
        from datetime import datetime, timezone
        evidence["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
        result = {"status": "ok", "evidence": evidence}
        code = 0
    except Exception as exc:
        result = {"status": "unavailable" if isinstance(exc, GeometryUnavailable) else "failed",
                  "error_type": type(exc).__name__}
        if isinstance(exc, (ValueError, RuntimeError, ImportError)):
            result["reason"] = str(exc)[:500]
        elif isinstance(exc, MemoryError):
            result["reason"] = "Geometry decoding could not allocate memory within its worker budget."
        elif getattr(exc, "response", None) is not None:
            result["reason"] = f"Public geometry source returned HTTP {exc.response.status_code}."
        else:
            result["reason"] = "Geometry retrieval or decoding failed. Retry the selected representation."
        code = 1
    result_path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    return code


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1]))
