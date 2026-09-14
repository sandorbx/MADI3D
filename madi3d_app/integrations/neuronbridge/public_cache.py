"""Append-only public metadata snapshots, independent of geometry downloads."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from uuid import UUID

from .cache import neuronbridge_cache_root
from .public_api import MAX_BYTES, PublicAPIError, canonical, checkpoint
from .records import SearchResults


class PublicSnapshotCache:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else neuronbridge_cache_root() / "public_metadata"

    def _path(self, key, suffix):
        return self.root / (str(UUID(key)) + suffix)

    def save(self, value, kind, cancel=None):
        if kind == "results":
            parsed = SearchResults.from_dict(value)
            key = parsed.session.session_id
            label = f"{parsed.session.query_reference} — {parsed.session.neuronbridge_data_version}"
        elif kind == "lookup":
            key = value["snapshot_id"]
            label = f"{value['identifier']} — {value['data_version']}"
        else:
            raise ValueError("Unknown public snapshot kind.")
        body = canonical(value).encode("utf-8")
        if len(body) > 4 * MAX_BYTES:
            raise PublicAPIError("Snapshot exceeds the offline cache size limit.")
        checkpoint(cancel)
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self._path(key, ".json")
        manifest_path = self._path(key, ".info.json")
        if destination.exists() or manifest_path.exists():
            raise PublicAPIError("An existing snapshot cannot be overwritten.")
        manifest = {"key": key, "kind": kind, "label": label,
                    "sha256": hashlib.sha256(body).hexdigest()}
        published = []
        try:
            for path, data in ((destination, body), (manifest_path, canonical(manifest).encode("utf-8"))):
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=self.root, suffix=".part", delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                    checkpoint(cancel)
                    os.replace(temporary, path)
                    published.append(path)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            # Manifest publication is the commit point. Cancellation after it
            # leaves a valid historical snapshot available for offline opening.
            return key
        except Exception:
            for path in published:
                path.unlink(missing_ok=True)
            raise

    def entries(self):
        entries = []
        if not self.root.exists():
            return entries
        # Directory scan never reads result payloads. Present the most recent
        # 200 manifests; older snapshots remain addressable by their saved ID.
        import heapq
        paths = heapq.nlargest(200, self.root.glob("*.info.json"), key=lambda p: p.stat().st_mtime_ns)
        for path in paths:
            try:
                if path.stat().st_size > 64 * 1024:
                    continue
                item = json.loads(path.read_text(encoding="utf-8"))
                if self._path(item["key"], ".info.json") == path and item["kind"] in {"lookup", "results"}:
                    entries.append(item)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return entries

    def read(self, key):
        manifest_path = self._path(key, ".info.json")
        if manifest_path.stat().st_size > 64 * 1024:
            raise PublicAPIError("Invalid cache manifest.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        path = self._path(key, ".json")
        if path.stat().st_size > 4 * MAX_BYTES:
            raise PublicAPIError("Cached snapshot exceeds the size limit.")
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != manifest["sha256"]:
            raise PublicAPIError("Cached metadata checksum mismatch. Use another snapshot or refresh explicitly.")
        value = json.loads(body)
        if manifest["kind"] == "results":
            result = SearchResults.from_dict(value)
            if result.session.session_id != key:
                raise PublicAPIError("Cached result session identity mismatch.")
            return result
        if manifest["kind"] != "lookup" or value["snapshot_id"] != key:
            raise PublicAPIError("Invalid cached snapshot identity.")
        return value
