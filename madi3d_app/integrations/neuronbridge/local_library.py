"""Persistent, resumable search libraries: SQLite inventory and lossless shards.

No GUI, executable installation or ambient authentication. A SQLite exclusive
lease outside each snapshot coordinates installers, searches and removal; OS
process death releases it automatically, including on Windows and macOS.
"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
from uuid import uuid4

import numpy as np

from madi3d_storage import atomic_write_json, data_dir
from .search_profiles import BRAIN, search_profile
from .local_catalog import (BASE_URL, IMAGE_BUCKET, BulkTransport, Library, ObjectMissing,
                            canonical, file_hash, manifest_members, member_url, now, parse_json)
from .local_scorer import SearchCancelled, checkpoint
from .public_api import file_url, source_identity
from .records import ImageIdentity, SourceIdentity

INDEX_FORMAT = "positive-cds-sparse-rgb-u32-v1"
PIXEL_DTYPE = np.dtype([("position", "<u4"), ("rgb", "u1", (3,))])
SHARD_BYTES = 16 * 1024 * 1024
MAX_OBJECT_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 128 * 1024 * 1024


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


def decoder_check():
    """Small real TIFF/PNG round trips, also invoked by the frozen smoke path."""
    import io
    from PIL import Image
    import tifffile
    rgb = np.array([[[19, 117, 250], [0, 0, 0]], [[2, 255, 40], [234, 52, 12]]], dtype=np.uint8)
    for compression in ("lzw", "deflate"):
        stream = io.BytesIO()
        tifffile.imwrite(stream, rgb, photometric="rgb", compression=compression)
        stream.seek(0)
        if not np.array_equal(tifffile.imread(stream), rgb):
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


def _close_map(array):
    if array is not None:
        array._mmap.close()


class LibraryBusy(RuntimeError):
    pass


class LibraryManager:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else data_dir() / "NeuronBridgeLibraries"
        self.root = self.root.expanduser().absolute()

    def path(self, snapshot_id):
        if not isinstance(snapshot_id, str) or not re.fullmatch(r"[0-9a-f]{32}", snapshot_id):
            raise ValueError("Invalid library snapshot identity.")
        path = self.root / "snapshots" / snapshot_id
        if path.is_symlink() or path.resolve().parent != (self.root / "snapshots").resolve():
            raise ValueError("Library path escapes the selected library root.")
        return path

    @contextmanager
    def lease(self, snapshot_id):
        self.path(snapshot_id)
        lock_dir = self.root / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock = sqlite3.connect(lock_dir / (snapshot_id + ".sqlite"), timeout=0)
        try:
            try:
                lock.execute("BEGIN EXCLUSIVE")
            except sqlite3.OperationalError as exc:
                raise LibraryBusy("This library is in use. Wait for its search or installation to stop.") from exc
            yield
        finally:
            lock.rollback()
            lock.close()

    def state(self, snapshot_id):
        value = json.loads((self.path(snapshot_id) / "snapshot.json").read_text(encoding="utf-8"))
        if value["snapshot_id"] != snapshot_id or value["index_format"] != INDEX_FORMAT:
            raise ValueError("Unsupported or inconsistent library snapshot.")
        Library(**value["library"])
        return value

    def entries(self):
        folder = self.root / "snapshots"
        if not folder.exists():
            return []
        result = []
        for path in sorted(folder.iterdir()):
            if path.is_dir():
                try:
                    state = self.state(path.name)
                    result.append(state)
                except (ValueError, OSError, KeyError, TypeError):
                    continue
        return result

    def create(self, library, *, snapshot_id=None):
        if not isinstance(library, Library):
            raise ValueError("Select a compatible public library.")
        snapshot_id = snapshot_id or uuid4().hex
        path = self.path(snapshot_id)
        path.mkdir(parents=True)
        for name in ("objects", "metadata", "index"):
            (path / name).mkdir()
        state = {"snapshot_id": snapshot_id, "library": library.to_dict(), "created_at": now(),
                 "status": "pending", "index_format": INDEX_FORMAT, "enumerated": False}
        with self.database(snapshot_id, create=True) as db:
            db.executescript("""
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
        atomic_write_json(path / "snapshot.json", state)
        return snapshot_id

    @contextmanager
    def database(self, snapshot_id, *, create=False):
        path = self.path(snapshot_id) / "inventory.sqlite"
        if not create and not path.is_file():
            raise FileNotFoundError("The exact snapshot inventory is missing at this library location.")
        db = sqlite3.connect(path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA cache_size=-2048")
        db.execute("PRAGMA temp_store=FILE")
        try:
            yield db
        finally:
            db.close()

    def remove(self, snapshot_id):
        with self.lease(snapshot_id):
            path = self.path(snapshot_id)
            self.state(snapshot_id)  # Only recognized, exact snapshots can be removed.
            shutil.rmtree(path)

    def install(self, snapshot_id, cancel=None, progress=None, *, transport=None, repair=False):
        report = progress or (lambda phase, done, total: None)
        with self.lease("0" * 32), self.lease(snapshot_id):
            state = self.state(snapshot_id)
            if state["status"] == "ready" and not repair:
                return state
            original_inventory = state.get("inventory_sha256")
            path = self.path(snapshot_id)
            library = Library(**state["library"])
            own_transport = transport is None
            transport = transport or BulkTransport(cancel)
            try:
                decoder_check()
                checkpoint(cancel)
                state["status"] = "installing"
                atomic_write_json(path / "snapshot.json", state)
                with self.database(snapshot_id) as db:
                    self._manifest(path, library, state, db, transport, cancel, report)
                    self._objects(path, library, db, transport, cancel, report)
                    self._indexes(path, db, cancel, report, search_profile(library.profile))
                    checkpoint(cancel)
                    inventory = self.inventory_digest(db, state, cancel)
                    if original_inventory and original_inventory != inventory:
                        raise ValueError("Repair changed snapshot identity. Select an explicit new snapshot instead.")
                    state.update(status="ready", inventory_sha256=inventory, completed_at=state.get("completed_at", now()),
                                 accounted_candidates=library.count, error=None)
                    checkpoint(cancel)
                    atomic_write_json(path / "snapshot.json", state)
                return state
            except Exception as exc:
                state.update(status="interrupted" if isinstance(exc, SearchCancelled) else "failed", error=str(exc))
                atomic_write_json(path / "snapshot.json", state)
                raise
            finally:
                if own_transport:
                    transport.close()

    def _space(self, path, required):
        if shutil.disk_usage(path).free < required + 64 * 1024 * 1024:
            raise OSError("Insufficient disk space. Free space or select another library location, then Resume.")

    def _manifest(self, path, library, state, db, transport, cancel, report):
        report("Checking library", 0, library.count)
        manifest = path / "membership.json"
        old = state.get("manifest")
        if not manifest.exists() or (old and file_hash(manifest, cancel) != old["sha256"]):
            staged = path / "membership.download"
            evidence = transport.download(library.manifest_url, staged, max_bytes=MAX_MANIFEST_BYTES)
            if old and evidence["sha256"] != old["sha256"]:
                staged.unlink(missing_ok=True)
                raise ValueError("Upstream membership changed; this snapshot cannot be repaired with new content.")
            os.replace(staged, manifest)
            if old is None:
                state["manifest"] = evidence
            else:
                db.execute("INSERT INTO repair_observations VALUES (?)", (canonical(evidence),))
                db.commit()
            atomic_write_json(path / "snapshot.json", state)
        elif old is None:
            # A crash before the manifest commit leaves an uncommitted object.
            manifest.unlink()
            return self._manifest(path, library, state, db, transport, cancel, report)
        if not state["enumerated"]:
            db.execute("DELETE FROM members")
            count = 0
            for count, key in enumerate(manifest_members(manifest, cancel), 1):
                member_url(library, key)
                object_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
                db.execute("INSERT INTO members(ordinal, member_key, object_id) VALUES (?, ?, ?)", (count - 1, key, object_id))
                if count % 512 == 0:
                    db.commit()
            db.commit()
            duplicate = db.execute("SELECT member_key FROM members GROUP BY member_key HAVING count(*)>1 LIMIT 1").fetchone()
            if duplicate:
                raise ValueError("Membership contains duplicate search members; all declarations were retained for inspection.")
            if count != library.count:
                raise ValueError(f"Incomplete library enumeration: catalog declares {library.count}, manifest contains {count}.")
            state["enumerated"] = True
            atomic_write_json(path / "snapshot.json", state)
        if db.execute("SELECT count(*) FROM members").fetchone()[0] != library.count:
            raise ValueError("Stored library membership is incomplete. Remove this incomplete snapshot and install again.")

    def _objects(self, path, library, db, transport, cancel, report):
        config = library.catalog["config"]["payload"]
        for cached in db.execute("SELECT url FROM metadata_cache"):
            checkpoint(cancel)
            self._metadata(path, cached["url"], db, transport, cancel)
        for row in db.execute("SELECT * FROM members ORDER BY ordinal"):
            checkpoint(cancel)
            target = path / "objects" / (row["object_id"] + ".tif")
            old = json.loads(row["image_evidence"]) if row["image_evidence"] else None
            report("Verifying" if old else "Downloading", row["ordinal"], library.count)
            valid = old and target.is_file() and target.stat().st_size == old["size"] and file_hash(target, cancel) == old["sha256"]
            if not valid:
                self._space(path, MAX_OBJECT_BYTES)
                staged = target.with_suffix(".download")
                evidence = transport.download(member_url(library, row["member_key"]), staged, max_bytes=MAX_OBJECT_BYTES)
                if old and evidence["sha256"] != old["sha256"]:
                    staged.unlink(missing_ok=True)
                    raise ValueError("Upstream search pixels changed. Repair cannot replace an earlier snapshot; install a new one.")
                report("Verifying", row["ordinal"], library.count)
                try:
                    rgb = decode_rgb(staged, search_profile(library.profile))
                    pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                    nonzero = int(np.count_nonzero(np.any(rgb != 0, axis=2)))
                    if row["pixel_sha256"] and pixel_hash != row["pixel_sha256"]:
                        raise ValueError("Decoded search pixels changed during repair.")
                    checkpoint(cancel)
                    os.replace(staged, target)
                finally:
                    staged.unlink(missing_ok=True)
                if old:
                    db.execute("INSERT INTO repair_observations VALUES (?)", (canonical(evidence),))
                db.execute("UPDATE members SET image_evidence=?,pixel_sha256=?,nonzero_pixels=? WHERE ordinal=?",
                           (canonical(old or evidence), pixel_hash, nonzero, row["ordinal"]))
                db.commit()
            if row["source"] is None:
                source, diagnostics, evidence = self._identity(path, library, row["member_key"], config, db, transport, cancel)
                db.execute("UPDATE members SET source=?,diagnostics=?,metadata_evidence=? WHERE ordinal=?",
                           (canonical(source.to_dict()), canonical(diagnostics), canonical(evidence), row["ordinal"]))
                db.commit()
        report("Verifying", library.count, library.count)

    def _identity(self, path, library, key, config, db, transport, cancel):
        unknown = SourceIdentity(kind="em", library=library.name, library_release=library.release,
                                 image=ImageIdentity(alignment_space=library.template_id, anatomical_area=library.region))
        # The pinned backend documents the leading filename token. It is only a
        # lookup selector until exact public metadata confirms the full identity.
        name = key.rsplit("/", 1)[-1]
        match = re.fullmatch(r"([^-]+)-" + re.escape(library.template_id) + r"-CDM(?:-[^.]+)?\.tif(?:f)?", name)
        if not match:
            return unknown, ["Library member has no supported metadata lookup token; biological identity remains unresolved."], {}
        token = match[1]
        from urllib.parse import quote
        url = f"{BASE_URL}/{library.data_version}/metadata/by_body/{quote(token, safe='')}.json"
        payload, evidence = self._metadata(path, url, db, transport, cancel)
        if evidence.get("status") == "missing":
            return unknown, ["Public metadata is missing; this exact search member remains visible but cannot load an arbitrary neuron."], evidence
        display_key = f"{library.template_id}/{library.name}/{token}-{library.template_id}-CDM.png"
        expected_name = library.published_name_prefix + ":" + token
        matches = [image for image in payload["results"] if isinstance(image, dict) and
                   image.get("type") == "EMImage" and image.get("libraryName") == library.name and
                   image.get("alignmentSpace") == library.template_id and image.get("anatomicalArea") == library.region and
                   image.get("publishedName") == expected_name and image.get("id") and
                   file_url(image, "CDM", config) == f"https://s3.amazonaws.com/{IMAGE_BUCKET}/{display_key}"]
        if len(matches) != 1 or set(payload) != {"results"}:
            return unknown, [f"Metadata resolution is ambiguous or incomplete ({len(matches)} exact images); retained member identity is authoritative."], evidence
        image = matches[0]
        source = source_identity(image, config)
        if source.biological.neuron_id not in (None, token):
            return unknown, ["Public body identity conflicts with the verified image name."], evidence
        source = replace(source, library_release=library.release,
                         biological=replace(source.biological, neuron_id=token))
        return source, [], dict(evidence, resolved_image=image, lookup_token=token,
                               resolution="exact_metadata_image_and_published_name")

    def _metadata(self, path, url, db, transport, cancel):
        cached = db.execute("SELECT evidence FROM metadata_cache WHERE url=?", (url,)).fetchone()
        md_path = path / "metadata" / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        evidence = json.loads(cached[0]) if cached else None
        if evidence is None or (evidence.get("status") != "missing" and
                (not md_path.exists() or file_hash(md_path, cancel) != evidence["sha256"])):
            staged = md_path.with_suffix(".download")
            try:
                observed = transport.download(url, staged, max_bytes=16 * 1024 * 1024)
                if evidence and evidence.get("sha256") != observed["sha256"]:
                    raise ValueError("Upstream identity metadata changed during snapshot repair.")
                parse_json(staged.read_bytes())
                os.replace(staged, md_path)
                if evidence:
                    db.execute("INSERT INTO repair_observations VALUES (?)", (canonical(observed),))
                evidence = evidence or observed
            except ObjectMissing:
                if evidence:
                    raise ValueError("Previously available snapshot identity metadata is now missing.")
                evidence = {"url": url, "retrieved_at": now(), "status": "missing"}
            finally:
                staged.unlink(missing_ok=True)
            db.execute("INSERT OR REPLACE INTO metadata_cache VALUES (?, ?)", (url, canonical(evidence)))
            db.commit()
        if evidence.get("status") == "missing":
            return None, evidence
        payload = parse_json(md_path.read_bytes())
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("Malformed public identity metadata.")
        return payload, evidence

    def _indexes(self, path, db, cancel, report, profile):
        total = db.execute("SELECT count(*) FROM members").fetchone()[0]
        for shard in db.execute("SELECT * FROM shards ORDER BY id"):
            checkpoint(cancel)
            target = path / "index" / f"{shard['id']:08d}.npy"
            if not target.exists() or target.stat().st_size != shard["size"] or file_hash(target, cancel) != shard["sha256"]:
                # Rebuild the same deterministic shard from verified original files.
                rows = db.execute("SELECT * FROM members WHERE shard=? ORDER BY ordinal", (shard["id"],)).fetchall()
                self._write_shard(path, db, shard["id"], rows, cancel, profile, expected=shard["sha256"])
        while True:
            checkpoint(cancel)
            rows, used = [], 0
            for row in db.execute("SELECT * FROM members WHERE shard IS NULL ORDER BY ordinal LIMIT 256"):
                size = row["nonzero_pixels"] * PIXEL_DTYPE.itemsize
                if rows and used + size > SHARD_BYTES:
                    break
                rows.append(row)
                used += size
            if not rows:
                break
            done = db.execute("SELECT count(*) FROM members WHERE shard IS NOT NULL").fetchone()[0]
            report("Indexing", done, total)
            shard_id = db.execute("SELECT coalesce(max(id), -1)+1 FROM shards").fetchone()[0]
            self._write_shard(path, db, shard_id, rows, cancel, profile)
        report("Indexing", total, total)

    def _write_shard(self, path, db, shard_id, rows, cancel, profile, expected=None):
        count = sum(row["nonzero_pixels"] for row in rows)
        # A zero-pixel shard has one unused sentinel (NumPy cannot mmap empty data).
        self._space(path, max(1, count) * PIXEL_DTYPE.itemsize + MAX_OBJECT_BYTES)
        target = path / "index" / f"{shard_id:08d}.npy"
        staged = target.with_suffix(".pending.npy")
        array = None
        assignments = []
        try:
            array = np.lib.format.open_memmap(staged, mode="w+", dtype=PIXEL_DTYPE, shape=(max(1, count),))
            if count == 0:
                array[:] = np.zeros(1, PIXEL_DTYPE)
            offset = 0
            for row in rows:
                checkpoint(cancel)
                rgb = decode_rgb(path / "objects" / (row["object_id"] + ".tif"), profile)
                if hashlib.sha256(rgb.tobytes()).hexdigest() != row["pixel_sha256"]:
                    raise ValueError("Index input differs from verified source pixels.")
                pixels = sparse_pixels(rgb)
                if len(pixels) != row["nonzero_pixels"]:
                    raise ValueError("Index pixel count disagrees with inventory.")
                array[offset:offset + len(pixels)] = pixels
                assignments.append((shard_id, offset, len(pixels), row["ordinal"]))
                offset += len(pixels)
            array.flush()
            _close_map(array)
            array = None
            with staged.open("r+b") as stream:
                os.fsync(stream.fileno())
            sha = file_hash(staged, cancel)
            if expected and sha != expected:
                raise ValueError("Rebuilt index does not reproduce the original snapshot.")
            checkpoint(cancel)
            os.replace(staged, target)
            with db:
                db.executemany("UPDATE members SET shard=?,pixel_offset=?,pixel_count=? WHERE ordinal=?", assignments)
                db.execute("INSERT OR REPLACE INTO shards VALUES (?, ?, ?)", (shard_id, sha, target.stat().st_size))
        finally:
            _close_map(array)
            staged.unlink(missing_ok=True)

    @staticmethod
    def inventory_digest(db, state, cancel=None):
        h = hashlib.sha256(canonical({"library": state["library"], "manifest": state["manifest"],
                                      "index_format": INDEX_FORMAT}).encode())
        expected_count = state["library"]["count"]
        count = 0
        for row in db.execute("SELECT * FROM members ORDER BY ordinal"):
            checkpoint(cancel)
            if any(row[key] is None for key in ("image_evidence", "pixel_sha256", "source", "metadata_evidence", "shard", "pixel_offset", "pixel_count")):
                raise ValueError("Library has incomplete objects, metadata or index shards.")
            if row["ordinal"] != count:
                raise ValueError("Library membership order is incomplete.")
            h.update(canonical(dict(row)).encode())
            count += 1
        if count != expected_count:
            raise ValueError("Library membership count does not match its declaration.")
        for row in db.execute("SELECT * FROM shards ORDER BY id"):
            h.update(canonical(dict(row)).encode())
        return h.hexdigest()

    @contextmanager
    def indexed_candidates(self, snapshot_id, cancel=None):
        """Yield a streaming iterator under a lease; all maps close on exit."""
        with self.lease(snapshot_id), self.database(snapshot_id) as db:
            state = self.state(snapshot_id)
            if state["status"] != "ready":
                raise ValueError("Library is incomplete. Resume or Repair before searching.")
            if self.inventory_digest(db, state, cancel) != state["inventory_sha256"]:
                raise ValueError("Library inventory changed. Repair is required.")
            mapped = None

            def candidates():
                nonlocal mapped
                active = None
                try:
                    for row in db.execute("SELECT * FROM members ORDER BY ordinal"):
                        checkpoint(cancel)
                        if row["shard"] != active:
                            _close_map(mapped)
                            mapped = None
                            active = row["shard"]
                            evidence = db.execute("SELECT * FROM shards WHERE id=?", (active,)).fetchone()
                            target = self.path(snapshot_id) / "index" / f"{active:08d}.npy"
                            if not evidence or target.stat().st_size != evidence["size"] or file_hash(target, cancel) != evidence["sha256"]:
                                raise ValueError("Search index is corrupt. Repair this exact snapshot.")
                            mapped = np.load(target, mmap_mode="r", allow_pickle=False)
                            if mapped.dtype != PIXEL_DTYPE or mapped.ndim != 1 or mapped.nbytes > SHARD_BYTES:
                                raise ValueError("Invalid or oversized search shard.")
                        start, count = row["pixel_offset"], row["pixel_count"]
                        if start < 0 or count < 0 or start + count > len(mapped):
                            raise ValueError("Invalid candidate index span.")
                        yield dict(row), mapped[start:start + count]
                finally:
                    _close_map(mapped)
                    mapped = None
            iterator = candidates()
            try:
                yield state, iterator
            finally:
                iterator.close()
