"""Project-owned offline queries. Completed evidence is never rewritten by previewing."""
from __future__ import annotations

import base64
import copy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
from contextlib import contextmanager
import shutil
import tempfile

import numpy as np
from PIL import Image

from .cdm import CDMResult, array_checksum, checkpoint, generate_cdm
from .cdm_export import export_cdm
from .cdm_records import (CDMArtifact, CDMGeometry, CDMMapping, CDMParameters,
                          CDMSelection, ImportedCDMImage, canonical_json)
from .search_profiles import search_profile

QUERY_KEY = "neuronbridge_queries"
ASSOCIATION_KEY = "neuronbridge_query_associations"
MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024


def generation_parameters(pixels, parameters, cancel=None):
    """Choose reproducible scalar conversion endpoints off the GUI thread."""
    if parameters.mode == "binary_mask" or parameters.display_range is not None or parameters.scalar_range is not None:
        return parameters
    if pixels.dtype == np.uint8:
        return parameters
    if pixels.dtype.kind not in "buif":
        raise ValueError("Choose a numeric scalar channel for Color-Depth MIP generation.")
    lo, hi = float("inf"), float("-inf")
    for plane in pixels:
        checkpoint(cancel)
        if not np.isfinite(plane).all():
            raise ValueError("Source contains non-finite values; resolve them before generating a Color-Depth MIP.")
        lo, hi = min(lo, float(plane.min())), max(hi, float(plane.max()))
    if lo == hi:
        lo, hi = (0., max(1., hi)) if lo >= 0 else (lo, 0.)
    if pixels.dtype.kind == "u" and pixels.dtype.itemsize == 2:
        return replace(parameters, display_range=(int(lo), int(hi)))
    return replace(parameters, scalar_range=(lo, hi))


@contextmanager
def disk_snapshot(launch, cancel=None):
    with disk_input_snapshots((launch.signal, launch.mask), cancel) as inputs:
        yield replace(launch, signal=inputs[0], mask=inputs[1])


@contextmanager
def disk_input_snapshots(inputs, cancel=None):
    """Own large input bytes on disk, with bounded RAM and guaranteed cleanup."""
    required = snapshot_bytes(inputs)
    if shutil.disk_usage(tempfile.gettempdir()).free < required + max(required // 10, 16 * 1024 * 1024):
        raise ValueError("Not enough temporary disk space to capture this Color-Depth MIP source safely.")
    buffers = {}
    with tempfile.TemporaryDirectory(prefix="madi3d-cdm-") as directory:
        try:
            def freeze(value):
                if value is None:
                    return None
                key = buffer_key(value.pixels)
                if key not in buffers:
                    path = Path(directory) / f"source-{len(buffers)}.npy"
                    target = np.lib.format.open_memmap(path, mode="w+", dtype=value.pixels.dtype.newbyteorder("="), shape=value.pixels.shape)
                    buffers[key] = target
                    for z in range(value.pixels.shape[0]):
                        checkpoint(cancel)
                        target[z] = value.pixels[z]
                    target.flush()
                    if array_checksum(target, cancel=cancel) != array_checksum(value.pixels, cancel=cancel):
                        raise ValueError("Source changed while capturing the Color-Depth MIP. Retry with a stable source.")
                    target._mmap.close()
                    buffers[key] = np.load(path, mmap_mode="r")
                return replace(value, pixels=buffers[key])
            captured = tuple(freeze(value) for value in inputs)
            checkpoint(cancel)
            yield captured
        finally:
            for array in buffers.values():
                array._mmap.close()


def buffer_key(pixels):
    """Identify the exact consumed view, including independently created aliases."""
    return (pixels.__array_interface__["data"][0], pixels.shape, pixels.strides, pixels.dtype.str)


def snapshot_bytes(inputs):
    buffers = {buffer_key(value.pixels): value.pixels.nbytes
               for value in inputs if value is not None and value.pixels is not None}
    return sum(buffers.values())


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QueryInput:
    selection: CDMSelection
    geometry: CDMGeometry
    pixels: np.ndarray | None
    # Runtime-only revision of the borrowed pixel owner. Saved evidence uses the
    # deterministic checksum, never process addresses or VTK modification times.
    content_revision: tuple | None = field(default=None, compare=False)
    scientific_revision: str | None = None

    def dependency(self, *, binary=False, cancel=None, pixel_checksum=None):
        result = {"selection": self.selection.to_dict(), "geometry": self.geometry.to_dict(),
                "pixels": pixel_checksum if pixel_checksum is not None else
                    None if self.pixels is None else array_checksum(self.pixels, binary=binary, cancel=cancel)}
        if self.scientific_revision is not None:
            result["scientific_revision"] = self.scientific_revision
        return result


@dataclass(frozen=True)
class QueryLaunch:
    signal: QueryInput
    mask: QueryInput | None
    mapping: CDMMapping
    parameters: CDMParameters

    @classmethod
    def capture(cls, signal, mask, mapping, parameters):
        if parameters.mode == "binary_mask":
            mask = signal
        inputs = [signal] + ([mask] if mask is not None else [])
        if any(i.pixels is None for i in inputs):
            raise ValueError("Load the selected signal and mask before generation.")
        if snapshot_bytes(inputs) > MAX_SNAPSHOT_BYTES:
            raise ValueError("Query snapshot exceeds 512 MiB. Extract the original signal region before generating.")
        if mask is not None:
            if (signal.pixels.shape != mask.pixels.shape or
                    signal.geometry.working_grid != mask.geometry.working_grid or
                    signal.geometry.pose != mask.geometry.pose):
                raise ValueError("Mask and signal must share the exact working grid and pose. Reformat them onto one grid first.")
        buffers = {}
        def freeze(value):
            # Own launch-time bytes: changes in the live volume cannot reach the worker.
            key = buffer_key(value.pixels)
            if key not in buffers:
                dtype = value.pixels.dtype.newbyteorder("=")
                buffers[key] = np.frombuffer(value.pixels.astype(dtype, copy=False).tobytes(), dtype=dtype).reshape(value.pixels.shape)
            pixels = buffers[key]
            return QueryInput(value.selection, value.geometry, pixels, value.content_revision, value.scientific_revision)
        return cls(freeze(signal), None if mask is None else freeze(mask), mapping, parameters)

    def dependencies(self, *, cancel=None, artifact=None):
        binary = self.parameters.mode == "binary_mask"
        values = [] if binary else [self.signal.dependency(cancel=cancel,
            pixel_checksum=artifact.source_pixel_sha256 if artifact else None)]
        if self.mask is not None:
            values.append(self.mask.dependency(binary=True, cancel=cancel,
                pixel_checksum=artifact.mask_pixel_sha256 if artifact else None))
        return values

    def generate(self, cancel=None):
        binary = self.parameters.mode == "binary_mask"
        return generate_cdm(None if binary else self.signal.pixels,
            selection=self.signal.selection, geometry=self.signal.geometry,
            mapping=self.mapping, parameters=self.parameters,
            mask=None if self.mask is None else self.mask.pixels,
            mask_selection=None if self.mask is None else self.mask.selection,
            mask_revision=None if self.mask is None else self.mask.selection.source_revision,
            cancel=cancel)


def complete_query(launch, destination, query_id, label, cancel=None):
    result = launch.generate(cancel)
    dependencies = launch.dependencies(cancel=cancel, artifact=result.artifact)
    checkpoint(cancel)
    artifact = export_cdm(result, destination, cancel=cancel, label=label)
    # Export's rename is the commit point. Retain committed evidence even if a
    # cancellation/supersession arrives immediately afterwards.
    from .cdm_export import cdm_image_name
    png = (Path(destination) / cdm_image_name(label)).read_bytes()
    record = {"version": 1, "query_id": query_id, "label": label,
              "artifact": artifact.to_dict(), "dependencies": dependencies,
              "export_directory": str(Path(destination).absolute()),
              "png_base64": base64.b64encode(png).decode("ascii"), "out_of_date": []}
    return result, json.loads(canonical_json(record))


def query_png(record):
    return base64.b64decode(record["png_base64"], validate=True)


def _read_query_file(path, limit, cancel):
    checkpoint(cancel)
    with Path(path).open("rb") as stream:
        content = stream.read(limit + 1)
    checkpoint(cancel)
    if len(content) > limit:
        raise ValueError(f"Color-Depth MIP file exceeds the {limit // (1024 * 1024)} MiB limit.")
    return content


def _require_rgb8(image, encoded):
    if image.format not in {"PNG", "TIFF", "JPEG"} or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
        raise ValueError("Choose a single 8-bit RGB Color-Depth MIP in PNG, TIFF, or JPEG format.")
    # Pillow can expose 16-bit RGB files as mode RGB after down-conversion.
    # Inspect the encoded precision before accepting the decoded color values.
    if image.format == "PNG" and (len(encoded) < 26 or encoded[24:26] != bytes((8, 2))):
        raise ValueError("Color-Depth MIP PNG must contain 8-bit RGB pixels; colors were not converted.")
    if image.format == "TIFF" and (tuple(image.tag_v2.get(258, ())) not in {(8,), (8, 8, 8)}
                                  or any(value != 1 for value in image.tag_v2.get(339, (1,)))):
        raise ValueError("Color-Depth MIP TIFF must contain unsigned 8-bit RGB pixels; colors were not converted.")


def import_query_image(path, query_id, cancel=None):
    """Capture an RGB image and optional matching export evidence off the GUI thread."""
    path = Path(path).absolute()
    original = _read_query_file(path, 8 * 1024 * 1024, cancel)
    with Image.open(io.BytesIO(original)) as image:
        _require_rgb8(image, original)
        if image.width * image.height > 16 * 1024 * 1024:
            raise ValueError("Color-Depth MIP exceeds the 16 megapixel import limit.")
        source_format = image.format
        pixels = np.asarray(image).copy()
        if pixels.dtype != np.uint8 or pixels.shape != (image.height, image.width, 3):
            raise ValueError("Choose an 8-bit RGB Color-Depth MIP.")
        if source_format == "PNG":
            png = original
        else:
            output = io.BytesIO()
            Image.fromarray(pixels).save(output, format="PNG")
            png = output.getvalue()
    checkpoint(cancel)
    if len(png) > 8 * 1024 * 1024:
        raise ValueError("The retained Color-Depth MIP PNG exceeds the 8 MiB limit.")
    png_sha = hashlib.sha256(png).hexdigest()
    manifest_json = None
    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_file():
        text = _read_query_file(manifest_path, 16 * 1024 * 1024, cancel).decode("utf-8")
        manifest = json.loads(text)
        if isinstance(manifest, dict):
            if "imported_query" in manifest and manifest.get("image") == path.name:
                record = manifest["imported_query"]
                if record.get("version") != 2:
                    raise ValueError("Unsupported imported Color-Depth MIP manifest.")
                validate_queries({QUERY_KEY: {record["query_id"]: record}})
                if query_png(record) != png:
                    raise ValueError("Color-Depth MIP manifest disagrees with the selected image.")
                checkpoint(cancel)
                return record
            artifact = manifest.get("artifact")
            if ((manifest.get("image") == path.name and ("artifact" in manifest or "context_sha256" in manifest))
                    or isinstance(artifact, dict) and artifact.get("png_file_sha256") == png_sha):
                manifest_json = text
    imported = ImportedCDMImage(str(path), source_format, hashlib.sha256(original).hexdigest(), len(original),
        datetime.now(timezone.utc).isoformat(), tuple(pixels.shape), png_sha,
        hashlib.sha256(pixels.tobytes()).hexdigest(), manifest_json)
    record = {"version": 2, "query_id": query_id, "label": path.stem,
              "imported_image": imported.to_dict(), "dependencies": [], "out_of_date": [],
              "export_directory": str(path.parent), "png_base64": base64.b64encode(png).decode("ascii")}
    validate_queries({QUERY_KEY: {query_id: record}})
    checkpoint(cancel)
    return json.loads(canonical_json(record))


def query_image_evidence(record):
    """Search consumes supplied image facts without inventing generation inputs."""
    if record["version"] == 2:
        image = ImportedCDMImage.from_dict(record["imported_image"])
        artifact = image.generation_artifact
        return {"png_file_sha256": image.png_file_sha256, "rgb_pixel_sha256": image.rgb_pixel_sha256,
                "warnings": list(image.warnings), "mapping": artifact.to_dict()["mapping"] if artifact else None}
    return record["artifact"]


def restore_query(record, destination, cancel=None):
    """Restore retained pixels through the same verified atomic export boundary."""
    checkpoint(cancel)
    if record["version"] == 2:
        validate_queries({QUERY_KEY: {record["query_id"]: record}})
        destination = Path(destination).absolute()
        if destination.exists():
            raise FileExistsError(destination)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.pending-", dir=destination.parent))
        try:
            from .cdm_export import cdm_image_name
            image_name = cdm_image_name(record["label"])
            files = {image_name: query_png(record),
                     "manifest.json": canonical_json({"image": image_name, "imported_query": record}).encode("utf-8")}
            for name, content in files.items():
                checkpoint(cancel)
                with (staging / name).open("xb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            checkpoint(cancel)
            if destination.exists():
                raise FileExistsError(destination)
            os.rename(staging, destination)
            return
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    with Image.open(io.BytesIO(query_png(record))) as image:
        pixels = np.asarray(image).copy()
    artifact = CDMArtifact.from_dict(record["artifact"])
    return export_cdm(CDMResult(pixels, artifact, None), destination, cancel=cancel, label=record["label"])


def validate_queries(metadata):
    queries = metadata.get(QUERY_KEY, {})
    for key, record in queries.items():
        if record["version"] not in (1, 2) or key != record["query_id"] or not record["label"]:
            raise ValueError("Invalid saved NeuronBridge query identity.")
        if record["version"] == 2:
            imported = ImportedCDMImage.from_dict(record["imported_image"])
            png = query_png(record)
            if len(png) > 8 * 1024 * 1024 or hashlib.sha256(png).hexdigest() != imported.png_file_sha256:
                raise ValueError("Saved imported query PNG checksum mismatch.")
            if imported.source_format == "PNG" and imported.source_size != len(png):
                raise ValueError("Saved imported PNG size disagrees with its source evidence.")
            with Image.open(io.BytesIO(png)) as image:
                _require_rgb8(image, png)
                if (image.format != "PNG" or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1
                        or (image.height, image.width, 3) != imported.shape_yx_rgb
                        or hashlib.sha256(image.tobytes()).hexdigest() != imported.rgb_pixel_sha256):
                    raise ValueError("Saved query pixels disagree with the imported image evidence.")
            if record["dependencies"] or record.get("artifact") is not None or record["out_of_date"]:
                raise ValueError("Imported images cannot invent live generation dependencies or geometry.")
            continue
        artifact = CDMArtifact.from_dict(record["artifact"])
        png = query_png(record)
        if len(png) > 8 * 1024 * 1024 or hashlib.sha256(png).hexdigest() != artifact.png_file_sha256:
            raise ValueError("Saved query PNG checksum mismatch.")
        with Image.open(io.BytesIO(png)) as image:
            if image.mode != "RGB" or image.size != search_profile(artifact.profile).search_canvas_yx[::-1]:
                raise ValueError("Invalid saved query canvas.")
            if hashlib.sha256(image.tobytes()).hexdigest() != artifact.rgb_pixel_sha256:
                raise ValueError("Saved query pixels disagree with generation evidence.")
        expected = 2 if artifact.parameters.mode == "signal_mask" else 1
        if len(record["dependencies"]) != expected:
            raise ValueError("Missing consumed query inputs.")
        for dependency in record["dependencies"]:
            CDMSelection.from_dict(dependency["selection"])
            CDMGeometry.from_dict(dependency["geometry"])
            revision = dependency.get("scientific_revision")
            if revision is not None and (not isinstance(revision, str) or len(revision) != 64 or
                                         any(c not in "0123456789abcdef" for c in revision)):
                raise ValueError("Invalid query scientific revision.")
        first = record["dependencies"][0]
        if (CDMSelection.from_dict(first["selection"]) != artifact.selection or
                CDMGeometry.from_dict(first["geometry"]) != artifact.geometry or
                first["pixels"] != (artifact.source_pixel_sha256 or artifact.mask_pixel_sha256)):
            raise ValueError("Query dependency disagrees with consumed generation evidence.")
        if artifact.mask_selection is not None:
            last = record["dependencies"][-1]
            if (CDMSelection.from_dict(last["selection"]) != artifact.mask_selection or
                    last["pixels"] != artifact.mask_pixel_sha256):
                raise ValueError("Query mask dependency disagrees with generation evidence.")
        validate_query_freshness(record)
    validate_query_associations(metadata)


def validate_query_freshness(record):
    if not isinstance(record["out_of_date"], list) or any(not isinstance(v, str) for v in record["out_of_date"]):
        raise ValueError("Invalid query freshness diagnostics.")


def validate_query_associations(metadata):
    queries = metadata.get(QUERY_KEY, {})
    for session, query in metadata.get(ASSOCIATION_KEY, {}).items():
        if session not in metadata.get("neuronbridge_sessions", {}) or query not in queries:
            raise ValueError("Query association references missing saved evidence.")


def add_query(metadata, record):
    result = copy.deepcopy(dict(metadata))
    registry = dict(result.get(QUERY_KEY, {}))
    result[QUERY_KEY] = registry
    key = record["query_id"]
    if key in registry and registry[key] != record:
        raise ValueError("A completed query cannot be overwritten.")
    registry[key] = copy.deepcopy(record)
    # Existing records are checked at the document boundary. Adding one query
    # must not decode all historical images before that boundary is reached.
    validate_queries({QUERY_KEY: {key: registry[key]}})
    validate_query_associations(result)
    return result


def associate_query(metadata, session_id, query_id):
    result = copy.deepcopy(dict(metadata))
    associations = result.setdefault(ASSOCIATION_KEY, {})
    if session_id not in result.get("neuronbridge_sessions", {}):
        raise ValueError("Select a saved search session.")
    if query_id is None:
        associations.pop(session_id, None)
    else:
        if query_id not in result.get(QUERY_KEY, {}):
            raise ValueError("Select a completed saved query.")
        associations[session_id] = query_id
    return result


def merge_query_metadata(metadata, incoming):
    """Merge separately imported projects/CSV carriers without guessing links."""
    result = copy.deepcopy(dict(metadata))
    for field in (QUERY_KEY, ASSOCIATION_KEY):
        values = incoming.get(field, {})
        if not values:
            continue
        target = dict(result.get(field, {}))
        result[field] = target
        for key, value in values.items():
            if key in target and canonical_json(target[key]) != canonical_json(value):
                raise ValueError(f"Conflicting saved NeuronBridge query evidence: {key}.")
            target[key] = copy.deepcopy(value)
    return result


def invalidate_queries(metadata, resolve, *, dependency_cache=None, resolve_mapping=None):
    """Compare consumed values, ignoring all rendering and camera properties.

    Unloaded inputs cannot be rechecked offline; retained evidence remains
    available. A resolver returns None only for an unloaded, still present input.
    Deletion or incompatible geometry is an error and marks the query stale.
    """
    result = dict(metadata)
    registry = dict(result.get(QUERY_KEY, {}))
    cache = dependency_cache if dependency_cache is not None else {}
    for key, record in registry.items():
        if record["out_of_date"] or record["version"] == 2:
            continue
        reasons = list(record["out_of_date"])
        mapping = CDMMapping(**record["artifact"]["mapping"])
        if (resolve_mapping is not None and mapping.evidence_json is not None and
                str(mapping.evidence_id or "").startswith("madi3d-geometry:")):
            try:
                channel_id = json.loads(mapping.evidence_json)["source_channel_id"]
                if resolve_mapping(channel_id).evidence_sha256 != mapping.evidence_sha256:
                    reasons.append("Consumed template mapping provenance changed.")
            except (ValueError, LookupError, RuntimeError) as exc:
                reasons.append(str(exc))
        for i, previous in enumerate(record["dependencies"]):
            selection = previous["selection"]
            binary = record["artifact"]["parameters"]["mode"] == "binary_mask" or i == 1
            cache_key = (digest(selection), binary)
            try:
                if cache_key not in cache:
                    current = resolve(selection)
                    cache[cache_key] = None if current is None else current.dependency(binary=binary)
                current = cache[cache_key]
                if current is not None and "scientific_revision" not in previous:
                    current = {k: v for k, v in current.items() if k != "scientific_revision"}
                if current is not None and current["pixels"] is None:
                    current = dict(current, pixels=previous["pixels"])
                if current is not None and canonical_json(current) != canonical_json(previous):
                    reasons.append(f"Consumed channel {selection['channel']['selector']}, frame {selection['frame_index']} changed.")
            except (ValueError, LookupError, RuntimeError) as exc:
                reasons.append(str(exc))
        reasons = list(dict.fromkeys(reasons))
        if reasons != record["out_of_date"]:
            registry[key] = dict(record, out_of_date=reasons)
    if registry:
        result[QUERY_KEY] = registry
    return result
