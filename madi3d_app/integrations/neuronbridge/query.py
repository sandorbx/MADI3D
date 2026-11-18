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
from PIL import Image, __version__ as pillow_version

from .cdm import CDMResult, array_checksum, checkpoint, generate_cdm
from .cdm_export import CDM_IMAGE_NAME, export_cdm, stage_cdm_bundle
from .cdm_records import (CDMArtifact, CDMGeometry, CDMMapping, CDMParameters,
                          CDMSelection, ImportedCDMImage, canonical_json)
from .search_profiles import search_profile
from .local_scorer import checkpoint as search_checkpoint

QUERY_KEY = "neuronbridge_queries"
ASSOCIATION_KEY = "neuronbridge_query_associations"
SUPPLEMENTARY_KEY = "neuronbridge_supplementary_inputs"
MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024
MAX_SUPPLEMENTARY_PNG_BYTES = 8 * 1024 * 1024


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
        captured = replace(launch, signal=inputs[0], mask=inputs[1])
        launch = None
        yield captured


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
            inputs = ()
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
    scientific_capture: object | None = field(default=None, compare=False, repr=False)

    def dependency(self, *, binary=False, cancel=None, pixel_checksum=None):
        result = {"selection": self.selection.to_dict(), "geometry": self.geometry.to_dict(),
                "pixels": pixel_checksum if pixel_checksum is not None else
                    None if self.pixels is None else array_checksum(self.pixels, binary=binary, cancel=cancel)}
        if self.scientific_revision is not None:
            result["scientific_revision"] = self.scientific_revision
        return result


def validate_query_inputs(signal, mask):
    """Require matching spatial inputs while preserving their evidence revisions."""
    if signal.pixels is None or (mask is not None and mask.pixels is None):
        raise ValueError("Load the selected signal and mask before generation.")
    if mask is not None and (
        signal.pixels.shape != mask.pixels.shape
        or signal.geometry.working_grid != mask.geometry.working_grid
        or signal.geometry.pose != mask.geometry.pose
    ):
        raise ValueError("Mask and signal must share the exact working grid and pose. Reformat them onto one grid first.")


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
        validate_query_inputs(signal, mask)
        if snapshot_bytes((signal, mask)) > MAX_SNAPSHOT_BYTES:
            raise ValueError("Query snapshot exceeds 512 MiB. Extract the original signal region before generating.")
        buffers = {}
        def freeze(value):
            # Own launch-time bytes: changes in the live volume cannot reach the worker.
            key = buffer_key(value.pixels)
            if key not in buffers:
                dtype = value.pixels.dtype.newbyteorder("=")
                buffers[key] = np.frombuffer(value.pixels.astype(dtype, copy=False).tobytes(), dtype=dtype).reshape(value.pixels.shape)
            pixels = buffers[key]
            return QueryInput(
                value.selection,
                value.geometry,
                pixels,
                value.content_revision,
                value.scientific_revision,
                value.scientific_capture,
            )
        return cls(freeze(signal), None if mask is None else freeze(mask), mapping, parameters)

    def generate(self, cancel=None):
        binary = self.parameters.mode == "binary_mask"
        validate_query_inputs(self.signal, self.mask)
        result = generate_cdm(None if binary else self.signal.pixels,
            selection=self.signal.selection, geometry=self.signal.geometry,
            mapping=self.mapping, parameters=self.parameters,
            mask=None if self.mask is None else self.mask.pixels,
            mask_selection=None if self.mask is None else self.mask.selection,
            mask_revision=None if self.mask is None else self.mask.selection.source_revision,
            cancel=cancel)
        artifact = replace(result.artifact,
            source_scientific_revision=None if binary else self.signal.scientific_revision,
            mask_scientific_revision=None if self.mask is None else self.mask.scientific_revision,
            mask_geometry_revision=None if self.mask is None else self.mask.geometry.revision)
        return replace(result, artifact=artifact)


class CommittedQueryExportError(RuntimeError):
    """The bundle was published, but its project record could not be prepared."""


def complete_query(launch, destination, query_id, label, cancel=None):
    result = launch.generate(cancel)
    checkpoint(cancel)
    if destination is None:
        # A searchable query needs the same verified RGB and PNG evidence as an
        # exported bundle, but does not need an intermediate filesystem copy.
        if (result.rgb.dtype != np.uint8 or
                result.rgb.shape != (*search_profile(result.artifact.profile).search_canvas_yx, 3) or
                hashlib.sha256(result.rgb.tobytes()).hexdigest() != result.artifact.rgb_pixel_sha256):
            raise ValueError("Generated Color-Depth MIP pixels disagree with the artifact.")
        stream = io.BytesIO()
        Image.fromarray(result.rgb).save(stream, format="PNG", compress_level=6)
        png = stream.getvalue()
        checkpoint(cancel)
        artifact = replace(result.artifact, png_file_sha256=hashlib.sha256(png).hexdigest(),
                           png_encoder=f"Pillow/{pillow_version}; PNG RGB8; compress_level=6")
    else:
        artifact = export_cdm(result, destination, cancel=cancel, label=label, query_id=query_id)
    try:
        if destination is not None:
            # The rename committed the bundle. Read its published filename even
            # if cancellation or supersession arrived immediately afterwards.
            directory = Path(destination)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            image = manifest["image"]
            if Path(image).name != image or image in ("", ".", ".."):
                raise ValueError("Invalid image filename in the exported manifest.")
            png = (directory / image).read_bytes()
            if hashlib.sha256(png).hexdigest() != artifact.png_file_sha256:
                raise ValueError("Exported PNG disagrees with its checksum.")
        record = {"version": 3, "query_id": query_id, "label": label,
                  "artifact": artifact.to_dict(),
                  "png_base64": base64.b64encode(png).decode("ascii"), "out_of_date": []}
        return result, json.loads(canonical_json(record))
    except Exception as exc:
        if destination is None:
            raise
        raise CommittedQueryExportError(
            f"Color-Depth MIP export succeeded at {destination}, but its project record "
            f"could not be prepared: {exc}. Open the exported image to add it to the project."
        ) from exc


def query_png(record):
    return base64.b64decode(record["png_base64"], validate=True)



MAX_QUERY_PNG_BYTES = 8 * 1024 * 1024
MAX_QUERY_PIXELS = 16 * 1024 * 1024
ROI_QUERY_VERSION = 4
ROI_METHOD = "xy-binary-clear-outside-v1"
_ROI_SUBPIXELS = 256


def query_rgb(record):
    """Decode a retained query without conversion or resampling."""
    encoded = record["png_base64"]
    if len(encoded) > ((MAX_QUERY_PNG_BYTES + 2) // 3) * 4:
        raise ValueError("The retained Color-Depth MIP PNG exceeds the 8 MiB limit.")
    png = query_png(record)
    if len(png) > MAX_QUERY_PNG_BYTES:
        raise ValueError("The retained Color-Depth MIP PNG exceeds the 8 MiB limit.")
    try:
        with Image.open(io.BytesIO(png)) as image:
            _require_rgb8(image, png)
            if image.format != "PNG" or image.width * image.height > MAX_QUERY_PIXELS:
                raise ValueError("Invalid retained RGB8 Color-Depth MIP canvas.")
            return np.array(image)
    except OSError as exc:
        raise ValueError("The retained Color-Depth MIP could not be decoded.") from exc


def polygon_mask(points, shape):
    """Even-odd fill at pixel centers, with top/left inclusive edges."""
    height, width = shape
    half = _ROI_SUBPIXELS // 2
    vertices = [(round(x * _ROI_SUBPIXELS), round(y * _ROI_SUBPIXELS)) for x, y in points]
    crossings = [[] for _ in range(height)]
    for (x0, y0), (x1, y1) in zip(vertices, vertices[1:] + vertices[:1]):
        if y0 == y1:
            continue
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        dy = y1 - y0
        divisor = dy * _ROI_SUBPIXELS
        first_row = max(0, -((half - y0) // _ROI_SUBPIXELS))
        last_row = min(height, -((half - y1) // _ROI_SUBPIXELS))
        for row in range(first_row, last_row):
            numerator = (x0 - half) * dy + (row * _ROI_SUBPIXELS + half - y0) * (x1 - x0)
            crossings[row].append(-(-numerator // divisor))
    mask = np.zeros(shape, dtype=bool)
    for row, columns in enumerate(crossings):
        columns.sort()
        for left, right in zip(columns[::2], columns[1::2]):
            mask[row, max(0, min(width, left)):max(0, min(width, right))] = True
    return mask


def mask_query_rgb(rgb, mask):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("The source must be an RGB8 Color-Depth MIP.")
    if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2]:
        raise ValueError("The search area must use the original image canvas.")
    result = np.zeros_like(rgb)
    np.copyto(result, rgb, where=mask[..., None])
    return result


def query_has_template_placement(record):
    """Whether the image canvas was placed by a retained generation mapping."""
    if record["version"] == ROI_QUERY_VERSION:
        return record["derivation"]["template_placement"]
    if record["version"] == 2:
        return ImportedCDMImage.from_dict(record["imported_image"]).generation_artifact is not None
    return True


def derive_masked_query(parent, mask, query_id, *, rgb=None, cancel=None):
    """Capture a new image and its exact row-major binary selection."""
    from .search_profiles import query_profile
    checkpoint(cancel)
    rgb = query_rgb(parent) if rgb is None else rgb
    evidence = query_image_evidence(parent)
    if hashlib.sha256(rgb.tobytes()).hexdigest() != evidence["rgb_pixel_sha256"]:
        raise ValueError("The search area belongs to a different source image.")
    if rgb.shape[0] * rgb.shape[1] > MAX_QUERY_PIXELS:
        raise ValueError("Color-Depth MIP exceeds the 16 megapixel import limit.")
    masked = mask_query_rgb(rgb, mask)
    if not masked.any():
        raise ValueError("The selection contains no signal. Draw around a brighter area or clear the mask.")
    output = io.BytesIO()
    Image.fromarray(masked).save(output, format="PNG", compress_level=6)
    png = output.getvalue()
    if len(png) > MAX_QUERY_PNG_BYTES:
        raise ValueError("The retained Color-Depth MIP PNG exceeds the 8 MiB limit.")
    checkpoint(cancel)
    profile = query_profile(parent)
    bits = np.packbits(mask, bitorder="big").tobytes()
    return {
        "version": ROI_QUERY_VERSION, "query_id": query_id,
        "label": f"{parent['label']} · Masked CDM",
        "png_base64": base64.b64encode(png).decode("ascii"), "out_of_date": [],
        "derivation": {
            "method": ROI_METHOD, "parent_query_id": parent["query_id"],
            "parent_rgb_sha256": evidence["rgb_pixel_sha256"],
            "shape_yx_rgb": list(rgb.shape),
            "roi_base64": base64.b64encode(bits).decode("ascii"),
            "roi_sha256": hashlib.sha256(bits).hexdigest(),
            "profile": profile.profile_id if profile else None,
            "template_placement": query_has_template_placement(parent),
            "warnings": list(evidence["warnings"]),
            "png_file_sha256": hashlib.sha256(png).hexdigest(),
            "rgb_pixel_sha256": hashlib.sha256(masked.tobytes()).hexdigest(),
        },
    }


def _validate_masked_query(record, parent=None):
    derived = record["derivation"]
    height, width, channels = derived["shape_yx_rgb"]
    if (any(type(size) is not int or size < 1 for size in (height, width))
            or channels != 3 or height * width > MAX_QUERY_PIXELS
            or derived["method"] != ROI_METHOD or not derived["parent_query_id"]
            or not isinstance(derived["parent_query_id"], str)
            or derived["parent_query_id"] == record["query_id"]
            or not _valid_sha256(derived["parent_rgb_sha256"])
            or type(derived["template_placement"]) is not bool
            or not isinstance(derived["warnings"], list)
            or any(not isinstance(warning, str) for warning in derived["warnings"])
            or record["out_of_date"]):
        raise ValueError("Invalid masked Color-Depth MIP derivation.")
    profile_id = derived["profile"]
    if ((profile_id is None and derived["template_placement"])
            or (profile_id is not None and search_profile(profile_id).search_canvas_yx != (height, width))):
        raise ValueError("Masked Color-Depth MIP canvas disagrees with its inherited search profile.")
    count = height * width
    size = (count + 7) // 8
    if len(derived["roi_base64"]) != ((size + 2) // 3) * 4:
        raise ValueError("Invalid retained search area size.")
    bits = base64.b64decode(derived["roi_base64"], validate=True)
    if (len(bits) != size or hashlib.sha256(bits).hexdigest() != derived["roi_sha256"]
            or (count % 8 and bits[-1] & ((1 << (8 - count % 8)) - 1))):
        raise ValueError("Invalid retained search area bits.")
    mask = np.unpackbits(np.frombuffer(bits, dtype=np.uint8), count=count, bitorder="big")
    mask = mask.reshape(height, width).view(np.bool_)
    if len(record["png_base64"]) > ((MAX_QUERY_PNG_BYTES + 2) // 3) * 4:
        raise ValueError("The retained Color-Depth MIP PNG exceeds the 8 MiB limit.")
    png = query_png(record)
    if len(png) > MAX_QUERY_PNG_BYTES or hashlib.sha256(png).hexdigest() != derived["png_file_sha256"]:
        raise ValueError("Saved masked query PNG checksum mismatch.")
    rgb = _png_array(png, role="Masked Color-Depth MIP", shape=(height, width, 3), dtype=np.uint8)
    if (hashlib.sha256(rgb.tobytes()).hexdigest() != derived["rgb_pixel_sha256"]
            or np.any(rgb, where=~mask[..., None]) or not rgb.any()):
        raise ValueError("Saved masked query pixels disagree with the search area or checksum.")
    if parent is not None:
        from .search_profiles import query_profile
        source = query_rgb(parent)
        profile = query_profile(parent)
        if (hashlib.sha256(source.tobytes()).hexdigest() != derived["parent_rgb_sha256"]
                or derived["profile"] != (profile.profile_id if profile else None)
                or derived["template_placement"] != query_has_template_placement(parent)
                or derived["warnings"] != list(query_image_evidence(parent)["warnings"])
                or not np.array_equal(rgb, mask_query_rgb(source, mask))):
            raise ValueError("Masked Color-Depth MIP disagrees with its retained source.")

def _canonical_pixel_hash(array):
    """Hash shape and scalar semantics as well as canonical C-order bytes."""
    array = np.asarray(array)
    if array.dtype.kind == "u" and array.dtype.itemsize == 2:
        dtype, pixels = "uint16-le", np.asarray(array, dtype="<u2", order="C")
    elif array.dtype.kind == "u" and array.dtype.itemsize == 1:
        dtype, pixels = "uint8", np.asarray(array, dtype=np.uint8, order="C")
    else:
        raise ValueError("Supplementary pixels must be uint8 or uint16.")
    header = canonical_json({"dtype": dtype, "shape": list(array.shape), "order": "C"}).encode("utf-8")
    checksum = hashlib.sha256(header + b"\0")
    checksum.update(memoryview(pixels))
    return checksum.hexdigest()


def _png_array(encoded, *, role, shape, dtype):
    if not isinstance(encoded, bytes) or not encoded.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"{role} must be retained as PNG bytes.")
    if len(encoded) > MAX_SUPPLEMENTARY_PNG_BYTES or len(encoded) < 26:
        raise ValueError(f"{role} PNG is outside the supported byte bound.")
    bit_depth, color_type = encoded[24], encoded[25]
    expected = (16, 0) if dtype == np.uint16 else ((8, 2) if len(shape) == 3 else (8, 0))
    if (bit_depth, color_type) != expected:
        raise ValueError(f"{role} PNG encoding must preserve {np.dtype(dtype).name} pixels without conversion.")
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1 or image.size != (shape[1], shape[0]):
                raise ValueError(f"{role} PNG canvas or frame count is invalid.")
            image.load()
            pixels = np.asarray(image)
            if pixels.shape != shape:
                raise ValueError(f"{role} PNG has an unexpected channel layout.")
            if dtype == np.uint16:
                pixels = pixels.astype(np.uint16, copy=False)
            elif pixels.dtype != np.uint8:
                raise ValueError(f"{role} PNG must decode as uint8.")
            return np.array(pixels, copy=True)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid {role} PNG: {exc}") from exc


def _query_pixel_evidence(record):
    if record["version"] == ROI_QUERY_VERSION:
        derived = record["derivation"]
        return {"png_sha256": derived["png_file_sha256"], "pixel_sha256": derived["rgb_pixel_sha256"],
                "shape": list(derived["shape_yx_rgb"])}
    if record["version"] == 2:
        imported = record["imported_image"]
        return {"png_sha256": imported["png_file_sha256"], "pixel_sha256": imported["rgb_pixel_sha256"],
                "shape": list(imported["shape_yx_rgb"])}
    artifact = record["artifact"]
    return {"png_sha256": artifact["png_file_sha256"], "pixel_sha256": artifact["rgb_pixel_sha256"],
            "shape": [*search_profile(artifact["profile"]).search_canvas_yx, 3]}


def _explicit_association(association):
    if not isinstance(association, dict) or association.get("computational_role") != "lm_target":
        raise ValueError("Declare the retained query's LM computational role explicitly.")
    origin = association.get("origin")
    if origin not in {"verified", "user-declared"}:
        raise ValueError("LM association origin must be verified or user-declared.")
    result = json.loads(canonical_json(association))
    alignment = result.get("alignment", "unknown")
    if alignment not in {"verified", "unknown", "unverified"}:
        raise ValueError("LM alignment must be explicitly verified, unknown, or unverified.")
    if alignment == "verified" and result.get("alignment_space") is None:
        raise ValueError("Verified LM alignment requires an explicit alignment space.")
    if origin == "verified" and not (result.get("evidence_id") or result.get("evidence")):
        raise ValueError("Verified LM association requires a bounded evidence identity.")
    if origin == "user-declared" and not (result.get("declaration") or result.get("evidence_id")):
        raise ValueError("User-declared LM association requires an explicit declaration identity.")
    result.setdefault("warnings", [])
    if not isinstance(result["warnings"], list) or any(not isinstance(item, str) for item in result["warnings"]):
        raise ValueError("LM association warnings must be a list of text.")
    if result.get("alignment") in (None, "unknown", "unverified"):
        warning = "LM biological alignment is unknown; role and pixel association are explicitly declared only."
        if warning not in result["warnings"]:
            result["warnings"].append(warning)
    if len(canonical_json(result).encode("utf-8")) > 16 * 1024:
        raise ValueError("LM association evidence exceeds its bound.")
    return result


def _valid_sha256(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(char in "0123456789abcdef" for char in value))


def _reference_zgap_preparation(label_policy, reference):
    clear = label_policy == "reference-labels"
    return {"identity": reference, "method": "madi3d-reference-prepared-zgap/1",
            "parameters": {"clear_name_label": clear, "clear_color_scale": clear,
                           "reference_revision": reference, "mask_threshold": 20, "maximum_radius": 10}}


def validate_supplementary_binding(bundle, record):
    if bundle["query_id"] != record["query_id"] or bundle["query"] != _query_pixel_evidence(record):
        raise ValueError("Supplementary bundle contradicts its retained query binding.")
    if record["version"] in (3, ROI_QUERY_VERSION):
        from .search_profiles import query_profile
        profile = query_profile(record)
        if profile is not None and bundle["profile"] == profile.profile_id:
            return
        raise ValueError("Supplementary bundle profile contradicts the retained query.")


def validate_supplementary_bundle(bundle, record=None, *, decoded=False, cancel=None):
    """Validate consumed evidence, optionally returning the verified pixel arrays."""
    search_checkpoint(cancel)
    if not isinstance(bundle, dict):
        raise ValueError("Supplementary bundle must be a mapping.")
    if record is not None:
        validate_supplementary_binding(bundle, record)
        if record["version"] in (2, ROI_QUERY_VERSION):
            qpng = query_png(record)
            if hashlib.sha256(qpng).hexdigest() != bundle["query"]["png_sha256"]:
                raise ValueError("Supplementary bundle query PNG disagrees with retained query bytes.")
            qpixels = _png_array(qpng, role="Query", shape=tuple(bundle["query"]["shape"]), dtype=np.uint8)
            actual_query_hash = hashlib.sha256(qpixels.tobytes()).hexdigest()
            del qpixels
        else:
            qpng = query_png(record)
            if hashlib.sha256(qpng).hexdigest() != bundle["query"]["png_sha256"]:
                raise ValueError("Supplementary bundle query PNG disagrees with retained query bytes.")
            with Image.open(io.BytesIO(qpng)) as image:
                _require_rgb8(image, qpng)
                if image.size != tuple(reversed(_query_pixel_evidence(record)["shape"][:2])):
                    raise ValueError("Retained query canvas contradicts supplementary bundle evidence.")
                actual_query_hash = hashlib.sha256(image.tobytes()).hexdigest()
        if actual_query_hash != bundle["query"]["pixel_sha256"]:
            raise ValueError("Supplementary bundle query pixels disagree with retained query bytes.")
        del qpng
    if bundle.get("schema") != 1 or bundle.get("origin") not in {"generated", "imported"}:
        raise ValueError("Invalid supplementary bundle schema or origin.")
    if not isinstance(bundle.get("query_id"), str) or not bundle["query_id"] or len(bundle["query_id"]) > 256:
        raise ValueError("Supplementary query identity is invalid.")
    payload_chars = 0
    for role in ("gradient", "zgap"):
        item = bundle.get(role)
        if not isinstance(item, dict) or not isinstance(item.get("png_base64"), str):
            raise ValueError(f"Supplementary {role} evidence must contain a PNG payload.")
        payload_chars += len(item["png_base64"])
        if len(item["png_base64"]) > 12 * 1024 * 1024:
            raise ValueError(f"Supplementary {role} PNG exceeds its encoded byte bound.")
    roi_input = bundle.get("roi")
    if roi_input is not None:
        if not isinstance(roi_input, dict) or not isinstance(roi_input.get("png_base64"), str):
            raise ValueError("Supplementary ROI evidence must contain a PNG payload.")
        payload_chars += len(roi_input["png_base64"])
        if len(roi_input["png_base64"]) > 12 * 1024 * 1024:
            raise ValueError("Supplementary ROI PNG exceeds its encoded byte bound.")
    if payload_chars > 36 * 1024 * 1024:
        raise ValueError("Supplementary PNG payloads exceed their combined byte bound.")
    profile = search_profile(bundle.get("profile"))
    shape = profile.search_canvas_yx
    expected_bundle_id = hashlib.sha256(canonical_json({k: v for k, v in bundle.items() if k != "bundle_id"}).encode("utf-8")).hexdigest()
    if bundle.get("bundle_id") != expected_bundle_id:
        raise ValueError("Supplementary bundle identity checksum mismatch.")
    association = _explicit_association(bundle.get("association"))
    if association != bundle.get("association"):
        raise ValueError("Supplementary LM association is not canonical.")
    if (association.get("alignment") == "verified" and
            association.get("alignment_space") != profile.alignment_space):
        raise ValueError("Verified LM alignment contradicts the selected search profile.")
    query = bundle.get("query")
    if (not isinstance(query, dict) or query.get("shape") != [*shape, 3]
            or not _valid_sha256(query.get("png_sha256")) or not _valid_sha256(query.get("pixel_sha256"))):
        raise ValueError("Supplementary query identity or canvas evidence is invalid.")
    producer, preparation = bundle.get("producer"), bundle.get("preparation")
    if not isinstance(producer, dict) or not isinstance(preparation, dict):
        raise ValueError("Supplementary producer and preparation evidence must be mappings.")
    if len(canonical_json({"producer": producer, "preparation": preparation}).encode("utf-8")) > 16 * 1024:
        raise ValueError("Supplementary producer or preparation evidence exceeds its bound.")
    if bundle["origin"] == "generated":
        required = {"implementation", "producer_revision", "imagej_revision", "scoring_reference"}
        if (not required.issubset(producer) or
                any(not isinstance(producer[key], str) or not producer[key] for key in required)):
            raise ValueError("Generated derivative producer identity is incomplete.")
        if (preparation.get("label_policy") not in {"reference-labels", "no-labels"} or
                preparation.get("max_filter_radius") != 20):
            raise ValueError("Generated derivative preparation policy is invalid.")
    else:
        if not producer or not (producer.get("identity") or producer.get("revision") or producer.get("source_id")):
            raise ValueError("Imported derivative producer requires an explicit source identity.")
        if "implementation" in producer and producer["implementation"] == "madi3d-lm-derivatives/1":
            raise ValueError("Imported images cannot claim the native generated producer identity.")
        if (preparation.get("label_policy") not in {"reference-labels", "no-labels"} or
                not isinstance(preparation.get("gradient_parameters"), dict)):
            raise ValueError("Imported derivative preparation policy is invalid.")
    gradient, zgap = bundle.get("gradient"), bundle.get("zgap")
    if not isinstance(gradient, dict) or not isinstance(zgap, dict):
        raise ValueError("Supplementary gradient and Z-gap evidence must be mappings.")
    if gradient.get("origin") != bundle["origin"] or zgap.get("origin") not in {"generated", "imported"}:
        raise ValueError("Supplementary derivative origins are inconsistent.")
    zgap_preparation = preparation.get("zgap")
    if (not isinstance(zgap_preparation, dict) or not zgap_preparation.get("identity") or
            not zgap_preparation.get("method") or not isinstance(zgap_preparation.get("parameters"), dict)):
        raise ValueError("Supplementary Z-gap identity, method, and parameters are required.")
    if zgap.get("origin") == "imported" and zgap_preparation["method"] == "madi3d-reference-prepared-zgap/1":
        raise ValueError("A supplied Z-gap requires its distinct preparation method identity.")
    if zgap.get("origin") == "generated" and zgap_preparation["method"] == "madi3d-reference-prepared-zgap/1":
        label_policy = preparation["label_policy"]
        reference = zgap_preparation["identity"]
        if (not isinstance(reference, str) or len(reference) != 40 or
                any(char not in "0123456789abcdef" for char in reference) or
                zgap_preparation != _reference_zgap_preparation(label_policy, reference) or
                bundle["origin"] == "generated" and producer["scoring_reference"] != reference):
            raise ValueError("Reference-prepared Z-gap parameters are inconsistent.")
    arrays = {}
    for role, dtype, expected_shape in (("gradient", np.uint16, shape), ("zgap", np.uint8, (*shape, 3))):
        search_checkpoint(cancel)
        item = bundle[role]
        if not isinstance(item, dict):
            raise ValueError(f"Supplementary {role} evidence must be a mapping.")
        encoded_text = item.get("png_base64")
        if not isinstance(encoded_text, str) or len(encoded_text) > 12 * 1024 * 1024:
            raise ValueError(f"Supplementary {role} PNG exceeds its encoded byte bound.")
        try:
            encoded = base64.b64decode(encoded_text, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Supplementary {role} PNG encoding is invalid.") from exc
        if (item.get("shape") != (list(shape) if role == "gradient" else [*shape, 3]) or
                item.get("dtype") != ("uint16-le" if role == "gradient" else "uint8") or
                not _valid_sha256(item.get("pixel_sha256")) or
                hashlib.sha256(encoded).hexdigest() != item.get("png_sha256")):
            raise ValueError(f"Supplementary {role} PNG checksum mismatch.")
        pixels = _png_array(encoded, role=role,
                            shape=expected_shape, dtype=dtype)
        if item["pixel_sha256"] != _canonical_pixel_hash(pixels):
            raise ValueError(f"Supplementary {role} pixels disagree with their hash.")
        if decoded:
            arrays[role] = pixels
    if bundle.get("roi") is not None:
        item = bundle["roi"]
        encoded_text = item.get("png_base64")
        if not isinstance(encoded_text, str) or len(encoded_text) > 12 * 1024 * 1024:
            raise ValueError("Supplementary ROI PNG exceeds its encoded byte bound.")
        try:
            encoded = base64.b64decode(encoded_text, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Supplementary ROI PNG encoding is invalid.") from exc
        if hashlib.sha256(encoded).hexdigest() != item["png_sha256"]:
            raise ValueError("Supplementary ROI PNG checksum mismatch.")
        pixels = _png_array(encoded, role="ROI", shape=shape, dtype=np.uint8)
        if not np.isin(pixels, (0, 255)).all() or item["pixel_sha256"] != _canonical_pixel_hash(pixels):
            raise ValueError("Supplementary ROI pixels are invalid.")
        if decoded:
            arrays["roi"] = pixels
    search_checkpoint(cancel)
    return arrays if decoded else None


def bundle_reference(bundle):
    reference = {"bundle_id": bundle["bundle_id"], "query_id": bundle["query_id"],
            "origin": bundle["origin"],
            "gradient_origin": bundle["gradient"]["origin"], "zgap_origin": bundle["zgap"]["origin"],
            "query_png_sha256": bundle["query"]["png_sha256"],
            "query_pixel_sha256": bundle["query"]["pixel_sha256"],
            "profile_id": bundle["profile"],
            "gradient_png_sha256": bundle["gradient"]["png_sha256"],
            "gradient_pixel_sha256": bundle["gradient"]["pixel_sha256"],
            "zgap_png_sha256": bundle["zgap"]["png_sha256"],
            "zgap_pixel_sha256": bundle["zgap"]["pixel_sha256"],
            "association": copy.deepcopy(bundle["association"]),
            "producer": copy.deepcopy(bundle["producer"]), "preparation": copy.deepcopy(bundle["preparation"])}
    if bundle["roi"] is not None:
        reference["roi_png_sha256"] = bundle["roi"]["png_sha256"]
        reference["roi_pixel_sha256"] = bundle["roi"]["pixel_sha256"]
    return reference


def query_dependencies(record):
    """Project validated generation facts without rehashing mapping histories."""
    if record["version"] in (2, ROI_QUERY_VERSION):
        return []
    if record["version"] == 1:
        return copy.deepcopy(record["dependencies"])
    artifact = record["artifact"]
    inputs = []
    if artifact["parameters"]["mode"] != "binary_mask":
        inputs.append((artifact["selection"], artifact["geometry"], artifact["source_pixel_sha256"],
                       artifact.get("source_scientific_revision")))
    if artifact["mask_selection"] is not None:
        geometry = artifact.get("mask_geometry") or dict(artifact["geometry"], revision=artifact["mask_geometry_revision"])
        inputs.append((artifact["mask_selection"], geometry, artifact["mask_pixel_sha256"],
                       artifact.get("mask_scientific_revision")))
    values = []
    for selection, geometry, pixels, revision in inputs:
        values.append(QueryInput(CDMSelection.from_dict(selection), CDMGeometry.from_dict(geometry), None,
            scientific_revision=revision).dependency(pixel_checksum=pixels))
    return values


def normalize_query_metadata(metadata):
    """Migrate at publication/load boundaries, never during validation or review.

    Check duplicated legacy facts before collapsing them. No live source is
    consulted: saved grids and revisions remain the ones actually consumed.
    """
    result = dict(metadata)
    registry = dict(result.get(QUERY_KEY, {}))
    for key, original in registry.items():
        if original["version"] != 1 and "export_directory" not in original:
            continue
        validate_queries({QUERY_KEY: {key: original}})
        record = copy.deepcopy(original)
        record.pop("export_directory", None)
        if record["version"] == 1:
            artifact = CDMArtifact.from_dict(record["artifact"])
            dependencies = record.pop("dependencies")
            binary = artifact.parameters.mode == "binary_mask"
            additions = {"source_scientific_revision": None if binary else dependencies[0].get("scientific_revision")}
            if artifact.mask_selection is not None:
                mask = dependencies[-1]
                geometry = CDMGeometry.from_dict(mask["geometry"])
                additions.update(mask_scientific_revision=mask.get("scientific_revision"),
                    mask_geometry_revision=geometry.revision,
                    mask_geometry=geometry if (geometry.working_grid != artifact.geometry.working_grid or
                                              geometry.pose != artifact.geometry.pose) else None)
            record["artifact"] = replace(artifact, **additions).to_dict()
            record["version"] = 3
        registry[key] = record
    if registry:
        result[QUERY_KEY] = registry
    # Older local searches copied the entire mapping. Collapse only a proven
    # duplicate of the retained query; unmatched external evidence stays intact.
    sessions = dict(result.get("neuronbridge_sessions", {}))
    for session_id, payload in sessions.items():
        context = payload.get("session", {}).get("context", {})
        query = registry.get(context.get("query_id"))
        if context.get("backend") != "madi3d-local" or "query_mapping" not in context or query is None:
            continue
        evidence = query_image_evidence(query)
        if (context.get("query_png_sha256") == evidence["png_file_sha256"] and
                context.get("query_pixel_sha256") == evidence["rgb_pixel_sha256"] and
                canonical_json(context["query_mapping"]) == canonical_json(evidence["mapping"])):
            context = {key: value for key, value in context.items() if key != "query_mapping"}
            sessions[session_id] = dict(payload, session=dict(payload["session"], context=context))
    if sessions:
        result["neuronbridge_sessions"] = sessions
    return result


def _read_query_file(path, limit, cancel, *, role="Color-Depth MIP"):
    checkpoint(cancel)
    with Path(path).open("rb") as stream:
        content = stream.read(limit + 1)
    checkpoint(cancel)
    if len(content) > limit:
        raise ValueError(f"{role} file exceeds the {limit // (1024 * 1024)} MiB limit.")
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
    original = _read_query_file(path, MAX_QUERY_PNG_BYTES, cancel)
    with Image.open(io.BytesIO(original)) as image:
        _require_rgb8(image, original)
        if image.width * image.height > MAX_QUERY_PIXELS:
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
    if len(png) > MAX_QUERY_PNG_BYTES:
        raise ValueError("The retained Color-Depth MIP PNG exceeds the 8 MiB limit.")
    png_sha = hashlib.sha256(png).hexdigest()
    manifest_json = None
    label = path.stem
    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_file():
        text = _read_query_file(manifest_path, 16 * 1024 * 1024, cancel).decode("utf-8")
        manifest = json.loads(text)
        if isinstance(manifest, dict):
            retained = manifest.get("derived_query", manifest.get("imported_query"))
            if retained is not None and manifest.get("image") == path.name:
                record = retained
                expected = ROI_QUERY_VERSION if "derived_query" in manifest else 2
                if record.get("version") != expected:
                    raise ValueError("Unsupported imported Color-Depth MIP manifest.")
                record = dict(record)
                record.setdefault("png_base64", base64.b64encode(png).decode("ascii"))
                record = normalize_query_metadata({QUERY_KEY: {record["query_id"]: record}})[QUERY_KEY][record["query_id"]]
                validate_queries({QUERY_KEY: {record["query_id"]: record}})
                if query_png(record) != png:
                    raise ValueError("Color-Depth MIP manifest disagrees with the selected image.")
                checkpoint(cancel)
                return record
            artifact = manifest.get("artifact")
            if ((manifest.get("image") == path.name and ("artifact" in manifest or "context_sha256" in manifest))
                    or isinstance(artifact, dict) and artifact.get("png_file_sha256") == png_sha):
                manifest_json = text
                if isinstance(manifest.get("label"), str) and manifest["label"].strip():
                    label = manifest["label"]
    imported = ImportedCDMImage(str(path), source_format, hashlib.sha256(original).hexdigest(), len(original),
        datetime.now(timezone.utc).isoformat(), tuple(pixels.shape), png_sha,
        hashlib.sha256(pixels.tobytes()).hexdigest(), manifest_json)
    record = {"version": 2, "query_id": query_id, "label": label,
              "imported_image": imported.to_dict(), "dependencies": [], "out_of_date": [],
              "png_base64": base64.b64encode(png).decode("ascii")}
    validate_queries({QUERY_KEY: {query_id: record}})
    checkpoint(cancel)
    return json.loads(canonical_json(record))


def query_image_evidence(record):
    """Search consumes supplied image facts without inventing generation inputs."""
    if record["version"] == ROI_QUERY_VERSION:
        return dict(record["derivation"], mapping=None)
    if record["version"] == 2:
        image = ImportedCDMImage.from_dict(record["imported_image"])
        artifact = image.generation_artifact
        return {"png_file_sha256": image.png_file_sha256, "rgb_pixel_sha256": image.rgb_pixel_sha256,
                "warnings": list(image.warnings), "mapping": artifact.to_dict()["mapping"] if artifact else None}
    return record["artifact"]


def restore_query(record, destination, cancel=None):
    """Restore retained pixels through the same verified atomic export boundary."""
    checkpoint(cancel)
    if record["version"] in (2, ROI_QUERY_VERSION):
        validate_queries({QUERY_KEY: {record["query_id"]: record}})
        with stage_cdm_bundle(destination, cancel) as staging:
            evidence = {key: value for key, value in record.items() if key not in ("png_base64", "export_directory")}
            kind = "derived_query" if record["version"] == ROI_QUERY_VERSION else "imported_query"
            manifest = {"image": CDM_IMAGE_NAME, kind: evidence}
            files = {CDM_IMAGE_NAME: query_png(record),
                     "manifest.json": canonical_json(manifest).encode("utf-8")}
            for name, content in files.items():
                checkpoint(cancel)
                with (staging / name).open("xb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
        return
    png = query_png(record)
    with Image.open(io.BytesIO(png)) as image:
        pixels = np.asarray(image).copy()
    artifact = CDMArtifact.from_dict(record["artifact"])
    return export_cdm(CDMResult(pixels, artifact, None), destination, cancel=cancel, label=record["label"],
                      query_id=record["query_id"], retained_png=png)


def validate_queries(metadata, *, derivation_parents=None):
    queries = metadata.get(QUERY_KEY, {})
    parents = queries if derivation_parents is None else derivation_parents
    for key, record in queries.items():
        if record["version"] not in (1, 2, 3, ROI_QUERY_VERSION) or key != record["query_id"] or not record["label"]:
            raise ValueError("Invalid saved NeuronBridge query identity.")
        if record["version"] == ROI_QUERY_VERSION:
            if any(field in record for field in ("artifact", "imported_image", "dependencies")):
                raise ValueError("Masked Color-Depth MIPs retain a 2D image derivation.")
            _validate_masked_query(record, parents.get(record["derivation"]["parent_query_id"]))
            continue
        if record["version"] == 2:
            imported = ImportedCDMImage.from_dict(record["imported_image"])
            png = query_png(record)
            if len(png) > MAX_QUERY_PNG_BYTES or hashlib.sha256(png).hexdigest() != imported.png_file_sha256:
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
        if record["version"] == 3:
            if "dependencies" in record or "export_directory" in record:
                raise ValueError("Query inputs belong to the generation artifact; export hints belong to preferences.")
            if artifact.mask_selection is not None and artifact.mask_geometry_revision is None:
                raise ValueError("Missing consumed mask geometry revision.")
        png = query_png(record)
        if len(png) > MAX_QUERY_PNG_BYTES or hashlib.sha256(png).hexdigest() != artifact.png_file_sha256:
            raise ValueError("Saved query PNG checksum mismatch.")
        with Image.open(io.BytesIO(png)) as image:
            _require_rgb8(image, png)
            if image.format != "PNG" or image.size != search_profile(artifact.profile).search_canvas_yx[::-1]:
                raise ValueError("Invalid saved query canvas.")
            if hashlib.sha256(image.tobytes()).hexdigest() != artifact.rgb_pixel_sha256:
                raise ValueError("Saved query pixels disagree with generation evidence.")
        expected = 2 if artifact.parameters.mode == "signal_mask" else 1
        dependencies = query_dependencies(record)
        if len(dependencies) != expected:
            raise ValueError("Missing consumed query inputs.")
        for dependency in dependencies:
            CDMSelection.from_dict(dependency["selection"])
            CDMGeometry.from_dict(dependency["geometry"])
            revision = dependency.get("scientific_revision")
            if revision is not None and (not isinstance(revision, str) or len(revision) != 64 or
                                         any(c not in "0123456789abcdef" for c in revision)):
                raise ValueError("Invalid query scientific revision.")
        first = dependencies[0]
        if (CDMSelection.from_dict(first["selection"]) != artifact.selection or
                CDMGeometry.from_dict(first["geometry"]) != artifact.geometry or
                first["pixels"] != (artifact.source_pixel_sha256 or artifact.mask_pixel_sha256)):
            raise ValueError("Query dependency disagrees with consumed generation evidence.")
        if (artifact.source_scientific_revision is not None and
                first.get("scientific_revision") != artifact.source_scientific_revision):
            raise ValueError("Query dependency disagrees with the consumed scientific revision.")
        if artifact.mask_selection is not None:
            last = dependencies[-1]
            if (CDMSelection.from_dict(last["selection"]) != artifact.mask_selection or
                    last["pixels"] != artifact.mask_pixel_sha256):
                raise ValueError("Query mask dependency disagrees with generation evidence.")
            if ((artifact.mask_scientific_revision is not None and
                    last.get("scientific_revision") != artifact.mask_scientific_revision) or
                    (artifact.mask_geometry_revision is not None and
                     last["geometry"]["revision"] != artifact.mask_geometry_revision) or
                    (artifact.mask_geometry is not None and
                     CDMGeometry.from_dict(last["geometry"]) != artifact.mask_geometry)):
                raise ValueError("Query mask dependency disagrees with the consumed geometry or scientific revision.")
        validate_query_freshness(record)
    validate_query_associations(metadata)
    for bundle_id, bundle in metadata.get(SUPPLEMENTARY_KEY, {}).items():
        if bundle_id != bundle.get("bundle_id"):
            raise ValueError("Supplementary bundle key disagrees with its identity.")
        record = queries.get(bundle.get("query_id"))
        if record is not None:
            validate_supplementary_binding(bundle, record)


def validate_query_freshness(record):
    if not isinstance(record["out_of_date"], list) or any(not isinstance(v, str) for v in record["out_of_date"]):
        raise ValueError("Invalid query freshness diagnostics.")


def validate_query_associations(metadata):
    queries = metadata.get(QUERY_KEY, {})
    for session, query in metadata.get(ASSOCIATION_KEY, {}).items():
        if session not in metadata.get("neuronbridge_sessions", {}) or query not in queries:
            raise ValueError("Query association references missing saved evidence.")


def add_query(metadata, record):
    result = copy.deepcopy(normalize_query_metadata(metadata))
    record = normalize_query_metadata({QUERY_KEY: {record["query_id"]: record}})[QUERY_KEY][record["query_id"]]
    registry = dict(result.get(QUERY_KEY, {}))
    result[QUERY_KEY] = registry
    key = record["query_id"]
    if key in registry and registry[key] != record:
        raise ValueError("A completed query cannot be overwritten.")
    registry[key] = copy.deepcopy(record)
    # Existing records are checked at the document boundary. Adding one query
    # must not decode all historical images before that boundary is reached.
    validate_queries({QUERY_KEY: {key: registry[key]}}, derivation_parents=registry)
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
    result = copy.deepcopy(normalize_query_metadata(metadata))
    incoming = normalize_query_metadata(incoming)
    for field in (QUERY_KEY, ASSOCIATION_KEY, SUPPLEMENTARY_KEY):
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
        if record["out_of_date"] or record["version"] in (2, ROI_QUERY_VERSION):
            continue
        reasons = list(record["out_of_date"])
        mapping = CDMMapping(**record["artifact"]["mapping"])
        if (resolve_mapping is not None and mapping.evidence_json is not None and
                str(mapping.evidence_id or "").startswith("madi3d-geometry:")):
            try:
                channel_id = json.loads(mapping.evidence_json)["source_channel_id"]
                current_mapping = resolve_mapping(channel_id)
                if current_mapping is not None and current_mapping.evidence_sha256 != mapping.evidence_sha256:
                    reasons.append("Consumed template mapping provenance changed.")
            except (ValueError, LookupError, RuntimeError) as exc:
                reasons.append(str(exc))
        for i, previous in enumerate(query_dependencies(record)):
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
