"""Native, GUI-independent prepared colour-depth projection.

Inputs are borrowed scalar ZYX arrays (including memmaps/strided views). The
caller must hold a read lease for the whole call. Geometry and selection must
come from authoritative snapshots, never actors or display settings.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from madi3d_app.volume.resampling import sample_affine_zyx
from .cdm_lut import LUT_BYTES
from .cdm_records import (
    CDMArtifact, CDMGeometry, CDMMapping, CDMParameters, CDMSelection,
)
from .search_profiles import BRAIN, search_profile, template_profile


class CDMCancelled(RuntimeError):
    """Generation/export cancelled; no completed artifact was published."""


def checkpoint(cancel):
    if cancel is not None and cancel():
        raise CDMCancelled("CDM operation cancelled.")


def array_checksum(array, *, cancel=None, binary=False):
    """SHA-256 of C-order, little-endian scalar bytes in <=64K-value chunks."""
    digest = hashlib.sha256()
    dtype = array.dtype.newbyteorder("<")
    with np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                   op_flags=["readonly"], op_dtypes=[dtype], order="C",
                   buffersize=65536) as chunks:
        for chunk in chunks:
            checkpoint(cancel)
            if binary and np.any((chunk != 0) & (chunk != 1)):
                raise ValueError("Mask must contain only explicit 0/1 membership.")
            digest.update(chunk.tobytes())
    return digest.hexdigest()


def _scalar(array, name, *, mask=False):
    if not isinstance(array, np.ndarray) or array.ndim != 3 or any(n <= 0 for n in array.shape):
        raise ValueError(f"{name} must be one explicitly selected nonempty scalar ZYX ndarray.")
    if mask:
        valid = array.dtype.kind in "bu" and array.dtype.itemsize in (1, 2)
    else:
        valid = array.dtype.kind == "u" and array.dtype.itemsize in (1, 2)
    if not valid:
        raise ValueError(f"{name} dtype is outside NB-02; float (including non-finite), signed and RGB inputs require a separate conversion contract.")


def _exact_view(array, mapping, profile=BRAIN):
    """Signed permutation + integer placement, without copying source pixels."""
    rotation = mapping[:3, :3]
    absolute = np.abs(rotation)
    if not (np.all((absolute == 0) | (absolute == 1)) and
            np.all(absolute.sum(axis=0) == 1) and np.all(absolute.sum(axis=1) == 1) and
            np.array_equal(mapping[:3, 3], np.floor(mapping[:3, 3]))):
        raise ValueError("Exact mapping requires a signed axis permutation and integer translation; explicitly select nearest for resampling.")
    source_axes = absolute.argmax(axis=1)
    signs = rotation[np.arange(3), source_axes].astype(int)
    view = array.transpose(tuple(2 - source_axes[g] for g in (2, 1, 0)))
    view = view[tuple(slice(None, None, int(signs[g])) for g in (2, 1, 0))]
    low = mapping[:3, 3] + np.minimum(signs, 0) * (np.asarray(view.shape[::-1]) - 1)
    high = low + np.asarray(view.shape[::-1])
    if np.any(low < 0) or np.any(high > profile.template_shape_zyx[::-1]):
        raise ValueError("Mapped source extends outside the selected template canvas.")
    return view, tuple(int(v) for v in low[::-1])


def _encode_table(profile=BRAIN):
    if hashlib.sha256(LUT_BYTES).hexdigest() != profile.lut_sha256:
        raise ValueError("Pinned CDM LUT checksum mismatch.")
    lut = np.loadtxt(LUT_BYTES.decode("ascii").splitlines(), dtype=np.uint8)
    depth = profile.template_shape_zyx[0]
    depths = np.floor(255 * (np.arange(depth, dtype=np.float64) / depth) + 0.5).astype(int)
    # Preserve ImageJ binary64 division, multiplication, then truncation.
    table = ((np.arange(256, dtype=np.float64)[None, :, None] / 255.0)
             * lut[depths, None, :]).astype(np.uint8)
    table[:, :2] = 0
    return table


def _reduce(accumulated, candidate):
    """Ordered MIP_right_color reduction on the pinned scalar-LUT domain only."""
    brighter = candidate.max(axis=-1) > accumulated.max(axis=-1)
    winner = np.where(brighter[..., None], candidate, accumulated)
    loser = np.where(brighter[..., None], accumulated, candidate)
    tertiary = winner.argmin(axis=-1)[..., None]
    low = np.take_along_axis(winner, tertiary, axis=-1)[..., 0]
    other = np.take_along_axis(loser, tertiary, axis=-1)[..., 0]
    second = np.sort(winner, axis=-1)[..., 1]
    saturated = (winner.max(axis=-1) == 255) & (loser.max(axis=-1) == 255)
    mixed = np.where((low < other) & (other < second) & ~saturated, other, low)
    np.put_along_axis(winner, tertiary, mixed[..., None], axis=-1)
    accumulated[:] = winner


@dataclass(frozen=True)
class CDMResources:
    input_bytes: int
    output_bytes: int
    peak_tile_voxels: int
    tiles_visited: int
    estimated_working_bytes: int


@dataclass(frozen=True)
class CDMResult:
    rgb: np.ndarray
    artifact: CDMArtifact
    resources: CDMResources


def generate_cdm(signal, *, selection: CDMSelection, geometry: CDMGeometry,
                 mapping: CDMMapping, parameters: CDMParameters,
                 mask=None, mask_selection: CDMSelection | None = None,
                 mask_revision: str | None = None, tile_rows: int = 32,
                 cancel=None) -> CDMResult:
    """Generate the selected profile RGB search canvas and immutable reproducibility evidence.

    ``signal=None`` is required for binary_mask; selection then identifies the
    mask. Optional masks are explicitly co-indexed on the frozen working grid.
    Exact mappings preserve values. Nearest uses the existing zero-background
    SciPy pull sampler on raw integers before NB-02 conversion/masking. No scene
    pose, working-grid affine or template preset is implicitly applied.
    """
    checkpoint(cancel)
    if not isinstance(selection, CDMSelection) or not isinstance(geometry, CDMGeometry):
        raise ValueError("Explicit selection and frozen geometry are required.")
    if not isinstance(mapping, CDMMapping) or not isinstance(parameters, CDMParameters):
        raise ValueError("Explicit mapping and generation parameters are required.")
    profile = search_profile(parameters.profile)
    if template_profile(mapping.template_id, mapping.template_sha256) != profile:
        raise ValueError("Mapping and generation search profiles disagree.")
    if type(tile_rows) is not int or not 1 <= tile_rows <= 32:
        raise ValueError("tile_rows must be an integer in 1..32.")
    if mapping.evidence_id is None and not parameters.allow_exploratory:
        raise ValueError("Unverified mapping requires allow_exploratory=True.")
    binary = parameters.mode == "binary_mask"
    if binary:
        if signal is not None or mask_selection != selection:
            raise ValueError("Binary mode requires no signal and selection equal to mask_selection.")
    else:
        if parameters.scalar_range is None:
            _scalar(signal, "signal")
        elif (not isinstance(signal, np.ndarray) or signal.ndim != 3 or
              any(n <= 0 for n in signal.shape) or signal.dtype.kind not in "buif"):
            raise ValueError("Scalar conversion needs one numeric ZYX frame.")
        else:
            for plane in signal:
                checkpoint(cancel)
                if not np.isfinite(plane).all():
                    raise ValueError("Source contains non-finite values; resolve them before generating a Color-Depth MIP.")
        if parameters.scalar_range is None and (signal.dtype.itemsize == 2) != (parameters.display_range is not None):
            raise ValueError("uint16 needs explicit display endpoints; uint8 must not rescale.")
    if binary and parameters.display_range is not None:
        raise ValueError("Binary masks have fixed 255 occupancy; display endpoints do not apply.")
    if parameters.mode != "signal":
        _scalar(mask, "mask", mask=True)
        if not isinstance(mask_selection, CDMSelection) or not isinstance(mask_revision, str) or not mask_revision.strip():
            raise ValueError("Explicit mask selection and revision are required.")
    elif any(v is not None for v in (mask, mask_selection, mask_revision)):
        raise ValueError("Declare signal_mask mode to consume a mask.")
    source = mask if binary else signal
    if tuple(source.shape[::-1]) != geometry.working_grid.dimensions:
        raise ValueError("Frozen working geometry does not match the selected scalar array.")
    if mask is not None and mask.shape != source.shape:
        raise ValueError("Mask must be co-indexed on the source working geometry.")
    affine = np.asarray(mapping.source_index_to_template_index)
    warnings = list(profile.warnings) + list(geometry.working_grid.warnings)
    warnings.extend(mapping.assumptions)
    if parameters.scalar_range is not None:
        warnings.append(f"Scalar values were clipped and converted to 8-bit using endpoints {parameters.scalar_range}.")
    warnings.extend(geometry.working_grid.physical_grid_diagnostics)
    warnings.extend(f"Source working geometry assumes {field}." for field in geometry.working_grid.assumed_fields)
    warnings.extend(f"Source working geometry replaces {field}." for field in geometry.working_grid.replaced_fields)
    if mapping.evidence_id is None:
        warnings.append("Exploratory output: mapping to the template has not been verified.")
    scipy_version = None
    if parameters.interpolation == "exact":
        source_view, (oz, oy, ox) = _exact_view(source, affine, profile)
        mask_view = None if mask is None else _exact_view(mask, affine, profile)[0]
        nz, ny, nx = source_view.shape
    else:
        import scipy
        for array in (source, mask):
            if array is not None and (not array.dtype.isnative or not array.flags.aligned):
                raise ValueError("Nearest resampling requires native-endian aligned source storage to avoid an implicit full-volume copy.")
        scipy_version = scipy.__version__
        warnings.append("Nearest resampling uses zero outside source voxel centres and clips to the full template canvas; resampling is not an NB-02 registration validation.")
        permutation = np.eye(4)[[2, 1, 0, 3]]
        pull = permutation @ np.linalg.inv(affine) @ permutation
        oz = oy = ox = 0
        nz, ny, nx = profile.template_shape_zyx
    source_hash = None if binary else array_checksum(signal, cancel=cancel)
    mask_hash = None if mask is None else array_checksum(mask, cancel=cancel, binary=True)
    output = np.zeros((*profile.search_canvas_yx, 3), dtype=np.uint8)
    py, px = profile.content_offset_yx
    table = _encode_table(profile)
    visited = peak = 0
    start, end = parameters.depth_range
    for z in range(max(oz, start), min(oz + nz - 1, end) + 1):
        for y in range(oy, oy + ny, tile_rows):
            checkpoint(cancel)
            rows = min(tile_rows, oy + ny - y)
            visited += 1
            peak = max(peak, rows * nx)
            if parameters.interpolation == "exact":
                values = source_view[z - oz, y - oy:y - oy + rows]
                membership = None if mask_view is None else mask_view[z - oz, y - oy:y - oy + rows]
            else:
                block_pull = pull.copy()
                block_pull[:3, 3] += pull[:3, :3] @ (z, y, ox)
                values = sample_affine_zyx(source, block_pull, (1, rows, nx), order=0,
                                           output_dtype=source.dtype.newbyteorder("="))[0]
                membership = None if mask is None else sample_affine_zyx(
                    mask, block_pull, (1, rows, nx), order=0, output_dtype=np.uint8)[0]
            if binary:
                intensity = values.astype(np.uint8) * np.uint8(255)
            elif parameters.scalar_range is not None:
                lo, hi = parameters.scalar_range
                intensity = np.rint(np.clip((values.astype(np.float64) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
            elif parameters.display_range is not None:
                lo, hi = parameters.display_range
                intensity = np.minimum(255, np.floor(np.maximum(values.astype(np.float64) - lo, 0)
                                       * (256.0 / (hi - lo + 1)) + 0.5)).astype(np.uint8)
            else:
                intensity = values.copy()
            if parameters.interpolation == "nearest" and parameters.scalar_range is not None:
                # Constant-zero samples outside the source are background even
                # when a signed/float conversion maps zero to a nonzero value.
                inside = np.ones((rows, nx), dtype=bool)
                yy, xx = np.arange(rows)[:, None], np.arange(nx)[None, :]
                for axis, length in enumerate(source.shape):
                    coordinate = block_pull[axis, 1] * yy + block_pull[axis, 2] * xx + block_pull[axis, 3]
                    inside &= (coordinate >= 0) & (coordinate <= length - 1)
                intensity[~inside] = 0
            intensity[intensity <= max(parameters.threshold, 1)] = 0
            if membership is not None:
                intensity[membership == 0] = 0
            if intensity.any():
                _reduce(output[y + py:y + py + rows, ox + px:ox + px + nx], table[z, intensity])
    checkpoint(cancel)
    # Detect persistent violations of the caller's read lease before completion.
    if not binary and source_hash != array_checksum(signal, cancel=cancel):
        raise ValueError("Source changed during CDM generation.")
    if mask is not None and mask_hash != array_checksum(mask, cancel=cancel, binary=True):
        raise ValueError("Mask changed during CDM generation.")
    pixels = output.tobytes()
    artifact = CDMArtifact(
        selection=selection, geometry=geometry, mapping=mapping, parameters=parameters,
        source_dtype=None if binary else signal.dtype.newbyteorder("<").str,
        source_pixel_sha256=source_hash, mask_selection=mask_selection,
        mask_revision=mask_revision, mask_dtype=None if mask is None else mask.dtype.newbyteorder("<").str,
        mask_pixel_sha256=mask_hash, rgb_pixel_sha256=hashlib.sha256(pixels).hexdigest(),
        warnings=tuple(dict.fromkeys(warnings)), numpy_version=np.__version__, scipy_version=scipy_version,
        schema_version=1 if profile == BRAIN else 2, profile=profile.profile_id, generator=profile.generator,
        template_id=profile.template_id, template_sha256=profile.template_sha256,
        template_shape_zyx=profile.template_shape_zyx, template_spacing_xyz_um=profile.spacing_xyz_um,
        template_working_origin_xyz_um=profile.origin_xyz_um, template_direction=profile.direction,
        **({} if profile == BRAIN else dict(search_canvas_shape_yx=profile.search_canvas_yx,
            content_offset_yx=profile.content_offset_yx, anatomical_area=profile.anatomical_area,
            alignment_space=profile.alignment_space)),
    )
    # Bytes-backed pixels cannot be made writeable by consumers.
    rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(output.shape)
    resources = CDMResources(
        input_bytes=(0 if binary else signal.nbytes) + (0 if mask is None else mask.nbytes),
        output_bytes=output.nbytes, peak_tile_voxels=peak, tiles_visited=visited,
        estimated_working_bytes=2 * output.nbytes + table.nbytes + 256 * peak + 262144,
    )
    return CDMResult(rgb, artifact, resources)


# Preserve the existing public API while callers adopt the profile-driven name.
generate_brain_cdm = generate_cdm
