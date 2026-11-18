"""Stitching fusion, completed-output verification and atomic publication."""
from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from madi3d_app.io_utils import NUMBERED_FILENAME_RESERVE, PUBLICATION_MARKER_SUFFIX, bounded_filename
from madi3d_app.project.scientific_records import ARTIFACT_COMPANION_SUFFIX
from madi3d_app.stitching.models import material_stitching_fusion_parameters
from madi3d_app.volume.geometry import (
    affine_support_bounds as _support_bounds,
    canonical_space_units,
    finite_tuple3,
    geometry_values_equivalent,
    grid_affine_from_components,
    grid_components_from_affine,
    invertible_affine4,
)
from madi3d_app.volume.probe import probe_volume_source

from . import common


_FUSION_MODES = {"distance_weighted", "mean", "max"}


_INTERPOLATION_MODES = {"nearest", "linear", "cubic"}


_OUTPUT_DTYPES = {"preserve", "float32"}


_OUTPUT_FORMATS = {"nrrd", "nifti", "tiff", "h5j"}


_SPACING_MODES = {"finest", "reference", "custom"}


_PRESERVABLE_SCALAR_DTYPES = frozenset(
    np.dtype(dtype)
    for dtype in (
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.float32,
        np.float64,
    )
)


_CUBIC_NEAREST_PAD = 12


_SPLINE_SLAB_BYTES = 8 * 1024**2


_LINEAR_SLAB_VOXELS = 4096


def _cast_output(array, dtype):
    dtype = np.dtype(dtype)
    x = np.asarray(array)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        x = np.clip(np.rint(x), limits.min, limits.max)
    return np.ascontiguousarray(x.astype(dtype, copy=False))


def _safe_stem(text):
    value = re.sub(r"[^\w.-]+", "_", str(text or "stitched")).strip("._")
    return value or "stitched"


def _validated_fusion_mode(value, subject="Stitching"):
    mode = "distance_weighted" if value in (None, "") else str(value)
    if mode not in _FUSION_MODES:
        raise ValueError(
            f'{subject} uses unsupported fusion mode "{mode}". '
            "Review the project and choose distance weighted, mean, or maximum fusion."
        )
    return mode


def _validate_fusion_options(options, subject="Stitching"):
    options = dict(options or {})
    _validated_fusion_mode(options.get("fusion_mode"), subject)
    interpolation = str(options.get("interpolation") or "linear").lower()
    if interpolation not in _INTERPOLATION_MODES:
        raise ValueError(
            f'{subject} uses unsupported interpolation "{interpolation}".'
        )
    output_dtype = str(options.get("output_dtype") or "preserve").lower()
    if output_dtype not in _OUTPUT_DTYPES:
        raise ValueError(
            f'{subject} uses unsupported output data type "{output_dtype}".'
        )
    output_format = str(options.get("output_format") or "nrrd").lower()
    if output_format not in _OUTPUT_FORMATS:
        raise ValueError(f'{subject} uses unsupported output format "{output_format}".')
    spacing_mode = str(options.get("spacing_mode") or "reference").lower()
    if spacing_mode not in _SPACING_MODES:
        raise ValueError(
            f'{subject} uses unsupported output spacing mode "{spacing_mode}".'
        )
    return {
        "fusion_mode": str(options.get("fusion_mode") or "distance_weighted"),
        "interpolation": interpolation,
        "output_dtype": output_dtype,
        "output_format": output_format,
        "spacing_mode": spacing_mode,
    }


def fusion_options_from_settings(settings, subject="Saved queued stitching job"):
    """Validate persisted open-ended settings at the fusion operation boundary."""
    settings = dict(settings or {})
    raw_custom = settings.get("custom_spacing")
    custom = [1.0, 1.0, 1.0] if raw_custom is None else list(raw_custom)

    def integer_setting(key, default, minimum):
        numeric = float(settings.get(key, default))
        if not math.isfinite(numeric):
            raise ValueError(f"{subject} setting {key!r} must be finite.")
        integer = int(numeric)
        if numeric != integer or integer < minimum:
            raise ValueError(
                f"{subject} setting {key!r} must be an integer of at least {minimum}."
            )
        return integer

    options = {
        "fusion_mode": _validated_fusion_mode(settings.get("fusion_mode"), subject),
        "output_format": str(settings.get("output_format") or "nrrd"),
        "interpolation": str(settings.get("interpolation") or "linear"),
        "output_dtype": str(settings.get("output_dtype") or "preserve"),
        "h5j_conversion_confirmed": bool(
            settings.get("h5j_conversion_confirmed", False)
        ),
        "spacing_mode": str(settings.get("spacing_mode") or "reference"),
        "custom_spacing": tuple(float(value) for value in custom),
        "padding": float(settings.get("padding", 0.0)),
        "chunk_depth": integer_setting("chunk_depth", 32, 1),
        "worker_count": integer_setting("worker_count", 0, 0),
    }
    options.update(_validate_fusion_options(options, subject))
    finite_tuple3(
        options["custom_spacing"],
        f"{subject} custom spacing",
        positive=True,
    )
    if not math.isfinite(options["padding"]) or options["padding"] < 0.0:
        raise ValueError(f"{subject} padding must be finite and non-negative.")
    return options


def _tile_support_corners_world(tile):
    dimensions = finite_tuple3(
        tile["dims"], "Stitching tile dimensions", positive=True, integer=True
    )
    corners = np.asarray(
        [
            [x, y, z, 1.0]
            for x in (-0.5, dimensions[0] - 0.5)
            for y in (-0.5, dimensions[1] - 0.5)
            for z in (-0.5, dimensions[2] - 0.5)
        ],
        dtype=float,
    )
    return (
        invertible_affine4(tile["world_affine"], "Stitching tile affine")
        @ corners.T
    ).T[:, :3]


def _output_grid(channel_tiles, spacing_mode, custom_spacing, padding):
    channel_tiles = common._prepared_tiles(channel_tiles)
    spacing_mode = str(spacing_mode or "reference").lower()
    if spacing_mode not in _SPACING_MODES:
        raise ValueError(
            f'Unsupported stitching output spacing mode "{spacing_mode}".'
        )
    reference_tile = next(
        (tile for tile in channel_tiles if bool(tile.get("anchor"))),
        channel_tiles[0],
    )
    lows = []
    highs = []
    spacings = []
    support_corners = []
    for tile in channel_tiles:
        lo, hi = _support_bounds(tile["world_affine"], tile["dims"])
        lows.append(lo)
        highs.append(hi)
        spacings.append(np.asarray(tile["spacing"], dtype=float))
        support_corners.append(_tile_support_corners_world(tile))
    padding = float(padding)
    if not math.isfinite(padding) or padding < 0.0:
        raise ValueError("Stitching output padding must be finite and greater than or equal to zero.")
    mins = np.min(np.vstack(lows), axis=0) - padding
    maxs = np.max(np.vstack(highs), axis=0) + padding

    if spacing_mode == "reference":
        reference_affine = invertible_affine4(
            reference_tile["world_affine"], "Reference stitching lattice"
        ).copy()
        spacing = np.linalg.norm(reference_affine[:3, :3], axis=0)
        reference_inverse = np.linalg.inv(reference_affine)
        world_corners = np.vstack(support_corners)
        homogeneous = np.column_stack(
            (world_corners, np.ones(len(world_corners), dtype=float))
        )
        reference_coordinates = (
            reference_inverse @ homogeneous.T
        ).T[:, :3]
        padding_steps = padding / spacing
        lattice_min = np.min(reference_coordinates, axis=0) - padding_steps
        lattice_max = np.max(reference_coordinates, axis=0) + padding_steps
        origin_index = lattice_min + 0.5
        affine = reference_affine.copy()
        affine[:3, 3] = (
            reference_affine @ np.r_[origin_index, 1.0]
        )[:3]
        raw_dims = lattice_max - lattice_min
        orientation_mode = "reference-lattice"
    elif spacing_mode == "custom":
        spacing = np.asarray(
            finite_tuple3(
                custom_spacing, "Custom output spacing", positive=True
            ),
            dtype=float,
        )
        origin = mins + 0.5 * spacing
        affine = grid_affine_from_components(
            origin, spacing, np.eye(3, dtype=float)
        )
        raw_dims = (maxs - mins) / spacing
        orientation_mode = "world-axis-aligned"
    else:
        spacing = np.min(np.vstack(spacings), axis=0)
        origin = mins + 0.5 * spacing
        affine = grid_affine_from_components(
            origin, spacing, np.eye(3, dtype=float)
        )
        raw_dims = (maxs - mins) / spacing
        orientation_mode = "world-axis-aligned"

    if not np.all(np.isfinite(raw_dims)) or np.any(raw_dims <= 0.0):
        raise ValueError("Validated stitching geometry produced invalid output dimensions.")
    rounding_tolerance = 1e-10 * np.maximum(1.0, np.abs(raw_dims))
    dims = tuple(int(v) for v in np.ceil(raw_dims - rounding_tolerance))
    origin = np.asarray(affine[:3, 3], dtype=float).copy()
    return {
        "affine": affine,
        "origin": origin,
        "spacing": spacing,
        "dims": dims,
        "support_min": mins,
        "support_max": maxs,
        "orientation_mode": orientation_mode,
        "reference_tile_id": str(reference_tile.get("tile_id") or ""),
        "coordinate_space_id": reference_tile.get("mosaic_coordinate_space_id"),
        "space_units": copy.deepcopy(reference_tile.get("space_units")),
        "geometry_status": copy.deepcopy(
            reference_tile.get("mosaic_geometry_status") or {}
        ),
    }


def _validate_fusion_resource_estimate(
    grid, dtype, output_dir, *, channel_count=1, simultaneous_copies=2,
    additional_working_bytes=0,
):
    dims = tuple(int(value) for value in grid["dims"])
    voxels = math.prod(dims) * max(1, int(channel_count))
    itemsize = int(np.dtype(dtype).itemsize)
    if voxels <= 0 or voxels > np.iinfo(np.intp).max // max(1, itemsize):
        raise ValueError(
            "The stitching output grid is too large for this platform's array index space."
        )
    output_bytes = voxels * itemsize
    required_bytes = (output_bytes * max(1, int(simultaneous_copies))
                      + max(0, int(additional_working_bytes)))
    try:
        free_bytes = int(shutil.disk_usage(Path(output_dir)).free)
    except Exception:
        free_bytes = 0
    if free_bytes > 0 and required_bytes > int(free_bytes * 0.90):
        raise RuntimeError(
            "The stitched output and its atomic working copy require approximately "
            f"{required_bytes / 1024**3:.2f} GiB, but only "
            f"{free_bytes / 1024**3:.2f} GiB is available."
        )
    return {
        "voxel_count": int(voxels),
        "output_bytes": int(output_bytes),
        "required_working_bytes": int(required_bytes),
        "available_disk_bytes": int(free_bytes),
    }


def _cubic_scratch_estimate(tiles):
    shapes = [tuple(int(n) + 2 * _CUBIC_NEAREST_PAD for n in tile["data"].shape)
              for tile in tiles]
    disk_bytes = sum(math.prod(shape) * 8 for shape in shapes)
    # Count the bounded in-place mapped slab, one padded source plane used
    # during staging, and its two explicit integer index vectors. Native filter
    # scratch remains unknown.
    working_bytes = max((
        min(_SPLINE_SLAB_BYTES, math.prod(shape) * np.dtype(np.float64).itemsize)
        + shape[1] * shape[2] * int(tile["data"].dtype.itemsize)
        + (shape[1] + shape[2]) * np.dtype(np.intp).itemsize
        for shape, tile in zip(shapes, tiles)
    ), default=0)
    return disk_bytes, working_bytes


def _fusion_channel_memory_estimate(channel_tiles, grid, options):
    """Return the canonical output and bounded chunk working-set estimate."""
    nx, ny, nz = (int(value) for value in grid["dims"])
    chunk_depth = max(1, int(options["chunk_depth"]))
    fusion_mode = _validated_fusion_mode(options["fusion_mode"])
    interpolation = str(options["interpolation"])
    output_dtype = _fusion_output_dtype(channel_tiles, options)
    working_dtype = _fusion_working_dtype(
        channel_tiles, fusion_mode=fusion_mode,
        interpolation=interpolation, output_dtype=output_dtype,
    )
    depth = min(chunk_depth, nz)
    chunk_voxels = max(1, depth * ny * nx)
    itemsize = int(working_dtype.itemsize)
    weight_bytes = itemsize if fusion_mode != "max" else 0
    vector_bytes = 8 * (depth + ny + nx)
    # Support retains inside + two Float64 coordinate fields; the distance
    # field, when requested, survives into subsequent axes.
    support_bytes = (
        chunk_voxels * (17 + (itemsize if fusion_mode == "distance_weighted" else 0))
        + vector_bytes
    )
    support_bytes += 4 * 4 * 8 + 3 * 8  # Mapping and source shape.
    if fusion_mode == "distance_weighted":
        support_bytes += 2 * 3 * 8  # Plane-normal lengths and distance scales.
    # Returned eligible/rejected masks, values and optional weights. Eligibility
    # construction additionally overlaps mask, finite_support and boolean temps.
    result_bytes = chunk_voxels * (2 + itemsize + weight_bytes)
    interpolation_bytes = chunk_voxels * (5 + itemsize + weight_bytes)
    if interpolation != "nearest":
        slab_depth = min(depth, max(1, _LINEAR_SLAB_VOXELS // (ny * nx)))
        slab_voxels = slab_depth * ny * nx
        source_itemsize = max(int(tile["data"].dtype.itemsize) for tile in channel_tiles)
        # Linear fallback: three intp lower indices, three Float64 fractions,
        # three intp selected indices, Float64 values/weight/product, and four
        # boolean masks (valid/consumed/finite/used), plus gathered source values.
        slab_bytes = slab_voxels * (
            6 * np.dtype(np.intp).itemsize + 6 * 8 + 4 + source_itemsize
        )
        # Broadcast axes, slab affine and output-to-source mapping.
        slab_bytes += 8 * (slab_depth + ny + nx) + 2 * 4 * 4 * 8
        interpolation_bytes = max(interpolation_bytes, result_bytes + slab_bytes)
    bytes_per_job = max(support_bytes, interpolation_bytes, result_bytes)
    worker_resolution = {}
    worker_count = common._resolved_worker_count(
        options.get("worker_count", 0), len(channel_tiles), auto_cap=8,
        estimated_bytes_per_job=bytes_per_job,
        diagnostics=worker_resolution,
        memory_reserved=bool(options.get("memory_reserved")),
    )
    coefficient_disk, coefficient_working = (
        _cubic_scratch_estimate(channel_tiles)
        if interpolation == "cubic" else (0, 0)
    )
    # Shared accumulators remain live while all workers produce/retain results.
    # Accumulation uses one product array, or first/overlap masks for maximum.
    if fusion_mode == "max":
        accumulator_bytes = chunk_voxels * (itemsize + 2)
    elif fusion_mode == "distance_weighted":
        accumulator_bytes = chunk_voxels * (3 * itemsize + 5)
    else:
        accumulator_bytes = chunk_voxels * (2 * itemsize + 1)
    accumulation_bytes = chunk_voxels * (2 if fusion_mode == "max" else itemsize)
    # Final output overlaps accumulators and support masks. Integer output
    # additionally holds rint, clip and converted arrays in _cast_output.
    finish_masks = 6 if fusion_mode == "distance_weighted" else 2 if fusion_mode == "max" else 3
    finish_bytes = chunk_voxels * (itemsize + finish_masks)
    if np.issubdtype(output_dtype, np.integer):
        rounded_itemsize = np.rint(np.empty(0, dtype=working_dtype)).dtype.itemsize
        finish_bytes += chunk_voxels * max(
            2 * rounded_itemsize, rounded_itemsize + output_dtype.itemsize
        )
    else:
        finish_bytes += chunk_voxels * (output_dtype.itemsize if output_dtype != working_dtype else 0)
    return {
        "output_bytes": math.prod((nx, ny, nz)) * int(output_dtype.itemsize),
        "estimated_bytes_per_job": bytes_per_job,
        "resolved_worker_count": worker_count,
        "worker_resolution": worker_resolution,
        "minimum_chunk_working_bytes": max(
            accumulator_bytes + max(
                bytes_per_job * worker_count,
                bytes_per_job * (worker_count - 1) + result_bytes + accumulation_bytes,
                finish_bytes,
            ), coefficient_working,
        ),
        "cubic_coefficient_disk_bytes": coefficient_disk,
        "cubic_preparation_working_bytes": coefficient_working,
    }


def _fusion_memory_estimate(channel_sets, options):
    """Estimate detached inputs, resolved output grids and canonical worker peaks."""
    memory_reserved = bool((options or {}).get("memory_reserved"))
    options = fusion_options_from_settings(options, "Stitching fusion")
    options["memory_reserved"] = memory_reserved
    input_bytes = common._registration_admission_bytes(
        tile for channel in channel_sets for tile in channel["tiles"]
    )
    output_bytes = 0
    working_bytes = 0
    grids = []
    channel_output_dtypes = []
    for channel in channel_sets:
        tiles = common._prepared_tiles(channel["tiles"])
        grid = _output_grid(
            tiles, options["spacing_mode"], options["custom_spacing"],
            options["padding"],
        )
        estimate = _fusion_channel_memory_estimate(tiles, grid, options)
        output_bytes += int(estimate["output_bytes"])
        grids.append(grid)
        channel_output_dtypes.append(_fusion_output_dtype(tiles, options))
        working_bytes = max(
            working_bytes, int(estimate["minimum_chunk_working_bytes"])
        )
    bundle_formats = {"nrrd", "tiff", "h5j"}
    if (
        len(grids) > 1
        and options["output_format"] in bundle_formats
        and all(tuple(grid["dims"]) == tuple(grids[0]["dims"]) for grid in grids[1:])
        and all(np.allclose(grid["affine"], grids[0]["affine"], atol=1e-8, rtol=1e-8)
                for grid in grids[1:])
    ):
        common_dtype = np.dtype(np.result_type(*channel_output_dtypes))
        voxels = math.prod(int(value) for value in grids[0]["dims"])
        output_bytes = voxels * len(grids) * int(common_dtype.itemsize)
    # The bundle writer runs after every channel kernel has returned and its
    # workers/scratch have retired. The NRRD/TIFF writers stream the combined
    # memmap in bounded (<=1 MiB) slabs, so no writer-sized private heap copy
    # is reserved; output_bytes remains the distinct disk-backed map footprint.
    future_bytes = output_bytes + working_bytes
    return {
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "working_bytes": working_bytes,
        "future_bytes": future_bytes,
        "memory_bytes": input_bytes + future_bytes,
    }


def _filter_spline_scratch(path, shape, *, cancelled):
    """Filter complete lines in bounded slabs; unmap each slab's touched pages."""
    from scipy import ndimage

    for axis in range(3):
        split_axis = max((a for a in range(3) if a != axis), key=lambda a: shape[a])
        plane_bytes = math.prod(shape[a] for a in range(3) if a != split_axis) * 8
        depth = max(1, _SPLINE_SLAB_BYTES // max(1, plane_bytes))
        for start in range(0, shape[split_axis], depth):
            if cancelled():
                raise InterruptedError("Stitching fusion was cancelled.")
            slab = [slice(None)] * 3
            slab[split_axis] = slice(start, min(shape[split_axis], start + depth))
            mapped = np.memmap(path, mode="r+", dtype=np.float64, shape=shape)
            try:
                view = mapped[tuple(slab)]
                ndimage.spline_filter1d(view, order=3, axis=axis, output=view,
                                       mode="nearest")
                mapped.flush()
            finally:
                view = None
                mapped._mmap.close()


def _prepare_cubic_source(tile, coefficient_path, cancelled):
    """Prepare SciPy's nearest-boundary coefficients once for this operation."""
    source = np.asarray(tile["data"])
    pad = _CUBIC_NEAREST_PAD
    shape = tuple(int(n) + 2 * pad for n in source.shape)
    mapped = np.memmap(coefficient_path, mode="w+", dtype=np.float64, shape=shape)
    mapped._mmap.close()
    invalid_samples = 0
    yi = np.clip(np.arange(shape[1]) - pad, 0, source.shape[1] - 1)
    xi = np.clip(np.arange(shape[2]) - pad, 0, source.shape[2] - 1)
    depth = max(1, _SPLINE_SLAB_BYTES // max(1, shape[1] * shape[2] * 8))
    for start in range(0, shape[0], depth):
        mapped = np.memmap(coefficient_path, mode="r+", dtype=np.float64, shape=shape)
        try:
            for z in range(start, min(shape[0], start + depth)):
                if cancelled():
                    raise InterruptedError("Stitching fusion was cancelled.")
                plane = source[min(max(z - pad, 0), source.shape[0] - 1)]
                if pad <= z < shape[0] - pad:
                    invalid_samples += int(np.count_nonzero(~np.isfinite(plane)))
                mapped[z] = plane[np.ix_(yi, xi)]
            mapped.flush()
        finally:
            mapped._mmap.close()
    if invalid_samples:
        # Spline coefficients depend on complete source lines. A local mask
        # cannot contain NaN/Inf contamination without inventing missing values.
        # Retain finite local support through explicit linear sampling instead.
        coefficient_path.unlink()
        return None, np.asarray(tile["world_affine"]), invalid_samples
    _filter_spline_scratch(coefficient_path, shape, cancelled=cancelled)
    affine = np.asarray(tile["world_affine"], dtype=float).copy()
    affine[:3, 3] -= affine[:3, :3] @ np.full(3, pad)
    return shape, affine, invalid_samples


def _finite_linear_resample_slab(source, source_affine, output_affine, shape):
    """Evaluate one bounded trilinear slab using Float64 accumulation."""
    mapping = common._output_to_source_zyx_mapping(source_affine, output_affine)
    axes = [np.arange(n, dtype=np.float64).reshape(
        tuple(n if a == axis else 1 for a in range(3)))
        for axis, n in enumerate(shape)]
    indices, fractions = [], []
    for axis in range(3):
        coordinate = (mapping[axis, 3] + mapping[axis, 0] * axes[0]
                      + mapping[axis, 1] * axes[1] + mapping[axis, 2] * axes[2])
        np.clip(coordinate, 0, source.shape[axis] - 1, out=coordinate)
        lower = coordinate.astype(np.intp)
        indices.append(lower)
        fractions.append(coordinate - lower)
        del coordinate
    values = np.zeros(shape, dtype=np.float64)
    valid = np.ones(shape, dtype=bool)
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                corner = (z, y, x)
                weight = np.ones(shape, dtype=np.float64)
                for axis, upper in enumerate(corner):
                    weight *= fractions[axis] if upper else 1 - fractions[axis]
                selected = tuple(np.minimum(indices[a] + corner[a], source.shape[a] - 1)
                                 for a in range(3))
                samples = source[selected]
                consumed = weight > 0
                finite = np.isfinite(samples)
                valid &= ~consumed | finite
                used = consumed & finite
                product = np.empty(shape, dtype=np.float64)
                np.multiply(samples, weight, out=product, where=used)
                np.add(values, product, out=values, where=used)
                del selected, samples, consumed, finite, used, weight, product
    return values, valid


def _finite_linear_resample(source, source_affine, output_affine, shape, dtype,
                            *, cancelled=lambda: False):
    """Trilinear nearest-boundary sampling with bounded nonfinite scratch."""
    values = np.empty(shape, dtype=dtype)
    valid = np.empty(shape, dtype=bool)
    slab_depth = max(1, _LINEAR_SLAB_VOXELS // (shape[1] * shape[2]))
    for start in range(0, shape[0], slab_depth):
        if cancelled():
            raise InterruptedError("Stitching fusion was cancelled.")
        stop = min(shape[0], start + slab_depth)
        affine = np.asarray(output_affine, dtype=float).copy()
        affine[:3, 3] += affine[:3, 2] * start
        slab_values, slab_valid = _finite_linear_resample_slab(
            source, source_affine, affine, (stop - start, *shape[1:]))
        values[start:stop] = slab_values
        valid[start:stop] = slab_valid
        del slab_values, slab_valid
    return values, valid


def _tile_chunk_contribution(
    tile,
    *,
    fusion_mode,
    interpolation,
    chunk_affine,
    chunk_shape,
    working_dtype,
    cubic_source=None,
    cancelled=lambda: False,
):
    """Resample one tile contribution. Safe to execute in a worker thread."""
    if cancelled():
        raise InterruptedError("Stitching fusion was cancelled.")

    mask, distance = common._tile_physical_support(
        tile,
        chunk_affine,
        chunk_shape,
        with_distance=(fusion_mode == "distance_weighted"),
        distance_dtype=working_dtype,
    )
    if not np.any(mask):
        return None
    if fusion_mode == "distance_weighted":
        weights = np.asarray(distance, dtype=working_dtype)
    elif fusion_mode == "mean":
        weights = mask.astype(working_dtype)
    else:
        weights = None

    if cancelled():
        raise InterruptedError("Stitching fusion was cancelled.")

    interpolation = str(interpolation or "linear").lower()
    value_order = 0 if interpolation == "nearest" else 3 if interpolation == "cubic" else 1
    source, source_affine = tile["data"], tile["world_affine"]
    coefficient_map = None
    if cubic_source is not None:
        path, shape, source_affine = cubic_source
        if shape is None:
            value_order = 1
        else:
            coefficient_map = np.memmap(path, mode="r", dtype=np.float64, shape=shape)
            source = coefficient_map
    try:
        if value_order == 1:
            values = common._resample_zyx(source, source_affine, chunk_affine, chunk_shape,
                                   order=1, output_dtype=working_dtype, mode="nearest")
            finite_support = np.isfinite(values)
            if np.any(mask & ~finite_support):
                # Keep the native finite-only fast path. Recompute contaminated
                # chunks so a zero-weight missing neighbor cannot reject a pixel.
                del values, finite_support
                values, finite_support = _finite_linear_resample(
                    source, source_affine, chunk_affine, chunk_shape, working_dtype,
                    cancelled=cancelled)
        else:
            values = common._resample_zyx(
                source, source_affine, chunk_affine, chunk_shape,
                order=value_order, output_dtype=working_dtype, mode="nearest",
                prefilter=False if cubic_source is not None else None,
            )
            finite_support = np.isfinite(values)
    finally:
        if coefficient_map is not None:
            coefficient_map._mmap.close()
    eligible = mask & finite_support & np.isfinite(values)
    invalid_support = mask & ~eligible
    if weights is not None:
        weights[~eligible] = 0
    return eligible, values, weights, invalid_support


def _hidden_temporary_output_path(path, purpose="partial"):
    """Return a unique sibling path that retains the scientific file suffix."""
    path = Path(path)
    suffix = ".nii.gz" if path.name.lower().endswith(".nii.gz") else path.suffix
    return path.with_name(f".{purpose}-{uuid.uuid4().hex}{suffix}")


def _flush_completed_file(path):
    """Ask the OS to flush a writer-closed file before validation/publication."""
    with open(path, "r+b") as stream:
        os.fsync(stream.fileno())


def _validate_stitching_output(
    path,
    *,
    output_format,
    expected_channel_count,
    expected_dimensions,
    expected_dtype,
    expected_grid,
    expected_space_units,
    subject,
):
    """Validate completed scientific output from container metadata."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(
            f"{subject} writer did not create a complete file: {path}"
        )

    probe = probe_volume_source(path)
    diagnostics = tuple(probe.errors) + tuple(probe.ambiguities)
    physical_diagnostics = tuple(probe.physical_geometry_diagnostics)
    if expected_space_units is None:
        physical_diagnostics = tuple(
            item
            for item in physical_diagnostics
            if not (
                "unit" in str(item).lower()
                and not any(
                    token in str(item).lower()
                    for token in ("spacing", "origin", "direction", "dimension")
                )
            )
        )
    diagnostics += physical_diagnostics
    if probe.requires_axis_resolution:
        diagnostics += ("the output axis contract requires manual resolution",)
    missing_fields = tuple(probe.missing_fields)
    if expected_space_units is None:
        missing_fields = tuple(
            field for field in missing_fields if "unit" not in str(field).lower()
        )
    if missing_fields:
        diagnostics += (
            "missing authoritative fields: " + ", ".join(missing_fields),
        )
    if diagnostics:
        raise RuntimeError(
            f"The completed {subject.lower()} failed header validation: "
            + "; ".join(str(item) for item in diagnostics)
        )

    expected_format = str(output_format or "").lower()
    compatible_formats = {
        "nrrd": {"nrrd", "nhdr"},
        "nifti": {"nifti"},
        "tiff": {"tiff", "imagej-tiff", "ome-tiff"},
        "h5j": {"h5j"},
    }
    if probe.container_format not in compatible_formats.get(expected_format, set()):
        raise RuntimeError(
            f"The completed {subject.lower()} has container type "
            f"{probe.container_format!r}, expected {expected_format!r}."
        )
    if int(probe.channel_count) != int(expected_channel_count):
        raise RuntimeError(
            f"The completed {subject.lower()} has "
            f"{probe.channel_count} channel(s), expected {expected_channel_count}."
        )
    if int(probe.time_count) != 1:
        raise RuntimeError(
            f"The completed {subject.lower()} has {probe.time_count} time points; "
            "stitching output must be one spatial volume."
        )
    expected_dimensions = tuple(int(value) for value in expected_dimensions)
    if tuple(probe.dimensions) != expected_dimensions:
        raise RuntimeError(
            f"The completed {subject.lower()} has dimensions "
            f"{tuple(probe.dimensions)}, expected {expected_dimensions}."
        )

    written_dtype = np.dtype(probe.scalar_dtype).name if probe.scalar_dtype else ""
    required_dtype = np.dtype(expected_dtype).name
    if written_dtype != required_dtype:
        raise RuntimeError(
            f"The completed {subject.lower()} has scalar type "
            f"{written_dtype or 'unknown'!r}, expected {required_dtype!r}."
        )

    expected_components = grid_components_from_affine(expected_grid["affine"])
    physical_fields = (
        ("spacing", probe.spacing, expected_components["spacing"]),
        ("origin", probe.origin, expected_components["origin"]),
        ("direction", probe.direction, expected_components["direction"]),
    )
    for name, actual, expected in physical_fields:
        if actual is None or not geometry_values_equivalent(actual, expected):
            raise RuntimeError(
                f"The completed {subject.lower()} has a mismatched "
                f"physical-grid {name}."
            )
    if expected_space_units is not None and canonical_space_units(
        probe.space_units
    ) != canonical_space_units(expected_space_units):
        raise RuntimeError(
            f"The completed {subject.lower()} has mismatched physical units."
        )
    return probe


def _validate_stitching_bundle_output(path, **expected):
    return _validate_stitching_output(
        path, subject="Multichannel stitching output", **expected
    )


def _validate_stitching_scalar_output(
    path,
    *,
    output_format,
    expected_dimensions,
    expected_dtype,
    expected_grid,
    expected_space_units,
):
    return _validate_stitching_output(
        path,
        output_format=output_format,
        expected_channel_count=1,
        expected_dimensions=expected_dimensions,
        expected_dtype=expected_dtype,
        expected_grid=expected_grid,
        expected_space_units=expected_space_units,
        subject="Scalar stitching output",
    )


def _atomic_write_json(path, payload, unique_path):
    """Write, parse-check, and atomically publish one JSON document."""
    requested_path = Path(path)
    temporary = _hidden_temporary_output_path(requested_path, "manifest")
    try:
        with open(temporary, "x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        parsed = json.loads(temporary.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise RuntimeError("Stitching project manifest must contain a JSON object.")
        final_path = unique_path(requested_path)
        os.replace(temporary, final_path)
        return final_path
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except Exception:
            pass


def _fusion_output_dtype(channel_tiles, options):
    _validate_fusion_options(options)
    source_dtype = np.result_type(*[np.dtype(tile["dtype"]) for tile in channel_tiles])
    if str(options.get("output_dtype", "preserve")) == "float32":
        return np.dtype(np.float32)
    if source_dtype not in _PRESERVABLE_SCALAR_DTYPES:
        source_types = ", ".join(
            sorted({np.dtype(tile["dtype"]).name for tile in channel_tiles})
        )
        raise ValueError(
            "Preserve source data type cannot represent the selected stitching "
            f"channels without conversion (source types: {source_types}; common "
            f"type: {source_dtype.name}). Choose an explicit Float32 conversion."
        )
    return source_dtype


def _fusion_bundle_output_dtype(channel_sets, options):
    """Resolve one exact bundle dtype without weakening a preserve request."""
    channel_dtypes = [
        _fusion_output_dtype(channel["tiles"], options) for channel in channel_sets
    ]
    common_dtype = np.dtype(np.result_type(*channel_dtypes))
    if str(options.get("output_dtype", "preserve")) == "float32":
        return np.dtype(np.float32)
    if common_dtype not in _PRESERVABLE_SCALAR_DTYPES:
        source_types = ", ".join(sorted({dtype.name for dtype in channel_dtypes}))
        raise ValueError(
            "Preserve source data type cannot represent all selected stitching "
            f"channels in one multichannel output without conversion (channel "
            f"types: {source_types}; common type: {common_dtype.name}). Choose "
            "an explicit Float32 conversion or a scalar output format."
        )
    return common_dtype


def _fusion_working_dtype(channel_tiles, *, fusion_mode, interpolation, output_dtype):
    """Return the numerical dtype used before the single final output cast.

    Nearest-neighbour maximum fusion performs no arithmetic and therefore keeps
    the exact output scalar type. Interpolation and overlap arithmetic use
    Float64 whenever Float32 cannot exactly carry the supported source contract.
    Float32 is sufficient for Float32 and <=16-bit integer inputs; the worker cap
    bounds exact integer accumulation for the latter below Float32's 24-bit
    significand before the final division.
    """
    source_dtype = np.result_type(*[np.dtype(tile["dtype"]) for tile in channel_tiles])
    output_dtype = np.dtype(output_dtype)
    if fusion_mode == "max" and interpolation == "nearest":
        return output_dtype
    if source_dtype in {np.dtype(np.int32), np.dtype(np.uint32), np.dtype(np.float64)}:
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def _fuse_one_channel(
    channel_tiles,
    output_path,
    options,
    progress_callback,
    cancelled,
    *,
    output_array,
):
    """Compute into the caller-owned array; the operation owns serialization."""
    validated_options = _validate_fusion_options(options)
    channel_tiles = common._prepared_tiles(channel_tiles)
    grid = _output_grid(
        channel_tiles,
        options["spacing_mode"],
        options["custom_spacing"],
        options["padding"],
    )
    nx, ny, nz = grid["dims"]
    chunk_depth = max(1, int(options["chunk_depth"]))
    fusion_mode = _validated_fusion_mode(validated_options["fusion_mode"])
    interpolation = validated_options["interpolation"]
    output_dtype = _fusion_output_dtype(channel_tiles, options)
    working_dtype = _fusion_working_dtype(
        channel_tiles,
        fusion_mode=fusion_mode,
        interpolation=interpolation,
        output_dtype=output_dtype,
    )
    grid["resource_estimate"] = _validate_fusion_resource_estimate(
        grid,
        output_dtype,
        Path(output_path).parent,
        simultaneous_copies=1,
    )

    tile_bounds = []
    for tile in channel_tiles:
        lo, hi = _support_bounds(tile["world_affine"], tile["dims"])
        tile_bounds.append((lo, hi))

    total_chunks = max(1, math.ceil(nz / chunk_depth))
    memory_estimate = _fusion_channel_memory_estimate(
        channel_tiles, grid, options,
    )
    minimum_chunk_working_bytes = memory_estimate["minimum_chunk_working_bytes"]
    coefficient_disk_bytes = memory_estimate["cubic_coefficient_disk_bytes"]
    coefficient_working_bytes = memory_estimate["cubic_preparation_working_bytes"]
    chunk_voxels = max(1, int(chunk_depth) * int(ny) * int(nx))
    estimated_bytes_per_job = memory_estimate["estimated_bytes_per_job"]
    available_memory = common._available_memory_bytes()
    memory_reserved = bool(options.get("memory_reserved"))
    if (not memory_reserved and available_memory > 0
            and coefficient_working_bytes > available_memory):
        raise RuntimeError("One cubic coefficient preparation slab exceeds the stitching memory budget. Free memory or choose linear interpolation.")
    if (
        not memory_reserved and available_memory > 0
        and minimum_chunk_working_bytes > available_memory
    ):
        raise RuntimeError(
            "One stitching fusion chunk requires approximately "
            f"{minimum_chunk_working_bytes / 1024**3:.2f} GiB, but only "
            f"{available_memory / 1024**3:.2f} GiB of physical memory is available. "
            "Reduce the fusion chunk depth."
        )
    grid["resource_estimate"].update(
        {
            "minimum_chunk_working_bytes": int(minimum_chunk_working_bytes),
            "available_memory_bytes": int(available_memory),
            "cubic_coefficient_disk_bytes": int(coefficient_disk_bytes),
            "cubic_preparation_working_bytes": int(coefficient_working_bytes),
        }
    )
    worker_count = memory_estimate["resolved_worker_count"]
    worker_resolution = memory_estimate["worker_resolution"]
    grid["resolved_worker_count"] = int(worker_count)
    grid["worker_resolution"] = worker_resolution
    grid["resource_estimate"]["worker_resolution"] = copy.deepcopy(
        worker_resolution
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if coefficient_disk_bytes:
        estimate = grid["resource_estimate"]
        combined_disk_bytes = estimate["required_working_bytes"] + coefficient_disk_bytes
        if estimate["available_disk_bytes"] > 0 and combined_disk_bytes > int(estimate["available_disk_bytes"] * 0.90):
            raise RuntimeError("Stitching output and cubic coefficient staging exceed available disk space. Free space or choose linear interpolation.")
        estimate["required_working_bytes"] = combined_disk_bytes
    executor = None
    output_map = np.asanyarray(output_array)
    if tuple(output_map.shape) != (int(nz), int(ny), int(nx)):
        raise RuntimeError(f"Fusion output shape {output_map.shape} does not match grid {(nz, ny, nx)}.")
    cubic_sources = {}
    coefficient_paths = []
    sample_validity = {"policy": "finite-nonzero-footprint-v1",
                       "excluded_contributions": 0, "unsupported_voxels": 0,
                       "invalid_support_voxels": 0,
                       "cubic_linear_fallback_sources": []}
    grid["sample_validity"] = sample_validity
    try:
        if interpolation == "cubic":
            for tile in channel_tiles:
                path = _hidden_temporary_output_path(output_path.with_suffix(".dat"), "spline")
                coefficient_paths.append(path)
                shape, affine, invalid_count = _prepare_cubic_source(tile, path, cancelled)
                cubic_sources[id(tile)] = (path, shape, affine)
                if invalid_count:
                    sample_validity["cubic_linear_fallback_sources"].append({
                        "tile_id": str(tile.get("tile_id") or ""),
                        "nonfinite_samples": invalid_count, "effective_interpolation": "linear",
                    })
        if worker_count > 1:
            executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="MADI3D-Stitch-Fuse",
            )

        with common._limit_native_threadpools(worker_count > 1):
            for chunk_index, z0 in enumerate(range(0, nz, chunk_depth), 1):
                if cancelled():
                    raise InterruptedError("Stitching fusion was cancelled.")
                z1 = min(nz, z0 + chunk_depth)
                depth = z1 - z0
                chunk_affine = np.asarray(grid["affine"], dtype=float).copy()
                chunk_affine[:3, 3] += chunk_affine[:3, 2] * z0
                chunk_shape = (depth, ny, nx)
                invalid_support = np.zeros(chunk_shape, dtype=bool)

                if fusion_mode == "max":
                    accumulator = np.zeros(chunk_shape, dtype=working_dtype)
                    valid_any = np.zeros(chunk_shape, dtype=bool)
                else:
                    weighted_sum = np.zeros(chunk_shape, dtype=working_dtype)
                    weight_sum = np.zeros(chunk_shape, dtype=working_dtype)
                    if fusion_mode == "distance_weighted":
                        value_sum = np.zeros(chunk_shape, dtype=working_dtype)
                        coverage_count = np.zeros(chunk_shape, dtype=np.uint32)

                chunk_support_lo, chunk_support_hi = _support_bounds(
                    chunk_affine,
                    (nx, ny, depth),
                )

                active_tiles = sorted(
                    (
                        tile
                        for tile, (tile_lo, tile_hi) in zip(channel_tiles, tile_bounds)
                        if not (
                            np.any(tile_hi <= chunk_support_lo)
                            or np.any(tile_lo >= chunk_support_hi)
                        )
                    ),
                    key=lambda tile: (
                        str(tile.get("tile_id") or ""),
                        str(tile.get("display_name") or ""),
                    ),
                )

                def contribution(tile):
                    return _tile_chunk_contribution(
                        tile,
                        fusion_mode=fusion_mode,
                        interpolation=interpolation,
                        chunk_affine=chunk_affine,
                        chunk_shape=chunk_shape,
                        working_dtype=working_dtype,
                        cubic_source=cubic_sources.get(id(tile)),
                        cancelled=cancelled,
                    )

                def accumulate(result):
                    if result is None:
                        return
                    mask, values, weights, rejected = result
                    invalid_support[:] |= rejected
                    sample_validity["excluded_contributions"] += int(np.count_nonzero(rejected))
                    if fusion_mode == "max":
                        first = mask & ~valid_any
                        overlap = mask & valid_any
                        np.copyto(accumulator, values, where=first)
                        np.maximum(accumulator, values, out=accumulator, where=overlap)
                        valid_any[:] |= mask
                    else:
                        product = np.empty(chunk_shape, dtype=working_dtype)
                        np.multiply(values, weights, out=product, where=mask)
                        np.add(weighted_sum, product, out=weighted_sum, where=mask)
                        np.add(weight_sum, weights, out=weight_sum, where=mask)
                        if fusion_mode == "distance_weighted":
                            np.add(value_sum, values, out=value_sum, where=mask)
                            coverage_count[:] += mask

                if executor is not None and len(active_tiles) > 1:
                    for batch_start in range(0, len(active_tiles), worker_count):
                        batch = active_tiles[batch_start : batch_start + worker_count]
                        futures = [executor.submit(contribution, tile) for tile in batch]
                        try:
                            for future in futures:
                                if cancelled():
                                    for pending in futures:
                                        pending.cancel()
                                    raise InterruptedError("Stitching fusion was cancelled.")
                                accumulate(future.result())
                        except Exception:
                            for pending in futures:
                                pending.cancel()
                            raise
                        finally:
                            future = None
                            del futures
                else:
                    for tile in active_tiles:
                        accumulate(contribution(tile))

                if fusion_mode == "max":
                    out = np.zeros(chunk_shape, dtype=working_dtype)
                    np.copyto(out, accumulator, where=valid_any)
                    supported = valid_any
                else:
                    out = np.zeros(chunk_shape, dtype=working_dtype)
                    if fusion_mode == "distance_weighted":
                        supported = coverage_count > 0
                        single = coverage_count == 1
                        np.copyto(out, value_sum, where=single)
                        weighted = (coverage_count > 1) & (weight_sum > 0.0)
                        np.divide(weighted_sum, weight_sum, out=out, where=weighted)
                        # Samples can lie exactly on multiple support faces, so
                        # every physical interior distance is legitimately zero.
                        # Their unweighted mean is deterministic and avoids an
                        # arbitrary macroscopic blend-width epsilon.
                        zero_weight_overlap = (coverage_count > 1) & ~weighted
                        np.divide(value_sum, coverage_count, out=out, where=zero_weight_overlap)
                    else:
                        valid = weight_sum > 0.0
                        supported = valid
                        np.divide(weighted_sum, weight_sum, out=out, where=valid)

                sample_validity["unsupported_voxels"] += int(np.count_nonzero(~supported))
                sample_validity["invalid_support_voxels"] += int(np.count_nonzero(invalid_support & ~supported))

                output_map[z0:z1] = _cast_output(out, output_dtype)
                progress_callback(chunk_index, total_chunks, worker_count)
                del out, supported, invalid_support
                if fusion_mode == "max":
                    del accumulator, valid_any
                else:
                    del weighted_sum, weight_sum
                    if fusion_mode == "distance_weighted":
                        del value_sum, coverage_count, single, weighted, zero_weight_overlap
                    else:
                        del valid

        if hasattr(output_map, "flush"):
            output_map.flush()
        if cancelled():
            raise InterruptedError("Stitching fusion was cancelled.")
        producing_operation = (options.get("scientific_provenance") or {}).get("producing_operation")
        if producing_operation is not None:
            producing_operation.setdefault("evidence", {}).setdefault("sample_validity", {})[
                str(options.get("channel_label", ""))
            ] = copy.deepcopy(sample_validity)
            if sample_validity["excluded_contributions"] or sample_validity["cubic_linear_fallback_sources"]:
                producing_operation["qc_status"] = "warning"
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for path in coefficient_paths:
            path.unlink(missing_ok=True)
    return grid


def _fusion_sample_warning(grid, label):
    validity = grid.get("sample_validity") or {}
    if not (validity.get("excluded_contributions") or validity.get("cubic_linear_fallback_sources")):
        return None
    text = (f"{label}: non-finite interpolation support was excluded; "
            f"{validity['invalid_support_voxels']} output voxels have no finite support and are zero-filled.")
    if validity.get("cubic_linear_fallback_sources"):
        text += " Affected cubic sources use linear interpolation."
    return text


class StitchFusionOperation:
    """GUI-independent fusion operation driven by explicit callbacks."""

    def __init__(
        self, channel_sets, output_dir, base_name, options, project_payload,
        writer_callback=None, bundle_writer_callback=None,
        ffmpeg_executable=None, supporting_operations=(), *, progress_callback=None, cancelled=None,
        completed_callback=None, failed_callback=None,
    ):
        self.channel_sets = channel_sets
        self.output_dir = str(output_dir)
        self.base_name = str(base_name or "stitched")
        self.options = dict(options)
        self.project_payload = dict(project_payload)
        self.writer_callback = writer_callback
        self.bundle_writer_callback = bundle_writer_callback
        self.ffmpeg_executable = os.fspath(ffmpeg_executable) if ffmpeg_executable else None
        self.supporting_operations = tuple(
            copy.deepcopy(record) for record in supporting_operations
        )
        self._progress_callback = progress_callback or (lambda _value, _text: None)
        self._cancelled = cancelled or (lambda: False)
        self._completed_callback = completed_callback or (lambda _result: None)
        self._failed_callback = failed_callback or (lambda _message: None)

    @staticmethod
    def _automatic_output_path(directory, stem, suffix, *, companion=True):
        # Reserve the collision counter and the longest sibling of the artifact.
        reserve = NUMBERED_FILENAME_RESERVE
        if companion:
            reserve += ARTIFACT_COMPANION_SUFFIX + PUBLICATION_MARKER_SUFFIX
        physical_stem = _safe_stem(stem)
        if re.search(r"(?:_stitched)+$", physical_stem, re.IGNORECASE):
            physical_stem = re.sub(r"(?:_stitched)+$", "", physical_stem, flags=re.IGNORECASE)
            suffix = "_stitched" + suffix
        name = bounded_filename(physical_stem, suffix, reserve=reserve,
                                directory=directory, hash_source=stem)
        return directory / name

    @staticmethod
    def _unique_output_path(path):
        path = Path(path)
        def available(candidate):
            companion = candidate.with_name(candidate.name + ARTIFACT_COMPANION_SUFFIX)
            siblings = (candidate,)
            if not candidate.name.endswith(".json"):
                siblings += (companion, companion.with_name(companion.name + PUBLICATION_MARKER_SUFFIX))
            return all(not sibling.exists() and not sibling.is_symlink() for sibling in siblings)

        if available(path):
            return path
        if path.name.lower().endswith(".nii.gz"):
            stem, ext = path.name[:-7], ".nii.gz"
        else:
            stem, ext = path.stem, path.suffix
        counter = 2
        while True:
            candidate = path.with_name(f"{stem}_{counter}{ext}")
            if available(candidate):
                return candidate
            counter += 1

    def _bundle_grid(self, channel_sets=None):
        channel_sets = self.channel_sets if channel_sets is None else channel_sets
        grids = [
            _output_grid(
                channel["tiles"],
                self.options["spacing_mode"],
                self.options["custom_spacing"],
                self.options["padding"],
            )
            for channel in channel_sets
        ]
        if not grids:
            return None
        first = grids[0]
        for grid in grids[1:]:
            if tuple(grid["dims"]) != tuple(first["dims"]):
                return None
            if not np.allclose(np.asarray(grid["affine"]), np.asarray(first["affine"]), atol=1e-8, rtol=1e-8):
                return None
        return first

    def run(self):
        combined_map = None
        combined_temp = None
        bundle_temporary = None
        scalar_temporaries = []
        published_outputs = []
        published_manifest = None
        transaction_committed = False
        sample_warnings = []
        scalar_stages = []
        try:
            self.memory_reserved = bool(self.options.get("memory_reserved"))
            self.options = fusion_options_from_settings(
                self.options, "Stitching fusion"
            )
            if self.writer_callback is None:
                raise RuntimeError(
                    "Stitching fusion requires MADI3D's canonical injected volume writer."
                )
            tile_counts = [
                len(channel.get("tiles", [])) for channel in self.channel_sets
            ]
            preparation = common._prepare_stitching_geometry(
                (
                    tile
                    for channel in self.channel_sets
                    for tile in channel.get("tiles", [])
                ),
                mosaic_coordinate_space_id=self.project_payload.get(
                    "mosaic_coordinate_space_id"
                ),
            )
            prepared_channel_sets = []
            offset = 0
            for channel, tile_count in zip(self.channel_sets, tile_counts):
                prepared_channel = dict(channel)
                prepared_channel["tiles"] = preparation.prepared_tiles[
                    offset : offset + tile_count
                ]
                prepared_channel_sets.append(prepared_channel)
                offset += tile_count
            self.channel_sets = prepared_channel_sets
            mosaic_provenance = preparation.provenance.to_dict()
            from .models import stitching_fusion_operation_record
            fusion_record = stitching_fusion_operation_record(
                self.project_payload, self.options, mosaic_provenance,
                "stitching-fusion:" + str(uuid.uuid4()),
            )
            raw_operation_inputs = [
                channel.get("scientific_operation_inputs")
                for channel in self.channel_sets
            ]
            scientific_inputs = []
            if raw_operation_inputs and all(raw_operation_inputs):
                scientific_inputs = list({
                    id(capture): capture for capture in raw_operation_inputs
                }.values())
                fusion_record["input_record_ids"] = list(dict.fromkeys(
                    reference["record_id"]
                    for capture in scientific_inputs
                    for reference in capture.get("references", ())
                ))
                captured_producers = {
                    capture["direct_records"][reference["record_id"]].get(
                        "producing_operation_id"
                    )
                    for capture in scientific_inputs
                    for reference in capture.get("references", ())
                }
                fusion_record["input_operation_ids"] = [
                    operation_id
                    for operation_id in fusion_record.get("input_operation_ids", ())
                    if operation_id not in captured_producers
                ]
            fusion_id = fusion_record["operation_id"]
            fusion_record["output_grids"] = {}
            for channel in self.channel_sets:
                output_grid = _output_grid(
                    channel["tiles"], self.options["spacing_mode"],
                    self.options["custom_spacing"], self.options["padding"],
                )
                fusion_record["output_grids"][str(channel["label"])] = {
                    "dims_xyz": list(output_grid["dims"]),
                    "world_index_affine": np.asarray(output_grid["affine"]).tolist(),
                    "output_dtype": str(np.dtype(np.uint8) if self.options["output_format"] == "h5j"
                                        else _fusion_output_dtype(channel["tiles"], self.options)),
                }
            self.options["scientific_provenance"] = {
                "producing_operation_id": fusion_id,
                "producing_operation": fusion_record,
                "supporting_operations": copy.deepcopy(self.supporting_operations),
            }
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            outputs = []
            channel_count = max(1, len(self.channel_sets))
            output_format = str(self.options.get("output_format", "nrrd"))
            extension = {
                "nrrd": ".nrrd",
                "nifti": ".nii.gz",
                "tiff": ".tif",
                "h5j": ".h5j",
            }.get(output_format, ".nrrd")

            # Source/bundle output: keep co-registered channels in one physical
            # container whenever both the format and the MADI3D writer support it.
            bundle_grid = self._bundle_grid() if channel_count > 1 else None
            bundle_mode = bool(
                channel_count > 1
                and output_format in {"nrrd", "tiff", "h5j"}
                and self.bundle_writer_callback is not None
                and bundle_grid is not None
            )

            if bundle_mode:
                nx, ny, nz = (int(v) for v in bundle_grid["dims"])
                common_dtype = _fusion_bundle_output_dtype(
                    self.channel_sets, self.options
                )
                bundle_grid["resource_estimate"] = _validate_fusion_resource_estimate(
                    bundle_grid,
                    common_dtype,
                    out_dir,
                    channel_count=channel_count,
                    simultaneous_copies=2,
                )
                combined_temp = _hidden_temporary_output_path(out_dir / "scratch.dat", "bundle")
                combined_map = np.memmap(
                    combined_temp, mode="w+", dtype=common_dtype,
                    shape=(channel_count, nz, ny, nx), order="C",
                )
                resolved_workers = []
                worker_resolutions = []
                for channel_index, channel in enumerate(self.channel_sets, 1):
                    if self._cancelled():
                        raise InterruptedError("Stitching fusion was cancelled.")
                    label = str(channel["label"])
                    local_options = dict(self.options)
                    local_options["memory_reserved"] = self.memory_reserved
                    local_options["channel_label"] = label
                    local_options["stitching_mosaic_geometry"] = (
                        mosaic_provenance
                    )

                    def update(chunk_index, total_chunks, worker_count, ci=channel_index, lab=label):
                        fraction = (ci - 1 + chunk_index / max(1, total_chunks)) / channel_count
                        self._progress_callback(
                            int(round(96.0 * fraction)),
                            f"Fusing {lab}: chunk {chunk_index}/{total_chunks} with {worker_count} CPU worker(s)",
                        )

                    grid = _fuse_one_channel(
                        channel["tiles"],
                        combined_temp,
                        local_options, update, self._cancelled,
                        output_array=combined_map[channel_index - 1],
                    )
                    if warning := _fusion_sample_warning(grid, label):
                        sample_warnings.append(warning)
                    resolved_workers.append(int(grid.get("resolved_worker_count", 1)))
                    worker_resolutions.append(
                        copy.deepcopy(grid.get("worker_resolution") or {})
                    )

                bundle_grid["resource_estimate"][
                    "channel_worker_resolutions"
                ] = worker_resolutions

                combined_map.flush()
                requested_path = self._automatic_output_path(out_dir, self.base_name, extension)
                bundle_temporary = _hidden_temporary_output_path(
                    requested_path, "multichannel"
                )
                labels = [
                    str(channel.get("output_name") or channel["label"])
                    for channel in self.channel_sets
                ]
                self._progress_callback(97, f"Writing multichannel {output_format.upper()} source")
                bundle_writer_kwargs = {}
                if output_format.lower() == "h5j":
                    bundle_writer_kwargs = {
                        "ffmpeg_executable": self.ffmpeg_executable,
                        "cancel_check": self._cancelled,
                    }
                final_path = self._unique_output_path(requested_path)
                bundle_options = dict(self.options)
                inputs = [channel.get("scientific_inputs") for channel in self.channel_sets]
                if all(inputs):
                    from madi3d_app.volume.provenance import volume_result_projection
                    bundle_options["portable_projection"] = volume_result_projection(inputs,
                        [{"display_name": label} for label in labels], [fusion_record]*len(labels),
                        supporting_operations=self.supporting_operations)
                bundle_options["stitching_mosaic_geometry"] = mosaic_provenance
                bundle_writer_result = self.bundle_writer_callback(
                    str(bundle_temporary), combined_map, output_format, bundle_grid,
                    labels, bundle_options,
                    bundle_grid.get("space_units"),
                    **bundle_writer_kwargs,
                )
                bundle_warning = (
                    str(bundle_writer_result[2])
                    if isinstance(bundle_writer_result, (tuple, list))
                    and len(bundle_writer_result) >= 3
                    and bundle_writer_result[2]
                    else None
                )
                if self._cancelled():
                    raise InterruptedError("Stitching fusion was cancelled.")
                _flush_completed_file(bundle_temporary)
                _validate_stitching_bundle_output(
                    bundle_temporary,
                    output_format=output_format,
                    expected_channel_count=channel_count,
                    expected_dimensions=bundle_grid["dims"],
                    expected_dtype=(
                        np.uint8 if output_format.lower() == "h5j" else common_dtype
                    ),
                    expected_grid=bundle_grid,
                    expected_space_units=bundle_grid.get("space_units"),
                )
                if self._cancelled():
                    raise InterruptedError("Stitching fusion was cancelled.")
                final_path = self._unique_output_path(requested_path)
                from madi3d_app.volume.export import publish_staged_volume_pair
                publish_staged_volume_pair(bundle_temporary, final_path, cancel_check=self._cancelled)
                final_companion = Path(str(final_path)+".madi3d-provenance.json")
                if final_companion.is_file():
                    published_outputs.append(final_companion)
                bundle_temporary = None
                published_outputs.append(final_path)
                outputs.append({
                    "path": str(final_path),
                    "label": "multichannel",
                    "display_name": self.base_name + (
                        final_path.name[len(requested_path.name) - len(extension):-len(extension)]
                    ),
                    "channels": labels,
                    "multichannel": True,
                    "grid": {
                        "dims": list(bundle_grid["dims"]),
                        "origin": np.asarray(bundle_grid["origin"], dtype=float).tolist(),
                        "spacing": np.asarray(bundle_grid["spacing"], dtype=float).tolist(),
                        "affine": common._matrix_to_json(bundle_grid["affine"]),
                        "orientation_mode": bundle_grid.get("orientation_mode"),
                        "reference_tile_id": bundle_grid.get("reference_tile_id"),
                        "resolved_worker_count": max(resolved_workers or [1]),
                        "resource_estimate": copy.deepcopy(
                            bundle_grid.get("resource_estimate") or {}
                        ),
                        "coordinate_space_id": bundle_grid.get(
                            "coordinate_space_id"
                        ),
                        "space_units": copy.deepcopy(
                            bundle_grid.get("space_units")
                        ),
                        "geometry_status": copy.deepcopy(
                            bundle_grid.get("geometry_status") or {}
                        ),
                    },
                    "geometry_provenance": preparation.provenance.to_dict(),
                    **({"provenance_warning": bundle_warning} if bundle_warning else {}),
                })
            else:
                planned_grids = [
                    _output_grid(channel["tiles"], self.options["spacing_mode"],
                                 self.options["custom_spacing"], self.options["padding"])
                    for channel in self.channel_sets
                ]
                scalar_bytes = [math.prod(grid["dims"]) * _fusion_output_dtype(channel["tiles"], self.options).itemsize
                                for channel, grid in zip(self.channel_sets, planned_grids)]
                retained_bytes = sum(scalar_bytes)
                for channel, grid, output_bytes in zip(self.channel_sets, planned_grids, scalar_bytes):
                    spline_bytes = (_cubic_scratch_estimate(channel["tiles"])[0]
                                    if self.options["interpolation"] == "cubic" else 0)
                    _validate_fusion_resource_estimate(
                        grid, _fusion_output_dtype(channel["tiles"], self.options), out_dir,
                        additional_working_bytes=retained_bytes - output_bytes + spline_bytes,
                    )
                for channel_index, channel in enumerate(self.channel_sets, 1):
                    if self._cancelled():
                        raise InterruptedError("Stitching fusion was cancelled.")
                    label = str(channel["label"])
                    output_name = str(channel.get("output_name") or "").strip()
                    if output_name:
                        output_stem = output_name
                    else:
                        suffix = "" if channel_count == 1 else "_" + label
                        output_stem = self.base_name + suffix
                    requested_path = self._automatic_output_path(out_dir, output_stem, extension)
                    staged_path = _hidden_temporary_output_path(
                        requested_path, "scalar"
                    )
                    scalar_temporaries.append(staged_path)
                    local_options = dict(self.options)
                    local_options["memory_reserved"] = self.memory_reserved
                    local_options["channel_label"] = label
                    local_options["stitching_mosaic_geometry"] = (
                        mosaic_provenance
                    )

                    def update(chunk_index, total_chunks, worker_count, ci=channel_index, lab=label):
                        fraction = (ci - 1 + chunk_index / max(1, total_chunks)) / channel_count
                        self._progress_callback(
                            int(round(100.0 * fraction)),
                            f"Fusing {lab}: chunk {chunk_index}/{total_chunks} with {worker_count} CPU worker(s)",
                        )

                    planned_grid = planned_grids[channel_index - 1]
                    shape = tuple(reversed(planned_grid["dims"]))
                    scratch_path = _hidden_temporary_output_path(out_dir / "scratch.dat", "scalar")
                    scalar_temporaries.append(scratch_path)
                    scalar_map = np.memmap(scratch_path, mode="w+",
                                           dtype=_fusion_output_dtype(channel["tiles"], local_options),
                                           shape=shape)
                    try:
                        grid = _fuse_one_channel(
                            channel["tiles"], staged_path, local_options, update, self._cancelled,
                            output_array=scalar_map,
                        )
                        scalar_map.flush()
                    finally:
                        scalar_map._mmap.close()
                    grid.setdefault("resource_estimate", {})["retained_scalar_staging_bytes"] = retained_bytes
                    if warning := _fusion_sample_warning(grid, label):
                        sample_warnings.append(warning)
                    scalar_stages.append((channel, grid, local_options, requested_path, staged_path,
                                          output_name, label, scratch_path, shape))

                # All channels share one completed operation record. Freeze its
                # numerical evidence before projecting it into any scalar file.
                inputs = [member.get("scientific_inputs") for member in self.channel_sets]
                for (channel, grid, local_options, requested_path, staged_path,
                     output_name, label, scratch_path, shape) in scalar_stages:
                    if self._cancelled():
                        raise InterruptedError("Stitching fusion was cancelled.")
                    path = self._unique_output_path(requested_path)
                    output_display_name = output_name
                    if output_name and path != requested_path:
                        output_display_name += path.name[len(requested_path.name) - len(extension):-len(extension)]
                    local_options["channel_label"] = output_display_name or label
                    if all(inputs):
                        from madi3d_app.volume.provenance import volume_result_projection
                        local_options["portable_projection"] = volume_result_projection(
                            inputs, [{"display_name": output_display_name or label}], [fusion_record],
                            supporting_operations=self.supporting_operations)
                    scalar_map = np.memmap(scratch_path, mode="r",
                                           dtype=_fusion_output_dtype(channel["tiles"], local_options), shape=shape)
                    try:
                        writer_kwargs = ({"ffmpeg_executable": self.ffmpeg_executable,
                                          "cancel_check": self._cancelled}
                                         if output_format == "h5j" else {})
                        writer_result = self.writer_callback(
                            str(staged_path), scalar_map, output_format, grid,
                            local_options["channel_label"], local_options,
                            channel["tiles"][0].get("space_units"), **writer_kwargs)
                        if isinstance(writer_result, (tuple, list)) and len(writer_result) >= 3 and writer_result[2]:
                            grid["provenance_warning"] = str(writer_result[2])
                    finally:
                        scalar_map._mmap.close()
                    scratch_path.unlink(missing_ok=True)
                    if self._cancelled():
                        raise InterruptedError("Stitching fusion was cancelled.")
                    _flush_completed_file(staged_path)
                    _validate_stitching_scalar_output(
                        staged_path,
                        output_format=output_format,
                        expected_dimensions=grid["dims"],
                        expected_dtype=(
                            np.uint8
                            if output_format.lower() == "h5j"
                            else _fusion_output_dtype(channel["tiles"], self.options)
                        ),
                        expected_grid=grid,
                        expected_space_units=grid.get("space_units"),
                    )
                    if self._cancelled():
                        raise InterruptedError("Stitching fusion was cancelled.")
                    final_path = self._unique_output_path(requested_path)
                    if output_name and final_path != path:
                        raise RuntimeError(
                            "Stitching output filename changed during fusion. Retry the output."
                        )
                    path = final_path
                    from madi3d_app.volume.export import publish_staged_volume_pair
                    publish_staged_volume_pair(staged_path, path, cancel_check=self._cancelled)
                    final_companion = Path(str(path)+".madi3d-provenance.json")
                    if final_companion.is_file():
                        published_outputs.append(final_companion)
                    scalar_temporaries.remove(staged_path)
                    published_outputs.append(path)
                    outputs.append({
                        "path": str(path),
                        "label": label,
                        **({"display_name": output_display_name} if output_name else {}),
                        "multichannel": False,
                        "grid": {
                            "dims": list(grid["dims"]),
                            "origin": grid["origin"].tolist(),
                            "spacing": grid["spacing"].tolist(),
                            "affine": common._matrix_to_json(grid["affine"]),
                            "orientation_mode": grid.get("orientation_mode"),
                            "reference_tile_id": grid.get("reference_tile_id"),
                            "resolved_worker_count": int(grid.get("resolved_worker_count", 1)),
                            "resource_estimate": copy.deepcopy(
                                grid.get("resource_estimate") or {}
                            ),
                            "coordinate_space_id": grid.get(
                                "coordinate_space_id"
                            ),
                            "space_units": copy.deepcopy(
                                grid.get("space_units")
                            ),
                            "geometry_status": copy.deepcopy(
                                grid.get("geometry_status") or {}
                            ),
                        },
                        "geometry_provenance": preparation.provenance.to_dict(),
                        **(
                            {"provenance_warning": grid["provenance_warning"]}
                            if grid.get("provenance_warning")
                            else {}
                        ),
                    })

            payload = dict(self.project_payload)
            payload["outputs"] = [
                {
                    key: copy.deepcopy(entry[key])
                    for key in ("path", "label", "channels", "multichannel")
                    if key in entry
                }
                | {
                    "grid": {
                        key: copy.deepcopy(entry["grid"][key])
                        for key in (
                            "dims", "origin", "spacing", "affine",
                            "orientation_mode", "reference_tile_id",
                            "coordinate_space_id", "space_units",
                            "geometry_status",
                        )
                        if key in entry["grid"]
                    }
                }
                for entry in outputs
            ]
            payload["fusion_parameters"] = material_stitching_fusion_parameters(
                self.options
            )
            payload.pop("mosaic_coordinate_space_id", None)
            payload["operation_id"] = fusion_id
            payload["execution_status"] = "completed"
            payload["mosaic_geometry"] = preparation.provenance.to_dict()
            completed_with_warnings = bool(
                preparation.warnings
                or preparation.assumptions
                or sample_warnings
            )
            payload["completed_with_warnings"] = completed_with_warnings
            if self._cancelled():
                raise InterruptedError("Stitching fusion was cancelled.")
            published_manifest = _atomic_write_json(
                self._automatic_output_path(
                    out_dir, self.base_name, "_stitching_project.json", companion=False,
                ),
                payload,
                self._unique_output_path,
            )
            project_path = published_manifest
            transaction_committed = True
            self._progress_callback(
                100,
                "Fusion completed with warnings"
                if completed_with_warnings
                else "Fusion complete",
            )
            self._completed_callback(
                {
                    "output_committed": True,
                    "outputs": outputs,
                    "project_path": str(project_path),
                    "operation_record": fusion_record,
                    "scientific_inputs": scientific_inputs,
                    "completed_with_warnings": completed_with_warnings,
                    "warnings": [*preparation.warnings, *sample_warnings],
                    "assumptions": preparation.assumptions,
                    "mosaic_geometry": preparation.provenance.to_dict(),
                }
            )
        except InterruptedError:
            return
        except Exception:
            self._failed_callback(traceback.format_exc())
        finally:
            if not transaction_committed:
                for path in reversed(published_outputs):
                    try:
                        if Path(path).is_file():
                            Path(path).unlink()
                    except Exception:
                        pass
                if published_manifest is not None:
                    try:
                        if Path(published_manifest).is_file():
                            Path(published_manifest).unlink()
                    except Exception:
                        pass
            if bundle_temporary is not None:
                try:
                    Path(str(bundle_temporary)+".madi3d-provenance.json").unlink(missing_ok=True)
                    if Path(bundle_temporary).exists():
                        Path(bundle_temporary).unlink()
                except Exception:
                    pass
            for scalar_temporary in scalar_temporaries:
                try:
                    Path(str(scalar_temporary)+".madi3d-provenance.json").unlink(missing_ok=True)
                    if Path(scalar_temporary).exists():
                        Path(scalar_temporary).unlink()
                except Exception:
                    pass
            if combined_map is not None:
                try:
                    combined_map.flush()
                except Exception:
                    pass
                try:
                    mmap_obj = getattr(combined_map, "_mmap", None)
                    if mmap_obj is not None:
                        mmap_obj.close()
                except Exception:
                    pass
                combined_map = None
            if combined_temp is not None:
                try:
                    if Path(combined_temp).exists():
                        Path(combined_temp).unlink()
                except Exception:
                    pass
            self.ffmpeg_executable = None
