"""Pure stitching registration, fusion, validation, and operation services."""
from __future__ import annotations

STITCHING_ALGORITHM_VERSION = (
    "stitching-registration-fusion-v6-conservative-model-reference-lattice"
)

DEFAULT_STITCHING_MINIMUM_NCC = 0.05
DEFAULT_STITCHING_MAX_ANGLE_DEG = 3.0
DEFAULT_STITCHING_MAX_SCALE_PERCENT = 5.0
DEFAULT_STITCHING_MAX_SHEAR = 0.03
MINIMUM_ADVANCED_MODEL_NCC_GAIN = 0.02
TRANSLATION_STITCHING_ACCEPTANCE_CONTRACT = {
    "contract_version": "translation-stitching-v1",
    "supported_fixture_accuracy_registration_voxels": 0.5,
    "default_minimum_pair_ncc": DEFAULT_STITCHING_MINIMUM_NCC,
    "pair_acceptance": (
        "Phase-correlation confidence must be numerically valid and final pair "
        "NCC must meet the configured minimum. Ambiguous but valid phase peaks "
        "remain reviewable rather than being treated as universally invalid."
    ),
    "global_inconsistency": (
        "Only cycle-redundant translation edges whose registration-grid residual "
        "is a unique outlier above max(0.5 voxel, median + 3 normalized MAD) "
        "are rejected. Rejection is bounded to eight edges and cannot disconnect "
        "the accepted graph; non-identifiable disagreement remains explicit."
    ),
    "preprocessing": (
        "Masked 1st-to-99.5th percentile normalization, Gaussian background "
        "subtraction, and renormalization. FFT phase-correlation inputs receive a "
        "cosine distance taper at the support boundary while NCC remains untapered. "
        "The deterministic boundary audit reduced mean supported-fixture translation "
        "error relative to a hard mask while retaining the 0.5-voxel worst-case "
        "accuracy contract."
    ),
    "scope": (
        "The 0.5-voxel value is an algorithmic synthetic-fixture accuracy and "
        "global-consistency threshold, not a universal microscopy or biological "
        "quality threshold."
    ),
}

import copy
import hashlib
import json
import math
import os
import re
import shutil
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from madi3d_app.stitching.models import (
    StitchingMosaicGeometry,
    StitchingRegistrationResult,
    StitchingRejection,
    stitching_grid_revision,
)
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

# -----------------------------------------------------------------------------
# Parallel execution helpers
# -----------------------------------------------------------------------------


def _cpu_count():
    return max(1, int(os.cpu_count() or 1))


def _available_memory_bytes():
    """Best-effort available physical memory without adding a hard dependency."""
    try:
        import psutil
        return max(0, int(psutil.virtual_memory().available))
    except Exception:
        pass

    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return max(0, int(status.ullAvailPhys))
        except Exception:
            pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return max(0, page_size * available_pages)
    except Exception:
        return 0


def _resolved_worker_count(
    requested,
    job_count=None,
    *,
    auto_cap=8,
    estimated_bytes_per_job=0,
    diagnostics=None,
):
    """Resolve requested maximum concurrency through CPU and memory limits."""
    try:
        requested = int(requested)
    except Exception:
        requested = 0

    cpu = _cpu_count()
    reductions = []
    if requested > 0:
        workers = min(requested, cpu, 32)
        if workers < requested:
            reductions.append(
                f"requested {requested}, limited to {workers} by available logical CPUs and the 32-worker cap"
            )
    else:
        workers = min(max(1, cpu - 1), max(1, int(auto_cap)))
    workers = max(1, int(workers))
    if job_count is not None:
        job_limit = max(1, int(job_count))
        if workers > job_limit:
            workers = job_limit
            reductions.append(f"limited to {workers} by the number of jobs")

    estimated = max(0, int(estimated_bytes_per_job or 0))
    available = _available_memory_bytes()
    memory_budget = int(available * 0.25) if available > 0 else 0
    memory_limited = False
    if estimated > 0 and memory_budget > 0:
        if estimated > memory_budget:
            raise RuntimeError(
                "One stitching worker is estimated to require approximately "
                f"{estimated / 1024**3:.2f} GiB of temporary memory, but the "
                "conservative stitching budget is only "
                f"{memory_budget / 1024**3:.2f} GiB from "
                f"{available / 1024**3:.2f} GiB currently available. Reduce the "
                "registration size or fusion chunk depth, close other applications, "
                "or free memory before retrying."
            )
        safe_workers = max(1, memory_budget // estimated)
        if workers > safe_workers:
            workers = int(safe_workers)
            memory_limited = True
            reductions.append(
                f"memory safety reduced concurrency to {workers} worker(s)"
            )

    workers = max(1, int(workers))
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "requested_workers": int(requested),
                "requested_mode": "explicit" if requested > 0 else "auto",
                "logical_cpu_count": int(cpu),
                "job_count": None if job_count is None else int(job_count),
                "estimated_bytes_per_job": int(estimated),
                "available_memory_bytes": int(available),
                "memory_budget_bytes": int(memory_budget),
                "memory_limited": bool(memory_limited),
                "resolved_workers": int(workers),
                "reductions": reductions,
            }
        )
    return workers


@contextmanager
def _limit_native_threadpools(enabled):
    """Prevent BLAS/FFT backends from multiplying each Python worker again."""
    if not enabled:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        yield
        return
    with threadpool_limits(limits=1):
        yield


# -----------------------------------------------------------------------------
# Geometry and array helpers
# -----------------------------------------------------------------------------

_XYZ_ZYX_PERMUTATION = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def _stitching_tile_label(tile):
    label = tile.get("display_name") or tile.get("tile_id") or tile.get("name") or "unnamed tile"
    return str(label).replace('"', "'")


_UNIT_POWER10_METERS = {
    "nm": -9,
    "micron": -6,
    "mm": -3,
    "cm": -2,
    "m": 0,
}


@dataclass
class StitchingGeometryPreparation:
    """Runtime tiles plus the complete serializable mosaic-space record."""

    prepared_tiles: list[dict]
    normalized_working_affines: list[np.ndarray]
    provenance: StitchingMosaicGeometry

    @property
    def warnings(self):
        return list(self.provenance.warnings)

    @property
    def assumptions(self):
        return list(self.provenance.assumptions)

    @property
    def unit_conversions(self):
        return list(self.provenance.unit_conversions)

    @property
    def output_geometry_status(self):
        return dict(self.provenance.output_geometry_status)


def _json_geometry_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_geometry_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_geometry_value(item) for item in value]
    return copy.deepcopy(value)


def _tile_source_units(tile, descriptor, working_geometry):
    if isinstance(working_geometry, dict) and "physical_units" in working_geometry:
        return copy.deepcopy(working_geometry.get("physical_units"))
    descriptor_geometry = descriptor.get("working_geometry")
    if isinstance(descriptor_geometry, dict) and "physical_units" in descriptor_geometry:
        return copy.deepcopy(descriptor_geometry.get("physical_units"))
    return copy.deepcopy(tile.get("space_units"))


def _operation_working_geometry(
    working_geometry, *, dims, local_affine, actor_matrix, world_affine
):
    """Return geometry provenance whose pose-dependent fields match this job."""
    geometry = copy.deepcopy(dict(working_geometry or {}))
    geometry["dimensions"] = [int(value) for value in dims]
    source_affine = local_affine if local_affine is not None else world_affine
    components = grid_components_from_affine(source_affine)
    geometry.update(
        {
            "spacing": [float(value) for value in components["spacing"]],
            "origin": [float(value) for value in components["origin"]],
            "direction": np.asarray(
                components["direction"], dtype=float
            ).tolist(),
            "local_index_to_working_affine": _matrix_to_json(source_affine),
        }
    )
    if actor_matrix is not None:
        geometry["pose"] = _matrix_to_json(actor_matrix)
        geometry["index_to_world_affine"] = _matrix_to_json(world_affine)
        support_min, support_max = _support_bounds(world_affine, dims)
        geometry["effective_support"] = [
            float(support_min[0]),
            float(support_max[0]),
            float(support_min[1]),
            float(support_max[1]),
            float(support_min[2]),
            float(support_max[2]),
        ]
    return geometry


def _validated_source_working_geometry(
    tile,
    descriptor,
    *,
    subject,
    dims,
    local_affine,
    actor_matrix,
    world_affine,
):
    """Validate a frozen channel grid and bind it to the operation pose."""
    frozen_revision = str(
        descriptor.get("channel_local_geometry_revision") or ""
    ).strip()
    current_revision = str(
        tile.get("channel_local_geometry_revision")
        or tile.get("geometry_revision")
        or ""
    ).strip()
    legacy_revision = str(descriptor.get("geometry_revision") or "").strip()
    frozen_source_checksum = str(
        descriptor.get("source_checksum") or ""
    ).strip()
    current_source_checksum = str(tile.get("source_checksum") or "").strip()

    if (
        frozen_source_checksum
        and current_source_checksum
        and frozen_source_checksum != current_source_checksum
    ):
        raise ValueError(
            f"{subject} source data checksum changed after it was added to the "
            "stitching project. Refresh the captured project source before "
            "continuing."
        )

    if frozen_revision:
        if not current_revision:
            raise ValueError(
                f"{subject} cannot verify its captured channel-local grid against "
                "the current MADI3D source. Refresh this source in the stitching "
                "project before continuing."
            )
        if current_revision != frozen_revision:
            raise ValueError(
                f"{subject} captured channel-local grid is stale because its "
                "numerical geometry changed after it was added to the stitching "
                "project. Refresh the captured project geometry and recalculate "
                "registration before continuing."
            )
    elif legacy_revision and current_revision:
        raise ValueError(
            f"{subject} uses a legacy stitching geometry revision that combines "
            "grid and pose and cannot be compared safely with the current channel "
            "grid. Refresh this source in the stitching project before continuing."
        )

    frozen_grid = descriptor.get("channel_local_working_grid")
    current_grid = tile.get("channel_local_working_grid") or tile.get(
        "working_grid"
    )
    if frozen_revision and frozen_grid:
        if not isinstance(current_grid, dict) or not current_grid:
            raise ValueError(
                f"{subject} has no current channel-local working grid to verify "
                "against its captured stitching descriptor."
            )
        if stitching_grid_revision(frozen_grid) != stitching_grid_revision(
            current_grid
        ):
            raise ValueError(
                f"{subject} captured grid payload disagrees with the current "
                "channel-local grid despite its recorded revision. Refresh the "
                "stitching project source before continuing."
            )

    runtime_geometry = tile.get("working_geometry")
    if frozen_revision and not runtime_geometry:
        raise ValueError(
            f"{subject} has no current runtime working geometry for this operation."
        )
    base_geometry = (
        runtime_geometry
        or descriptor.get("working_geometry")
        or current_grid
        or frozen_grid
        or {}
    )
    geometry = _operation_working_geometry(
        base_geometry,
        dims=dims,
        local_affine=local_affine,
        actor_matrix=actor_matrix,
        world_affine=world_affine,
    )
    revision = (
        current_revision
        or frozen_revision
        or legacy_revision
        or stitching_grid_revision(current_grid or frozen_grid or geometry)
    )
    return geometry, revision


def _canonical_units_or_none(raw_units):
    try:
        return canonical_space_units(raw_units), "recognized"
    except (TypeError, ValueError):
        missing = raw_units is None or raw_units == "" or raw_units == [] or raw_units == ()
        return None, "missing" if missing else "unsupported"


def _stable_mosaic_coordinate_space_id(source_records, chosen_output_unit):
    identity_payload = {
        "chosen_output_unit": chosen_output_unit,
        "sources": [
            {
                "tile_id": record["tile_id"],
                "source_id": record["source_id"],
                "channel_id": record["channel_id"],
                "geometry_revision": record["geometry_revision"],
                "source_coordinate_space_id": record[
                    "source_coordinate_space_id"
                ],
                "source_working_affine": record["source_working_affine"],
                "initial_pose": record["initial_pose"],
                "unit_conversion": record["unit_conversion"],
            }
            for record in source_records
        ],
    }
    encoded = json.dumps(
        identity_payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "madi3d:mosaic:sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_source_tile_ownership(tiles, subject="Stitching project"):
    """Require one logical SourceID to map to exactly one pose-graph tile."""
    ownership = {}
    labels = {}
    for index, raw_tile in enumerate(tiles):
        tile = dict(raw_tile or {})
        descriptor = dict(tile.get("source_descriptor") or {})
        source_id = str(
            descriptor.get("source_id") or tile.get("source_id") or ""
        ).strip()
        if not source_id:
            continue
        tile_id = str(tile.get("tile_id") or index)
        ownership.setdefault(source_id, set()).add(tile_id)
        labels[tile_id] = _stitching_tile_label(tile)
    conflicts = {
        source_id: sorted(tile_ids)
        for source_id, tile_ids in ownership.items()
        if len(tile_ids) > 1
    }
    if conflicts:
        details = []
        for source_id, tile_ids in sorted(conflicts.items()):
            described_tiles = ", ".join(
                f"{labels.get(tile_id, tile_id)!r} ({tile_id})" for tile_id in tile_ids
            )
            details.append(f"SourceID {source_id!r} is split across {described_tiles}")
        raise ValueError(
            f"{subject} is invalid: "
            + "; ".join(details)
            + ". All sibling channels from one acquisition must remain in one "
            "stitching tile so that the source has one authoritative solved pose."
        )
    return ownership


def validate_selection_source_ownership(tiles, subject="Stitching project"):
    records = [
        {
            "tile_id": tile.tile_id,
            "display_name": tile.display_name,
            "source_descriptor": descriptor,
        }
        for tile in tiles
        for descriptor in tile.channel_descriptors.values()
    ]
    return _validate_source_tile_ownership(records, subject=subject)


def _prepare_stitching_geometry(
    tiles,
    *,
    mosaic_coordinate_space_id=None,
    chosen_output_unit=None,
    require_data=True,
):
    """Prepare arbitrary sources in one explicit numerical mosaic space.

    Physical unit conversion is exact for recognized units. Missing or unsupported
    units keep their current numeric working grid and remain explicitly uncertain.
    Source coordinate-space identities are retained only as provenance; the result
    always owns a new job-local mosaic identity.
    """

    tiles = list(tiles)
    if not tiles:
        raise ValueError("Stitching requires at least one three-dimensional tile.")
    _validate_source_tile_ownership(tiles)

    inspected = []
    recognized_units = []
    for tile_index, tile in enumerate(tiles):
        tile = dict(tile or {})
        label = _stitching_tile_label(tile)
        subject = f'Tile "{label}"'
        dims = finite_tuple3(
            tile.get("dims"), f"{subject} dimensions", positive=True, integer=True
        )
        data = tile.get("data")
        if data is None:
            if require_data:
                raise ValueError(f"{subject} has no scalar image data.")
            data_shape = None
            data_dtype = (
                str(np.dtype(tile["dtype"]))
                if tile.get("dtype") is not None
                else None
            )
        else:
            array = np.asanyarray(data)
            data_shape = tuple(int(value) for value in array.shape)
            data_dtype = str(array.dtype)
            if array.ndim != 3:
                raise ValueError(
                    f"{subject} image data has shape {data_shape}. Stitching requires exactly three spatial dimensions."
                )
            if not (
                np.issubdtype(array.dtype, np.integer)
                or np.issubdtype(array.dtype, np.floating)
            ) or np.issubdtype(array.dtype, np.complexfloating):
                raise ValueError(
                    f"{subject} image data type {array.dtype} is not a real scalar type."
                )
            if data_shape != dims[::-1]:
                raise ValueError(
                    f"{subject} dimensions {dims} (XYZ) do not match image data shape {data_shape} (ZYX)."
                )

        world_affine = invertible_affine4(
            tile.get("world_affine"), f"{subject} working affine"
        ).copy()
        local_value = tile.get("local_affine", tile.get("local_index_affine"))
        local_affine = (
            invertible_affine4(local_value, f"{subject} local working affine").copy()
            if local_value is not None
            else None
        )
        actor_value = tile.get("actor_matrix", tile.get("world_transform"))
        actor_matrix = (
            invertible_affine4(actor_value, f"{subject} initial pose").copy()
            if actor_value is not None
            else None
        )
        if local_affine is not None and actor_matrix is not None:
            expected_world = actor_matrix @ local_affine
            if not geometry_values_equivalent(expected_world, world_affine):
                raise ValueError(
                    f"{subject} has an internal runtime/model disagreement: its pose and local working affine do not reproduce its working affine."
                )

        descriptor = copy.deepcopy(dict(tile.get("source_descriptor") or {}))
        working_geometry, geometry_revision = _validated_source_working_geometry(
            tile,
            descriptor,
            subject=subject,
            dims=dims,
            local_affine=local_affine,
            actor_matrix=actor_matrix,
            world_affine=world_affine,
        )
        descriptor_dims = working_geometry.get("dimensions")
        if descriptor_dims is not None:
            exact_descriptor_dims = finite_tuple3(
                descriptor_dims,
                f"{subject} frozen working dimensions",
                positive=True,
                integer=True,
            )
            if exact_descriptor_dims != dims:
                raise ValueError(
                    f"{subject} frozen working dimensions {exact_descriptor_dims} do not match the runtime array dimensions {dims}."
                )

        raw_units = _tile_source_units(tile, descriptor, working_geometry)
        units, unit_state = _canonical_units_or_none(raw_units)
        if units is not None:
            recognized_units.append(units[0])
        inspected.append(
            {
                "tile_index": tile_index,
                "tile": tile,
                "label": label,
                "subject": subject,
                "dims": dims,
                "data_shape": data_shape,
                "data_dtype": data_dtype,
                "world_affine": world_affine,
                "local_affine": local_affine,
                "actor_matrix": actor_matrix,
                "descriptor": descriptor,
                "working_geometry": working_geometry,
                "geometry_revision": geometry_revision,
                "raw_units": raw_units,
                "units": units,
                "unit_state": unit_state,
            }
        )

    if chosen_output_unit is None:
        target_unit = recognized_units[0] if recognized_units else None
    else:
        target_unit = canonical_space_units(chosen_output_unit)[0]

    prepared_tiles = []
    normalized_affines = []
    source_records = []
    conversion_records = []
    warnings = []
    assumptions = []
    calibration_states = []
    source_coordinate_spaces = []

    for item in inspected:
        tile = item["tile"]
        descriptor = item["descriptor"]
        working_geometry = item["working_geometry"]
        units = item["units"]
        if units is not None and target_unit is not None:
            scale = 10.0 ** (
                _UNIT_POWER10_METERS[units[0]]
                - _UNIT_POWER10_METERS[target_unit]
            )
            conversion_mode = "exact-physical-unit-conversion"
        else:
            scale = 1.0
            conversion_mode = "unconverted-numeric-working-grid"
        conversion = np.eye(4, dtype=float)
        conversion[:3, :3] *= float(scale)
        normalized_affine = conversion @ item["world_affine"]
        normalized_affine = invertible_affine4(
            normalized_affine, f"{item['subject']} normalized working affine"
        ).copy()
        normalized_spacing = np.linalg.norm(normalized_affine[:3, :3], axis=0)

        actor_matrix = item["actor_matrix"]
        initial_pose = tile.get("initial_pose")
        if initial_pose is None:
            initial_pose = actor_matrix if actor_matrix is not None else np.eye(4)
        initial_pose = invertible_affine4(
            initial_pose, f"{item['subject']} operation initial pose"
        ).copy()
        solved_mosaic = invertible_affine4(
            tile.get("solved_correction_mosaic", np.eye(4)),
            f"{item['subject']} mosaic correction",
        ).copy()
        solved_source = invertible_affine4(
            tile.get("solved_correction_source", np.eye(4)),
            f"{item['subject']} source correction",
        ).copy()

        source_coordinate_space = str(
            working_geometry.get("source_coordinate_space_id")
            or tile.get("coordinate_space_id")
            or descriptor.get("source_id")
            or ""
        ).strip()
        source_coordinate_spaces.append(source_coordinate_space or None)
        source_id = str(
            descriptor.get("source_id") or tile.get("source_id") or ""
        )
        channel_id = str(
            descriptor.get("channel_id") or tile.get("channel_id") or ""
        )
        source_working_affine = (
            item["local_affine"]
            if item["local_affine"] is not None
            else item["world_affine"]
        )
        if not working_geometry:
            source_components = grid_components_from_affine(source_working_affine)
            working_geometry = {
                "dimensions": list(item["dims"]),
                "spacing": list(source_components["spacing"]),
                "origin": list(source_components["origin"]),
                "direction": np.asarray(
                    source_components["direction"], dtype=float
                ).tolist(),
                "local_index_to_working_affine": _matrix_to_json(
                    source_working_affine
                ),
                "physical_units": (
                    list(units) if units is not None else None
                ),
                "source_coordinate_space_id": source_coordinate_space or None,
                "coordinate_mode": "physical" if units is not None else "numerical",
                "geometry_basis": "operation-runtime-grid",
                "physical_grid_state": "unrecorded",
                "warnings": [
                    "Working geometry was captured from the operation snapshot."
                ],
            }
        geometry_revision = item["geometry_revision"]

        conversion_record = {
            "tile_id": str(tile.get("tile_id") or item["tile_index"]),
            "source_id": source_id,
            "channel_id": channel_id,
            "source_units_raw": _json_geometry_value(item["raw_units"]),
            "source_units_canonical": list(units) if units is not None else None,
            "chosen_output_unit": target_unit,
            "scale_to_output": float(scale),
            "conversion_matrix": _matrix_to_json(conversion),
            "mode": conversion_mode,
            "uncertainty": item["unit_state"] if units is None else None,
        }
        conversion_records.append(conversion_record)

        initial_mosaic_mapping = conversion @ initial_pose
        mosaic_mapping = (
            conversion @ actor_matrix
            if actor_matrix is not None
            else conversion.copy()
        )
        source_record = {
            "tile_id": conversion_record["tile_id"],
            "display_name": item["label"],
            "source_id": source_id,
            "channel_id": channel_id,
            "source_checksum": copy.deepcopy(
                tile.get("source_checksum")
                or descriptor.get("source_checksum")
            ),
            "geometry_checksum": copy.deepcopy(
                tile.get("geometry_checksum")
                or descriptor.get("geometry_checksum")
            ),
            "data_shape_zyx": (
                list(item["data_shape"])
                if item["data_shape"] is not None
                else None
            ),
            "data_dtype": item["data_dtype"],
            "backing_source_id": str(
                descriptor.get("backing_source_id")
                or tile.get("backing_source_id")
                or ""
            ),
            "source_coordinate_space_id": source_coordinate_space or None,
            "original_working_grid": _json_geometry_value(working_geometry),
            "operation_working_grid": _json_geometry_value(working_geometry),
            "source_working_affine": _matrix_to_json(source_working_affine),
            "operation_world_affine": _matrix_to_json(item["world_affine"]),
            "operation_pose": (
                _matrix_to_json(actor_matrix)
                if actor_matrix is not None
                else _matrix_to_json(initial_pose)
            ),
            "initial_pose": _matrix_to_json(initial_pose),
            "initial_mosaic_mapping": _matrix_to_json(initial_mosaic_mapping),
            "solved_correction_mosaic": _matrix_to_json(solved_mosaic),
            "solved_correction_source": _matrix_to_json(solved_source),
            "unit_conversion": copy.deepcopy(conversion_record),
            "mosaic_space_mapping": _matrix_to_json(mosaic_mapping),
            "normalized_working_affine": _matrix_to_json(normalized_affine),
            "geometry_revision": geometry_revision,
            "channel_local_geometry_revision": geometry_revision,
            "project_capture_pose": copy.deepcopy(
                descriptor.get("project_capture_pose")
                or descriptor.get("source_pose")
            ),
            "geometry_status": copy.deepcopy(
                descriptor.get("geometry_status") or {}
            ),
            "warnings": copy.deepcopy(descriptor.get("warnings") or []),
            "assumptions": copy.deepcopy(descriptor.get("assumptions") or []),
        }
        source_records.append(source_record)
        normalized_affines.append(normalized_affine)

        status_code = str(
            (descriptor.get("geometry_status") or {}).get("code") or ""
        )
        calibration_states.append(status_code)
        for warning in descriptor.get("warnings") or ():
            warnings.append(
                {
                    "code": "source-geometry-warning",
                    "tile_id": source_record["tile_id"],
                    "message": str(
                        warning.get("message") if isinstance(warning, dict) else warning
                    ),
                    "source": _json_geometry_value(warning),
                }
            )
        for assumption in descriptor.get("assumptions") or ():
            assumptions.append(
                {
                    "code": "source-working-assumption",
                    "tile_id": source_record["tile_id"],
                    "message": str(
                        assumption.get("message")
                        if isinstance(assumption, dict)
                        else assumption
                    ),
                    "source": _json_geometry_value(assumption),
                }
            )
        if units is None:
            warnings.append(
                {
                    "code": f"{item['unit_state']}-source-units",
                    "tile_id": source_record["tile_id"],
                    "message": (
                        f"{item['label']} has {item['unit_state']} spatial units; "
                        "its current numeric working grid was used without conversion."
                    ),
                }
            )
            assumptions.append(
                {
                    "code": "numeric-grid-without-unit-conversion",
                    "tile_id": source_record["tile_id"],
                    "message": (
                        "The source numeric coordinates were mapped one-for-one into "
                        "the mosaic while retaining unknown unit provenance."
                    ),
                }
            )

        prepared = dict(tile)
        prepared.update(
            {
                "dims": item["dims"],
                "world_affine": normalized_affine,
                "world_index_affine": normalized_affine.copy(),
                "spacing": tuple(float(value) for value in normalized_spacing),
                "space_units": None,
                "source_world_affine": item["world_affine"].copy(),
                "unit_conversion": copy.deepcopy(conversion_record),
                "mosaic_space_mapping": mosaic_mapping,
                "_mosaic_prepared": True,
            }
        )
        prepared_tiles.append(prepared)

    known_count = len(recognized_units)
    if known_count == 0:
        status = {
            "code": "assumed",
            "label": "assumed numerical mosaic geometry",
            "physical_calibration": False,
        }
        output_space_units = None
    elif known_count != len(inspected):
        status = {
            "code": "mixed",
            "label": "mixed physical and unit-unknown mosaic geometry",
            "physical_calibration": False,
        }
        output_space_units = None
    elif calibration_states and any(
        state and state != "calibrated" for state in calibration_states
    ):
        status = {
            "code": "assumed",
            "label": "unit-normalized mosaic with unverified source geometry",
            "physical_calibration": False,
        }
        output_space_units = (target_unit,) * 3
    else:
        status = {
            "code": "calibrated",
            "label": "unit-normalized physical mosaic geometry",
            "physical_calibration": True,
        }
        output_space_units = (target_unit,) * 3

    distinct_spaces = {
        value for value in source_coordinate_spaces if value not in (None, "")
    }
    if len(distinct_spaces) > 1:
        warnings.append(
            {
                "code": "independent-source-coordinate-spaces",
                "message": (
                    "Inputs came from different source coordinate spaces. Their "
                    "current/manual poses define their mappings into this job-local mosaic."
                ),
                "source_coordinate_space_ids": sorted(distinct_spaces),
            }
        )

    identity = str(mosaic_coordinate_space_id or "").strip()
    if not identity:
        identity = _stable_mosaic_coordinate_space_id(source_records, target_unit)
    for prepared in prepared_tiles:
        prepared["coordinate_space_id"] = identity
        prepared["mosaic_coordinate_space_id"] = identity
        prepared["mosaic_geometry_status"] = copy.deepcopy(status)
        prepared["space_units"] = (
            list(output_space_units) if output_space_units is not None else None
        )

    provenance = StitchingMosaicGeometry(
        coordinate_space_id=identity,
        output_geometry_status=status,
        output_space_units=output_space_units,
        chosen_output_unit=target_unit,
        source_tiles=tuple(source_records),
        normalized_working_affines=tuple(
            {
                "tile_id": source_records[index]["tile_id"],
                "source_id": source_records[index]["source_id"],
                "channel_id": source_records[index]["channel_id"],
                "affine": _matrix_to_json(affine),
            }
            for index, affine in enumerate(normalized_affines)
        ),
        unit_conversions=tuple(conversion_records),
        warnings=tuple(warnings),
        assumptions=tuple(assumptions),
    )
    return StitchingGeometryPreparation(
        prepared_tiles=prepared_tiles,
        normalized_working_affines=normalized_affines,
        provenance=provenance,
    )


def _prepared_tiles(tiles, *, require_data=True):
    tiles = list(tiles)
    identities = {
        str(tile.get("mosaic_coordinate_space_id") or "") for tile in tiles
    }
    if (
        tiles
        and all(bool(tile.get("_mosaic_prepared")) for tile in tiles)
        and len(identities) == 1
        and "" not in identities
    ):
        return tiles
    return _prepare_stitching_geometry(
        tiles, require_data=require_data
    ).prepared_tiles


def _resample_zyx(
    source_zyx: np.ndarray,
    source_world_affine: np.ndarray,
    output_world_affine: np.ndarray,
    output_shape_zyx: tuple[int, int, int],
    *,
    order: int = 1,
    cval: float = 0.0,
    output_dtype=np.float32,
    mode: str = "constant",
):
    try:
        from scipy import ndimage
    except Exception as exc:
        raise RuntimeError(
            "3-D stitching requires SciPy. Install it with: pip install scipy"
        ) from exc

    source_zyx = np.ascontiguousarray(np.asarray(source_zyx))
    mapping = _output_to_source_zyx_mapping(source_world_affine, output_world_affine)
    return ndimage.affine_transform(
        source_zyx,
        matrix=mapping[:3, :3],
        offset=mapping[:3, 3],
        output_shape=tuple(int(v) for v in output_shape_zyx),
        output=output_dtype,
        order=int(order),
        mode=str(mode),
        cval=float(cval),
        prefilter=(int(order) > 1),
    )


_UNIT_MASK_SCALAR = np.array(1, dtype=np.uint8)


def _unit_mask_view(shape_zyx):
    """Return a zero-allocation all-valid mask view for a rectangular source volume.

    ``np.ones(source.shape)`` used to be rebuilt inside every registration/fusion
    chunk.  A broadcast scalar has the same read semantics for SciPy's nearest-
    neighbour affine sampler, but owns only one byte regardless of source size.
    """
    shape = finite_tuple3(
        shape_zyx, "Volume data shape", positive=True, integer=True
    )
    return np.broadcast_to(_UNIT_MASK_SCALAR, shape)


def _tile_support_mask(tile):
    mask = tile.get("_support_mask")
    expected = tuple(int(v) for v in np.asarray(tile["data"]).shape)
    if mask is None or tuple(mask.shape) != expected:
        mask = _unit_mask_view(expected)
        tile["_support_mask"] = mask
    return mask


def _output_to_source_zyx_mapping(source_world_affine, output_world_affine):
    source_world_affine = invertible_affine4(
        source_world_affine, "Source physical transform"
    )
    output_world_affine = invertible_affine4(
        output_world_affine, "Output physical transform"
    )
    try:
        out_xyz_to_in_xyz = np.linalg.solve(source_world_affine, output_world_affine)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("A selected volume transform is singular.") from exc
    return _XYZ_ZYX_PERMUTATION @ out_xyz_to_in_xyz @ _XYZ_ZYX_PERMUTATION


def _tile_physical_support(
    tile,
    output_world_affine,
    output_shape_zyx,
    *,
    with_distance,
    distance_dtype=np.float32,
):
    """Return sample coverage and optional distance to a tile's nearest physical face."""
    shape = finite_tuple3(
        output_shape_zyx, "Fusion output shape", positive=True, integer=True
    )

    mapping = _output_to_source_zyx_mapping(tile["world_affine"], output_world_affine)
    matrix = np.asarray(mapping[:3, :3], dtype=float)
    offset = np.asarray(mapping[:3, 3], dtype=float)

    z = np.arange(shape[0], dtype=float)[:, None, None]
    y = np.arange(shape[1], dtype=float)[None, :, None]
    x = np.arange(shape[2], dtype=float)[None, None, :]

    src_shape = np.asarray(np.asarray(tile["data"]).shape, dtype=float)
    face_distance_scales_zyx = None
    if with_distance:
        linear_xyz = np.asarray(tile["world_affine"], dtype=float)[:3, :3]
        plane_normal_lengths_xyz = np.linalg.norm(np.linalg.inv(linear_xyz).T, axis=0)
        face_distance_scales_zyx = (1.0 / plane_normal_lengths_xyz)[::-1]
    coordinate_tolerance = 64.0 * np.finfo(float).eps * max(1.0, float(np.max(src_shape)))

    # Voxel support extends half a voxel beyond the first and last centres:
    # [-0.5, n - 0.5]. For source face s_i=c, the world-space distance is
    # abs(s_i-c) / ||L^-T e_i||. Build one coordinate field at a time so fusion
    # never retains three full coordinate volumes for a chunk.
    inside = np.ones(shape, dtype=bool)
    distance = None
    for axis in range(3):
        src = (
            matrix[axis, 0] * z
            + matrix[axis, 1] * y
            + matrix[axis, 2] * x
            + offset[axis]
        )
        src += 0.5
        upper_distance = src_shape[axis] - src
        np.minimum(src, upper_distance, out=src)
        del upper_distance
        inside &= src >= -coordinate_tolerance
        if with_distance:
            np.maximum(src, 0.0, out=src)
            src *= face_distance_scales_zyx[axis]
            axis_distance = src.astype(distance_dtype)
            if distance is None:
                distance = axis_distance
            else:
                np.minimum(distance, axis_distance, out=distance)
                del axis_distance
        del src

    if distance is not None:
        distance[~inside] = 0.0
    return inside, distance


def _robust_normalize(volume, mask=None):
    x = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(x)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    values = x[finite]
    if values.size < 32:
        return np.zeros_like(x, dtype=np.float32)
    positive = values[values > 0]
    if positive.size >= 256:
        values = positive
    lo, hi = np.percentile(values, (1.0, 99.5))
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    out = np.clip((x - float(lo)) / float(hi - lo), 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32, copy=False)


def _registration_preprocess(volume, mask):
    try:
        from scipy import ndimage
    except Exception as exc:
        raise RuntimeError("3-D stitching requires SciPy.") from exc

    x = _robust_normalize(volume, mask)
    if min(x.shape) >= 8:
        background = ndimage.gaussian_filter(x, sigma=2.0)
        x = x - background
        x = _robust_normalize(x, mask)
    x *= np.asarray(mask, dtype=np.float32)
    return np.ascontiguousarray(x)


def _registration_support_taper(volume, mask):
    """Condition support boundaries for FFT phase correlation only."""
    try:
        from scipy import ndimage
    except Exception as exc:
        raise RuntimeError("3-D stitching requires SciPy.") from exc

    x = np.asarray(volume, dtype=np.float32).copy()
    support = np.asarray(mask, dtype=bool)
    padded_support = np.pad(support, 1, constant_values=False)
    distance = ndimage.distance_transform_edt(padded_support)[1:-1, 1:-1, 1:-1]
    taper_width = max(2.0, min(6.0, 0.125 * min(x.shape)))
    support_taper = np.sin(
        0.5 * np.pi * np.clip(distance / taper_width, 0.0, 1.0)
    ) ** 2
    x *= support_taper.astype(np.float32)
    return np.ascontiguousarray(x)


def _ncc(a, b, mask=None):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if np.count_nonzero(valid) < 128:
        return -1.0
    av = a[valid]
    bv = b[valid]
    # Keep registration focused on signal while retaining enough samples.
    signal = (av > np.percentile(av, 40.0)) | (bv > np.percentile(bv, 40.0))
    if np.count_nonzero(signal) >= 128:
        av = av[signal]
        bv = bv[signal]
    av = av - float(av.mean())
    bv = bv - float(bv.mean())
    denom = math.sqrt(float(np.dot(av, av)) * float(np.dot(bv, bv)))
    if denom <= 1e-12:
        return -1.0
    return float(np.dot(av, bv) / denom)


def _phase_correlation_shift(fixed, moving, max_shift_zyx=None):
    """Return displacement plus an interpretable phase-peak ambiguity metric.

    Confidence compares the primary peak with the strongest peak outside a
    two-voxel periodic neighbourhood. A ratio below 1.5 or relative prominence
    below 0.25 is conservatively flagged for review; it is not an edge-rejection
    rule. Flat/no-information inputs report invalid ambiguous confidence.
    """
    fixed = np.asarray(fixed, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)
    if fixed.shape != moving.shape:
        raise ValueError("Phase-correlation arrays must have the same shape.")

    f = np.fft.fftn(fixed)
    m = np.fft.fftn(moving)
    cross = f * np.conj(m)
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.ifftn(cross).real

    allowed = None
    if max_shift_zyx is not None:
        max_shift = np.asarray(max_shift_zyx, dtype=int)
        allowed = np.zeros(corr.shape, dtype=bool)
        index_sets = []
        for n, radius in zip(corr.shape, max_shift):
            radius = max(0, min(int(radius), n // 2))
            idx = np.concatenate((np.arange(0, radius + 1), np.arange(n - radius, n)))
            index_sets.append(np.unique(idx))
        allowed[np.ix_(*index_sets)] = True
        masked = np.where(allowed, corr, -np.inf)
        peak_index = np.unravel_index(int(np.argmax(masked)), corr.shape)
    else:
        masked = corr
        peak_index = np.unravel_index(int(np.argmax(corr)), corr.shape)

    primary_peak = float(corr[peak_index])
    secondary_search = np.array(masked, dtype=float, copy=True)
    exclusion_radius = 2
    excluded_indices = [
        np.unique(
            [
                (int(peak_index[axis]) + offset) % int(corr.shape[axis])
                for offset in range(-exclusion_radius, exclusion_radius + 1)
            ]
        )
        for axis in range(3)
    ]
    secondary_search[np.ix_(*excluded_indices)] = -np.inf
    finite_secondary = secondary_search[np.isfinite(secondary_search)]
    secondary_peak = (
        float(np.max(finite_secondary)) if finite_secondary.size else float("nan")
    )
    signal_norm = math.sqrt(
        float(np.vdot(fixed, fixed).real) * float(np.vdot(moving, moving).real)
    )
    valid_confidence = bool(
        math.isfinite(primary_peak)
        and math.isfinite(secondary_peak)
        and signal_norm > 1e-12
        and primary_peak > 1e-12
    )
    if valid_confidence:
        peak_ratio = float(primary_peak / max(secondary_peak, 1e-12))
        peak_prominence = float(
            (primary_peak - secondary_peak) / max(abs(primary_peak), 1e-12)
        )
        ambiguous = bool(peak_ratio < 1.5 or peak_prominence < 0.25)
    else:
        peak_ratio = None
        peak_prominence = None
        ambiguous = True

    shift = np.asarray(peak_index, dtype=float)
    shape = np.asarray(corr.shape, dtype=float)
    shift[shift > shape / 2.0] -= shape[shift > shape / 2.0]

    # Independent parabolic interpolation around the periodic correlation peak.
    for axis in range(3):
        idx0 = list(peak_index)
        idxm = list(peak_index)
        idxp = list(peak_index)
        idxm[axis] = (idxm[axis] - 1) % corr.shape[axis]
        idxp[axis] = (idxp[axis] + 1) % corr.shape[axis]
        fm = float(corr[tuple(idxm)])
        f0 = float(corr[tuple(idx0)])
        fp = float(corr[tuple(idxp)])
        denom = fm - 2.0 * f0 + fp
        if abs(denom) > 1e-12:
            delta = 0.5 * (fm - fp) / denom
            if abs(delta) <= 1.0:
                shift[axis] += delta
    return {
        "shift_zyx": shift,
        "primary_peak": primary_peak,
        "secondary_peak": secondary_peak if math.isfinite(secondary_peak) else None,
        "peak_ratio": peak_ratio,
        "peak_prominence": peak_prominence,
        "valid": valid_confidence,
        "ambiguous": ambiguous,
        "secondary_exclusion_radius_voxels": exclusion_radius,
    }


def _translation_matrix_xyz(translation_xyz):
    out = np.eye(4, dtype=float)
    out[:3, 3] = np.asarray(translation_xyz, dtype=float)
    return out


def _rigid_matrix(params, center_xyz=(0.0, 0.0, 0.0)):
    """World correction matrix from tx,ty,tz,rx,ry,rz (degrees)."""
    try:
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        raise RuntimeError("Rigid stitching requires scipy.spatial.transform.") from exc

    p = np.asarray(params, dtype=float)
    t = p[:3]
    rot = Rotation.from_euler("xyz", p[3:6], degrees=True).as_matrix()
    center = np.asarray(center_xyz, dtype=float)
    out = np.eye(4, dtype=float)
    out[:3, :3] = rot
    out[:3, 3] = t + center - rot @ center
    return out


def _proper_rotation_from_linear(matrix3):
    """Closest proper rotation to a rigid or affine 3x3 linear matrix."""
    linear = np.asarray(matrix3, dtype=float).reshape(3, 3)
    u, _s, vh = np.linalg.svd(linear)
    rot = u @ vh
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vh
    return rot


def _rotation_vector_degrees(matrix3):
    try:
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        raise RuntimeError("Rigid stitching requires scipy.spatial.transform.") from exc
    rot = _proper_rotation_from_linear(matrix3)
    return Rotation.from_matrix(rot).as_rotvec() * (180.0 / math.pi)


def _affine_matrix(params, center_xyz=(0.0, 0.0, 0.0)):
    """World correction from 12 parameters.

    Parameters are translation XYZ, Euler rotation XYZ in degrees, logarithmic
    scale XYZ, and upper-triangular shear XY/XZ/YZ. This is a complete
    orientation-preserving 12-DOF affine model without projective terms.
    """
    try:
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        raise RuntimeError("Affine stitching requires scipy.spatial.transform.") from exc

    p = np.asarray(params, dtype=float).reshape(12)
    translation = p[:3]
    rotation = Rotation.from_euler("xyz", p[3:6], degrees=True).as_matrix()
    scales = np.exp(np.clip(p[6:9], -6.0, 6.0))
    shear = np.eye(3, dtype=float)
    shear[0, 1] = p[9]
    shear[0, 2] = p[10]
    shear[1, 2] = p[11]
    linear = rotation @ shear @ np.diag(scales)
    center = np.asarray(center_xyz, dtype=float)
    out = np.eye(4, dtype=float)
    out[:3, :3] = linear
    out[:3, 3] = translation + center - linear @ center
    return out


def _affine_linear_metrics(matrix3):
    """Return approximate rotation, principal scales, and shear magnitudes."""
    linear = np.asarray(matrix3, dtype=float).reshape(3, 3)
    rotation = _proper_rotation_from_linear(linear)
    stretch = rotation.T @ linear
    scales = np.diag(stretch).copy()
    safe = scales.copy()
    safe[np.abs(safe) < 1e-9] = 1.0
    normalized = stretch @ np.diag(1.0 / safe)
    shear = np.array([normalized[0, 1], normalized[0, 2], normalized[1, 2]], dtype=float)
    return rotation, scales, shear


def _matrix_to_json(matrix):
    return np.asarray(matrix, dtype=float).round(12).tolist()


def matrix_maps_equal(first, second, tolerance=1e-6):
    if set(first) != set(second):
        return False
    return all(
        np.allclose(
            np.asarray(first[key]),
            np.asarray(second[key]),
            atol=tolerance,
            rtol=tolerance,
        )
        for key in first
    )


def assign_source_target(targets, source_key, target):
    target = invertible_affine4(
        target, f"Registered target pose for {source_key}"
    ).copy()
    existing = targets.get(source_key)
    if existing is not None and not np.allclose(
        np.asarray(existing), target, atol=1e-8, rtol=1e-8
    ):
        raise RuntimeError(
            f"Registration produced conflicting target poses for {source_key}. "
            "One SourceID cannot be assigned to multiple stitching tiles."
        )
    targets[source_key] = target


def layout_delta(layout, tile_id):
    return np.asarray(
        (layout or {}).get("placement_deltas", {}).get(tile_id, np.eye(4)),
        dtype=float,
    )


def annotate_fusion_geometry_provenance(
    snapshot,
    *,
    descriptor,
    display_name,
    tile_id,
    source_key,
    actor_matrix,
    registration_result,
    pose_source,
):
    """Attach exact operation poses/corrections to a detached fusion snapshot."""
    result = registration_result or {}
    snapshot["source_descriptor"] = copy.deepcopy(descriptor or {})
    if pose_source == "registered" and result:
        base = (result.get("base_matrices_by_source") or {}).get(source_key)
        if base is None:
            raise RuntimeError(
                f"Registered stitching result has no initial pose for {display_name!r}."
            )
        initial_pose = layout_delta(result.get("initial_layout", {}), tile_id) @ np.asarray(
            base, dtype=float
        )
        mosaic_correction = np.asarray(
            (result.get("mosaic_corrections") or {}).get(tile_id, np.eye(4)),
            dtype=float,
        )
        source_correction = np.asarray(
            (result.get("corrections") or {}).get(tile_id, np.eye(4)),
            dtype=float,
        )
    else:
        initial_pose = np.asarray(actor_matrix, dtype=float)
        mosaic_correction = np.eye(4, dtype=float)
        source_correction = np.eye(4, dtype=float)
    snapshot["initial_pose"] = invertible_affine4(
        initial_pose, f"Initial stitching pose for {display_name!r}"
    ).copy()
    snapshot["solved_correction_mosaic"] = invertible_affine4(
        mosaic_correction, f"Mosaic correction for {display_name!r}"
    ).copy()
    snapshot["solved_correction_source"] = invertible_affine4(
        source_correction, f"Source correction for {display_name!r}"
    ).copy()


def _cast_output(array, dtype):
    dtype = np.dtype(dtype)
    x = np.asarray(array)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        x = np.clip(np.rint(x), limits.min, limits.max)
    return np.ascontiguousarray(x.astype(dtype, copy=False))


def _safe_stem(text):
    import re

    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "stitched")).strip("._")
    return value or "stitched"


# -----------------------------------------------------------------------------
# Registration engine
# -----------------------------------------------------------------------------


def _candidate_pairs(tiles, search_margin, minimum_overlap_fraction):
    tiles = _prepared_tiles(tiles)
    pairs = []
    margin = max(0.0, float(search_margin))
    min_fraction = max(0.0, float(minimum_overlap_fraction))
    bounds = {}
    for tile in tiles:
        bounds[tile["tile_id"]] = _support_bounds(tile["world_affine"], tile["dims"])

    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            a = tiles[i]
            b = tiles[j]
            alo, ahi = bounds[a["tile_id"]]
            blo, bhi = bounds[b["tile_id"]]
            actual_lo = np.maximum(alo, blo)
            actual_hi = np.minimum(ahi, bhi)
            actual_extent = np.maximum(0.0, actual_hi - actual_lo)
            actual_volume = float(np.prod(actual_extent))
            min_volume = max(1e-12, min(float(np.prod(ahi - alo)), float(np.prod(bhi - blo))))
            overlap_fraction = actual_volume / min_volume

            search_lo = np.maximum(alo - margin, blo - margin)
            search_hi = np.minimum(ahi + margin, bhi + margin)
            if np.any(search_hi <= search_lo):
                continue
            if actual_volume > 0.0 and overlap_fraction < min_fraction:
                continue

            pairs.append(
                {
                    "i": i,
                    "j": j,
                    "fixed_id": a["tile_id"],
                    "moving_id": b["tile_id"],
                    "actual_overlap_fraction": overlap_fraction,
                    "search_lo": search_lo,
                    "search_hi": search_hi,
                }
            )
    return pairs


def _pair_registration_grid(tile_a, tile_b, pair, coarse_max_dim):
    lo = np.asarray(pair["search_lo"], dtype=float)
    hi = np.asarray(pair["search_hi"], dtype=float)
    raw_extent = hi - lo
    if lo.shape != (3,) or hi.shape != (3,) or not np.all(np.isfinite(raw_extent)) or np.any(raw_extent <= 0.0):
        raise ValueError("A registration pair has invalid physical search bounds.")
    extent = np.maximum(raw_extent, 1e-6)

    spacing_a = np.asarray(tile_a["spacing"], dtype=float)
    spacing_b = np.asarray(tile_b["spacing"], dtype=float)
    base_spacing = np.maximum(np.minimum(spacing_a, spacing_b), 1e-6)
    dims = np.ceil(extent / base_spacing).astype(int) + 1
    max_dim = max(16, int(coarse_max_dim))
    scale = max(1.0, float(np.max(dims)) / float(max_dim))
    spacing = base_spacing * scale
    dims = np.maximum(8, np.ceil(extent / spacing).astype(int) + 1)
    origin = lo + 0.5 * spacing
    affine = grid_affine_from_components(origin, spacing, np.eye(3, dtype=float))
    shape_zyx = (int(dims[2]), int(dims[1]), int(dims[0]))
    return affine, shape_zyx, spacing, lo, hi


def _estimate_registration_pair_bytes(tile_a, tile_b, pair, settings, mode):
    """Conservative temporary-memory estimate for one coarse pair task.

    The factor covers fixed/moving resamples, support masks, normalized and
    background-filtered arrays, FFT inputs/outputs and cross-power data,
    transformed buffers, and optimizer work. It deliberately includes a fixed
    allowance for SciPy/FFT plans and allocator overhead.
    """
    _affine, shape_zyx, _spacing, _lo, _hi = _pair_registration_grid(
        tile_a, tile_b, pair, settings["coarse_max_dim"]
    )
    voxels = math.prod(int(value) for value in shape_zyx)
    bytes_per_voxel = {
        "translation": 96,
        "rigid": 112,
        "affine": 128,
    }.get(str(mode or "translation").lower(), 128)
    return int(64 * 1024**2 + voxels * bytes_per_voxel)


def _register_pair(tile_a, tile_b, pair, settings, mode):
    tile_a, tile_b = _prepared_tiles([tile_a, tile_b])
    try:
        from scipy import ndimage, optimize
    except Exception as exc:
        raise RuntimeError("3-D stitching requires SciPy.") from exc

    mode = str(mode or "translation").lower()
    if mode not in {"translation", "rigid", "affine"}:
        raise ValueError(f"Unsupported stitching registration mode: {mode}")

    grid_affine, shape_zyx, spacing_xyz, lo, hi = _pair_registration_grid(
        tile_a, tile_b, pair, settings["coarse_max_dim"]
    )

    fixed = _resample_zyx(
        tile_a["data"], tile_a["world_affine"], grid_affine, shape_zyx,
        order=1, output_dtype=np.float32,
    )
    moving = _resample_zyx(
        tile_b["data"], tile_b["world_affine"], grid_affine, shape_zyx,
        order=1, output_dtype=np.float32,
    )
    fixed_mask = _resample_zyx(
        _tile_support_mask(tile_a),
        tile_a["world_affine"], grid_affine, shape_zyx,
        order=0, output_dtype=np.uint8,
    ) > 0
    moving_mask = _resample_zyx(
        _tile_support_mask(tile_b),
        tile_b["world_affine"], grid_affine, shape_zyx,
        order=0, output_dtype=np.uint8,
    ) > 0

    fixed_p = _registration_preprocess(fixed, fixed_mask)
    moving_p = _registration_preprocess(moving, moving_mask)
    initial_common = fixed_mask & moving_mask
    initial_score = _ncc(fixed_p, moving_p, initial_common)

    search_margin = max(0.0, float(settings["search_margin"]))
    spacing_zyx = np.asarray(spacing_xyz[::-1], dtype=float)
    max_shift_zyx = np.ceil(search_margin / np.maximum(spacing_zyx, 1e-9)).astype(int)
    max_shift_zyx = np.minimum(max_shift_zyx, np.asarray(shape_zyx) // 2)
    fixed_phase = _registration_support_taper(fixed_p, fixed_mask)
    moving_phase = _registration_support_taper(moving_p, moving_mask)
    phase_correlation = _phase_correlation_shift(
        fixed_phase, moving_phase, max_shift_zyx
    )
    shift_zyx = np.asarray(phase_correlation["shift_zyx"], dtype=float)

    moved_translation = ndimage.shift(
        moving_p,
        shift=shift_zyx,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    moved_mask_translation = ndimage.shift(
        moving_mask.astype(np.uint8),
        shift=shift_zyx,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ) > 0
    translation_common = fixed_mask & moved_mask_translation
    translation_score = _ncc(fixed_p, moved_translation, translation_common)

    translation_xyz = shift_zyx[::-1] * np.asarray(spacing_xyz, dtype=float)
    correction = _translation_matrix_xyz(translation_xyz)
    final_score = float(translation_score)
    final_common = translation_common
    rigid_params = np.r_[translation_xyz, [0.0, 0.0, 0.0]]
    affine_params = np.r_[rigid_params, np.zeros(6, dtype=float)]
    selected_model = "translation"
    center_xyz = 0.5 * (lo + hi)
    grid_inv = np.linalg.inv(grid_affine)

    def transform_moving_matrix(corr, order=1, mask=False):
        # The already-resampled moving image lives on grid_affine. Moving it by
        # corr requires output world points to sample corr^-1 world points.
        output_to_input = grid_inv @ np.linalg.inv(corr) @ grid_affine
        mapping = _XYZ_ZYX_PERMUTATION @ output_to_input @ _XYZ_ZYX_PERMUTATION
        source = moving_mask.astype(np.uint8) if mask else moving_p
        return ndimage.affine_transform(
            source,
            matrix=mapping[:3, :3],
            offset=mapping[:3, 3],
            output_shape=shape_zyx,
            output=np.uint8 if mask else np.float32,
            order=0 if mask else order,
            mode="constant",
            cval=0,
            prefilter=False,
        )

    def evaluated_correction(corr):
        moved = transform_moving_matrix(corr)
        moved_mask = transform_moving_matrix(corr, mask=True) > 0
        valid = fixed_mask & moved_mask
        return float(_ncc(fixed_p, moved, valid)), valid

    model_selection = {
        "requested_model": mode,
        "selected_model": selected_model,
        "minimum_advanced_ncc_gain": MINIMUM_ADVANCED_MODEL_NCC_GAIN,
        "candidates": {
            "translation": {
                "score": float(final_score),
                "selected": True,
                "source": "phase-correlation",
                "bound_hits": [],
            }
        },
    }

    max_angle = max(
        0.0,
        float(settings.get("max_angle_deg", DEFAULT_STITCHING_MAX_ANGLE_DEG)),
    )
    translation_radius = max(
        float(np.max(spacing_xyz)) * 2.0,
        search_margin * 0.35,
    )

    if mode in {"rigid", "affine"}:
        translation_lower = translation_xyz - translation_radius
        translation_upper = translation_xyz + translation_radius

        def translation_objective(params):
            score, _valid = evaluated_correction(_translation_matrix_xyz(params))
            return -score

        translation_result = optimize.minimize(
            translation_objective,
            translation_xyz,
            method="Powell",
            bounds=list(zip(translation_lower, translation_upper)),
            options={
                "maxiter": int(settings.get("translation_refinement_iterations", 40)),
                "xtol": 1e-3,
                "ftol": 1e-4,
                "disp": False,
            },
        )
        refined_translation = np.asarray(translation_result.x, dtype=float)
        refined_correction = _translation_matrix_xyz(refined_translation)
        refined_score, refined_common = evaluated_correction(refined_correction)
        if refined_score > final_score + 1e-8:
            translation_xyz = refined_translation
            correction = refined_correction
            final_score = refined_score
            final_common = refined_common
        rigid_params = np.r_[translation_xyz, [0.0, 0.0, 0.0]]
        affine_params = np.r_[rigid_params, np.zeros(6, dtype=float)]
        model_selection["candidates"]["translation"] = {
            "score": float(final_score),
            "selected": True,
            "source": "bounded-continuous-refinement",
            "phase_score": float(translation_score),
            "bound_hits": [],
        }

        lower = np.r_[translation_xyz - translation_radius, [-max_angle] * 3]
        upper = np.r_[translation_xyz + translation_radius, [max_angle] * 3]

        def rigid_objective(params):
            corr = _rigid_matrix(params, center_xyz)
            score, _valid = evaluated_correction(corr)
            angular_penalty = 1e-5 * float(np.dot(params[3:6], params[3:6]))
            return -score + angular_penalty

        result = optimize.minimize(
            rigid_objective,
            rigid_params,
            method="Powell",
            bounds=list(zip(lower, upper)),
            options={
                "maxiter": int(settings.get("rigid_iterations", 60)),
                "xtol": 1e-3,
                "ftol": 1e-4,
                "disp": False,
            },
        )
        rigid_candidate_params = np.asarray(result.x, dtype=float)
        rigid_candidate_corr = _rigid_matrix(rigid_candidate_params, center_xyz)
        rigid_candidate_score, rigid_candidate_common = evaluated_correction(
            rigid_candidate_corr
        )
        rigid_gain = float(rigid_candidate_score - final_score)
        rigid_bound_hits = []
        if max_angle > 0.0:
            bound_tolerance = max(1e-6, max_angle * 1e-3)
            for axis, value in zip("XYZ", rigid_candidate_params[3:6]):
                if abs(abs(float(value)) - max_angle) <= bound_tolerance:
                    rigid_bound_hits.append(f"rotation_{axis.lower()}")
        rigid_selected = rigid_gain >= MINIMUM_ADVANCED_MODEL_NCC_GAIN
        model_selection["candidates"]["rigid"] = {
            "score": float(rigid_candidate_score),
            "gain_over_translation": rigid_gain,
            "required_gain": MINIMUM_ADVANCED_MODEL_NCC_GAIN,
            "selected": bool(rigid_selected),
            "bound_hits": rigid_bound_hits,
        }
        if rigid_selected:
            selected_model = "rigid"
            rigid_params = rigid_candidate_params
            correction = rigid_candidate_corr
            final_score = float(rigid_candidate_score)
            final_common = rigid_candidate_common

    if mode == "affine":
        affine_seed = np.r_[rigid_candidate_params, np.zeros(6, dtype=float)]
        max_scale_fraction = max(
            0.0,
            min(
                0.95,
                float(
                    settings.get(
                        "max_scale_percent", DEFAULT_STITCHING_MAX_SCALE_PERCENT
                    )
                )
                / 100.0,
            ),
        )
        minimum_scale = max(0.05, 1.0 - max_scale_fraction)
        maximum_scale = max(minimum_scale + 1e-6, 1.0 + max_scale_fraction)
        log_scale_lower = math.log(minimum_scale)
        log_scale_upper = math.log(maximum_scale)
        max_shear = max(
            0.0, float(settings.get("max_shear", DEFAULT_STITCHING_MAX_SHEAR))
        )
        lower = np.r_[
            rigid_candidate_params[:3] - translation_radius,
            [-max_angle] * 3,
            [log_scale_lower] * 3,
            [-max_shear] * 3,
        ]
        upper = np.r_[
            rigid_candidate_params[:3] + translation_radius,
            [max_angle] * 3,
            [log_scale_upper] * 3,
            [max_shear] * 3,
        ]

        def affine_objective(params):
            corr = _affine_matrix(params, center_xyz)
            determinant = float(np.linalg.det(corr[:3, :3]))
            if not math.isfinite(determinant) or determinant <= 1e-5:
                return 10.0 + abs(determinant)
            moved = transform_moving_matrix(corr)
            moved_mask = transform_moving_matrix(corr, mask=True) > 0
            valid = fixed_mask & moved_mask
            score = _ncc(fixed_p, moved, valid)
            regularizer = (
                1e-5 * float(np.dot(params[3:6], params[3:6]))
                + 5e-4 * float(np.dot(params[6:9], params[6:9]))
                + 5e-4 * float(np.dot(params[9:12], params[9:12]))
            )
            return -score + regularizer

        result = optimize.minimize(
            affine_objective,
            affine_seed,
            method="Powell",
            bounds=list(zip(lower, upper)),
            options={
                "maxiter": int(settings.get("affine_iterations", 90)),
                "xtol": 5e-4,
                "ftol": 5e-5,
                "disp": False,
            },
        )
        affine_candidate_params = np.asarray(result.x, dtype=float)
        affine_candidate_corr = _affine_matrix(affine_candidate_params, center_xyz)
        affine_candidate_score, affine_candidate_common = evaluated_correction(
            affine_candidate_corr
        )
        best_simpler_score = max(
            float(model_selection["candidates"]["translation"]["score"]),
            float(model_selection["candidates"]["rigid"]["score"]),
        )
        affine_gain = float(affine_candidate_score - best_simpler_score)
        affine_bound_hits = []
        if max_angle > 0.0:
            angle_tolerance = max(1e-6, max_angle * 1e-3)
            for axis, value in zip("XYZ", affine_candidate_params[3:6]):
                if abs(abs(float(value)) - max_angle) <= angle_tolerance:
                    affine_bound_hits.append(f"rotation_{axis.lower()}")
        scale_tolerance = max(1e-8, (log_scale_upper - log_scale_lower) * 1e-3)
        for axis, value in zip("XYZ", affine_candidate_params[6:9]):
            if abs(float(value) - log_scale_lower) <= scale_tolerance:
                affine_bound_hits.append(f"scale_{axis.lower()}_minimum")
            elif abs(float(value) - log_scale_upper) <= scale_tolerance:
                affine_bound_hits.append(f"scale_{axis.lower()}_maximum")
        if max_shear > 0.0:
            shear_tolerance = max(1e-8, max_shear * 1e-3)
            for plane, value in zip(("xy", "xz", "yz"), affine_candidate_params[9:12]):
                if abs(abs(float(value)) - max_shear) <= shear_tolerance:
                    affine_bound_hits.append(f"shear_{plane}")
        affine_selected = affine_gain >= MINIMUM_ADVANCED_MODEL_NCC_GAIN
        model_selection["candidates"]["affine"] = {
            "score": float(affine_candidate_score),
            "gain_over_best_simpler_model": affine_gain,
            "best_simpler_score": best_simpler_score,
            "required_gain": MINIMUM_ADVANCED_MODEL_NCC_GAIN,
            "selected": bool(affine_selected),
            "bound_hits": affine_bound_hits,
        }
        if affine_selected:
            selected_model = "affine"
            rigid_params = affine_candidate_params[:6].copy()
            affine_params = affine_candidate_params
            correction = affine_candidate_corr
            final_score = float(affine_candidate_score)
            final_common = affine_candidate_common
        else:
            affine_params = np.r_[rigid_params, np.zeros(6, dtype=float)]

    for candidate_name, candidate_record in model_selection["candidates"].items():
        candidate_record["selected"] = candidate_name == selected_model
    model_selection["selected_model"] = selected_model

    _rot, scales, shear = _affine_linear_metrics(correction[:3, :3])
    physical_translation = _transform_displacement_at_point(correction, center_xyz)
    overlap_voxels = int(np.count_nonzero(final_common))
    weight = max(1e-4, (max(-1.0, min(1.0, final_score)) + 1.0) * 0.5)
    weight *= max(1.0, math.sqrt(max(1, overlap_voxels)))

    return {
        "fixed_id": tile_a["tile_id"],
        "moving_id": tile_b["tile_id"],
        "fixed_name": tile_a["display_name"],
        "moving_name": tile_b["display_name"],
        "mode": selected_model,
        "requested_mode": mode,
        "selected_model": selected_model,
        "model_selection": model_selection,
        "correction": correction,
        # For rigid/affine corrections the homogeneous translation column
        # includes pivot compensation and changes when the world origin moves.
        # Report the physical displacement at the registration overlap center.
        "translation_xyz": physical_translation,
        "overlap_center_xyz": np.asarray(center_xyz, dtype=float).copy(),
        "phase_translation_xyz": np.asarray(translation_xyz, dtype=float).copy(),
        "rigid_params": np.asarray(rigid_params, dtype=float),
        "affine_params": np.asarray(affine_params, dtype=float),
        "scale_xyz": np.asarray(scales, dtype=float),
        "shear_xyz": np.asarray(shear, dtype=float),
        "score": float(final_score),
        "initial_score": float(initial_score),
        "translation_score": float(translation_score),
        "weight": float(weight),
        "overlap_voxels": overlap_voxels,
        "overlap_fraction": float(pair["actual_overlap_fraction"]),
        "grid_shape_zyx": list(shape_zyx),
        "grid_spacing_xyz": np.asarray(spacing_xyz, dtype=float).tolist(),
        "grid_affine": _matrix_to_json(grid_affine),
        "translation_grid_voxels_xyz": np.linalg.solve(
            np.asarray(grid_affine, dtype=float)[:3, :3], physical_translation
        ),
        "preprocessing": {
            "normalization_percentiles": [1.0, 99.5],
            "gaussian_background_sigma_registration_voxels": 2.0,
            "support_boundary": "cosine-distance-taper",
            "support_taper_width_registration_voxels": max(
                2.0, min(6.0, 0.125 * min(shape_zyx))
            ),
            "boundary_audit_decision": "cosine-taper-adopted",
        },
        "phase_correlation": {
            key: value
            for key, value in phase_correlation.items()
            if key != "shift_zyx"
        },
    }

def _connected_components(tile_ids, edges):
    tile_ids = list(tile_ids)
    adjacency = {tile_id: set() for tile_id in tile_ids}
    for edge in edges:
        adjacency[edge["fixed_id"]].add(edge["moving_id"])
        adjacency[edge["moving_id"]].add(edge["fixed_id"])
    components = []
    unseen = set(tile_ids)
    for start in tile_ids:
        if start not in unseen:
            continue
        unseen.remove(start)
        comp_set = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nxt in adjacency[cur]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    comp_set.add(nxt)
                    stack.append(nxt)
        # Preserve the original tile order; component[0] is its fixed anchor.
        components.append([tile_id for tile_id in tile_ids if tile_id in comp_set])
    return components


def _edge_overlap_center(edge):
    center = np.asarray(edge.get("overlap_center_xyz", (0.0, 0.0, 0.0)), dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("A stitching edge has an invalid physical overlap center.")
    return center


def _transform_displacement_at_point(transform, point_xyz):
    """Return T(point) - point without using an origin-dependent column alone."""
    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    point = np.asarray(point_xyz, dtype=float).reshape(3)
    return (matrix[:3, :3] - np.eye(3, dtype=float)) @ point + matrix[:3, 3]


def _transform_rebased_to_origin(transform, origin_xyz):
    """Conjugate a world transform into coordinates centered at origin_xyz."""
    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    rebased = matrix.copy()
    rebased[:3, 3] = _transform_displacement_at_point(matrix, origin_xyz)
    return rebased


def _transform_rebased_to_world(transform, origin_xyz):
    """Undo _transform_rebased_to_origin without forming large translations."""
    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    origin = np.asarray(origin_xyz, dtype=float).reshape(3)
    restored = matrix.copy()
    restored[:3, 3] = (
        matrix[:3, 3] + origin - matrix[:3, :3] @ origin
    )
    return restored


def _edge_rebased_to_origin(edge, origin_xyz):
    rebased = dict(edge)
    rebased["correction"] = _transform_rebased_to_origin(
        edge["correction"], origin_xyz
    )
    rebased["overlap_center_xyz"] = (
        _edge_overlap_center(edge) - np.asarray(origin_xyz, dtype=float)
    )
    return rebased


def _edge_measured_displacement(edge):
    return _transform_displacement_at_point(
        edge["correction"], _edge_overlap_center(edge)
    )


def _edge_translation_error_vector(edge, fixed_correction, moving_correction):
    """Physical predicted-minus-measured displacement at the pair pivot."""
    predicted = np.linalg.inv(np.asarray(fixed_correction, dtype=float)) @ np.asarray(
        moving_correction, dtype=float
    )
    center = _edge_overlap_center(edge)
    return _transform_displacement_at_point(
        predicted, center
    ) - _transform_displacement_at_point(edge["correction"], center)


def _solve_translation_pose_graph(tile_ids, edges):
    tile_ids = list(tile_ids)
    corrections = {tile_id: np.eye(4, dtype=float) for tile_id in tile_ids}
    components = _connected_components(tile_ids, edges)
    if not edges:
        return corrections, components
    try:
        from scipy import optimize
    except Exception as exc:
        raise RuntimeError("Translation pose-graph optimization requires SciPy.") from exc

    for component in components:
        if len(component) <= 1:
            continue
        anchor = component[0]
        variables = [tile_id for tile_id in component if tile_id != anchor]
        index = {tile_id: i for i, tile_id in enumerate(variables)}
        component_edges = [
            edge for edge in edges
            if edge["fixed_id"] in component and edge["moving_id"] in component
        ]
        if not component_edges:
            continue

        raw_weights = np.asarray(
            [max(1e-8, float(edge["weight"])) for edge in component_edges],
            dtype=float,
        )
        weight_reference = max(1e-8, float(np.median(raw_weights)))

        rows = []
        rhs = []
        for edge in component_edges:
            i_id = edge["fixed_id"]
            j_id = edge["moving_id"]
            weight = math.sqrt(max(1e-8, float(edge["weight"])) / weight_reference)
            measured = _edge_measured_displacement(edge)
            for axis in range(3):
                row = np.zeros(3 * len(variables), dtype=float)
                if i_id != anchor:
                    row[3 * index[i_id] + axis] -= weight
                if j_id != anchor:
                    row[3 * index[j_id] + axis] += weight
                rows.append(row)
                rhs.append(weight * measured[axis])
        matrix = np.vstack(rows)
        vector = np.asarray(rhs, dtype=float)
        x0, *_ = np.linalg.lstsq(matrix, vector, rcond=None)

        measured_norms = [
            np.linalg.norm(_edge_measured_displacement(edge))
            for edge in component_edges
        ]
        residual_scale = max(1.0, float(np.median(measured_norms or [1.0])))

        def residual(params):
            values = []
            for edge in component_edges:
                i_id = edge["fixed_id"]
                j_id = edge["moving_id"]
                ti = np.zeros(3, dtype=float) if i_id == anchor else params[3 * index[i_id] : 3 * index[i_id] + 3]
                tj = np.zeros(3, dtype=float) if j_id == anchor else params[3 * index[j_id] : 3 * index[j_id] + 3]
                measured = _edge_measured_displacement(edge)
                weight = math.sqrt(max(1e-8, float(edge["weight"])) / weight_reference)
                values.extend((weight * (tj - ti - measured) / residual_scale).tolist())
            return np.asarray(values, dtype=float)

        robust = optimize.least_squares(
            residual,
            x0,
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=150,
        )
        solution = robust.x
        for tile_id in variables:
            corrections[tile_id] = _translation_matrix_xyz(
                solution[3 * index[tile_id] : 3 * index[tile_id] + 3]
            )
    return corrections, components


def _solve_rigid_pose_graph(tile_ids, edges):
    tile_ids = list(tile_ids)
    translation_start, components = _solve_translation_pose_graph(tile_ids, edges)
    corrections = {tile_id: np.eye(4, dtype=float) for tile_id in tile_ids}
    if not edges:
        return corrections, components
    try:
        from scipy import optimize
    except Exception as exc:
        raise RuntimeError("Rigid pose-graph optimization requires SciPy.") from exc

    for component in components:
        if len(component) <= 1:
            continue
        anchor = component[0]
        variables = [tile_id for tile_id in component if tile_id != anchor]
        index = {tile_id: i for i, tile_id in enumerate(variables)}
        component_edges = [
            edge for edge in edges
            if edge["fixed_id"] in component and edge["moving_id"] in component
        ]
        component_center = np.median(
            np.vstack([_edge_overlap_center(edge) for edge in component_edges]),
            axis=0,
        )
        local_edges = [
            _edge_rebased_to_origin(edge, component_center)
            for edge in component_edges
        ]
        x0 = np.zeros(6 * len(variables), dtype=float)
        for tile_id in variables:
            x0[6 * index[tile_id] : 6 * index[tile_id] + 3] = (
                _transform_displacement_at_point(
                    translation_start[tile_id], component_center
                )
            )
        translation_scale = np.median(
            [
                max(1e-6, np.linalg.norm(_edge_measured_displacement(edge)))
                for edge in component_edges
            ]
            or [1.0]
        )
        translation_scale = max(1.0, float(translation_scale))
        weight_reference = max(
            1e-8,
            float(np.median([max(1e-8, float(edge["weight"])) for edge in component_edges])),
        )

        def matrices(params):
            out = {anchor: np.eye(4, dtype=float)}
            for tile_id in variables:
                p = params[6 * index[tile_id] : 6 * index[tile_id] + 6]
                out[tile_id] = _rigid_matrix(p)
            return out

        def residual(params):
            mats = matrices(params)
            values = []
            for edge in local_edges:
                ci = mats[edge["fixed_id"]]
                cj = mats[edge["moving_id"]]
                measured = np.asarray(edge["correction"], dtype=float)
                relative = np.linalg.inv(ci) @ cj
                weight = math.sqrt(
                    max(1e-8, float(edge["weight"])) / weight_reference
                )
                translation_error = _edge_translation_error_vector(edge, ci, cj)
                values.extend((weight * translation_error / translation_scale).tolist())
                linear_error = np.linalg.inv(measured[:3, :3]) @ relative[:3, :3]
                rot_deg = _rotation_vector_degrees(linear_error)
                values.extend((weight * rot_deg / 5.0).tolist())
            return np.asarray(values, dtype=float)

        result = optimize.least_squares(
            residual,
            x0,
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=250,
        )
        solved = matrices(result.x)
        corrections.update({
            tile_id: _transform_rebased_to_world(matrix, component_center)
            for tile_id, matrix in solved.items()
        })

    return corrections, components


def _solve_affine_pose_graph(tile_ids, edges):
    """Solve a robust global 12-DOF affine correction graph.

    Pairwise affine edges are already bounded during image registration. The
    global stage uses direct 3x3 linear matrices plus translation for each
    non-anchor tile and a soft determinant barrier to avoid singular solutions.
    """
    tile_ids = list(tile_ids)
    rigid_start, components = _solve_rigid_pose_graph(tile_ids, edges)
    corrections = {tile_id: np.eye(4, dtype=float) for tile_id in tile_ids}
    if not edges:
        return corrections, components
    try:
        from scipy import optimize
    except Exception as exc:
        raise RuntimeError("Affine pose-graph optimization requires SciPy.") from exc

    for component in components:
        if len(component) <= 1:
            continue
        anchor = component[0]
        variables = [tile_id for tile_id in component if tile_id != anchor]
        index = {tile_id: i for i, tile_id in enumerate(variables)}
        component_edges = [
            edge for edge in edges
            if edge["fixed_id"] in component and edge["moving_id"] in component
        ]
        if not component_edges:
            continue

        translation_scale = max(
            1.0,
            float(np.median([
                max(1e-6, np.linalg.norm(_edge_measured_displacement(edge)))
                for edge in component_edges
            ] or [1.0])),
        )
        weight_reference = max(
            1e-8,
            float(np.median([
                max(1e-8, float(edge["weight"])) for edge in component_edges
            ])),
        )
        component_center = np.median(
            np.vstack([_edge_overlap_center(edge) for edge in component_edges]),
            axis=0,
        )
        local_edges = [
            _edge_rebased_to_origin(edge, component_center)
            for edge in component_edges
        ]
        x0 = np.zeros(12 * len(variables), dtype=float)
        for tile_id in variables:
            matrix = _transform_rebased_to_origin(
                rigid_start.get(tile_id, np.eye(4)), component_center
            )
            offset = 12 * index[tile_id]
            x0[offset : offset + 3] = matrix[:3, 3]
            x0[offset + 3 : offset + 12] = (
                matrix[:3, :3] - np.eye(3)
            ).reshape(-1)

        def matrices(params):
            out = {anchor: np.eye(4, dtype=float)}
            for tile_id in variables:
                offset = 12 * index[tile_id]
                matrix = np.eye(4, dtype=float)
                matrix[:3, :3] = np.eye(3) + params[offset + 3 : offset + 12].reshape(3, 3)
                matrix[:3, 3] = params[offset : offset + 3]
                out[tile_id] = matrix
            return out

        def residual(params):
            mats = matrices(params)
            values = []
            for edge in local_edges:
                ci = mats[edge["fixed_id"]]
                cj = mats[edge["moving_id"]]
                measured = np.asarray(edge["correction"], dtype=float)
                try:
                    relative = np.linalg.inv(ci) @ cj
                    linear_error = np.linalg.inv(measured[:3, :3]) @ relative[:3, :3]
                except np.linalg.LinAlgError:
                    values.extend([100.0] * 15)
                    continue
                weight = math.sqrt(
                    max(1e-8, float(edge["weight"])) / weight_reference
                )
                translation_error = _edge_translation_error_vector(edge, ci, cj)
                values.extend((weight * translation_error / translation_scale).tolist())
                rotation = _proper_rotation_from_linear(linear_error)
                rot_deg = _rotation_vector_degrees(rotation)
                values.extend((weight * rot_deg / 5.0).tolist())
                stretch = rotation.T @ linear_error
                values.extend((weight * (stretch - np.eye(3)).reshape(-1) / 0.05).tolist())

            # Constant-length soft barrier and weak regularization keep node
            # matrices invertible without suppressing genuine scale/shear.
            for tile_id in variables:
                matrix = mats[tile_id]
                determinant = float(np.linalg.det(matrix[:3, :3]))
                values.append(max(0.0, 0.10 - determinant) * 50.0)
                start = np.asarray(rigid_start.get(tile_id, np.eye(4)), dtype=float)
                start = _transform_rebased_to_origin(start, component_center)
                action_delta = matrix[:3, 3] - start[:3, 3]
                values.extend((0.002 * action_delta).tolist())
                values.extend(
                    (0.002 * (matrix[:3, :3] - start[:3, :3]).reshape(-1)).tolist()
                )
            return np.asarray(values, dtype=float)

        result = optimize.least_squares(
            residual,
            x0,
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=350,
        )
        solved = {
            tile_id: _transform_rebased_to_world(matrix, component_center)
            for tile_id, matrix in matrices(result.x).items()
        }
        for tile_id, matrix in solved.items():
            if tile_id == anchor:
                continue
            determinant = float(np.linalg.det(matrix[:3, :3]))
            if math.isfinite(determinant) and determinant > 1e-5:
                corrections[tile_id] = matrix
            else:
                corrections[tile_id] = rigid_start.get(tile_id, np.eye(4, dtype=float))

    return corrections, components


def _edge_residuals(edges, corrections):
    for edge in edges:
        ci = corrections[edge["fixed_id"]]
        cj = corrections[edge["moving_id"]]
        measured = np.asarray(edge["correction"], dtype=float)
        relative = np.linalg.inv(ci) @ cj
        linear_error = np.linalg.inv(measured[:3, :3]) @ relative[:3, :3]
        translation_error = _edge_translation_error_vector(edge, ci, cj)
        edge["global_translation_residual_xyz"] = translation_error.copy()
        edge["global_translation_residual"] = float(np.linalg.norm(translation_error))
        edge["global_rotation_residual_deg"] = float(
            np.linalg.norm(_rotation_vector_degrees(linear_error))
        )
        rotation = _proper_rotation_from_linear(linear_error)
        stretch = rotation.T @ linear_error
        edge["global_affine_residual"] = float(
            np.linalg.norm(stretch - np.eye(3), ord="fro")
        )


_POSE_GRAPH_QC_THRESHOLDS = {
    # These are review thresholds, not rejection thresholds. Half a registration
    # voxel catches loop disagreement at the scale of the established synthetic
    # accuracy contract without claiming a microscopy-specific acceptance limit.
    "translation_voxels": 0.5,
    "rotation_degrees": 0.5,
    "affine_frobenius": 0.01,
}


def _pose_graph_qc_summary(edges):
    """Summarize accepted-edge global consistency and phase ambiguity."""
    edges = list(edges)
    for edge in edges:
        residual_vector = np.asarray(
            edge.get("global_translation_residual_xyz", ()), dtype=float
        )
        grid_affine = np.asarray(edge.get("grid_affine", ()), dtype=float)
        spacing = np.asarray(edge.get("grid_spacing_xyz") or (), dtype=float)
        if residual_vector.shape == (3,) and np.all(np.isfinite(residual_vector)):
            if grid_affine.shape == (4, 4):
                grid_linear = grid_affine[:3, :3]
            elif spacing.shape == (3,) and np.all(np.isfinite(spacing)) and np.all(spacing > 0.0):
                grid_linear = np.diag(spacing)
            else:
                grid_linear = np.eye(3, dtype=float)
            try:
                residual_index = np.linalg.solve(grid_linear, residual_vector)
            except np.linalg.LinAlgError:
                residual_index = residual_vector
            edge["global_translation_residual_voxels_xyz"] = residual_index.copy()
            edge["global_translation_residual_voxels"] = float(
                np.linalg.norm(residual_index)
            )
        else:
            # Historical payloads retain only a scalar world residual and cannot
            # recover its physical direction for anisotropic conversion.
            finite_spacing = spacing[np.isfinite(spacing) & (spacing > 0.0)]
            voxel_scale = float(np.min(finite_spacing)) if finite_spacing.size else 1.0
            edge["global_translation_residual_voxels"] = float(
                edge.get("global_translation_residual", 0.0)
            ) / voxel_scale

    metric_specs = {
        "translation_voxels": (
            "global_translation_residual_voxels",
            _POSE_GRAPH_QC_THRESHOLDS["translation_voxels"],
        ),
        "rotation_degrees": (
            "global_rotation_residual_deg",
            _POSE_GRAPH_QC_THRESHOLDS["rotation_degrees"],
        ),
        "affine_frobenius": (
            "global_affine_residual",
            _POSE_GRAPH_QC_THRESHOLDS["affine_frobenius"],
        ),
    }
    metrics = {}
    residual_warning_edges = []
    for name, (field, threshold) in metric_specs.items():
        values = np.asarray([float(edge.get(field, 0.0)) for edge in edges])
        maximum = float(np.max(values)) if values.size else 0.0
        largest = [
            {
                "fixed_id": str(edge.get("fixed_id") or ""),
                "moving_id": str(edge.get("moving_id") or ""),
                "fixed_name": str(edge.get("fixed_name") or edge.get("fixed_id") or ""),
                "moving_name": str(edge.get("moving_name") or edge.get("moving_id") or ""),
                "value": float(value),
            }
            for edge, value in zip(edges, values)
            if values.size and np.isclose(value, maximum, rtol=1e-9, atol=1e-12)
        ]
        metrics[name] = {
            "count": int(values.size),
            "mean": float(np.mean(values)) if values.size else 0.0,
            "median": float(np.median(values)) if values.size else 0.0,
            "p95": float(np.percentile(values, 95.0)) if values.size else 0.0,
            "maximum": maximum,
            "warning_threshold": float(threshold),
            "largest_edges": largest,
        }
        if maximum > threshold:
            residual_warning_edges.extend(
                f"{record['fixed_id']}->{record['moving_id']}" for record in largest
            )

    ambiguous_edges = []
    evaluated_phase_edges = 0
    for edge in edges:
        phase = edge.get("phase_correlation")
        if not isinstance(phase, dict):
            continue
        evaluated_phase_edges += 1
        if bool(phase.get("ambiguous", True)):
            ambiguous_edges.append(
                {
                    "fixed_id": str(edge.get("fixed_id") or ""),
                    "moving_id": str(edge.get("moving_id") or ""),
                    "fixed_name": str(edge.get("fixed_name") or edge.get("fixed_id") or ""),
                    "moving_name": str(edge.get("moving_name") or edge.get("moving_id") or ""),
                    "valid": bool(phase.get("valid", False)),
                    "peak_ratio": phase.get("peak_ratio"),
                    "peak_prominence": phase.get("peak_prominence"),
                }
            )

    warning = bool(residual_warning_edges or ambiguous_edges)
    return {
        "status": "warning" if warning else "passed" if edges else "not-evaluated",
        "accepted_edge_count": len(edges),
        "metrics": metrics,
        "residual_warning_edges": sorted(set(residual_warning_edges)),
        "phase_correlation": {
            "evaluated_edge_count": int(evaluated_phase_edges),
            "ambiguous_edge_count": len(ambiguous_edges),
            "ambiguous_edges": ambiguous_edges,
            "warning_policy": (
                "Peak ratio below 1.5, relative prominence below 0.25, or "
                "invalid confidence triggers review but does not reject an edge."
            ),
        },
    }


def _translation_pose_graph_with_global_rejection(tile_ids, edges):
    """Solve translation constraints and conservatively reject unique outliers.

    Global residuals can identify disagreement but cannot always identify which
    member of a loop is wrong. Remove only a unique robust residual outlier whose
    removal preserves the graph components. Tied or otherwise non-identifiable
    disagreement remains explicit QC.
    """
    tile_ids = list(tile_ids)
    active_edges = list(edges)
    rejected = []
    iterations = []
    ambiguities = []
    initial_components = _connected_components(tile_ids, active_edges)
    initial_cycle_rank = max(
        0, len(active_edges) - len(tile_ids) + len(initial_components)
    )
    maximum_rejections = min(8, initial_cycle_rank)
    numerical_threshold = float(
        TRANSLATION_STITCHING_ACCEPTANCE_CONTRACT[
            "supported_fixture_accuracy_registration_voxels"
        ]
    )

    for iteration_index in range(maximum_rejections):
        corrections, components = _solve_translation_pose_graph(
            tile_ids, active_edges
        )
        _edge_residuals(active_edges, corrections)
        _pose_graph_qc_summary(active_edges)

        removable = []
        component_count = len(components)
        for edge_index, edge in enumerate(active_edges):
            remaining = active_edges[:edge_index] + active_edges[edge_index + 1 :]
            if len(_connected_components(tile_ids, remaining)) != component_count:
                continue
            removable.append(
                (
                    edge_index,
                    edge,
                    float(edge.get("global_translation_residual_voxels", 0.0)),
                )
            )

        values = np.asarray([record[2] for record in removable], dtype=float)
        median = float(np.median(values)) if values.size else 0.0
        mad = (
            float(np.median(np.abs(values - median))) if values.size else 0.0
        )
        normalized_mad = 1.4826 * mad
        cutoff = max(numerical_threshold, median + 3.0 * normalized_mad)
        outliers = [
            record for record in removable if record[2] > cutoff + 1e-9
        ]
        residual_warnings = [
            record for record in removable if record[2] > numerical_threshold
        ]
        audit = {
            "iteration": iteration_index + 1,
            "accepted_edge_count_before": len(active_edges),
            "component_count_before": component_count,
            "removable_edge_count": len(removable),
            "residual_voxels": {
                "median": median,
                "normalized_mad": normalized_mad,
                "outlier_cutoff": cutoff,
                "maximum": float(np.max(values)) if values.size else 0.0,
            },
            "candidate_edges": [
                {
                    "fixed_id": str(edge.get("fixed_id") or ""),
                    "moving_id": str(edge.get("moving_id") or ""),
                    "residual_voxels": residual,
                }
                for _index, edge, residual in outliers
            ],
        }
        iterations.append(audit)

        if not outliers:
            if residual_warnings:
                ambiguities.append(
                    {
                        "code": "global-inconsistency-not-identifiable",
                        "reason": (
                            "Translation constraints exceed the 0.5 registration-"
                            "voxel consistency threshold, but no unique robust "
                            "cycle outlier can be identified."
                        ),
                        "edges": [
                            f"{edge.get('fixed_id', '')}->{edge.get('moving_id', '')}"
                            for _index, edge, _residual in residual_warnings
                        ],
                    }
                )
            break

        maximum = max(record[2] for record in outliers)
        strongest = [
            record
            for record in outliers
            if math.isclose(record[2], maximum, rel_tol=1e-9, abs_tol=1e-9)
        ]
        if len(strongest) != 1:
            ambiguities.append(
                {
                    "code": "global-inconsistency-tied-outliers",
                    "reason": (
                        "Multiple translation constraints are tied as the largest "
                        "global residual; rejecting one would assign blame without "
                        "independent evidence."
                    ),
                    "edges": [
                        f"{edge.get('fixed_id', '')}->{edge.get('moving_id', '')}"
                        for _index, edge, _residual in strongest
                    ],
                }
            )
            break

        edge_index, rejected_edge, residual_voxels = strongest[0]
        rejected.append(
            {
                "iteration": iteration_index + 1,
                "criterion": (
                    "unique cycle-redundant residual > max(0.5 registration voxel, "
                    "median + 3 normalized MAD)"
                ),
                "outlier_cutoff_voxels": cutoff,
                "residual_voxels": residual_voxels,
                "constraint": copy.deepcopy(rejected_edge),
            }
        )
        audit["rejected_edge"] = {
            "fixed_id": str(rejected_edge.get("fixed_id") or ""),
            "moving_id": str(rejected_edge.get("moving_id") or ""),
            "residual_voxels": residual_voxels,
        }
        del active_edges[edge_index]

    corrections, components = _solve_translation_pose_graph(tile_ids, active_edges)
    _edge_residuals(active_edges, corrections)
    pose_graph_qc = _pose_graph_qc_summary(active_edges)
    final_cycle_rank = max(
        0, len(active_edges) - len(tile_ids) + len(components)
    )
    if (
        len(rejected) >= maximum_rejections
        and maximum_rejections > 0
        and pose_graph_qc["residual_warning_edges"]
    ):
        ambiguities.append(
            {
                "code": "global-inconsistency-rejection-bound-reached",
                "reason": (
                    "The bounded global inconsistency pass reached its rejection "
                    "limit while residual disagreement remained."
                ),
                "edges": list(pose_graph_qc["residual_warning_edges"]),
            }
        )

    global_qc = {
        "policy": TRANSLATION_STITCHING_ACCEPTANCE_CONTRACT[
            "global_inconsistency"
        ],
        "numerical_threshold_registration_voxels": numerical_threshold,
        "maximum_rejections": maximum_rejections,
        "iteration_count": len(iterations),
        "iterations": iterations,
        "rejected_edge_count": len(rejected),
        "rejected_edges": [
            {
                "fixed_id": str(item["constraint"].get("fixed_id") or ""),
                "moving_id": str(item["constraint"].get("moving_id") or ""),
                "iteration": item["iteration"],
                "residual_voxels": item["residual_voxels"],
                "outlier_cutoff_voxels": item["outlier_cutoff_voxels"],
            }
            for item in rejected
        ],
        "ambiguity_count": len(ambiguities),
        "ambiguities": ambiguities,
        "initial_component_count": len(initial_components),
        "final_component_count": len(components),
        "initial_cycle_rank": initial_cycle_rank,
        "final_cycle_rank": final_cycle_rank,
    }
    pose_graph_qc["component_count"] = len(components)
    pose_graph_qc["cycle_rank"] = final_cycle_rank
    pose_graph_qc["global_inconsistency"] = global_qc
    if rejected or ambiguities:
        pose_graph_qc["status"] = "warning"
    return corrections, components, active_edges, rejected, pose_graph_qc


def _source_corrections_from_mosaic(mosaic_corrections, preparation):
    conversions = {}
    for record in preparation.provenance.unit_conversions:
        tile_id = str(record.get("tile_id") or "")
        conversions.setdefault(
            tile_id,
            invertible_affine4(
                record.get("conversion_matrix"),
                f"Unit conversion for stitching tile {tile_id!r}",
            ),
        )
    source_corrections = {}
    for tile_id, correction in mosaic_corrections.items():
        conversion = conversions.get(str(tile_id), np.eye(4, dtype=float))
        source_corrections[str(tile_id)] = invertible_affine4(
            np.linalg.inv(conversion) @ np.asarray(correction, dtype=float) @ conversion,
            f"Source-frame stitching correction for tile {tile_id!r}",
        ).copy()
    return source_corrections


def _registration_geometry_provenance(
    preparation, mosaic_corrections, source_corrections
):
    payload = preparation.provenance.to_dict()
    for record in payload["source_tiles"]:
        tile_id = str(record.get("tile_id") or "")
        mosaic_correction = np.asarray(
            mosaic_corrections.get(tile_id, np.eye(4)), dtype=float
        )
        source_correction = np.asarray(
            source_corrections.get(tile_id, np.eye(4)), dtype=float
        )
        record["solved_correction_mosaic"] = _matrix_to_json(mosaic_correction)
        record["solved_correction_source"] = _matrix_to_json(source_correction)
        initial_mapping = np.asarray(record["initial_mosaic_mapping"], dtype=float)
        record["mosaic_space_mapping"] = _matrix_to_json(
            mosaic_correction @ initial_mapping
        )
    return payload


def validated_registration_settings(settings, subject="Stitching registration"):
    """Validate open-ended persisted settings at the registration boundary."""
    values = dict(settings or {})
    required = {
        "search_margin",
        "minimum_overlap_fraction",
        "minimum_score",
        "coarse_max_dim",
        "max_angle_deg",
        "max_scale_percent",
        "max_shear",
        "worker_count",
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"{subject} is missing settings: {', '.join(missing)}.")
    for key in (
        "search_margin",
        "minimum_overlap_fraction",
        "minimum_score",
        "max_angle_deg",
        "max_scale_percent",
        "max_shear",
    ):
        values[key] = float(values[key])
        if not math.isfinite(values[key]):
            raise ValueError(f"{subject} setting {key!r} must be finite.")
    if values["search_margin"] < 0.0:
        raise ValueError(f"{subject} search margin cannot be negative.")
    if not 0.0 <= values["minimum_overlap_fraction"] <= 1.0:
        raise ValueError(f"{subject} minimum overlap fraction must be within 0..1.")
    if not -1.0 <= values["minimum_score"] <= 1.0:
        raise ValueError(f"{subject} minimum NCC score must be within -1..1.")
    for key in ("max_angle_deg", "max_scale_percent", "max_shear"):
        if values[key] < 0.0:
            raise ValueError(f"{subject} setting {key!r} cannot be negative.")
    for key, minimum in (
        ("coarse_max_dim", 1),
        ("worker_count", 0),
        ("rigid_iterations", 1),
        ("affine_iterations", 1),
    ):
        if key not in values:
            continue
        numeric = float(values[key])
        if not math.isfinite(numeric):
            raise ValueError(f"{subject} setting {key!r} must be finite.")
        integer = int(numeric)
        if numeric != integer or integer < minimum:
            raise ValueError(
                f"{subject} setting {key!r} must be an integer of at least {minimum}."
            )
        values[key] = integer
    return values


class StitchRegistrationOperation:
    """GUI-independent registration operation driven by explicit callbacks."""

    def __init__(
        self,
        tiles,
        settings,
        mode,
        *,
        progress_callback=None,
        cancelled=None,
        completed_callback=None,
        failed_callback=None,
    ):
        self.tiles = tiles
        self.settings = dict(settings)
        self.mode = str(mode)
        self._progress_callback = progress_callback or (lambda _value, _text: None)
        self._cancelled = cancelled or (lambda: False)
        self._completed_callback = completed_callback or (lambda _result: None)
        self._failed_callback = failed_callback or (lambda _message: None)

    def run(self):
        try:
            self.settings = validated_registration_settings(self.settings)
            if self.mode not in {"translation", "rigid", "affine"}:
                raise ValueError(
                    f"Unsupported stitching registration mode {self.mode!r}."
                )
            preparation = _prepare_stitching_geometry(self.tiles)
            self.tiles = preparation.prepared_tiles
            registration_inputs = []
            for tile in self.tiles:
                support_min, support_max = _support_bounds(
                    tile["world_affine"], tile["dims"]
                )
                registration_inputs.append(
                    {
                        "tile_id": tile["tile_id"],
                        "display_name": tile["display_name"],
                        "world_affine": _matrix_to_json(tile["world_affine"]),
                        "support_bounds": {
                            "minimum_xyz": np.asarray(
                                support_min, dtype=float
                            ).tolist(),
                            "maximum_xyz": np.asarray(
                                support_max, dtype=float
                            ).tolist(),
                        },
                    }
                )
            pairs = _candidate_pairs(
                self.tiles,
                self.settings["search_margin"],
                self.settings["minimum_overlap_fraction"],
            )
            # Reference-enabled tiles can constrain neighbors. Fusion-only tiles
            # remain in the project/output but do not create edges solely between
            # one another. If only the second tile is a reference, orient the edge
            # so registration treats it as fixed.
            filtered_pairs = []
            for pair in pairs:
                a = self.tiles[pair["i"]]
                b = self.tiles[pair["j"]]
                a_ref = bool(a.get("reference_enabled", True))
                b_ref = bool(b.get("reference_enabled", True))
                if not (a_ref or b_ref):
                    continue
                pair = dict(pair)
                if b_ref and not a_ref:
                    pair["i"], pair["j"] = pair["j"], pair["i"]
                    pair["fixed_id"], pair["moving_id"] = pair["moving_id"], pair["fixed_id"]
                filtered_pairs.append(pair)
            pairs = filtered_pairs
            # Build the immutable broadcast support masks before pair tasks start,
            # so worker threads only read tile dictionaries.
            for tile in self.tiles:
                _tile_support_mask(tile)

            total = len(pairs)
            estimated_pair_bytes = max(
                (
                    _estimate_registration_pair_bytes(
                        self.tiles[pair["i"]],
                        self.tiles[pair["j"]],
                        pair,
                        self.settings,
                        self.mode,
                    )
                    for pair in pairs
                ),
                default=0,
            )
            worker_resolution = {}
            worker_count = _resolved_worker_count(
                self.settings.get("worker_count", 0),
                total,
                auto_cap=8,
                estimated_bytes_per_job=estimated_pair_bytes,
                diagnostics=worker_resolution,
            )
            self.settings["resolved_worker_count"] = int(worker_count)
            self.settings["worker_resolution"] = worker_resolution
            self._progress_callback(
                0,
                f"Registering {total} candidate pair(s) with {worker_count} CPU worker(s)",
            )

            edges = []
            pair_failures = []
            rejections = []
            pair_evaluations = []
            score_rejections = 0
            invalid_pair_rejections = 0
            minimum_score = float(self.settings["minimum_score"])
            recoverable_pair_errors = (
                np.linalg.LinAlgError,
                FloatingPointError,
                OverflowError,
                ValueError,
                RuntimeError,
            )

            def register_pair(pair):
                if self._cancelled():
                    raise InterruptedError
                a = self.tiles[pair["i"]]
                b = self.tiles[pair["j"]]
                edge = _register_pair(a, b, pair, self.settings, self.mode)
                return pair, edge

            def pair_evaluation(pair, outcome, *, edge=None, error=None):
                a = self.tiles[pair["i"]]
                b = self.tiles[pair["j"]]
                record = {
                    "fixed_id": a["tile_id"],
                    "moving_id": b["tile_id"],
                    "fixed_name": a["display_name"],
                    "moving_name": b["display_name"],
                    "actual_overlap_fraction": float(
                        pair["actual_overlap_fraction"]
                    ),
                    "search_bounds": {
                        "minimum_xyz": np.asarray(
                            pair["search_lo"], dtype=float
                        ).tolist(),
                        "maximum_xyz": np.asarray(
                            pair["search_hi"], dtype=float
                        ).tolist(),
                    },
                    "minimum_accepted_ncc": minimum_score,
                    "outcome": str(outcome),
                }
                if edge is not None:
                    record.update(
                        {
                            "phase_correlation_translation_xyz": np.asarray(
                                edge.get(
                                    "phase_translation_xyz",
                                    edge.get("translation_xyz", (0.0, 0.0, 0.0)),
                                ),
                                dtype=float,
                            ).tolist(),
                            "final_translation_xyz": np.asarray(
                                edge.get("translation_xyz", (0.0, 0.0, 0.0)),
                                dtype=float,
                            ).tolist(),
                            "initial_ncc": float(
                                edge.get("initial_score", edge["translation_score"])
                            ),
                            "translation_ncc": float(edge["translation_score"]),
                            "final_ncc": float(edge["score"]),
                            "translation_grid_voxels_xyz": np.asarray(
                                edge.get(
                                    "translation_grid_voxels_xyz",
                                    (0.0, 0.0, 0.0),
                                ),
                                dtype=float,
                            ).tolist(),
                            "overlap_voxels": int(edge["overlap_voxels"]),
                            "preprocessing": copy.deepcopy(
                                edge.get("preprocessing") or {}
                            ),
                            "phase_correlation": copy.deepcopy(
                                edge.get("phase_correlation") or {}
                            ),
                        }
                    )
                if error is not None:
                    record["error_type"] = type(error).__name__
                    record["error"] = str(error)
                pair_evaluations.append(record)
                return record

            def record_pair_failure(pair, exc, completed_count):
                a = self.tiles[pair["i"]]
                b = self.tiles[pair["j"]]
                failure = {
                    "fixed_id": a["tile_id"],
                    "moving_id": b["tile_id"],
                    "fixed_name": a["display_name"],
                    "moving_name": b["display_name"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                pair_failures.append(failure)
                pair_evaluation(pair, "numerical_failure", error=exc)
                rejections.append(
                    StitchingRejection(
                        code="numerical_failure",
                        reason=f"{type(exc).__name__}: {exc}",
                        fixed_id=a["tile_id"],
                        moving_id=b["tile_id"],
                        fixed_name=a["display_name"],
                        moving_name=b["display_name"],
                    )
                )
                self._progress_callback(
                    int(completed_count * 75 / total),
                    f"Rejected {a['display_name']} ↔ {b['display_name']} after "
                    f"{type(exc).__name__}: {exc}",
                )

            def record_score_rejection(pair, edge):
                nonlocal score_rejections
                a = self.tiles[pair["i"]]
                b = self.tiles[pair["j"]]
                score_rejections += 1
                evaluation = pair_evaluation(
                    pair, "below_minimum_score", edge=edge
                )
                rejections.append(
                    StitchingRejection(
                        code="below_minimum_score",
                        reason=(
                            f"Registration score {edge['score']:.6g} is below "
                            f"the required minimum {minimum_score:.6g}."
                        ),
                        fixed_id=a["tile_id"],
                        moving_id=b["tile_id"],
                        fixed_name=a["display_name"],
                        moving_name=b["display_name"],
                        score=edge["score"],
                        details={
                            "constraint": copy.deepcopy(edge),
                            "pair_evaluation": copy.deepcopy(evaluation),
                        },
                    )
                )

            def record_invalid_pair_rejection(pair, edge):
                nonlocal invalid_pair_rejections
                a = self.tiles[pair["i"]]
                b = self.tiles[pair["j"]]
                invalid_pair_rejections += 1
                evaluation = pair_evaluation(
                    pair, "invalid_phase_correlation", edge=edge
                )
                rejections.append(
                    StitchingRejection(
                        code="invalid_phase_correlation",
                        reason=(
                            "Phase correlation did not produce numerically valid "
                            "confidence for this pair."
                        ),
                        fixed_id=a["tile_id"],
                        moving_id=b["tile_id"],
                        fixed_name=a["display_name"],
                        moving_name=b["display_name"],
                        score=edge["score"],
                        details={
                            "constraint": copy.deepcopy(edge),
                            "phase_correlation": copy.deepcopy(
                                edge.get("phase_correlation") or {}
                            ),
                            "pair_evaluation": copy.deepcopy(evaluation),
                        },
                    )
                )

            def accept_or_reject_pair(pair, edge):
                phase_valid = (edge.get("phase_correlation") or {}).get("valid")
                if phase_valid is False:
                    record_invalid_pair_rejection(pair, edge)
                elif edge["score"] >= minimum_score:
                    edges.append(edge)
                    pair_evaluation(pair, "accepted", edge=edge)
                else:
                    record_score_rejection(pair, edge)

            with _limit_native_threadpools(worker_count > 1):
                if worker_count == 1:
                    for completed_count, pair in enumerate(pairs, 1):
                        if self._cancelled():
                            return
                        a = self.tiles[pair["i"]]
                        b = self.tiles[pair["j"]]
                        self._progress_callback(
                            int((completed_count - 1) * 75 / total),
                            f"Registering {a['display_name']} ↔ {b['display_name']}",
                        )
                        try:
                            _pair, edge = register_pair(pair)
                        except recoverable_pair_errors as exc:
                            record_pair_failure(pair, exc, completed_count)
                            continue
                        accept_or_reject_pair(pair, edge)
                else:
                    executor = ThreadPoolExecutor(
                        max_workers=worker_count,
                        thread_name_prefix="MADI3D-Stitch-Reg",
                    )
                    futures = {
                        executor.submit(register_pair, pair): pair
                        for pair in pairs
                    }
                    try:
                        completed_count = 0
                        for future in as_completed(futures):
                            if self._cancelled():
                                for pending in futures:
                                    pending.cancel()
                                return
                            try:
                                pair, edge = future.result()
                            except recoverable_pair_errors as exc:
                                pair = futures[future]
                                completed_count += 1
                                record_pair_failure(pair, exc, completed_count)
                                continue
                            completed_count += 1
                            accept_or_reject_pair(pair, edge)
                            a = self.tiles[pair["i"]]
                            b = self.tiles[pair["j"]]
                            self._progress_callback(
                                int(completed_count * 75 / total),
                                f"Completed {completed_count}/{total}: "
                                f"{a['display_name']} ↔ {b['display_name']}",
                            )
                    finally:
                        executor.shutdown(wait=True, cancel_futures=True)

            # Parallel completion order is nondeterministic; keep pose-graph
            # residual ordering and project JSON stable across runs.
            edges.sort(key=lambda edge: (
                str(edge.get("fixed_id", "")),
                str(edge.get("moving_id", "")),
            ))
            pair_evaluations.sort(
                key=lambda record: (
                    str(record.get("fixed_id", "")),
                    str(record.get("moving_id", "")),
                )
            )

            candidate_pairs = []
            for pair in pairs:
                a = self.tiles[pair["i"]]
                b = self.tiles[pair["j"]]
                candidate_pairs.append(
                    {
                        "fixed_id": a["tile_id"],
                        "moving_id": b["tile_id"],
                        "fixed_name": a["display_name"],
                        "moving_name": b["display_name"],
                        "actual_overlap_fraction": float(
                            pair["actual_overlap_fraction"]
                        ),
                        "search_bounds": {
                            "minimum_xyz": np.asarray(
                                pair["search_lo"], dtype=float
                            ).tolist(),
                            "maximum_xyz": np.asarray(
                                pair["search_hi"], dtype=float
                            ).tolist(),
                        },
                    }
                )

            tile_ids = [tile["tile_id"] for tile in self.tiles]
            self._progress_callback(80, "Optimizing the global tile layout")
            globally_rejected = []
            model_rank = {"translation": 0, "rigid": 1, "affine": 2}
            selected_edge_models = [
                str(edge.get("selected_model") or edge.get("mode") or "translation")
                for edge in edges
            ]
            global_mode = max(
                selected_edge_models or ["translation"],
                key=lambda value: model_rank.get(value, 0),
            )
            if global_mode == "affine":
                corrections, components = _solve_affine_pose_graph(tile_ids, edges)
            elif global_mode == "rigid":
                corrections, components = _solve_rigid_pose_graph(tile_ids, edges)
            else:
                (
                    corrections,
                    components,
                    edges,
                    globally_rejected,
                    pose_graph_qc,
                ) = _translation_pose_graph_with_global_rejection(tile_ids, edges)
            if global_mode != "translation":
                _edge_residuals(edges, corrections)
                pose_graph_qc = _pose_graph_qc_summary(edges)

            evaluation_by_pair = {
                (record["fixed_id"], record["moving_id"]): record
                for record in pair_evaluations
            }
            for rejected_record in globally_rejected:
                constraint = rejected_record["constraint"]
                fixed_id = str(constraint.get("fixed_id") or "")
                moving_id = str(constraint.get("moving_id") or "")
                evaluation = evaluation_by_pair.get((fixed_id, moving_id))
                if evaluation is not None:
                    evaluation["outcome"] = "global_inconsistency_rejected"
                    evaluation["global_rejection"] = {
                        key: copy.deepcopy(value)
                        for key, value in rejected_record.items()
                        if key != "constraint"
                    }
                rejection_details = copy.deepcopy(rejected_record)
                if evaluation is not None:
                    rejection_details["pair_evaluation"] = copy.deepcopy(
                        evaluation
                    )
                rejections.append(
                    StitchingRejection(
                        code="global_inconsistency_outlier",
                        reason=(
                            "Rejected globally inconsistent translation constraint "
                            f"at {rejected_record['residual_voxels']:.6g} "
                            "registration voxels (cutoff "
                            f"{rejected_record['outlier_cutoff_voxels']:.6g})."
                        ),
                        fixed_id=fixed_id,
                        moving_id=moving_id,
                        fixed_name=str(constraint.get("fixed_name") or fixed_id),
                        moving_name=str(constraint.get("moving_name") or moving_id),
                        score=constraint.get("score"),
                        details=rejection_details,
                    )
                )
            mosaic_corrections = corrections
            source_corrections = _source_corrections_from_mosaic(
                mosaic_corrections, preparation
            )

            execution_warnings = list(preparation.warnings)
            if global_mode != self.mode:
                execution_warnings.append(
                    {
                        "code": "registration-model-complexity-not-supported",
                        "message": (
                            f"{self.mode.title()} registration was requested, but "
                            f"the accepted pair evidence supported only {global_mode}. "
                            "The global solution used the simpler model."
                        ),
                        "requested_model": self.mode,
                        "selected_model": global_mode,
                    }
                )
            selected_bound_hits = []
            for edge in edges:
                selection = edge.get("model_selection") or {}
                selected = str(selection.get("selected_model") or "translation")
                candidate = (selection.get("candidates") or {}).get(selected) or {}
                for bound_name in candidate.get("bound_hits") or []:
                    selected_bound_hits.append(
                        f"{edge['fixed_id']}->{edge['moving_id']}:{bound_name}"
                    )
            if selected_bound_hits:
                execution_warnings.append(
                    {
                        "code": "registration-model-bound-hit",
                        "message": (
                            "Accepted advanced registration parameters reached a "
                            "configured deformation bound. Review: "
                            + ", ".join(selected_bound_hits)
                        ),
                        "bound_hits": selected_bound_hits,
                    }
                )
            if not pairs:
                execution_warnings.append(
                    {
                        "code": "no-candidate-registration-edges",
                        "message": (
                            "No candidate overlap was found from the current tile poses "
                            "and configured overlap extent. No alignment was obtained; "
                            "the images were not registered."
                        ),
                    }
                )
            elif not edges:
                if (
                    score_rejections
                    and not pair_failures
                    and not invalid_pair_rejections
                ):
                    rejected_scores = [
                        record["final_ncc"]
                        for record in pair_evaluations
                        if record["outcome"] == "below_minimum_score"
                    ]
                    if len(rejected_scores) == 1:
                        detail = (
                            "1 candidate pair was evaluated, but its NCC "
                            f"{rejected_scores[0]:.6g} was below the required "
                            f"{minimum_score:.6g}."
                        )
                    else:
                        detail = (
                            f"{score_rejections} candidate pairs were below the "
                            f"required NCC {minimum_score:.6g}."
                        )
                elif (
                    pair_failures
                    and not score_rejections
                    and not invalid_pair_rejections
                ):
                    detail = (
                        f"{len(pair_failures)} candidate pair(s) were found, but "
                        "registration failed numerically. Review the rejected pair "
                        "errors."
                    )
                elif (
                    invalid_pair_rejections
                    and not score_rejections
                    and not pair_failures
                ):
                    detail = (
                        f"{invalid_pair_rejections} candidate pair(s) had invalid "
                        "phase-correlation confidence. Review the rejected pair "
                        "diagnostics."
                    )
                else:
                    detail = (
                        f"{score_rejections} pair(s) were below the required NCC and "
                        f"{invalid_pair_rejections} pair(s) had invalid phase "
                        f"confidence; {len(pair_failures)} pair(s) failed numerically."
                    )
                execution_warnings.append(
                    {
                        "code": "no-accepted-registration-edges",
                        "message": (
                            detail
                            + " No alignment was obtained; the images were not "
                            "registered and their existing poses remain unchanged."
                        ),
                    }
                )
            elif len(components) > 1:
                execution_warnings.append(
                    {
                        "code": "disconnected-registration-components",
                        "message": (
                            f"The accepted registration graph has {len(components)} "
                            "components; each component retained its initial placement."
                        ),
                    }
                )
            if pose_graph_qc["residual_warning_edges"]:
                execution_warnings.append(
                    {
                        "code": "pose-graph-residual-warning",
                        "message": (
                            "The accepted constraints are globally inconsistent at "
                            "or above the conservative review threshold. Review edge(s): "
                            + ", ".join(pose_graph_qc["residual_warning_edges"])
                        ),
                        "edges": pose_graph_qc["residual_warning_edges"],
                    }
                )
            global_inconsistency = pose_graph_qc.get("global_inconsistency") or {}
            globally_rejected_edges = (
                global_inconsistency.get("rejected_edges") or []
            )
            if globally_rejected_edges:
                edge_labels = [
                    f"{edge['fixed_id']}->{edge['moving_id']}"
                    for edge in globally_rejected_edges
                ]
                execution_warnings.append(
                    {
                        "code": "global-inconsistency-edges-rejected",
                        "message": (
                            "Globally inconsistent translation constraint(s) were "
                            "removed and the mosaic was re-solved: "
                            + ", ".join(edge_labels)
                        ),
                        "edges": edge_labels,
                    }
                )
            global_ambiguities = global_inconsistency.get("ambiguities") or []
            if global_ambiguities:
                execution_warnings.append(
                    {
                        "code": "global-inconsistency-ambiguous",
                        "message": (
                            "Global translation disagreement remains ambiguous; "
                            "no constraint was discarded without independent evidence."
                        ),
                        "ambiguities": copy.deepcopy(global_ambiguities),
                    }
                )
            ambiguous_phase = pose_graph_qc["phase_correlation"]["ambiguous_edges"]
            if ambiguous_phase:
                edge_labels = [
                    f"{edge['fixed_id']}->{edge['moving_id']}"
                    for edge in ambiguous_phase
                ]
                execution_warnings.append(
                    {
                        "code": "ambiguous-phase-correlation",
                        "message": (
                            "Phase correlation did not have a unique interpretable "
                            "peak for edge(s): " + ", ".join(edge_labels)
                        ),
                        "edges": edge_labels,
                    }
                )

            result = {
                "algorithm_version": STITCHING_ALGORITHM_VERSION,
                "execution_status": "succeeded",
                "qc_status": (
                    "warning"
                    if execution_warnings or preparation.assumptions or rejections
                    else "passed"
                ),
                "user_decision": "unapplied",
                "mode": global_mode,
                "requested_mode": self.mode,
                "corrections": source_corrections,
                "mosaic_corrections": mosaic_corrections,
                "edges": edges,
                "components": components,
                "pose_graph_qc": pose_graph_qc,
                "translation_acceptance_contract": (
                    copy.deepcopy(TRANSLATION_STITCHING_ACCEPTANCE_CONTRACT)
                    if global_mode == "translation"
                    else None
                ),
                "tile_ids": tile_ids,
                "registration_channel_key": self.settings.get(
                    "registration_channel_key"
                ),
                "registration_channel_label": self.settings.get(
                    "registration_channel_label"
                ),
                "registration_inputs": registration_inputs,
                "candidate_pairs": candidate_pairs,
                "pair_evaluations": pair_evaluations,
                "candidate_pair_count": len(pairs),
                "evaluated_pair_count": len(pair_evaluations),
                "accepted_edge_count": len(edges),
                "below_minimum_score_count": score_rejections,
                "invalid_pair_count": invalid_pair_rejections,
                "global_inconsistency_rejected_count": len(globally_rejected),
                "numerical_failure_count": len(pair_failures),
                "pair_failures": pair_failures,
                "rejections": [item.to_dict() for item in rejections],
                "settings": dict(self.settings),
                "mosaic_geometry": _registration_geometry_provenance(
                    preparation, mosaic_corrections, source_corrections
                ),
                "warnings": execution_warnings,
                "assumptions": preparation.assumptions,
                "completed_with_warnings": bool(
                    execution_warnings or preparation.assumptions or rejections
                ),
            }
            failure_note = (
                f"; {len(pair_failures)} numerical pair failure(s) rejected"
                if pair_failures
                else ""
            )
            self._progress_callback(
                100,
                (
                    "Registration completed with warnings"
                    if result["completed_with_warnings"]
                    else "Registration complete"
                )
                + f" using {worker_count} CPU worker(s){failure_note}",
            )
            self._completed_callback(StitchingRegistrationResult.from_runtime(result))
        except InterruptedError:
            return
        except Exception:
            self._failed_callback(traceback.format_exc())


# -----------------------------------------------------------------------------
# Fusion engine
# -----------------------------------------------------------------------------

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
    channel_tiles = _prepared_tiles(channel_tiles)
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
    grid, dtype, output_dir, *, channel_count=1, simultaneous_copies=2
):
    dims = tuple(int(value) for value in grid["dims"])
    voxels = math.prod(dims) * max(1, int(channel_count))
    itemsize = int(np.dtype(dtype).itemsize)
    if voxels <= 0 or voxels > np.iinfo(np.intp).max // max(1, itemsize):
        raise ValueError(
            "The stitching output grid is too large for this platform's array index space."
        )
    output_bytes = voxels * itemsize
    required_bytes = output_bytes * max(1, int(simultaneous_copies))
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


def _tile_chunk_contribution(
    tile,
    *,
    fusion_mode,
    interpolation,
    chunk_affine,
    chunk_shape,
    working_dtype,
    cancelled=lambda: False,
):
    """Resample one tile contribution. Safe to execute in a worker thread."""
    if cancelled():
        raise InterruptedError("Stitching fusion was cancelled.")

    mask, distance = _tile_physical_support(
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
    values = _resample_zyx(
        tile["data"],
        tile["world_affine"],
        chunk_affine,
        chunk_shape,
        order=value_order,
        output_dtype=working_dtype,
        mode="nearest",
    )
    return mask, values, weights


def _partial_output_path(path):
    path = Path(path)
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return path.with_name(name[:-7] + ".partial.nii.gz")
    return path.with_name(path.stem + ".partial" + path.suffix)


def _hidden_temporary_output_path(path, purpose="partial"):
    """Return a unique sibling path that retains the scientific file suffix."""
    path = Path(path)
    if path.name.lower().endswith(".nii.gz"):
        stem, suffix = path.name[:-7], ".nii.gz"
    else:
        stem, suffix = path.stem, path.suffix
    return path.with_name(
        f".{stem}.{uuid.uuid4().hex}.{_safe_stem(purpose)}{suffix}"
    )


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
    writer_callback=None,
    ffmpeg_executable=None,
    output_array=None,
    defer_write=False,
):
    validated_options = _validate_fusion_options(options)
    channel_tiles = _prepared_tiles(channel_tiles)
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
        simultaneous_copies=2 if output_array is None else 1,
    )

    tile_bounds = []
    for tile in channel_tiles:
        lo, hi = _support_bounds(tile["world_affine"], tile["dims"])
        tile_bounds.append((lo, hi))

    total_chunks = max(1, math.ceil(nz / chunk_depth))
    chunk_voxels = max(1, int(chunk_depth) * int(ny) * int(nx))
    working_itemsize = int(working_dtype.itemsize)
    if fusion_mode == "distance_weighted":
        # One Float64 source-coordinate field is built at a time. Returned
        # distance/value arrays use the selected working precision.
        estimated_bytes_per_job = chunk_voxels * (12 + 3 * working_itemsize)
    else:
        estimated_bytes_per_job = chunk_voxels * (2 + 2 * working_itemsize)
    minimum_chunk_working_bytes = estimated_bytes_per_job * 2
    available_memory = _available_memory_bytes()
    if (
        available_memory > 0
        and minimum_chunk_working_bytes > int(available_memory * 0.75)
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
        }
    )
    worker_resolution = {}
    worker_count = _resolved_worker_count(
        options.get("worker_count", 0),
        len(channel_tiles),
        auto_cap=8,
        estimated_bytes_per_job=estimated_bytes_per_job,
        diagnostics=worker_resolution,
    )
    grid["resolved_worker_count"] = int(worker_count)
    grid["worker_resolution"] = worker_resolution
    grid["resource_estimate"]["worker_resolution"] = copy.deepcopy(
        worker_resolution
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _partial_output_path(output_path)
    data_temp = None if output_array is not None else output_path.with_name(
        "." + output_path.name.replace(os.sep, "_") + f".{uuid.uuid4().hex}.fusion.dat"
    )
    for candidate in (temporary, data_temp):
        if candidate is None:
            continue
        try:
            if candidate.exists():
                candidate.unlink()
        except Exception:
            pass

    executor = None
    output_map = None
    owns_output_map = output_array is None
    try:
        if output_array is None:
            output_map = np.memmap(
                data_temp,
                mode="w+",
                dtype=output_dtype,
                shape=(int(nz), int(ny), int(nx)),
                order="C",
            )
        else:
            output_map = np.asanyarray(output_array)
            if tuple(output_map.shape) != (int(nz), int(ny), int(nx)):
                raise RuntimeError(
                    f"Shared multichannel output shape {output_map.shape} does not match fusion grid {(nz, ny, nx)}."
                )
        if worker_count > 1:
            executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="MADI3D-Stitch-Fuse",
            )

        with _limit_native_threadpools(worker_count > 1):
            for chunk_index, z0 in enumerate(range(0, nz, chunk_depth), 1):
                if cancelled():
                    raise InterruptedError("Stitching fusion was cancelled.")
                z1 = min(nz, z0 + chunk_depth)
                depth = z1 - z0
                chunk_affine = np.asarray(grid["affine"], dtype=float).copy()
                chunk_affine[:3, 3] += chunk_affine[:3, 2] * z0
                chunk_shape = (depth, ny, nx)

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
                        cancelled=cancelled,
                    )

                def accumulate(result):
                    if result is None:
                        return
                    mask, values, weights = result
                    if fusion_mode == "max":
                        first = mask & ~valid_any
                        overlap = mask & valid_any
                        accumulator[first] = values[first]
                        accumulator[overlap] = np.maximum(
                            accumulator[overlap], values[overlap]
                        )
                        valid_any[mask] = True
                    else:
                        weighted_sum[:] += values * weights
                        weight_sum[:] += weights
                        if fusion_mode == "distance_weighted":
                            value_sum[mask] += values[mask]
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
                            del futures
                else:
                    for tile in active_tiles:
                        accumulate(contribution(tile))

                if fusion_mode == "max":
                    out = np.zeros(chunk_shape, dtype=working_dtype)
                    out[valid_any] = accumulator[valid_any]
                else:
                    out = np.zeros(chunk_shape, dtype=working_dtype)
                    if fusion_mode == "distance_weighted":
                        single = coverage_count == 1
                        out[single] = value_sum[single]
                        weighted = (coverage_count > 1) & (weight_sum > 0.0)
                        out[weighted] = weighted_sum[weighted] / weight_sum[weighted]
                        # Samples can lie exactly on multiple support faces, so
                        # every physical interior distance is legitimately zero.
                        # Their unweighted mean is deterministic and avoids an
                        # arbitrary macroscopic blend-width epsilon.
                        zero_weight_overlap = (coverage_count > 1) & ~weighted
                        out[zero_weight_overlap] = (
                            value_sum[zero_weight_overlap]
                            / coverage_count[zero_weight_overlap]
                        )
                    else:
                        valid = weight_sum > 0.0
                        out[valid] = weighted_sum[valid] / weight_sum[valid]

                output_map[z0:z1] = _cast_output(out, output_dtype)
                progress_callback(chunk_index, total_chunks, worker_count)

        if hasattr(output_map, "flush"):
            output_map.flush()
        if cancelled():
            raise InterruptedError("Stitching fusion was cancelled.")
        if not defer_write:
            if writer_callback is None:
                raise RuntimeError(
                    "Stitching fusion requires MADI3D's canonical injected volume writer."
                )
            writer_kwargs = {}
            if str(options.get("output_format", "nrrd")).lower() == "h5j":
                writer_kwargs = {
                    "ffmpeg_executable": ffmpeg_executable,
                    "cancel_check": cancelled,
                }
            writer_callback(
                str(temporary),
                output_map,
                str(options.get("output_format", "nrrd")),
                grid,
                str(options.get("channel_label", "")),
                dict(options),
                channel_tiles[0].get("space_units"),
                **writer_kwargs,
            )
            # Explicitly close our private mmap before deleting its backing file.
            if owns_output_map:
                mmap_obj = getattr(output_map, "_mmap", None)
                if mmap_obj is not None:
                    mmap_obj.close()
                output_map = None
            os.replace(temporary, output_path)
    except Exception:
        try:
            if temporary.exists():
                temporary.unlink()
        except Exception:
            pass
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if output_map is not None and owns_output_map:
            try:
                output_map.flush()
            except Exception:
                pass
            try:
                mmap_obj = getattr(output_map, "_mmap", None)
                if mmap_obj is not None:
                    mmap_obj.close()
            except Exception:
                pass
            output_map = None
        if data_temp is not None:
            try:
                if data_temp.exists():
                    data_temp.unlink()
            except Exception:
                pass
    return grid


class StitchFusionOperation:
    """GUI-independent fusion operation driven by explicit callbacks."""

    def __init__(
        self, channel_sets, output_dir, base_name, options, project_payload,
        writer_callback=None, bundle_writer_callback=None,
        ffmpeg_executable=None, *, progress_callback=None, cancelled=None,
        completed_callback=None, failed_callback=None,
    ):
        self.channel_sets = channel_sets
        self.output_dir = str(output_dir)
        self.base_name = _safe_stem(base_name)
        self.options = dict(options)
        self.project_payload = dict(project_payload)
        self.writer_callback = writer_callback
        self.bundle_writer_callback = bundle_writer_callback
        self.ffmpeg_executable = os.fspath(ffmpeg_executable) if ffmpeg_executable else None
        self._progress_callback = progress_callback or (lambda _value, _text: None)
        self._cancelled = cancelled or (lambda: False)
        self._completed_callback = completed_callback or (lambda _result: None)
        self._failed_callback = failed_callback or (lambda _message: None)

    @staticmethod
    def _unique_output_path(path):
        path = Path(path)
        if not path.exists():
            return path
        if path.name.lower().endswith(".nii.gz"):
            stem, ext = path.name[:-7], ".nii.gz"
        else:
            stem, ext = path.stem, path.suffix
        counter = 2
        while True:
            candidate = path.with_name(f"{stem}_{counter}{ext}")
            if not candidate.exists():
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
        try:
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
            preparation = _prepare_stitching_geometry(
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
                combined_temp = out_dir / f".{self.base_name}.{uuid.uuid4().hex}.multichannel-fusion.dat"
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
                        out_dir / f".{self.base_name}.{channel_index}.stage{extension}",
                        local_options, update, self._cancelled,
                        writer_callback=self.writer_callback,
                        ffmpeg_executable=self.ffmpeg_executable,
                        output_array=combined_map[channel_index - 1],
                        defer_write=True,
                    )
                    resolved_workers.append(int(grid.get("resolved_worker_count", 1)))
                    worker_resolutions.append(
                        copy.deepcopy(grid.get("worker_resolution") or {})
                    )

                bundle_grid["resource_estimate"][
                    "channel_worker_resolutions"
                ] = worker_resolutions

                combined_map.flush()
                requested_path = out_dir / f"{self.base_name}{extension}"
                bundle_temporary = _hidden_temporary_output_path(
                    requested_path, "multichannel"
                )
                labels = [str(channel["label"]) for channel in self.channel_sets]
                self._progress_callback(97, f"Writing multichannel {output_format.upper()} source")
                bundle_writer_kwargs = {}
                if output_format.lower() == "h5j":
                    bundle_writer_kwargs = {
                        "ffmpeg_executable": self.ffmpeg_executable,
                        "cancel_check": self._cancelled,
                    }
                bundle_options = dict(self.options)
                bundle_options["stitching_mosaic_geometry"] = mosaic_provenance
                self.bundle_writer_callback(
                    str(bundle_temporary), combined_map, output_format, bundle_grid,
                    labels, bundle_options,
                    bundle_grid.get("space_units"),
                    **bundle_writer_kwargs,
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
                if self._cancelled():
                    raise InterruptedError("Stitching fusion was cancelled.")
                os.replace(bundle_temporary, final_path)
                bundle_temporary = None
                published_outputs.append(final_path)
                outputs.append({
                    "path": str(final_path),
                    "label": "multichannel",
                    "channels": labels,
                    "multichannel": True,
                    "grid": {
                        "dims": list(bundle_grid["dims"]),
                        "origin": np.asarray(bundle_grid["origin"], dtype=float).tolist(),
                        "spacing": np.asarray(bundle_grid["spacing"], dtype=float).tolist(),
                        "affine": _matrix_to_json(bundle_grid["affine"]),
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
                })
            else:
                for channel_index, channel in enumerate(self.channel_sets, 1):
                    if self._cancelled():
                        raise InterruptedError("Stitching fusion was cancelled.")
                    label = str(channel["label"])
                    suffix = "" if channel_count == 1 else "_" + _safe_stem(label)
                    requested_path = out_dir / f"{self.base_name}{suffix}{extension}"
                    staged_path = _hidden_temporary_output_path(
                        requested_path, "scalar"
                    )
                    scalar_temporaries.append(staged_path)
                    local_options = dict(self.options)
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

                    grid = _fuse_one_channel(
                        channel["tiles"], staged_path, local_options, update,
                        self._cancelled, writer_callback=self.writer_callback,
                        ffmpeg_executable=self.ffmpeg_executable,
                    )
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
                    path = self._unique_output_path(requested_path)
                    if self._cancelled():
                        raise InterruptedError("Stitching fusion was cancelled.")
                    os.replace(staged_path, path)
                    scalar_temporaries.remove(staged_path)
                    published_outputs.append(path)
                    outputs.append({
                        "path": str(path),
                        "label": label,
                        "multichannel": False,
                        "grid": {
                            "dims": list(grid["dims"]),
                            "origin": grid["origin"].tolist(),
                            "spacing": grid["spacing"].tolist(),
                            "affine": _matrix_to_json(grid["affine"]),
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
                    })

            payload = dict(self.project_payload)
            payload["outputs"] = outputs
            payload["fusion_options"] = self.options
            payload["mosaic_coordinate_space_id"] = (
                preparation.provenance.coordinate_space_id
            )
            payload["mosaic_geometry"] = preparation.provenance.to_dict()
            registration_rejections = list(payload.get("rejections") or [])
            completed_with_warnings = bool(
                preparation.warnings
                or preparation.assumptions
                or registration_rejections
                or payload.get("completed_with_warnings")
            )
            payload["completed_with_warnings"] = completed_with_warnings
            if self._cancelled():
                raise InterruptedError("Stitching fusion was cancelled.")
            published_manifest = _atomic_write_json(
                out_dir / f"{self.base_name}_stitching_project.json",
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
                    "outputs": outputs,
                    "project_path": str(project_path),
                    "completed_with_warnings": completed_with_warnings,
                    "warnings": preparation.warnings,
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
                    if Path(bundle_temporary).exists():
                        Path(bundle_temporary).unlink()
                except Exception:
                    pass
            for scalar_temporary in scalar_temporaries:
                try:
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
