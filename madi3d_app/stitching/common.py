"""Shared stitching geometry, array sampling and resource accounting."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from madi3d_app.stitching.models import (
    StitchingCapture,
    StitchingMosaicGeometry,
    stitching_grid_mismatches,
    stitching_grid_revision,
)
from madi3d_app.stitching.correlation import check_cancelled
from madi3d_app.volume.geometry import (
    affine_support_bounds as _support_bounds,
    canonical_space_units,
    finite_tuple3,
    geometry_values_equivalent,
    grid_components_from_affine,
    invertible_affine4,
)


STITCHING_ALGORITHM_VERSION = (
    "stitching-registration-fusion-v12-reviewable-discovery"
)


_UNIT_POWER10_METERS = {
    "nm": -9,
    "micron": -6,
    "mm": -3,
    "cm": -2,
    "m": 0,
}


_UNIT_MASK_SCALAR = np.array(1, dtype=np.uint8)


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
    memory_reserved=False,
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
    memory_budget = available
    memory_limited = False
    if estimated > 0 and memory_budget > 0 and not memory_reserved:
        if estimated > memory_budget:
            raise RuntimeError(
                "One stitching worker is estimated to require approximately "
                f"{estimated / 1024**3:.2f} GiB of temporary memory, but the "
                "available memory is only "
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
                f"available memory limited concurrency to {workers} worker(s)"
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
    frozen_grid = descriptor.get("channel_local_working_grid")
    current_grid = tile.get("channel_local_working_grid")

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

    comparable_grids = (
        isinstance(frozen_grid, dict)
        and bool(frozen_grid)
        and isinstance(current_grid, dict)
        and bool(current_grid)
    )
    grid_changes = None
    if comparable_grids:
        try:
            grid_changes = stitching_grid_mismatches(frozen_grid, current_grid)
        except ValueError as exc:
            raise ValueError(
                f"{subject} has an invalid captured/current channel-local grid "
                "payload and cannot verify stitching freshness."
            ) from exc

    if frozen_revision:
        if not current_revision:
            raise ValueError(
                f"{subject} cannot verify its captured channel-local grid against "
                "the current MADI3D source. Refresh this source in the stitching "
                "project before continuing."
            )
        if grid_changes:
            changed = ", ".join(grid_changes)
            raise ValueError(
                f"{subject} captured channel-local grid is stale because its "
                f"numerical geometry changed in: {changed}. Refresh the captured "
                "project geometry and recalculate registration before continuing."
            )
        if current_revision != frozen_revision and grid_changes is None:
            raise ValueError(
                f"{subject} channel-local geometry evidence changed after project "
                "capture, and the numerical grid cannot be verified as equivalent. "
                "Refresh the captured project geometry and recalculate registration "
                "before continuing."
            )
    elif legacy_revision and current_revision:
        raise ValueError(
            f"{subject} uses a legacy stitching geometry revision that combines "
            "grid and pose and cannot be compared safely with the current channel "
            "grid. Refresh this source in the stitching project before continuing."
        )

    if frozen_revision and frozen_grid and not comparable_grids:
        raise ValueError(
            f"{subject} has no current channel-local working grid to verify "
            "against its captured stitching descriptor."
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


def _validate_source_tile_ownership(tiles, subject="Stitching project", *, cancelled=None):
    """Require one logical SourceID to map to exactly one pose-graph tile."""
    ownership = {}
    labels = {}
    for index, raw_tile in enumerate(tiles):
        check_cancelled(cancelled)
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
    tiles: list[StitchingCapture],
    *,
    mosaic_coordinate_space_id=None,
    chosen_output_unit=None,
    require_data=True,
    cancelled=None,
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
    check_cancelled(cancelled)
    _validate_source_tile_ownership(tiles, cancelled=cancelled)

    inspected = []
    recognized_units = []
    for tile_index, tile in enumerate(tiles):
        check_cancelled(cancelled)
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
        local_value = tile.get("local_affine")
        local_affine = (
            invertible_affine4(local_value, f"{subject} local working affine").copy()
            if local_value is not None
            else None
        )
        actor_value = tile.get("actor_matrix")
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
        check_cancelled(cancelled)
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
        check_cancelled(cancelled)
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


def _prepared_tiles(tiles, *, require_data=True, cancelled=None):
    tiles = list(tiles)
    identities = set()
    all_prepared = True
    for tile in tiles:
        check_cancelled(cancelled)
        identities.add(str(tile.get("mosaic_coordinate_space_id") or ""))
        all_prepared = all_prepared and bool(tile.get("_mosaic_prepared"))
    if (
        tiles
        and all_prepared
        and len(identities) == 1
        and "" not in identities
    ):
        return tiles
    return _prepare_stitching_geometry(
        tiles, require_data=require_data, cancelled=cancelled
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
    prefilter: bool | None = None,
):
    try:
        from scipy import ndimage
    except Exception as exc:
        raise RuntimeError(
            "3-D stitching requires SciPy. Install it with: pip install scipy"
        ) from exc

    # SciPy accepts strided arrays and broadcast support masks. Materializing
    # either here can copy an entire native volume for each bounded resample.
    source_zyx = np.asarray(source_zyx)
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
        prefilter=(int(order) > 1) if prefilter is None else bool(prefilter),
    )


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


def _matrix_to_json(matrix):
    return np.asarray(matrix, dtype=float).tolist()


def _registration_admission_bytes(tiles):
    """Captured backing only; resolved worker stages admit their actual demand."""
    backings = {}
    for tile in tiles:
        data = tile.get("data")
        if data is None:
            continue
        while isinstance(getattr(data, "base", None), np.ndarray):
            data = data.base
        backings[id(data)] = int(data.nbytes)
    return sum(backings.values())
