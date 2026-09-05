"""Registration algorithms, CMTK execution, QC, cancellation, and reformat workers."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6 import QtCore

from madi3d_app.operation_status import execution_succeeded
from madi3d_app.integrations.cmtk.process import CMTKProcessError
from madi3d_app.integrations.cmtk.registration import (
    CMTKLinearSettings,
    CMTKRegistrationRunner,
    CMTKWarpSettings,
    canonical_cmtk_grid,
    cmtk_output_type_for_dtype,
    cmtk_working_moving_to_reference,
    read_nrrd_zyx,
    update_artifact_bundle_qc,
    write_reference_grid_nrrd,
    write_volume_nrrd,
    write_working_nrrd,
)
from madi3d_app.integrations.cmtk.xform import write_cmtk_matrix_xform
from madi3d_app.registration.models import (
    REGISTRATION_ALGORITHM_VERSION,
    RegistrationLogEntry,
    RegistrationSettings,
    RegistrationTransformChain,
    TransformStageResult,
    _canonical_local_geometry_payload,
)
from madi3d_app.registration.output import partial_output_path
from madi3d_app.volume.geometry import (
    affine_matrix4 as _matrix4,
    affine_support_bounds as _support_bounds,
    canonical_space_units,
    finite_tuple3,
    geometry_values_equivalent,
    grid_affine_from_components,
    general_grid_affine_from_components,
    grid_components_from_affine,
    invertible_affine4,
)

# -----------------------------------------------------------------------------
# Small geometry helpers
# -----------------------------------------------------------------------------

_XYZ_ZYX_PERMUTATION = np.array(
    [[0.0, 0.0, 1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=float,
)

_UNIT_POWER10_METERS = {
    "nm": -9,
    "micron": -6,
    "mm": -3,
    "cm": -2,
    "m": 0,
}


@dataclass(frozen=True)
class RegistrationWorkingSpacePreparation:
    """Validated runtime inputs and their explicit job-local mapping record."""

    fixed: dict
    moving: dict
    provenance: dict


@dataclass(frozen=True)
class RegistrationLocalGrid:
    """Numerical voxel lattice; physical calibration is optional evidence."""

    dimensions: tuple[int, int, int]
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    direction: tuple[tuple[float, float, float], ...]
    space_units: tuple[str, str, str] | None
    coordinate_space_id: str


def _matrix_to_json(value):
    return _matrix4(value).round(12).tolist()


def _canonical_units_or_none(raw_units):
    try:
        return canonical_space_units(raw_units), "recognized"
    except (TypeError, ValueError):
        missing = raw_units is None or raw_units == "" or raw_units == [] or raw_units == ()
        return None, "missing" if missing else "unsupported"


def _working_geometry_payload(snapshot):
    geometry = copy.deepcopy(dict(snapshot.get("working_geometry") or {}))
    if geometry:
        return geometry
    local_affine = snapshot.get("local_index_affine")
    if local_affine is None:
        local_affine = general_grid_affine_from_components(
            snapshot.get("array_origin"),
            snapshot.get("source_spacing"),
            snapshot.get("source_direction"),
        )
    components = grid_components_from_affine(local_affine)
    raw_units = copy.deepcopy(snapshot.get("space_units"))
    units, _state = _canonical_units_or_none(raw_units)
    return {
        "dimensions": list(finite_tuple3(
            snapshot.get("dims"), "Working dimensions", positive=True, integer=True
        )),
        "spacing": list(components["spacing"]),
        "origin": list(components["origin"]),
        "direction": np.asarray(components["direction"], dtype=float).tolist(),
        "local_index_to_working_affine": _matrix_to_json(local_affine),
        "physical_units": list(units) if units is not None else None,
        "coordinate_space_id": str(snapshot.get("coordinate_space_id") or ""),
        "coordinate_mode": "physical" if units is not None else "numerical",
        "geometry_basis": "operation-runtime-grid",
        "assumed_fields": [],
        "replaced_fields": [],
    }


def _local_geometry_payload(snapshot):
    geometry = _working_geometry_payload(snapshot)
    grid = _validated_local_grid(
        {
            "origin": geometry.get("origin"),
            "spacing": geometry.get("spacing"),
            "direction": geometry.get("direction"),
            "dims_xyz": geometry.get("dimensions", snapshot.get("dims")),
            "space_units": geometry.get("physical_units"),
            "coordinate_space_id": geometry.get("coordinate_space_id"),
        },
        "registration source grid",
    )
    return {
        "origin": list(grid.origin),
        "spacing": list(grid.spacing),
        "direction": [list(row) for row in grid.direction],
        "dims_xyz": list(grid.dimensions),
        "space_units": list(grid.space_units) if grid.space_units is not None else None,
        "coordinate_space_id": grid.coordinate_space_id,
    }


def _validated_local_grid(geometry, label="registration grid"):
    geometry = _canonical_local_geometry_payload(geometry, label)
    dims = finite_tuple3(
        geometry.get("dims_xyz"), f"{label} dimensions", positive=True, integer=True
    )
    affine = general_grid_affine_from_components(
        geometry.get("origin"), geometry.get("spacing"), geometry.get("direction")
    )
    components = grid_components_from_affine(affine)
    units, _unit_state = _canonical_units_or_none(geometry.get("space_units"))
    return RegistrationLocalGrid(
        dimensions=dims,
        origin=components["origin"],
        spacing=components["spacing"],
        direction=tuple(
            tuple(float(value) for value in row)
            for row in np.asarray(components["direction"], dtype=float)
        ),
        space_units=units,
        coordinate_space_id=str(geometry.get("coordinate_space_id") or ""),
    )


def _local_grid_mismatches(reference, candidate):
    mismatches = []
    for field, label in (
        ("dimensions", "dimensions"),
        ("space_units", "spatial units"),
        ("coordinate_space_id", "coordinate-space identity"),
    ):
        if getattr(reference, field) != getattr(candidate, field):
            mismatches.append(label)
    for field, label in (
        ("origin", "origin"),
        ("spacing", "spacing"),
        ("direction", "direction"),
    ):
        if not geometry_values_equivalent(
            getattr(reference, field), getattr(candidate, field)
        ):
            mismatches.append(label)
    return tuple(mismatches)


def _registration_geometry_revision(working_geometry, initial_pose):
    payload = {
        "working_geometry": copy.deepcopy(working_geometry),
        "initial_pose": _matrix_to_json(initial_pose),
    }
    encoded = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _registration_identity(snapshot):
    descriptor = dict(snapshot.get("descriptor") or {})
    return {
        "dataset_id": str(snapshot.get("dataset_id") or ""),
        "acquisition_id": str(snapshot.get("acquisition_id") or descriptor.get("acquisition_id") or ""),
        "source_id": str(snapshot.get("source_id") or descriptor.get("source_id") or ""),
        "channel_id": str(snapshot.get("channel_id") or descriptor.get("channel_id") or ""),
        "backing_source_id": str(
            snapshot.get("backing_source_id")
            or descriptor.get("backing_source_id")
            or ""
        ),
        "entry_id": str(snapshot.get("entry_id") or descriptor.get("entry_id") or ""),
    }


def _inspect_registration_input(snapshot, label):
    # The snapshot contains the full volume array. Share that immutable payload
    # while constructing a separate job-local mapping record around it.
    snapshot = dict(snapshot or {})
    subject = f'{label} volume "{snapshot.get("display_name") or "unnamed"}"'
    dims = finite_tuple3(
        snapshot.get("dims"), f"{subject} dimensions", positive=True, integer=True
    )
    data = snapshot.get("data")
    if data is None:
        raise ValueError(f"{subject} has no scalar image data.")
    array = np.asanyarray(data)
    if array.ndim != 3:
        raise ValueError(
            f"{subject} image data has shape {tuple(array.shape)}; registration requires exactly three spatial dimensions."
        )
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ) or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{subject} image data type {array.dtype} is not a real scalar type.")
    if tuple(array.shape) != dims[::-1]:
        raise ValueError(
            f"{subject} dimensions {dims} (XYZ) do not match image data shape {tuple(array.shape)} (ZYX)."
        )

    working_geometry = _working_geometry_payload(snapshot)
    frozen_dims = finite_tuple3(
        working_geometry.get("dimensions", dims),
        f"{subject} frozen working dimensions",
        positive=True,
        integer=True,
    )
    if frozen_dims != dims:
        raise ValueError(
            f"{subject} frozen working dimensions {frozen_dims} do not match runtime dimensions {dims}."
        )
    local_affine = invertible_affine4(
        working_geometry.get("local_index_to_working_affine", snapshot.get("local_index_affine")),
        f"{subject} local working affine",
    ).copy()
    actor_matrix = invertible_affine4(
        snapshot.get("actor_matrix", snapshot.get("world_transform")),
        f"{subject} initial pose",
    ).copy()
    world_affine = invertible_affine4(
        snapshot.get("world_affine", snapshot.get("world_index_affine")),
        f"{subject} working affine",
    ).copy()
    if not geometry_values_equivalent(actor_matrix @ local_affine, world_affine):
        raise ValueError(
            f"{subject} has an internal runtime/model disagreement: its initial pose and local working affine do not reproduce its working affine."
        )

    raw_units = copy.deepcopy(
        working_geometry.get("physical_units")
        if "physical_units" in working_geometry
        else snapshot.get("space_units")
    )
    units, unit_state = _canonical_units_or_none(raw_units)
    coordinate_space_id = str(
        working_geometry.get("coordinate_space_id")
        or working_geometry.get("source_coordinate_space_id")
        or snapshot.get("coordinate_space_id")
        or ""
    )
    return {
        "snapshot": snapshot,
        "label": label,
        "dims": dims,
        "working_geometry": working_geometry,
        "local_affine": local_affine,
        "actor_matrix": actor_matrix,
        "world_affine": world_affine,
        "raw_units": raw_units,
        "units": units,
        "unit_state": unit_state,
        "coordinate_space_id": coordinate_space_id,
        "identity": _registration_identity(snapshot),
        "source_observation": copy.deepcopy(snapshot.get("source_observation") or {}),
        "channel_observation": copy.deepcopy(snapshot.get("channel_observation") or {}),
    }


def _prepare_registration_pair(fixed, moving):
    """Map arbitrary exact input grids into the Reference numerical working space."""

    target = _inspect_registration_input(fixed, "Reference")
    source = _inspect_registration_input(moving, "Moving")
    target_unit = target["units"][0] if target["units"] is not None else None
    warnings = []
    assumptions = []
    prepared = []
    conversions = []
    for item in (target, source):
        if target_unit is not None and item["units"] is not None:
            scale = 10.0 ** (
                _UNIT_POWER10_METERS[item["units"][0]]
                - _UNIT_POWER10_METERS[target_unit]
            )
            mode = "exact-physical-unit-conversion"
        else:
            scale = 1.0
            mode = "unconverted-numeric-working-grid"
        conversion = np.eye(4, dtype=float)
        conversion[:3, :3] *= float(scale)
        normalized_world = invertible_affine4(
            conversion @ item["world_affine"],
            f"{item['label']} normalized working affine",
        )
        normalized_actor = invertible_affine4(
            conversion @ item["actor_matrix"],
            f"{item['label']} normalized initial pose",
        )
        runtime = dict(item["snapshot"])
        runtime.update({
            "dims": item["dims"],
            "data": np.ascontiguousarray(np.asanyarray(item["snapshot"]["data"])),
            "world_affine": normalized_world,
            "world_index_affine": normalized_world.copy(),
            "actor_matrix": normalized_actor,
            "world_transform": normalized_actor.copy(),
            "operation_space_mapping": conversion.copy(),
            "source_world_affine": item["world_affine"].copy(),
            "source_actor_matrix": item["actor_matrix"].copy(),
        })
        conversion_record = {
            "role": item["label"].lower(),
            "source_units_raw": copy.deepcopy(item["raw_units"]),
            "source_units_canonical": list(item["units"]) if item["units"] is not None else None,
            "target_working_unit": target_unit,
            "scale_to_target_working_space": float(scale),
            "conversion_matrix": _matrix_to_json(conversion),
            "mode": mode,
            "uncertainty": item["unit_state"] if item["units"] is None else None,
        }
        if item["units"] is None:
            warning = (
                f"{item['label']} spatial units are {item['unit_state']}; its current numerical working grid is used without conversion."
            )
            warnings.append(warning)
            assumptions.append(
                f"{item['label']} numerical coordinates map one-for-one into the Reference working space because exact unit conversion is unavailable."
            )
        prepared.append(runtime)
        conversions.append(conversion_record)

    if source["coordinate_space_id"] != target["coordinate_space_id"]:
        warnings.append(
            "Moving and Reference inputs have distinct source coordinate-space identities; their captured poses define the initial moving-to-reference mapping."
        )
    for item in (target, source):
        diagnostics = dict(item["snapshot"].get("snapshot_diagnostics") or {})
        for record in diagnostics.get("warnings") or ():
            warnings.append(str(record.get("message") if isinstance(record, dict) else record))
        for record in diagnostics.get("assumptions") or ():
            assumptions.append(str(record.get("message") if isinstance(record, dict) else record))
        for field in item["working_geometry"].get("assumed_fields") or ():
            assumptions.append(f"{item['label']} working field is assumed: {field}.")
        for field in item["working_geometry"].get("replaced_fields") or ():
            assumptions.append(f"{item['label']} working field replaces invalid source evidence: {field}.")

    provenance = {
        "target_working_unit": target_unit,
        "source_identity": source["identity"],
        "target_identity": target["identity"],
        "source_space_id": source["coordinate_space_id"],
        "target_space_id": target["coordinate_space_id"],
        "source_observation": source["source_observation"],
        "target_observation": target["source_observation"],
        "source_channel_observation": source["channel_observation"],
        "target_channel_observation": target["channel_observation"],
        "source_working_grid": copy.deepcopy(source["working_geometry"]),
        "target_working_grid": copy.deepcopy(target["working_geometry"]),
        "source_grid_state": str(
            source["snapshot"].get("grid_state")
            or source["working_geometry"].get("physical_grid_state")
            or ""
        ).strip(),
        "target_grid_state": str(
            target["snapshot"].get("grid_state")
            or target["working_geometry"].get("physical_grid_state")
            or ""
        ).strip(),
        "source_grid_diagnostics": copy.deepcopy(
            source["snapshot"].get("grid_diagnostics")
            or source["working_geometry"].get("physical_grid_diagnostics")
            or []
        ),
        "target_grid_diagnostics": copy.deepcopy(
            target["snapshot"].get("grid_diagnostics")
            or target["working_geometry"].get("physical_grid_diagnostics")
            or []
        ),
        "unit_conversions": conversions,
        "source_to_operation": copy.deepcopy(conversions[1]["conversion_matrix"]),
        "target_to_operation": copy.deepcopy(conversions[0]["conversion_matrix"]),
        "source_geometry_revision": _registration_geometry_revision(
            source["working_geometry"], source["actor_matrix"]
        ),
        "target_geometry_revision": _registration_geometry_revision(
            target["working_geometry"], target["actor_matrix"]
        ),
        "source_initial_pose": _matrix_to_json(source["actor_matrix"]),
        "target_initial_pose": _matrix_to_json(target["actor_matrix"]),
        "warnings": list(dict.fromkeys(value for value in warnings if value)),
        "assumptions": list(dict.fromkeys(value for value in assumptions if value)),
    }
    return RegistrationWorkingSpacePreparation(
        fixed=prepared[0], moving=prepared[1], provenance=provenance
    )


def _map_stages_to_source_target_space(stages, working_space):
    """Return stages whose public matrices map source coordinates to target coordinates."""

    source_to_operation = invertible_affine4(
        working_space.get("source_to_operation"), "Source-to-registration working mapping"
    )
    target_to_operation = invertible_affine4(
        working_space.get("target_to_operation"), "Target-to-registration working mapping"
    )
    operation_to_target = np.linalg.inv(target_to_operation)
    mapped = []
    previous = np.eye(4, dtype=float)
    for stage in stages:
        public_cumulative = operation_to_target @ _matrix4(
            stage.cumulative_moving_to_fixed
        ) @ source_to_operation
        public_incremental = public_cumulative @ np.linalg.inv(previous)
        details = copy.deepcopy(stage.details or {})
        details["operation_space_cumulative_moving_to_reference"] = copy.deepcopy(
            stage.cumulative_moving_to_fixed
        )
        details["operation_space_incremental_moving_to_reference"] = copy.deepcopy(
            stage.incremental_moving_to_fixed
        )
        details["source_to_operation"] = _matrix_to_json(source_to_operation)
        details["target_to_operation"] = _matrix_to_json(target_to_operation)
        mapped.append(TransformStageResult(
            name=stage.name,
            kind=stage.kind,
            cumulative_moving_to_fixed=_matrix_to_json(public_cumulative),
            incremental_moving_to_fixed=_matrix_to_json(public_incremental),
            metric_value=stage.metric_value,
            ncc=stage.ncc,
            iterations=stage.iterations,
            stop_condition=stage.stop_condition,
            execution_status=stage.execution_status,
            qc_status=stage.qc_status,
            user_decision=stage.user_decision,
            details=details,
            logs=copy.deepcopy(stage.logs),
        ))
        previous = public_cumulative
    return mapped


def _prepare_reformat_target(record, original_fixed, working_space):
    """Capture the default Reference or one explicit output lattice in target units."""

    explicit = bool(record)
    if explicit:
        record = copy.deepcopy(dict(record or {}))
        geometry = copy.deepcopy(dict(record.get("local_geometry") or {}))
        actor = invertible_affine4(
            record.get("actor_matrix"), "Explicit Reformat output-grid pose"
        )
        descriptor = copy.deepcopy(dict(record.get("descriptor") or {}))
        label = str(record.get("display_name") or "explicit output grid")
    else:
        geometry = _local_geometry_payload(original_fixed)
        actor = invertible_affine4(
            working_space.get("target_initial_pose"), "Reference output-grid pose"
        )
        descriptor = copy.deepcopy(dict(original_fixed.get("descriptor") or {}))
        label = str(original_fixed.get("display_name") or "Reference")
        record = {}
    captured = record if explicit else original_fixed
    selected_working_grid = copy.deepcopy(
        dict(
            captured.get("working_grid")
            or captured.get("working_geometry")
            or {}
        )
    )
    selected_physical_grid = copy.deepcopy(
        captured.get("physical_grid")
        or captured.get("canonical_physical_grid")
    )
    assumed_fields = list(selected_working_grid.get("assumed_fields") or ())
    replaced_fields = list(selected_working_grid.get("replaced_fields") or ())
    geometry_basis = str(
        selected_working_grid.get("geometry_basis") or ""
    ).strip()
    coordinate_mode = str(
        selected_working_grid.get("coordinate_mode") or ""
    ).strip()
    physical_grid_state = str(
        captured.get("physical_grid_state")
        or captured.get("grid_state")
        or selected_working_grid.get("physical_grid_state")
        or ""
    ).strip().lower()
    if not physical_grid_state:
        if replaced_fields or geometry_basis == "sanitized-invalid-source":
            physical_grid_state = "inconsistent"
        elif (
            assumed_fields
            or coordinate_mode in {"working-grid", "numerical"}
            or geometry_basis
            in {
                "partially-assumed",
                "fully-voxel-default",
                "sanitized-invalid-source",
            }
        ):
            physical_grid_state = "unresolved"
        elif selected_working_grid.get("physical_units") is not None:
            physical_grid_state = "resolved"
        else:
            physical_grid_state = "unresolved"
    if physical_grid_state not in {"resolved", "unresolved", "inconsistent"}:
        raise ValueError(
            f"{label} Reformat output grid has unsupported physical state {physical_grid_state!r}."
        )
    physical_grid_diagnostics = copy.deepcopy(
        captured.get("physical_grid_diagnostics")
        or captured.get("grid_diagnostics")
        or selected_working_grid.get("physical_grid_diagnostics")
        or []
    )
    grid = _validated_local_grid(geometry, f"{label} Reformat output grid")
    source_units = grid.space_units
    target_unit = working_space.get("target_working_unit")
    if source_units is not None and target_unit:
        scale = 10.0 ** (
            _UNIT_POWER10_METERS[source_units[0]] - _UNIT_POWER10_METERS[target_unit]
        )
        mode = "exact-physical-unit-conversion"
    else:
        scale = 1.0
        mode = "unconverted-numeric-working-grid"
    conversion = np.eye(4, dtype=float)
    conversion[:3, :3] *= float(scale)
    local = general_grid_affine_from_components(
        grid.origin, grid.spacing, grid.direction
    )
    normalized_local = invertible_affine4(
        conversion @ local, "Normalized Reformat output-grid affine"
    )
    normalized_components = grid_components_from_affine(normalized_local)
    normalized_actor = invertible_affine4(
        conversion @ actor @ np.linalg.inv(conversion),
        "Normalized Reformat output-grid pose",
    )
    output_unit = (
        str(target_unit)
        if source_units is not None and target_unit
        else str(source_units[0])
        if source_units is not None
        else None
    )
    verified_output_units = [output_unit] * 3 if output_unit else None
    if physical_grid_state == "resolved" and verified_output_units is None:
        raise ValueError(
            f"{label} Reformat output grid is resolved but has no valid spatial units."
        )
    output_coordinate_space_id = str(
        working_space.get("target_space_id") or grid.coordinate_space_id or ""
    )
    normalized_geometry = {
        "dims_xyz": list(grid.dimensions),
        "origin": list(normalized_components["origin"]),
        "spacing": list(normalized_components["spacing"]),
        "direction": np.asarray(normalized_components["direction"], dtype=float).tolist(),
        "space_units": verified_output_units,
        "coordinate_space_id": output_coordinate_space_id,
    }
    if not coordinate_mode:
        coordinate_mode = (
            "physical-grid"
            if physical_grid_state == "resolved"
            else "working-grid"
        )
    if not geometry_basis:
        geometry_basis = (
            "verified-source-physical"
            if physical_grid_state == "resolved"
            else "partially-assumed"
        )
    normalized_working_grid = {
        **selected_working_grid,
        "dimensions": list(grid.dimensions),
        "spacing": list(normalized_components["spacing"]),
        "origin": list(normalized_components["origin"]),
        "direction": np.asarray(
            normalized_components["direction"], dtype=float
        ).tolist(),
        "local_index_to_working_affine": _matrix_to_json(normalized_local),
        "physical_units": verified_output_units,
        "source_coordinate_space_id": output_coordinate_space_id,
        "coordinate_mode": coordinate_mode,
        "geometry_basis": geometry_basis,
        "physical_grid_state": physical_grid_state,
        "physical_grid_diagnostics": copy.deepcopy(
            physical_grid_diagnostics
        ),
        "assumed_fields": assumed_fields,
        "replaced_fields": replaced_fields,
        "warnings": list(selected_working_grid.get("warnings") or ()),
    }
    normalized_physical_grid = None
    if physical_grid_state == "resolved":
        physical_source = dict(selected_physical_grid or {})
        normalized_physical_grid = {
            "dimensions": list(grid.dimensions),
            "spacing": list(normalized_components["spacing"]),
            "spatial_units": verified_output_units,
            "origin": list(normalized_components["origin"]),
            "direction": np.asarray(
                normalized_components["direction"], dtype=float
            ).tolist(),
            "time_point_count": int(
                physical_source.get("time_point_count")
                or selected_working_grid.get("time_point_count")
                or 1
            ),
            "time_interval": float(
                physical_source.get("time_interval")
                or selected_working_grid.get("time_interval")
                or 1.0
            ),
            "time_units": str(
                physical_source.get("time_units")
                or selected_working_grid.get("time_units")
                or "frame"
            ),
            "coordinate_space_id": output_coordinate_space_id,
        }
    result = {
        "selection_mode": "explicit" if explicit else "reference-default",
        "display_name": label,
        "entry_id": str(record.get("entry_id") or descriptor.get("entry_id") or ""),
        "descriptor": descriptor,
        "selected_output_grid_identity": _registration_identity(captured),
        "selected_working_grid": selected_working_grid,
        "selected_physical_grid": selected_physical_grid,
        "physical_grid_state": physical_grid_state,
        "physical_grid_diagnostics": physical_grid_diagnostics,
        "source_observation": copy.deepcopy(
            captured.get("source_observation") or {}
        ),
        "channel_observation": copy.deepcopy(
            captured.get("channel_observation") or {}
        ),
        "geometry_revision": str(
            captured.get("geometry_revision") or ""
        ),
        "source_geometry": geometry,
        "source_actor_matrix": _matrix_to_json(actor),
        "unit_conversion": {
            "source_units_canonical": list(source_units) if source_units is not None else None,
            "target_working_unit": target_unit,
            "scale_to_target_working_space": float(scale),
            "conversion_matrix": _matrix_to_json(conversion),
            "mode": mode,
        },
        "normalized_geometry": normalized_geometry,
        "normalized_working_grid": normalized_working_grid,
        "normalized_physical_grid": normalized_physical_grid,
        "normalized_actor_matrix": _matrix_to_json(normalized_actor),
    }

    def append_unique(key, value):
        values = working_space.setdefault(key, [])
        if value and value not in values:
            values.append(value)

    if source_units is None:
        message = (
            f"{label} output-grid units are unknown; its numerical lattice is used one-for-one in the Reference working space."
        )
        append_unique("warnings", message)
        append_unique("assumptions", message)
    if physical_grid_state != "resolved":
        append_unique(
            "warnings",
            f"{label} output-grid physical calibration is {physical_grid_state}; Reformat preserves its exact numerical lattice without claiming verified calibration.",
        )
    for field in assumed_fields:
        append_unique(
            "assumptions", f"{label} output-grid field is assumed: {field}."
        )
    return result


def _output_to_source_zyx_mapping(source_world_affine, output_world_affine):
    try:
        out_xyz_to_in_xyz = np.linalg.solve(
            _matrix4(source_world_affine), _matrix4(output_world_affine)
        )
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Registration input transform is singular.") from exc
    return _XYZ_ZYX_PERMUTATION @ out_xyz_to_in_xyz @ _XYZ_ZYX_PERMUTATION


def _resample_zyx(source_zyx, source_world_affine, output_world_affine,
                  output_shape_zyx, *, order=1, output_dtype=np.float32):
    try:
        from scipy import ndimage
    except Exception as exc:
        raise RuntimeError("Registration requires SciPy. Install it with: pip install scipy") from exc
    source = np.ascontiguousarray(np.asarray(source_zyx))
    mapping = _output_to_source_zyx_mapping(source_world_affine, output_world_affine)
    return ndimage.affine_transform(
        source,
        matrix=mapping[:3, :3],
        offset=mapping[:3, 3],
        output_shape=tuple(int(v) for v in output_shape_zyx),
        output=output_dtype,
        order=int(order),
        mode="constant",
        cval=0.0,
        prefilter=(int(order) > 1),
    )


def _robust_normalize(array):
    x = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(x)
    vals = x[finite]
    if vals.size < 32:
        return np.zeros_like(x, dtype=np.float32)
    positive = vals[vals > 0]
    if positive.size >= 256:
        vals = positive
    lo, hi = np.percentile(vals, (0.5, 99.7))
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x - float(lo)) / float(hi - lo), 0.0, 1.0)
    y[~finite] = 0.0
    return y.astype(np.float32, copy=False)


def _ncc(a, b, mask=None):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if np.count_nonzero(valid) < 128:
        return float("nan")
    av = a[valid].astype(np.float64, copy=False)
    bv = b[valid].astype(np.float64, copy=False)
    av -= av.mean(); bv -= bv.mean()
    denom = math.sqrt(float(np.dot(av, av)) * float(np.dot(bv, bv)))
    if denom <= 1e-15:
        return float("nan")
    return float(np.dot(av, bv) / denom)


def _weighted_world_moments(snapshot, max_samples=300000):
    """Return weighted COM and principal axes in MADI world coordinates."""
    data = _robust_normalize(snapshot["data"])
    flat = data.ravel()
    positive = np.flatnonzero(flat > max(0.02, float(np.percentile(flat, 70.0))))
    if positive.size < 64:
        positive = np.flatnonzero(flat > 0)
    if positive.size < 16:
        lo, hi = _support_bounds(snapshot["world_affine"], snapshot["dims"])
        center = 0.5 * (lo + hi)
        return center, np.eye(3, dtype=float)
    if positive.size > max_samples:
        # Deterministic sub-sampling keeps PCA reproducible across queue runs.
        idx = np.linspace(0, positive.size - 1, int(max_samples), dtype=np.int64)
        positive = positive[idx]
    z, y, x = np.unravel_index(positive, data.shape)
    xyz = np.column_stack((x, y, z)).astype(np.float64)
    weights = flat[positive].astype(np.float64)
    weights = np.maximum(weights, 1e-6)
    world = (
        _matrix4(snapshot["world_affine"])
        @ np.column_stack((xyz, np.ones(len(xyz), dtype=float))).T
    ).T[:, :3]
    wsum = float(weights.sum())
    center = (world * weights[:, None]).sum(axis=0) / max(wsum, 1e-12)
    centered = world - center
    cov = (centered * weights[:, None]).T @ centered / max(wsum, 1e-12)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axes = vecs[:, order]
    # Right handed principal frame.
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    return center, axes


def _initial_moving_to_fixed(fixed, moving, mode):
    mode = str(mode or "current").lower()
    identity = np.eye(4, dtype=float)
    if mode == "current":
        return identity, "Current MADI3D pose"

    f_lo, f_hi = _snapshot_support_bounds(fixed)
    m_lo, m_hi = _snapshot_support_bounds(moving)
    f_center = 0.5 * (f_lo + f_hi)
    m_center = 0.5 * (m_lo + m_hi)

    if mode == "centers":
        out = identity.copy()
        out[:3, 3] = f_center - m_center
        return out, "Volume centers"

    f_com, f_axes = _weighted_world_moments(fixed)
    m_com, m_axes = _weighted_world_moments(moving)
    if mode == "com":
        out = identity.copy(); out[:3, 3] = f_com - m_com
        return out, "Intensity center of mass"

    if mode == "pca":
        # Principal axes have sign ambiguity. Choose the proper sign assignment
        # producing the smallest rotation from the current MADI orientation.
        best_r = np.eye(3, dtype=float)
        best_trace = -1e99
        for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            signed = f_axes @ np.diag(signs)
            r = signed @ m_axes.T
            if np.linalg.det(r) <= 0:
                continue
            score = float(np.trace(r))
            if score > best_trace:
                best_trace = score
                best_r = r
        out = identity.copy()
        out[:3, :3] = best_r
        out[:3, 3] = f_com - best_r @ m_com
        return out, "Principal axes + center of mass"

    return identity, "Current MADI3D pose"



_LANDMARK_INITIAL_ALIGNMENT_MODES = frozenset(("landmarks", "landmarks_9dof"))


def _is_landmark_initial_alignment(value):
    return str(value or "current") in _LANDMARK_INITIAL_ALIGNMENT_MODES


def _validate_landmark_9dof_pipeline(settings):
    settings = dict(settings or {})
    if str(settings.get("initial_alignment") or "current") != "landmarks_9dof":
        return
    if _linear_engine(settings) != "cmtk":
        raise RuntimeError(
            "Landmark 9-DOF initialization requires the CMTK linear engine. "
            "ITK's anisotropic-scale transform uses a different matrix parameterization and cannot preserve this no-shear CMTK initializer exactly."
        )
    model = str(settings.get("global_model") or "affine").lower()
    if model not in {"scaleversor", "affine"}:
        raise RuntimeError(
            "Landmark 9-DOF initialization requires a 9-DOF or 12-DOF global transform so its independent XYZ scales are retained during intensity refinement."
        )


def _landmark_correspondences(reference_points, moving_points, weights, minimum_pairs):
    try:
        fixed = np.asarray(reference_points, dtype=np.float64).reshape(-1, 3)
        moving = np.asarray(moving_points, dtype=np.float64).reshape(-1, 3)
    except Exception as exc:
        raise RuntimeError("Landmark coordinates must be complete 3-D points.") from exc
    if fixed.shape != moving.shape or fixed.shape[0] < int(minimum_pairs):
        raise RuntimeError(
            f"Landmark initialization requires at least {int(minimum_pairs)} complete landmark pairs."
        )
    if not np.all(np.isfinite(fixed)) or not np.all(np.isfinite(moving)):
        raise RuntimeError("Landmark coordinates contain non-finite values.")

    if weights is None:
        normalized_weights = np.ones(fixed.shape[0], dtype=np.float64)
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if normalized_weights.size != fixed.shape[0]:
            raise RuntimeError("Landmark weight count does not match the landmark-pair count.")
        if not np.all(np.isfinite(normalized_weights)) or np.any(normalized_weights <= 0.0):
            raise RuntimeError("Landmark weights must be finite and greater than zero.")
    normalized_weights /= max(float(normalized_weights.sum()), 1e-12)
    return fixed, moving, normalized_weights


def _landmark_rigid_moving_to_fixed(reference_points, moving_points, weights=None):
    """Least-squares rigid map from moving-world landmarks to reference-world landmarks.

    Landmarks are already expressed in MADI3D world/physical coordinates.  A
    weighted Kabsch solve gives the one proper rotation + translation that best
    maps the moving points onto their paired reference points.  Scaling and
    reflection are intentionally excluded from Phase 1 landmark initialization.
    """
    fixed, moving, w = _landmark_correspondences(
        reference_points, moving_points, weights, minimum_pairs=3
    )

    fixed_center = np.sum(fixed * w[:, None], axis=0)
    moving_center = np.sum(moving * w[:, None], axis=0)
    fixed_centered = fixed - fixed_center
    moving_centered = moving - moving_center

    # Three points are sufficient only when they actually establish a 3-D
    # orientation.  Collinear points leave rotation around their common line
    # unconstrained and should be rejected before an optimizer is started.
    scale = max(
        1e-9,
        float(np.max(np.linalg.norm(fixed_centered, axis=1))),
        float(np.max(np.linalg.norm(moving_centered, axis=1))),
    )
    if np.linalg.matrix_rank(fixed_centered, tol=scale * 1e-6) < 2:
        raise RuntimeError("Reference landmarks are nearly collinear. Place landmarks across the specimen, not along one line.")
    if np.linalg.matrix_rank(moving_centered, tol=scale * 1e-6) < 2:
        raise RuntimeError("Registration landmarks are nearly collinear. Place landmarks across the specimen, not along one line.")

    covariance = (moving_centered * w[:, None]).T @ fixed_centered
    u, singular, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    translation = fixed_center - rotation @ moving_center

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    mapped = moving @ rotation.T + translation
    residuals = np.linalg.norm(mapped - fixed, axis=1)
    return matrix, residuals, singular


def _landmark_9dof_moving_to_fixed(reference_points, moving_points, weights=None):
    """Fit translation, proper rotation and independent positive XYZ scales.

    The 3x3 block is constrained to ``rotation @ diag(scale_xyz)``. This is the
    no-shear 9-DOF model used as a CMTK-compatible initializer, not a 12-DOF
    affine fit. Four genuinely non-coplanar pairs are the mathematical minimum.
    """
    fixed, moving, w = _landmark_correspondences(
        reference_points, moving_points, weights, minimum_pairs=4
    )
    fixed_center = np.sum(fixed * w[:, None], axis=0)
    moving_center = np.sum(moving * w[:, None], axis=0)
    fixed_centered = fixed - fixed_center
    moving_centered = moving - moving_center
    sqrt_w = np.sqrt(w)[:, None]
    fixed_spread = np.linalg.svd(fixed_centered * sqrt_w, compute_uv=False)
    moving_spread = np.linalg.svd(moving_centered * sqrt_w, compute_uv=False)

    for label, singular in (
        ("Reference", fixed_spread),
        ("Registration", moving_spread),
    ):
        if (
            singular.size < 3
            or not np.all(np.isfinite(singular))
            or float(singular[2]) <= max(1e-9, float(singular[0]) * 1e-4)
        ):
            raise RuntimeError(
                f"{label} landmarks are coplanar or nearly coplanar. "
                "A 9-DOF fit needs points spread through X, Y and Z to constrain all three scales."
            )

    try:
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        raise RuntimeError(
            "Landmark 9-DOF initialization requires SciPy. Install the normal MADI3D dependencies and retry."
        ) from exc

    rigid, _rigid_residuals, _rigid_singular = _landmark_rigid_moving_to_fixed(
        fixed, moving, w
    )
    initial_rotation = np.asarray(rigid[:3, :3], dtype=np.float64)
    fixed_in_initial_axes = fixed_centered @ initial_rotation
    denominators = np.sum(w[:, None] * moving_centered * moving_centered, axis=0)
    scale_guess = np.sum(
        w[:, None] * moving_centered * fixed_in_initial_axes, axis=0
    ) / np.maximum(denominators, 1e-12)
    scale_guess = np.clip(scale_guess, 0.05, 20.0)
    initial = np.concatenate((
        Rotation.from_matrix(initial_rotation).as_rotvec(),
        np.log(scale_guess),
    ))

    def weighted_residual(parameters):
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        scales = np.exp(parameters[3:])
        linear = rotation @ np.diag(scales)
        difference = moving_centered @ linear.T - fixed_centered
        return (difference * sqrt_w).reshape(-1)

    result = least_squares(
        weighted_residual,
        initial,
        bounds=(
            np.array([-np.inf, -np.inf, -np.inf, *([math.log(0.05)] * 3)]),
            np.array([np.inf, np.inf, np.inf, *([math.log(20.0)] * 3)]),
        ),
        method="trf",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=4000,
    )
    if not bool(result.success) or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            "Landmark 9-DOF fit did not converge to a finite transform: "
            + str(result.message or "unknown optimizer failure")
        )
    if np.any(np.asarray(result.active_mask[3:], dtype=int) != 0):
        raise RuntimeError(
            "Landmark 9-DOF fit requires an XYZ scale outside the safe 0.05–20 range. "
            "Check landmark pairing, units, and coordinate spaces."
        )

    rotation = Rotation.from_rotvec(result.x[:3]).as_matrix()
    scales = np.exp(result.x[3:])
    linear = rotation @ np.diag(scales)
    translation = fixed_center - linear @ moving_center
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation
    valid, reason, geometry_qc = _linear_transform_sanity(matrix, kind="scaleversor")
    if not valid:
        raise RuntimeError(f"Landmark 9-DOF fit is not physically usable: {reason}.")

    mapped = moving @ linear.T + translation
    residuals = np.linalg.norm(mapped - fixed, axis=1)
    return matrix, residuals, {
        "scale_xyz": [float(v) for v in scales],
        "fixed_spread_singular_values": [float(v) for v in fixed_spread],
        "moving_spread_singular_values": [float(v) for v in moving_spread],
        "optimizer_evaluations": int(result.nfev),
        "optimizer_cost": float(result.cost),
        "optimizer_optimality": float(result.optimality),
        "linear_geometry_qc": geometry_qc,
    }


def _effective_world_spacing(snapshot):
    """Return effective source sampling along MADI world X/Y/Z axes."""
    linear = _matrix4(snapshot["world_affine"])[:3, :3]
    try:
        inverse_linear = np.linalg.inv(linear)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Registration input transform is singular.") from exc
    spacing = np.array(
        [
            1.0 / float(np.linalg.norm(inverse_linear[:, axis]))
            for axis in range(3)
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise ValueError("Registration input transform has invalid working spacing.")
    return spacing


def _index_box_world_bounds(affine, lo_xyz, hi_xyz):
    """World AABB for an inclusive XYZ voxel-index box, including voxel support."""
    lo = np.asarray(lo_xyz, dtype=float).reshape(3)
    hi = np.asarray(hi_xyz, dtype=float).reshape(3)
    corners = np.array(
        [[x, y, z, 1.0]
         for x in (lo[0] - 0.5, hi[0] + 0.5)
         for y in (lo[1] - 0.5, hi[1] + 0.5)
         for z in (lo[2] - 0.5, hi[2] + 0.5)],
        dtype=float,
    )
    world = (_matrix4(affine) @ corners.T).T[:, :3]
    return world.min(axis=0), world.max(axis=0)


def _foreground_detector_normalize(array):
    """Normalize detector samples without erasing dim signal beside zero margins."""
    x = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(x)
    values = x[finite]
    if values.size < 32:
        return np.zeros_like(x, dtype=np.float32)
    positive = values[values > 0.0]
    if positive.size < 16:
        return np.zeros_like(x, dtype=np.float32)
    # Exact zero padding must not change detector contrast. When zeros exist,
    # keep zero as the baseline and estimate only the upper robust signal scale
    # from positive data. For genuinely non-zero detector backgrounds, use the
    # low positive percentile as the baseline.
    lo = 0.0 if np.any(values <= 0.0) else float(np.percentile(positive, 10.0))
    hi = float(np.percentile(positive, 99.5))
    if not math.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    out = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32, copy=False)


def _foreground_threshold(normalized):
    """Deterministic Otsu-style threshold on a robust detector image."""
    x = np.asarray(normalized, dtype=np.float32)
    values = x[np.isfinite(x) & (x > 0.0)]
    if values.size < 32:
        return None
    hist, edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    total = float(hist.sum())
    if total <= 0.0:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight1 = np.cumsum(hist, dtype=np.float64)
    weight2 = total - weight1
    mean1_num = np.cumsum(hist * centers, dtype=np.float64)
    mean_total = float(mean1_num[-1])
    valid = (weight1 > 0.0) & (weight2 > 0.0)
    between = np.full(hist.shape, -np.inf, dtype=np.float64)
    if np.any(valid):
        mean1 = np.zeros_like(weight1)
        mean2 = np.zeros_like(weight2)
        mean1[valid] = mean1_num[valid] / weight1[valid]
        mean2[valid] = (mean_total - mean1_num[valid]) / weight2[valid]
        between[valid] = weight1[valid] * weight2[valid] * (mean1[valid] - mean2[valid]) ** 2
        otsu = float(centers[int(np.argmax(between))])
    else:
        otsu = float(np.percentile(values, 50.0))

    # Keep the detector conservative. A very low Otsu split in non-zero detector
    # noise should not turn nearly the entire confocal frame into foreground, while
    # the upper clamp guarantees that a dim but spatially coherent brain remains.
    upper = float(np.percentile(values, 85.0))
    threshold = min(max(0.02, otsu), max(0.02, upper))
    return float(threshold)


def _foreground_support_detector(array_zyx, max_dim=160):
    """Find robust connected foreground support on a bounded-resolution detector grid."""
    source = np.asarray(array_zyx)
    if source.ndim != 3 or min(source.shape) < 1:
        return {"valid": False, "reason": "not a scalar 3-D image"}
    max_dim = max(32, int(max_dim))
    shape = np.asarray(source.shape, dtype=int)
    stride = np.maximum(1, np.ceil(shape / float(max_dim)).astype(int))
    sample = source[::int(stride[0]), ::int(stride[1]), ::int(stride[2])]
    normalized = _foreground_detector_normalize(sample)
    threshold = _foreground_threshold(normalized)
    if threshold is None:
        return {
            "valid": False,
            "reason": "insufficient non-zero signal",
            "detector_stride_zyx": [int(v) for v in stride],
            "detector_shape_zyx": [int(v) for v in sample.shape],
        }

    candidate = np.isfinite(normalized) & (normalized >= float(threshold))
    if int(np.count_nonzero(candidate)) < 16:
        return {
            "valid": False,
            "reason": "foreground threshold produced too few voxels",
            "threshold": float(threshold),
            "detector_stride_zyx": [int(v) for v in stride],
            "detector_shape_zyx": [int(v) for v in sample.shape],
        }

    try:
        from scipy import ndimage
    except Exception as exc:
        raise RuntimeError(
            "Registration foreground detection requires SciPy. Install it with: pip install scipy"
        ) from exc

    structure = np.ones((3, 3, 3), dtype=bool)
    candidate = ndimage.binary_closing(candidate, structure=structure, iterations=1)
    labels, count = ndimage.label(candidate, structure=structure)
    if int(count) < 1:
        return {
            "valid": False,
            "reason": "no connected foreground component",
            "threshold": float(threshold),
            "detector_stride_zyx": [int(v) for v in stride],
            "detector_shape_zyx": [int(v) for v in sample.shape],
        }

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(sizes.max()) if sizes.size else 0
    minimum_component = max(8, int(math.ceil(0.01 * max(1, largest))))
    keep = np.flatnonzero(sizes >= minimum_component)
    if keep.size == 0 and largest > 0:
        keep = np.array([int(np.argmax(sizes))], dtype=int)
    support = np.isin(labels, keep)
    support = ndimage.binary_fill_holes(support)
    # One detector voxel of dilation protects thin neuropil edges and weakly
    # separated lobes before the physical bounding-box padding below.
    support = ndimage.binary_dilation(support, structure=structure, iterations=1)
    coordinates = np.argwhere(support)
    if coordinates.size == 0:
        return {
            "valid": False,
            "reason": "connected foreground support became empty",
            "threshold": float(threshold),
            "detector_stride_zyx": [int(v) for v in stride],
            "detector_shape_zyx": [int(v) for v in sample.shape],
        }

    lo_sample = coordinates.min(axis=0)
    hi_sample = coordinates.max(axis=0)
    lo_native = lo_sample * stride
    hi_native = np.minimum(shape - 1, (hi_sample + 1) * stride - 1)
    retained = int(np.count_nonzero(support))
    return {
        "valid": True,
        "threshold": float(threshold),
        "bbox_zyx": [
            [int(v) for v in lo_native],
            [int(v) for v in hi_native],
        ],
        "detector_stride_zyx": [int(v) for v in stride],
        "detector_shape_zyx": [int(v) for v in sample.shape],
        "component_count": int(count),
        "retained_component_count": int(keep.size),
        "largest_component_voxels": int(largest),
        "retained_detector_fraction": float(retained / max(1, support.size)),
        "_support_mask_detector": support,
    }


def _foreground_support_bounds(snapshot, target_spacing=1.5, detector_max_dim=160):
    """Return a conservative world-space brain/support box for registration."""
    full_lo, full_hi = _support_bounds(snapshot["world_affine"], snapshot["dims"])
    detector = _foreground_support_detector(snapshot["data"], max_dim=detector_max_dim)
    details = {k: v for k, v in detector.items() if not str(k).startswith("_")}
    if not detector.get("valid"):
        details.update({
            "fallback_to_full_support": True,
            "padding_world": 0.0,
            "support_bounds_world": [full_lo.tolist(), full_hi.tolist()],
        })
        return (full_lo, full_hi), details

    lo_zyx = np.asarray(detector["bbox_zyx"][0], dtype=float)
    hi_zyx = np.asarray(detector["bbox_zyx"][1], dtype=float)
    lo_xyz = lo_zyx[::-1]
    hi_xyz = hi_zyx[::-1]
    foreground_lo, foreground_hi = _index_box_world_bounds(
        snapshot["world_affine"], lo_xyz, hi_xyz
    )
    foreground_extent = np.maximum(foreground_hi - foreground_lo, 0.0)
    target = float(np.max(np.atleast_1d(np.asarray(target_spacing, dtype=float))))
    padding_world = max(3.0 * max(target, 1e-6), 0.04 * float(np.max(foreground_extent)))
    lo = np.maximum(full_lo, foreground_lo - padding_world)
    hi = np.minimum(full_hi, foreground_hi + padding_world)
    if np.any(hi <= lo):
        details.update({
            "fallback_to_full_support": True,
            "reason": "foreground world bounds were degenerate",
            "padding_world": 0.0,
            "support_bounds_world": [full_lo.tolist(), full_hi.tolist()],
        })
        return (full_lo, full_hi), details

    details.update({
        "fallback_to_full_support": False,
        "padding_world": float(padding_world),
        "foreground_bounds_world": [foreground_lo.tolist(), foreground_hi.tolist()],
        "support_bounds_world": [lo.tolist(), hi.tolist()],
        "full_bounds_world": [full_lo.tolist(), full_hi.tolist()],
    })
    return (lo, hi), details


def _snapshot_support_bounds(snapshot):
    value = snapshot.get("registration_support_bounds_world")
    if value is not None:
        try:
            lo = np.asarray(value[0], dtype=float).reshape(3)
            hi = np.asarray(value[1], dtype=float).reshape(3)
            if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(hi > lo):
                return lo, hi
        except Exception:
            pass
    return _support_bounds(snapshot["world_affine"], snapshot["dims"])


def _working_grid(snapshot, target_spacing=1.5, max_dim=512, support_bounds=None):
    """Build a foreground-bounded, world-aligned registration working image.

    ``target_spacing`` is the scientific resolution control in physical scene
    units (normally micrometres for microscopy data). Source sampling is never
    oversampled. ``max_dim`` is only a RAM/runtime safety ceiling: it coarsens the
    requested spacing uniformly when the foreground support would exceed it.
    """
    if support_bounds is None:
        lo, hi = _snapshot_support_bounds(snapshot)
    else:
        lo = np.asarray(support_bounds[0], dtype=float).reshape(3)
        hi = np.asarray(support_bounds[1], dtype=float).reshape(3)
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(hi <= lo):
        raise ValueError("Registration working-grid support bounds are invalid.")

    target = np.asarray(target_spacing, dtype=float).reshape(-1)
    if target.size == 1:
        target = np.repeat(target, 3)
    if target.size != 3 or not np.all(np.isfinite(target)) or np.any(target <= 0.0):
        raise ValueError("Registration target working spacing must contain positive finite values.")

    source_spacing = _effective_world_spacing(snapshot)
    requested_spacing = np.maximum(source_spacing, target)
    extent = np.maximum(hi - lo, 1e-6)
    max_dim = max(32, int(max_dim))
    safety_scale = max(
        1.0,
        float(np.max(extent / np.maximum(requested_spacing * float(max_dim - 1), 1e-12))),
    )
    spacing = requested_spacing * safety_scale
    dims = np.ceil(extent / spacing).astype(int) + 1
    dims = np.maximum(8, dims)
    # Guard floating-point ceil at the safety boundary without making max_dim a
    # scientific resolution control again.
    if int(np.max(dims)) > max_dim:
        correction = float(np.max(dims)) / float(max_dim)
        spacing *= correction
        safety_scale *= correction
        dims = np.ceil(extent / spacing).astype(int) + 1
        dims = np.maximum(8, dims)
    origin = np.asarray(lo, dtype=float) + 0.5 * spacing
    direction = ((1.0, 0.0, 0.0),
                 (0.0, 1.0, 0.0),
                 (0.0, 0.0, 1.0))
    affine = grid_affine_from_components(origin, spacing, direction)
    return {
        "origin": tuple(float(v) for v in origin),
        "spacing": tuple(float(v) for v in spacing),
        "direction": direction,
        "requested_spacing": tuple(float(v) for v in requested_spacing),
        "target_spacing": tuple(float(v) for v in target),
        "source_world_spacing": tuple(float(v) for v in source_spacing),
        "safety_scale": float(safety_scale),
        "safety_max_dim": int(max_dim),
        "support_bounds_world": [
            [float(v) for v in lo],
            [float(v) for v in hi],
        ],
        "dims_xyz": tuple(int(v) for v in dims),
        "shape_zyx": (int(dims[2]), int(dims[1]), int(dims[0])),
        "affine": affine,
    }


def _native_working_array(snapshot, grid, *, normalize=False):
    """Resample one working image directly from the captured native source.

    Global ITK and nonlinear CMTK lattices are intentionally independent. Both
    therefore start from the original captured voxel array and world geometry;
    a coarse global working image is never used as the source for a finer CMTK
    image. CMTK receives finite float32 intensities without ITK's robust
    normalization, preserving the source intensity relationship for its metric.
    """
    array = _resample_zyx(
        snapshot["data"], snapshot["world_affine"], grid["affine"],
        grid["shape_zyx"], order=1, output_dtype=np.float32,
    )
    array = np.ascontiguousarray(array, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        np.nan_to_num(array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return _robust_normalize(array) if normalize else array


def _cmtk_nonlinear_working_grid(snapshot, support_bounds, settings):
    """Build the CMTK input lattice independently from global ITK settings."""
    settings = dict(settings or {})
    target_spacing = max(1e-6, float(settings.get("cmtk_working_spacing", 0.75)))
    max_dim = max(32, int(settings.get("cmtk_max_grid_dim", 768)))
    return _working_grid(
        snapshot, target_spacing, max_dim, support_bounds=support_bounds
    )


def _working_grid_payload(grid):
    """Return the serializable physical-grid fields used by registration results."""
    payload = {
        "origin": list(grid["origin"]),
        "spacing": list(grid["spacing"]),
        "dims_xyz": list(grid["dims_xyz"]),
        "direction": (
            np.asarray(grid["direction"], dtype=float).reshape(-1).tolist()
        ),
    }
    for key in (
        "requested_spacing", "target_spacing", "source_world_spacing",
        "safety_scale", "safety_max_dim", "support_bounds_world",
    ):
        if key in grid:
            payload[key] = copy.deepcopy(grid[key])
    return payload


def _transformed_support_bounds(snapshot, moving_to_fixed):
    explicit = snapshot.get("registration_support_bounds_world")
    if explicit is not None:
        lo, hi = _snapshot_support_bounds(snapshot)
        corners = np.array(
            [[x, y, z, 1.0]
             for x in (lo[0], hi[0])
             for y in (lo[1], hi[1])
             for z in (lo[2], hi[2])],
            dtype=float,
        )
        moved = (_matrix4(moving_to_fixed) @ corners.T).T[:, :3]
        return moved.min(axis=0), moved.max(axis=0)

    nx, ny, nz = (max(1, int(v)) for v in snapshot["dims"])
    corners = np.array(
        [[x, y, z, 1.0]
         for x in (-0.5, nx - 0.5)
         for y in (-0.5, ny - 0.5)
         for z in (-0.5, nz - 0.5)],
        dtype=float,
    )
    world = (_matrix4(snapshot["world_affine"]) @ corners.T).T
    moved = (_matrix4(moving_to_fixed) @ world.T).T[:, :3]
    return moved.min(axis=0), moved.max(axis=0)


def _support_overlap_fraction(fixed, moving, moving_to_fixed):
    flo, fhi = _snapshot_support_bounds(fixed)
    mlo, mhi = _transformed_support_bounds(moving, moving_to_fixed)
    inter = np.maximum(0.0, np.minimum(fhi, mhi) - np.maximum(flo, mlo))
    inter_volume = float(np.prod(inter))
    fixed_volume = max(1e-12, float(np.prod(np.maximum(0.0, fhi - flo))))
    moving_volume = max(1e-12, float(np.prod(np.maximum(0.0, mhi - mlo))))
    return inter_volume / max(1e-12, min(fixed_volume, moving_volume))


def _overlap_metric_error(exc):
    text = str(exc or "").lower()
    return (
        "all samples map outside moving image buffer" in text
        or "images do not sufficiently overlap" in text
        or "insufficient overlap" in text
        or "no valid points were found" in text
    )


def _platform_multithreader_error(exc):
    text = str(exc or "").lower()
    return (
        "platformmultithreader" in text
        or "singlemethodexecute" in text
        or "exception occurred during singlemethodexecute" in text
    )


def _itk_runtime_error(exc):
    text = str(exc or "").lower()
    return (
        "itk error" in text
        or "simpleitk" in text
        or _platform_multithreader_error(exc)
        or _overlap_metric_error(exc)
    )


def _set_process_threads(process, count):
    """Best-effort per-process concurrency cap for current ITK threaders."""
    target = max(1, int(count))
    changed = False
    # In ITK 5 SetNumberOfThreads is a maximum-thread cap, while work units
    # control how many subtasks the ProcessObject requests. Set both when
    # available so the single-thread safety paths remain single-work-unit too.
    for name in ("SetNumberOfThreads", "SetNumberOfWorkUnits"):
        try:
            fn = getattr(process, name, None)
            if callable(fn):
                fn(target)
                changed = True
        except Exception:
            pass
    return changed


def _sitk_module():
    try:
        import SimpleITK as sitk
    except Exception as exc:
        raise RuntimeError(
            "Registration requires SimpleITK for global rigid/affine optimization. "
            "Install it in the MADI3D environment with: pip install SimpleITK"
        ) from exc
    return sitk


def _sitk_image(array_zyx, grid):
    sitk = _sitk_module()
    image = sitk.GetImageFromArray(np.ascontiguousarray(array_zyx, dtype=np.float32))
    image.SetOrigin(tuple(grid["origin"]))
    image.SetSpacing(tuple(grid["spacing"]))
    image.SetDirection(
        tuple(np.asarray(grid["direction"], dtype=float).reshape(-1))
    )
    return image


_LEGACY_DEFORMABLE_SETTING_KEYS = frozenset({
    "bspline",
    "bspline_grid_spacing",
    "bspline_refinements",
    "bspline_final_spacing",
    "bspline_iterations",
    "bspline_max_step_size",
    "bspline_step_control",
    "bspline_max_displacement",
    "bspline_learning_rate",
    "bspline_learning_rate_decay",
    "bspline_regularization_sigma",
    "deformable_metric",
    "deformable_optimizer",
    "deformable_thread_cap",
    "deformable_convergence_minimum",
    "deformable_convergence_window",
    "jacobian_max_dim",
    "reject_folding",
    "minimum_jacobian",
    "deformable_retry_single_thread",
})


_CMTK_LINEAR_DOF_BY_MODEL = {
    "rigid": 6,
    "similarity": 7,
    "scaleversor": 9,
    "affine": 12,
}
_CMTK_LINEAR_METRIC_BY_GLOBAL_METRIC = {
    "mattes": "nmi",
    "joint_mi": "mi",
    "ants_ncc": "ncc",
    "correlation": "ncc",
    "meansquares": "msd",
}


def _linear_engine(settings):
    """Return the explicitly selected global linear engine."""
    engine = str(dict(settings or {}).get("linear_engine", "itk") or "itk").strip().lower()
    if engine not in {"itk", "cmtk"}:
        raise RuntimeError(
            f"Unsupported linear registration engine: {engine!r}. Expected 'itk' or 'cmtk'."
        )
    return engine


def _cmtk_linear_backend_ui_state(manager):
    """Return display-only readiness from persisted state; never probe CMTK/WSL."""
    required = tuple(CMTKRegistrationRunner.LINEAR_REQUIRED_TOOLS)
    try:
        status = manager.persisted_status()
    except Exception:
        status = getattr(manager, "last_status", None)
    ready = bool(getattr(status, "ready", False))
    if ready:
        label = str(getattr(status, "backend", "") or "CMTK")
        version = str(getattr(status, "version", "") or "").strip()
        display_version = version[5:].strip() if version.lower().startswith("cmtk ") else version
        suffix = f" ({display_version})" if display_version else ""
        return {
            "ready": True,
            "combo_text": "CMTK",
            "summary": f"CMTK ready{suffix}: {label}.",
            "required_tools": required,
        }

    prior = str(getattr(status, "summary", "") or "").strip()
    state = str(getattr(status, "state", "") or "").lower()
    if state == "unavailable":
        summary = (
            "CMTK setup/repair is required. Selecting CMTK is allowed; Calculate registration "
            "will open the dependency setup path. MADI3D will not silently fall back to ITK."
        )
    else:
        summary = (
            "CMTK has not been validated on this installation yet. Selecting CMTK is allowed; "
            "the first CMTK action will run the dependency setup/check path. "
            "MADI3D will not silently fall back to ITK."
        )
    if prior and prior not in {"CMTK is ready.", "CMTK has not been checked yet."}:
        summary += f" Last CMTK status: {prior}"
    return {
        "ready": False,
        "combo_text": "CMTK — setup required",
        "summary": summary,
        "required_tools": required,
    }


def _cmtk_linear_settings_from_registration(settings):
    """Map common registration settings onto the first-class CMTK linear runner."""
    settings = dict(settings or {})
    model = str(settings.get("global_model", "affine") or "affine").lower()
    if model not in _CMTK_LINEAR_DOF_BY_MODEL:
        raise RuntimeError(f"Unsupported global registration model for CMTK: {model!r}.")
    metric = str(settings.get("cmtk_linear_metric") or "").strip().lower()
    if not metric:
        metric = _CMTK_LINEAR_METRIC_BY_GLOBAL_METRIC.get(
            str(settings.get("metric", "mattes") or "mattes").lower(), "nmi"
        )
    return CMTKLinearSettings(
        metric=metric,
        final_dof=_CMTK_LINEAR_DOF_BY_MODEL[model],
        exploration=float(settings.get("cmtk_linear_exploration", 16.0) or 16.0),
        accuracy=float(settings.get("cmtk_linear_accuracy", 0.8) or 0.8),
        coarsest=float(settings.get("cmtk_linear_coarsest", 8.0) or 8.0),
        threads=max(1, int(settings.get("cmtk_linear_threads", settings.get("cmtk_threads", 1)) or 1)),
    ).validated()


def _cmtk_required_tools_for_registration(settings):
    """Return the exact CMTK capabilities required by the selected workflow."""
    settings = dict(settings or {})
    tools = []
    if _linear_engine(settings) == "cmtk":
        tools.extend(CMTKRegistrationRunner.LINEAR_REQUIRED_TOOLS)
    if bool(settings.get("deformable", False)):
        tools.extend(CMTKRegistrationRunner.REQUIRED_TOOLS)
    return tuple(dict.fromkeys(tools))


def _cmtk_working_to_world_moving_to_reference(
    moving_to_reference_working, reference_origin_world, floating_origin_world
):
    """Invert ``cmtk_working_moving_to_reference`` without changing direction."""
    local = _matrix4(moving_to_reference_working)
    reference_origin = np.asarray(reference_origin_world, dtype=float).reshape(3)
    floating_origin = np.asarray(floating_origin_world, dtype=float).reshape(3)
    if not np.all(np.isfinite(reference_origin)) or not np.all(np.isfinite(floating_origin)):
        raise ValueError("CMTK working-grid origins must be finite.")
    from_reference_local = np.eye(4, dtype=float)
    from_reference_local[:3, 3] = reference_origin
    to_floating_local = np.eye(4, dtype=float)
    to_floating_local[:3, 3] = -floating_origin
    return _matrix4(from_reference_local @ local @ to_floating_local)


def _cmtk_linear_path_labels(dof_sequence):
    labels = {6: "Rigid", 7: "Similarity", 9: "Scale", 12: "Affine"}
    return [labels[int(dof)] for dof in tuple(dof_sequence or ()) if int(dof) in labels]


def _cmtk_warp_settings_from_registration(settings):
    """Return validated settings for the only supported nonlinear engine: CMTK."""
    settings = dict(settings or {})
    legacy = sorted(key for key in _LEGACY_DEFORMABLE_SETTING_KEYS if key in settings)
    if legacy:
        raise RuntimeError(
            "This registration configuration contains removed SimpleITK deformable settings: "
            + ", ".join(legacy)
            + ". Reapply a current CMTK preset or configure the CMTK controls directly."
        )

    cmtk = CMTKWarpSettings(
        metric=str(settings.get("cmtk_metric", "nmi") or "nmi").lower(),
        exploration=float(settings.get("cmtk_exploration", 26.0) or 26.0),
        accuracy=float(settings.get("cmtk_accuracy", 0.8) or 0.8),
        coarsest=float(settings.get("cmtk_coarsest", 8.0) or 8.0),
        grid_spacing=float(settings.get("cmtk_grid_spacing", 80.0) or 80.0),
        refine=max(0, int(settings.get("cmtk_refine", 4) or 0)),
        mode=str(settings.get("cmtk_mode", "fast") or "fast"),
        threads=max(1, int(settings.get("cmtk_threads", 1) or 1)),
        energy_weight=float(settings.get("cmtk_energy_weight", 0.1) or 0.0),
        jacobian_weight=float(settings.get("cmtk_jacobian_weight", 0.0) or 0.0),
        inverse_consistency_weight=float(settings.get("cmtk_inverse_consistency_weight", 0.0) or 0.0),
        omit_original_data=bool(settings.get("cmtk_omit_original_data", False)),
        match_histograms=bool(settings.get("cmtk_match_histograms", False)),
    ).validated()
    mapping = {
        "cmtk_metric": cmtk.metric,
        "grid_spacing": float(cmtk.grid_spacing),
        "refine": int(cmtk.refine),
        "approx_final_spacing": float(cmtk.grid_spacing) / float(2 ** int(cmtk.refine)),
        "threads": int(cmtk.threads),
    }
    return cmtk, mapping



def _cleanup_runtime_cmtk_chains(chains):
    """Remove only MADI-created Commit-4 temporary CMTK workspaces."""
    temp_root = Path(tempfile.gettempdir()).resolve()
    for chain in list(chains or []):
        try:
            model = dict(getattr(chain, "deformation_model", {}) or {})
            if model.get("type") != "cmtk_warp" or model.get("artifact_scope") != "runtime_temporary":
                continue
            raw_root = str(model.get("runtime_artifact_root") or "").strip()
            if not raw_root:
                continue
            root = Path(raw_root).resolve()
            if root.parent != temp_root or not root.name.startswith("madi3d-cmtk-registration-"):
                continue
            shutil.rmtree(root, ignore_errors=True)
        except Exception:
            pass


def _sitk_linear_matrix(transform):
    """Return a homogeneous matrix for a centered SimpleITK linear transform.

    Treat linear transforms without GetTranslation() as zero-translation
    centered maps instead of assuming one common accessor set.
    """
    matrix = np.asarray(transform.GetMatrix(), dtype=float).reshape(3, 3)
    try:
        center = np.asarray(transform.GetCenter(), dtype=float)
    except Exception:
        center = np.zeros(3, dtype=float)
    getter = getattr(transform, "GetTranslation", None)
    if callable(getter):
        translation = np.asarray(getter(), dtype=float)
    else:
        translation = np.zeros(3, dtype=float)
    out = np.eye(4, dtype=float)
    out[:3, :3] = matrix
    out[:3, 3] = center + translation - matrix @ center
    return out


def _translation_parameter_for_matrix(matrix4, center):
    arr = _matrix4(matrix4)
    c = np.asarray(center, dtype=float)
    return arr[:3, 3] - c + arr[:3, :3] @ c


def _polar_rotation_scale(linear):
    a = np.asarray(linear, dtype=float).reshape(3, 3)
    u, s, vh = np.linalg.svd(a)
    r = u @ vh
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        s[-1] *= -1.0
        r = u @ vh
    return r, s


def _linear_transform_for_stage(stage, fixed_to_moving_matrix, center):
    """Construct a SimpleITK transform initialized from a fixed->moving matrix."""
    sitk = _sitk_module()
    arr = _matrix4(fixed_to_moving_matrix)
    c = tuple(float(v) for v in center)
    linear = arr[:3, :3]
    t = _translation_parameter_for_matrix(arr, center)
    kind = str(stage)

    if kind == "rigid":
        r, _ = _polar_rotation_scale(linear)
        desired = np.eye(4, dtype=float)
        desired[:3, :3] = r
        desired[:3, 3] = arr[:3, 3]
        rigid_t = _translation_parameter_for_matrix(desired, center)
        tx = sitk.Euler3DTransform()
        tx.SetCenter(c)
        tx.SetMatrix(tuple(r.ravel()))
        tx.SetTranslation(tuple(float(v) for v in rigid_t))
        return tx

    if kind == "similarity":
        r, _scales = _polar_rotation_scale(linear)
        scale = float(np.cbrt(max(1e-12, abs(np.linalg.det(linear)))))
        sim_linear = r * scale
        desired = np.eye(4, dtype=float)
        desired[:3, :3] = sim_linear
        desired[:3, 3] = arr[:3, 3]
        sim_t = _translation_parameter_for_matrix(desired, center)
        tx = sitk.Similarity3DTransform()
        tx.SetCenter(c)
        tx.SetMatrix(tuple(sim_linear.ravel()))
        tx.SetTranslation(tuple(float(v) for v in sim_t))
        return tx

    if kind == "scaleversor":
        # ScaleVersor3DTransform is the native joint 9-parameter ITK model:
        # versor rotation (3), translation (3), and independent XYZ scale (3).
        # The preceding rigid stage provides the warm-start, so initialize from
        # its proper rotation/translation and neutral scale without resampling
        # the moving image or freezing any parameter group.
        r, _ = _polar_rotation_scale(linear)
        desired = np.eye(4, dtype=float)
        desired[:3, :3] = r
        desired[:3, 3] = arr[:3, 3]
        scaleversor_t = _translation_parameter_for_matrix(desired, center)

        rigid_seed = sitk.VersorRigid3DTransform()
        rigid_seed.SetMatrix(tuple(float(v) for v in r.ravel()))

        tx = sitk.ScaleVersor3DTransform()
        tx.SetCenter(c)
        tx.SetRotation(rigid_seed.GetVersor())
        tx.SetTranslation(tuple(float(v) for v in scaleversor_t))
        tx.SetScale((1.0, 1.0, 1.0))
        return tx

    if kind == "affine":
        tx = sitk.AffineTransform(3)
        tx.SetCenter(c)
        tx.SetMatrix(tuple(linear.ravel()))
        tx.SetTranslation(tuple(float(v) for v in t))
        return tx

    raise ValueError(f"Unsupported linear registration stage: {stage}")


def _affine_qc_components(matrix4):
    """Return scientist-facing affine geometry diagnostics in MADI direction.

    Principal stretches are the singular values of the 3x3 linear block.  The
    symmetric polar stretch matrix separates rotation from scale/shear; its
    normalized off-diagonal terms provide dimensionless XY/XZ/YZ shear
    diagnostics while the transformed unit-axis lengths report intuitive XYZ
    axis stretch.
    """
    arr = _matrix4(matrix4)
    linear = np.asarray(arr[:3, :3], dtype=float)
    determinant = float(np.linalg.det(linear))
    _u, singular, vh = np.linalg.svd(linear)
    stretch = vh.T @ np.diag(singular) @ vh
    axis_stretch = np.linalg.norm(linear, axis=0)
    minimum = float(np.min(singular))
    maximum = float(np.max(singular))
    condition = maximum / max(minimum, 1e-12)

    shear = {}
    for name, i, j in (("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)):
        denom = math.sqrt(max(1e-24, float(stretch[i, i] * stretch[j, j])))
        shear[name] = float(stretch[i, j] / denom)

    return {
        "determinant": determinant,
        # Keep the established key for saved-result readers and existing tests.
        "singular_values": [float(v) for v in singular],
        "principal_stretches": [float(v) for v in singular],
        "axis_stretch_xyz": [float(v) for v in axis_stretch],
        "scale_xyz": [float(stretch[i, i]) for i in range(3)],
        "symmetric_stretch_matrix": [
            [float(v) for v in row] for row in np.asarray(stretch, dtype=float)
        ],
        "shear_xy": shear["xy"],
        "shear_xz": shear["xz"],
        "shear_yz": shear["yz"],
        "max_abs_shear": float(max(abs(v) for v in shear.values())),
        "condition_number": float(condition),
        "translation_xyz": [float(v) for v in arr[:3, 3]],
    }


def _linear_qc_thresholds():
    return {
        "minimum_positive_determinant": 1.0e-8,
        "minimum_principal_stretch": 1.0e-8,
        "minimum_scale": 0.05,
        "maximum_scale": 20.0,
        "maximum_condition_number": 100.0,
        "minimum_support_overlap_fraction": 1.0e-5,
        "substantial_nmi_absolute_drop": 0.03,
        "substantial_nmi_relative_drop": 0.02,
        "substantial_ncc_drop": 0.15,
        "substantial_overlap_absolute_drop": 0.05,
        "substantial_overlap_remaining_fraction": 0.5,
    }


def _deformation_qc_thresholds():
    thresholds = _linear_qc_thresholds()
    return {
        "maximum_nonpositive_jacobian_fraction": 0.0,
        "substantial_nmi_absolute_drop": thresholds["substantial_nmi_absolute_drop"],
        "substantial_nmi_relative_drop": thresholds["substantial_nmi_relative_drop"],
        "substantial_ncc_drop": thresholds["substantial_ncc_drop"],
        "substantial_overlap_absolute_drop": thresholds["substantial_overlap_absolute_drop"],
        "substantial_overlap_remaining_fraction": thresholds["substantial_overlap_remaining_fraction"],
    }


def _linear_transform_sanity(matrix4, kind="affine"):
    """Evaluate transform plausibility without discarding invertible output.

    The returned diagnostics use MADI's moving-to-reference direction. A false
    result is scientific QC, not an artifact failure; finite invertible
    candidates remain available for inspection and explicit use.
    """
    thresholds = _linear_qc_thresholds()
    try:
        details = _affine_qc_components(matrix4)
        determinant = float(details["determinant"])
        singular = np.asarray(details["principal_stretches"], dtype=float)
        condition = float(details["condition_number"])
    except Exception as exc:
        return False, f"invalid linear transform: {exc}", {}

    if not math.isfinite(determinant) or determinant <= thresholds["minimum_positive_determinant"]:
        return False, f"non-positive or singular determinant ({determinant:.6g})", details
    if not np.all(np.isfinite(singular)) or float(np.min(singular)) <= thresholds["minimum_principal_stretch"]:
        return False, "non-finite or near-zero singular value", details

    minimum = float(np.min(singular))
    maximum = float(np.max(singular))

    # These are safety rails, not biological priors. They are deliberately broad
    # enough for legitimate cross-specimen affine registration while catching an
    # optimizer that has escaped into an obviously destructive solution.
    if minimum < thresholds["minimum_scale"] or maximum > thresholds["maximum_scale"]:
        return False, f"extreme affine scale (singular values {singular.tolist()})", details
    if condition > thresholds["maximum_condition_number"]:
        return False, f"ill-conditioned affine transform (condition {condition:.6g})", details

    return True, "", details


def _linear_transform_artifact_error(matrix4):
    """Return an artifact error only for unusable affine output.

    Plausibility thresholds remain in ``_linear_transform_sanity`` and are QC,
    not execution gates. A finite invertible optimizer result therefore remains
    inspectable even when its scale, overlap, or conditioning fails QC.
    """
    try:
        invertible_affine4(matrix4, "Optimized registration transform")
    except Exception as exc:
        return f"invalid optimized transform artifact: {exc}"
    return ""


def _deformation_result_qc_status(deformation_qc):
    deformation_qc = dict(deformation_qc or {})
    warnings = [str(value).lower() for value in deformation_qc.get("warnings") or ()]
    jacobian = dict(deformation_qc.get("jacobian_local") or {})
    if (
        float(jacobian.get("nonpositive_fraction", 0.0) or 0.0)
        > _deformation_qc_thresholds()["maximum_nonpositive_jacobian_fraction"]
    ):
        return "failed"
    failure_terms = (
        "decreased substantially",
        "overlap fell substantially",
        "similarity qc is incomplete",
        "implausible displacement",
        "landmark disagreement",
    )
    if any(any(term in warning for term in failure_terms) for warning in warnings):
        return "failed"
    return "warning" if warnings else "passed"


def _linear_stage_pipeline(global_model):
    """Return the scientifically ordered ITK linear stages for one final model."""
    model = str(global_model or "affine").lower()
    if model == "rigid":
        return [("rigid", "Rigid 6 DOF", True)]
    if model == "similarity":
        return [
            ("rigid", "Rigid warm-start", True),
            ("similarity", "Similarity 7 DOF", True),
        ]
    if model == "scaleversor":
        return [
            ("rigid", "Rigid warm-start", True),
            ("scaleversor", "Anisotropic scale 9 DOF", True),
        ]
    if model == "affine":
        return [
            ("rigid", "Rigid warm-start", True),
            ("scaleversor", "Anisotropic scale 9 DOF (affine warm-start)", True),
            ("affine", "Affine 12 DOF", True),
        ]
    raise ValueError(f"Unsupported global registration model: {global_model!r}.")


def _registration_stage_path(stages):
    """Return the successfully executed transform path in compact terms."""
    labels = {
        "rigid": "Rigid",
        "similarity": "Similarity",
        "scaleversor": "Scale",
        "affine": "Affine",
        "cmtk_warp": "Warp",
    }
    path = []
    for stage in list(stages or []):
        status = stage.get("execution_status", "pending") if isinstance(stage, dict) else getattr(stage, "execution_status", "pending")
        kind = stage.get("kind", "") if isinstance(stage, dict) else getattr(stage, "kind", "")
        details = stage.get("details", {}) if isinstance(stage, dict) else getattr(stage, "details", {})
        if not execution_succeeded(status):
            continue
        if str(dict(details or {}).get("engine") or "").lower() == "cmtk":
            for cmtk_label in _cmtk_linear_path_labels(dict(details or {}).get("dof_sequence") or ()):
                if not path or path[-1] != cmtk_label:
                    path.append(cmtk_label)
            if dict(details or {}).get("dof_sequence"):
                continue
        label = labels.get(str(kind or "").lower())
        if not label:
            continue
        if not path or path[-1] != label:
            path.append(label)
    return " → ".join(path) if path else "Initial only"


def _linear_qc_text(qc, precision=4):
    """Format decomposed polar scale and shear values without hiding components."""
    qc = dict(qc or {})
    try:
        digits = max(1, int(precision))
    except Exception:
        digits = 4
    parts = []
    scale = list(qc.get("scale_xyz") or qc.get("axis_stretch_xyz") or ())
    if len(scale) == 3:
        parts.append("scale XYZ=" + "/".join(f"{float(v):.{digits}f}" for v in scale))
    shears = [qc.get("shear_xy"), qc.get("shear_xz"), qc.get("shear_yz")]
    if all(value is not None for value in shears):
        parts.append(
            "shear XY/XZ/YZ="
            + "/".join(f"{float(v):+.{digits}f}" for v in shears)
        )
        if qc.get("max_abs_shear") is not None:
            parts.append(f"max |shear|={float(qc['max_abs_shear']):.{digits}f}")
    return "; ".join(parts)


def _configure_metric(
    registration, metric_name, bins, sampling, seed,
    fixed_mask=None, moving_mask=None, neighborhood_radius=4,
    sampling_per_level=None, sampling_strategy="random",
    gradient_memory_strategy="precompute",
):
    """Configure one SimpleITK metric with reproducible, explicit sampling behavior.

    ``sampling`` remains the fine-level/user target.  When ``sampling_per_level``
    is supplied, SimpleITK receives one percentage for each pyramid level so a
    coarse fluorescence level does not accidentally collapse to only a few valid
    metric samples.  Gradient filters are configurable because precomputed image
    gradients are fast but can dominate RAM on large working grids.
    """
    sitk = _sitk_module()
    metric = str(metric_name or "mattes").lower()
    if metric == "joint_mi":
        registration.SetMetricAsJointHistogramMutualInformation(max(8, int(bins)))
    elif metric == "ants_ncc":
        registration.SetMetricAsANTSNeighborhoodCorrelation(max(1, int(neighborhood_radius)))
    elif metric == "correlation":
        registration.SetMetricAsCorrelation()
    elif metric == "meansquares":
        registration.SetMetricAsMeanSquares()
    else:
        registration.SetMetricAsMattesMutualInformation(max(8, int(bins)))

    strategy = str(sampling_strategy or "random").lower()
    sitk_strategy = registration.REGULAR if strategy == "regular" else registration.RANDOM
    percentages = None
    if sampling_per_level is not None:
        try:
            percentages = [max(0.001, min(1.0, float(v))) for v in list(sampling_per_level)]
        except Exception:
            percentages = None
    if percentages:
        if all(v >= 0.999 for v in percentages):
            registration.SetMetricSamplingStrategy(registration.NONE)
        else:
            registration.SetMetricSamplingStrategy(sitk_strategy)
            registration.SetMetricSamplingPercentagePerLevel(percentages, int(seed) & 0xFFFFFFFF)
    else:
        pct = max(0.001, min(1.0, float(sampling)))
        if pct < 0.999:
            registration.SetMetricSamplingStrategy(sitk_strategy)
            registration.SetMetricSamplingPercentage(pct, int(seed) & 0xFFFFFFFF)
        else:
            registration.SetMetricSamplingStrategy(registration.NONE)

    gradient_mode = str(gradient_memory_strategy or "precompute").lower()
    if gradient_mode == "on_demand":
        try:
            registration.MetricUseFixedImageGradientFilterOff()
            registration.MetricUseMovingImageGradientFilterOff()
        except Exception:
            pass
    else:
        try:
            registration.MetricUseFixedImageGradientFilterOn()
            registration.MetricUseMovingImageGradientFilterOn()
        except Exception:
            pass

    if fixed_mask is not None:
        registration.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        registration.SetMetricMovingMask(moving_mask)
    registration.SetInterpolator(sitk.sitkLinear)


def _registration_foreground_mask(image, array, enabled, detector_max_dim=160):
    """Return a conservative connected foreground mask for the ITK metric."""
    if not enabled:
        return None
    sitk = _sitk_module()
    detector = _foreground_support_detector(array, max_dim=detector_max_dim)
    if not detector.get("valid"):
        return None
    support = np.asarray(detector.get("_support_mask_detector"), dtype=bool)
    stride = np.asarray(detector.get("detector_stride_zyx"), dtype=int)
    expanded = support
    for axis in range(3):
        if int(stride[axis]) > 1:
            expanded = np.repeat(expanded, int(stride[axis]), axis=axis)
    shape = np.asarray(array).shape
    expanded = expanded[:shape[0], :shape[1], :shape[2]]
    if expanded.shape != tuple(shape):
        padded = np.zeros(shape, dtype=bool)
        slices = tuple(slice(0, min(shape[i], expanded.shape[i])) for i in range(3))
        padded[slices] = expanded[slices]
        expanded = padded
    if int(np.count_nonzero(expanded)) < 64:
        return None
    mask = sitk.GetImageFromArray(np.ascontiguousarray(expanded, dtype=np.uint8))
    mask.CopyInformation(image)
    return mask


def _parse_float_schedule(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        raw = [part for part in re.split(r"[,;\s]+", str(value).strip()) if part]
    out = []
    for part in raw:
        try:
            out.append(max(0.0, float(part)))
        except Exception:
            return []
    return out


def _configure_multires(registration, shrink_factors, smoothing_mode="auto", smoothing_values=None):
    """Configure the image pyramid using working-grid voxel-aware smoothing.

    Auto sigma values are expressed in working-image voxel units, not hard-coded
    scene units.  This keeps the same anti-aliasing behavior when registration
    working spacing changes between datasets.  Full resolution is intentionally
    left unsmoothed.
    """
    shrink = [max(1, int(v)) for v in shrink_factors]
    registration.SetShrinkFactorsPerLevel(shrink)
    mode = str(smoothing_mode or "auto").lower()
    if mode == "none":
        sigmas = [0.0 for _ in shrink]
    elif mode == "custom":
        requested = _parse_float_schedule(smoothing_values)
        if len(requested) != len(shrink):
            raise RuntimeError(
                "Custom pyramid smoothing must provide one non-negative sigma for each resolution level."
            )
        sigmas = requested
    else:
        sigmas = [0.0 if int(v) <= 1 else 0.5 * float(v) for v in shrink]
    registration.SetSmoothingSigmasPerLevel(sigmas)
    try:
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    except Exception:
        try:
            registration.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(False)
        except Exception:
            pass
    return [float(v) for v in sigmas]


def _auto_metric_sampling_schedule(image, shrink_factors, fine_fraction, minimum_candidates=8000):
    """Boost coarse-level sampling only enough to avoid tiny candidate sets."""
    shrink = [max(1, int(v)) for v in shrink_factors]
    fine = max(0.001, min(1.0, float(fine_fraction)))
    size = np.asarray(image.GetSize(), dtype=np.int64)
    schedule = []
    for factor in shrink:
        level_size = np.maximum(1, np.ceil(size / float(factor)).astype(np.int64))
        candidates = max(1, int(np.prod(level_size, dtype=np.int64)))
        required = float(max(256, int(minimum_candidates))) / float(candidates)
        schedule.append(max(fine, min(1.0, required)))
    return schedule


def _resolve_gradient_memory_strategy(setting, image, sampling_fraction):
    mode = str(setting or "auto").lower()
    if mode in {"precompute", "on_demand"}:
        return mode
    try:
        voxels = int(np.prod(np.asarray(image.GetSize(), dtype=np.int64), dtype=np.int64))
    except Exception:
        voxels = 0
    return "on_demand" if voxels >= 2_000_000 and float(sampling_fraction) < 0.35 else "precompute"


def _normalized_mutual_information(a, b, mask=None, bins=64):
    """Deterministic full-grid NMI diagnostic for normalized registration images.

    This is deliberately independent from the optimizer's stochastic metric
    sampling.  It uses one fixed 64-bin joint histogram by default and the
    entropy-ratio definition (H(A)+H(B))/H(A,B), where larger values indicate
    stronger statistical dependence.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if np.count_nonzero(valid) < 128:
        return float("nan")
    av = np.clip(a[valid].astype(np.float64, copy=False), 0.0, 1.0)
    bv = np.clip(b[valid].astype(np.float64, copy=False), 0.0, 1.0)
    hist, _x, _y = np.histogram2d(
        av, bv, bins=max(8, int(bins)), range=((0.0, 1.0), (0.0, 1.0))
    )
    total = float(hist.sum())
    if total <= 0.0:
        return float("nan")
    pxy = hist / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    def entropy(prob):
        values = np.asarray(prob, dtype=np.float64)
        values = values[values > 0.0]
        return -float(np.sum(values * np.log(values))) if values.size else 0.0

    hxy = entropy(pxy)
    if hxy <= 1e-15:
        return float("nan")
    return float((entropy(px) + entropy(py)) / hxy)



def _array_similarity_diagnostics(fixed_array, moving_array, *, nmi_bins=64):
    """Return deterministic NMI/NCC and foreground overlap on one shared grid."""
    fixed = _robust_normalize(np.asarray(fixed_array, dtype=np.float32))
    moving = _robust_normalize(np.asarray(moving_array, dtype=np.float32))
    if fixed.shape != moving.shape:
        raise ValueError(
            f"Similarity QC arrays must share one grid; got {fixed.shape} and {moving.shape}."
        )
    fixed_fg = np.isfinite(fixed) & (fixed > 0.02)
    moving_fg = np.isfinite(moving) & (moving > 0.02)
    fixed_count = int(np.count_nonzero(fixed_fg))
    moving_count = int(np.count_nonzero(moving_fg))
    overlap_count = int(np.count_nonzero(fixed_fg & moving_fg))
    foreground_overlap = (
        float(overlap_count / max(1, min(fixed_count, moving_count)))
        if fixed_count >= 128 and moving_count >= 128 else float("nan")
    )
    metric_mask = fixed_fg if fixed_count >= 128 else None
    valid = np.isfinite(fixed) & np.isfinite(moving)
    if metric_mask is not None:
        valid &= metric_mask
    return {
        "ncc": _ncc(fixed, moving, metric_mask),
        "nmi": _normalized_mutual_information(
            fixed, moving, metric_mask, bins=nmi_bins
        ),
        "valid_voxels": int(np.count_nonzero(valid)),
        "nmi_bins": int(max(8, int(nmi_bins))),
        "foreground_overlap_fraction": foreground_overlap,
        "fixed_foreground_voxels": fixed_count,
        "moving_foreground_voxels": moving_count,
        "overlap_foreground_voxels": overlap_count,
    }


def _deformation_qc_warnings(linear_similarity, warp_similarity, jacobian, configured_metric):
    """Combine common image QC with CMTK-local Jacobian warnings."""
    thresholds = _deformation_qc_thresholds()
    linear_similarity = dict(linear_similarity or {})
    warp_similarity = dict(warp_similarity or {})
    jacobian = dict(jacobian or {})
    warnings = list(jacobian.get("warnings") or [])
    warnings.extend(
        _linear_quality_warnings(
            linear_similarity,
            warp_similarity,
            configured_metric,
        )
    )

    before_overlap = float(
        linear_similarity.get("foreground_overlap_fraction", float("nan"))
    )
    after_overlap = float(
        warp_similarity.get("foreground_overlap_fraction", float("nan"))
    )
    if (
        math.isfinite(before_overlap) and math.isfinite(after_overlap)
        and before_overlap - after_overlap > thresholds["substantial_overlap_absolute_drop"]
        and after_overlap < thresholds["substantial_overlap_remaining_fraction"] * max(before_overlap, 1e-12)
    ):
        warnings.append(
            f"foreground overlap fell substantially ({before_overlap:.5f} -> {after_overlap:.5f})"
        )

    for label, similarity in (("affine-seed", linear_similarity), ("final-warp", warp_similarity)):
        nmi = float(similarity.get("nmi", float("nan")))
        ncc = float(similarity.get("ncc", float("nan")))
        if not math.isfinite(nmi) or not math.isfinite(ncc):
            warnings.append(
                f"{label} deterministic similarity QC is incomplete (NMI={nmi}, NCC={ncc})"
            )
    return list(dict.fromkeys(str(value) for value in warnings if str(value).strip()))


def _resampled_similarity_diagnostics(
    fixed_image, moving_image, fixed_to_moving_transform, fixed_mask=None, *, nmi_bins=64
):
    """Return deterministic NCC/NMI after one fixed-grid resampling."""
    sitk = _sitk_module()
    warped = sitk.Resample(
        moving_image, fixed_image, fixed_to_moving_transform,
        sitk.sitkLinear, 0.0, sitk.sitkFloat32,
    )
    a = sitk.GetArrayViewFromImage(fixed_image)
    b = sitk.GetArrayViewFromImage(warped)
    mask = None
    if fixed_mask is not None:
        mask = sitk.GetArrayViewFromImage(fixed_mask) > 0
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return {
        "ncc": _ncc(a, b, mask),
        "nmi": _normalized_mutual_information(a, b, mask, bins=nmi_bins),
        "valid_voxels": int(np.count_nonzero(valid)),
        "nmi_bins": int(max(8, int(nmi_bins))),
    }


def _resampled_ncc(fixed_image, moving_image, fixed_to_moving_transform, fixed_mask=None):
    return float(
        _resampled_similarity_diagnostics(
            fixed_image, moving_image, fixed_to_moving_transform, fixed_mask
        )["ncc"]
    )


def _linear_quality_warnings(before, after, configured_metric, *, overlap_before=None, overlap_after=None):
    """Return strong QC warnings without making image similarity a hard gate."""
    thresholds = _linear_qc_thresholds()
    before = dict(before or {})
    after = dict(after or {})
    warnings = []

    before_nmi = float(before.get("nmi", float("nan")))
    after_nmi = float(after.get("nmi", float("nan")))
    if math.isfinite(before_nmi) and math.isfinite(after_nmi):
        drop = before_nmi - after_nmi
        relative = drop / max(abs(before_nmi), 1e-12)
        if (
            drop > thresholds["substantial_nmi_absolute_drop"]
            and relative > thresholds["substantial_nmi_relative_drop"]
        ):
            warnings.append(
                f"deterministic NMI decreased substantially ({before_nmi:.5f} -> {after_nmi:.5f})"
            )

    before_ncc = float(before.get("ncc", float("nan")))
    after_ncc = float(after.get("ncc", float("nan")))
    if math.isfinite(before_ncc) and math.isfinite(after_ncc):
        if before_ncc - after_ncc > thresholds["substantial_ncc_drop"]:
            warnings.append(
                f"deterministic NCC decreased substantially ({before_ncc:.5f} -> {after_ncc:.5f})"
            )

    if overlap_before is not None and overlap_after is not None:
        before_overlap = float(overlap_before)
        after_overlap = float(overlap_after)
        if (
            math.isfinite(before_overlap) and math.isfinite(after_overlap)
            and before_overlap - after_overlap > thresholds["substantial_overlap_absolute_drop"]
            and after_overlap < thresholds["substantial_overlap_remaining_fraction"] * max(before_overlap, 1e-12)
        ):
            warnings.append(
                f"support overlap fell substantially ({before_overlap:.5f} -> {after_overlap:.5f})"
            )

    metric = str(configured_metric or "mattes").lower()
    if warnings and metric in {"mattes", "joint_mi", "mi", "nmi"}:
        warnings.append(
            "optimizer used an information-theoretic metric; deterministic NCC is secondary QC and was not used as a rejection gate"
        )
    return warnings


def _sitk_affine_from_matrix(matrix4):
    """Return a generic physical-space affine for an explicit 4x4 matrix."""
    sitk = _sitk_module()
    arr = _matrix4(matrix4)
    tx = sitk.AffineTransform(3)
    tx.SetCenter((0.0, 0.0, 0.0))
    tx.SetMatrix(tuple(float(v) for v in arr[:3, :3].ravel()))
    tx.SetTranslation(tuple(float(v) for v in arr[:3, 3]))
    return tx


def _native_fixed_to_moving_transform(transform_payload):
    """Return the linear reference->moving map required by SimpleITK Reformat."""
    payload = dict(transform_payload or {})
    moving_to_fixed = _matrix4(payload.get("cumulative_moving_to_fixed") or np.eye(4))
    return _sitk_affine_from_matrix(np.linalg.inv(moving_to_fixed))


def _reformat_local_transform(transform_payload, reference_actor_matrix, source_actor_matrix):
    """Map reference-local physical coordinates to source-local physical coordinates."""
    sitk = _sitk_module()
    reference_to_world = _sitk_affine_from_matrix(reference_actor_matrix)
    world_registration = _native_fixed_to_moving_transform(transform_payload)
    world_to_source = _sitk_affine_from_matrix(np.linalg.inv(_matrix4(source_actor_matrix)))
    composite = sitk.CompositeTransform(3)
    # Desired map: world_to_source(world_registration(reference_to_world(p))).
    composite.AddTransform(world_to_source)
    composite.AddTransform(world_registration)
    composite.AddTransform(reference_to_world)
    try:
        composite.FlattenTransform()
    except Exception:
        pass
    return composite



def _polydata_points_fingerprint(polydata):
    """Stable SHA-256 fingerprint of current mesh point coordinates."""
    if polydata is None or polydata.GetPoints() is None:
        return ""
    points = polydata.GetPoints()
    try:
        from vtkmodules.util import numpy_support
        arr = np.asarray(numpy_support.vtk_to_numpy(points.GetData()))
        contiguous = np.ascontiguousarray(arr)
        digest = hashlib.sha256()
        digest.update(str(contiguous.shape).encode("ascii", "replace"))
        digest.update(str(contiguous.dtype).encode("ascii", "replace"))
        digest.update(memoryview(contiguous).cast("B"))
        return digest.hexdigest()
    except Exception:
        digest = hashlib.sha256()
        for index in range(int(points.GetNumberOfPoints())):
            digest.update(np.asarray(points.GetPoint(index), dtype=np.float64).tobytes())
        return digest.hexdigest()


def _apply_points_affine(points_xyz, matrix4):
    """Apply one homogeneous 4x4 affine to an N x 3 point array."""
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    matrix = _matrix4(matrix4)
    if points.size == 0:
        return points.copy()
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _sitk_image_native(array_zyx, geometry):
    sitk = _sitk_module()
    arr = np.ascontiguousarray(array_zyx)
    grid = _validated_local_grid(geometry, "native registration image")
    if tuple(arr.shape[-3:]) != grid.dimensions[::-1]:
        raise ValueError(
            "Native registration data shape does not match its working grid."
        )
    image = sitk.GetImageFromArray(arr)
    image.SetOrigin(grid.origin)
    image.SetSpacing(grid.spacing)
    image.SetDirection(tuple(np.asarray(grid.direction, dtype=float).reshape(-1)))
    return image


def _sitk_reference_image(geometry, pixel_id):
    sitk = _sitk_module()
    grid = _validated_local_grid(
        geometry, "reference registration image"
    )
    image = sitk.Image(list(grid.dimensions), int(pixel_id))
    image.SetOrigin(grid.origin)
    image.SetSpacing(grid.spacing)
    image.SetDirection(tuple(np.asarray(grid.direction, dtype=float).reshape(-1)))
    return image
# -----------------------------------------------------------------------------
# Registration worker - only NumPy/plain data cross the GUI-thread boundary.
# -----------------------------------------------------------------------------

class RegistrationWorker(QtCore.QThread):
    progress = QtCore.Signal(int, str)
    diagnostic = QtCore.Signal(str, str, str)
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, tasks, settings, cmtk_backend=None, parent=None):
        super().__init__(parent)
        self.tasks = list(tasks or [])
        self.settings = RegistrationSettings.from_dict(settings)
        self.cmtk_backend = cmtk_backend
        self._cancel = False
        self._active_method = None
        self.logs: list[RegistrationLogEntry] = []

    def _diag(self, message, level="INFO", details=""):
        """Emit detailed worker diagnostics without touching Qt widgets directly."""
        entry = RegistrationLogEntry(
            level=str(level or "INFO"),
            message=str(message or ""),
            details=str(details or ""),
            source="registration-worker",
        )
        self.logs.append(entry)
        try:
            self.diagnostic.emit(entry.level, entry.message, entry.details)
        except Exception:
            pass

    def request_cancel(self):
        # Do not call into a SimpleITK registration object from the GUI thread.
        # The iteration callback runs in the worker thread and calls
        # StopRegistration() there, avoiding a cross-thread C++ method call.
        self._cancel = True
        try:
            self.requestInterruption()
        except Exception:
            pass

    def _check_cancel(self):
        if self._cancel or self.isInterruptionRequested():
            raise InterruptedError("Registration cancelled")

    def run(self):
        sitk = None
        previous_global_threads = None
        results = []
        try:
            sitk = _sitk_module()
            threads = max(1, int(self.settings.get("threads", max(1, (os.cpu_count() or 2) - 1))))
            try:
                previous_global_threads = int(sitk.ProcessObject.GetGlobalDefaultNumberOfThreads())
                sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
            except Exception:
                previous_global_threads = None
            self._diag(
                "Registration worker started",
                details=(
                    f"tasks={len(self.tasks)}; preset={self.settings.get('preset', 'custom')}; requested_threads={threads}; "
                    f"model={self.settings.get('global_model')}; linear_engine={_linear_engine(self.settings)}; deformable={bool(self.settings.get('deformable', False))}; "
                    f"global_metric={self.settings.get('metric')}; cmtk_metric={self.settings.get('cmtk_metric', 'nmi')}; "
                    f"sampling={100.0 * float(self.settings.get('sampling', 0.0)):.3g}%; global_spacing={self.settings.get('working_spacing')}; global_safety_max={self.settings.get('max_grid_dim')}; "
                    f"CMTK_input_spacing={self.settings.get('cmtk_working_spacing', 0.75)}; CMTK_input_safety_max={self.settings.get('cmtk_max_grid_dim', 768)}; "
                    f"ITK_threads={threads}; CMTK_threads={self.settings.get('cmtk_threads', 1)}; "
                    f"SimpleITK={getattr(sitk, 'Version_VersionString', lambda: 'unknown')()}; "
                    f"nonlinear_engine={'CMTK warp' if bool(self.settings.get('deformable', False)) else 'disabled'}; "
                    f"cmtk_backend={getattr(self.cmtk_backend, 'label', 'not requested') if self.cmtk_backend is not None else 'not requested'}"
                ),
            )
            total = max(1, len(self.tasks))
            for index, task in enumerate(self.tasks):
                self._check_cancel()
                task_log_start = len(self.logs)
                prefix = int(round(100.0 * index / total))
                self.progress.emit(prefix, f"Preparing {task['moving']['display_name']} → {task['fixed']['display_name']}")
                chain = self._register_pair(task, index, total)
                chain.logs = copy.deepcopy(self.logs[task_log_start:])
                results.append(chain)
                # Batch snapshots are large NumPy arrays. Release each completed
                # moving task immediately instead of retaining the whole batch
                # until QThread teardown. The shared fixed snapshot remains alive
                # through the still-pending task dictionaries.
                try:
                    self.tasks[index] = None
                except Exception:
                    pass
            self._emit_finishing("Finalizing registration results…")
            self.progress.emit(100, "Registration complete")
            self.succeeded.emit(results)
        except InterruptedError:
            _cleanup_runtime_cmtk_chains(results)
            self.cancelled.emit()
        except Exception:
            _cleanup_runtime_cmtk_chains(results)
            self.failed.emit(traceback.format_exc())
        finally:
            if sitk is not None and previous_global_threads is not None:
                try:
                    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous_global_threads)
                except Exception:
                    pass

    def _method_progress_callback(self, method, base, span, label, optimizer_fraction=0.90):
        """Report SimpleITK global-stage progress with headroom for validation.

        SimpleITK's optimizer iteration is zero-based. Keep the final portion of
        every linear stage for transform validation and present iterations as
        human-friendly 1..N values.
        """
        if self._cancel or self.isInterruptionRequested():
            try:
                method.StopRegistration()
            except Exception:
                pass
            return
        try:
            iteration = int(method.GetOptimizerIteration())
        except Exception:
            iteration = 0
        maximum = max(1, int(self._current_stage_iterations))
        shown_iteration = max(1, min(maximum, iteration + 1))
        optimizer_fraction = max(0.50, min(0.95, float(optimizer_fraction)))
        frac = min(1.0, max(0.0, shown_iteration / maximum)) * optimizer_fraction
        self.progress.emit(
            int(round(base + span * frac)),
            f"{label}: optimizing — iteration {shown_iteration}/{maximum}",
        )

    def _emit_finishing(self, text):
        """Switch the panel progress bar to an indeterminate finishing state.

        A negative progress value is a private worker/panel convention for work
        such as CMTK execution and validation whose duration is not meaningfully
        estimated from SimpleITK optimizer iterations.
        """
        self.progress.emit(-1, str(text))

    def _run_cmtk_linear_stage(
        self, *, fixed, moving, fixed_grid, moving_grid, fixed_arr, moving_arr,
        fixed_img, moving_img, fixed_mask, cumulative_m2f, settings, base, stage_span
    ):
        """Run one complete CMTK staged global registration and apply common MADI QC."""
        if self.cmtk_backend is None:
            raise RuntimeError(
                "CMTK is selected as the linear registration engine, but no validated CMTK backend was supplied. "
                "Run the CMTK setup/validation path and retry. MADI3D will not silently fall back to ITK."
            )

        cmtk_settings = _cmtk_linear_settings_from_registration(settings)
        dof_sequence = tuple(int(v) for v in cmtk_settings.dof_sequence)
        path_labels = _cmtk_linear_path_labels(dof_sequence)
        label = "CMTK global " + " → ".join(path_labels)
        global_model = str(settings.get("global_model", "affine") or "affine").lower()
        stage_start_overlap = float(_support_overlap_fraction(fixed, moving, cumulative_m2f))
        initial_f2m = np.linalg.inv(cumulative_m2f)
        try:
            similarity_before = _resampled_similarity_diagnostics(
                fixed_img, moving_img, _sitk_affine_from_matrix(initial_f2m), fixed_mask
            )
        except Exception as exc:
            similarity_before = {
                "ncc": float("nan"), "nmi": float("nan"), "valid_voxels": 0,
                "nmi_bins": 64, "error": str(exc),
            }

        runtime_root = Path(tempfile.mkdtemp(prefix="madi3d-cmtk-linear-"))
        reference_path = runtime_root / "reference.nrrd"
        floating_path = runtime_root / "floating.nrrd"
        workspace = runtime_root / "linear"
        result = None
        cmtk_local_initial = cmtk_working_moving_to_reference(
            cumulative_m2f, fixed_grid["origin"], moving_grid["origin"]
        )
        self.progress.emit(int(round(base + stage_span * 0.05)), f"{label}: staging global working images")
        try:
            write_working_nrrd(reference_path, fixed_arr, canonical_cmtk_grid(fixed_grid))
            write_working_nrrd(floating_path, moving_arr, canonical_cmtk_grid(moving_grid))
            self._check_cancel()

            def cmtk_stdout(text):
                chunk = str(text or "").rstrip()
                if chunk:
                    self._diag("CMTK linear stdout", details=chunk)

            def cmtk_stderr(text):
                chunk = str(text or "").rstrip()
                if chunk:
                    self._diag("CMTK linear stderr", level="WARNING", details=chunk)

            self._diag(
                f"{label} prepared",
                details=(
                    f"metric={cmtk_settings.metric}; DOFs={list(dof_sequence)}; "
                    f"exploration={cmtk_settings.exploration:g}; accuracy={cmtk_settings.accuracy:g}; "
                    f"coarsest={cmtk_settings.coarsest:g}; threads={cmtk_settings.threads}; "
                    f"input source=shared global linear working grids; "
                    f"initial moving→reference={np.asarray(cumulative_m2f).round(6).tolist()}"
                ),
            )
            self._emit_finishing(f"{label}: CMTK registration running…")
            result = CMTKRegistrationRunner(self.cmtk_backend).run_linear(
                reference_image=reference_path,
                floating_image=floating_path,
                moving_to_reference=cmtk_local_initial,
                reference_grid=canonical_cmtk_grid(fixed_grid),
                floating_grid=canonical_cmtk_grid(moving_grid),
                workspace=workspace,
                settings=cmtk_settings,
                on_stdout=cmtk_stdout,
                on_stderr=cmtk_stderr,
                cancel_check=lambda: bool(self._cancel or self.isInterruptionRequested()),
                timeout=None,
            )
            self._check_cancel()
        except CMTKProcessError as exc:
            proc_result = getattr(exc, "result", None)
            if self._cancel or self.isInterruptionRequested() or bool(getattr(proc_result, "cancelled", False)):
                raise InterruptedError("Registration cancelled") from exc
            raise RuntimeError(
                f"CMTK global linear registration failed: {exc}. No ITK fallback was attempted."
            ) from exc
        except InterruptedError:
            raise
        except Exception as exc:
            if self._cancel or self.isInterruptionRequested():
                raise InterruptedError("Registration cancelled") from exc
            raise RuntimeError(
                f"CMTK global linear registration failed: {exc}. No ITK fallback was attempted."
            ) from exc
        finally:
            if result is None:
                shutil.rmtree(runtime_root, ignore_errors=True)

        try:
            candidate_m2f = _cmtk_working_to_world_moving_to_reference(
                result.moving_to_reference, fixed_grid["origin"], moving_grid["origin"]
            )
            artifact_error = _linear_transform_artifact_error(candidate_m2f)
            _qc_passed, qc_failure, sanity = _linear_transform_sanity(
                candidate_m2f, global_model
            )
            final_overlap = float(_support_overlap_fraction(fixed, moving, candidate_m2f))
            if final_overlap < _linear_qc_thresholds()["minimum_support_overlap_fraction"]:
                qc_failure = "optimized transform leaves essentially no fixed/moving support overlap"

            try:
                candidate_f2m = np.linalg.inv(candidate_m2f)
                similarity_after = _resampled_similarity_diagnostics(
                    fixed_img, moving_img, _sitk_affine_from_matrix(candidate_f2m), fixed_mask
                )
            except Exception as exc:
                similarity_after = {
                    "ncc": float("nan"), "nmi": float("nan"), "valid_voxels": 0,
                    "nmi_bins": 64, "error": str(exc),
                }
            quality_warnings = _linear_quality_warnings(
                similarity_before, similarity_after, cmtk_settings.metric,
                overlap_before=stage_start_overlap, overlap_after=final_overlap,
            ) if not artifact_error else []
            serialization_status = str(
                (result.affine_serialization_qc or {}).get("status") or "not-evaluated"
            )
            if serialization_status in {"warning", "failed"}:
                quality_warnings.append(
                    "CMTK affine round-trip serialization QC " + serialization_status
                )
            qc_failures = [qc_failure] if qc_failure else []
            if serialization_status == "failed":
                qc_failures.append("CMTK affine round-trip serialization QC failed")
            stage_qc_status = (
                "not-evaluated" if artifact_error else
                "failed" if qc_failure or serialization_status == "failed" else
                "warning" if quality_warnings else
                "passed"
            )

            similarity_delta = {}
            for diagnostic_name in ("nmi", "ncc"):
                before_value = float(similarity_before.get(diagnostic_name, float("nan")))
                after_value = float(similarity_after.get(diagnostic_name, float("nan")))
                similarity_delta[diagnostic_name] = (
                    float(after_value - before_value)
                    if math.isfinite(before_value) and math.isfinite(after_value) else None
                )

            result_matrix = candidate_m2f if not artifact_error else cumulative_m2f
            incremental = (
                result_matrix @ np.linalg.inv(cumulative_m2f)
                if not artifact_error else np.eye(4, dtype=float)
            )
            ncc = float(similarity_after.get("ncc", float("nan")))
            self._diag(
                f"{label} execution {'succeeded' if not artifact_error else 'failed'}; QC {stage_qc_status}",
                level="WARNING" if (artifact_error or stage_qc_status != "passed") else "INFO",
                details=(
                    f"engine=CMTK; metric={cmtk_settings.metric}; DOFs={list(dof_sequence)}; "
                    f"NMI={similarity_before.get('nmi')} -> {similarity_after.get('nmi')} "
                    f"(delta={similarity_delta.get('nmi')}); "
                    f"NCC={similarity_before.get('ncc')} -> {similarity_after.get('ncc')} "
                    f"(delta={similarity_delta.get('ncc')}); "
                    f"overlap={stage_start_overlap:.6g} -> {final_overlap:.6g}; "
                    f"det={sanity.get('determinant')}; {_linear_qc_text(sanity, precision=5)}; "
                    f"condition={sanity.get('condition_number')}; warnings={quality_warnings or 'none'}; "
                    f"artifact error={artifact_error or 'none'}; QC failures={qc_failures or 'none'}"
                ),
            )
            stage = TransformStageResult(
                name=label,
                kind=global_model,
                cumulative_moving_to_fixed=_matrix_to_json(result_matrix),
                incremental_moving_to_fixed=_matrix_to_json(incremental),
                metric_value=None,
                ncc=None if not math.isfinite(ncc) else ncc,
                iterations=0,
                stop_condition=(
                    "CMTK registration completed; scientific QC passed"
                    if stage_qc_status == "passed" else
                    "CMTK registration completed; scientific QC requires review"
                    if not artifact_error else
                    "CMTK registration produced an invalid transform artifact"
                ),
                execution_status="failed" if artifact_error else "succeeded",
                qc_status=stage_qc_status,
                user_decision="unapplied",
                details={
                    "engine": "cmtk",
                    "linear_engine": "cmtk",
                    "dof_sequence": list(dof_sequence),
                    "configured_metric": cmtk_settings.metric,
                    "configured_exploration": float(cmtk_settings.exploration),
                    "configured_accuracy": float(cmtk_settings.accuracy),
                    "configured_coarsest": float(cmtk_settings.coarsest),
                    "configured_threads": int(cmtk_settings.threads),
                    "cmtk_version": str(result.cmtk_version or ""),
                    "command": list(result.command),
                    "affine_serialization_qc": copy.deepcopy(
                        result.affine_serialization_qc
                    ),
                    "qc_thresholds": {
                        "linear": _linear_qc_thresholds(),
                        "affine_serialization": copy.deepcopy(
                            (result.affine_serialization_qc or {}).get("thresholds") or {}
                        ),
                    },
                    "input_image_source": "shared_global_linear_working_grid",
                    "reference_working_grid": _working_grid_payload(fixed_grid),
                    "floating_working_grid": _working_grid_payload(moving_grid),
                    "initial_moving_to_reference_matrix": _matrix_to_json(cumulative_m2f),
                    "cmtk_initial_working_moving_to_reference_matrix": _matrix_to_json(cmtk_local_initial),
                    "cmtk_optimized_working_moving_to_reference_matrix": _matrix_to_json(result.moving_to_reference),
                    "optimized_moving_to_reference_matrix": _matrix_to_json(candidate_m2f),
                    "deterministic_similarity_before": similarity_before,
                    "deterministic_similarity_after": similarity_after,
                    "deterministic_similarity_delta": similarity_delta,
                    "linear_sanity": sanity,
                    "quality_warnings": list(quality_warnings),
                    "artifact_error": artifact_error,
                    "qc_failures": list(qc_failures),
                    "support_overlap_fraction_at_stage_start": stage_start_overlap,
                    "support_overlap_fraction": final_overlap,
                    "support_overlap_delta": float(final_overlap - stage_start_overlap),
                },
            )
            return stage, np.asarray(result_matrix, dtype=float).copy()
        finally:
            shutil.rmtree(runtime_root, ignore_errors=True)

    def _register_pair(self, task, task_index, task_total):
        settings = self.settings
        original_fixed = task["fixed"]
        original_moving = task["moving"]
        preparation = _prepare_registration_pair(original_fixed, original_moving)
        fixed = preparation.fixed
        moving = preparation.moving
        working_space = copy.deepcopy(preparation.provenance)
        working_space["reformat_target"] = _prepare_reformat_target(
            task.get("reformat_target"), original_fixed, working_space
        )
        for warning in working_space.get("warnings") or ():
            self._diag("Registration working-space warning", level="WARNING", details=warning)
        for assumption in working_space.get("assumptions") or ():
            self._diag("Registration working-space assumption", level="WARNING", details=assumption)
        sitk = _sitk_module()

        # Keep the fixed and moving working images compact. Their physical origins
        # remain in MADI world coordinates, so ITK can register them without one
        # enormous mostly-empty union canvas. Foreground support defines the ROI;
        # working spacing defines numerical resolution, while max_grid_dim is
        # retained only as a memory/runtime safety ceiling.
        target_working_spacing = max(1e-6, float(settings.get("working_spacing", 1.5)))
        max_working_dim = max(32, int(settings.get("max_grid_dim", 512)))
        use_foreground = bool(settings.get("ignore_background", True))
        if use_foreground:
            fixed_support, fixed_support_details = _foreground_support_bounds(
                fixed, target_working_spacing
            )
            moving_support, moving_support_details = _foreground_support_bounds(
                moving, target_working_spacing
            )
        else:
            fixed_support = _support_bounds(fixed["world_affine"], fixed["dims"])
            moving_support = _support_bounds(moving["world_affine"], moving["dims"])
            fixed_support_details = {"valid": False, "foreground_disabled": True}
            moving_support_details = {"valid": False, "foreground_disabled": True}

        if use_foreground:
            fixed["registration_support_bounds_world"] = [
                np.asarray(fixed_support[0], dtype=float).tolist(),
                np.asarray(fixed_support[1], dtype=float).tolist(),
            ]
            moving["registration_support_bounds_world"] = [
                np.asarray(moving_support[0], dtype=float).tolist(),
                np.asarray(moving_support[1], dtype=float).tolist(),
            ]
        else:
            fixed.pop("registration_support_bounds_world", None)
            moving.pop("registration_support_bounds_world", None)
        fixed_grid = _working_grid(
            fixed, target_working_spacing, max_working_dim, support_bounds=fixed_support
        )
        moving_grid = _working_grid(
            moving, target_working_spacing, max_working_dim, support_bounds=moving_support
        )
        fixed_arr = _native_working_array(fixed, fixed_grid, normalize=True)
        moving_arr = _native_working_array(moving, moving_grid, normalize=True)
        fixed_img = _sitk_image(fixed_arr, fixed_grid)
        moving_img = _sitk_image(moving_arr, moving_grid)
        fixed_mask = _registration_foreground_mask(fixed_img, fixed_arr, use_foreground)
        moving_mask = _registration_foreground_mask(moving_img, moving_arr, use_foreground)
        self._diag(
            f"Prepared working images: {moving.get('display_name')} → {fixed.get('display_name')}",
            details=(
                f"target working spacing={target_working_spacing:g}; safety max axis={max_working_dim}; "
                f"fixed dims XYZ={fixed_grid['dims_xyz']}, spacing={tuple(round(float(v), 6) for v in fixed_grid['spacing'])}, safety scale={fixed_grid['safety_scale']:.6g}; "
                f"moving dims XYZ={moving_grid['dims_xyz']}, spacing={tuple(round(float(v), 6) for v in moving_grid['spacing'])}, safety scale={moving_grid['safety_scale']:.6g}; "
                f"foreground support={use_foreground}; fixed detector={fixed_support_details}; moving detector={moving_support_details}; "
                f"metric masks fixed={fixed_mask is not None}, moving={moving_mask is not None}"
            ),
        )

        requested_initial = str(settings.get("initial_alignment", "current") or "current")
        landmark_details = {}
        if _is_landmark_initial_alignment(requested_initial):
            _validate_landmark_9dof_pipeline(settings)
            dataset_id = str(moving.get("dataset_id") or "")
            records = list((settings.get("landmarks_by_dataset") or {}).get(dataset_id) or [])
            minimum_pairs = 4 if requested_initial == "landmarks_9dof" else 3
            if len(records) < minimum_pairs:
                raise RuntimeError(
                    f"Landmark initialization for {moving.get('group_name') or moving.get('display_name') or dataset_id} "
                    f"has only {len(records)} complete pair(s); at least {minimum_pairs} are required."
                )
            target_mapping = _matrix4(working_space["target_to_operation"])
            source_mapping = _matrix4(working_space["source_to_operation"])
            reference_landmarks = _apply_points_affine(
                [record.get("reference_world") for record in records], target_mapping
            )
            moving_landmarks = _apply_points_affine(
                [record.get("moving_world") for record in records], source_mapping
            )
            landmark_weights = [record.get("weight", 1.0) for record in records]
            if requested_initial == "landmarks_9dof":
                initial_m2f, residuals, fit_details = _landmark_9dof_moving_to_fixed(
                    reference_landmarks, moving_landmarks, landmark_weights
                )
                initial_label = "Landmarks 9 DOF rigid + scale fit"
                landmark_dof = 9
                singular = fit_details.get("moving_spread_singular_values") or []
            else:
                initial_m2f, residuals, singular = _landmark_rigid_moving_to_fixed(
                    reference_landmarks, moving_landmarks, landmark_weights
                )
                initial_label = "Landmarks rigid fit"
                landmark_dof = 6
                fit_details = {}
            landmark_details = {
                "landmark_count": int(len(records)),
                "landmark_dof": int(landmark_dof),
                "landmark_mean_residual": float(np.mean(residuals)),
                "landmark_max_residual": float(np.max(residuals)),
                "landmark_residuals": [
                    {"name": str(record.get("name") or f"L{i+1}"), "residual": float(residuals[i])}
                    for i, record in enumerate(records)
                ],
                "landmark_singular_values": [float(v) for v in np.asarray(singular).reshape(-1)],
                "landmark_scale_xyz": list(fit_details.get("scale_xyz") or []),
                "landmark_fit_details": copy.deepcopy(fit_details),
                "reference_landmarks_operation_space": np.asarray(
                    reference_landmarks, dtype=float
                ).tolist(),
                "moving_landmarks_operation_space": np.asarray(
                    moving_landmarks, dtype=float
                ).tolist(),
            }
        else:
            initial_m2f, initial_label = _initial_moving_to_fixed(fixed, moving, requested_initial)
        initial_overlap = _support_overlap_fraction(fixed, moving, initial_m2f)
        fallback_used = False
        fallback_overlap = initial_overlap

        # Metric optimizers cannot bootstrap from truly disjoint support. If the
        # requested initialization has essentially no numerical overlap, use the
        # deterministic center translation as a safety initializer. This is not
        # applied when the requested/manual pose already overlaps.
        if not _is_landmark_initial_alignment(requested_initial) and bool(settings.get("auto_overlap_recovery", True)) and initial_overlap < 1e-4:
            center_m2f, center_label = _initial_moving_to_fixed(fixed, moving, "centers")
            center_overlap = _support_overlap_fraction(fixed, moving, center_m2f)
            if center_overlap > initial_overlap + 1e-6:
                initial_m2f = center_m2f
                fallback_overlap = center_overlap
                fallback_used = True
                initial_label = f"{initial_label} → center-overlap recovery"

        self._diag(
            f"Initial alignment validated: {initial_label}",
            level="WARNING" if fallback_used else "INFO",
            details=(
                f"requested_overlap={float(initial_overlap):.6g}; final_overlap={float(fallback_overlap):.6g}; "
                f"automatic_center_recovery={fallback_used}; moving→reference matrix={np.asarray(initial_m2f).round(6).tolist()}"
            ),
        )
        stages = []
        previous_m2f = np.eye(4, dtype=float)
        cumulative_m2f = initial_m2f.copy()
        stages.append(TransformStageResult(
            name=initial_label,
            kind="initial",
            cumulative_moving_to_fixed=_matrix_to_json(cumulative_m2f),
            incremental_moving_to_fixed=_matrix_to_json(cumulative_m2f),
            ncc=None,
            execution_status="succeeded",
            qc_status="not-evaluated",
            user_decision="unapplied",
            details={
                "mode": requested_initial,
                "support_overlap_fraction": float(fallback_overlap),
                "requested_support_overlap_fraction": float(initial_overlap),
                "automatic_center_fallback": bool(fallback_used),
                **landmark_details,
            },
        ))
        previous_m2f = cumulative_m2f.copy()

        global_model = str(settings.get("global_model", "affine") or "affine").lower()
        try:
            pipeline = _linear_stage_pipeline(global_model)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        linear_engine = _linear_engine(settings)
        if linear_engine == "cmtk" and self.cmtk_backend is None:
            raise RuntimeError(
                "CMTK is selected as the linear registration engine, but no validated CMTK backend is available. "
                "Use MADI3D's CMTK setup/validation path and retry. No ITK fallback will be used."
            )
        enabled_linear = list(pipeline) if linear_engine == "itk" else [("cmtk_linear", "CMTK global linear", True)]
        deformable_enabled = bool(settings.get("deformable", False))
        stage_count = len(enabled_linear) + (1 if deformable_enabled else 0)
        task_span = 100.0 / max(1, task_total)
        task_base = task_index * task_span
        stage_span = task_span / max(1, stage_count)
        stage_number = 0
        affine_scale_ready = global_model != "affine"
        prior_successful_similarity = None

        shrink = list(settings.get("shrink_factors", [8, 4, 2, 1]))
        retry_enabled = bool(settings.get("linear_retry_enabled", True))
        retry_factor = max(0.05, min(0.9, float(settings.get("linear_retry_factor", 0.25))))
        linear_attempt_factors = (1.0, retry_factor) if retry_enabled else (1.0,)
        sampling = float(settings.get("sampling", 0.2))
        bins = int(settings.get("histogram_bins", 48))
        seed = int(settings.get("random_seed", 121212)) + int(task_index)
        center = np.asarray(fixed_grid["origin"], dtype=float) + 0.5 * (
            (np.asarray(fixed_grid["dims_xyz"], dtype=float) - 1.0)
            * np.asarray(fixed_grid["spacing"], dtype=float)
        )

        if linear_engine == "itk":
            for kind, label, enabled in pipeline:
                if not enabled:
                    continue
                self._check_cancel()
                stage_number += 1
                iterations = int(settings.get("linear_iterations", 120))
                self._current_stage_iterations = iterations
                base = task_base + (stage_number - 1) * stage_span

                if global_model == "affine" and kind == "affine" and not affine_scale_ready:
                    reason = (
                        "Affine 12 DOF requires a successfully executed anisotropic 9-DOF warm-start; "
                        "the prerequisite stage failed, so direct rigid-to-affine optimization was not attempted."
                    )
                    stages.append(TransformStageResult(
                        name=label,
                        kind=kind,
                        cumulative_moving_to_fixed=_matrix_to_json(cumulative_m2f),
                        incremental_moving_to_fixed=_matrix_to_json(np.eye(4)),
                        metric_value=None,
                        ncc=None,
                        iterations=0,
                        stop_condition="Skipped because the required 9-DOF affine warm-start failed",
                        execution_status="failed",
                        qc_status="not-evaluated",
                        user_decision="unapplied",
                        details={
                            "execution_error": reason,
                            "prerequisite_stage": "scaleversor",
                            "configured_metric": str(settings.get("metric", "mattes") or "mattes").lower(),
                            "support_overlap_fraction_at_stage_start": float(
                                _support_overlap_fraction(fixed, moving, cumulative_m2f)
                            ),
                        },
                    ))
                    self._diag(f"{label} skipped", level="WARNING", details=reason)
                    self.progress.emit(int(round(base + stage_span)), f"{label} skipped; 9-DOF prerequisite failed")
                    continue

                f2m_initial = np.linalg.inv(cumulative_m2f)
                stage_start_overlap = float(_support_overlap_fraction(fixed, moving, cumulative_m2f))
                if prior_successful_similarity is not None:
                    similarity_before = copy.deepcopy(prior_successful_similarity)
                else:
                    try:
                        similarity_before = _resampled_similarity_diagnostics(
                            fixed_img, moving_img, _sitk_affine_from_matrix(f2m_initial), fixed_mask
                        )
                    except Exception as exc:
                        similarity_before = {
                            "ncc": float("nan"), "nmi": float("nan"), "valid_voxels": 0,
                            "nmi_bins": 64, "error": str(exc),
                        }

                registration = None
                tx = None
                execute_error = None
                attempt_info = []
                stage_parameter_details = {
                    "internal_affine_warm_start": bool(global_model == "affine" and kind == "scaleversor"),
                }
                # Regular-step GD avoids the line-search extrapolations which can jump
                # entirely outside the moving image and throw Mattes MI's
                # "All samples map outside moving image buffer" exception. A smaller
                # step retry is retained as an additional safety path.
                initial_learning_rate = max(1e-4, float(settings.get("linear_learning_rate", 1.0)))

                for attempt_index, factor in enumerate(linear_attempt_factors):
                    self._check_cancel()
                    tx = _linear_transform_for_stage(kind, f2m_initial, center)
                    registration = sitk.ImageRegistrationMethod()
                    _set_process_threads(
                        registration, max(1, int(settings.get("threads", max(1, (os.cpu_count() or 2) - 1))))
                    )
                    self._active_method = registration
                    attempt_fixed_mask = fixed_mask
                    attempt_moving_mask = moving_mask
                    _configure_metric(
                        registration, settings.get("metric", "mattes"), bins, sampling,
                        seed,
                        fixed_mask=attempt_fixed_mask, moving_mask=attempt_moving_mask,
                        neighborhood_radius=int(settings.get("neighborhood_radius", 4)),
                        sampling_per_level=(
                            _auto_metric_sampling_schedule(fixed_img, shrink, sampling)
                            if bool(settings.get("metric_sampling_auto", True)) else None
                        ),
                        sampling_strategy=str(settings.get("metric_sampling_strategy", "random") or "random"),
                        gradient_memory_strategy=_resolve_gradient_memory_strategy(
                            settings.get("gradient_memory_strategy", "auto"), fixed_img, sampling
                        ),
                    )
                    _configure_multires(
                        registration, shrink,
                        smoothing_mode=str(settings.get("pyramid_smoothing_mode", "auto") or "auto"),
                        smoothing_values=settings.get("pyramid_smoothing_sigmas", ()),
                    )
                    learning_rate = initial_learning_rate * factor
                    min_step = max(1e-6, learning_rate * 1e-3)
                    registration.SetOptimizerAsRegularStepGradientDescent(
                        learningRate=float(learning_rate),
                        minStep=float(min_step),
                        numberOfIterations=iterations,
                        relaxationFactor=0.5,
                        gradientMagnitudeTolerance=1e-5,
                    )
                    registration.SetInitialTransform(tx, inPlace=True)
                    optimizer_scale_estimator = "physical_shift"
                    try:
                        registration.SetOptimizerScalesFromPhysicalShift()
                    except Exception:
                        optimizer_scale_estimator = "index_shift"
                        try:
                            registration.SetOptimizerScalesFromIndexShift()
                        except Exception:
                            optimizer_scale_estimator = "default_parameter_scales"
                    registration.AddCommand(
                        sitk.sitkIterationEvent,
                        lambda r=registration, b=base, s=stage_span, l=label:
                            self._method_progress_callback(r, b, s, l, optimizer_fraction=0.90)
                    )
                    try:
                        registration.Execute(fixed_img, moving_img)
                        execute_error = None
                        if kind == "scaleversor":
                            stage_parameter_details.update({
                                "implementation": "joint_scaleversor_9dof",
                                "optimized_scale_xyz": [float(v) for v in tx.GetScale()],
                                "optimized_versor": [float(v) for v in tx.GetVersor()],
                                "optimized_translation_xyz": [float(v) for v in tx.GetTranslation()],
                                "center_xyz": [float(v) for v in tx.GetCenter()],
                                "parameter_count_optimized_in_stage": len(tx.GetParameters()),
                                "initialized_fixed_to_moving": _matrix_to_json(f2m_initial),
                            })
                        attempt_info.append({
                            "learning_rate": learning_rate,
                            "background_masks": bool(attempt_fixed_mask is not None or attempt_moving_mask is not None),
                            "optimizer_scale_estimator": optimizer_scale_estimator,
                            "status": "complete",
                        })
                        break
                    except Exception as exc:
                        if self._cancel or self.isInterruptionRequested():
                            raise InterruptedError("Registration cancelled")
                        attempt_info.append({
                            "learning_rate": learning_rate,
                            "background_masks": bool(attempt_fixed_mask is not None or attempt_moving_mask is not None),
                            "optimizer_scale_estimator": optimizer_scale_estimator,
                            "status": "failed", "error": str(exc), "error_traceback": traceback.format_exc(),
                        })
                        self._diag(
                            f"{label} optimizer attempt failed",
                            level="ERROR",
                            details=traceback.format_exc(),
                        )
                        if not _overlap_metric_error(exc):
                            raise
                        execute_error = exc
                        if attempt_index + 1 < len(linear_attempt_factors):
                            self.progress.emit(
                                int(round(base)),
                                f"{label}: overlap lost — retrying once with learning rate × {linear_attempt_factors[attempt_index + 1]:g}"
                            )
                        else:
                            self.progress.emit(int(round(base)), f"{label}: overlap lost — rejecting this stage")

                if execute_error is not None or registration is None or tx is None:
                    if global_model == "affine" and kind == "scaleversor":
                        affine_scale_ready = False
                    stages.append(TransformStageResult(
                        name=label,
                        kind=kind,
                        cumulative_moving_to_fixed=_matrix_to_json(previous_m2f),
                        incremental_moving_to_fixed=_matrix_to_json(np.eye(4)),
                        metric_value=None,
                        ncc=None,
                        iterations=0,
                        stop_condition="Execution failed after loss of fixed/moving overlap",
                        execution_status="failed",
                        qc_status="not-evaluated",
                        user_decision="unapplied",
                        details={
                            "optimizer_attempts": attempt_info,
                            "execution_error": str(execute_error or "Registration stage failed"),
                            "configured_metric": str(settings.get("metric", "mattes") or "mattes").lower(),
                            "configured_sampling_fraction": float(sampling),
                            "configured_histogram_bins": int(bins),
                            "deterministic_similarity_before": similarity_before,
                            "support_overlap_fraction_at_stage_start": float(stage_start_overlap),
                            "support_overlap_fraction": float(stage_start_overlap),
                            "support_overlap_delta": 0.0,
                            **stage_parameter_details,
                        },
                    ))
                    cumulative_m2f = previous_m2f.copy()
                    prior_successful_similarity = copy.deepcopy(similarity_before)
                    self.progress.emit(int(round(base + stage_span)), f"{label} failed; retained previous valid stage")
                    continue

                self._check_cancel()
                self.progress.emit(
                    int(round(base + stage_span * 0.92)),
                    f"{label}: optimizer complete — validating transform",
                )
                new_f2m = _sitk_linear_matrix(tx)
                candidate_m2f = None
                sanity = {}
                artifact_error = ""
                qc_failure = ""
                try:
                    candidate_m2f = np.linalg.inv(new_f2m)
                except np.linalg.LinAlgError as exc:
                    artifact_error = f"optimized transform is not invertible: {exc}"
                else:
                    # Validate and report geometry in MADI's authoritative
                    # moving->reference direction so the displayed scale/shear values
                    # correspond to how the moving specimen is actually transformed.
                    artifact_error = _linear_transform_artifact_error(candidate_m2f)
                    _qc_passed, qc_failure, sanity = _linear_transform_sanity(
                        candidate_m2f, kind
                    )

                # An optimizer may terminate without an exception while leaving only a
                # negligible intersection. Keep the finite transform inspectable,
                # but make the scientific QC failure explicit.
                final_overlap = 0.0
                if candidate_m2f is not None:
                    final_overlap = float(_support_overlap_fraction(fixed, moving, candidate_m2f))
                    if final_overlap < _linear_qc_thresholds()["minimum_support_overlap_fraction"]:
                        qc_failure = "optimized transform leaves essentially no fixed/moving support overlap"

                try:
                    similarity_after = _resampled_similarity_diagnostics(
                        fixed_img, moving_img, tx, fixed_mask
                    )
                except Exception as exc:
                    similarity_after = {
                        "ncc": float("nan"), "nmi": float("nan"), "valid_voxels": 0,
                        "nmi_bins": 64, "error": str(exc),
                    }
                ncc = float(similarity_after.get("ncc", float("nan")))
                quality_warnings = _linear_quality_warnings(
                    similarity_before, similarity_after, settings.get("metric", "mattes"),
                    overlap_before=stage_start_overlap, overlap_after=final_overlap,
                ) if not artifact_error else []
                qc_failures = [qc_failure] if qc_failure else []
                stage_qc_status = (
                    "not-evaluated" if artifact_error else
                    "failed" if qc_failure else
                    "warning" if quality_warnings else
                    "passed"
                )

                if not artifact_error and candidate_m2f is not None:
                    cumulative_m2f = candidate_m2f
                    incremental = cumulative_m2f @ np.linalg.inv(previous_m2f)
                else:
                    cumulative_m2f = previous_m2f.copy()
                    incremental = np.eye(4, dtype=float)

                if global_model == "affine" and kind == "scaleversor":
                    affine_scale_ready = not bool(artifact_error)

                similarity_delta = {}
                for diagnostic_name in ("nmi", "ncc"):
                    before_value = float(similarity_before.get(diagnostic_name, float("nan")))
                    after_value = float(similarity_after.get(diagnostic_name, float("nan")))
                    similarity_delta[diagnostic_name] = (
                        float(after_value - before_value)
                        if math.isfinite(before_value) and math.isfinite(after_value) else None
                    )

                try:
                    metric_value = float(registration.GetMetricValue())
                except Exception:
                    metric_value = None
                try:
                    optimizer_iterations = int(registration.GetOptimizerIteration())
                except Exception:
                    optimizer_iterations = 0
                try:
                    stop_condition = str(registration.GetOptimizerStopConditionDescription())
                except Exception:
                    stop_condition = ""
                configured_metric = str(settings.get("metric", "mattes") or "mattes").lower()
                self._diag(
                    f"{label} execution {'succeeded' if not artifact_error else 'failed'}; QC {stage_qc_status}",
                    level="WARNING" if (artifact_error or stage_qc_status != "passed") else "INFO",
                    details=(
                        f"configured_metric={configured_metric}; optimizer_metric={metric_value}; "
                        f"NMI={similarity_before.get('nmi')} -> {similarity_after.get('nmi')} "
                        f"(delta={similarity_delta.get('nmi')}); "
                        f"NCC={similarity_before.get('ncc')} -> {similarity_after.get('ncc')} "
                        f"(delta={similarity_delta.get('ncc')}); "
                        f"iterations={optimizer_iterations}; overlap={stage_start_overlap:.6g} -> {float(final_overlap):.6g}; "
                        f"det={sanity.get('determinant')}; principal stretches={sanity.get('principal_stretches')}; "
                        f"{_linear_qc_text(sanity, precision=5)}; "
                        f"condition={sanity.get('condition_number')}; warnings={quality_warnings or 'none'}; "
                        f"artifact error={artifact_error or 'none'}; QC failures={qc_failures or 'none'}; stop={stop_condition or 'n/a'}"
                    ),
                )

                stages.append(TransformStageResult(
                    name=label,
                    kind=kind,
                    cumulative_moving_to_fixed=_matrix_to_json(cumulative_m2f),
                    incremental_moving_to_fixed=_matrix_to_json(incremental),
                    metric_value=metric_value,
                    ncc=None if not math.isfinite(ncc) else float(ncc),
                    iterations=optimizer_iterations,
                    stop_condition=stop_condition,
                    execution_status="failed" if artifact_error else "succeeded",
                    qc_status=stage_qc_status,
                    user_decision="unapplied",
                    details={
                        "fixed_to_moving_parameters": [float(v) for v in tx.GetParameters()],
                        "fixed_parameters": [float(v) for v in tx.GetFixedParameters()],
                        "configured_metric": configured_metric,
                        "configured_sampling_fraction": float(sampling),
                        "configured_histogram_bins": int(bins),
                        "deterministic_similarity_before": similarity_before,
                        "deterministic_similarity_after": similarity_after,
                        "deterministic_similarity_delta": similarity_delta,
                        "linear_sanity": sanity,
                        "quality_warnings": list(quality_warnings),
                        "artifact_error": artifact_error,
                        "qc_failures": list(qc_failures),
                        "qc_thresholds": {"linear": _linear_qc_thresholds()},
                        "optimizer_attempts": attempt_info,
                        "support_overlap_fraction_at_stage_start": float(stage_start_overlap),
                        "support_overlap_fraction": float(final_overlap),
                        "support_overlap_delta": float(final_overlap - stage_start_overlap),
                        **stage_parameter_details,
                    },
                ))
                if not artifact_error:
                    previous_m2f = cumulative_m2f.copy()
                    prior_successful_similarity = copy.deepcopy(similarity_after)
                    self.progress.emit(
                        int(round(base + stage_span)),
                        f"{label} complete; QC {stage_qc_status}",
                    )
                else:
                    prior_successful_similarity = copy.deepcopy(similarity_before)
                    self.progress.emit(int(round(base + stage_span)), f"{label} failed; retained previous valid stage")

        else:
            self._check_cancel()
            stage_number += 1
            base = task_base + (stage_number - 1) * stage_span
            cmtk_stage, cumulative_m2f = self._run_cmtk_linear_stage(
                fixed=fixed, moving=moving,
                fixed_grid=fixed_grid, moving_grid=moving_grid,
                fixed_arr=fixed_arr, moving_arr=moving_arr,
                fixed_img=fixed_img, moving_img=moving_img, fixed_mask=fixed_mask,
                cumulative_m2f=cumulative_m2f, settings=settings,
                base=base, stage_span=stage_span,
            )
            stages.append(cmtk_stage)
            if execution_succeeded(cmtk_stage.execution_status):
                previous_m2f = cumulative_m2f.copy()
                self.progress.emit(int(round(base + stage_span)), f"{cmtk_stage.name} complete")
            else:
                cumulative_m2f = previous_m2f.copy()
                self.progress.emit(
                    int(round(base + stage_span)),
                    f"{cmtk_stage.name} failed; retained previous valid stage",
                )

        deformation_model = {}
        if deformable_enabled:
            self._check_cancel()
            if self.cmtk_backend is None:
                raise RuntimeError(
                    "CMTK nonlinear registration was requested, but no validated CMTK backend was supplied to the worker."
                )
            stage_number += 1
            label = "CMTK nonlinear warp"
            base = task_base + (stage_number - 1) * stage_span
            cmtk_settings, cmtk_mapping = _cmtk_warp_settings_from_registration(settings)
            runtime_root = Path(tempfile.mkdtemp(prefix="madi3d-cmtk-registration-"))
            reference_path = runtime_root / "reference.nrrd"
            floating_path = runtime_root / "floating.nrrd"
            workspace = runtime_root / "warp"
            self.progress.emit(int(round(base + stage_span * 0.05)), f"{label}: building independent nonlinear input grids")

            # Linear optimization is complete. Release the coarse ITK working
            # images before allocating the finer nonlinear arrays so the two image
            # domains do not unnecessarily coexist in memory. Native snapshots and
            # their world geometry remain available as the CMTK staging source.
            self._active_method = None
            registration = None
            tx = None
            fixed_img = None
            moving_img = None
            fixed_mask = None
            moving_mask = None
            fixed_arr = None
            moving_arr = None

            # The nonlinear image domain is deliberately independent from the
            # coarse global-search domain. Re-sample each CMTK input exactly once
            # from the original captured source using the same anatomical support
            # box but its own finer working spacing and safety ceiling. The
            # latest successfully executed linear matrix remains an explicit transform initializer; it
            # is not baked into either staged image.
            fixed_cmtk_world_grid = _cmtk_nonlinear_working_grid(
                fixed, fixed_support, settings
            )
            moving_cmtk_world_grid = _cmtk_nonlinear_working_grid(
                moving, moving_support, settings
            )
            fixed_cmtk_arr = _native_working_array(
                fixed, fixed_cmtk_world_grid, normalize=False
            )
            self._check_cancel()
            moving_cmtk_arr = _native_working_array(
                moving, moving_cmtk_world_grid, normalize=False
            )
            self._check_cancel()
            fixed_cmtk_grid = canonical_cmtk_grid(fixed_cmtk_world_grid)
            moving_cmtk_grid = canonical_cmtk_grid(moving_cmtk_world_grid)
            cmtk_local_m2f = cmtk_working_moving_to_reference(
                cumulative_m2f,
                fixed_cmtk_world_grid["origin"],
                moving_cmtk_world_grid["origin"],
            )

            try:
                # CMTK sees zero-origin lattices because its oriented NRRD loader
                # normalizes RAS pose metadata. The nonlinear working-grid world
                # origins and the complete successful moving→reference matrix are
                # carried explicitly by cmtk_local_m2f.
                write_working_nrrd(reference_path, fixed_cmtk_arr, fixed_cmtk_grid)
                write_working_nrrd(floating_path, moving_cmtk_arr, moving_cmtk_grid)
                self._check_cancel()
                self._diag(
                    f"{label} prepared",
                    details=(
                        f"backend={getattr(self.cmtk_backend, 'label', getattr(self.cmtk_backend, 'kind', 'CMTK'))}; "
                        f"input=native volume direct resample; frame=zero-origin nonlinear working grids; "
                        f"fixed dims XYZ={fixed_cmtk_world_grid['dims_xyz']}, spacing={tuple(round(float(v), 6) for v in fixed_cmtk_world_grid['spacing'])}, safety scale={fixed_cmtk_world_grid['safety_scale']:.6g}; "
                        f"moving dims XYZ={moving_cmtk_world_grid['dims_xyz']}, spacing={tuple(round(float(v), 6) for v in moving_cmtk_world_grid['spacing'])}, safety scale={moving_cmtk_world_grid['safety_scale']:.6g}; "
                        f"fixed_world_origin={tuple(round(float(v), 6) for v in fixed_cmtk_world_grid['origin'])}; "
                        f"moving_world_origin={tuple(round(float(v), 6) for v in moving_cmtk_world_grid['origin'])}; "
                        f"successful linear moving→reference={np.asarray(cumulative_m2f).round(6).tolist()}; "
                        f"metric={cmtk_settings.metric}; exploration={cmtk_settings.exploration:g}; "
                        f"accuracy={cmtk_settings.accuracy:g}; coarsest={cmtk_settings.coarsest:g}; "
                        f"grid={cmtk_mapping['grid_spacing']:g}; refine={cmtk_mapping['refine']}; "
                        f"approx final spacing={cmtk_mapping['approx_final_spacing']:g}; "
                        f"mode={cmtk_settings.mode}; threads={cmtk_mapping['threads']}; "
                        f"energy={cmtk_settings.energy_weight:g}; Jacobian={cmtk_settings.jacobian_weight:g}; "
                        f"inverse-consistency={cmtk_settings.inverse_consistency_weight:g}; "
                        f"omit-original-data={bool(cmtk_settings.omit_original_data)}; "
                        f"match-histograms={bool(cmtk_settings.match_histograms)}"
                    ),
                )
                self._emit_finishing(f"{label}: CMTK warp running…")

                def cmtk_stdout(text):
                    chunk = str(text or "").rstrip()
                    if chunk:
                        self._diag("CMTK stdout", details=chunk)

                def cmtk_stderr(text):
                    chunk = str(text or "").rstrip()
                    if chunk:
                        self._diag("CMTK stderr", level="WARNING", details=chunk)

                runner = CMTKRegistrationRunner(self.cmtk_backend)
                artifacts = runner.run_warp(
                    reference_image=reference_path,
                    floating_image=floating_path,
                    moving_to_reference=cmtk_local_m2f,
                    reference_grid=fixed_cmtk_grid,
                    floating_grid=moving_cmtk_grid,
                    workspace=workspace,
                    settings=cmtk_settings,
                    on_stdout=cmtk_stdout,
                    on_stderr=cmtk_stderr,
                    cancel_check=lambda: bool(self._cancel or self.isInterruptionRequested()),
                    timeout=None,
                )


                # Final nonlinear QC is performed on the same zero-origin CMTK
                # working grids used by the optimizer. Compare the successful
                # affine seed against the completed warp using MADI's common
                # deterministic NMI/NCC diagnostics, then measure the local
                # deformation Jacobian with global affine scale removed.
                qc_linear_path = workspace / "qc-linear.nrrd"
                qc_warp_path = workspace / "qc-warp.nrrd"
                qc_jacobian_path = workspace / "qc-jacobian-local.nrrd"
                try:
                    runner.run_reformat(
                        reference_image=reference_path,
                        floating_image=floating_path,
                        xforms=[artifacts.affine_xform],
                        output_image=qc_linear_path,
                        interpolation="linear",
                        output_type="float",
                        pad_out=0.0,
                        cancel_check=lambda: bool(self._cancel or self.isInterruptionRequested()),
                        timeout=None,
                    )
                    runner.run_reformat(
                        reference_image=reference_path,
                        floating_image=floating_path,
                        xforms=[artifacts.warp_xform],
                        output_image=qc_warp_path,
                        interpolation="linear",
                        output_type="float",
                        pad_out=0.0,
                        cancel_check=lambda: bool(self._cancel or self.isInterruptionRequested()),
                        timeout=None,
                    )
                    linear_reformatted, _linear_header = read_nrrd_zyx(qc_linear_path)
                    warp_reformatted, _warp_header = read_nrrd_zyx(qc_warp_path)
                    linear_similarity = _array_similarity_diagnostics(
                        fixed_cmtk_arr, linear_reformatted
                    )
                    warp_similarity = _array_similarity_diagnostics(
                        fixed_cmtk_arr, warp_reformatted
                    )
                    jacobian_qc = runner.run_jacobian_qc(
                        reference_image=reference_path,
                        xforms=[artifacts.warp_xform],
                        output_image=qc_jacobian_path,
                        correct_global=True,
                        threads=int(cmtk_settings.threads),
                        on_stdout=cmtk_stdout,
                        on_stderr=cmtk_stderr,
                        cancel_check=lambda: bool(self._cancel or self.isInterruptionRequested()),
                        timeout=None,
                    )
                    qc_warnings = _deformation_qc_warnings(
                        linear_similarity,
                        warp_similarity,
                        jacobian_qc,
                        cmtk_settings.metric,
                    )
                    deformation_qc = {
                        "policy": "warn_and_keep",
                        "thresholds": _deformation_qc_thresholds(),
                        "linear_similarity": copy.deepcopy(linear_similarity),
                        "warp_similarity": copy.deepcopy(warp_similarity),
                        "jacobian_local": copy.deepcopy(jacobian_qc),
                        "warnings": list(qc_warnings),
                    }
                    update_artifact_bundle_qc(artifacts.workspace, deformation_qc)
                except CMTKProcessError as exc:
                    result = getattr(exc, "result", None)
                    if self._cancel or self.isInterruptionRequested() or bool(getattr(result, "cancelled", False)):
                        raise InterruptedError("Registration cancelled") from exc
                    raise RuntimeError(f"CMTK nonlinear QC failed: {exc}") from exc
                except InterruptedError:
                    raise
                except Exception as exc:
                    raise RuntimeError(f"CMTK nonlinear QC failed: {exc}") from exc
                finally:
                    output_policy = dict(settings.get("result_output") or {})
                    keep_qc_volumes = bool(
                        output_policy.get("enabled") and output_policy.get("write_qc_volumes")
                    )
                    if not keep_qc_volumes:
                        for qc_path in (qc_linear_path, qc_warp_path, qc_jacobian_path):
                            try:
                                Path(qc_path).unlink(missing_ok=True)
                            except Exception:
                                pass
            except CMTKProcessError as exc:
                shutil.rmtree(runtime_root, ignore_errors=True)
                result = getattr(exc, "result", None)
                if self._cancel or self.isInterruptionRequested() or bool(getattr(result, "cancelled", False)):
                    raise InterruptedError("Registration cancelled") from exc
                raise RuntimeError(f"CMTK nonlinear warp failed: {exc}") from exc
            except InterruptedError:
                shutil.rmtree(runtime_root, ignore_errors=True)
                raise
            except Exception:
                shutil.rmtree(runtime_root, ignore_errors=True)
                raise

            deformation_model = {
                "type": "cmtk_warp",
                "engine": "cmtk",
                "native_direction": "reference_to_floating",
                "madi_direction": "moving_to_reference",
                "coordinate_frame": "working_grid_local_v1",
                "linear_moving_to_reference_matrix": _matrix_to_json(cumulative_m2f),
                "cmtk_linear_moving_to_reference_matrix": _matrix_to_json(cmtk_local_m2f),
                "reference_working_origin_world": [float(v) for v in fixed_cmtk_world_grid["origin"]],
                "floating_working_origin_world": [float(v) for v in moving_cmtk_world_grid["origin"]],
                "reference_working_grid": _working_grid_payload(fixed_cmtk_world_grid),
                "floating_working_grid": _working_grid_payload(moving_cmtk_world_grid),
                "input_image_source": "native_volume_direct_resample",
                "runtime_artifact_root": str(runtime_root),
                "workspace": str(artifacts.workspace),
                "reference_image": str(artifacts.reference_image),
                "floating_image": str(artifacts.floating_image),
                "affine_xform": str(artifacts.affine_xform),
                "warp_xform": str(artifacts.warp_xform),
                "manifest_path": str(artifacts.manifest_path),
                "stdout_log": str(artifacts.stdout_log),
                "stderr_log": str(artifacts.stderr_log),
                "cmtk_version": str(artifacts.cmtk_version or ""),
                "command": list(artifacts.command),
                "affine_serialization_qc": copy.deepcopy(
                    artifacts.affine_serialization_qc
                ),
                "qc_thresholds": {
                    "deformation": _deformation_qc_thresholds(),
                    "affine_serialization": copy.deepcopy(
                        (artifacts.affine_serialization_qc or {}).get("thresholds") or {}
                    ),
                },
                "artifact_scope": "runtime_temporary",
                "persistence_ready": True,
                "reformat_ready": True,
                "effective_settings": {
                    "metric": cmtk_settings.metric,
                    "exploration": float(cmtk_settings.exploration),
                    "accuracy": float(cmtk_settings.accuracy),
                    "coarsest": float(cmtk_settings.coarsest),
                    "grid_spacing": float(cmtk_settings.grid_spacing),
                    "refine": int(cmtk_settings.refine),
                    "mode": str(cmtk_settings.mode),
                    "threads": int(cmtk_settings.threads),
                    "energy_weight": float(cmtk_settings.energy_weight),
                    "jacobian_weight": float(cmtk_settings.jacobian_weight),
                    "inverse_consistency_weight": float(cmtk_settings.inverse_consistency_weight),
                    "omit_original_data": bool(cmtk_settings.omit_original_data),
                    "match_histograms": bool(cmtk_settings.match_histograms),
                    "input_working_spacing": float(settings.get("cmtk_working_spacing", 0.75)),
                    "input_max_grid_dim": int(settings.get("cmtk_max_grid_dim", 768)),
                },
                "settings_mapping": copy.deepcopy(cmtk_mapping),
                "deformation_qc": copy.deepcopy(deformation_qc),
                "runtime_qc_files": ({
                    "linear": str(qc_linear_path),
                    "warp": str(qc_warp_path),
                    "jacobian": str(qc_jacobian_path),
                } if bool(dict(settings.get("result_output") or {}).get("enabled")
                          and dict(settings.get("result_output") or {}).get("write_qc_volumes")) else {}),
            }
            details = {
                "engine": "cmtk",
                "cmtk_version": str(artifacts.cmtk_version or ""),
                "cmtk_settings": copy.deepcopy(deformation_model["effective_settings"]),
                "settings_mapping": copy.deepcopy(cmtk_mapping),
                "artifact_scope": "runtime_temporary",
                "coordinate_frame": "working_grid_local_v1",
                "linear_moving_to_reference_matrix": _matrix_to_json(cumulative_m2f),
                "cmtk_linear_moving_to_reference_matrix": _matrix_to_json(cmtk_local_m2f),
                "reference_working_grid": _working_grid_payload(fixed_cmtk_world_grid),
                "floating_working_grid": _working_grid_payload(moving_cmtk_world_grid),
                "input_image_source": "native_volume_direct_resample",
                "workspace": str(artifacts.workspace),
                "warp_xform": str(artifacts.warp_xform),
                "manifest_path": str(artifacts.manifest_path),
                "stdout_log": str(artifacts.stdout_log),
                "stderr_log": str(artifacts.stderr_log),
                "command": list(artifacts.command),
                "affine_serialization_qc": copy.deepcopy(
                    artifacts.affine_serialization_qc
                ),
                "qc_thresholds": copy.deepcopy(
                    deformation_model.get("qc_thresholds") or {}
                ),
                "persistence_ready": True,
                "reformat_ready": True,
                "deformation_qc": copy.deepcopy(deformation_qc),
                "runtime_qc_files": copy.deepcopy(deformation_model.get("runtime_qc_files") or {}),
            }
            stages.append(TransformStageResult(
                name=label,
                kind="cmtk_warp",
                cumulative_moving_to_fixed=_matrix_to_json(cumulative_m2f),
                incremental_moving_to_fixed=_matrix_to_json(np.eye(4)),
                metric_value=None,
                ncc=float(deformation_qc["warp_similarity"]["ncc"]),
                iterations=0,
                stop_condition=(
                    "CMTK warp completed; scientific QC requires review"
                    if deformation_qc.get("warnings") else
                    "CMTK warp completed and scientific QC passed"
                ),
                execution_status="succeeded",
                qc_status=_deformation_result_qc_status(deformation_qc),
                user_decision="unapplied",
                details=details,
            ))
            self._diag(
                f"{label} execution succeeded",
                details=(
                    f"CMTK={artifacts.cmtk_version or 'unknown'}; warp={artifacts.warp_xform}; "
                    f"NMI {deformation_qc['linear_similarity']['nmi']:.6g}→{deformation_qc['warp_similarity']['nmi']:.6g}; "
                    f"NCC {deformation_qc['linear_similarity']['ncc']:.6g}→{deformation_qc['warp_similarity']['ncc']:.6g}; "
                    f"Jacobian median={deformation_qc['jacobian_local']['median']:.6g}; "
                    f"folding={100.0 * deformation_qc['jacobian_local']['nonpositive_fraction']:.6g}%; "
                    f"warnings={deformation_qc.get('warnings') or 'none'}; "
                    f"manifest={artifacts.manifest_path}; command={list(artifacts.command)}"
                ),
            )
            if deformation_qc.get("warnings"):
                self._diag(
                    f"{label} QC warning",
                    level="WARNING",
                    details=" | ".join(map(str, deformation_qc.get("warnings") or [])),
                )
            self.progress.emit(int(round(base + stage_span)), f"{label} complete")

        fixed_payload = _working_grid_payload(fixed_grid)
        moving_payload = _working_grid_payload(moving_grid)
        # Keep the fixed-grid keys at the top level for backward readers while
        # recording the two actual field domains explicitly.
        registration_grid = copy.deepcopy(fixed_payload)
        registration_grid["fixed"] = fixed_payload
        registration_grid["moving"] = moving_payload

        public_stages = _map_stages_to_source_target_space(stages, working_space)
        if deformation_model:
            deformation_model["source_to_operation"] = copy.deepcopy(
                working_space["source_to_operation"]
            )
            deformation_model["target_to_operation"] = copy.deepcopy(
                working_space["target_to_operation"]
            )
            deformation_model["source_to_target_linear_matrix"] = copy.deepcopy(
                public_stages[-1].cumulative_moving_to_fixed
            )
        chain = RegistrationTransformChain(
            registration_id=str(uuid.uuid4()),
            source_dataset_id=str(moving.get("dataset_id") or ""),
            target_dataset_id=str(fixed.get("dataset_id") or ""),
            source_space_uid=str(working_space.get("source_space_id") or ""),
            target_space_uid=str(working_space.get("target_space_id") or ""),
            source_name=str(moving["display_name"]),
            target_name=str(fixed["display_name"]),
            source_actor_matrix=copy.deepcopy(working_space["source_initial_pose"]),
            target_actor_matrix=copy.deepcopy(working_space["target_initial_pose"]),
            registration_grid=registration_grid,
            working_space=copy.deepcopy(working_space),
            source_geometry=_local_geometry_payload(original_moving),
            target_geometry=_local_geometry_payload(original_fixed),
            source_descriptor=copy.deepcopy(moving.get("descriptor") or {}),
            target_descriptor=copy.deepcopy(fixed.get("descriptor") or {}),
            stages=public_stages,
            settings=copy.deepcopy(settings),
            attachments=copy.deepcopy(task.get("attachments") or []),
            reformat_volumes=copy.deepcopy(task.get("reformat_volumes") or []),
            source_volumes=copy.deepcopy(task.get("source_volumes") or []),
            deformation_model=copy.deepcopy(deformation_model),
            algorithm_version=REGISTRATION_ALGORITHM_VERSION,
        )
        return chain


# -----------------------------------------------------------------------------
# Reformat worker - resampling only; VTK/project objects stay on the GUI thread.
# -----------------------------------------------------------------------------

class ReformatWorker(QtCore.QThread):
    progress = QtCore.Signal(int, str)
    diagnostic = QtCore.Signal(str, str, str)
    outputReady = QtCore.Signal(object)
    succeeded = QtCore.Signal(int)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(
        self, tasks, settings, cmtk_backend=None, volume_output_writer=None,
        ffmpeg_executable=None, parent=None,
    ):
        super().__init__(parent)
        self.tasks = list(tasks or [])
        self.settings = RegistrationSettings.from_dict(settings)
        self.cmtk_backend = cmtk_backend
        self.volume_output_writer = volume_output_writer
        self.ffmpeg_executable = os.fspath(ffmpeg_executable) if ffmpeg_executable else None
        self._cancel = False

    def request_cancel(self):
        self._cancel = True
        try:
            self.requestInterruption()
        except Exception:
            pass

    def _check_cancel(self):
        if self._cancel or self.isInterruptionRequested():
            raise InterruptedError("Reformat cancelled")

    def _diag(self, message, level="INFO", details=""):
        try:
            self.diagnostic.emit(str(level or "INFO"), str(message or ""), str(details or ""))
        except Exception:
            pass

    def _itk_reformat_frame(self, frame, task, interpolation, dtype_mode, progress_callback):
        sitk = _sitk_module()
        interpolator = {
            "nearest": sitk.sitkNearestNeighbor,
            "cubic": sitk.sitkBSpline,
            "linear": sitk.sitkLinear,
        }.get(interpolation, sitk.sitkLinear)
        moving_img = _sitk_image_native(frame, task["source_geometry"])
        output_pixel_id = sitk.sitkFloat32 if dtype_mode == "float32" else moving_img.GetPixelID()
        reference_img = _sitk_reference_image(task["reference_geometry"], output_pixel_id)
        local_transform = _reformat_local_transform(
            task["transform"], task["reference_actor_matrix"], task["source_actor_matrix"]
        )
        resampler = sitk.ResampleImageFilter()
        _set_process_threads(resampler, min(8, max(1, int(os.cpu_count() or 1))))
        resampler.SetReferenceImage(reference_img)
        resampler.SetTransform(local_transform)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetOutputPixelType(output_pixel_id)

        def resample_progress():
            if self._cancel or self.isInterruptionRequested():
                try:
                    resampler.Abort()
                except Exception:
                    pass
                return
            try:
                progress_callback(max(0.0, min(1.0, float(resampler.GetProgress()))))
            except Exception:
                pass

        resampler.AddCommand(sitk.sitkProgressEvent, resample_progress)
        out = resampler.Execute(moving_img)
        self._check_cancel()
        return np.asarray(sitk.GetArrayFromImage(out))

    def _cmtk_reformat_frame(
        self, frame, task, interpolation, dtype_mode, runner, frame_workspace
    ):
        if runner is None or self.cmtk_backend is None:
            raise RuntimeError("CMTK Reformat requires a validated CMTK backend.")
        frame_workspace.mkdir(parents=True, exist_ok=False)
        floating = frame_workspace / "floating.nrrd"
        reference = frame_workspace / "reference-grid.nrrd"
        reference_bridge = frame_workspace / "reference-to-working.xform"
        source_bridge = frame_workspace / "working-to-source.xform"
        output = frame_workspace / "output.nrrd"
        write_volume_nrrd(floating, frame, task["cmtk_source_grid"])
        write_reference_grid_nrrd(reference, task["cmtk_reference_grid"])
        write_cmtk_matrix_xform(
            self.cmtk_backend,
            task["cmtk_reference_to_working_matrix"],
            reference_bridge,
            reference_grid=task["cmtk_reference_grid"],
            floating_grid=task["cmtk_fixed_working_grid"],
            on_qc=lambda qc: (
                self._diag(
                    "CMTK reference bridge serialization QC",
                    level="WARNING",
                    details=qc.diagnostic(),
                ) if qc.status in {"warning", "failed"} else None
            ),
        )
        write_cmtk_matrix_xform(
            self.cmtk_backend,
            task["cmtk_working_to_source_matrix"],
            source_bridge,
            reference_grid=task["cmtk_moving_working_grid"],
            floating_grid=task["cmtk_source_grid"],
            on_qc=lambda qc: (
                self._diag(
                    "CMTK source bridge serialization QC",
                    level="WARNING",
                    details=qc.diagnostic(),
                ) if qc.status in {"warning", "failed"} else None
            ),
        )
        output_type = "float" if dtype_mode == "float32" else cmtk_output_type_for_dtype(frame.dtype)

        def stdout(text):
            if str(text or "").strip():
                self._diag("CMTK reformatx", details=str(text).rstrip())

        def stderr(text):
            if str(text or "").strip():
                self._diag("CMTK reformatx diagnostic", level="WARNING", details=str(text).rstrip())

        try:
            runner.run_reformat(
                reference_image=reference,
                floating_image=floating,
                xforms=[
                    reference_bridge,
                    task["cmtk_warp_xform"],
                    source_bridge,
                ],
                output_image=output,
                interpolation=interpolation,
                output_type=output_type,
                pad_out=0.0,
                on_stdout=stdout,
                on_stderr=stderr,
                cancel_check=lambda: self._cancel or self.isInterruptionRequested(),
            )
        except CMTKProcessError as exc:
            result = getattr(exc, "result", None)
            if self._cancel or self.isInterruptionRequested() or bool(getattr(result, "cancelled", False)):
                raise InterruptedError("Reformat cancelled") from exc
            raise
        self._check_cancel()
        array, _header = read_nrrd_zyx(output)
        expected = tuple(
            int(v) for v in reversed(tuple(task["cmtk_reference_grid"]["dims_xyz"]))
        )
        if tuple(array.shape) != expected:
            raise RuntimeError(
                "CMTK reformatx output geometry does not match the captured Reference grid: "
                f"expected ZYX={expected}, got {tuple(array.shape)}."
            )
        return array

    def run(self):
        temp_root = None
        try:
            interpolation = str(self.settings.get("interpolation", "linear"))
            dtype_mode = str(self.settings.get("dtype", "preserve"))
            total_tasks = max(1, len(self.tasks))
            needs_cmtk = any(
                str(task.get("engine") or "itk").lower() == "cmtk"
                for task in self.tasks
                if str(task.get("kind") or "volume").lower() == "volume"
            )
            cmtk_runner = None
            if needs_cmtk:
                if self.cmtk_backend is None:
                    raise RuntimeError("CMTK nonlinear Reformat was requested without a validated backend.")
                temp_root = Path(tempfile.mkdtemp(prefix="madi3d-cmtk-reformat-"))
                cmtk_runner = CMTKRegistrationRunner(self.cmtk_backend)
            created = 0
            for task_index, task in enumerate(self.tasks):
                self._check_cancel()
                task_kind = str(task.get("kind") or "volume").lower()
                engine = str(task.get("engine") or "itk").lower()
                if task_kind == "mesh":
                    if engine == "cmtk":
                        raise RuntimeError(
                            "CMTK nonlinear mesh baking is not implemented. "
                            "This task should have been filtered before Reformat started."
                        )
                    points_local = np.asarray(task.get("points_local"), dtype=np.float64).reshape(-1, 3)
                    source_world = _apply_points_affine(points_local, task["source_actor_matrix"])
                    reference_actor_inverse = np.linalg.inv(_matrix4(task["reference_actor_matrix"]))
                    self.progress.emit(-1, f"Reformatting attached mesh {task['display_name']} — transforming vertices")
                    reference_world = _apply_points_affine(
                        source_world, task["transform"]["cumulative_moving_to_fixed"]
                    )
                    reference_local = _apply_points_affine(reference_world, reference_actor_inverse)
                    self.outputReady.emit({
                        "kind": "mesh",
                        "points_local": reference_local.astype(np.float64, copy=False),
                        "display_name": task["output_name"],
                        "source_descriptor": copy.deepcopy(task.get("source_descriptor") or {}),
                        "source_actor_id": task.get("source_actor_id"),
                        "reference_actor_matrix": copy.deepcopy(task["reference_actor_matrix"]),
                        "registration_id": str(task.get("registration_id") or ""),
                        "registration_stage": str(task.get("registration_stage") or ""),
                        "source_name": str(task.get("display_name") or "Mesh"),
                        "reference_name": str(task.get("reference_name") or "Reference"),
                        "disk_output_path": str(task.get("disk_output_path") or ""),
                        "written_format": str(task.get("disk_output_format") or ""),
                    })
                    created += 1
                    continue

                data = np.asanyarray(task["data"])
                frames = data if data.ndim == 4 else data[np.newaxis, ...]
                if frames.ndim != 4:
                    raise RuntimeError(f"Reformat expects Z,Y,X or T,Z,Y,X data, got {data.shape}.")
                output = None
                total_frames = max(1, int(frames.shape[0]))
                for frame_index in range(total_frames):
                    self._check_cancel()
                    frame = np.ascontiguousarray(frames[frame_index])
                    if dtype_mode == "float32":
                        frame = frame.astype(np.float32, copy=False)
                    self.progress.emit(
                        -1,
                        f"Reformatting {task['display_name']} — frame {frame_index + 1}/{total_frames} "
                        f"({'CMTK reformatx' if engine == 'cmtk' else 'resampling'})",
                    )

                    def frame_progress(local_fraction):
                        task_fraction = (frame_index + float(local_fraction)) / total_frames
                        overall = (task_index + task_fraction) / total_tasks
                        self.progress.emit(
                            max(0, min(98, int(round(98.0 * overall)))),
                            f"Reformatting {task['display_name']} — frame {frame_index + 1}/{total_frames} "
                            f"({100.0 * float(local_fraction):.0f}%)",
                        )

                    if engine == "cmtk":
                        frame_workspace = temp_root / f"task-{task_index:04d}-frame-{frame_index:04d}"
                        out_array = self._cmtk_reformat_frame(
                            frame, task, interpolation, dtype_mode, cmtk_runner, frame_workspace
                        )
                    else:
                        out_array = self._itk_reformat_frame(
                            frame, task, interpolation, dtype_mode, frame_progress
                        )
                    self._check_cancel()
                    if total_frames == 1:
                        output = out_array
                    else:
                        if output is None:
                            output = np.empty((total_frames,) + tuple(out_array.shape), dtype=out_array.dtype)
                        output[frame_index] = out_array
                    fraction = (task_index + (frame_index + 1) / total_frames) / total_tasks
                    self.progress.emit(
                        max(0, min(99, int(round(99.0 * fraction)))),
                        f"Finished {task['display_name']} — frame {frame_index + 1}/{total_frames}",
                    )
                if output is None:
                    raise RuntimeError(f"Reformat produced no frames for {task['display_name']}.")
                written_path = ""
                disk_output_path = str(task.get("disk_output_path") or "").strip()
                if disk_output_path:
                    if not callable(self.volume_output_writer):
                        raise RuntimeError(
                            "Registration result writing requires MADI3D's central volume output writer."
                        )
                    self.progress.emit(-1, f"Writing reformatted volume {task['display_name']} to disk")
                    final_path = Path(disk_output_path)
                    temporary_path = partial_output_path(final_path)
                    temporary_path.unlink(missing_ok=True)
                    try:
                        writer_kwargs = {}
                        registration_provenance = copy.deepcopy(
                            task.get("registration_provenance") or {}
                        )
                        registration_provenance.update(
                            {
                                "operation": "registration_reformat",
                                "registration_id": str(
                                    task.get("registration_id") or ""
                                ),
                                "registration_stage": str(
                                    task.get("registration_stage") or ""
                                ),
                                "source_name": str(
                                    task.get("display_name") or "Volume"
                                ),
                                "reference_name": str(
                                    task.get("reference_name") or "Reference"
                                ),
                                "source_entry_id": str(
                                    task.get("source_entry_id") or ""
                                ),
                                "source_acquisition_id": str(
                                    task.get("source_acquisition_id") or ""
                                ),
                                "source_channel_id": str(
                                    task.get("source_channel_id") or ""
                                ),
                                "source_backing_source_id": str(
                                    task.get("source_backing_source_id") or ""
                                ),
                                "source_geometry_revision": str(
                                    task.get("source_geometry_revision") or ""
                                ),
                                "reference_acquisition_id": str(
                                    task.get("reference_acquisition_id") or ""
                                ),
                                "reference_channel_id": str(
                                    task.get("reference_channel_id") or ""
                                ),
                                "reference_geometry_revision": str(
                                    task.get("reference_geometry_revision") or ""
                                ),
                                "source_operation_ids": list(
                                    task.get("source_operation_ids") or ()
                                ),
                                "reference_operation_ids": list(
                                    task.get("reference_operation_ids") or ()
                                ),
                                "supporting_operations": copy.deepcopy(
                                    task.get("supporting_operations") or ()
                                ),
                                "result_transform_id": (
                                    str(task.get("registration_id") or "")
                                    + ":"
                                    + str(task.get("registration_stage") or "")
                                    if task.get("registration_id")
                                    and task.get("registration_stage")
                                    else ""
                                ),
                                "result_transform": copy.deepcopy(
                                    task.get("transform")
                                ),
                                "engine": engine,
                                "interpolation": interpolation,
                                "dtype_mode": dtype_mode,
                                "output_source_id": str(
                                    task.get("output_source_id") or ""
                                ),
                                "output_source_channel": copy.deepcopy(
                                    task.get("output_source_channel")
                                ),
                                "output_acquisition_name": str(
                                    task.get("output_acquisition_name") or ""
                                ),
                                "output_acquisition_size": int(
                                    task.get("output_acquisition_size") or 1
                                ),
                                "output_channel_order": int(
                                    task.get("output_channel_order") or 0
                                ),
                            }
                        )
                        writer_kwargs["registration_provenance"] = (
                            registration_provenance
                        )
                        if str(task.get("disk_output_format") or "nrrd").lower() == "h5j":
                            writer_kwargs.update({
                                "ffmpeg_executable": self.ffmpeg_executable,
                                "cancel_check": lambda: bool(
                                    self._cancel or self.isInterruptionRequested()
                                ),
                            })
                        self.volume_output_writer(
                            str(temporary_path),
                            output,
                            str(task.get("disk_output_format") or "nrrd"),
                            copy.deepcopy(task["reference_geometry"]),
                            copy.deepcopy(task["reference_actor_matrix"]),
                            copy.deepcopy(task.get("source_metadata") or {}),
                            copy.deepcopy(task.get("source_time") or {}),
                            str(task.get("output_name") or task.get("display_name") or "registered"),
                            copy.deepcopy(task.get("reference_space_units")),
                            str(task.get("coordinate_space_id") or ""),
                            **writer_kwargs,
                        )
                        if not temporary_path.is_file():
                            raise RuntimeError(
                                f"Registration volume writer returned without creating: {temporary_path}"
                            )
                        final_path.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(temporary_path, final_path)
                    except BaseException:
                        temporary_path.unlink(missing_ok=True)
                        raise
                    written_path = str(final_path)

                payload = {
                    "kind": "volume",
                    "display_name": task["output_name"],
                    "source_descriptor": copy.deepcopy(task.get("source_descriptor") or {}),
                    "source_metadata": copy.deepcopy(task.get("source_metadata") or {}),
                    "source_time": copy.deepcopy(task.get("source_time") or {}),
                    "reference_geometry": copy.deepcopy(task["reference_geometry"]),
                    "reference_actor_matrix": copy.deepcopy(task["reference_actor_matrix"]),
                    "reference_space_units": copy.deepcopy(task.get("reference_space_units")),
                    "registration_provenance": copy.deepcopy(
                        task.get("registration_provenance") or {}
                    ),
                    "coordinate_space_id": str(task.get("coordinate_space_id") or ""),
                    "registration_id": str(task.get("registration_id") or ""),
                    "registration_stage": str(task.get("registration_stage") or ""),
                    "source_name": str(task.get("display_name") or "Volume"),
                    "reference_name": str(task.get("reference_name") or "Reference"),
                    "source_entry_id": str(task.get("source_entry_id") or ""),
                    "source_acquisition_id": str(
                        task.get("source_acquisition_id") or ""
                    ),
                    "source_channel_id": str(
                        task.get("source_channel_id") or ""
                    ),
                    "source_backing_source_id": str(
                        task.get("source_backing_source_id") or ""
                    ),
                    "source_geometry_revision": str(
                        task.get("source_geometry_revision") or ""
                    ),
                    "reference_acquisition_id": str(
                        task.get("reference_acquisition_id") or ""
                    ),
                    "reference_channel_id": str(
                        task.get("reference_channel_id") or ""
                    ),
                    "reference_geometry_revision": str(
                        task.get("reference_geometry_revision") or ""
                    ),
                    "source_operation_ids": list(
                        task.get("source_operation_ids") or ()
                    ),
                    "reference_operation_ids": list(
                        task.get("reference_operation_ids") or ()
                    ),
                    "supporting_operations": copy.deepcopy(
                        task.get("supporting_operations") or ()
                    ),
                    "output_source_id": str(task.get("output_source_id") or ""),
                    "output_source_channel": task.get("output_source_channel"),
                    "output_acquisition_name": str(
                        task.get("output_acquisition_name") or ""
                    ),
                    "output_acquisition_size": int(
                        task.get("output_acquisition_size") or 1
                    ),
                    "output_channel_order": int(
                        task.get("output_channel_order") or 0
                    ),
                    "written_path": written_path,
                    "written_format": str(task.get("disk_output_format") or ""),
                }
                if bool(task.get("emit_data", True)):
                    payload["data"] = output
                self.outputReady.emit(payload)
                created += 1
            self.progress.emit(100, "Reformat complete")
            self.succeeded.emit(created)
        except InterruptedError:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            if temp_root is not None:
                shutil.rmtree(temp_root, ignore_errors=True)
            self.tasks = []
            self.cmtk_backend = None
            self.volume_output_writer = None
            self.ffmpeg_executable = None
