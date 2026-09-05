"""GUI-independent stitching job, edge, residual, and QC records."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

import numpy as np

from madi3d_app.operation_status import (
    execution_status as validate_execution_status,
    qc_status as validate_qc_status,
    user_decision as validate_user_decision,
)
from madi3d_app.volume.geometry import (
    WORKING_GRID_BASIS_FULLY_VOXEL_DEFAULT,
    canonical_space_units,
    invertible_affine4,
)


STITCHING_JOB_SCHEMA_VERSION = "MADI3D_stitching_job_v2"
STITCHING_WORKSPACE_SCHEMA_VERSION = "MADI3D_stitching_workspace_v1"


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return copy.deepcopy(value)


def _finite_json_value(value, field_name):
    payload = _json_value(value)
    try:
        json.dumps(payload, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain finite JSON values.") from exc
    return payload


def _validated_project_state(value, field_name="Stitching project state"):
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    payload = _finite_json_value(value, field_name)
    tiles = payload.get("tiles", [])
    if not isinstance(tiles, list):
        raise ValueError(f"{field_name} tiles must be a list.")
    if tiles:
        if not str(payload.get("project_id") or "").strip():
            raise ValueError(f"{field_name} requires a stable project ID.")
        schema = str(payload.get("schema") or "")
        if schema != "MADI3D_stitching_project_tree_v2":
            raise ValueError(f"{field_name} uses unsupported schema {schema!r}.")
    tile_ids = []
    for tile in tiles:
        if not isinstance(tile, Mapping):
            raise ValueError(f"{field_name} tile records must be mappings.")
        tile_id = str(tile.get("tile_id") or "").strip()
        if not tile_id:
            raise ValueError(f"{field_name} tiles require stable tile IDs.")
        tile_ids.append(tile_id)
        channels = tile.get("channels", [])
        if not isinstance(channels, list):
            raise ValueError(f"{field_name} tile channels must be a list.")
        for channel in channels:
            if not isinstance(channel, Mapping):
                raise ValueError(f"{field_name} channel records must be mappings.")
            descriptor = channel.get("descriptor")
            if not isinstance(descriptor, Mapping):
                raise ValueError(
                    f"{field_name} channels require source descriptors."
                )
            migration_status = str(descriptor.get("migration_status") or "")
            if migration_status != "unresolved-legacy-source":
                if not str(descriptor.get("source_id") or "").strip():
                    raise ValueError(
                        f"{field_name} channels require authoritative SourceID."
                    )
                if not str(descriptor.get("channel_id") or "").strip():
                    raise ValueError(
                        f"{field_name} channels require authoritative channel identity."
                    )
            pose = descriptor.get("project_capture_pose")
            if pose is not None:
                invertible_affine4(pose, "Stitching project captured pose")
    if len(tile_ids) != len(set(tile_ids)):
        raise ValueError(f"{field_name} tile IDs must be unique.")
    return payload


def _validated_pose_state(value, field_name):
    payload = _finite_json_value(value or {}, field_name)
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    for key in ("before_matrices", "after_matrices", "exact_target_matrices"):
        matrices = payload.get(key, {})
        if matrices:
            if not isinstance(matrices, Mapping):
                raise ValueError(f"{field_name} {key} must be a mapping.")
            for source_key, matrix in matrices.items():
                invertible_affine4(
                    matrix, f"{field_name} {key} {source_key}"
                )
    return payload


def _validated_initial_layout(value):
    payload = _finite_json_value(value or {}, "Stitching initial-placement evidence")
    if not isinstance(payload, dict):
        raise ValueError("Stitching initial-placement evidence must be a mapping.")
    deltas = payload.get("placement_deltas", {})
    if not isinstance(deltas, Mapping):
        raise ValueError("Stitching initial-placement deltas must be a mapping.")
    for tile_id, matrix in deltas.items():
        invertible_affine4(
            matrix, f"Stitching initial-placement delta {tile_id}"
        )
    return payload


def _validated_pose_undo_stack(value):
    payload = _finite_json_value(value or [], "Stitching pose undo stack")
    if not isinstance(payload, list):
        raise ValueError("Stitching pose undo stack must be a list.")
    for index, entry in enumerate(payload):
        if not isinstance(entry, Mapping):
            raise ValueError("Stitching pose undo entries must be mappings.")
        matrices = entry.get("matrices", {})
        if not isinstance(matrices, Mapping):
            raise ValueError("Stitching pose undo matrices must be a mapping.")
        for source_key, matrix in matrices.items():
            invertible_affine4(
                matrix,
                f"Stitching pose undo entry {index} matrix {source_key}",
            )
    return payload


def _finite_float(value, field_name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite.")
    return result


def _float_vector(value, length, field_name, default):
    source = default if value is None else value
    array = np.asarray(source, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain {length} finite values.")
    return tuple(float(component) for component in array)


_POSE_DEPENDENT_GEOMETRY_FIELDS = {
    "effective_support",
    "index_to_world_affine",
    "pose",
}


def stitching_grid_revision(working_geometry):
    """Return a stable revision for a channel-local grid, excluding scene pose."""
    if not isinstance(working_geometry, Mapping) or not working_geometry:
        raise ValueError("A stitching source requires exact working geometry.")
    payload = {
        str(key): _json_value(value)
        for key, value in working_geometry.items()
        if str(key) not in _POSE_DEPENDENT_GEOMETRY_FIELDS
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def stitching_geometry_status(grid_state, working_geometry):
    """Return the stable nonmodal status stored with a source descriptor."""
    geometry = dict(working_geometry or {})
    state = str(grid_state or "").strip().lower()
    if state == "resolved":
        return {"code": "calibrated", "label": "calibrated"}
    if state == "inconsistent":
        return {
            "code": "source-metadata-conflicting",
            "label": "source metadata conflicting",
        }
    if (
        str(geometry.get("geometry_basis") or "")
        == WORKING_GRID_BASIS_FULLY_VOXEL_DEFAULT
        or geometry.get("assumed_fields")
        or geometry.get("replaced_fields")
    ):
        return {
            "code": "working-defaults-used",
            "label": "working defaults used",
        }
    return {
        "code": "source-geometry-unverified",
        "label": "source geometry unverified",
    }


@dataclass(frozen=True)
class StitchingMosaicGeometry:
    """Serializable geometry and provenance for one job-local mosaic space."""

    coordinate_space_id: str
    output_geometry_status: dict[str, Any]
    output_space_units: tuple[str, str, str] | None
    chosen_output_unit: str | None
    source_tiles: tuple[dict[str, Any], ...] = ()
    normalized_working_affines: tuple[dict[str, Any], ...] = ()
    unit_conversions: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    assumptions: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        identity = str(self.coordinate_space_id or "").strip()
        if not identity:
            raise ValueError("A stitching mosaic requires a coordinate-space identity.")
        object.__setattr__(self, "coordinate_space_id", identity)
        if not isinstance(self.output_geometry_status, Mapping):
            raise ValueError("Stitching output geometry status must be a mapping.")
        status = _json_value(self.output_geometry_status)
        if not str(status.get("code") or "").strip():
            raise ValueError("Stitching output geometry status requires a code.")
        object.__setattr__(self, "output_geometry_status", status)

        if self.output_space_units is not None:
            units = canonical_space_units(self.output_space_units)
            object.__setattr__(self, "output_space_units", units)
            chosen = str(self.chosen_output_unit or units[0]).strip()
            if chosen != units[0]:
                raise ValueError(
                    "The chosen stitching output unit must match output space units."
                )
            object.__setattr__(self, "chosen_output_unit", chosen)
        elif self.chosen_output_unit is not None:
            chosen = canonical_space_units(self.chosen_output_unit)[0]
            object.__setattr__(self, "chosen_output_unit", chosen)

        for field_name in (
            "source_tiles",
            "normalized_working_affines",
            "unit_conversions",
            "warnings",
            "assumptions",
        ):
            values = tuple(
                _json_value(dict(value)) for value in (getattr(self, field_name) or ())
            )
            # This is also the strict finite-value boundary for the manifest.
            json.dumps(values, allow_nan=False, ensure_ascii=False, sort_keys=True)
            object.__setattr__(self, field_name, values)

    def to_dict(self):
        return {
            "coordinate_space_id": self.coordinate_space_id,
            "output_geometry_status": _json_value(self.output_geometry_status),
            "output_space_units": (
                list(self.output_space_units)
                if self.output_space_units is not None
                else None
            ),
            "chosen_output_unit": self.chosen_output_unit,
            "source_tiles": _json_value(self.source_tiles),
            "normalized_working_affines": _json_value(
                self.normalized_working_affines
            ),
            "unit_conversions": _json_value(self.unit_conversions),
            "warnings": _json_value(self.warnings),
            "assumptions": _json_value(self.assumptions),
        }

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        return cls(
            coordinate_space_id=payload.get("coordinate_space_id", ""),
            output_geometry_status=dict(
                payload.get("output_geometry_status") or {}
            ),
            output_space_units=(
                tuple(payload["output_space_units"])
                if payload.get("output_space_units") is not None
                else None
            ),
            chosen_output_unit=payload.get("chosen_output_unit"),
            source_tiles=tuple(payload.get("source_tiles") or ()),
            normalized_working_affines=tuple(
                payload.get("normalized_working_affines") or ()
            ),
            unit_conversions=tuple(payload.get("unit_conversions") or ()),
            warnings=tuple(payload.get("warnings") or ()),
            assumptions=tuple(payload.get("assumptions") or ()),
        )


@dataclass(frozen=True)
class PoseGraphResidual:
    translation_world: float = 0.0
    rotation_degrees: float = 0.0
    affine_frobenius: float = 0.0

    def __post_init__(self):
        values = (
            self.translation_world,
            self.rotation_degrees,
            self.affine_frobenius,
        )
        if any(float(value) < 0.0 for value in values):
            raise ValueError("Pose-graph residuals cannot be negative.")
        object.__setattr__(
            self,
            "translation_world",
            _finite_float(self.translation_world, "translation residual"),
        )
        object.__setattr__(
            self,
            "rotation_degrees",
            _finite_float(self.rotation_degrees, "rotation residual"),
        )
        object.__setattr__(
            self,
            "affine_frobenius",
            _finite_float(self.affine_frobenius, "affine residual"),
        )

    @classmethod
    def from_edge(cls, edge):
        return cls(
            translation_world=edge.get("global_translation_residual", 0.0),
            rotation_degrees=edge.get("global_rotation_residual_deg", 0.0),
            affine_frobenius=edge.get("global_affine_residual", 0.0),
        )

    def to_dict(self):
        return {
            "translation_world": self.translation_world,
            "rotation_degrees": self.rotation_degrees,
            "affine_frobenius": self.affine_frobenius,
        }


@dataclass(frozen=True)
class StitchingRejection:
    code: str
    reason: str
    fixed_id: str = ""
    moving_id: str = ""
    fixed_name: str = ""
    moving_name: str = ""
    score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not str(self.code).strip():
            raise ValueError("A stitching rejection requires a code.")
        if not str(self.reason).strip():
            raise ValueError("A stitching rejection requires a reason.")
        if self.score is not None:
            object.__setattr__(
                self, "score", _finite_float(self.score, "rejected edge score")
            )
        details = _json_value(dict(self.details or {}))
        json.dumps(details, allow_nan=False, ensure_ascii=False, sort_keys=True)
        object.__setattr__(self, "details", details)

    @classmethod
    def from_dict(cls, payload):
        return cls(
            code=str(payload.get("code") or payload.get("error_type") or "rejected"),
            reason=str(payload.get("reason") or payload.get("error") or "Rejected"),
            fixed_id=str(payload.get("fixed_id") or ""),
            moving_id=str(payload.get("moving_id") or ""),
            fixed_name=str(payload.get("fixed_name") or ""),
            moving_name=str(payload.get("moving_name") or ""),
            score=payload.get("score"),
            details=dict(payload.get("details") or {}),
        )

    def to_dict(self):
        payload = {
            "code": str(self.code),
            "reason": str(self.reason),
            "fixed_id": str(self.fixed_id),
            "moving_id": str(self.moving_id),
            "fixed_name": str(self.fixed_name),
            "moving_name": str(self.moving_name),
        }
        if self.score is not None:
            payload["score"] = self.score
        if self.details:
            payload["details"] = _json_value(self.details)
        return payload


@dataclass(frozen=True)
class StitchingEdgeResult:
    fixed_id: str
    moving_id: str
    fixed_name: str
    moving_name: str
    score: float
    correction: np.ndarray = field(repr=False)
    translation_xyz: tuple[float, float, float]
    affine_params: tuple[float, ...]
    scale_xyz: tuple[float, float, float]
    shear_xyz: tuple[float, float, float]
    residual: PoseGraphResidual
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "score", _finite_float(self.score, "edge score"))
        object.__setattr__(
            self,
            "correction",
            invertible_affine4(self.correction, "Stitching edge correction").copy(),
        )
        object.__setattr__(
            self,
            "translation_xyz",
            _float_vector(self.translation_xyz, 3, "edge translation", (0.0,) * 3),
        )
        affine = tuple(float(value) for value in self.affine_params)
        if len(affine) != 12 or not np.all(np.isfinite(affine)):
            raise ValueError("edge affine parameters must contain 12 finite values.")
        object.__setattr__(self, "affine_params", affine)
        object.__setattr__(
            self,
            "scale_xyz",
            _float_vector(self.scale_xyz, 3, "edge scale", (1.0,) * 3),
        )
        object.__setattr__(
            self,
            "shear_xyz",
            _float_vector(self.shear_xyz, 3, "edge shear", (0.0,) * 3),
        )

    @classmethod
    def from_dict(cls, payload):
        known = {
            "fixed_id", "moving_id", "fixed_name", "moving_name", "score",
            "correction", "translation_xyz", "affine_params", "scale_xyz",
            "shear_xyz", "global_translation_residual",
            "global_rotation_residual_deg", "global_affine_residual", "residual",
        }
        residual_payload = payload.get("residual")
        residual = (
            PoseGraphResidual(**residual_payload)
            if isinstance(residual_payload, Mapping)
            else PoseGraphResidual.from_edge(payload)
        )
        return cls(
            fixed_id=str(payload.get("fixed_id") or ""),
            moving_id=str(payload.get("moving_id") or ""),
            fixed_name=str(payload.get("fixed_name") or payload.get("fixed_id") or ""),
            moving_name=str(payload.get("moving_name") or payload.get("moving_id") or ""),
            score=payload.get("score", 0.0),
            correction=payload.get("correction", np.eye(4)),
            translation_xyz=_float_vector(
                payload.get("translation_xyz"), 3, "edge translation", (0.0,) * 3
            ),
            affine_params=tuple(payload.get("affine_params", (0.0,) * 12)),
            scale_xyz=_float_vector(
                payload.get("scale_xyz"), 3, "edge scale", (1.0,) * 3
            ),
            shear_xyz=_float_vector(
                payload.get("shear_xyz"), 3, "edge shear", (0.0,) * 3
            ),
            residual=residual,
            extra={key: copy.deepcopy(value) for key, value in payload.items() if key not in known},
        )

    def to_runtime_dict(self):
        payload = copy.deepcopy(self.extra)
        payload.update({
            "fixed_id": self.fixed_id,
            "moving_id": self.moving_id,
            "fixed_name": self.fixed_name,
            "moving_name": self.moving_name,
            "score": self.score,
            "correction": self.correction.copy(),
            "translation_xyz": np.asarray(self.translation_xyz, dtype=float),
            "affine_params": np.asarray(self.affine_params, dtype=float),
            "scale_xyz": np.asarray(self.scale_xyz, dtype=float),
            "shear_xyz": np.asarray(self.shear_xyz, dtype=float),
            "global_translation_residual": self.residual.translation_world,
            "global_rotation_residual_deg": self.residual.rotation_degrees,
            "global_affine_residual": self.residual.affine_frobenius,
        })
        return payload

    def to_dict(self):
        payload = _json_value(self.to_runtime_dict())
        payload["residual"] = self.residual.to_dict()
        return payload


def detached_stitching_settings(value, field_name="Stitching settings"):
    """Return plain, detached, finite JSON settings for persistence/dispatch."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    payload = _finite_json_value(value, field_name)
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return payload


@dataclass
class StitchingRegistrationResult(MutableMapping[str, Any]):
    data: dict[str, Any] = field(default_factory=dict)
    edge_results: list[StitchingEdgeResult] = field(default_factory=list)
    rejections: list[StitchingRejection] = field(default_factory=list)
    execution_status: str = "succeeded"
    qc_status: str = "not-evaluated"
    user_decision: str = "unapplied"

    _MATRIX_MAP_FIELDS = {
        "corrections",
        "mosaic_corrections",
        "base_matrices_by_source",
        "target_matrices_by_source",
    }

    def __post_init__(self):
        self.execution_status = validate_execution_status(self.execution_status)
        self.qc_status = validate_qc_status(self.qc_status)
        self.user_decision = validate_user_decision(self.user_decision)

    def __getitem__(self, key):
        if key in {"execution_status", "qc_status", "user_decision"}:
            return getattr(self, key)
        if key == "edges":
            return [edge.to_runtime_dict() for edge in self.edge_results]
        if key == "rejections":
            return [rejection.to_dict() for rejection in self.rejections]
        return self.data[key]

    def __setitem__(self, key, value):
        if key == "execution_status":
            self.execution_status = validate_execution_status(value)
            return
        if key == "qc_status":
            self.qc_status = validate_qc_status(value)
            return
        if key == "user_decision":
            self.user_decision = validate_user_decision(value)
            return
        if key == "edges":
            self.edge_results = [StitchingEdgeResult.from_dict(item) for item in value]
            return
        if key == "rejections":
            self.rejections = [StitchingRejection.from_dict(item) for item in value]
            return
        if key in self._MATRIX_MAP_FIELDS:
            self.data[key] = {
                str(name): invertible_affine4(
                    matrix, f"Stitching {key} {name}"
                ).copy()
                for name, matrix in dict(value or {}).items()
            }
            return
        self.data[str(key)] = copy.deepcopy(value)

    def __delitem__(self, key):
        if key in {"execution_status", "qc_status", "user_decision"}:
            raise KeyError(f"Cannot delete required stitching result field: {key}")
        if key == "edges":
            self.edge_results.clear()
        elif key == "rejections":
            self.rejections.clear()
        else:
            del self.data[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.data
        yield "execution_status"
        yield "qc_status"
        yield "user_decision"
        yield "edges"
        yield "rejections"

    def __len__(self):
        return len(self.data) + 5

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        result = cls()
        for key, value in payload.items():
            result[key] = value
        return result

    from_runtime = from_dict

    def to_runtime_dict(self):
        payload = copy.deepcopy(self.data)
        payload["execution_status"] = self.execution_status
        payload["qc_status"] = self.qc_status
        payload["user_decision"] = self.user_decision
        payload["edges"] = [edge.to_runtime_dict() for edge in self.edge_results]
        payload["rejections"] = [item.to_dict() for item in self.rejections]
        return payload

    def to_dict(self):
        return _json_value(self.to_runtime_dict())

    def to_provenance_dict(self):
        """Return the stable persisted result without transient solver weights."""
        payload = self.to_dict()
        for edge in payload["edges"]:
            edge.pop("weight", None)
        return payload


def _matrix_payload(matrix):
    return np.asarray(matrix, dtype=float).round(12).tolist()


def _tile_field(tile, name, default=None):
    if isinstance(tile, Mapping):
        return tile.get(name, default)
    return getattr(tile, name, default)


def _json_safe_pose_state(value):
    state = copy.deepcopy(value or {})
    for field_name in ("before_matrices", "after_matrices", "exact_target_matrices"):
        if field_name in state:
            state[field_name] = {
                key: _matrix_payload(matrix)
                for key, matrix in state[field_name].items()
            }
    for item in (state.get("before_states") or {}).values():
        if isinstance(item, dict) and "matrix" in item:
            item["matrix"] = _matrix_payload(item["matrix"])
    return _finite_json_value(state, "Stitching applied pose state")


def build_stitching_fusion_manifest(
    *,
    schema,
    algorithm_version,
    result,
    initial_layout,
    applied_state,
    registration_channel,
    registration_channel_key,
    registration_channel_label,
    selected_initial_pose_source,
    tiles,
    project_state,
    channel_sets,
    queue_job=None,
):
    """Build the detached scientific manifest shared by interactive and queued fusion."""
    tiles = list(tiles)
    channel_sets = list(channel_sets)
    result = result or {}
    serialized_result = StitchingRegistrationResult.from_runtime(
        result
    ).to_provenance_dict()
    edges = serialized_result["edges"]
    rejections = serialized_result["rejections"]
    raw_initial = initial_layout or {
        "mode": "current",
        "summary": "Current MADI3D poses used as the initial layout.",
        "placement_deltas": {},
        "records": [],
    }
    serialized_initial = {
        key: copy.deepcopy(value)
        for key, value in raw_initial.items()
        if key != "placement_deltas"
    }
    serialized_initial["placement_deltas"] = {
        tile_id: _matrix_payload(matrix)
        for tile_id, matrix in raw_initial.get("placement_deltas", {}).items()
    }
    channel_layouts = {
        str(channel["label"]): [
            {
                "tile_id": tile["tile_id"],
                "display_name": tile["display_name"],
                "source_id": tile.get("source_id", ""),
                "channel_id": tile.get("channel_id", ""),
                "backing_source_id": tile.get("backing_source_id", ""),
                "channel_selector": tile.get("channel"),
                "channel_role": tile.get("channel_role", "other"),
                "channel_order": int(tile.get("channel_order", 0)),
                "source_path": tile.get("source_path", tile.get("name", "")),
                "source_operation_ids": list(tile.get("source_operation_ids", [])),
                "dims_xyz": list(tile["dims"]),
                "dtype": str(tile["dtype"]),
                "world_index_affine": _matrix_payload(tile["world_affine"]),
            }
            for tile in channel["tiles"]
        ]
        for channel in channel_sets
    }
    corrections = serialized_result.get("corrections", {})
    payload = {
        "schema": str(schema),
        "stitching_algorithm_version": str(algorithm_version),
        "created": datetime.now().isoformat(timespec="seconds"),
        "registration_mode": result.get("mode", "current_poses"),
        "execution_status": serialized_result.get("execution_status", "pending"),
        "qc_status": serialized_result.get("qc_status", "not-evaluated"),
        "user_decision": serialized_result.get("user_decision", "unapplied"),
        "registration_channel": str(registration_channel or ""),
        "registration_channel_key": copy.deepcopy(registration_channel_key),
        "registration_channel_label": str(registration_channel_label or ""),
        "selected_initial_pose_source": str(
            selected_initial_pose_source or "current"
        ),
        "initial_layout": serialized_initial,
        "tile_count": len(tiles),
        "project_tree": copy.deepcopy(project_state),
        "applied_state": _json_safe_pose_state(applied_state),
        "channels": [channel["label"] for channel in channel_sets],
        "channel_layouts": channel_layouts,
        "mosaic_coordinate_space_id": (
            (result.get("mosaic_geometry") or {}).get("coordinate_space_id")
        ),
        "registration_mosaic_geometry": copy.deepcopy(
            result.get("mosaic_geometry") or {}
        ),
        "corrections": corrections,
        "registered_actor_matrices": {
            key: _matrix_payload(value)
            for key, value in (result.get("target_matrices_by_source") or {}).items()
        },
        "edges": edges,
        "rejections": rejections,
        "registration_inputs": copy.deepcopy(result.get("registration_inputs", [])),
        "candidate_pairs": copy.deepcopy(result.get("candidate_pairs", [])),
        "pair_evaluations": copy.deepcopy(result.get("pair_evaluations", [])),
        "candidate_pair_count": int(
            result.get("candidate_pair_count", len(edges) + len(rejections))
        ),
        "evaluated_pair_count": int(
            result.get("evaluated_pair_count", len(edges) + len(rejections))
        ),
        "accepted_edge_count": int(result.get("accepted_edge_count", len(edges))),
        "below_minimum_score_count": int(
            result.get(
                "below_minimum_score_count",
                sum(item.get("code") == "below_minimum_score" for item in rejections),
            )
        ),
        "invalid_pair_count": int(result.get("invalid_pair_count", 0)),
        "global_inconsistency_rejected_count": int(
            result.get("global_inconsistency_rejected_count", 0)
        ),
        "translation_acceptance_contract": copy.deepcopy(
            result.get("translation_acceptance_contract")
        ),
        "numerical_failure_count": int(
            result.get(
                "numerical_failure_count",
                sum(item.get("code") == "numerical_failure" for item in rejections),
            )
        ),
        "pair_failures": copy.deepcopy(result.get("pair_failures", [])),
        "components": copy.deepcopy(result.get("components", [])),
        "pose_graph_qc": copy.deepcopy(result.get("pose_graph_qc") or {}),
        "registration_warnings": copy.deepcopy(result.get("warnings", [])),
        "registration_assumptions": copy.deepcopy(result.get("assumptions", [])),
        "completed_with_warnings": bool(result.get("completed_with_warnings")),
        "registration_settings": copy.deepcopy(result.get("settings", {})),
        "anchor_tile": next(
            (
                str(_tile_field(tile, "display_name", ""))
                for tile in tiles
                if bool(_tile_field(tile, "anchor", False))
            ),
            None,
        ),
        "queue_job": copy.deepcopy(queue_job),
        "tiles": [
            {
                "tile_id": str(_tile_field(tile, "tile_id", "")),
                "display_name": str(_tile_field(tile, "display_name", "")),
                "multichannel": bool(_tile_field(tile, "multichannel", False)),
                "anchor": bool(_tile_field(tile, "anchor", False)),
                "reference_enabled": bool(
                    _tile_field(tile, "reference_enabled", True)
                ),
                "channels": copy.deepcopy(
                    _tile_field(tile, "channel_labels", {})
                ),
            }
            for tile in tiles
        ],
    }
    return _finite_json_value(payload, "Stitching fusion manifest")


@dataclass
class StitchingProjectState:
    """Serializable active stitching project, independent of panel/runtime objects."""

    project_state: dict[str, Any]
    settings: dict[str, Any]
    registration_result: StitchingRegistrationResult | None = None
    initial_layout: dict[str, Any] = field(default_factory=dict)
    applied_state: dict[str, Any] = field(default_factory=dict)
    pose_undo_stack: list[dict[str, Any]] = field(default_factory=list)
    loaded_job_id: int | None = None
    qc_status: str = "not-evaluated"
    user_decision: str = "unapplied"
    binding_issues: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.project_state = _validated_project_state(self.project_state)
        self.settings = detached_stitching_settings(
            self.settings, "Stitching project settings"
        )
        self.registration_result = (
            self.registration_result
            if isinstance(self.registration_result, StitchingRegistrationResult)
            else StitchingRegistrationResult.from_dict(self.registration_result)
            if self.registration_result
            else None
        )
        self.applied_state = _validated_pose_state(
            self.applied_state, "Stitching applied state"
        )
        self.initial_layout = _validated_initial_layout(self.initial_layout)
        self.pose_undo_stack = _validated_pose_undo_stack(self.pose_undo_stack)
        self.loaded_job_id = (
            int(self.loaded_job_id) if self.loaded_job_id is not None else None
        )
        self.qc_status = validate_qc_status(self.qc_status)
        self.user_decision = validate_user_decision(self.user_decision)
        issues = _finite_json_value(
            self.binding_issues or [], "Stitching project binding issues"
        )
        if not isinstance(issues, list) or any(
            not isinstance(value, Mapping) for value in issues
        ):
            raise ValueError("Stitching project binding issues must be a list of mappings.")
        self.binding_issues = [dict(value) for value in issues]

    def to_dict(self):
        return {
            "project_state": copy.deepcopy(self.project_state),
            "settings": copy.deepcopy(self.settings),
            "registration_result": (
                self.registration_result.to_dict()
                if self.registration_result is not None
                else None
            ),
            "initial_layout": copy.deepcopy(self.initial_layout),
            "applied_state": copy.deepcopy(self.applied_state),
            "pose_undo_stack": copy.deepcopy(self.pose_undo_stack),
            "loaded_job_id": self.loaded_job_id,
            "qc_status": self.qc_status,
            "user_decision": self.user_decision,
            "binding_issues": copy.deepcopy(self.binding_issues),
        }

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        return cls(
            project_state=copy.deepcopy(payload.get("project_state") or {}),
            settings=detached_stitching_settings(
                payload.get("settings"), "Stitching project settings"
            ),
            registration_result=(
                StitchingRegistrationResult.from_dict(payload["registration_result"])
                if payload.get("registration_result")
                else None
            ),
            initial_layout=copy.deepcopy(payload.get("initial_layout") or {}),
            applied_state=copy.deepcopy(payload.get("applied_state") or {}),
            pose_undo_stack=copy.deepcopy(payload.get("pose_undo_stack") or []),
            loaded_job_id=payload.get("loaded_job_id"),
            qc_status=payload.get("qc_status", "not-evaluated"),
            user_decision=payload.get("user_decision", "unapplied"),
            binding_issues=copy.deepcopy(payload.get("binding_issues") or []),
        )


@dataclass
class StitchingJob:
    job_id: int
    name: str
    registration_mode: str
    settings: dict[str, Any]
    project_state: dict[str, Any]
    registration_result: StitchingRegistrationResult | None = None
    initial_layout: dict[str, Any] = field(default_factory=dict)
    applied_state: dict[str, Any] = field(default_factory=dict)
    pose_undo_stack: list[dict[str, Any]] = field(default_factory=list)
    selection_descriptors: list[dict[str, Any]] = field(default_factory=list)
    execution_status: str = "pending"
    qc_status: str = "not-evaluated"
    user_decision: str = "unapplied"
    execution_phase: str = ""
    outputs: list[str] = field(default_factory=list)
    error: str = ""
    binding_issues: list[dict[str, Any]] = field(default_factory=list)
    created: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def __post_init__(self):
        self.job_id = int(self.job_id)
        self.execution_status = validate_execution_status(self.execution_status)
        self.qc_status = validate_qc_status(self.qc_status)
        self.user_decision = validate_user_decision(self.user_decision)
        self.settings = detached_stitching_settings(
            self.settings, "Stitching job settings"
        )
        self.registration_result = (
            self.registration_result
            if isinstance(self.registration_result, StitchingRegistrationResult)
            else StitchingRegistrationResult.from_dict(self.registration_result)
            if self.registration_result
            else None
        )
        self.project_state = _finite_json_value(
            self.project_state or {}, "Stitching job project state"
        )
        self.applied_state = _validated_pose_state(
            self.applied_state, "Stitching job applied state"
        )
        self.initial_layout = _validated_initial_layout(self.initial_layout)
        self.pose_undo_stack = _validated_pose_undo_stack(self.pose_undo_stack)
        issues = _finite_json_value(
            self.binding_issues or [], "Stitching job binding issues"
        )
        if not isinstance(issues, list) or any(
            not isinstance(value, Mapping) for value in issues
        ):
            raise ValueError("Stitching job binding issues must be a list of mappings.")
        self.binding_issues = [dict(value) for value in issues]

    def to_dict(self):
        return {
            "schema": STITCHING_JOB_SCHEMA_VERSION,
            "job_id": self.job_id,
            "name": str(self.name),
            "registration_mode": str(self.registration_mode),
            "settings": copy.deepcopy(self.settings),
            "project_state": _json_value(self.project_state),
            "registration_result": (
                self.registration_result.to_dict() if self.registration_result else None
            ),
            "initial_layout": copy.deepcopy(self.initial_layout),
            "applied_state": _json_value(self.applied_state),
            "pose_undo_stack": _json_value(self.pose_undo_stack),
            "selection_descriptors": _json_value(self.selection_descriptors),
            "execution_status": self.execution_status,
            "qc_status": self.qc_status,
            "user_decision": self.user_decision,
            "execution_phase": str(self.execution_phase),
            "outputs": [str(path) for path in self.outputs],
            "error": str(self.error),
            "binding_issues": copy.deepcopy(self.binding_issues),
            "created": str(self.created),
        }

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        schema = str(payload.get("schema") or STITCHING_JOB_SCHEMA_VERSION)
        if schema != STITCHING_JOB_SCHEMA_VERSION:
            raise ValueError(f"Unsupported stitching job schema: {schema}")
        return cls(
            job_id=payload["job_id"],
            name=payload.get("name", "Stitching job"),
            registration_mode=payload.get("registration_mode", "needs_review"),
            settings=detached_stitching_settings(
                payload.get("settings"), "Stitching job settings"
            ),
            project_state=copy.deepcopy(payload.get("project_state") or {}),
            registration_result=(
                StitchingRegistrationResult.from_dict(payload["registration_result"])
                if payload.get("registration_result") else None
            ),
            initial_layout=copy.deepcopy(payload.get("initial_layout") or {}),
            applied_state=copy.deepcopy(payload.get("applied_state") or {}),
            pose_undo_stack=copy.deepcopy(payload.get("pose_undo_stack") or []),
            selection_descriptors=copy.deepcopy(
                payload.get("selection_descriptors") or []
            ),
            execution_status=payload.get("execution_status", "pending"),
            qc_status=payload.get("qc_status", "not-evaluated"),
            user_decision=payload.get("user_decision", "unapplied"),
            execution_phase=payload.get("execution_phase", ""),
            outputs=[str(path) for path in payload.get("outputs") or []],
            error=payload.get("error", ""),
            binding_issues=copy.deepcopy(payload.get("binding_issues") or []),
            created=payload.get("created", ""),
        )


@dataclass
class StitchingWorkspaceState:
    """Authoritative active stitching project plus its ordered execution queue."""

    active_project: StitchingProjectState | None = None
    jobs: list[StitchingJob] = field(default_factory=list)
    job_counter: int = 0

    def __post_init__(self):
        self.active_project = (
            self.active_project
            if isinstance(self.active_project, StitchingProjectState)
            else StitchingProjectState.from_dict(self.active_project)
            if self.active_project
            else None
        )
        self.jobs = [
            job if isinstance(job, StitchingJob) else StitchingJob.from_dict(job)
            for job in self.jobs
        ]
        job_ids = [job.job_id for job in self.jobs]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("Stitching workspace job IDs must be unique.")
        self.job_counter = int(self.job_counter)
        if self.job_counter < 0 or self.job_counter < max(job_ids, default=0):
            raise ValueError(
                "Stitching workspace job counter must cover every queued job ID."
            )
        for job in self.jobs:
            _validated_project_state(
                job.project_state,
                f"Stitching job {job.job_id} project state",
            )
        if (
            self.active_project is not None
            and self.active_project.loaded_job_id is not None
            and self.active_project.loaded_job_id not in set(job_ids)
        ):
            raise ValueError(
                "The active stitching project refers to a missing queued job."
            )

    @property
    def is_empty(self):
        return self.active_project is None and not self.jobs

    def to_dict(self):
        return {
            "schema": STITCHING_WORKSPACE_SCHEMA_VERSION,
            "active_project": (
                self.active_project.to_dict()
                if self.active_project is not None
                else None
            ),
            "jobs": [job.to_dict() for job in self.jobs],
            "job_counter": self.job_counter,
        }

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        if not payload:
            return cls()
        schema = str(payload.get("schema") or STITCHING_WORKSPACE_SCHEMA_VERSION)
        if schema != STITCHING_WORKSPACE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported stitching workspace schema: {schema}")
        return cls(
            active_project=(
                StitchingProjectState.from_dict(payload["active_project"])
                if payload.get("active_project")
                else None
            ),
            jobs=[StitchingJob.from_dict(job) for job in payload.get("jobs") or []],
            job_counter=payload.get("job_counter", 0),
        )


__all__ = [
    "PoseGraphResidual",
    "STITCHING_JOB_SCHEMA_VERSION",
    "STITCHING_WORKSPACE_SCHEMA_VERSION",
    "StitchingEdgeResult",
    "StitchingJob",
    "detached_stitching_settings",
    "StitchingMosaicGeometry",
    "StitchingProjectState",
    "StitchingRegistrationResult",
    "StitchingRejection",
    "StitchingWorkspaceState",
    "build_stitching_fusion_manifest",
    "stitching_grid_revision",
    "stitching_geometry_status",
]
