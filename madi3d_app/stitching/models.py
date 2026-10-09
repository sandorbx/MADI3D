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
from madi3d_app.volume.model import physical_grid_input_reference


STITCHING_JOB_SCHEMA_VERSION = "MADI3D_stitching_job_v3"
STITCHING_WORKSPACE_SCHEMA_VERSION = "MADI3D_stitching_workspace_v2"
UNAVAILABLE_STITCHING_RESULT_CODE = "unavailable-result"
_LEGACY_STITCHING_JOB_SCHEMAS = {"MADI3D_stitching_job_v2"}
_LEGACY_STITCHING_WORKSPACE_SCHEMAS = {"MADI3D_stitching_workspace_v1"}


class InvalidStitchingWorkspaceInputError(ValueError):
    """A serialized stitching workspace failed an explicit model check."""

_REGISTRATION_RESULT_FIELDS = {
    "assumptions",
    "base_matrices_by_source",
    "channel_grid_revisions",
    "components",
    "corrections",
    "edges",
    "input_pixel_fingerprints",
    "initial_layout",
    "mode",
    "mosaic_corrections",
    "mosaic_geometry",
    "pose_graph_qc",
    "registration_channel_key",
    "registration_channel_label",
    "registration_inputs",
    "rejections",
    "requested_mode",
    "search_coverage",
    "settings",
    "target_matrices_by_source",
    "tile_ids",
    "translation_acceptance_contract",
    "warnings",
}
_REGISTRATION_RUNTIME_SETTING_FIELDS = {
    "initial_layout",
    "preview_cache_mb",
    "registration_channel_key",
    "registration_channel_label",
    "registration_memory_mb",
    "resolved_worker_count",
    "worker_count",
    "worker_resolution",
}
_INITIAL_LAYOUT_CONVENIENCE_FIELDS = {
    "anchor_display_name",
    "kind",
    "label",
    "partial",
    "requires_review",
    "summary",
}
_WORKSPACE_RESULT_OVERRIDE_FIELDS = {
    "project_revision",
    "user_decision",
}
_WORKSPACE_RESOURCE_SETTING_FIELDS = {
    "chunk_depth",
    "h5j_conversion_confirmed",
    "registration_memory_mb",
    "worker_count",
}
_MATERIAL_FUSION_PARAMETER_FIELDS = {
    "custom_spacing",
    "fusion_mode",
    "interpolation",
    "output_dtype",
    "output_format",
    "padding",
    "spacing_mode",
}


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


def _workspace_integer(value, label, *, minimum=0):
    if isinstance(value, bool):
        raise InvalidStitchingWorkspaceInputError(f"{label} must be an integer.")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        result = int(value)
    elif isinstance(value, str):
        try:
            result = int(value.strip())
        except ValueError as exc:
            raise InvalidStitchingWorkspaceInputError(
                f"{label} must be an integer."
            ) from exc
    else:
        raise InvalidStitchingWorkspaceInputError(f"{label} must be an integer.")
    if result < minimum:
        raise InvalidStitchingWorkspaceInputError(
            f"{label} must be at least {minimum}."
        )
    return result


def _workspace_status(value, validator, label, default):
    try:
        return validator(default if value is None else value)
    except ValueError as exc:
        raise InvalidStitchingWorkspaceInputError(
            f"{label} is invalid."
        ) from exc


def _workspace_registration_result(value):
    if value is None or isinstance(value, StitchingRegistrationResult):
        return value
    if not isinstance(value, Mapping):
        raise InvalidStitchingWorkspaceInputError(
            "Stitching registration result must be an object or null."
        )
    registration_id = value.get("registration_id")
    if not isinstance(registration_id, str) or not registration_id.strip():
        raise InvalidStitchingWorkspaceInputError(
            "Stitching registration result requires a nonblank registration ID."
        )
    for key, validator, label, default in (
        ("execution_status", validate_execution_status, "execution status", "succeeded"),
        ("qc_status", validate_qc_status, "QC status", "not-evaluated"),
        ("user_decision", validate_user_decision, "user decision", "unapplied"),
    ):
        if key in value:
            _workspace_status(value[key], validator, f"Stitching result {label}", default)
    for key in StitchingRegistrationResult._MATRIX_MAP_FIELDS:
        if key not in value:
            continue
        matrices = value[key]
        if not isinstance(matrices, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                f"Stitching registration result {key} must be an object."
            )
        for name, matrix in matrices.items():
            try:
                invertible_affine4(matrix, f"Stitching registration result {key} {name}")
            except (TypeError, ValueError) as exc:
                raise InvalidStitchingWorkspaceInputError(
                    f"Stitching registration result {key} contains an invalid transform."
                ) from exc
    for key in ("edges", "rejections"):
        if key in value and (
            not isinstance(value[key], (list, tuple))
            or any(not isinstance(item, Mapping) for item in value[key])
        ):
            raise InvalidStitchingWorkspaceInputError(
                f"Stitching registration result {key} must be a list of objects."
            )
    return value


def _workspace_result_ids(value, label):
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str)
        or not item.strip().startswith("stitching-registration:")
        or not item.strip().partition(":")[2].strip()
        for item in value
    ):
        raise InvalidStitchingWorkspaceInputError(
            f"{label} must be a list of stable stitching result IDs."
        )
    return list(dict.fromkeys(value))


def _finite_json_value(value, field_name):
    payload = _json_value(value)
    try:
        json.dumps(payload, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise InvalidStitchingWorkspaceInputError(f"{field_name} must contain finite JSON values.") from exc
    return payload


def _validated_project_state(value, field_name="Stitching project state"):
    if not isinstance(value, Mapping):
        raise InvalidStitchingWorkspaceInputError(f"{field_name} must be a mapping.")
    payload = _finite_json_value(value, field_name)
    tiles = payload.get("tiles", [])
    if not isinstance(tiles, list):
        raise InvalidStitchingWorkspaceInputError(f"{field_name} tiles must be a list.")
    if tiles:
        if not str(payload.get("project_id") or "").strip():
            raise InvalidStitchingWorkspaceInputError(f"{field_name} requires a stable project ID.")
        schema = str(payload.get("schema") or "")
        if schema != "MADI3D_stitching_project_tree_v2":
            raise InvalidStitchingWorkspaceInputError(f"{field_name} uses unsupported schema {schema!r}.")
    tile_ids = []
    for tile in tiles:
        if not isinstance(tile, Mapping):
            raise InvalidStitchingWorkspaceInputError(f"{field_name} tile records must be mappings.")
        tile_id = str(tile.get("tile_id") or "").strip()
        if not tile_id:
            raise InvalidStitchingWorkspaceInputError(f"{field_name} tiles require stable tile IDs.")
        tile_ids.append(tile_id)
        channels = tile.get("channels", [])
        if not isinstance(channels, list):
            raise InvalidStitchingWorkspaceInputError(f"{field_name} tile channels must be a list.")
        for channel in channels:
            if not isinstance(channel, Mapping):
                raise InvalidStitchingWorkspaceInputError(f"{field_name} channel records must be mappings.")
            descriptor = channel.get("descriptor")
            if not isinstance(descriptor, Mapping):
                raise InvalidStitchingWorkspaceInputError(
                    f"{field_name} channels require source descriptors."
                )
            migration_status = str(descriptor.get("migration_status") or "")
            if migration_status != "unresolved-legacy-source":
                if not str(descriptor.get("source_id") or "").strip():
                    raise InvalidStitchingWorkspaceInputError(
                        f"{field_name} channels require authoritative SourceID."
                    )
                if not str(descriptor.get("channel_id") or "").strip():
                    raise InvalidStitchingWorkspaceInputError(
                        f"{field_name} channels require authoritative channel identity."
                    )
            pose = descriptor.get("project_capture_pose")
            if pose is not None:
                try:
                    invertible_affine4(pose, "Stitching project captured pose")
                except (TypeError, ValueError) as exc:
                    raise InvalidStitchingWorkspaceInputError(
                        f"{field_name} contains an invalid captured pose."
                    ) from exc
    if len(tile_ids) != len(set(tile_ids)):
        raise InvalidStitchingWorkspaceInputError(f"{field_name} tile IDs must be unique.")
    return payload


def _validated_pose_state(value, field_name):
    payload = _finite_json_value(value or {}, field_name)
    if not isinstance(payload, dict):
        raise InvalidStitchingWorkspaceInputError(f"{field_name} must be a mapping.")
    for key in ("before_matrices", "after_matrices", "exact_target_matrices"):
        matrices = payload.get(key, {})
        if matrices:
            if not isinstance(matrices, Mapping):
                raise InvalidStitchingWorkspaceInputError(f"{field_name} {key} must be a mapping.")
            for source_key, matrix in matrices.items():
                try:
                    invertible_affine4(
                        matrix, f"{field_name} {key} {source_key}"
                    )
                except (TypeError, ValueError) as exc:
                    raise InvalidStitchingWorkspaceInputError(
                        f"{field_name} contains an invalid transform."
                    ) from exc
    return payload


def _validated_initial_layout(value):
    payload = _finite_json_value(value or {}, "Stitching initial-placement evidence")
    if not isinstance(payload, dict):
        raise InvalidStitchingWorkspaceInputError("Stitching initial-placement evidence must be a mapping.")
    deltas = payload.get("placement_deltas", {})
    if not isinstance(deltas, Mapping):
        raise InvalidStitchingWorkspaceInputError("Stitching initial-placement deltas must be a mapping.")
    for tile_id, matrix in deltas.items():
        try:
            invertible_affine4(
                matrix, f"Stitching initial-placement delta {tile_id}"
            )
        except (TypeError, ValueError) as exc:
            raise InvalidStitchingWorkspaceInputError(
                "Stitching initial-placement evidence contains an invalid transform."
            ) from exc
    return payload


def _validated_pose_undo_stack(value):
    payload = _finite_json_value(value or [], "Stitching pose undo stack")
    if not isinstance(payload, list):
        raise InvalidStitchingWorkspaceInputError("Stitching pose undo stack must be a list.")
    for index, entry in enumerate(payload):
        if not isinstance(entry, Mapping):
            raise InvalidStitchingWorkspaceInputError("Stitching pose undo entries must be mappings.")
        matrices = entry.get("matrices", {})
        if not isinstance(matrices, Mapping):
            raise InvalidStitchingWorkspaceInputError("Stitching pose undo matrices must be a mapping.")
        for source_key, matrix in matrices.items():
            try:
                invertible_affine4(
                    matrix,
                    f"Stitching pose undo entry {index} matrix {source_key}",
                )
            except (TypeError, ValueError) as exc:
                raise InvalidStitchingWorkspaceInputError(
                    "Stitching pose undo stack contains an invalid transform."
                ) from exc
    return payload


def _finite_float(value, field_name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidStitchingWorkspaceInputError(
            f"{field_name} must be finite."
        ) from exc
    if not math.isfinite(result):
        raise InvalidStitchingWorkspaceInputError(f"{field_name} must be finite.")
    return result


def _float_vector(value, length, field_name, default):
    source = default if value is None else value
    try:
        array = np.asarray(source, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidStitchingWorkspaceInputError(
            f"{field_name} must contain {length} finite values."
        ) from exc
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise InvalidStitchingWorkspaceInputError(
            f"{field_name} must contain {length} finite values."
        )
    return tuple(float(component) for component in array)


_STITCHING_NUMERICAL_GRID_FIELDS = (
    "dimensions",
    "spacing",
    "origin",
    "direction",
    "physical_units",
)


def _stitching_numerical_grid_payload(working_geometry):
    """Return only grid fields that can change stitching calculations."""
    if not isinstance(working_geometry, Mapping) or not working_geometry:
        raise ValueError("A stitching source requires exact working geometry.")
    payload = {
        field: _json_value(working_geometry[field])
        for field in _STITCHING_NUMERICAL_GRID_FIELDS
        if field in working_geometry
    }
    if not payload:
        raise ValueError("A stitching source requires numerical grid fields.")
    return payload


def stitching_grid_mismatches(reference_geometry, candidate_geometry):
    """Return operation-relevant numerical grid fields that differ."""
    reference = _stitching_numerical_grid_payload(reference_geometry)
    candidate = _stitching_numerical_grid_payload(candidate_geometry)
    missing = object()
    return tuple(
        field
        for field in _STITCHING_NUMERICAL_GRID_FIELDS
        if reference.get(field, missing) != candidate.get(field, missing)
    )


def stitching_grid_revision(working_geometry):
    """Return a stable revision for the numerical channel grid used by stitching."""
    payload = _stitching_numerical_grid_payload(working_geometry)
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
        for field_name, label in (
            ("translation_world", "translation residual"),
            ("rotation_degrees", "rotation residual"),
            ("affine_frobenius", "affine residual"),
        ):
            value = _finite_float(getattr(self, field_name), label)
            if value < 0.0:
                raise InvalidStitchingWorkspaceInputError(
                    "Pose-graph residuals cannot be negative."
                )
            object.__setattr__(self, field_name, value)

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
            raise InvalidStitchingWorkspaceInputError("A stitching rejection requires a code.")
        if not str(self.reason).strip():
            raise InvalidStitchingWorkspaceInputError("A stitching rejection requires a reason.")
        if self.score is not None:
            object.__setattr__(
                self, "score", _finite_float(self.score, "rejected edge score")
            )
        if self.details is not None and not isinstance(self.details, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching rejection details must be an object."
            )
        details = _finite_json_value(self.details or {}, "Stitching rejection details")
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
            details=payload.get("details", {}),
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
        try:
            correction = invertible_affine4(self.correction, "Stitching edge correction")
        except (TypeError, ValueError) as exc:
            raise InvalidStitchingWorkspaceInputError(str(exc)) from exc
        object.__setattr__(
            self,
            "correction",
            correction.copy(),
        )
        object.__setattr__(
            self,
            "translation_xyz",
            _float_vector(self.translation_xyz, 3, "edge translation", (0.0,) * 3),
        )
        affine = _float_vector(self.affine_params, 12, "edge affine parameters", None)
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
        if residual_payload is not None and (
            not isinstance(residual_payload, Mapping)
            or set(residual_payload) - set(PoseGraphResidual.__dataclass_fields__)
        ):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching edge residual must contain supported residual fields."
            )
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
            affine_params=payload.get("affine_params", (0.0,) * 12),
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
        raise InvalidStitchingWorkspaceInputError(f"{field_name} must be a mapping.")
    payload = _finite_json_value(value, field_name)
    if not isinstance(payload, dict):
        raise InvalidStitchingWorkspaceInputError(f"{field_name} must be a mapping.")
    return payload


def workspace_stitching_settings(value, field_name="Stitching workspace settings"):
    """Keep material configuration and output policy, not runtime resource state."""

    payload = detached_stitching_settings(value, field_name)
    for key in _WORKSPACE_RESOURCE_SETTING_FIELDS:
        payload.pop(key, None)
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
    return np.asarray(matrix, dtype=float).tolist()


def _material_initial_layout(value):
    payload = _finite_json_value(
        value or {}, "Stitching registration initial placement"
    )
    for key in _INITIAL_LAYOUT_CONVENIENCE_FIELDS:
        payload.pop(key, None)
    placement_deltas = payload.get("placement_deltas") or {}
    current_identity_layout = (
        str(payload.get("mode") or "current") == "current"
        and not payload.get("records")
        and not payload.get("warnings")
        and not payload.get("assumptions")
        and not payload.get("base_pose_provenance")
        and all(
            np.allclose(np.asarray(matrix, dtype=float), np.eye(4), atol=0.0, rtol=0.0)
            for matrix in placement_deltas.values()
        )
    )
    return {} if current_identity_layout else payload


def _material_registration_settings(value):
    return {
        str(key): copy.deepcopy(item)
        for key, item in _finite_json_value(
            value or {}, "Stitching registration settings"
        ).items()
        if str(key) not in _REGISTRATION_RUNTIME_SETTING_FIELDS
    }


def _registration_result_payload(result):
    """Project one completed solve into its canonical scientific payload."""
    payload = StitchingRegistrationResult.from_runtime(result).to_provenance_dict()
    projected = {
        key: copy.deepcopy(payload[key])
        for key in _REGISTRATION_RESULT_FIELDS
        if key in payload
    }
    settings = _material_registration_settings(
        payload.get("settings") or {}
    )
    if settings:
        projected["settings"] = settings
    else:
        projected.pop("settings", None)
    layout = _material_initial_layout(
        payload.get("initial_layout") or payload.get("placement_evidence") or {}
    )
    if layout:
        projected["initial_layout"] = layout
    else:
        projected.pop("initial_layout", None)
    for item in projected.get("registration_inputs", ()):
        item.pop("display_name", None)
        item.pop("source_operation_ids", None)
    for item in (*projected.get("edges", ()), *projected.get("rejections", ())):
        item.pop("fixed_name", None)
        item.pop("moving_name", None)
    return _finite_json_value(projected, "Stitching registration result")


def materialize_stitching_registration(record):
    """Return the typed view of one canonical stitching-registration record."""
    record = dict(record or {})
    if record.get("operation_type") != "stitching_registration":
        raise ValueError("Scientific operation is not a stitching registration result.")
    operation_id = str(record.get("operation_id") or "")
    prefix = "stitching-registration:"
    if not operation_id.startswith(prefix) or not operation_id[len(prefix):]:
        raise ValueError("Stitching registration operation has an invalid identity.")
    payload = copy.deepcopy(dict(record.get("result") or {}))
    payload["registration_id"] = operation_id[len(prefix):]
    payload["execution_status"] = record.get("execution_status", "succeeded")
    payload["qc_status"] = record.get("qc_status", "not-evaluated")
    payload.setdefault("user_decision", "unapplied")
    algorithm = record.get("algorithm") or {}
    if isinstance(algorithm, Mapping) and algorithm.get("version"):
        payload["algorithm_version"] = str(algorithm["version"])
    return StitchingRegistrationResult.from_dict(payload)


def normalize_stitching_registration_operations(
    model, *, tolerate_unavailable=False
):
    """Replace historical full-result payloads with the current canonical form."""
    legacy_workspace_overrides = {}
    for operation_id, raw_record in list(model.operation_records.items()):
        if not isinstance(raw_record, Mapping):
            continue
        if raw_record.get("operation_type") != "stitching_registration":
            continue
        legacy_result = raw_record.get("result", {})
        try:
            if not isinstance(legacy_result, Mapping):
                raise InvalidStitchingWorkspaceInputError(
                    "Historical stitching result must be an object."
                )
            prefix = "stitching-registration:"
            if (
                not isinstance(operation_id, str)
                or not operation_id.startswith(prefix)
                or not operation_id[len(prefix):].strip()
            ):
                raise InvalidStitchingWorkspaceInputError(
                    "Historical stitching operation has an invalid identity."
                )
            candidate = dict(legacy_result)
            candidate.setdefault("registration_id", operation_id[len(prefix):])
            _workspace_registration_result(candidate)
            _workspace_status(
                raw_record.get("execution_status"),
                validate_execution_status,
                "Historical stitching execution status",
                "succeeded",
            )
            _workspace_status(
                raw_record.get("qc_status"),
                validate_qc_status,
                "Historical stitching QC status",
                "not-evaluated",
            )
            if "project_revision" in legacy_result:
                _workspace_integer(
                    legacy_result["project_revision"],
                    "Historical stitching project revision",
                )
            if "user_decision" in legacy_result:
                _workspace_status(
                    legacy_result["user_decision"],
                    validate_user_decision,
                    "Historical stitching result user decision",
                    "unapplied",
                )
            if "user_decision" in raw_record:
                _workspace_status(
                    raw_record["user_decision"],
                    validate_user_decision,
                    "Historical stitching operation user decision",
                    "unapplied",
                )
        except InvalidStitchingWorkspaceInputError:
            if not tolerate_unavailable:
                raise
            model.operation_records.pop(operation_id, None)
            model.unavailable_history_operation_ids.add(operation_id)
            continue
        overrides = {
            key: copy.deepcopy(legacy_result[key])
            for key in _WORKSPACE_RESULT_OVERRIDE_FIELDS
            if key in legacy_result
        }
        if "user_decision" in raw_record:
            overrides.setdefault(
                "user_decision", copy.deepcopy(raw_record["user_decision"])
            )
        if overrides:
            legacy_workspace_overrides[operation_id] = overrides
        try:
            typed = materialize_stitching_registration(raw_record)
        except InvalidStitchingWorkspaceInputError:
            if not tolerate_unavailable:
                raise
            model.operation_records.pop(operation_id, None)
            model.unavailable_history_operation_ids.add(operation_id)
            continue
        record = copy.deepcopy(raw_record)
        record["result"] = _registration_result_payload(typed)
        record["execution_status"] = typed.execution_status
        record["qc_status"] = typed.qc_status
        record.pop("user_decision", None)
        algorithm_version = str(typed.get("algorithm_version") or "").strip()
        if algorithm_version:
            record["algorithm"] = {
                "name": "MADI3D stitching registration",
                "version": algorithm_version,
            }
        model.operation_records[operation_id] = record
    return legacy_workspace_overrides


def retain_stitching_registration(model, result, *, capture=None):
    """Retain each solved attempt independently of later fusion and review."""
    from madi3d_app.operation_status import execution_succeeded

    typed = StitchingRegistrationResult.from_runtime(result)
    if not execution_succeeded(typed.execution_status):
        raise ValueError(
            "Only a successfully completed stitching registration can become "
            "scientific provenance."
        )
    registration_id = str(typed.get("registration_id") or "").strip()
    if not registration_id:
        raise ValueError("Completed stitching registration requires a stable identity.")
    operation_id = "stitching-registration:" + registration_id
    payload = _registration_result_payload(typed)
    algorithm_version = str(typed.get("algorithm_version") or "").strip()
    algorithm = (
        {
            "name": "MADI3D stitching registration",
            "version": algorithm_version,
        }
        if algorithm_version
        else None
    )
    existing = model.operation_records.get(operation_id)
    if existing is not None:
        if (
            existing.get("operation_type") != "stitching_registration"
            or _registration_result_payload(
                materialize_stitching_registration(existing)
            ) != payload
            or existing.get("execution_status") != typed.execution_status
            or existing.get("qc_status") != typed.qc_status
            or existing.get("algorithm") != algorithm
        ):
            raise ValueError(
                f"Stitching registration {registration_id!r} has conflicting evidence."
            )
        return operation_id

    raw_payload = typed.to_provenance_dict()
    source_operation_ids = list(dict.fromkeys(
        oid for item in raw_payload.get("registration_inputs", [])
        for oid in (item.get("source_operation_ids") or [])
    ))
    modeled_channel_ids = list(dict.fromkeys(
        str(item.get("channel_id") or "")
        for item in raw_payload.get("registration_inputs", ())
        if str(item.get("channel_id") or "")
    ))
    operation = {
        "operation_id": operation_id,
        "operation_type": "stitching_registration",
        "input_operation_ids": source_operation_ids,
        "result": payload,
        "input_backing_source_ids": list(dict.fromkeys(
            item["backing_source_id"]
            for item in raw_payload.get("registration_inputs", [])
            if item.get("backing_source_id")
        )),
        "execution_status": typed.execution_status,
        "qc_status": typed.qc_status,
    }
    if algorithm is not None:
        operation["algorithm"] = algorithm
    if modeled_channel_ids:
        if capture is None:
            raise ValueError(
                "A new modeled-volume stitching result requires its dispatch-time input capture."
            )
        captured_channel_ids = {
            str(record.get("channel_id") or "")
            for record in capture.get("direct_records", {}).values()
        }
        missing = set(modeled_channel_ids).difference(captured_channel_ids)
        if missing:
            raise ValueError(
                "Stitching dispatch capture is missing modeled channel(s): "
                + ", ".join(sorted(missing))
            )
        from madi3d_app.volume.provenance import (
            publish_captured_volume_operation,
        )
        captured_producers = {
            record.get("producing_operation_id")
            for record in capture.get("direct_records", {}).values()
            if record.get("producing_operation_id")
        }
        operation["input_operation_ids"] = [
            source_id
            for source_id in source_operation_ids
            if source_id not in captured_producers
        ]
        publish_captured_volume_operation(model, [capture], operation)
    else:
        model.publish_operation(operation)
    return operation_id


def material_stitching_fusion_parameters(options):
    """Keep only parameters that can change fused scientific output."""
    values = _finite_json_value(options or {}, "Stitching fusion parameters")
    return {
        key: copy.deepcopy(values[key])
        for key in _MATERIAL_FUSION_PARAMETER_FIELDS
        if key in values
    }


def stitching_export_operation_projection(model, manifest):
    """Bounded transport closure for output companions, never manifest state."""
    manifest = dict(manifest or {})
    inputs = list(dict.fromkeys([
        *(
            [str(manifest["registration_operation_id"])]
            if manifest.get("registration_operation_id")
            else []
        ),
        *(
            str(operation_id)
            for tiles in (manifest.get("channel_layouts") or {}).values()
            for tile in tiles
            for operation_id in tile.get("source_operation_ids", ())
            if str(operation_id).strip()
        ),
    ]))
    if not inputs:
        return []
    return [
        copy.deepcopy(model.operation_records[operation_id])
        for operation_id in model.operation_dependency_closure(inputs)
    ]


def stitching_fusion_operation_record(manifest, options, mosaic_geometry, operation_id):
    """One operation owns membership, propagation and the actual fusion layout."""
    estimation_id = manifest.get("registration_operation_id")
    evidence = {
        key: copy.deepcopy(manifest[key])
        for key in (
            "registration_operation_id",
            "fusion_pose_source",
            "channel_layouts",
            "mosaic_coordinate_space_id",
        )
        if key in manifest
    }
    input_ids = list(dict.fromkeys([
        *([estimation_id] if estimation_id else []),
        *(oid for tiles in manifest.get("channel_layouts", {}).values()
          for tile in tiles for oid in tile.get("source_operation_ids", [])),
    ]))
    fusion_parameters = material_stitching_fusion_parameters(options)
    geometry = copy.deepcopy(mosaic_geometry)
    fusion_warnings = list(geometry.get("warnings") or ())
    fusion_assumptions = list(geometry.get("assumptions") or ())
    return _finite_json_value({
        "operation_id": operation_id, "operation_type": "tile_stitching",
        "input_operation_ids": input_ids,
        "input_backing_source_ids": list(dict.fromkeys(
            tile["backing_source_id"] for tiles in manifest.get("channel_layouts", {}).values()
            for tile in tiles if tile.get("backing_source_id")
        )),
        "execution_status": "succeeded",
        "qc_status": "warning" if fusion_warnings or fusion_assumptions else "passed",
        "algorithm": {
            "name": "MADI3D tile stitching fusion",
            "version": str(manifest.get("stitching_algorithm_version") or ""),
        },
        "evidence": evidence,
        "fusion_parameters": fusion_parameters,
        "mosaic_geometry": geometry,
        "transform_application": "channel_layout_world_index_affines_baked_once",
    }, "Stitching fusion operation")


def project_workspace_results(
    workspace,
    operation_records,
    *,
    restore=False,
    legacy_result_overrides=None,
    unavailable_operation_ids=None,
    tolerate_unavailable=False,
):
    """Project bounded result references and expand them only for live UI use."""

    if not isinstance(workspace, Mapping):
        raise InvalidStitchingWorkspaceInputError(
            "Stitching workspace must be an object."
        )
    payload = copy.deepcopy(dict(workspace))
    if not payload:
        return {}
    reviews = copy.deepcopy(payload.get("result_reviews", {}))
    if not isinstance(reviews, Mapping):
        raise InvalidStitchingWorkspaceInputError(
            "Stitching workspace reviews must be an object."
        )
    reviews = dict(reviews)
    for operation_id, review in reviews.items():
        _workspace_result_ids([operation_id], "Stitching workspace review IDs")
        if not isinstance(review, Mapping) or set(review) - {"decision"}:
            raise InvalidStitchingWorkspaceInputError(
                "Stitching workspace review entries must contain only a decision."
            )
        _workspace_status(
            review.get("decision"), validate_user_decision,
            "Stitching workspace review decision", "unapplied",
        )
    active_project = payload.get("active_project")
    jobs = payload.get("jobs", [])
    if active_project is not None and not isinstance(active_project, Mapping):
        raise InvalidStitchingWorkspaceInputError(
            "Stitching active project must be an object or null."
        )
    if not isinstance(jobs, (list, tuple)) or any(
        not isinstance(owner, Mapping) for owner in jobs
    ):
        raise InvalidStitchingWorkspaceInputError(
            "Stitching workspace jobs must be a list of objects."
        )
    unavailable_operation_ids = (
        unavailable_operation_ids
        if unavailable_operation_ids is not None
        else set()
    )
    owners = [active_project, *jobs]
    for owner in owners:
        if owner is None:
            continue
        result = owner.get("registration_result")
        if result is not None and not isinstance(result, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching workspace registration result must be an object or null."
            )
        result_ids = _workspace_result_ids(
            owner.get("result_operation_ids", []),
            "Stitching workspace result references",
        )
        if result is not None:
            if "operation_id" in result:
                if set(result) - {"operation_id", "result_overrides"}:
                    raise InvalidStitchingWorkspaceInputError(
                        "Compact stitching result references contain unsupported fields."
                    )
                compact_id = result.get("operation_id")
                compact_ids = _workspace_result_ids(
                    [compact_id], "Compact stitching result references"
                )
                result_ids = list(dict.fromkeys([*compact_ids, *result_ids]))
            else:
                _workspace_registration_result(result)
            result_overrides = result.get("result_overrides", {})
            if not isinstance(result_overrides, Mapping):
                raise InvalidStitchingWorkspaceInputError(
                    "Stitching result overrides must be an object."
                )
            for key, value in result_overrides.items():
                if key not in _WORKSPACE_RESULT_OVERRIDE_FIELDS:
                    raise InvalidStitchingWorkspaceInputError(
                        f"Unsupported stitching result override: {key}."
                    )
                if key == "project_revision":
                    _workspace_integer(value, "Stitching project revision")
                else:
                    _workspace_status(
                        value, validate_user_decision,
                        "Stitching result user decision", "unapplied",
                    )
            for key in _WORKSPACE_RESULT_OVERRIDE_FIELDS.intersection(result):
                value = result[key]
                if key == "project_revision":
                    _workspace_integer(value, "Stitching project revision")
                else:
                    _workspace_status(
                        value, validate_user_decision,
                        "Stitching result user decision", "unapplied",
                    )
        if restore:
            legacy_operation_id = str(
                (result or {}).get("operation_id") or ""
            ).strip()
            if legacy_operation_id:
                result_ids = list(dict.fromkeys([
                    legacy_operation_id, *result_ids,
                ]))
            missing_result_ids = [
                operation_id for operation_id in result_ids
                if operation_id not in operation_records
            ]
            undeclared_missing = [
                operation_id for operation_id in missing_result_ids
                if operation_id not in unavailable_operation_ids
            ]
            if undeclared_missing and not tolerate_unavailable:
                raise InvalidStitchingWorkspaceInputError(
                    "Stitching workspace references an unavailable completed result."
                )
            if tolerate_unavailable:
                unavailable_operation_ids.update(missing_result_ids)

            binding_issues = [
                copy.deepcopy(issue)
                for issue in owner.get("binding_issues") or ()
                if isinstance(issue, Mapping)
                and issue.get("code") != UNAVAILABLE_STITCHING_RESULT_CODE
            ]
            if missing_result_ids:
                shown = missing_result_ids[:5]
                suffix = (
                    f" (+{len(missing_result_ids) - len(shown)} more)"
                    if len(missing_result_ids) > len(shown)
                    else ""
                )
                binding_issues.append({
                    "code": UNAVAILABLE_STITCHING_RESULT_CODE,
                    "result_operation_ids": shown,
                    "unavailable_result_count": len(missing_result_ids),
                    "message": (
                        "Historical stitching result unavailable: "
                        + ", ".join(shown)
                        + suffix
                        + ". Recalculate registration or use current visible poses."
                    ),
                })
            owner["binding_issues"] = binding_issues
            registration_operation_id = next(
                (
                    operation_id for operation_id in result_ids
                    if operation_records.get(operation_id, {}).get("operation_type")
                    == "stitching_registration"
                ),
                "",
            )
            if not registration_operation_id:
                owner.pop("registration_result", None)
                owner["result_operation_ids"] = list(dict.fromkeys(result_ids))
                continue
            record = operation_records.get(registration_operation_id)
            overrides = copy.deepcopy(
                (legacy_result_overrides or {}).get(registration_operation_id, {})
            )
            if result:
                overrides.update(copy.deepcopy(result.get("result_overrides") or {}))
                overrides.update({
                    key: copy.deepcopy(result[key])
                    for key in _WORKSPACE_RESULT_OVERRIDE_FIELDS
                    if key in result
                })
            unsupported = sorted(set(overrides) - _WORKSPACE_RESULT_OVERRIDE_FIELDS)
            if unsupported:
                raise InvalidStitchingWorkspaceInputError(
                    "Stitching workspace registration reference contains unsupported "
                    "overrides: " + ", ".join(unsupported) + "."
                )
            expanded = materialize_stitching_registration(record).to_dict()
            expanded.update(overrides)
            review = reviews.get(registration_operation_id) or {}
            if review:
                expanded["user_decision"] = validate_user_decision(
                    review.get("decision", "unapplied")
                )
            owner["registration_result"] = expanded
            owner["result_operation_ids"] = list(dict.fromkeys(result_ids or [
                registration_operation_id
            ]))
            continue

        if result and result.get("registration_id"):
            operation_id = "stitching-registration:" + str(result["registration_id"])
            record = operation_records.get(operation_id)
            if record is None:
                raise InvalidStitchingWorkspaceInputError(
                    "Stitching workspace result has no canonical registration operation."
                )
            result_ids.insert(0, operation_id)
            decision = validate_user_decision(
                result.get("user_decision", "unapplied")
            )
            if decision != "unapplied":
                prior = reviews.get(operation_id)
                review = {"decision": decision}
                if prior is not None and prior != review:
                    raise InvalidStitchingWorkspaceInputError(
                        "Stitching workspace contains conflicting review state for "
                        f"{operation_id}."
                    )
                reviews[operation_id] = review
        result_ids = list(dict.fromkeys(result_ids))
        for operation_id in result_ids:
            if (
                operation_id not in operation_records
                and operation_id not in unavailable_operation_ids
            ):
                raise InvalidStitchingWorkspaceInputError(
                    "Stitching workspace references an unavailable completed result."
                )
        owner["result_operation_ids"] = result_ids
        owner["settings"] = workspace_stitching_settings(owner.get("settings"))
        if owner.get("execution_status") == "running":
            owner["execution_status"] = "interrupted"
        owner.pop("registration_result", None)
        for field_name in (
            "binding_issues",
            "created",
            "execution_phase",
            "loaded_job_id",
            "outputs",
            "pose_undo_stack",
            "qc_status",
            "selection_descriptors",
            "user_decision",
        ):
            owner.pop(field_name, None)
    payload["schema"] = STITCHING_WORKSPACE_SCHEMA_VERSION
    for job in payload.get("jobs", []):
        job["schema"] = STITCHING_JOB_SCHEMA_VERSION
    payload["result_reviews"] = reviews
    return payload


def build_stitching_fusion_manifest(
    *,
    schema,
    algorithm_version,
    result,
    channel_sets,
    pose_source,
    operation_model=None,
):
    """Build the bounded scientific manifest shared by interactive and queued fusion."""
    channel_sets = list(channel_sets)
    result = result or {}
    pose_source = str(pose_source or "current")
    registration_operation_id = None
    if pose_source == "registered" and result.get("registration_id"):
        registration_operation_id = (
            "stitching-registration:" + str(result["registration_id"])
        )
        if operation_model is not None:
            retain_stitching_registration(operation_model, result)

    def captured_record_id(channel, tile):
        capture = channel.get("scientific_inputs") or {}
        channel_id = str(tile.get("channel_id") or "")
        refs, _origin = (capture.get("by_channel") or {}).get(
            channel_id, ((), {})
        )
        records = getattr(capture.get("model"), "scientific_records", {})
        return next(
            (
                str(reference.get("record_id") or "")
                for reference in refs
                if (records.get(str(reference.get("record_id") or "")) or {}).get(
                    "record_kind"
                )
                == "volume_revision"
            ),
            "",
        )

    channel_layouts = {
        str(channel["label"]): [
            {
                "tile_id": tile["tile_id"],
                "source_id": tile.get("source_id", ""),
                "channel_id": tile.get("channel_id", ""),
                "backing_source_id": tile.get("backing_source_id", ""),
                "input_record_id": captured_record_id(channel, tile),
                "source_operation_ids": list(tile.get("source_operation_ids", [])),
                "geometry_revision": tile.get("geometry_revision"),
                "dims_xyz": list(tile["dims"]),
                "dtype": str(tile["dtype"]),
                "world_index_affine": _matrix_payload(tile["world_affine"]),
            }
            for tile in channel["tiles"]
        ]
        for channel in channel_sets
    }
    payload = {
        "schema": str(schema),
        "stitching_algorithm_version": str(algorithm_version),
        "registration_operation_id": registration_operation_id,
        "fusion_pose_source": pose_source,
        "channel_layouts": channel_layouts,
        "mosaic_coordinate_space_id": (
            (result.get("mosaic_geometry") or {}).get("coordinate_space_id")
        ),
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
    result_operation_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.project_state = _validated_project_state(self.project_state)
        self.settings = detached_stitching_settings(
            self.settings, "Stitching project settings"
        )
        if self.registration_result is not None:
            _workspace_registration_result(self.registration_result)
        self.registration_result = (
            self.registration_result
            if isinstance(self.registration_result, StitchingRegistrationResult)
            else StitchingRegistrationResult.from_dict(self.registration_result)
            if self.registration_result is not None
            else None
        )
        if self.registration_result is not None:
            registration_id = str(
                self.registration_result.get("registration_id") or ""
            ).strip()
            if registration_id:
                self.result_operation_ids = [
                    "stitching-registration:" + registration_id,
                    *self.result_operation_ids,
                ]
        self.result_operation_ids = _workspace_result_ids(
            self.result_operation_ids, "Stitching result references"
        )
        self.applied_state = _validated_pose_state(
            self.applied_state, "Stitching applied state"
        )
        self.initial_layout = _validated_initial_layout(self.initial_layout)
        self.pose_undo_stack = _validated_pose_undo_stack(self.pose_undo_stack)
        self.loaded_job_id = (
            _workspace_integer(self.loaded_job_id, "Loaded stitching job ID")
            if self.loaded_job_id is not None else None
        )
        self.qc_status = _workspace_status(
            self.qc_status, validate_qc_status, "Stitching project QC status",
            "not-evaluated",
        )
        self.user_decision = _workspace_status(
            self.user_decision, validate_user_decision,
            "Stitching project user decision", "unapplied",
        )
        issues = _finite_json_value(
            self.binding_issues or [], "Stitching project binding issues"
        )
        if not isinstance(issues, list) or any(
            not isinstance(value, Mapping) for value in issues
        ):
            raise InvalidStitchingWorkspaceInputError("Stitching project binding issues must be a list of mappings.")
        self.binding_issues = [dict(value) for value in issues]

    def to_dict(self):
        return {
            "project_state": copy.deepcopy(self.project_state),
            "settings": workspace_stitching_settings(self.settings),
            "initial_layout": copy.deepcopy(self.initial_layout),
            "applied_state": copy.deepcopy(self.applied_state),
            "result_operation_ids": list(self.result_operation_ids),
        }

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching active project must be an object."
            )
        payload = dict(payload)
        project_state = payload.get("project_state", {})
        settings = payload.get("settings", {})
        registration_result = payload.get("registration_result")
        initial_layout = payload.get("initial_layout", {})
        applied_state = payload.get("applied_state", {})
        pose_undo_stack = payload.get("pose_undo_stack", [])
        binding_issues = payload.get("binding_issues", [])
        result_ids = payload.get("result_operation_ids", [])
        loaded_job_id = payload.get("loaded_job_id")
        if not isinstance(project_state, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching project state must be an object."
            )
        if not isinstance(settings, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching project settings must be an object."
            )
        if not isinstance(initial_layout, Mapping) or not isinstance(applied_state, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching layout and applied state must be objects."
            )
        if not isinstance(pose_undo_stack, (list, tuple)):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching pose undo stack must be a list."
            )
        if not isinstance(binding_issues, (list, tuple)):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching project binding issues must be a list."
            )
        _workspace_result_ids(result_ids, "Stitching result references")
        if loaded_job_id is not None:
            _workspace_integer(loaded_job_id, "Loaded stitching job ID")
        _workspace_registration_result(registration_result)
        registration_result = (
            StitchingRegistrationResult.from_dict(registration_result)
            if registration_result is not None else None
        )
        return cls(
            project_state=copy.deepcopy(project_state),
            settings=detached_stitching_settings(
                settings, "Stitching project settings"
            ),
            registration_result=registration_result,
            initial_layout=copy.deepcopy(initial_layout),
            applied_state=copy.deepcopy(applied_state),
            pose_undo_stack=copy.deepcopy(pose_undo_stack),
            loaded_job_id=loaded_job_id,
            qc_status=payload.get(
                "qc_status",
                registration_result.qc_status
                if registration_result is not None else "not-evaluated",
            ),
            user_decision=payload.get(
                "user_decision",
                registration_result.user_decision
                if registration_result is not None else "unapplied",
            ),
            binding_issues=copy.deepcopy(binding_issues),
            result_operation_ids=copy.deepcopy(result_ids),
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
    result_operation_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.job_id = _workspace_integer(self.job_id, "Stitching job ID", minimum=1)
        self.execution_status = _workspace_status(
            self.execution_status, validate_execution_status,
            "Stitching job execution status", "pending",
        )
        self.qc_status = _workspace_status(
            self.qc_status, validate_qc_status, "Stitching job QC status",
            "not-evaluated",
        )
        self.user_decision = _workspace_status(
            self.user_decision, validate_user_decision,
            "Stitching job user decision", "unapplied",
        )
        if not isinstance(self.settings, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching job settings must be an object."
            )
        self.settings = detached_stitching_settings(
            self.settings, "Stitching job settings"
        )
        if not isinstance(self.project_state, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching job project state must be an object."
            )
        if self.registration_result is not None:
            _workspace_registration_result(self.registration_result)
        self.registration_result = (
            self.registration_result
            if isinstance(self.registration_result, StitchingRegistrationResult)
            else StitchingRegistrationResult.from_dict(self.registration_result)
            if self.registration_result is not None
            else None
        )
        if self.registration_result is not None:
            registration_id = str(
                self.registration_result.get("registration_id") or ""
            ).strip()
            if registration_id:
                self.result_operation_ids = [
                    "stitching-registration:" + registration_id,
                    *self.result_operation_ids,
                ]
        self.result_operation_ids = _workspace_result_ids(
            self.result_operation_ids, "Stitching result references"
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
            raise InvalidStitchingWorkspaceInputError("Stitching job binding issues must be a list of mappings.")
        self.binding_issues = [dict(value) for value in issues]

    def physical_grid_dependency_record(self) -> dict[str, Any]:
        """Query frozen tile inputs without serializing runtime status or results."""
        return {
            "job_id": str(self.job_id),
            "name": str(self.name),
            "input_dependencies": tuple(
                physical_grid_input_reference(channel.get("descriptor", {}))
                for tile in self.project_state.get("tiles", ())
                for channel in tile.get("channels", ())
            ),
        }

    def to_dict(self):
        persisted_status = (
            "interrupted" if self.execution_status == "running"
            else self.execution_status
        )
        if persisted_status == "succeeded" and not self.result_operation_ids:
            raise InvalidStitchingWorkspaceInputError(
                "A completed stitching job must reference its result operation."
            )
        return {
            "schema": STITCHING_JOB_SCHEMA_VERSION,
            "job_id": self.job_id,
            "name": str(self.name),
            "registration_mode": str(self.registration_mode),
            "settings": workspace_stitching_settings(self.settings),
            "project_state": _json_value(self.project_state),
            "initial_layout": copy.deepcopy(self.initial_layout),
            "applied_state": _json_value(self.applied_state),
            "execution_status": persisted_status,
            "error": str(self.error)[-4000:],
            "result_operation_ids": list(self.result_operation_ids),
        }

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching job must be an object."
            )
        payload = dict(payload)
        schema_value = payload.get("schema")
        schema = str(
            STITCHING_JOB_SCHEMA_VERSION
            if schema_value is None else schema_value
        )
        if schema not in {
            STITCHING_JOB_SCHEMA_VERSION, *_LEGACY_STITCHING_JOB_SCHEMAS,
        }:
            raise InvalidStitchingWorkspaceInputError(f"Unsupported stitching job schema: {schema}")
        if "job_id" not in payload:
            raise InvalidStitchingWorkspaceInputError(
                "Stitching jobs require an explicit job ID."
            )
        job_id = _workspace_integer(payload["job_id"], "Stitching job ID", minimum=1)
        project_state = payload.get("project_state", {})
        settings = payload.get("settings", {})
        result = payload.get("registration_result")
        initial_layout = payload.get("initial_layout", {})
        applied_state = payload.get("applied_state", {})
        pose_undo_stack = payload.get("pose_undo_stack", [])
        selection_descriptors = payload.get("selection_descriptors", [])
        binding_issues = payload.get("binding_issues", [])
        result_ids = payload.get("result_operation_ids", [])
        outputs = payload.get("outputs", [])
        if not isinstance(project_state, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching job project state must be an object."
            )
        if not isinstance(settings, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching job settings must be an object."
            )
        if not isinstance(initial_layout, Mapping) or not isinstance(applied_state, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching job layout and applied state must be objects."
            )
        for value, label in (
            (pose_undo_stack, "Stitching job pose undo stack"),
            (selection_descriptors, "Stitching job selections"),
            (binding_issues, "Stitching job binding issues"),
            (outputs, "Stitching job outputs"),
        ):
            if not isinstance(value, (list, tuple)):
                raise InvalidStitchingWorkspaceInputError(f"{label} must be a list.")
        _workspace_result_ids(result_ids, "Stitching result references")
        _workspace_registration_result(result)
        execution_status = payload.get("execution_status", "pending")
        if execution_status == "running":
            execution_status = "interrupted"
        execution_status = _workspace_status(
            execution_status, validate_execution_status,
            "Stitching job execution status", "pending",
        )
        qc_status = _workspace_status(
            payload.get("qc_status"), validate_qc_status,
            "Stitching job QC status",
            result.get("qc_status", "not-evaluated")
            if isinstance(result, Mapping) else "not-evaluated",
        )
        user_decision = _workspace_status(
            payload.get("user_decision"), validate_user_decision,
            "Stitching job user decision",
            result.get("user_decision", "unapplied")
            if isinstance(result, Mapping) else "unapplied",
        )
        registration_result = (
            StitchingRegistrationResult.from_dict(result)
            if result is not None else None
        )
        return cls(
            job_id=job_id,
            name=payload.get("name", "Stitching job"),
            registration_mode=payload.get("registration_mode", "needs_review"),
            settings=detached_stitching_settings(
                settings, "Stitching job settings"
            ),
            project_state=copy.deepcopy(project_state),
            registration_result=registration_result,
            initial_layout=copy.deepcopy(initial_layout),
            applied_state=copy.deepcopy(applied_state),
            pose_undo_stack=copy.deepcopy(pose_undo_stack),
            selection_descriptors=copy.deepcopy(selection_descriptors),
            execution_status=execution_status,
            qc_status=qc_status,
            user_decision=user_decision,
            execution_phase=payload.get("execution_phase", ""),
            outputs=[str(path) for path in outputs],
            error=payload.get("error", ""),
            binding_issues=copy.deepcopy(binding_issues),
            created=payload.get("created", ""),
            result_operation_ids=copy.deepcopy(result_ids),
        )


@dataclass
class StitchingWorkspaceState:
    """Authoritative active stitching project plus its ordered execution queue."""

    active_project: StitchingProjectState | None = None
    jobs: list[StitchingJob] = field(default_factory=list)
    job_counter: int = 0
    result_reviews: dict[str, dict[str, Any]] = field(default_factory=dict)

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
        reviews = _finite_json_value(
            self.result_reviews or {}, "Stitching workspace reviews"
        )
        if not isinstance(reviews, dict):
            raise InvalidStitchingWorkspaceInputError("Stitching workspace reviews must be keyed by operation ID.")
        for owner in [self.active_project, *self.jobs]:
            if owner is None or owner.registration_result is None:
                continue
            operation_id = next((
                value for value in owner.result_operation_ids
                if value.startswith("stitching-registration:")
            ), "")
            decision = owner.registration_result.user_decision
            if operation_id and decision != "unapplied":
                review = {"decision": decision}
                if operation_id in reviews and reviews[operation_id] != review:
                    raise InvalidStitchingWorkspaceInputError(
                        "Stitching workspace contains conflicting review state for "
                        f"{operation_id}."
                    )
                reviews[operation_id] = review
        for operation_id, review in reviews.items():
            if (
                not isinstance(operation_id, str)
                or not operation_id.startswith("stitching-registration:")
                or not operation_id.partition(":")[2].strip()
                or not isinstance(review, Mapping)
            ):
                raise InvalidStitchingWorkspaceInputError("Stitching workspace review entries are invalid.")
            unknown = set(review) - {"decision"}
            if unknown:
                raise InvalidStitchingWorkspaceInputError(
                    "Stitching workspace review contains unsupported fields: "
                    + ", ".join(sorted(unknown))
                    + "."
                )
            review["decision"] = _workspace_status(
                review.get("decision"), validate_user_decision,
                "Stitching workspace review decision", "unapplied",
            )
        referenced_result_ids = {
            operation_id
            for owner in [self.active_project, *self.jobs]
            if owner is not None
            for operation_id in owner.result_operation_ids
        }
        orphan_reviews = sorted(set(reviews) - referenced_result_ids)
        if orphan_reviews:
            raise InvalidStitchingWorkspaceInputError(
                "Stitching workspace reviews refer to unlisted results: "
                + ", ".join(orphan_reviews)
                + "."
            )
        self.result_reviews = reviews
        job_ids = [job.job_id for job in self.jobs]
        if len(job_ids) != len(set(job_ids)):
            raise InvalidStitchingWorkspaceInputError("Stitching workspace job IDs must be unique.")
        self.job_counter = _workspace_integer(
            self.job_counter, "Stitching workspace job counter"
        )
        if self.job_counter < max(job_ids, default=0):
            raise InvalidStitchingWorkspaceInputError(
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
            raise InvalidStitchingWorkspaceInputError(
                "The active stitching project refers to a missing queued job."
            )

    @property
    def is_empty(self):
        return self.active_project is None and not self.jobs and not self.result_reviews

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
            "result_reviews": copy.deepcopy(self.result_reviews),
        }

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching workspace must be an object."
            )
        payload = dict(payload)
        if not payload:
            return cls()
        schema_value = payload.get("schema")
        schema = str(
            STITCHING_WORKSPACE_SCHEMA_VERSION
            if schema_value is None else schema_value
        )
        if schema not in {
            STITCHING_WORKSPACE_SCHEMA_VERSION,
            *_LEGACY_STITCHING_WORKSPACE_SCHEMAS,
        }:
            raise InvalidStitchingWorkspaceInputError(f"Unsupported stitching workspace schema: {schema}")
        active_payload = payload.get("active_project")
        jobs_payload = payload.get("jobs", [])
        job_counter = payload.get("job_counter", 0)
        result_reviews = payload.get("result_reviews", {})
        if active_payload is not None and not isinstance(active_payload, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching active project must be an object or null."
            )
        if not isinstance(jobs_payload, (list, tuple)):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching workspace jobs must be a list."
            )
        _workspace_integer(job_counter, "Stitching workspace job counter")
        if not isinstance(result_reviews, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching workspace reviews must be an object."
            )
        return cls(
            active_project=(
                StitchingProjectState.from_dict(active_payload)
                if active_payload is not None
                else None
            ),
            jobs=[StitchingJob.from_dict(job) for job in jobs_payload],
            job_counter=job_counter,
            result_reviews=copy.deepcopy(dict(result_reviews)),
        )


def stage_stitching_project_state(raw, volume_sources, *, report_loss):
    """Project recoverable workspace state against available result owners."""
    legacy_result_overrides = normalize_stitching_registration_operations(
        volume_sources, tolerate_unavailable=True
    )
    if raw is None or (isinstance(raw, Mapping) and not raw):
        return {}
    if not isinstance(raw, Mapping):
        report_loss(
            "invalid-stitching-root", "stitching_workspace",
            "The stitching workspace was omitted; completed scientific results remain available.",
        )
        return {}
    schema = raw.get("schema")
    if schema not in {
        STITCHING_WORKSPACE_SCHEMA_VERSION, *_LEGACY_STITCHING_WORKSPACE_SCHEMAS
    }:
        report_loss(
            "invalid-stitching-schema", "stitching_workspace.schema",
            "The stitching workspace was omitted; completed scientific results remain available.",
        )
        return {}
    if set(raw) - {
        "schema", "active_project", "jobs", "job_counter", "result_reviews",
    }:
        report_loss(
            "unsupported-stitching-fields", "stitching_workspace",
            "The stitching workspace was omitted because it contains unsupported fields.",
        )
        return {}
    raw_reviews = raw.get("result_reviews", {})
    if not isinstance(raw_reviews, Mapping):
        report_loss(
            "invalid-stitching-reviews", "stitching_workspace.result_reviews",
            "The stitching workspace was omitted because its review state was malformed.",
        )
        return {}

    raw_jobs = raw.get("jobs", ())
    if not isinstance(raw_jobs, (list, tuple)):
        raw_jobs = ()
        report_loss(
            "invalid-stitching-job-list", "stitching_workspace.jobs",
            "Queued stitching jobs were omitted because their list shape was invalid.",
        )

    def project_owner(owner, *, active_project):
        if not isinstance(owner, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching workspace owner must be an object."
            )
        if not active_project and owner.get("schema") not in (
            None,
            STITCHING_JOB_SCHEMA_VERSION,
            *_LEGACY_STITCHING_JOB_SCHEMAS,
        ):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching job schema is unsupported."
            )
        projected_workspace = project_workspace_results(
            {
                "schema": schema,
                "active_project": owner if active_project else None,
                "jobs": [] if active_project else [owner],
                "job_counter": raw.get("job_counter", 0),
                "result_reviews": {},
            },
            volume_sources.operation_records,
            restore=True,
            legacy_result_overrides=legacy_result_overrides,
            unavailable_operation_ids=set(
                volume_sources.unavailable_history_operation_ids
            ),
            tolerate_unavailable=True,
        )
        projected_owner = (
            projected_workspace.get("active_project")
            if active_project else projected_workspace["jobs"][0]
        )
        if active_project:
            StitchingProjectState.from_dict(projected_owner)
        else:
            validated_job = StitchingJob.from_dict(projected_owner)
            projected_owner["execution_status"] = validated_job.execution_status
        return projected_owner

    active = None
    if raw.get("active_project") is not None:
        try:
            active = project_owner(raw["active_project"], active_project=True)
        except InvalidStitchingWorkspaceInputError:
            report_loss(
                "invalid-stitching-active-project",
                "stitching_workspace.active_project",
                "The active stitching project was omitted; valid queued jobs were retained.",
            )
    jobs = []
    for index, value in enumerate(raw_jobs):
        try:
            jobs.append(project_owner(value, active_project=False))
        except InvalidStitchingWorkspaceInputError:
            report_loss(
                "invalid-stitching-job", f"stitching_workspace.jobs[{index}]",
                "This queued stitching job was omitted; valid siblings were retained.",
            )

    candidate = {
        "schema": schema,
        "active_project": active,
        "jobs": jobs,
        "job_counter": raw.get("job_counter", 0),
        "result_reviews": copy.deepcopy(dict(raw_reviews)),
    }
    try:
        projected = project_workspace_results(
            candidate,
            volume_sources.operation_records,
            restore=True,
            legacy_result_overrides=legacy_result_overrides,
            unavailable_operation_ids=set(
                volume_sources.unavailable_history_operation_ids
            ),
            tolerate_unavailable=True,
        )
        normalized_state = StitchingWorkspaceState.from_dict(projected)
        normalized = projected if not normalized_state.is_empty else {}
    except InvalidStitchingWorkspaceInputError:
        report_loss(
            "invalid-stitching-relationships", "stitching_workspace",
            "The stitching workspace was omitted because result references, reviews, or queued-job relationships could not be reconciled.",
        )
        return {}

    available_operations = volume_sources.operation_records
    for owner_name, owner in (
        [("active_project", normalized.get("active_project"))]
        + [
            (f"jobs[{index}]", job)
            for index, job in enumerate(normalized.get("jobs", ()))
        ]
    ):
        if owner is None:
            continue
        for operation_id in owner.get("result_operation_ids", ()):
            if operation_id not in available_operations:
                report_loss(
                    "unavailable-stitching-result",
                    f"stitching_workspace.{owner_name}.result_operation_ids",
                    "The saved registration result was unavailable; its reference and current project state were retained.",
                    approval_required=False,
                    source_protection_required=False,
                )
    return normalized

__all__ = [
    "PoseGraphResidual",
    "STITCHING_JOB_SCHEMA_VERSION",
    "STITCHING_WORKSPACE_SCHEMA_VERSION",
    "UNAVAILABLE_STITCHING_RESULT_CODE",
    "StitchingEdgeResult",
    "StitchingJob",
    "detached_stitching_settings",
    "workspace_stitching_settings",
    "StitchingMosaicGeometry",
    "StitchingProjectState",
    "StitchingRegistrationResult",
    "StitchingRejection",
    "StitchingWorkspaceState",
    "build_stitching_fusion_manifest",
    "material_stitching_fusion_parameters",
    "materialize_stitching_registration",
    "normalize_stitching_registration_operations",
    "stage_stitching_project_state",
    "retain_stitching_registration",
    "stitching_export_operation_projection",
    "stitching_fusion_operation_record",
    "stitching_grid_mismatches",
    "stitching_grid_revision",
    "stitching_geometry_status",
]
