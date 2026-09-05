"""GUI-independent registration jobs, transforms, landmarks, logs, grids, and QC."""
from __future__ import annotations

import copy
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from madi3d_app.operation_status import (
    aggregate_qc_status,
    execution_status as validate_execution_status,
    execution_succeeded,
    qc_status as validate_qc_status,
    user_decision as validate_user_decision,
)
from madi3d_app.integrations.cmtk.registration import (
    persist_artifact_bundle,
    verify_artifact_bundle,
)
from madi3d_app.volume.geometry import (
    affine_matrix4 as _matrix4,
    direction_matrix3,
    finite_tuple3,
    grid_affine_from_components,
    general_grid_affine_from_components,
    invertible_affine4,
)

REGISTRATION_CHAIN_SCHEMA_VERSION = 9
REGISTRATION_ALGORITHM_VERSION = "staged-registration-outcomes-v3"


def _plain_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return copy.deepcopy(value)


def _canonical_local_geometry_payload(payload, label="registration local geometry"):
    """Validate one MADI-owned local grid and return its canonical encoding."""
    normalized = _plain_value(dict(payload or {}))
    if not normalized:
        return {}
    dimensions = finite_tuple3(
        normalized.get("dims_xyz"), f"{label} dimensions", positive=True, integer=True
    )
    origin = finite_tuple3(normalized.get("origin"), f"{label} origin")
    spacing = finite_tuple3(
        normalized.get("spacing"), f"{label} spacing", positive=True
    )
    try:
        serialized_direction = np.asarray(normalized.get("direction"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} direction must contain exactly nine numeric values."
        ) from exc
    if serialized_direction.size != 9:
        raise ValueError(
            f"{label} direction must contain exactly nine numeric values."
        )
    direction = direction_matrix3(serialized_direction.reshape(3, 3))
    general_grid_affine_from_components(origin, spacing, direction)
    normalized.update({
        "dims_xyz": list(dimensions),
        "origin": list(origin),
        "spacing": list(spacing),
        "direction": direction.tolist(),
    })
    return normalized


def _canonical_local_geometry_records(records, label):
    normalized = []
    for index, value in enumerate(records or []):
        record = _plain_value(dict(value or {}))
        if record.get("local_geometry"):
            record["local_geometry"] = _canonical_local_geometry_payload(
                record["local_geometry"], f"{label} {index + 1} local geometry"
            )
        normalized.append(record)
    return normalized


@dataclass(frozen=True)
class RegistrationLandmarkPair:
    pair_id: str
    name: str
    reference_world: tuple[float, float, float]
    moving_world: tuple[float, float, float]
    weight: float = 1.0

    def __post_init__(self):
        object.__setattr__(
            self,
            "reference_world",
            finite_tuple3(self.reference_world, "Reference landmark"),
        )
        object.__setattr__(
            self,
            "moving_world",
            finite_tuple3(self.moving_world, "Moving landmark"),
        )
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("Landmark weight must be finite and positive.")
        object.__setattr__(self, "weight", weight)

    @classmethod
    def from_dict(cls, payload):
        return cls(
            pair_id=str(payload.get("pair_id") or ""),
            name=str(payload.get("name") or "Landmark"),
            reference_world=payload.get("reference_world"),
            moving_world=payload.get("moving_world"),
            weight=payload.get("weight", 1.0),
        )

    def to_dict(self):
        return {
            "pair_id": self.pair_id,
            "name": self.name,
            "reference_world": list(self.reference_world),
            "moving_world": list(self.moving_world),
            "weight": self.weight,
        }


@dataclass(frozen=True)
class RegistrationLandmarkSet:
    dataset_id: str
    pairs: tuple[RegistrationLandmarkPair, ...] = ()

    @classmethod
    def from_records(cls, dataset_id, records):
        return cls(
            dataset_id=str(dataset_id),
            pairs=tuple(RegistrationLandmarkPair.from_dict(item) for item in records),
        )

    def to_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


class RegistrationSettings(dict):
    """Serializable settings mapping with typed landmark access."""

    def __init__(self, values=None):
        super().__init__(_plain_value(dict(values or {})))

    @classmethod
    def from_dict(cls, payload):
        return cls(payload)

    def landmark_sets(self):
        return tuple(
            RegistrationLandmarkSet.from_records(dataset_id, records)
            for dataset_id, records in dict(self.get("landmarks_by_dataset") or {}).items()
        )

    def to_dict(self):
        return _plain_value(self)


class RegistrationOutputGrid(dict):
    """Validated fixed/moving working-grid definition."""

    def __init__(self, values=None):
        payload = _plain_value(dict(values or {}))
        self._validate_domain(payload, "registration output grid")
        for name in ("fixed", "moving"):
            nested = payload.get(name)
            if nested:
                self._validate_domain(nested, f"{name} registration grid")
        super().__init__(payload)

    @staticmethod
    def _validate_domain(payload, label):
        if not payload:
            return
        required = ("origin", "spacing", "dims_xyz")
        if not all(key in payload for key in required):
            raise ValueError(f"{label} is missing origin, spacing, or dimensions.")
        origin = finite_tuple3(payload["origin"], f"{label} origin")
        spacing = finite_tuple3(payload["spacing"], f"{label} spacing", positive=True)
        finite_tuple3(payload["dims_xyz"], f"{label} dimensions", positive=True, integer=True)
        direction = np.asarray(payload.get("direction", np.eye(3)), dtype=float).reshape(3, 3)
        direction = direction_matrix3(direction)
        general_grid_affine_from_components(origin, spacing, direction)

    @classmethod
    def from_dict(cls, payload):
        return cls(payload)

    def to_dict(self):
        return _plain_value(self)


@dataclass(frozen=True)
class RegistrationLogEntry:
    level: str
    message: str
    details: str = ""
    source: str = "registration"
    created: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    @classmethod
    def from_dict(cls, payload):
        return cls(
            level=str(payload.get("level") or "INFO"),
            message=str(payload.get("message") or ""),
            details=str(payload.get("details") or ""),
            source=str(payload.get("source") or "registration"),
            created=str(payload.get("created") or ""),
        )

    def to_dict(self):
        return {
            "level": self.level,
            "message": self.message,
            "details": self.details,
            "source": self.source,
            "created": self.created,
        }


@dataclass(frozen=True)
class RegistrationQCResult:
    stage: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "status", validate_qc_status(self.status))
        object.__setattr__(self, "metrics", _plain_value(self.metrics))
        object.__setattr__(self, "thresholds", _plain_value(self.thresholds))

    @classmethod
    def from_stage(cls, stage):
        details = dict(stage.details or {})
        metrics = {
            "metric_value": stage.metric_value,
            "ncc": stage.ncc,
            "iterations": int(stage.iterations),
        }
        for key, value in details.items():
            lowered = str(key).lower()
            if (
                "threshold" not in lowered
                and any(token in lowered for token in ("qc", "similarity", "sanity", "overlap"))
            ):
                metrics[str(key)] = _plain_value(value)
        thresholds = _plain_value(details.get("qc_thresholds") or {})
        thresholds.update({
            str(key): _plain_value(value)
            for key, value in details.items()
            if key != "qc_thresholds"
            and ("threshold" in str(key).lower() or "limit" in str(key).lower())
        })
        warnings = details.get("quality_warnings")
        if warnings is None and isinstance(details.get("deformation_qc"), dict):
            warnings = details["deformation_qc"].get("warnings")
        return cls(
            stage=str(stage.name),
            status=str(stage.qc_status),
            metrics=_plain_value(metrics),
            thresholds=thresholds,
            warnings=tuple(str(item) for item in (warnings or ())),
        )

    @classmethod
    def from_dict(cls, payload):
        return cls(
            stage=str(payload.get("stage") or "Stage"),
            status=str(payload.get("status") or "not-evaluated"),
            metrics=_plain_value(payload.get("metrics") or {}),
            thresholds=_plain_value(payload.get("thresholds") or {}),
            warnings=tuple(str(item) for item in payload.get("warnings") or ()),
        )

    def to_dict(self):
        return {
            "stage": self.stage,
            "status": self.status,
            "metrics": _plain_value(self.metrics),
            "thresholds": _plain_value(self.thresholds),
            "warnings": list(self.warnings),
        }

# -----------------------------------------------------------------------------
# Serializable registration results
# -----------------------------------------------------------------------------

@dataclass
class TransformStageResult:
    name: str
    kind: str
    cumulative_moving_to_fixed: list[list[float]]
    incremental_moving_to_fixed: list[list[float]]
    metric_value: Optional[float] = None
    ncc: Optional[float] = None
    iterations: int = 0
    stop_condition: str = ""
    execution_status: str = "succeeded"
    qc_status: str = "not-evaluated"
    user_decision: str = "unapplied"
    details: dict[str, Any] = field(default_factory=dict)
    qc: Optional[RegistrationQCResult] = None
    logs: list[RegistrationLogEntry] = field(default_factory=list)

    def __post_init__(self):
        self.execution_status = validate_execution_status(self.execution_status)
        self.qc_status = validate_qc_status(self.qc_status)
        self.user_decision = validate_user_decision(self.user_decision)
        self.cumulative_moving_to_fixed = invertible_affine4(
            self.cumulative_moving_to_fixed, "Cumulative registration transform"
        ).tolist()
        self.incremental_moving_to_fixed = invertible_affine4(
            self.incremental_moving_to_fixed, "Incremental registration transform"
        ).tolist()
        if self.qc is None:
            self.qc = RegistrationQCResult.from_stage(self)
        elif not isinstance(self.qc, RegistrationQCResult):
            self.qc = RegistrationQCResult.from_dict(self.qc)
        self.logs = [
            item if isinstance(item, RegistrationLogEntry) else RegistrationLogEntry.from_dict(item)
            for item in self.logs
        ]

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "cumulative_moving_to_fixed": self.cumulative_moving_to_fixed,
            "incremental_moving_to_fixed": self.incremental_moving_to_fixed,
            "metric_value": self.metric_value,
            "ncc": self.ncc,
            "iterations": int(self.iterations),
            "stop_condition": self.stop_condition,
            "execution_status": self.execution_status,
            "qc_status": self.qc_status,
            "user_decision": self.user_decision,
            "details": copy.deepcopy(self.details),
            "qc": self.qc.to_dict() if self.qc else None,
            "logs": [entry.to_dict() for entry in self.logs],
        }

    @classmethod
    def from_dict(cls, raw):
        return cls(
            name=str(raw.get("name") or "Stage"),
            kind=str(raw.get("kind") or "unknown"),
            cumulative_moving_to_fixed=copy.deepcopy(
                raw.get("cumulative_moving_to_fixed") or np.eye(4).tolist()
            ),
            incremental_moving_to_fixed=copy.deepcopy(
                raw.get("incremental_moving_to_fixed") or np.eye(4).tolist()
            ),
            metric_value=raw.get("metric_value"),
            ncc=raw.get("ncc"),
            iterations=int(raw.get("iterations") or 0),
            stop_condition=str(raw.get("stop_condition") or ""),
            execution_status=str(raw.get("execution_status") or "pending"),
            qc_status=str(raw.get("qc_status") or "not-evaluated"),
            user_decision=str(raw.get("user_decision") or "unapplied"),
            details=copy.deepcopy(raw.get("details") or {}),
            qc=raw.get("qc"),
            logs=copy.deepcopy(raw.get("logs") or []),
        )


@dataclass
class RegistrationTransformChain:
    registration_id: str
    source_dataset_id: str
    target_dataset_id: str
    source_space_uid: str
    target_space_uid: str
    source_name: str
    target_name: str
    source_actor_matrix: list[list[float]]
    target_actor_matrix: list[list[float]]
    registration_grid: RegistrationOutputGrid
    working_space: dict[str, Any] = field(default_factory=dict)
    source_geometry: dict[str, Any] = field(default_factory=dict)
    target_geometry: dict[str, Any] = field(default_factory=dict)
    source_descriptor: dict[str, Any] = field(default_factory=dict)
    target_descriptor: dict[str, Any] = field(default_factory=dict)
    stages: list[TransformStageResult] = field(default_factory=list)
    settings: RegistrationSettings = field(default_factory=RegistrationSettings)
    attachments: list[dict[str, Any]] = field(default_factory=list)
    reformat_volumes: list[dict[str, Any]] = field(default_factory=list)  # compatibility: outputs selected when calculated
    source_volumes: list[dict[str, Any]] = field(default_factory=list)  # every moving-group volume + captured pose
    # CMTK is the sole nonlinear engine; its native transform bundle is authoritative.
    deformation_model: dict[str, Any] = field(default_factory=dict)
    landmarks: list[RegistrationLandmarkSet] = field(default_factory=list)
    qc_results: list[RegistrationQCResult] = field(default_factory=list)
    logs: list[RegistrationLogEntry] = field(default_factory=list)
    execution_status: str = "succeeded"
    qc_status: str = "not-evaluated"
    user_decision: str = "unapplied"
    algorithm_version: str = REGISTRATION_ALGORITHM_VERSION
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __post_init__(self):
        self.execution_status = validate_execution_status(self.execution_status)
        self.user_decision = validate_user_decision(self.user_decision)
        self.source_actor_matrix = _matrix4(
            self.source_actor_matrix, "Source actor matrix"
        ).tolist()
        self.target_actor_matrix = _matrix4(
            self.target_actor_matrix, "Target actor matrix"
        ).tolist()
        self.registration_grid = (
            self.registration_grid
            if isinstance(self.registration_grid, RegistrationOutputGrid)
            else RegistrationOutputGrid.from_dict(self.registration_grid)
        )
        self.working_space = _plain_value(dict(self.working_space or {}))
        for key in ("source_to_operation", "target_to_operation"):
            if self.working_space.get(key) is not None:
                self.working_space[key] = invertible_affine4(
                    self.working_space[key], f"Registration working-space {key}"
                ).tolist()
        reformat_target = dict(self.working_space.get("reformat_target") or {})
        for key, label in (
            ("source_geometry", "Reformat target source geometry"),
            ("normalized_geometry", "Reformat target normalized geometry"),
        ):
            if reformat_target.get(key):
                reformat_target[key] = _canonical_local_geometry_payload(
                    reformat_target[key], label
                )
        if reformat_target:
            self.working_space["reformat_target"] = reformat_target
        self.source_geometry = _canonical_local_geometry_payload(
            self.source_geometry, "Registration source geometry"
        )
        self.target_geometry = _canonical_local_geometry_payload(
            self.target_geometry, "Registration target geometry"
        )
        self.source_volumes = _canonical_local_geometry_records(
            self.source_volumes, "Registration source volume"
        )
        self.reformat_volumes = _canonical_local_geometry_records(
            self.reformat_volumes, "Registration Reformat volume"
        )
        self.settings = (
            self.settings
            if isinstance(self.settings, RegistrationSettings)
            else RegistrationSettings.from_dict(self.settings)
        )
        self.stages = [
            stage if isinstance(stage, TransformStageResult) else TransformStageResult.from_dict(stage)
            for stage in self.stages
        ]
        failed_stages = [
            stage for stage in self.stages
            if str(stage.kind or "").lower() != "initial"
            and not execution_succeeded(stage.execution_status)
        ]
        if failed_stages:
            self.execution_status = "failed"
        stage_qc = [stage.qc_status for stage in self.stages]
        self.qc_status = validate_qc_status(
            self.qc_status
            if self.qc_status != "not-evaluated" or not stage_qc
            else aggregate_qc_status(stage_qc)
        )
        self.landmarks = [
            item
            if isinstance(item, RegistrationLandmarkSet)
            else RegistrationLandmarkSet.from_records(item.get("dataset_id"), item.get("pairs") or [])
            for item in (self.landmarks or self.settings.landmark_sets())
        ]
        self.qc_results = [
            item if isinstance(item, RegistrationQCResult) else RegistrationQCResult.from_dict(item)
            for item in (self.qc_results or [stage.qc for stage in self.stages if stage.qc])
        ]
        self.logs = [
            item if isinstance(item, RegistrationLogEntry) else RegistrationLogEntry.from_dict(item)
            for item in self.logs
        ]

    def final_linear_matrix(self):
        for stage in reversed(self.stages):
            if execution_succeeded(stage.execution_status):
                return _matrix4(stage.cumulative_moving_to_fixed)
        return np.eye(4, dtype=float)

    def has_deformation(self):
        return str((self.deformation_model or {}).get("type") or "").lower() == "cmtk_warp"

    def to_dict(self, include_field_metadata=True):
        payload = {
            "format": "MADI3D Registration Transform Chain",
            "format_version": REGISTRATION_CHAIN_SCHEMA_VERSION,
            "registration_schema_version": REGISTRATION_CHAIN_SCHEMA_VERSION,
            "registration_algorithm_version": str(self.algorithm_version),
            "registration_id": self.registration_id,
            "source_dataset_id": self.source_dataset_id,
            "target_dataset_id": self.target_dataset_id,
            "direction": "moving_to_reference",
            "source_space_uid": self.source_space_uid,
            "target_space_uid": self.target_space_uid,
            "source_name": self.source_name,
            "target_name": self.target_name,
            "execution_status": self.execution_status,
            "qc_status": self.qc_status,
            "user_decision": self.user_decision,
            "source_actor_matrix": self.source_actor_matrix,
            "target_actor_matrix": self.target_actor_matrix,
            "working_space": copy.deepcopy(self.working_space),
            "source_geometry": copy.deepcopy(self.source_geometry),
            "target_geometry": copy.deepcopy(self.target_geometry),
            "source_descriptor": copy.deepcopy(self.source_descriptor),
            "target_descriptor": copy.deepcopy(self.target_descriptor),
            "registration_grid": self.registration_grid.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
            "settings": self.settings.to_dict(),
            "attachments": copy.deepcopy(self.attachments),
            "reformat_volumes": copy.deepcopy(self.reformat_volumes),
            "source_volumes": copy.deepcopy(self.source_volumes),
            "landmarks": [item.to_dict() for item in self.landmarks],
            "qc_results": [item.to_dict() for item in self.qc_results],
            "logs": [item.to_dict() for item in self.logs],
            "created": self.created,
            "has_deformation": self.has_deformation(),
        }
        if include_field_metadata and self.has_deformation():
            payload["deformation"] = {
                "model": copy.deepcopy(self.deformation_model),
                "grid": self.registration_grid.to_dict(),
                "storage": "cmtk_xform_bundle",
            }
        return payload

    def export(self, json_path):
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict(include_field_metadata=True)
        model_type = str((self.deformation_model or {}).get("type") or "").lower()
        if model_type == "cmtk_warp":
            model = dict(self.deformation_model or {})
            source_bundle = str(model.get("workspace") or model.get("artifact_bundle_resolved") or "").strip()
            if not source_bundle:
                raise RuntimeError(
                    "The CMTK nonlinear result has no accessible transform bundle. "
                    "Reload the original registration project or recalculate the registration."
                )
            bundle_dir = json_path.with_name(json_path.stem + ".cmtk")
            persisted = persist_artifact_bundle(source_bundle, bundle_dir)
            manifest = dict(persisted.get("manifest") or {})
            portable_model = copy.deepcopy(model)
            for key in (
                "runtime_artifact_root", "artifact_bundle_resolved", "reference_image", "floating_image",
                "workspace", "affine_xform", "warp_xform", "manifest_path", "stdout_log", "stderr_log",
                "runtime_qc_files",
            ):
                portable_model.pop(key, None)
            portable_model.update({
                "artifact_scope": "persisted",
                "artifact_bundle": bundle_dir.name,
                "affine_xform": "affine.xform",
                "warp_xform": "warp.xform",
                "manifest_path": "manifest.json",
                "stdout_log": "cmtk-stdout.log",
                "stderr_log": "cmtk-stderr.log",
                "cmtk_version": str(manifest.get("cmtk_version") or model.get("cmtk_version") or ""),
                "effective_settings": copy.deepcopy(manifest.get("settings") or model.get("effective_settings") or {}),
                "command": copy.deepcopy(manifest.get("command") or model.get("command") or []),
                "deformation_qc": copy.deepcopy(manifest.get("deformation_qc") or model.get("deformation_qc") or {}),
                "persistence_ready": True,
                "reformat_ready": True,
            })
            payload.setdefault("deformation", {})["model"] = portable_model
            payload["deformation"]["storage"] = "cmtk_xform_bundle"
            payload["deformation"]["artifact_bundle"] = bundle_dir.name
            for stage_payload in payload.get("stages") or []:
                if str(stage_payload.get("kind") or "").lower() != "cmtk_warp":
                    continue
                stage_details = dict(stage_payload.get("details") or {})
                for key in (
                    "workspace", "warp_xform", "manifest_path", "stdout_log", "stderr_log",
                    "runtime_qc_files",
                ):
                    stage_details.pop(key, None)
                stage_details.update({
                    "artifact_scope": "persisted",
                    "artifact_bundle": bundle_dir.name,
                    "warp_xform": "warp.xform",
                    "manifest_path": "manifest.json",
                    "persistence_ready": True,
                    "reformat_ready": True,
                })
                stage_payload["details"] = stage_details
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(json_path)

    @classmethod
    def from_file(cls, json_path):
        json_path = Path(json_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("format") != "MADI3D Registration Transform Chain":
            raise RuntimeError(f"Not a MADI3D registration transform chain: {json_path}")
        schema = int(payload.get("registration_schema_version") or 0)
        if schema != REGISTRATION_CHAIN_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported registration transform schema {schema}; "
                f"expected {REGISTRATION_CHAIN_SCHEMA_VERSION}."
            )
        stages = [TransformStageResult.from_dict(raw) for raw in payload.get("stages") or []]
        deformation = payload.get("deformation") or {}
        deformation_model = copy.deepcopy(deformation.get("model") or {})
        expects_deformation = bool(payload.get("has_deformation") or deformation)
        deformation_type = str(deformation_model.get("type") or "").lower()
        if expects_deformation and deformation_type != "cmtk_warp":
            raise RuntimeError(
                "This registration transform uses a removed non-CMTK deformable format. "
                "Recalculate the registration with the current CMTK nonlinear pipeline."
            )
        if deformation_type == "cmtk_warp":
            bundle_name = str(
                deformation.get("artifact_bundle")
                or deformation_model.get("artifact_bundle")
                or ""
            ).strip()
            if not bundle_name:
                raise RuntimeError(
                    f"Registration transform declares a CMTK warp but no artifact bundle is recorded: {json_path}"
                )
            bundle_relative = Path(bundle_name)
            if bundle_relative.is_absolute() or ".." in bundle_relative.parts:
                raise RuntimeError(f"Registration transform has an unsafe CMTK artifact path: {bundle_name}")
            bundle_path = (json_path.parent / bundle_relative).resolve()
            if bundle_path.parent != json_path.parent.resolve():
                raise RuntimeError(f"Registration transform CMTK artifact must sit beside its JSON: {bundle_name}")
            verified = verify_artifact_bundle(bundle_path)
            manifest = dict(verified.get("manifest") or {})
            deformation_model.update({
                "artifact_scope": "persisted",
                "artifact_bundle": bundle_name,
                "artifact_bundle_resolved": str(bundle_path),
                "workspace": str(bundle_path),
                "runtime_artifact_root": "",
                "reference_image": "",
                "floating_image": "",
                "affine_xform": str(verified["affine_xform"]),
                "warp_xform": str(verified["warp_xform"]),
                "manifest_path": str(verified["manifest_path"]),
                "stdout_log": str(verified.get("stdout_log") or ""),
                "stderr_log": str(verified.get("stderr_log") or ""),
                "cmtk_version": str(manifest.get("cmtk_version") or deformation_model.get("cmtk_version") or ""),
                "effective_settings": copy.deepcopy(manifest.get("settings") or deformation_model.get("effective_settings") or {}),
                "command": copy.deepcopy(manifest.get("command") or deformation_model.get("command") or []),
                "deformation_qc": copy.deepcopy(manifest.get("deformation_qc") or deformation_model.get("deformation_qc") or {}),
                "persistence_ready": True,
                "reformat_ready": True,
            })
        return cls(
            registration_id=str(payload.get("registration_id") or uuid.uuid4()),
            source_dataset_id=str(payload.get("source_dataset_id") or ""),
            target_dataset_id=str(payload.get("target_dataset_id") or ""),
            source_space_uid=str(payload.get("source_space_uid") or ""),
            target_space_uid=str(payload.get("target_space_uid") or ""),
            source_name=str(payload.get("source_name") or "Moving"),
            target_name=str(payload.get("target_name") or "Reference"),
            execution_status=str(payload.get("execution_status") or "pending"),
            qc_status=str(payload.get("qc_status") or "not-evaluated"),
            user_decision=str(payload.get("user_decision") or "unapplied"),
            source_actor_matrix=copy.deepcopy(payload.get("source_actor_matrix") or np.eye(4).tolist()),
            target_actor_matrix=copy.deepcopy(payload.get("target_actor_matrix") or np.eye(4).tolist()),
            registration_grid=copy.deepcopy(payload.get("registration_grid") or {}),
            working_space=copy.deepcopy(payload.get("working_space") or {}),
            source_geometry=copy.deepcopy(payload.get("source_geometry") or {}),
            target_geometry=copy.deepcopy(payload.get("target_geometry") or {}),
            source_descriptor=copy.deepcopy(payload.get("source_descriptor") or {}),
            target_descriptor=copy.deepcopy(payload.get("target_descriptor") or {}),
            stages=stages,
            settings=copy.deepcopy(payload.get("settings") or {}),
            attachments=copy.deepcopy(payload.get("attachments") or []),
            reformat_volumes=copy.deepcopy(payload.get("reformat_volumes") or []),
            source_volumes=copy.deepcopy(payload.get("source_volumes") or []),
            deformation_model=deformation_model,
            landmarks=copy.deepcopy(payload.get("landmarks") or []),
            qc_results=copy.deepcopy(payload.get("qc_results") or []),
            logs=copy.deepcopy(payload.get("logs") or []),
            algorithm_version=str(
                payload.get("registration_algorithm_version")
                or REGISTRATION_ALGORITHM_VERSION
            ),
            created=str(payload.get("created") or datetime.now().isoformat(timespec="seconds")),
        )


@dataclass
class RegistrationDataset:
    dataset_id: str
    display_name: str
    channel_descriptors: dict[str, dict[str, Any]] = field(default_factory=dict)
    channel_items: dict[str, Any] = field(default_factory=dict, repr=False)
    attachments: list[dict[str, Any]] = field(default_factory=list)
    attachment_items: list[Any] = field(default_factory=list, repr=False)
    reference: bool = False  # compatibility mirror of a child volume with role=reference
    captured_actor_matrix: Optional[list[list[float]]] = field(default=None, repr=False)
    captured_actor_matrices: dict[str, list[list[float]]] = field(default_factory=dict, repr=False)


@dataclass
class RegistrationJob:
    job_id: str
    name: str
    project_state: dict[str, Any]
    settings: RegistrationSettings
    execution_status: str = "pending"
    qc_status: str = "not-evaluated"
    user_decision: str = "unapplied"
    execution_phase: str = ""
    error: str = ""
    results: list[RegistrationTransformChain] = field(default_factory=list)
    output_directory: str = ""

    def __post_init__(self):
        self.execution_status = validate_execution_status(self.execution_status)
        self.qc_status = validate_qc_status(self.qc_status)
        self.user_decision = validate_user_decision(self.user_decision)
        self.settings = (
            self.settings
            if isinstance(self.settings, RegistrationSettings)
            else RegistrationSettings.from_dict(self.settings)
        )

    def to_dict(self):
        return {
            "job_id": str(self.job_id),
            "name": str(self.name),
            "project_state": _plain_value(self.project_state),
            "settings": self.settings.to_dict(),
            "execution_status": self.execution_status,
            "qc_status": self.qc_status,
            "user_decision": self.user_decision,
            "execution_phase": str(self.execution_phase),
            "error": str(self.error),
            "output_directory": str(self.output_directory),
        }

    @classmethod
    def from_dict(cls, payload, *, results=None):
        payload = dict(payload or {})
        execution = str(payload.get("execution_status") or "pending")
        if execution == "running":
            execution = "pending"
        return cls(
            job_id=str(payload.get("job_id") or uuid.uuid4()),
            name=str(payload.get("name") or "Registration"),
            project_state=copy.deepcopy(payload.get("project_state") or {}),
            settings=copy.deepcopy(payload.get("settings") or {}),
            execution_status=execution,
            qc_status=str(payload.get("qc_status") or "not-evaluated"),
            user_decision=str(payload.get("user_decision") or "unapplied"),
            execution_phase="",
            error=str(payload.get("error") or ""),
            results=list(results or ()),
            output_directory=str(payload.get("output_directory") or ""),
        )
