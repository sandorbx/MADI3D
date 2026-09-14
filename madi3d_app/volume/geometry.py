"""Canonical physical geometry helpers for scientific image volumes.

The functions in this module operate on NumPy arrays and plain mappings only.
They deliberately contain no Qt or VTK objects so loading, registration,
stitching, export, and relationship validation can share one geometry contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


GRID_NUMERICAL_TOLERANCE = 1e-9
WORKING_COORDINATE_MODE_PHYSICAL = "physical-grid"
WORKING_COORDINATE_MODE_NUMERICAL = "working-grid"

WORKING_GRID_BASIS_VERIFIED_SOURCE = "verified-source-physical"
WORKING_GRID_BASIS_PARTIALLY_ASSUMED = "partially-assumed"
WORKING_GRID_BASIS_FULLY_VOXEL_DEFAULT = "fully-voxel-default"
WORKING_GRID_BASIS_SANITIZED_SOURCE = "sanitized-invalid-source"
WORKING_GRID_BASIS_USER_CALIBRATED = "user-calibrated"
WORKING_GRID_BASIS_EXACT_SOURCE_AFFINE = "exact-source-affine"
WORKING_GRID_BASES = frozenset(
    {
        WORKING_GRID_BASIS_VERIFIED_SOURCE,
        WORKING_GRID_BASIS_PARTIALLY_ASSUMED,
        WORKING_GRID_BASIS_FULLY_VOXEL_DEFAULT,
        WORKING_GRID_BASIS_SANITIZED_SOURCE,
        WORKING_GRID_BASIS_USER_CALIBRATED,
        WORKING_GRID_BASIS_EXACT_SOURCE_AFFINE,
    }
)


# PhysicalGridObservation.raw_fields is reserved for values reported by an
# external source.  These keys name MADI3D-owned state which must live in the
# scientific model instead of masquerading as immutable source evidence.
INTERNAL_SOURCE_EVIDENCE_KEYS = frozenset(
    {
        "cache",
        "cache_state",
        "channel_record",
        "backing_source_record",
        "acquisition_record",
        "generated_operation",
        "generated_parent_index_relation",
        "generated_lineage",
        "geometry_revisions",
        "local_geometry",
        "madi3d_geometry_provenance",
        "object_info",
        "object_info_state",
        "output_working_grid",
        "parent_grid_state",
        "parent_observation",
        "physical_grid",
        "processing_history",
        "provenance",
        "registration_provenance",
        "runtime",
        "runtime_state",
        "scientific_provenance",
        "transform_chain",
        "volume_sources",
        "working_grid",
        "working_grid_decision",
    }
)


def json_safe_source_value(value: Any) -> Any:
    """Convert plain source metadata to deterministic JSON-safe values."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): json_safe_source_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [json_safe_source_value(item) for item in value]
    raise ValueError(
        f"Source geometry value of type {type(value).__name__} is not JSON-safe."
    )


def split_internal_source_state(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[tuple[str, Any], ...]]:
    """Separate external raw evidence from MADI3D-owned state.

    The returned internal entries retain their complete JSON-safe values so a
    project migration can move them to the operation registry without data
    loss.  New observations must reject such entries rather than relying on
    this migration helper.
    """

    internal: list[tuple[str, Any]] = []

    def visit(node: Any, path: str) -> Any:
        if isinstance(node, Mapping):
            clean_mapping = {}
            for raw_key, raw_value in dict(node).items():
                key = str(raw_key)
                child_path = f"{path}.{key}" if path else key
                if key.strip().casefold() in INTERNAL_SOURCE_EVIDENCE_KEYS:
                    internal.append(
                        (child_path, json_safe_source_value(raw_value))
                    )
                else:
                    clean_mapping[key] = visit(raw_value, child_path)
            return clean_mapping
        if isinstance(node, (list, tuple)):
            return [
                visit(item, f"{path}[{index}]")
                for index, item in enumerate(node)
            ]
        return json_safe_source_value(node)

    clean = visit(value or {}, "")
    return json_safe_source_value(clean), tuple(internal)


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def geometry_values_equivalent(first, second) -> bool:
    """Apply the one numerical tolerance used for physical grid comparison."""
    try:
        return bool(
            np.allclose(
                np.asarray(first, dtype=float),
                np.asarray(second, dtype=float),
                rtol=GRID_NUMERICAL_TOLERANCE,
                atol=GRID_NUMERICAL_TOLERANCE,
            )
        )
    except (TypeError, ValueError):
        return False


def physical_grid_mismatches(reference, candidate) -> tuple[str, ...]:
    """Return authoritative grid fields that do not satisfy equivalence policy."""
    mismatches = []
    exact_fields = (
        ("dimensions", "dimensions"),
        ("spatial_units", "spatial units"),
        ("time_point_count", "time-point count"),
        ("time_units", "time units"),
        ("coordinate_space_id", "coordinate-space identity"),
    )
    for field_name, label in exact_fields:
        if getattr(reference, field_name) != getattr(candidate, field_name):
            mismatches.append(label)
    numerical_fields = (
        ("spacing", "spacing"),
        ("origin", "origin"),
        ("direction", "direction"),
        ("time_interval", "time interval"),
    )
    for field_name, label in numerical_fields:
        if not geometry_values_equivalent(
            getattr(reference, field_name), getattr(candidate, field_name)
        ):
            mismatches.append(label)
    return tuple(mismatches)


@dataclass(frozen=True)
class PhysicalGridObservation:
    """Immutable source evidence retained independently of a canonical grid."""

    dimensions: tuple[int, int, int] | None = None
    validated_spacing: tuple[float, float, float] | None = None
    validated_spatial_units: tuple[str, str, str] | None = None
    validated_origin: tuple[float, float, float] | None = None
    validated_direction: tuple[tuple[float, float, float], ...] | None = None
    time_point_count: int | None = None
    time_interval: float | None = None
    time_units: str | None = None
    raw_fields: Mapping[str, Any] = field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    resolution_provenance: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self):
        if self.dimensions is not None:
            object.__setattr__(
                self,
                "dimensions",
                finite_tuple3(
                    self.dimensions,
                    "Observed dimensions",
                    positive=True,
                    integer=True,
                ),
            )
        if self.validated_spacing is not None:
            object.__setattr__(
                self,
                "validated_spacing",
                finite_tuple3(
                    self.validated_spacing, "Observed spacing", positive=True
                ),
            )
        if self.validated_spatial_units is not None:
            object.__setattr__(
                self,
                "validated_spatial_units",
                canonical_space_units(self.validated_spatial_units),
            )
        if self.validated_origin is not None:
            object.__setattr__(
                self,
                "validated_origin",
                finite_tuple3(self.validated_origin, "Observed origin"),
            )
        if self.validated_direction is not None:
            direction = direction_matrix3(self.validated_direction)
            object.__setattr__(
                self,
                "validated_direction",
                tuple(tuple(float(item) for item in row) for row in direction),
            )
        if self.time_point_count is not None:
            if isinstance(self.time_point_count, (bool, np.bool_)):
                raise ValueError("Observed time-point count must be a positive integer.")
            try:
                count = int(self.time_point_count)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "Observed time-point count must be a positive integer."
                ) from exc
            if count <= 0 or count != self.time_point_count:
                raise ValueError("Observed time-point count must be a positive integer.")
            object.__setattr__(self, "time_point_count", count)
        if self.time_interval is not None:
            try:
                interval = float(self.time_interval)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Observed time interval must be finite and positive."
                ) from exc
            if not math.isfinite(interval) or interval <= 0:
                raise ValueError("Observed time interval must be finite and positive.")
            object.__setattr__(self, "time_interval", interval)
        if self.time_units is not None:
            units = str(self.time_units).strip()
            if not units:
                raise ValueError("Observed time units must be non-empty when present.")
            object.__setattr__(self, "time_units", units)

        raw_fields, internal = split_internal_source_state(self.raw_fields or {})
        if internal:
            keys = ", ".join(sorted({key for key, _value in internal}))
            raise ValueError(
                "Physical-grid raw_fields may contain external source evidence only; "
                f"move MADI3D-owned state out of: {keys}."
            )
        object.__setattr__(self, "raw_fields", _freeze_json_value(raw_fields))
        object.__setattr__(
            self,
            "missing_fields",
            tuple(
                str(value).strip()
                for value in self.missing_fields
                if str(value).strip()
            ),
        )
        object.__setattr__(
            self,
            "diagnostics",
            tuple(
                str(value).strip()
                for value in self.diagnostics
                if str(value).strip()
            ),
        )
        provenance = json_safe_source_value(list(self.resolution_provenance or ()))
        if any(not isinstance(entry, Mapping) for entry in provenance):
            raise ValueError("Resolution provenance entries must be mappings.")
        object.__setattr__(
            self,
            "resolution_provenance",
            tuple(_freeze_json_value(entry) for entry in provenance),
        )

    @property
    def is_empty(self) -> bool:
        return not any(
            value
            for value in (
                self.dimensions,
                self.validated_spacing,
                self.validated_spatial_units,
                self.validated_origin,
                self.validated_direction,
                self.time_point_count,
                self.time_interval,
                self.time_units,
                self.raw_fields,
                self.missing_fields,
                self.diagnostics,
                self.resolution_provenance,
            )
        )

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": list(self.dimensions) if self.dimensions is not None else None,
            "validated_spacing": (
                list(self.validated_spacing)
                if self.validated_spacing is not None
                else None
            ),
            "validated_spatial_units": (
                list(self.validated_spatial_units)
                if self.validated_spatial_units is not None
                else None
            ),
            "validated_origin": (
                list(self.validated_origin)
                if self.validated_origin is not None
                else None
            ),
            "validated_direction": (
                [list(row) for row in self.validated_direction]
                if self.validated_direction is not None
                else None
            ),
            "time_point_count": self.time_point_count,
            "time_interval": self.time_interval,
            "time_units": self.time_units,
            "raw_fields": _thaw_json_value(self.raw_fields),
            "missing_fields": list(self.missing_fields),
            "diagnostics": list(self.diagnostics),
            "resolution_provenance": [
                _thaw_json_value(entry) for entry in self.resolution_provenance
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PhysicalGridObservation":
        return cls(**dict(value or {}))


@dataclass(frozen=True)
class AcquisitionPhysicalGrid:
    """Immutable canonical XYZ grid stored by one logical acquisition.

    Only independent grid fields are serialized. Affines and physical support
    are derived so they cannot drift from dimensions, spacing, origin, or
    direction. The acquisition's scene pose remains a separate model value.
    """

    dimensions: tuple[int, int, int]
    spacing: tuple[float, float, float]
    spatial_units: tuple[str, str, str]
    origin: tuple[float, float, float]
    direction: tuple[tuple[float, float, float], ...]
    time_point_count: int
    time_interval: float
    time_units: str
    coordinate_space_id: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "dimensions",
            finite_tuple3(
                self.dimensions, "Dimensions", positive=True, integer=True
            ),
        )
        object.__setattr__(
            self,
            "spacing",
            finite_tuple3(self.spacing, "Spacing", positive=True),
        )
        object.__setattr__(
            self, "spatial_units", canonical_space_units(self.spatial_units)
        )
        object.__setattr__(self, "origin", finite_tuple3(self.origin, "Origin"))
        direction = direction_matrix3(self.direction, require_orthogonal=True)
        object.__setattr__(
            self,
            "direction",
            tuple(tuple(float(value) for value in row) for row in direction),
        )
        if isinstance(self.time_point_count, bool):
            raise ValueError("Time-point count must be a positive integer.")
        try:
            time_point_count = int(self.time_point_count)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Time-point count must be a positive integer.") from exc
        if time_point_count <= 0 or time_point_count != self.time_point_count:
            raise ValueError("Time-point count must be a positive integer.")
        object.__setattr__(self, "time_point_count", time_point_count)
        try:
            time_interval = float(self.time_interval)
        except (TypeError, ValueError) as exc:
            raise ValueError("Time interval must be finite and positive.") from exc
        if not math.isfinite(time_interval) or time_interval <= 0:
            raise ValueError("Time interval must be finite and positive.")
        object.__setattr__(self, "time_interval", time_interval)
        time_units = str(self.time_units or "").strip()
        if not time_units:
            raise ValueError("Time units must be explicit and non-empty.")
        object.__setattr__(self, "time_units", time_units)
        object.__setattr__(
            self,
            "coordinate_space_id",
            coordinate_space_id(self.coordinate_space_id),
        )

    @property
    def local_index_to_physical_affine(self) -> np.ndarray:
        return grid_affine_from_components(
            self.origin, self.spacing, self.direction
        )

    @property
    def physical_support_bounds(self) -> tuple[float, ...]:
        return flattened_support_bounds(
            self.local_index_to_physical_affine, self.dimensions
        )

    def world_affine(self, shared_pose) -> np.ndarray:
        return (
            invertible_affine4(shared_pose, "Acquisition shared pose")
            @ self.local_index_to_physical_affine
        )

    def world_support_bounds(self, shared_pose) -> tuple[float, ...]:
        return flattened_support_bounds(self.world_affine(shared_pose), self.dimensions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": list(self.dimensions),
            "spacing": list(self.spacing),
            "spatial_units": list(self.spatial_units),
            "origin": list(self.origin),
            "direction": [list(row) for row in self.direction],
            "time_point_count": self.time_point_count,
            "time_interval": self.time_interval,
            "time_units": self.time_units,
            "coordinate_space_id": self.coordinate_space_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AcquisitionPhysicalGrid":
        return cls(**dict(value))


@dataclass(frozen=True)
class VolumeWorkingGrid:
    """Exact numerical grid used for indexing and runtime volume operations.

    This record is deliberately distinct from :class:`AcquisitionPhysicalGrid`.
    It can retain valid source components while making explicit numerical
    assumptions for missing or invalid external metadata.  Its coordinate-space
    identity records provenance; it does not by itself claim physical calibration.
    """

    dimensions: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[tuple[float, float, float], ...]
    physical_units: tuple[str, str, str] | None
    source_coordinate_space_id: str
    coordinate_mode: str
    geometry_basis: str
    physical_grid_state: str
    physical_grid_diagnostics: tuple[str, ...] = ()
    assumed_fields: tuple[str, ...] = ()
    replaced_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    # Runtime verification input retained for direct constructors and legacy
    # projects.  It is exactly derived from spacing/origin/direction and is not
    # an independent persisted field.
    local_index_to_working_affine: (
        tuple[tuple[float, float, float, float], ...] | None
    ) = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "dimensions",
            finite_tuple3(
                self.dimensions, "Working dimensions", positive=True, integer=True
            ),
        )
        object.__setattr__(
            self,
            "spacing",
            finite_tuple3(self.spacing, "Working spacing", positive=True),
        )
        object.__setattr__(
            self, "origin", finite_tuple3(self.origin, "Working origin")
        )
        direction = direction_matrix3(self.direction)
        object.__setattr__(
            self,
            "direction",
            tuple(tuple(float(value) for value in row) for row in direction),
        )
        expected_affine = general_grid_affine_from_components(
            self.origin, self.spacing, self.direction
        )
        affine = (
            expected_affine
            if self.local_index_to_working_affine is None
            else invertible_affine4(
                self.local_index_to_working_affine,
                "Local index-to-working-space affine",
            )
        )
        if not geometry_values_equivalent(affine, expected_affine):
            raise ValueError(
                "The working-grid affine does not match spacing, origin, and direction."
            )
        object.__setattr__(
            self,
            "local_index_to_working_affine",
            tuple(tuple(float(value) for value in row) for row in affine),
        )
        if self.physical_units is not None:
            object.__setattr__(
                self, "physical_units", canonical_space_units(self.physical_units)
            )
        object.__setattr__(
            self,
            "source_coordinate_space_id",
            coordinate_space_id(self.source_coordinate_space_id),
        )
        coordinate_mode = str(self.coordinate_mode or "").strip()
        if coordinate_mode not in {
            WORKING_COORDINATE_MODE_PHYSICAL,
            WORKING_COORDINATE_MODE_NUMERICAL,
        }:
            raise ValueError(f"Unsupported working coordinate mode: {coordinate_mode!r}.")
        object.__setattr__(self, "coordinate_mode", coordinate_mode)
        basis = str(self.geometry_basis or "").strip()
        if basis not in WORKING_GRID_BASES:
            raise ValueError(f"Unsupported working-grid basis: {basis!r}.")
        object.__setattr__(self, "geometry_basis", basis)
        object.__setattr__(
            self, "physical_grid_state", str(self.physical_grid_state or "").strip()
        )
        for field_name in (
            "physical_grid_diagnostics",
            "assumed_fields",
            "replaced_fields",
            "warnings",
        ):
            values = tuple(
                str(value).strip()
                for value in (getattr(self, field_name) or ())
                if str(value).strip()
            )
            if len(set(values)) != len(values):
                raise ValueError(f"Working-grid {field_name} must not contain duplicates.")
            object.__setattr__(self, field_name, values)
        if coordinate_mode == WORKING_COORDINATE_MODE_PHYSICAL:
            if self.physical_units is None:
                raise ValueError("A physical working grid requires explicit spatial units.")
            if basis not in {
                WORKING_GRID_BASIS_VERIFIED_SOURCE,
                WORKING_GRID_BASIS_USER_CALIBRATED,
                WORKING_GRID_BASIS_EXACT_SOURCE_AFFINE,
            }:
                raise ValueError(
                    "Only verified source or user-calibrated grids may claim physical coordinates."
                )
        elif basis in {
            WORKING_GRID_BASIS_VERIFIED_SOURCE,
            WORKING_GRID_BASIS_USER_CALIBRATED,
            WORKING_GRID_BASIS_EXACT_SOURCE_AFFINE,
        }:
            raise ValueError(
                "Verified or user-calibrated geometry must use physical coordinates."
            )

    def vtk_geometry(self) -> dict[str, Any]:
        """Return the complete numeric geometry required by the VTK bridge."""
        return {
            "spacing": self.spacing,
            "origin": self.origin,
            "direction": self.direction,
        }

    def state_metadata(self) -> dict[str, Any]:
        """Return runtime metadata without promoting assumptions to calibration."""
        metadata = {
            "working_coordinate_mode": self.coordinate_mode,
            "working_grid_basis": self.geometry_basis,
            "grid_state": self.physical_grid_state,
            "grid_diagnostics": self.physical_grid_diagnostics,
            "working_grid_assumed_fields": self.assumed_fields,
            "working_grid_replaced_fields": self.replaced_fields,
            "working_grid_warnings": self.warnings,
            "working_grid": self.to_dict(),
            "geometry_status": working_grid_geometry_status(self),
        }
        if self.physical_units is not None:
            metadata["space_units"] = self.physical_units
        return metadata

    def to_dict(self) -> dict[str, Any]:
        """Serialize independent working-grid facts only."""

        return {
            "dimensions": list(self.dimensions),
            "spacing": list(self.spacing),
            "origin": list(self.origin),
            "direction": [list(row) for row in self.direction],
            "physical_units": (
                list(self.physical_units) if self.physical_units is not None else None
            ),
            "source_coordinate_space_id": self.source_coordinate_space_id,
            "coordinate_mode": self.coordinate_mode,
            "geometry_basis": self.geometry_basis,
            "physical_grid_state": self.physical_grid_state,
            "physical_grid_diagnostics": list(self.physical_grid_diagnostics),
            "assumed_fields": list(self.assumed_fields),
            "replaced_fields": list(self.replaced_fields),
            "warnings": list(self.warnings),
        }

    def revision_dict(self) -> dict[str, Any]:
        """Return the stable geometry-revision payload used before normalization.

        The affine remains in the digest so existing revision identities do not
        change merely because the deterministic copy was removed from storage.
        """

        payload = self.to_dict()
        payload["local_index_to_working_affine"] = [
            list(row) for row in self.local_index_to_working_affine
        ]
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VolumeWorkingGrid":
        return cls(**dict(value))


def working_grid_geometry_status(working_grid) -> dict[str, Any]:
    """Describe numeric geometry without promoting working defaults to calibration."""
    if not isinstance(working_grid, VolumeWorkingGrid):
        working_grid = VolumeWorkingGrid.from_dict(working_grid)
    units = (
        list(working_grid.physical_units)
        if working_grid.physical_units is not None
        else None
    )
    if (
        working_grid.physical_grid_state == "resolved"
        and working_grid.coordinate_mode == WORKING_COORDINATE_MODE_PHYSICAL
        and units is not None
    ):
        code = "verified-physical"
        label = "verified physical geometry"
        verified = True
    elif (
        working_grid.geometry_basis == WORKING_GRID_BASIS_FULLY_VOXEL_DEFAULT
        and units is None
    ):
        code = "voxel-index"
        label = "voxel/index working geometry"
        verified = False
    else:
        code = "unverified-working"
        label = "unverified working geometry"
        verified = False
    return {
        "code": code,
        "label": label,
        "verified": verified,
        "physical_units": units,
        "geometry_basis": working_grid.geometry_basis,
        "physical_grid_state": working_grid.physical_grid_state,
        "assumed_fields": list(working_grid.assumed_fields),
        "replaced_fields": list(working_grid.replaced_fields),
    }


def finite_tuple3(values, name, *, positive=False, integer=False):
    """Validate and return exactly three finite numeric values."""
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly three values.") from exc
    if len(raw) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    converted = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} contains a non-numeric value.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contains a non-numeric value.") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} contains a non-finite value.")
        if integer:
            if not number.is_integer():
                raise ValueError(
                    f"{name} values must be integers; {value!r} is fractional."
                )
            converted.append(int(number))
        else:
            converted.append(number)
    converted = tuple(converted)
    if positive and any(value <= 0 for value in converted):
        raise ValueError(f"{name} values must be positive.")
    return converted


def affine_matrix4(value, name="Transform") -> np.ndarray:
    """Return a finite affine 4 x 4 matrix without requiring invertibility."""
    matrix = np.asarray(value, dtype=float)
    if matrix.size != 16:
        raise ValueError(f"{name} must contain exactly 16 values.")
    matrix = matrix.reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains a non-finite value.")
    if not np.allclose(
        matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-9
    ):
        raise ValueError(f"{name} is not an affine 4 x 4 matrix.")
    return matrix


def invertible_affine4(value, name="Transform") -> np.ndarray:
    matrix = affine_matrix4(value, name)
    if abs(float(np.linalg.det(matrix[:3, :3]))) <= 1e-12:
        raise ValueError(f"{name} is singular.")
    return matrix


def direction_matrix3(value, *, require_orthogonal=False) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("Direction must be a 3 x 3 matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Direction contains a non-finite value.")
    if abs(float(np.linalg.det(matrix))) <= 1e-12:
        raise ValueError("Direction matrix is singular.")
    if require_orthogonal and not np.allclose(
        matrix.T @ matrix, np.eye(3), rtol=1e-6, atol=1e-6
    ):
        raise ValueError(
            "Direction matrix must be orthogonal; sheared grid bases are not supported."
        )
    return matrix


def grid_affine_from_components(origin, spacing, direction):
    spacing = np.asarray(
        finite_tuple3(spacing, "Spacing", positive=True), dtype=float
    )
    origin = np.asarray(finite_tuple3(origin, "Origin"), dtype=float)
    direction = direction_matrix3(direction, require_orthogonal=True)
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return affine


def general_grid_affine_from_components(origin, spacing, direction):
    """Compose an exact finite nonsingular grid affine, including shear/reflection."""
    spacing = np.asarray(
        finite_tuple3(spacing, "Spacing", positive=True), dtype=float
    )
    origin = np.asarray(finite_tuple3(origin, "Origin"), dtype=float)
    direction = direction_matrix3(direction)
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return invertible_affine4(affine, "Grid affine")


def grid_components_from_affine(affine):
    affine = affine_matrix4(affine, "Grid affine")
    linear = affine[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 1e-12):
        raise ValueError("Grid affine contains zero, negative, or invalid spacing.")
    direction = linear / spacing[np.newaxis, :]
    direction_matrix3(direction)
    return {
        "affine": affine,
        "origin": tuple(float(value) for value in affine[:3, 3]),
        "spacing": tuple(float(value) for value in spacing),
        "direction": direction,
    }


def affine_support_bounds(affine, dims_xyz):
    """Return min/max physical support including half-voxel borders."""
    dimensions = finite_tuple3(
        dims_xyz, "Dimensions", positive=True, integer=True
    )
    corners = np.array(
        [
            [x, y, z, 1.0]
            for x in (-0.5, dimensions[0] - 0.5)
            for y in (-0.5, dimensions[1] - 0.5)
            for z in (-0.5, dimensions[2] - 0.5)
        ],
        dtype=float,
    )
    world = (affine_matrix4(affine, "Geometry affine") @ corners.T).T[:, :3]
    return world.min(axis=0), world.max(axis=0)


def flattened_support_bounds(affine, dims_xyz):
    minimum, maximum = affine_support_bounds(affine, dims_xyz)
    return (
        float(minimum[0]),
        float(maximum[0]),
        float(minimum[1]),
        float(maximum[1]),
        float(minimum[2]),
        float(maximum[2]),
    )


_SPACE_UNIT_ALIASES = {
    "nm": "nm",
    "nanometer": "nm",
    "nanometers": "nm",
    "nanometre": "nm",
    "nanometres": "nm",
    "um": "micron",
    "µm": "micron",
    "μm": "micron",
    "micron": "micron",
    "microns": "micron",
    "micrometer": "micron",
    "micrometers": "micron",
    "micrometre": "micron",
    "micrometres": "micron",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
}


def canonical_space_units(value):
    if isinstance(value, str):
        raw = (value,) * 3
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        raw = tuple(value)
    else:
        raise ValueError(
            "Units must explicitly contain one physical unit for each of the three axes."
        )
    units = []
    for item in raw:
        key = str(item or "").strip().strip('"').lower()
        unit = _SPACE_UNIT_ALIASES.get(key)
        if unit is None:
            raise ValueError(f"Unsupported or ambiguous physical unit: {item!r}.")
        units.append(unit)
    if len(set(units)) != 1:
        raise ValueError("All three spatial axes must use the same physical unit.")
    return tuple(units)


def _working_raw_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "none"}
    return True


def _working_affine_candidate(value: Any) -> np.ndarray | None:
    if isinstance(value, (str, bytes)):
        try:
            import json

            value = json.loads(
                value.decode("utf-8", "replace") if isinstance(value, bytes) else value
            )
        except Exception:
            return None
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return matrix.reshape(4, 4) if matrix.size == 16 else None


def _working_spacing_evidence(observation: PhysicalGridObservation):
    """Return per-axis source spacing values and whether evidence was supplied."""
    if observation.validated_spacing is not None:
        return list(observation.validated_spacing), [True, True, True]

    raw = observation.raw_fields
    values = [None, None, None]
    present = [False, False, False]

    def consider(sequence, *, vector_norm=False, reciprocal=False):
        try:
            items = tuple(sequence)
        except TypeError:
            return
        if len(items) < 3:
            return
        for index, item in enumerate(items[:3]):
            if not _working_raw_present(item):
                continue
            present[index] = True
            try:
                if vector_norm:
                    vector = np.asarray(item, dtype=float).reshape(-1)
                    candidate = float(np.linalg.norm(vector[:3])) if vector.size >= 3 else math.nan
                elif reciprocal:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        candidate = float(item[0]) / float(item[1])
                    else:
                        candidate = float(item)
                    candidate = 1.0 / candidate
                else:
                    candidate = float(item)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if values[index] is None and math.isfinite(candidate) and candidate > 0:
                values[index] = candidate

    if "space_directions" in raw:
        consider(raw["space_directions"], vector_norm=True)
    for key in ("spacing", "spacings"):
        if key in raw:
            consider(raw[key])

    for key in ("affine", "madi3d_affine"):
        if key not in raw or not _working_raw_present(raw[key]):
            continue
        matrix = _working_affine_candidate(raw[key])
        if matrix is None:
            present[:] = [True, True, True]
            continue
        consider(np.linalg.norm(matrix[:3, :3], axis=0))

    ome_sizes = raw.get("ome_physical_sizes")
    if isinstance(ome_sizes, Mapping):
        consider(tuple(ome_sizes.get(axis) for axis in ("X", "Y", "Z")))
    resolution = raw.get("resolution")
    if isinstance(resolution, Mapping):
        consider(
            (
                resolution.get("XResolution"),
                resolution.get("YResolution"),
                raw.get("imagej_z_spacing"),
            ),
            reciprocal=True,
        )
        # ImageJ Z spacing is already distance-per-slice, not pixels-per-unit.
        if _working_raw_present(raw.get("imagej_z_spacing")):
            present[2] = True
            try:
                candidate = float(raw["imagej_z_spacing"])
            except (TypeError, ValueError):
                candidate = math.nan
            if math.isfinite(candidate) and candidate > 0:
                values[2] = candidate
    return values, present


def _working_origin_evidence(observation: PhysicalGridObservation):
    if observation.validated_origin is not None:
        return observation.validated_origin, True
    raw = observation.raw_fields
    if "origin" in raw and _working_raw_present(raw["origin"]):
        return raw["origin"], True
    for key in ("affine", "madi3d_affine"):
        if key in raw and _working_raw_present(raw[key]):
            matrix = _working_affine_candidate(raw[key])
            return (matrix[:3, 3] if matrix is not None else None), True
    return None, False


def _working_direction_evidence(observation: PhysicalGridObservation):
    if observation.validated_direction is not None:
        return observation.validated_direction, True
    raw = observation.raw_fields
    if "direction" in raw and _working_raw_present(raw["direction"]):
        return raw["direction"], True
    for key in ("affine", "madi3d_affine"):
        if key in raw and _working_raw_present(raw[key]):
            matrix = _working_affine_candidate(raw[key])
            if matrix is None:
                return None, True
            spacing = np.linalg.norm(matrix[:3, :3], axis=0)
            if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
                return None, True
            return matrix[:3, :3] / spacing[np.newaxis, :], True
    if "space_directions" in raw:
        directions = raw["space_directions"]
        try:
            items = tuple(directions)
        except TypeError:
            return None, _working_raw_present(directions)
        supplied = any(_working_raw_present(item) for item in items)
        if len(items) >= 3 and all(_working_raw_present(item) for item in items[:3]):
            try:
                columns = np.column_stack(
                    [np.asarray(item, dtype=float).reshape(-1)[:3] for item in items[:3]]
                )
                spacing = np.linalg.norm(columns, axis=0)
                if np.all(np.isfinite(spacing)) and np.all(spacing > 0):
                    return columns / spacing[np.newaxis, :], supplied
            except (TypeError, ValueError):
                pass
        return None, supplied
    return None, False


def _working_units_evidence(observation: PhysicalGridObservation):
    if observation.validated_spatial_units is not None:
        return observation.validated_spatial_units, True, False
    raw = observation.raw_fields
    candidates = []
    for key in (
        "space_units",
        "unit",
        "madi3d_space_units",
        "ome_space_units",
        "imagej_unit",
    ):
        value = raw.get(key)
        if not _working_raw_present(value):
            continue
        if key == "madi3d_space_units" and isinstance(value, str):
            try:
                import json

                value = json.loads(value)
            except Exception:
                pass
        try:
            candidates.append(canonical_space_units(value))
        except (TypeError, ValueError):
            return None, True, True
    if not candidates:
        return None, False, False
    if len(set(candidates)) != 1:
        return None, True, True
    return candidates[0], True, False


def resolve_volume_working_grid(
    *,
    dimensions,
    observation: PhysicalGridObservation,
    source_coordinate_space_id: str,
    physical_grid_state: str,
    physical_grid: AcquisitionPhysicalGrid | None = None,
    physical_grid_basis: str = WORKING_GRID_BASIS_VERIFIED_SOURCE,
    physical_grid_diagnostics=(),
    additional_assumed_fields=(),
    additional_replaced_fields=(),
    additional_warnings=(),
) -> VolumeWorkingGrid:
    """Resolve immutable source evidence into one reproducible numerical grid."""
    dimensions = finite_tuple3(
        dimensions, "Working dimensions", positive=True, integer=True
    )
    if not isinstance(observation, PhysicalGridObservation):
        observation = PhysicalGridObservation.from_dict(observation)

    warnings = list(observation.diagnostics)
    warnings.extend(str(value).strip() for value in additional_warnings if str(value).strip())
    assumed = list(str(value).strip() for value in additional_assumed_fields if str(value).strip())
    replaced = list(str(value).strip() for value in additional_replaced_fields if str(value).strip())

    if physical_grid is not None and dimensions == physical_grid.dimensions:
        spacing = physical_grid.spacing
        origin = physical_grid.origin
        direction = physical_grid.direction
        units = physical_grid.spatial_units
        basis = physical_grid_basis
        coordinate_mode = WORKING_COORDINATE_MODE_PHYSICAL
    else:
        if physical_grid is not None:
            spacing = physical_grid.spacing
            origin = physical_grid.origin
            direction = physical_grid.direction
            units = physical_grid.spatial_units
            replaced.append("dimensions")
            warnings.append(
                "Decoded array dimensions replace contradictory source dimensions in the working grid."
            )
        else:
            raw_spacing, spacing_present = _working_spacing_evidence(observation)
            spacing_values = []
            for index, axis in enumerate("xyz"):
                candidate = raw_spacing[index]
                if candidate is not None and math.isfinite(float(candidate)) and float(candidate) > 0:
                    spacing_values.append(float(candidate))
                    continue
                spacing_values.append(1.0)
                field_name = f"spacing_{axis}"
                target = replaced if spacing_present[index] else assumed
                target.append(field_name)
                reason = "invalid" if spacing_present[index] else "missing"
                warnings.append(
                    f"Working {axis.upper()} spacing uses 1.0 because source spacing is {reason}."
                )
            spacing = tuple(spacing_values)

            raw_origin, origin_present = _working_origin_evidence(observation)
            try:
                origin = finite_tuple3(raw_origin, "Source origin")
            except (TypeError, ValueError):
                origin = (0.0, 0.0, 0.0)
                (replaced if origin_present else assumed).append("origin")
                warnings.append(
                    "Working origin uses (0, 0, 0) because source origin is "
                    + ("invalid." if origin_present else "missing.")
                )

            raw_direction, direction_present = _working_direction_evidence(observation)
            try:
                direction_array = direction_matrix3(raw_direction)
                direction = tuple(
                    tuple(float(value) for value in row) for row in direction_array
                )
            except (TypeError, ValueError, np.linalg.LinAlgError):
                direction = tuple(
                    tuple(float(value) for value in row) for row in np.eye(3)
                )
                (replaced if direction_present else assumed).append("direction")
                warnings.append(
                    "Working direction uses identity because source direction is "
                    + ("invalid." if direction_present else "missing.")
                )

            units, units_present, units_invalid = _working_units_evidence(observation)
            if units is None:
                if units_invalid:
                    replaced.append("physical_units")
                    warnings.append(
                        "Invalid or conflicting source spatial units are unset in the working grid."
                    )
                else:
                    warnings.append(
                        "Physical spatial units are unavailable; working coordinates are not calibrated."
                    )
            elif not units_present:
                units = None

        if not replaced and not assumed and units is not None:
            basis = WORKING_GRID_BASIS_EXACT_SOURCE_AFFINE
            coordinate_mode = WORKING_COORDINATE_MODE_PHYSICAL
        else:
            basis = (
                WORKING_GRID_BASIS_SANITIZED_SOURCE
                if replaced
                else WORKING_GRID_BASIS_FULLY_VOXEL_DEFAULT
                if units is None
                and set(assumed)
                >= {"spacing_x", "spacing_y", "spacing_z", "origin", "direction"}
                else WORKING_GRID_BASIS_PARTIALLY_ASSUMED
            )
            coordinate_mode = WORKING_COORDINATE_MODE_NUMERICAL

    def unique(values):
        return tuple(dict.fromkeys(value for value in values if value))

    affine = general_grid_affine_from_components(origin, spacing, direction)
    return VolumeWorkingGrid(
        dimensions=dimensions,
        spacing=spacing,
        origin=origin,
        direction=direction,
        local_index_to_working_affine=tuple(
            tuple(float(value) for value in row) for row in affine
        ),
        physical_units=units,
        source_coordinate_space_id=source_coordinate_space_id,
        coordinate_mode=coordinate_mode,
        geometry_basis=basis,
        physical_grid_state=physical_grid_state,
        physical_grid_diagnostics=tuple(physical_grid_diagnostics or ()),
        assumed_fields=unique(assumed),
        replaced_fields=unique(replaced),
        warnings=unique(warnings),
    )


def revise_volume_working_grid(
    working_grid: VolumeWorkingGrid,
    *,
    dimensions=None,
    physical_grid_state=None,
    physical_grid_diagnostics=None,
    replaced_fields=(),
    warnings=(),
) -> VolumeWorkingGrid:
    """Return an explicit revised decision without reinterpreting source evidence."""
    if not isinstance(working_grid, VolumeWorkingGrid):
        raise TypeError("Expected a VolumeWorkingGrid.")
    dimensions = finite_tuple3(
        dimensions if dimensions is not None else working_grid.dimensions,
        "Working dimensions",
        positive=True,
        integer=True,
    )
    replaced = list(working_grid.replaced_fields)
    replaced.extend(str(value).strip() for value in replaced_fields if str(value).strip())
    decision_warnings = list(working_grid.warnings)
    decision_warnings.extend(
        str(value).strip() for value in warnings if str(value).strip()
    )
    basis = working_grid.geometry_basis
    coordinate_mode = working_grid.coordinate_mode
    if dimensions != working_grid.dimensions:
        replaced.append("dimensions")
        decision_warnings.append(
            "Decoded array dimensions replace contradictory source dimensions in the working grid."
        )
        basis = WORKING_GRID_BASIS_SANITIZED_SOURCE
        coordinate_mode = WORKING_COORDINATE_MODE_NUMERICAL
    state = (
        str(physical_grid_state).strip()
        if physical_grid_state is not None
        else working_grid.physical_grid_state
    )
    diagnostics = (
        tuple(physical_grid_diagnostics)
        if physical_grid_diagnostics is not None
        else working_grid.physical_grid_diagnostics
    )
    if state != "resolved" and coordinate_mode == WORKING_COORDINATE_MODE_PHYSICAL:
        basis = WORKING_GRID_BASIS_SANITIZED_SOURCE
        coordinate_mode = WORKING_COORDINATE_MODE_NUMERICAL

    def unique(values):
        return tuple(dict.fromkeys(value for value in values if value))

    return VolumeWorkingGrid(
        dimensions=dimensions,
        spacing=working_grid.spacing,
        origin=working_grid.origin,
        direction=working_grid.direction,
        local_index_to_working_affine=working_grid.local_index_to_working_affine,
        physical_units=working_grid.physical_units,
        source_coordinate_space_id=working_grid.source_coordinate_space_id,
        coordinate_mode=coordinate_mode,
        geometry_basis=basis,
        physical_grid_state=state,
        physical_grid_diagnostics=diagnostics,
        assumed_fields=working_grid.assumed_fields,
        replaced_fields=unique(replaced),
        warnings=unique(decision_warnings),
    )


def coordinate_space_id(value):
    identifier = str(value or "").strip()
    if not identifier:
        raise ValueError("Coordinate-space identity must be a non-empty string.")
    return identifier


@dataclass(frozen=True)
class RegistryGrid:
    """Validated scientific grid at an import, processing, or export boundary."""

    object_id: str
    object_name: str
    source_id: str | None
    dimensions: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    space_units: tuple[str, str, str] | None
    coordinate_space_id: str
    world_transform: tuple[float, ...]
    local_index_affine: tuple[float, ...]
    world_index_affine: tuple[float, ...]
    effective_support: tuple[float, float, float, float, float, float]

    @property
    def local_affine_matrix(self) -> np.ndarray:
        return np.asarray(self.local_index_affine, dtype=float).reshape(4, 4)

    @property
    def world_affine_matrix(self) -> np.ndarray:
        return np.asarray(self.world_index_affine, dtype=float).reshape(4, 4)

    @property
    def world_transform_matrix(self) -> np.ndarray:
        return np.asarray(self.world_transform, dtype=float).reshape(4, 4)


def _support_tuple6(value, name="Effective support"):
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly six values.") from exc
    if len(raw) != 6:
        raise ValueError(f"{name} must contain exactly six values.")
    try:
        support = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a non-numeric value.") from exc
    if not all(math.isfinite(item) for item in support):
        raise ValueError(f"{name} contains a non-finite value.")
    if any(support[index] > support[index + 1] for index in (0, 2, 4)):
        raise ValueError(f"{name} minima must not exceed maxima.")
    return support


def registry_grid_from_snapshot(
    snapshot: Mapping[str, Any], *, allow_unknown_space_units: bool = False
) -> RegistryGrid:
    """Construct one complete scientific grid without inventing missing fields.

    The input names are the established MADI3D scientific snapshot contract.
    Optional redundant affine and support fields are checked against the
    independent grid components so damaged snapshots cannot cross a boundary.
    """

    snapshot = dict(snapshot or {})
    object_id = str(snapshot.get("id") or "")
    object_name = str(snapshot.get("display_name") or object_id or "unnamed volume")

    def required(key, label):
        if key not in snapshot or snapshot[key] is None:
            raise ValueError(
                f"Volume {object_name!r} is missing required geometry field {label!r}."
            )
        return snapshot[key]

    def validated(label, function, value):
        try:
            return function(value)
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(
                f"Volume {object_name!r} has invalid geometry field {label!r}: {exc}"
            ) from exc

    dimensions = validated(
        "dimensions",
        lambda value: finite_tuple3(
            value, "Dimensions", positive=True, integer=True
        ),
        required("dims", "dimensions"),
    )
    spacing = validated(
        "spacing",
        lambda value: finite_tuple3(value, "Spacing", positive=True),
        required("source_spacing", "spacing"),
    )
    origin = validated(
        "origin",
        lambda value: finite_tuple3(value, "Origin"),
        required("array_origin", "origin"),
    )
    direction = validated(
        "direction",
        lambda value: direction_matrix3(value),
        required("source_direction", "direction"),
    )
    local_affine = validated(
        "direction",
        lambda value: general_grid_affine_from_components(
            origin, spacing, value
        ),
        direction,
    )
    raw_units = snapshot.get("space_units")
    if raw_units is None and allow_unknown_space_units:
        units = None
    else:
        units = validated(
            "units", canonical_space_units, required("space_units", "units")
        )
    coordinate_space = validated(
        "coordinate_space_id",
        coordinate_space_id,
        required("coordinate_space_id", "coordinate_space_id"),
    )
    world_transform = validated(
        "world_transform",
        lambda value: invertible_affine4(value, "World transform"),
        required("world_transform", "world_transform"),
    )
    world_affine = world_transform @ local_affine
    effective_support = flattened_support_bounds(world_affine, dimensions)

    for key, label, expected in (
        ("local_index_affine", "local_index_affine", local_affine),
        ("world_index_affine", "world_index_affine", world_affine),
        ("world_affine", "world_affine", world_affine),
        ("actor_matrix", "actor_matrix", world_transform),
    ):
        if snapshot.get(key) is None:
            continue
        provided = validated(
            label,
            lambda value, field=label: affine_matrix4(value, field),
            snapshot[key],
        )
        if not geometry_values_equivalent(provided, expected):
            raise ValueError(
                f"Volume {object_name!r} has inconsistent geometry field {label!r}; "
                "it does not match the required physical grid components."
            )

    if snapshot.get("effective_support") is not None:
        provided_support = validated(
            "effective_support", _support_tuple6, snapshot["effective_support"]
        )
        if not geometry_values_equivalent(provided_support, effective_support):
            raise ValueError(
                f"Volume {object_name!r} has inconsistent geometry field "
                "'effective_support'; it does not match the required physical grid."
            )

    return RegistryGrid(
        object_id=object_id,
        object_name=object_name,
        source_id=(
            None
            if snapshot.get("source_id") in (None, "")
            else str(snapshot.get("source_id"))
        ),
        dimensions=dimensions,
        spacing=spacing,
        origin=origin,
        direction=tuple(float(value) for value in direction.flat),
        space_units=units,
        coordinate_space_id=coordinate_space,
        world_transform=tuple(float(value) for value in world_transform.flat),
        local_index_affine=tuple(float(value) for value in local_affine.flat),
        world_index_affine=tuple(float(value) for value in world_affine.flat),
        effective_support=effective_support,
    )


def registry_grid_mismatches(reference, candidate) -> tuple[str, ...]:
    """Compare two validated registry grids using the canonical tolerance."""

    mismatches = []
    for field_name, label in (
        ("dimensions", "dimensions"),
        ("space_units", "spatial units"),
        ("coordinate_space_id", "coordinate-space identity"),
    ):
        if getattr(reference, field_name) != getattr(candidate, field_name):
            mismatches.append(label)
    for field_name, label in (
        ("spacing", "spacing"),
        ("origin", "origin"),
        ("direction", "direction"),
        ("world_transform", "world transform"),
        ("world_index_affine", "world affine"),
        ("effective_support", "effective support"),
    ):
        if not geometry_values_equivalent(
            getattr(reference, field_name), getattr(candidate, field_name)
        ):
            mismatches.append(label)
    return tuple(mismatches)


def world_to_index_affine(index_to_world):
    return np.linalg.inv(invertible_affine4(index_to_world, "Index-to-world affine"))


def index_to_world_points(points_xyz, index_to_world):
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Index points must be a finite N x 3 array.")
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=float)], axis=1
    )
    return (invertible_affine4(index_to_world, "Index-to-world affine") @ homogeneous.T).T[:, :3]


def world_to_index_points(points_xyz, index_to_world):
    return index_to_world_points(points_xyz, world_to_index_affine(index_to_world))


def geometry_record_from_components(
    *,
    dimensions,
    spacing,
    spatial_units,
    origin,
    direction,
    pose,
    time_point_count=1,
    time_interval=1.0,
    time_units="frame",
    coordinate_space_id_value,
):
    """Build a validated model record from canonical grid components."""
    from .model import VolumeGeometryRecord

    grid = AcquisitionPhysicalGrid(
        dimensions=dimensions,
        spacing=spacing,
        spatial_units=spatial_units,
        origin=origin,
        direction=direction,
        time_point_count=time_point_count,
        time_interval=time_interval,
        time_units=time_units,
        coordinate_space_id=coordinate_space_id_value,
    )
    pose_matrix = invertible_affine4(pose, "Volume pose")
    world_affine = grid.world_affine(pose_matrix)
    return VolumeGeometryRecord(
        dimensions=grid.dimensions,
        spacing=grid.spacing,
        spatial_units=grid.spatial_units,
        origin=grid.origin,
        direction=grid.direction,
        index_to_world_affine=world_affine.tolist(),
        effective_support=grid.world_support_bounds(pose_matrix),
        pose=pose_matrix.tolist(),
        time_point_count=grid.time_point_count,
        time_interval=grid.time_interval,
        time_units=grid.time_units,
        coordinate_space_id=grid.coordinate_space_id,
    )


def geometry_record_to_mapping(record) -> dict[str, Any]:
    """Return a detached plain mapping for a ``VolumeGeometryRecord``."""
    if not hasattr(record, "to_dict"):
        raise TypeError("Expected a VolumeGeometryRecord-compatible object.")
    return dict(record.to_dict())


__all__ = [
    "AcquisitionPhysicalGrid",
    "GRID_NUMERICAL_TOLERANCE",
    "PhysicalGridObservation",
    "RegistryGrid",
    "VolumeWorkingGrid",
    "WORKING_COORDINATE_MODE_NUMERICAL",
    "WORKING_COORDINATE_MODE_PHYSICAL",
    "WORKING_GRID_BASES",
    "WORKING_GRID_BASIS_FULLY_VOXEL_DEFAULT",
    "WORKING_GRID_BASIS_EXACT_SOURCE_AFFINE",
    "WORKING_GRID_BASIS_PARTIALLY_ASSUMED",
    "WORKING_GRID_BASIS_SANITIZED_SOURCE",
    "WORKING_GRID_BASIS_USER_CALIBRATED",
    "WORKING_GRID_BASIS_VERIFIED_SOURCE",
    "affine_matrix4",
    "affine_support_bounds",
    "canonical_space_units",
    "coordinate_space_id",
    "direction_matrix3",
    "finite_tuple3",
    "flattened_support_bounds",
    "geometry_record_from_components",
    "geometry_record_to_mapping",
    "geometry_values_equivalent",
    "general_grid_affine_from_components",
    "grid_affine_from_components",
    "grid_components_from_affine",
    "json_safe_source_value",
    "index_to_world_points",
    "invertible_affine4",
    "physical_grid_mismatches",
    "registry_grid_from_snapshot",
    "working_grid_geometry_status",
    "registry_grid_mismatches",
    "resolve_volume_working_grid",
    "revise_volume_working_grid",
    "world_to_index_affine",
    "world_to_index_points",
]
