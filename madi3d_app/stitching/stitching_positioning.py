"""GUI-independent automatic position resolution for stitching tiles."""

from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from madi3d_app.volume.geometry import affine_matrix4


_RESOLVER_SCHEMA = "stitching-positioning-v3-persisted-observations"
_VALID_STATUSES = {"usable", "partial", "ambiguous", "invalid", "absent"}
_AXES = ("X", "Y", "Z")
_OME_XY_TOLERANCE = 1e-3
_OME_Z_SPACING_RELATIVE_TOLERANCE = 0.05
_FAMILY_PRECEDENCE = (
    "native",
    "ome_plane",
    "ome_stage_label",
    "olympus",
    "leica-lif",
    "zeiss_lsm",
    "micromanager",
    "imagej",
    "h5j",
    "sidecar",
    "sidecar_grid",
    "micromanager_grid",
)
_STAGE_SEMANTICS = {
    "ome_plane_stage",
    "ome_stage_label",
    "micromanager_stage",
    "imagej_stage_text",
    "h5j_stage",
    "sidecar_stage",
    "olympus_stage",
    "zeiss_lsm_stage",
    "zeiss_lsm_tile",
    "reported-stage-position",
    "reported-tile-position",
}


def _unique_text(values) -> tuple[str, ...]:
    result: list[str] = []
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _optional_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_index(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        converted = int(number)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number != converted or converted < 0:
        return None
    return converted


def _triple(values, default=None) -> tuple[Any, Any, Any]:
    if values is None:
        return (default, default, default)
    if isinstance(values, (str, bytes)):
        return (values, values, values)
    try:
        result = list(values)
    except TypeError:
        return (values, values, values)
    if not result:
        return (default, default, default)
    if len(result) == 1:
        result *= 3
    return tuple((result + [default, default, default])[:3])


def _number_triple(values) -> tuple[float | None, float | None, float | None]:
    return tuple(_optional_number(value) for value in _triple(values))


def _normalized_unit(value) -> str | None:
    text = str(value or "").strip().lower().replace("μ", "µ")
    aliases = {
        "nm": "nm",
        "nanometer": "nm",
        "nanometers": "nm",
        "nanometre": "nm",
        "nanometres": "nm",
        "um": "um",
        "µm": "um",
        "micron": "um",
        "microns": "um",
        "micrometer": "um",
        "micrometers": "um",
        "micrometre": "um",
        "micrometres": "um",
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
    return aliases.get(text)


def _normalized_units(values) -> tuple[str | None, str | None, str | None]:
    return tuple(_normalized_unit(value) for value in _triple(values))


def _source_unit_tokens(values) -> tuple[str | None, str | None, str | None]:
    """Retain declared unit evidence without treating it as convertible."""
    return tuple(
        str(value).strip() if value not in (None, "") else None
        for value in _triple(values)
    )


def _unit_scale_microns(unit) -> float | None:
    return {
        "nm": 1e-3,
        "um": 1.0,
        "mm": 1e3,
        "cm": 1e4,
        "m": 1e6,
    }.get(_normalized_unit(unit))


@dataclass(frozen=True)
class SourceFingerprint:
    path: str
    size: int | None
    mtime_ns: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }


def _file_fingerprint(path) -> SourceFingerprint:
    raw = str(path or "")
    normalized = os.path.normcase(os.path.abspath(raw)) if raw else ""
    if not raw:
        return SourceFingerprint("", None, None)
    try:
        stat = os.stat(raw)
    except OSError:
        return SourceFingerprint(normalized, None, None)
    return SourceFingerprint(normalized, int(stat.st_size), int(stat.st_mtime_ns))


def _fingerprints(value) -> tuple[SourceFingerprint, ...]:
    if value is None:
        return ()
    if isinstance(value, SourceFingerprint):
        return (value,)
    if isinstance(value, Mapping):
        def fingerprint_integer(item):
            if item is None or isinstance(item, bool):
                return None
            try:
                result = int(item)
            except (TypeError, ValueError, OverflowError):
                return None
            return result if result >= 0 and str(item).strip() == str(result) else None

        return (
            SourceFingerprint(
                str(value.get("path") or ""),
                fingerprint_integer(value.get("size")),
                fingerprint_integer(value.get("mtime_ns")),
            ),
        )
    result = []
    try:
        values = tuple(value)
    except TypeError:
        values = (value,)
    for item in values:
        result.extend(_fingerprints(item))
    return tuple(result)


@dataclass(frozen=True)
class PositionCandidate:
    family: str
    semantic_kind: str
    position_xyz: tuple[float | None, float | None, float | None]
    units_xyz: tuple[str | None, str | None, str | None]
    source_label: str
    source_fields: tuple[str, ...]
    raw_position_xyz: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )
    source_units_xyz: tuple[str | None, str | None, str | None] = (
        None,
        None,
        None,
    )
    conversion_factors: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )
    series_index: int | None = None
    series_identity: str | None = None
    scene_identity: str | None = None
    tile_identity: str | None = None
    channel_index: int | None = None
    channel_identity: str | None = None
    time_index: int | None = None
    time_identity: str | None = None
    position_index: int | None = None
    coordinate_frame: str | None = None
    status: str = "usable"
    warnings: tuple[str, ...] = ()
    rejection_reason: str = ""
    source_fingerprint: tuple[SourceFingerprint, ...] | SourceFingerprint | Mapping[
        str, Any
    ] | None = None
    current_source_fingerprint: tuple[SourceFingerprint, ...] | SourceFingerprint | Mapping[
        str, Any
    ] | None = None
    fingerprint_state: str = "not-checked"
    source_checksum: str | None = None
    current_source_checksum: str | None = None
    checksum_state: str = "not-checked"
    persisted_observation: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        status = str(self.status or "absent").strip().lower()
        if status not in _VALID_STATUSES:
            raise ValueError(f"Unsupported position-candidate status {status!r}.")
        object.__setattr__(self, "family", str(self.family))
        object.__setattr__(self, "semantic_kind", str(self.semantic_kind))
        object.__setattr__(self, "position_xyz", _number_triple(self.position_xyz))
        object.__setattr__(self, "units_xyz", _normalized_units(self.units_xyz))
        object.__setattr__(
            self, "raw_position_xyz", _number_triple(self.raw_position_xyz)
        )
        object.__setattr__(
            self, "source_units_xyz", _source_unit_tokens(self.source_units_xyz)
        )
        object.__setattr__(
            self, "conversion_factors", _number_triple(self.conversion_factors)
        )
        object.__setattr__(
            self, "source_fields", tuple(str(value) for value in self.source_fields)
        )
        for field_name in (
            "series_identity",
            "scene_identity",
            "tile_identity",
            "channel_identity",
            "time_identity",
            "coordinate_frame",
        ):
            value = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                str(value).strip() if value not in (None, "") else None,
            )
        object.__setattr__(self, "warnings", _unique_text(self.warnings))
        object.__setattr__(self, "rejection_reason", str(self.rejection_reason or ""))
        object.__setattr__(
            self, "source_fingerprint", _fingerprints(self.source_fingerprint)
        )
        object.__setattr__(
            self,
            "current_source_fingerprint",
            _fingerprints(self.current_source_fingerprint),
        )
        object.__setattr__(
            self, "fingerprint_state", str(self.fingerprint_state or "not-checked")
        )
        object.__setattr__(
            self,
            "source_checksum",
            str(self.source_checksum).strip() if self.source_checksum else None,
        )
        object.__setattr__(
            self,
            "current_source_checksum",
            (
                str(self.current_source_checksum).strip()
                if self.current_source_checksum
                else None
            ),
        )
        object.__setattr__(
            self, "checksum_state", str(self.checksum_state or "not-checked")
        )
        observation = self.persisted_observation
        if observation is not None:
            if not isinstance(observation, Mapping):
                raise ValueError("Persisted position observation must be a mapping.")
            object.__setattr__(
                self,
                "persisted_observation",
                MappingProxyType(json.loads(json.dumps(dict(observation), allow_nan=False))),
            )
        object.__setattr__(self, "status", status)

    @property
    def axes_present(self) -> tuple[str, ...]:
        return tuple(
            axis
            for axis, value in zip(_AXES, self.raw_position_xyz)
            if value is not None
        )

    @property
    def usable_axes(self) -> tuple[str, ...]:
        return tuple(
            axis for axis, value in zip(_AXES, self.position_xyz) if value is not None
        )

    def position_array(self) -> np.ndarray:
        return np.asarray(
            [np.nan if value is None else value for value in self.position_xyz],
            dtype=float,
        )

    def to_dict(self) -> dict[str, Any]:
        fingerprints = tuple(self.source_fingerprint)
        fingerprint_payload: Any
        if not fingerprints:
            fingerprint_payload = None
        elif len(fingerprints) == 1:
            fingerprint_payload = fingerprints[0].to_dict()
        else:
            fingerprint_payload = [value.to_dict() for value in fingerprints]
        current_fingerprints = tuple(self.current_source_fingerprint)
        current_fingerprint_payload: Any
        if not current_fingerprints:
            current_fingerprint_payload = None
        elif len(current_fingerprints) == 1:
            current_fingerprint_payload = current_fingerprints[0].to_dict()
        else:
            current_fingerprint_payload = [
                value.to_dict() for value in current_fingerprints
            ]
        return {
            "family": self.family,
            "semantic_kind": self.semantic_kind,
            "status": self.status,
            "position_xyz": list(self.position_xyz),
            "normalized_position_xyz": list(self.position_xyz),
            "units_xyz": list(self.units_xyz),
            "raw_position_xyz": list(self.raw_position_xyz),
            "source_units_xyz": list(self.source_units_xyz),
            "conversion_factors": list(self.conversion_factors),
            "source_label": self.source_label,
            "source_fields": list(self.source_fields),
            "series_index": self.series_index,
            "series_identity": self.series_identity,
            "scene_identity": self.scene_identity,
            "tile_identity": self.tile_identity,
            "channel_index": self.channel_index,
            "channel_identity": self.channel_identity,
            "time_index": self.time_index,
            "time_identity": self.time_identity,
            "position_index": self.position_index,
            "coordinate_frame": self.coordinate_frame,
            "axes_present": list(self.axes_present),
            "usable_axes": list(self.usable_axes),
            "warnings": list(self.warnings),
            "rejection_reason": self.rejection_reason,
            "source_fingerprint": fingerprint_payload,
            "current_source_fingerprint": current_fingerprint_payload,
            "fingerprint_state": self.fingerprint_state,
            "source_checksum": self.source_checksum,
            "current_source_checksum": self.current_source_checksum,
            "checksum_state": self.checksum_state,
            "persisted_observation": (
                dict(self.persisted_observation)
                if self.persisted_observation is not None
                else None
            ),
        }


@dataclass(frozen=True)
class InitialPlacementSettings:
    mode: str = "auto"
    grid_overlap_percent: float = 10.0
    stage_xy_mapping: str = "identity"
    invert_stage_x: bool = False
    invert_stage_y: bool = False
    invert_stage_z: bool = False

    def stage_axis_mapping(self) -> dict[str, Any]:
        return {
            "stage_xy_mapping": str(self.stage_xy_mapping),
            "invert_stage_x": bool(self.invert_stage_x),
            "invert_stage_y": bool(self.invert_stage_y),
            "invert_stage_z": bool(self.invert_stage_z),
        }


@dataclass(frozen=True)
class InitialPlacementRecord:
    tile_id: str
    display_name: str
    candidate: PositionCandidate | None
    target_world_origin: tuple[float, float, float]
    kept_current_pose: bool
    applied_axes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    geometry_revision: str = ""

    def to_dict(self) -> dict[str, Any]:
        candidate = self.candidate
        return {
            "tile_id": self.tile_id,
            "display_name": self.display_name,
            "status": (
                candidate.status
                if candidate is not None
                else ("partial" if self.kept_current_pose else "absent")
            ),
            "selected_family": candidate.family if candidate is not None else None,
            "candidate": candidate.to_dict() if candidate is not None else None,
            "source": (
                candidate.source_label
                if candidate is not None
                else "current MADI3D pose (position unavailable)"
            ),
            "kind": candidate.semantic_kind if candidate is not None else "current",
            "position": (
                list(candidate.position_xyz)
                if candidate is not None
                else [None, None, None]
            ),
            "applied_axes": list(self.applied_axes),
            "warnings": list(_unique_text(self.warnings)),
            "rejection_reason": (
                candidate.rejection_reason if candidate is not None else ""
            ),
            "target_world_origin": list(self.target_world_origin),
            "kept_current_pose": bool(self.kept_current_pose),
            "geometry_revision": str(self.geometry_revision or ""),
        }


@dataclass(frozen=True)
class InitialPlacementResult:
    mode: str
    label: str
    selected_family: str | None
    semantic_kind: str | None
    status: str
    requires_review: bool
    placement_deltas: Mapping[str, tuple[tuple[float, ...], ...]]
    common_axes: tuple[str, ...]
    anchor_tile_id: str
    anchor_display_name: str
    records: tuple[InitialPlacementRecord, ...]
    warnings: tuple[str, ...]
    assumptions: tuple[str, ...]
    stage_axis_mapping: Mapping[str, Any]
    grid_overlap_percent: float | None
    grid_step: tuple[float, float, float] | None
    summary: str

    def __post_init__(self) -> None:
        status = str(self.status).strip().lower()
        if status not in _VALID_STATUSES:
            raise ValueError(f"Unsupported initial-placement status {status!r}.")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "placement_deltas",
            MappingProxyType(dict(self.placement_deltas)),
        )
        object.__setattr__(
            self, "stage_axis_mapping", MappingProxyType(dict(self.stage_axis_mapping))
        )
        object.__setattr__(self, "warnings", _unique_text(self.warnings))
        object.__setattr__(self, "assumptions", _unique_text(self.assumptions))

    @property
    def partial(self) -> bool:
        return self.status == "partial"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "label": self.label,
            "selected_family": self.selected_family,
            "semantic_kind": self.semantic_kind,
            "kind": self.semantic_kind or "current",
            "status": self.status,
            "requires_review": bool(self.requires_review),
            "placement_deltas": {
                tile_id: [list(row) for row in matrix]
                for tile_id, matrix in self.placement_deltas.items()
            },
            "common_axes": list(self.common_axes),
            "anchor_tile_id": self.anchor_tile_id,
            "anchor_display_name": self.anchor_display_name,
            "records": [record.to_dict() for record in self.records],
            "warnings": list(self.warnings),
            "assumptions": list(self.assumptions),
            "partial": self.partial,
            "stage_axis_mapping": dict(self.stage_axis_mapping),
            "filename_grid_overlap_percent": self.grid_overlap_percent,
            "grid_step": list(self.grid_step) if self.grid_step is not None else None,
            "summary": self.summary,
        }


_POSITION_CACHE: dict[tuple[Any, ...], tuple[PositionCandidate, ...]] = {}


def clear_position_cache() -> None:
    _POSITION_CACHE.clear()


def _selector(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("source_channel", "channel_selector"):
        value = snapshot.get(key)
        if isinstance(value, Mapping):
            return value
    value = snapshot.get("channel")
    return value if isinstance(value, Mapping) else {}


def _snapshot_index(snapshot, explicit_key, *selector_keys) -> int | None:
    value = _optional_index(snapshot.get(explicit_key))
    if value is not None:
        return value
    selector = _selector(snapshot)
    for key in selector_keys:
        value = _optional_index(selector.get(key))
        if value is not None:
            return value
    return None


def _series_index(snapshot) -> int | None:
    return _snapshot_index(snapshot, "series_index", "series_index", "series")


def _channel_index(snapshot) -> int | None:
    explicit = _snapshot_index(
        snapshot, "channel_index", "channel_index", "channel", "c"
    )
    if explicit is not None:
        return explicit
    channel = snapshot.get("channel")
    return _optional_index(channel)


def _time_index(snapshot) -> int | None:
    value = _snapshot_index(snapshot, "time_index", "time_index", "time", "t")
    return 0 if value is None else value


def _position_index(snapshot) -> int | None:
    return _snapshot_index(
        snapshot, "position_index", "position_index", "position", "scene"
    )


def _target_units(snapshot) -> tuple[str | None, str | None, str | None]:
    working = snapshot.get("working_geometry")
    if isinstance(working, Mapping) and working.get("physical_units") is not None:
        return _normalized_units(working.get("physical_units"))
    if snapshot.get("space_units") is not None:
        return _normalized_units(snapshot.get("space_units"))
    return (None, None, None)


def _candidate_fingerprints(path, extras=()) -> tuple[SourceFingerprint, ...]:
    values = []
    if path:
        values.append(_file_fingerprint(path))
    for extra in extras:
        values.append(_file_fingerprint(extra))
    return tuple(values)


def _convert_axes(raw_values, source_units, target_units):
    raw = _number_triple(raw_values)
    source = _normalized_units(source_units)
    target = _normalized_units(target_units)
    converted: list[float | None] = [None, None, None]
    output_units: list[str | None] = [None, None, None]
    factors: list[float | None] = [None, None, None]
    warnings: list[str] = []
    for index, axis in enumerate(_AXES):
        value = raw[index]
        if value is None:
            continue
        source_scale = _unit_scale_microns(source[index])
        target_scale = _unit_scale_microns(target[index])
        if source_scale is None:
            warnings.append(
                f"{axis} has an unknown or missing source unit; its value was not applied."
            )
            continue
        if target_scale is None:
            warnings.append(
                f"{axis} has an unknown target working unit; its value was not applied."
            )
            continue
        factor = source_scale / target_scale
        converted[index] = float(value * factor)
        output_units[index] = target[index]
        factors[index] = float(factor)
    return (
        raw,
        source,
        tuple(converted),
        tuple(output_units),
        tuple(factors),
        tuple(warnings),
    )


def _physical_candidate(
    *,
    family,
    semantic_kind,
    raw_values,
    source_units,
    target_units,
    source_label,
    source_fields,
    series_index=None,
    series_identity=None,
    scene_identity=None,
    tile_identity=None,
    channel_index=None,
    channel_identity=None,
    time_index=None,
    time_identity=None,
    position_index=None,
    coordinate_frame=None,
    warnings=(),
    rejection_reason="",
    status=None,
    source_fingerprint=None,
) -> PositionCandidate:
    (
        raw,
        _normalized_source_units,
        converted,
        normalized_target_units,
        factors,
        conversion_warnings,
    ) = _convert_axes(raw_values, source_units, target_units)
    all_warnings = _unique_text((*warnings, *conversion_warnings))
    raw_count = sum(value is not None for value in raw)
    usable_count = sum(value is not None for value in converted)
    if status is None:
        if raw_count == 0:
            status = "absent"
        elif usable_count == 0:
            status = "ambiguous"
        elif usable_count == 3 and not all_warnings:
            status = "usable"
        else:
            status = "partial"
    return PositionCandidate(
        family=family,
        semantic_kind=semantic_kind,
        position_xyz=converted,
        units_xyz=normalized_target_units,
        source_label=str(source_label),
        source_fields=tuple(source_fields),
        raw_position_xyz=raw,
        source_units_xyz=_source_unit_tokens(source_units),
        conversion_factors=factors,
        series_index=_optional_index(series_index),
        series_identity=(
            str(series_identity) if series_identity not in (None, "") else None
        ),
        scene_identity=scene_identity,
        tile_identity=tile_identity,
        channel_index=_optional_index(channel_index),
        channel_identity=channel_identity,
        time_index=_optional_index(time_index),
        time_identity=time_identity,
        position_index=_optional_index(position_index),
        coordinate_frame=(
            str(coordinate_frame) if coordinate_frame not in (None, "") else None
        ),
        status=status,
        warnings=all_warnings,
        rejection_reason=str(rejection_reason or ""),
        source_fingerprint=source_fingerprint,
    )


def _diagnostic_candidate(
    family,
    semantic_kind,
    source_label,
    status,
    reason,
    *,
    snapshot=None,
    source_fields=(),
    fingerprints=(),
) -> PositionCandidate:
    snapshot = snapshot or {}
    return PositionCandidate(
        family=family,
        semantic_kind=semantic_kind,
        position_xyz=(None, None, None),
        units_xyz=(None, None, None),
        raw_position_xyz=(None, None, None),
        source_units_xyz=(None, None, None),
        conversion_factors=(None, None, None),
        source_label=source_label,
        source_fields=tuple(source_fields),
        series_index=_series_index(snapshot),
        series_identity=snapshot.get("series_identity"),
        scene_identity=snapshot.get("scene_identity"),
        tile_identity=snapshot.get("tile_identity"),
        channel_index=_channel_index(snapshot),
        channel_identity=snapshot.get("channel_identity"),
        time_index=_time_index(snapshot),
        time_identity=snapshot.get("time_identity"),
        position_index=_position_index(snapshot),
        status=status,
        rejection_reason=reason,
        source_fingerprint=fingerprints,
    )


def _tag_name(element) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def _image_identity(image) -> str:
    return str(image.attrib.get("Name") or image.attrib.get("ID") or "")


def _select_ome_image(images, snapshot):
    requested_index = _series_index(snapshot)
    requested_identity = str(snapshot.get("series_identity") or "").strip()
    indexed = None
    if requested_index is not None and 0 <= requested_index < len(images):
        indexed = images[requested_index]
    identity_matches = []
    if requested_identity:
        identity_matches = [
            image
            for image in images
            if requested_identity
            in {
                str(image.attrib.get("ID") or ""),
                str(image.attrib.get("Name") or ""),
                _image_identity(image),
            }
        ]
    if indexed is not None and identity_matches and indexed not in identity_matches:
        return None, "Loaded TIFF series index and OME Image identity disagree."
    if len(identity_matches) > 1:
        return None, "Loaded TIFF series identity matches several OME Images."
    if identity_matches:
        return identity_matches[0], ""
    if requested_identity:
        return None, (
            f"Loaded TIFF series identity {requested_identity!r} has no exact OME Image."
        )
    if indexed is not None:
        return indexed, ""
    if requested_index is not None:
        return None, f"Loaded TIFF series index {requested_index} has no OME Image."
    if len(images) == 1:
        return images[0], ""
    return None, (
        "OME metadata contains multiple Images but the loaded series has no usable "
        "series index or identity."
    )


def _ome_axis_values(element, prefix):
    values = []
    units = []
    fields = []
    for axis in _AXES:
        key = f"{prefix}{axis}"
        unit_key = f"{key}Unit"
        values.append(_optional_number(element.attrib.get(key)))
        units.append(element.attrib.get(unit_key))
        if key in element.attrib:
            fields.append(key)
        if unit_key in element.attrib:
            fields.append(unit_key)
    return tuple(values), tuple(units), tuple(fields)


def _loaded_z_spacing(snapshot) -> float | None:
    for key in ("local_affine", "local_index_affine"):
        value = snapshot.get(key)
        if value is None:
            continue
        try:
            matrix = affine_matrix4(value)
        except (TypeError, ValueError):
            continue
        spacing = float(np.linalg.norm(matrix[:3, 2]))
        if math.isfinite(spacing) and spacing > 0:
            return spacing
    spacing = snapshot.get("spacing")
    try:
        value = float(_triple(spacing)[2])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def ome_position_candidates(
    ome_xml: str | None, snapshot: Mapping[str, Any]
) -> tuple[PositionCandidate, ...]:
    if not ome_xml:
        return ()
    fingerprints = _candidate_fingerprints(snapshot.get("source_path"))
    try:
        root = ET.fromstring(str(ome_xml))
    except (ET.ParseError, ValueError) as exc:
        return (
            _diagnostic_candidate(
                "ome_plane",
                "ome_plane_stage",
                "OME metadata",
                "invalid",
                f"OME XML could not be parsed: {exc}",
                snapshot=snapshot,
                fingerprints=fingerprints,
            ),
        )
    images = [node for node in root.iter() if _tag_name(node) == "Image"]
    if not images:
        return ()
    image, selection_error = _select_ome_image(images, snapshot)
    if image is None:
        return (
            _diagnostic_candidate(
                "ome_plane",
                "ome_plane_stage",
                "OME Plane metadata",
                "ambiguous",
                selection_error,
                snapshot=snapshot,
                fingerprints=fingerprints,
            ),
        )

    image_index = images.index(image)
    image_identity = str(
        snapshot.get("series_identity") or _image_identity(image)
    )
    channel_index = _channel_index(snapshot)
    if channel_index is None:
        # A scalar TIFF still addresses OME channel zero. Treating an absent
        # snapshot selector as a wildcard would admit unrelated Plane records.
        channel_index = 0
    time_index = _time_index(snapshot)
    pixels = next(
        (node for node in image.iter() if _tag_name(node) == "Pixels"), None
    )
    candidates: list[PositionCandidate] = []
    if pixels is not None:
        planes = [node for node in pixels if _tag_name(node) == "Plane"]
        if channel_index is None:
            channels = {
                _optional_index(plane.attrib.get("TheC"))
                for plane in planes
                if plane.attrib.get("TheC") is not None
            }
            channels.discard(None)
            if len(channels) > 1:
                candidates.append(
                    _diagnostic_candidate(
                        "ome_plane",
                        "ome_plane_stage",
                        "OME Plane metadata",
                        "ambiguous",
                        "OME Plane metadata contains multiple channels but the loaded channel identity is unavailable.",
                        snapshot=snapshot,
                        fingerprints=fingerprints,
                    )
                )
                planes = []
            elif channels:
                channel_index = next(iter(channels))
        selected_planes = []
        missing_plane_identity = False
        for plane in planes:
            plane_channel = _optional_index(plane.attrib.get("TheC"))
            plane_time = _optional_index(plane.attrib.get("TheT"))
            plane_z = _optional_index(plane.attrib.get("TheZ"))
            if plane_channel is None or plane_time is None or plane_z is None:
                missing_plane_identity = True
                continue
            if channel_index is not None and plane_channel != channel_index:
                continue
            if time_index is not None and plane_time != time_index:
                continue
            selected_planes.append(plane)
        if not selected_planes and planes:
            reason = (
                "OME Plane records omit required TheC/TheT/TheZ identity and "
                "cannot be matched to the loaded channel, time point, and stack plane."
                if missing_plane_identity
                else "No OME Plane record exactly matches the loaded channel and time point."
            )
            candidates.append(
                _diagnostic_candidate(
                    "ome_plane",
                    "ome_plane_stage",
                    "OME Plane metadata",
                    "ambiguous",
                    reason,
                    snapshot=snapshot,
                    fingerprints=fingerprints,
                )
            )
        if selected_planes:
            anchor_plane = next(
                (
                    plane
                    for plane in selected_planes
                    if _optional_index(plane.attrib.get("TheZ")) == 0
                ),
                None,
            )
            selection_warnings = []
            if missing_plane_identity:
                selection_warnings.append(
                    "Other OME Plane records omit required TheC/TheT/TheZ identity and were not used."
                )
            if anchor_plane is None:
                ordered = sorted(
                    selected_planes,
                    key=lambda plane: (
                        _optional_index(plane.attrib.get("TheZ"))
                        if _optional_index(plane.attrib.get("TheZ")) is not None
                        else 10**9
                    ),
                )
                anchor_plane = ordered[0]
                selection_warnings.append(
                    "OME stack has no TheZ=0 Plane; the lowest available Z plane was retained as the stack anchor."
                )
            raw, units, fields = _ome_axis_values(anchor_plane, "Position")
            converted_planes = []
            plane_conversion_warnings = []
            for plane in selected_planes:
                plane_raw, plane_units, _plane_fields = _ome_axis_values(
                    plane, "Position"
                )
                converted = _convert_axes(
                    plane_raw, plane_units, _target_units(snapshot)
                )
                converted_planes.append(
                    (_optional_index(plane.attrib.get("TheZ")), converted[2])
                )
                plane_conversion_warnings.extend(converted[5])
            rejection_reasons = []
            for axis_index, axis in enumerate(("X", "Y")):
                values = [
                    converted[axis_index]
                    for _z, converted in converted_planes
                    if converted[axis_index] is not None
                ]
                target_scale = _unit_scale_microns(
                    _target_units(snapshot)[axis_index]
                )
                tolerance = (
                    _OME_XY_TOLERANCE / target_scale
                    if target_scale is not None
                    else _OME_XY_TOLERANCE
                )
                if len(values) >= 2 and max(values) - min(values) > tolerance:
                    rejection_reasons.append(
                        f"OME {axis} stage position is not constant across the selected stack planes."
                    )
            z_values = [
                (z, converted[2])
                for z, converted in converted_planes
                if z is not None and converted[2] is not None
            ]
            if len(z_values) >= 2:
                z_values.sort(key=lambda item: item[0])
                z_indices = np.asarray([z for z, _value in z_values], dtype=float)
                z_positions = np.asarray(
                    [value for _z, value in z_values], dtype=float
                )
                index_differences = np.diff(z_indices)
                position_differences = np.diff(z_positions)
                if np.any(index_differences <= 0):
                    rejection_reasons.append(
                        "OME stack contains duplicate or non-increasing TheZ indices."
                    )
                elif not (
                    np.all(position_differences > 0)
                    or np.all(position_differences < 0)
                ):
                    rejection_reasons.append(
                        "OME Z stage positions are not monotonic across the selected stack planes."
                    )
                else:
                    observed = float(
                        np.median(
                            np.abs(position_differences / index_differences)
                        )
                    )
                    expected = _loaded_z_spacing(snapshot)
                    if (
                        expected is not None
                        and not math.isclose(
                            observed,
                            expected,
                            rel_tol=_OME_Z_SPACING_RELATIVE_TOLERANCE,
                            abs_tol=1e-6,
                        )
                    ):
                        rejection_reasons.append(
                            "OME Z-plane spacing is inconsistent with the loaded Z spacing "
                            f"({observed:.9g} versus {expected:.9g} working units)."
                        )
            plane_c = _optional_index(anchor_plane.attrib.get("TheC"))
            plane_t = _optional_index(anchor_plane.attrib.get("TheT"))
            plane_z = _optional_index(anchor_plane.attrib.get("TheZ"))
            exact_fields = [
                f"OME Image[{image_index}] ID={image.attrib.get('ID', '')!r} Name={image.attrib.get('Name', '')!r}",
                f"OME Plane TheC={plane_c} TheT={plane_t} TheZ={plane_z}",
                *(f"OME Plane {field}" for field in fields),
            ]
            candidates.append(
                _physical_candidate(
                    family="ome_plane",
                    semantic_kind="ome_plane_stage",
                    raw_values=raw,
                    source_units=units,
                    target_units=_target_units(snapshot),
                    source_label="OME Plane stage position",
                    source_fields=exact_fields,
                    series_index=image_index,
                    series_identity=image_identity,
                    channel_index=plane_c,
                    time_index=plane_t,
                    coordinate_frame="ome-stage",
                    warnings=(*selection_warnings, *plane_conversion_warnings),
                    rejection_reason=" ".join(rejection_reasons),
                    status="ambiguous" if rejection_reasons else None,
                    source_fingerprint=fingerprints,
                )
            )
    stage_label = next(
        (node for node in image if _tag_name(node) == "StageLabel"), None
    )
    if stage_label is not None:
        raw, units, fields = _ome_axis_values(stage_label, "")
        candidates.append(
            _physical_candidate(
                family="ome_stage_label",
                semantic_kind="ome_stage_label",
                raw_values=raw,
                source_units=units,
                target_units=_target_units(snapshot),
                source_label="OME StageLabel",
                source_fields=(
                    f"OME Image[{image_index}] ID={image.attrib.get('ID', '')!r} Name={image.attrib.get('Name', '')!r}",
                    *(f"OME StageLabel {field}" for field in fields),
                ),
                series_index=image_index,
                series_identity=image_identity,
                channel_index=channel_index,
                time_index=time_index,
                coordinate_frame="ome-stage",
                source_fingerprint=fingerprints,
            )
        )
    return tuple(candidates)


_STRICT_STAGE_ALIASES = {
    "stagex": (0, None),
    "stagey": (1, None),
    "stagez": (2, None),
    "xposition": (0, None),
    "yposition": (1, None),
    "zposition": (2, None),
    "positionx": (0, None),
    "positiony": (1, None),
    "positionz": (2, None),
    "xpositionum": (0, "um"),
    "ypositionum": (1, "um"),
    "zpositionum": (2, "um"),
}
_OLYMPUS_STAGE_ALIASES = {
    "abspositionvaluex": (0, None),
    "abspositionvaluey": (1, None),
    "abspositionvaluez": (2, None),
}
_OLYMPUS_AXIS_UNIT_ALIASES = {
    "abspositionunitnamex": 0,
    "abspositionunitnamey": 1,
    "abspositionunitnamez": 2,
}
_NUMBER_UNIT_RE = re.compile(
    r"^\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s*(?P<unit>nm|(?:u|µ|μ)m|microns?|mm|cm|m))?\s*$",
    re.IGNORECASE,
)
_STAGE_LINE_RE = re.compile(
    r"(?im)^\s*(?P<key>Stage\s*[XYZ]|[XYZ]\s*Position|Position\s*[XYZ]|"
    r"[XYZ]Position(?:Um)?|AbsPositionValue[XYZ])\s*[:=]\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s*(?P<unit>nm|(?:u|µ|μ)m|microns?|mm|cm|m))?\s*$"
)
_OLYMPUS_UNIT_LINE_RE = re.compile(
    r"(?im)^\s*(?P<key>AbsPositionUnitName(?:[XYZ])?)\s*[:=]\s*"
    r"(?P<unit>[^\r\n]*?)\s*$"
)
_OLYMPUS_UNSUFFIXED_VALUE_LINE_RE = re.compile(
    r"(?im)^\s*(?P<key>AbsPositionValue)\s*[:=]\s*"
    r"(?P<value>[^\r\n]*?)\s*$"
)


def _normalized_stage_key(value) -> str:
    return re.sub(r"[\s_\-]+", "", str(value or "")).lower()


_STAGE_UNIT_KEYS = (
    "PositionUnit",
    "StageUnit",
    "SpatialUnit",
    "Unit",
    "Units",
    "position_unit",
    "stage_unit",
    "spatial_unit",
    "unit",
    "units",
)
_STAGE_FRAME_KEYS = ("CoordinateFrame", "coordinate_frame")


def _shared_unit(mapping) -> Any:
    for key in _STAGE_UNIT_KEYS:
        if key in mapping:
            return mapping.get(key)
    return None


def _unit_evidence_token(value) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _normalized_unit(value) or str(value).strip().casefold()


def _olympus_unit_context(mapping):
    axis_declarations: list[list[tuple[str, Any]]] = [[], [], []]
    shared_declarations: list[tuple[str, Any]] = []
    ignored_value_fields: list[tuple[str, Any]] = []
    for raw_key, raw_value in mapping.items():
        normalized = _normalized_stage_key(raw_key)
        axis = _OLYMPUS_AXIS_UNIT_ALIASES.get(normalized)
        if axis is not None:
            if _unit_evidence_token(raw_value) is not None:
                axis_declarations[axis].append((str(raw_key), raw_value))
        elif normalized == "abspositionunitname":
            if _unit_evidence_token(raw_value) is not None:
                shared_declarations.append((str(raw_key), raw_value))
        elif normalized == "abspositionvalue":
            ignored_value_fields.append((str(raw_key), raw_value))

    conflicts = []
    for axis, declarations in enumerate(axis_declarations):
        tokens = {_unit_evidence_token(value) for _key, value in declarations}
        if len(tokens) > 1:
            conflicts.append(
                f"Conflicting Olympus stage {_AXES[axis]} unit fields: "
                + ", ".join(f"{key}={value!r}" for key, value in declarations)
                + "."
            )
    shared_tokens = {
        _unit_evidence_token(value) for _key, value in shared_declarations
    }
    if len(shared_tokens) > 1:
        conflicts.append(
            "Conflicting Olympus shared stage unit fields: "
            + ", ".join(
                f"{key}={value!r}" for key, value in shared_declarations
            )
            + "."
        )
    axis_units = tuple(
        declarations[0][1] if declarations else None
        for declarations in axis_declarations
    )
    unit_fields = tuple(
        key
        for declarations in axis_declarations
        for key, _value in declarations
    ) + tuple(key for key, _value in shared_declarations)
    shared_unit = shared_declarations[0][1] if shared_declarations else None
    warnings = tuple(
        f"Ignored unsuffixed {field}={value!r} because flattened Olympus metadata "
        "does not retain its original axis-table context."
        for field, value in ignored_value_fields
    )
    return axis_units, shared_unit, unit_fields, tuple(conflicts), warnings


def _stage_value_equivalent(left_value, left_unit, right_value, right_unit) -> bool:
    left_scale = _unit_scale_microns(left_unit)
    right_scale = _unit_scale_microns(right_unit)
    if left_scale is not None and right_scale is not None:
        left_value = float(left_value) * left_scale
        right_value = float(right_value) * right_scale
    else:
        left_token = (
            str(left_unit).strip().casefold()
            if left_unit not in (None, "")
            else None
        )
        right_token = (
            str(right_unit).strip().casefold()
            if right_unit not in (None, "")
            else None
        )
        if left_token != right_token:
            return False
    return math.isclose(
        float(left_value),
        float(right_value),
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


def _conflicting_stage_context(mapping) -> list[str]:
    conflicts = []
    for label, keys, normalizer in (
        (
            "unit",
            _STAGE_UNIT_KEYS,
            lambda value: _normalized_unit(value)
            or str(value).strip().casefold(),
        ),
        ("coordinate-frame", _STAGE_FRAME_KEYS, lambda value: str(value).strip()),
    ):
        declarations = []
        for key in keys:
            value = mapping.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            declarations.append((key, value))
        if len({normalizer(value) for _key, value in declarations}) > 1:
            conflicts.append(
                f"Conflicting stage {label} fields: "
                + ", ".join(f"{key}={value!r}" for key, value in declarations)
                + "."
            )
    return conflicts


def _strict_stage_mapping(mapping, *, include_olympus=False):
    values: list[float | None] = [None, None, None]
    units: list[Any] = [None, None, None]
    fields: list[str] = []
    conflicts = _conflicting_stage_context(mapping)
    parser_warnings = ()
    olympus_axis_units = (None, None, None)
    olympus_shared_unit = None
    olympus_unit_fields = ()
    if include_olympus:
        (
            olympus_axis_units,
            olympus_shared_unit,
            olympus_unit_fields,
            olympus_conflicts,
            parser_warnings,
        ) = _olympus_unit_context(mapping)
        conflicts.extend(olympus_conflicts)
    declarations: list[tuple[float, Any, str] | None] = [None, None, None]
    shared_unit = _shared_unit(mapping)
    for raw_key, raw_value in mapping.items():
        normalized = _normalized_stage_key(raw_key)
        alias = _STRICT_STAGE_ALIASES.get(normalized)
        if alias is None and include_olympus:
            alias = _OLYMPUS_STAGE_ALIASES.get(normalized)
        if alias is None:
            continue
        axis, implied_unit = alias
        match = _NUMBER_UNIT_RE.match(str(raw_value))
        if match is None:
            number = _optional_number(raw_value)
            explicit_unit = None
        else:
            number = _optional_number(match.group("value"))
            explicit_unit = match.group("unit")
        if number is None:
            continue
        is_olympus_position = normalized in _OLYMPUS_STAGE_ALIASES
        olympus_axis_unit = (
            olympus_axis_units[axis]
            if is_olympus_position
            else None
        )
        if (
            olympus_axis_unit is not None
            and explicit_unit is not None
            and _unit_evidence_token(olympus_axis_unit)
            != _unit_evidence_token(explicit_unit)
        ):
            conflicts.append(
                f"Conflicting Olympus stage {_AXES[axis]} inline and axis-specific "
                f"units: {explicit_unit!r} and {olympus_axis_unit!r}."
            )
        unit = (
            implied_unit
            or olympus_axis_unit
            or explicit_unit
            or (olympus_shared_unit if is_olympus_position else None)
            or shared_unit
        )
        previous = declarations[axis]
        if previous is None:
            values[axis] = number
            units[axis] = unit
            declarations[axis] = (number, unit, str(raw_key))
        elif not _stage_value_equivalent(previous[0], previous[1], number, unit):
            conflicts.append(
                f"Conflicting stage {_AXES[axis]} fields: "
                f"{previous[2]}={previous[0]!r} {previous[1] or '(unit missing)'}; "
                f"{raw_key}={number!r} {unit or '(unit missing)'}."
            )
        fields.append(str(raw_key))
    if any(value is not None for value in values):
        for raw_key in _STAGE_UNIT_KEYS:
            if raw_key in mapping:
                fields.append(raw_key)
        for raw_key in _STAGE_FRAME_KEYS:
            if raw_key in mapping:
                fields.append(raw_key)
        fields.extend(olympus_unit_fields)
    return (
        tuple(values),
        tuple(units),
        tuple(fields),
        tuple(conflicts),
        tuple(parser_warnings),
    )


def _strict_stage_text(text):
    values: list[float | None] = [None, None, None]
    units: list[Any] = [None, None, None]
    fields: list[str] = []
    olympus_context = {
        match.group("key").strip(): match.group("unit").strip()
        for match in _OLYMPUS_UNIT_LINE_RE.finditer(str(text or ""))
    }
    for match in _OLYMPUS_UNSUFFIXED_VALUE_LINE_RE.finditer(str(text or "")):
        olympus_context[match.group("key").strip()] = match.group("value").strip()
    (
        olympus_axis_units,
        olympus_shared_unit,
        olympus_unit_fields,
        olympus_conflicts,
        parser_warnings,
    ) = _olympus_unit_context(olympus_context)
    conflicts: list[str] = list(olympus_conflicts)
    declarations: list[tuple[float, Any, str] | None] = [None, None, None]
    for match in _STAGE_LINE_RE.finditer(str(text or "")):
        normalized = _normalized_stage_key(match.group("key"))
        alias = _STRICT_STAGE_ALIASES.get(normalized) or _OLYMPUS_STAGE_ALIASES.get(
            normalized
        )
        if alias is None:
            continue
        axis, implied_unit = alias
        number = float(match.group("value"))
        explicit_unit = match.group("unit")
        is_olympus_position = normalized in _OLYMPUS_STAGE_ALIASES
        olympus_axis_unit = (
            olympus_axis_units[axis]
            if is_olympus_position
            else None
        )
        if (
            olympus_axis_unit is not None
            and explicit_unit is not None
            and _unit_evidence_token(olympus_axis_unit)
            != _unit_evidence_token(explicit_unit)
        ):
            conflicts.append(
                f"Conflicting Olympus stage {_AXES[axis]} inline and axis-specific "
                f"units: {explicit_unit!r} and {olympus_axis_unit!r}."
            )
        unit = (
            implied_unit
            or olympus_axis_unit
            or explicit_unit
            or (olympus_shared_unit if is_olympus_position else None)
        )
        field = match.group("key").strip()
        previous = declarations[axis]
        if previous is None:
            values[axis] = number
            units[axis] = unit
            declarations[axis] = (number, unit, field)
        elif not _stage_value_equivalent(previous[0], previous[1], number, unit):
            conflicts.append(
                f"Conflicting stage {_AXES[axis]} fields: "
                f"{previous[2]}={previous[0]!r} {previous[1] or '(unit missing)'}; "
                f"{field}={number!r} {unit or '(unit missing)'}."
            )
        fields.append(field)
    if any(value is not None for value in values):
        fields.extend(olympus_unit_fields)
    return (
        tuple(values),
        tuple(units),
        tuple(fields),
        tuple(conflicts),
        tuple(parser_warnings),
    )


def imagej_position_candidate(
    mapping_or_text,
    target_units,
    *,
    source_label="ImageJ/Fiji metadata",
    source_fingerprint=None,
) -> PositionCandidate | None:
    if isinstance(mapping_or_text, Mapping):
        info = mapping_or_text.get("Info")
        if info not in (None, ""):
            values, units, fields, conflicts, parser_warnings = _strict_stage_text(
                info
            )
            if any(value is not None for value in values):
                return _physical_candidate(
                    family="imagej",
                    semantic_kind="imagej_stage_text",
                    raw_values=values,
                    source_units=units,
                    target_units=target_units,
                    source_label=source_label,
                    source_fields=tuple(f"Info: {field}" for field in fields),
                    coordinate_frame="imagej-stage",
                    warnings=parser_warnings,
                    status="ambiguous" if conflicts else None,
                    rejection_reason=" ".join(conflicts),
                    source_fingerprint=source_fingerprint,
                )
        values, units, fields, conflicts, parser_warnings = _strict_stage_mapping(
            mapping_or_text, include_olympus=True
        )
    else:
        values, units, fields, conflicts, parser_warnings = _strict_stage_text(
            mapping_or_text
        )
    if not any(value is not None for value in values):
        return None
    return _physical_candidate(
        family="imagej",
        semantic_kind="imagej_stage_text",
        raw_values=values,
        source_units=units,
        target_units=target_units,
        source_label=source_label,
        source_fields=fields,
        coordinate_frame="imagej-stage",
        warnings=parser_warnings,
        status="ambiguous" if conflicts else None,
        rejection_reason=" ".join(conflicts),
        source_fingerprint=source_fingerprint,
    )


def _mapping_value(mapping, *keys):
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _record_index(mapping, *keys) -> int | None:
    return _optional_index(_mapping_value(mapping, *keys))


def _mm_records_from_metadata(metadata) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    if not isinstance(metadata, Mapping):
        return records
    record_keys = {
        "XPositionUm",
        "YPositionUm",
        "ZPositionUm",
        "GridRow",
        "GridCol",
    }
    if record_keys.intersection(metadata):
        records.append(metadata)
    for key in (
        "Records",
        "records",
        "Frames",
        "frames",
        "PerFrame",
        "FrameMetadata",
        "Metadata",
        "IndexMap",
    ):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            for nested_key, nested in value.items():
                if isinstance(nested, Mapping):
                    merged = dict(nested)
                    if str(nested_key).startswith("FrameKey-"):
                        merged.setdefault("FrameKey", str(nested_key))
                    records.extend(_mm_records_from_metadata(merged))
        elif isinstance(value, (list, tuple)):
            for nested in value:
                if isinstance(nested, Mapping):
                    records.extend(_mm_records_from_metadata(nested))
    for key, value in metadata.items():
        if str(key).startswith("FrameKey-") and isinstance(value, Mapping):
            records.extend(_mm_records_from_metadata(value))
    return records


def _filter_mm_records(records, snapshot):
    selected = list(records)
    requested_position = _position_index(snapshot)
    positions = {
        value
        for record in selected
        if (
            value := _record_index(
                record, "Position", "PositionIndex", "position", "position_index"
            )
        )
        is not None
    }
    if requested_position is None and len(positions) > 1:
        return [], (
            "Micro-Manager metadata contains multiple acquisition positions but "
            "the loaded position identity is unavailable."
        )
    if requested_position is None and len(positions) == 1:
        requested_position = next(iter(positions))
    filters = (
        (
            requested_position,
            ("Position", "PositionIndex", "position", "position_index"),
            "position",
        ),
        (
            _channel_index(snapshot),
            ("ChannelIndex", "Channel", "channel", "channel_index"),
            "channel",
        ),
        (
            _time_index(snapshot),
            ("Frame", "FrameIndex", "Time", "time", "frame"),
            "frame",
        ),
    )
    for requested, keys, label in filters:
        if requested is None:
            continue
        indexed = [
            (record, _record_index(record, *keys)) for record in selected
        ]
        if any(value is not None for _record, value in indexed):
            selected = [
                record for record, value in indexed if value == requested
            ]
            if not selected:
                return [], (
                    f"No Micro-Manager frame matches loaded {label} index {requested}."
                )
    sliced = [
        (record, _record_index(record, "Slice", "SliceIndex", "slice"))
        for record in selected
    ]
    if any(value is not None for _record, value in sliced):
        zero = [record for record, value in sliced if value == 0]
        if zero:
            selected = zero
    return selected, ""


def _mm_stage_values(record):
    values = []
    units = []
    fields = []
    shared_unit = _shared_unit(record)
    for axis in _AXES:
        suffixed = f"{axis}PositionUm"
        generic = f"{axis}Position"
        if suffixed in record:
            value = record.get(suffixed)
            unit = "um"
            field = suffixed
        elif generic in record:
            value = record.get(generic)
            unit = shared_unit
            field = generic
        else:
            value = None
            unit = None
            field = ""
        values.append(_optional_number(value))
        units.append(unit)
        if field:
            fields.append(field)
    return tuple(values), tuple(units), tuple(fields)


def micromanager_position_candidates(
    metadata,
    records,
    snapshot: Mapping[str, Any],
) -> tuple[PositionCandidate, ...]:
    records = [
        record for record in (records or ()) if isinstance(record, Mapping)
    ]
    if not records:
        records = _mm_records_from_metadata(metadata)
    if not records:
        return ()
    fingerprints = _candidate_fingerprints(snapshot.get("source_path"))
    selected, selection_error = _filter_mm_records(records, snapshot)
    if selection_error:
        return (
            _diagnostic_candidate(
                "micromanager",
                "micromanager_stage",
                "Micro-Manager frame metadata",
                "ambiguous",
                selection_error,
                snapshot=snapshot,
                fingerprints=fingerprints,
            ),
        )
    candidates: list[PositionCandidate] = []
    stage_records = []
    for record in selected:
        raw, units, fields = _mm_stage_values(record)
        if any(value is not None for value in raw):
            stage_records.append((record, raw, units, fields))
    if stage_records:
        normalized_positions = []
        for _record, raw, units, _fields in stage_records:
            normalized_positions.append(_convert_axes(raw, units, _target_units(snapshot))[2])
        distinct = {
            tuple(value for value in position)
            for position in normalized_positions
        }
        if len(distinct) > 1:
            candidates.append(
                _diagnostic_candidate(
                    "micromanager",
                    "micromanager_stage",
                    "Micro-Manager frame metadata",
                    "ambiguous",
                    "Several exact Micro-Manager records match the loaded identity but report different stage positions.",
                    snapshot=snapshot,
                    fingerprints=fingerprints,
                )
            )
        else:
            record, raw, units, fields = stage_records[0]
            position_index = _record_index(
                record,
                "Position",
                "PositionIndex",
                "position",
                "position_index",
            )
            scene_identity = _mapping_value(
                record, "PositionName", "SceneIdentity", "scene_identity"
            )
            channel_identity = _mapping_value(
                record, "ChannelName", "ChannelIdentity", "channel_identity"
            )
            identity_fields = [
                key
                for key in (
                    "Position",
                    "PositionIndex",
                    "PositionName",
                    "Channel",
                    "ChannelIndex",
                    "Slice",
                    "Frame",
                )
                if key in record
            ]
            candidates.append(
                _physical_candidate(
                    family="micromanager",
                    semantic_kind="micromanager_stage",
                    raw_values=raw,
                    source_units=units,
                    target_units=_target_units(snapshot),
                    source_label="Micro-Manager frame stage position",
                    source_fields=tuple((*identity_fields, *fields)),
                    series_index=_series_index(snapshot),
                    series_identity=snapshot.get("series_identity"),
                    scene_identity=(
                        scene_identity or snapshot.get("scene_identity")
                    ),
                    channel_index=_channel_index(snapshot),
                    channel_identity=(
                        channel_identity or snapshot.get("channel_identity")
                    ),
                    time_index=_time_index(snapshot),
                    time_identity=snapshot.get("time_identity"),
                    position_index=(
                        position_index
                        if position_index is not None
                        else _position_index(snapshot)
                    ),
                    coordinate_frame="micromanager-stage",
                    source_fingerprint=fingerprints,
                )
            )
    if not stage_records:
        grid_records = [
            record
            for record in selected
            if _optional_number(record.get("GridRow")) is not None
            and _optional_number(record.get("GridCol")) is not None
        ]
        if grid_records:
            grid_record = grid_records[0]
            row = _optional_number(grid_record.get("GridRow"))
            column = _optional_number(grid_record.get("GridCol"))
            position_index = _record_index(
                grid_record,
                "Position",
                "PositionIndex",
                "position",
                "position_index",
            )
            candidates.append(
                PositionCandidate(
                    family="micromanager_grid",
                    semantic_kind="filename_grid",
                    position_xyz=(column, row, None),
                    units_xyz=(None, None, None),
                    raw_position_xyz=(column, row, None),
                    source_units_xyz=(None, None, None),
                    conversion_factors=(None, None, None),
                    source_label="Micro-Manager acquisition grid",
                    source_fields=("GridCol", "GridRow"),
                    series_index=_series_index(snapshot),
                    series_identity=snapshot.get("series_identity"),
                    scene_identity=(
                        _mapping_value(
                            grid_record,
                            "PositionName",
                            "SceneIdentity",
                            "scene_identity",
                        )
                        or snapshot.get("scene_identity")
                    ),
                    channel_index=_channel_index(snapshot),
                    channel_identity=snapshot.get("channel_identity"),
                    time_index=_time_index(snapshot),
                    time_identity=snapshot.get("time_identity"),
                    position_index=(
                        position_index
                        if position_index is not None
                        else _position_index(snapshot)
                    ),
                    coordinate_frame="micromanager-grid",
                    status="partial",
                    source_fingerprint=fingerprints,
                )
            )
    return tuple(candidates)


def position_sidecar_candidates(path) -> tuple[Path, ...]:
    source = Path(path)
    return (
        source.with_suffix(".json"),
        Path(str(source) + ".json"),
        source.with_name(source.stem + "_metadata.json"),
    )


def _sidecar_position_candidate(path, target_units, source_path, snapshot=None):
    snapshot = dict(snapshot or {})
    sidecar = Path(path)
    fingerprints = _candidate_fingerprints(source_path, (sidecar,))
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _diagnostic_candidate(
            "sidecar",
            "sidecar_stage",
            f"sidecar {sidecar.name}",
            "invalid",
            f"Stage-position sidecar is malformed or unreadable: {exc}",
            source_fields=(str(sidecar),),
            fingerprints=fingerprints,
            snapshot=snapshot,
        )
    if not isinstance(payload, Mapping):
        return _diagnostic_candidate(
            "sidecar",
            "sidecar_stage",
            f"sidecar {sidecar.name}",
            "invalid",
            "Stage-position sidecar must contain one JSON object.",
            source_fields=(str(sidecar),),
            fingerprints=fingerprints,
            snapshot=snapshot,
        )
    coordinate_frame = payload.get("CoordinateFrame") or payload.get(
        "coordinate_frame"
    )
    series_index = _record_index(payload, "SeriesIndex", "series_index", "series")
    channel_index = _record_index(
        payload, "ChannelIndex", "channel_index", "channel"
    )
    time_index = _record_index(payload, "TimeIndex", "time_index", "time", "frame")
    position_index = _record_index(
        payload, "PositionIndex", "position_index", "position", "scene"
    )
    series_identity = _mapping_value(payload, "SeriesIdentity", "series_identity")
    scene_identity = _mapping_value(payload, "SceneIdentity", "scene_identity")
    tile_identity = _mapping_value(payload, "TileIdentity", "tile_identity")
    channel_identity = _mapping_value(
        payload, "ChannelIdentity", "channel_identity"
    )
    time_identity = _mapping_value(payload, "TimeIdentity", "time_identity")
    vector_key = next(
        (
            key
            for key in ("StagePosition", "stage_position")
            if key in payload
        ),
        None,
    )
    if vector_key is not None:
        conflicts = ()
        parser_warnings = ()
        vector_value = payload.get(vector_key)
        if isinstance(vector_value, (str, bytes, Mapping)):
            vector_values = ()
        else:
            try:
                vector_values = tuple(vector_value)
            except TypeError:
                vector_values = ()
        if len(vector_values) not in {2, 3}:
            return _diagnostic_candidate(
                "sidecar",
                "sidecar_stage",
                f"sidecar {sidecar.name}",
                "invalid",
                f"{vector_key} must contain two or three finite coordinate values.",
                source_fields=(vector_key, str(sidecar)),
                fingerprints=fingerprints,
                snapshot=snapshot,
            )
        raw = _number_triple(vector_values)
        if any(value is None for value in raw[: len(vector_values)]):
            return _diagnostic_candidate(
                "sidecar",
                "sidecar_stage",
                f"sidecar {sidecar.name}",
                "invalid",
                f"{vector_key} contains a non-finite or non-numeric coordinate.",
                source_fields=(vector_key, str(sidecar)),
                fingerprints=fingerprints,
                snapshot=snapshot,
            )
        units_value = (
            payload.get("PositionUnits")
            or payload.get("StageUnits")
            or payload.get("Unit")
            or payload.get("Units")
        )
        units = _triple(units_value)
        fields = [vector_key]
        if units_value is not None:
            fields.append(
                next(
                    key
                    for key in ("PositionUnits", "StageUnits", "Unit", "Units")
                    if key in payload
                )
            )
    else:
        raw, units, fields, conflicts, parser_warnings = _strict_stage_mapping(
            payload
        )
    if not any(value is not None for value in raw):
        row = _optional_number(_mapping_value(payload, "GridRow", "grid_row"))
        column = _optional_number(_mapping_value(payload, "GridCol", "grid_col"))
        if row is None or column is None:
            return None
        stat = sidecar.stat()
        return PositionCandidate(
            family="sidecar_grid",
            semantic_kind="filename_grid",
            position_xyz=(column, row, None),
            units_xyz=(None, None, None),
            raw_position_xyz=(column, row, None),
            source_units_xyz=(None, None, None),
            conversion_factors=(None, None, None),
            source_label=f"sidecar {sidecar.name} grid",
            source_fields=(
                "GridCol",
                "GridRow",
                f"sidecar_path={os.path.abspath(str(sidecar))}",
                f"sidecar_size={int(stat.st_size)}",
                f"sidecar_mtime_ns={int(stat.st_mtime_ns)}",
            ),
            series_index=(series_index if series_index is not None else _series_index(snapshot)),
            series_identity=(
                str(series_identity)
                if series_identity not in (None, "")
                else snapshot.get("series_identity")
            ),
            scene_identity=(
                scene_identity
                if scene_identity not in (None, "")
                else snapshot.get("scene_identity")
            ),
            tile_identity=(
                tile_identity
                if tile_identity not in (None, "")
                else snapshot.get("tile_identity")
            ),
            channel_index=(
                channel_index if channel_index is not None else _channel_index(snapshot)
            ),
            channel_identity=(
                channel_identity
                if channel_identity not in (None, "")
                else snapshot.get("channel_identity")
            ),
            time_index=time_index if time_index is not None else _time_index(snapshot),
            time_identity=(
                time_identity
                if time_identity not in (None, "")
                else snapshot.get("time_identity")
            ),
            position_index=(
                position_index
                if position_index is not None
                else _position_index(snapshot)
            ),
            coordinate_frame=coordinate_frame or "sidecar-grid",
            status="partial",
            source_fingerprint=fingerprints,
        )
    stat = sidecar.stat()
    fields = (
        *fields,
        f"sidecar_path={os.path.abspath(str(sidecar))}",
        f"sidecar_size={int(stat.st_size)}",
        f"sidecar_mtime_ns={int(stat.st_mtime_ns)}",
    )
    return _physical_candidate(
        family="sidecar",
        semantic_kind="sidecar_stage",
        raw_values=raw,
        source_units=units,
        target_units=target_units,
        source_label=f"sidecar {sidecar.name}",
        source_fields=fields,
        series_index=series_index if series_index is not None else _series_index(snapshot),
        series_identity=(
            str(series_identity)
            if series_identity not in (None, "")
            else snapshot.get("series_identity")
        ),
        scene_identity=(
            scene_identity
            if scene_identity not in (None, "")
            else snapshot.get("scene_identity")
        ),
        tile_identity=(
            tile_identity
            if tile_identity not in (None, "")
            else snapshot.get("tile_identity")
        ),
        channel_index=(
            channel_index if channel_index is not None else _channel_index(snapshot)
        ),
        channel_identity=(
            channel_identity
            if channel_identity not in (None, "")
            else snapshot.get("channel_identity")
        ),
        time_index=time_index if time_index is not None else _time_index(snapshot),
        time_identity=(
            time_identity
            if time_identity not in (None, "")
            else snapshot.get("time_identity")
        ),
        position_index=(
            position_index if position_index is not None else _position_index(snapshot)
        ),
        coordinate_frame=coordinate_frame or "sidecar-stage",
        warnings=parser_warnings,
        status="ambiguous" if conflicts else None,
        rejection_reason=" ".join(conflicts),
        source_fingerprint=fingerprints,
    )


def catalog_position_candidates(
    snapshot: Mapping[str, Any],
) -> tuple[PositionCandidate, ...]:
    """Extract generic source-origin, sidecar, and filename evidence at catalog time."""
    snapshot = dict(snapshot)
    candidates: list[PositionCandidate] = []
    origin = _number_triple(snapshot.get("source_origin"))
    if any(value is not None for value in origin):
        candidates.append(
            _physical_candidate(
                family="native",
                semantic_kind="native_grid_origin",
                raw_values=origin,
                source_units=snapshot.get("source_units"),
                target_units=_target_units(snapshot),
                source_label="source-declared native grid origin",
                source_fields=tuple(snapshot.get("source_origin_fields") or ("origin",)),
                series_index=_series_index(snapshot),
                series_identity=snapshot.get("series_identity"),
                scene_identity=snapshot.get("scene_identity"),
                tile_identity=snapshot.get("tile_identity"),
                coordinate_frame=(
                    snapshot.get("source_coordinate_frame") or "native-physical"
                ),
                warnings=tuple(snapshot.get("source_warnings") or ()),
                source_fingerprint=_candidate_fingerprints(
                    snapshot.get("source_path")
                ),
            )
        )

    source_path = str(snapshot.get("source_path") or "")
    if source_path:
        seen_sidecars = set()
        for sidecar in position_sidecar_candidates(source_path):
            normalized = os.path.normcase(os.path.abspath(str(sidecar)))
            if normalized in seen_sidecars or not sidecar.is_file():
                continue
            seen_sidecars.add(normalized)
            candidates.append(
                _sidecar_position_candidate(
                    sidecar, _target_units(snapshot), source_path, snapshot
                )
            )
        filename = filename_position_candidate(snapshot)
        if filename is not None:
            candidates.append(
                replace(
                    filename,
                    series_index=_series_index(snapshot),
                    series_identity=(
                        str(snapshot.get("series_identity"))
                        if snapshot.get("series_identity") not in (None, "")
                        else None
                    ),
                    scene_identity=snapshot.get("scene_identity"),
                    tile_identity=snapshot.get("tile_identity"),
                    channel_index=None,
                    channel_identity=None,
                    time_index=None,
                    time_identity=None,
                    position_index=_position_index(snapshot),
                )
            )
    return _deduplicated_candidates(candidates)


_FILENAME_AXIS_RE = re.compile(
    r"(?i)(?<![a-z0-9])(?P<axis>x|y|z|row|col|r|c)\s*"
    r"(?P<assign>[=:]?)\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?)\s*(?P<unit>microns?|mm|nm|um|µm)?"
)


def filename_position_candidate(
    snapshot: Mapping[str, Any],
) -> PositionCandidate | None:
    path = str(snapshot.get("source_path") or snapshot.get("name") or "")
    stem = Path(path).stem if path else str(snapshot.get("display_name") or "")
    matches = {
        match.group("axis").lower(): match
        for match in _FILENAME_AXIS_RE.finditer(stem)
    }
    if not ({"r", "c"} <= set(matches)):
        matches.pop("r", None)
        matches.pop("c", None)
    values: list[float | None] = [None, None, None]
    units: list[Any] = [None, None, None]
    fields: list[str] = []
    physical_syntax = False
    axis_map = {
        "x": 0,
        "col": 0,
        "c": 0,
        "y": 1,
        "row": 1,
        "r": 1,
        "z": 2,
    }
    for token, match in matches.items():
        axis = axis_map[token]
        values[axis] = _optional_number(match.group("value"))
        units[axis] = match.group("unit")
        fields.append(match.group(0))
        physical_syntax |= bool(
            match.group("assign")
            or match.group("unit")
            or "." in match.group("value")
            or "e" in match.group("value").lower()
            or match.group("value").startswith(("+", "-"))
        )
    if not any(value is not None for value in values):
        pair = re.search(
            r"(?i)(?:tile|position|pos)[_\- ]*(\d+)[_\- ]+(\d+)", stem
        )
        if pair:
            values[1], values[0] = float(pair.group(1)), float(pair.group(2))
            fields.append(pair.group(0))
        else:
            single = re.search(
                r"(?i)(?:tile|position|pos)[_\- ]*(\d+)(?![_\- ]+\d)", stem
            )
            if single:
                values[0] = float(single.group(1))
                fields.append(single.group(0))
    if not any(value is not None for value in values):
        return None
    fingerprints = _candidate_fingerprints(path)
    if physical_syntax:
        return _physical_candidate(
            family="filename",
            semantic_kind="filename_physical",
            raw_values=values,
            source_units=units,
            target_units=_target_units(snapshot),
            source_label=f"filename {Path(path).name or stem}",
            source_fields=fields,
            coordinate_frame=f"filename:{os.path.normcase(os.path.abspath(str(Path(path).parent)))}",
            source_fingerprint=fingerprints,
        )
    return PositionCandidate(
        family="filename",
        semantic_kind="filename_grid",
        position_xyz=tuple(values),
        units_xyz=(None, None, None),
        raw_position_xyz=tuple(values),
        source_units_xyz=(None, None, None),
        conversion_factors=(None, None, None),
        source_label=f"filename {Path(path).name or stem}",
        source_fields=tuple(fields),
        coordinate_frame=f"filename-grid:{os.path.normcase(os.path.abspath(str(Path(path).parent)))}",
        status=(
            "usable" if all(value is not None for value in values) else "partial"
        ),
        source_fingerprint=fingerprints,
    )


def _h5j_position_candidate(snapshot):
    metadata = snapshot.get("source_metadata")
    if not isinstance(metadata, Mapping):
        return None
    mappings = []
    for key in (
        "selected_channel_attrs",
        "channel_attrs",
        "dataset_attrs",
        "root_attrs",
        "attrs",
    ):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            selector = _selector(snapshot)
            channel = selector.get("channel")
            if channel is None:
                selected = snapshot.get("channel_selector")
                if not isinstance(selected, Mapping):
                    channel = selected
            if (
                key == "channel_attrs"
                and channel in value
                and isinstance(value.get(channel), Mapping)
            ):
                mappings.append((f"H5J selected channel {channel!r} attributes", value[channel]))
            else:
                mappings.append((f"H5J {key.replace('_', ' ')}", value))
    mappings.append(("H5J retained source metadata", metadata))
    for label, mapping in mappings:
        raw, units, fields, conflicts, parser_warnings = _strict_stage_mapping(
            mapping
        )
        if not any(value is not None for value in raw):
            continue
        coordinate_frame = (
            mapping.get("CoordinateFrame")
            or mapping.get("coordinate_frame")
            or "h5j-stage"
        )
        return _physical_candidate(
            family="h5j",
            semantic_kind="h5j_stage",
            raw_values=raw,
            source_units=units,
            target_units=_target_units(snapshot),
            source_label=label,
            source_fields=fields,
            series_index=_series_index(snapshot),
            series_identity=snapshot.get("series_identity"),
            scene_identity=snapshot.get("scene_identity"),
            tile_identity=snapshot.get("tile_identity"),
            channel_index=_channel_index(snapshot),
            channel_identity=snapshot.get("channel_identity"),
            time_index=_time_index(snapshot),
            time_identity=snapshot.get("time_identity"),
            coordinate_frame=coordinate_frame,
            warnings=parser_warnings,
            status="ambiguous" if conflicts else None,
            rejection_reason=" ".join(conflicts),
            source_fingerprint=_candidate_fingerprints(
                snapshot.get("source_path")
            ),
        )
    return None


def h5j_position_candidate(snapshot):
    """Return retained H5J stage evidence without inspecting pixels or sidecars."""
    return _h5j_position_candidate(snapshot)


def _selected_tiff_description(handle, series_index):
    try:
        if series_index is not None and 0 <= series_index < len(handle.series):
            pages = handle.series[series_index].pages
            if len(pages):
                return pages[0].description
        if len(handle.series) == 1 and len(handle.series[0].pages):
            return handle.series[0].pages[0].description
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    return None


def _micromanager_tiff_records(handle, series_index):
    records = []
    manager_metadata = getattr(handle, "micromanager_metadata", None) or {}
    index_records = _mm_records_from_metadata(manager_metadata)
    index_map = manager_metadata.get("IndexMap")
    index_rows_by_offset = {}
    index_rows_in_order = []
    try:
        index_array = np.asarray(index_map)
        if index_array.ndim == 2 and index_array.shape[1] >= 5:
            for row in index_array:
                record = {
                    "ChannelIndex": int(row[0]),
                    "Slice": int(row[1]),
                    "Frame": int(row[2]),
                    "Position": int(row[3]),
                    "IFDOffset": int(row[4]),
                }
                index_rows_in_order.append(record)
                index_rows_by_offset[int(row[4])] = record
    except (TypeError, ValueError, OverflowError):
        index_rows_by_offset = {}
        index_rows_in_order = []
    pages = ()
    try:
        if series_index is not None and 0 <= series_index < len(handle.series):
            pages = tuple(handle.series[series_index].pages)
        elif len(handle.series) == 1:
            pages = tuple(handle.series[0].pages)
    except (AttributeError, IndexError, TypeError, ValueError):
        pages = ()
    for page_index, page in enumerate(pages):
        value = None
        try:
            tag = page.tags.get("MicroManagerMetadata")
            value = tag.value if tag is not None else None
        except (AttributeError, KeyError, TypeError, ValueError):
            value = None
        if isinstance(value, (bytes, str)):
            try:
                value = json.loads(
                    value.decode("utf-8", "replace")
                    if isinstance(value, bytes)
                    else value
                )
            except (UnicodeError, json.JSONDecodeError):
                value = None
        if isinstance(value, Mapping):
            merged = dict(value)
            page_offset = _optional_index(getattr(page, "offset", None))
            index_record = index_rows_by_offset.get(page_offset)
            if index_record is None and page_index < len(index_rows_in_order):
                index_record = index_rows_in_order[page_index]
            if index_record is None and page_index < len(index_records):
                index_record = index_records[page_index]
            if index_record is not None:
                for key, item in index_record.items():
                    merged.setdefault(key, item)
            records.append(merged)
    return records or index_records


def _cache_key(snapshot):
    observation_values = snapshot.get("stage_position_observations") or ()
    fingerprint_paths = []
    for raw_observation in observation_values:
        if hasattr(raw_observation, "to_dict"):
            raw_observation = raw_observation.to_dict()
        if not isinstance(raw_observation, Mapping):
            continue
        source_fields = raw_observation.get("source_fields") or {}
        if not isinstance(source_fields, Mapping):
            continue
        fingerprints = source_fields.get("source_fingerprints") or ()
        if isinstance(fingerprints, Mapping):
            fingerprints = (fingerprints,)
        for fingerprint in fingerprints:
            if isinstance(fingerprint, Mapping) and fingerprint.get("path"):
                fingerprint_paths.append(str(fingerprint["path"]))
    file_state = tuple(_file_fingerprint(path) for path in fingerprint_paths)
    local_value = snapshot.get("local_affine")
    if local_value is None:
        local_value = snapshot.get("local_index_affine")
    try:
        local_key = tuple(np.asarray(local_value, dtype=float).reshape(-1))
    except (TypeError, ValueError):
        local_key = ()
    selector = _selector(snapshot)
    try:
        selector_key = json.dumps(selector, sort_keys=True, default=str)
    except (TypeError, ValueError):
        selector_key = repr(selector)
    working = snapshot.get("working_geometry")
    if isinstance(working, Mapping):
        provenance_key = (
            str(working.get("geometry_basis") or ""),
            str(working.get("coordinate_mode") or ""),
            tuple(str(value) for value in (working.get("assumed_fields") or ())),
            tuple(str(value) for value in (working.get("replaced_fields") or ())),
        )
    else:
        provenance_key = ()
    try:
        metadata_key = json.dumps(
            observation_values,
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):
        metadata_key = repr(observation_values)
    return (
        _RESOLVER_SCHEMA,
        file_state,
        str(snapshot.get("source_id") or ""),
        str(snapshot.get("acquisition_id") or ""),
        str(snapshot.get("backing_source_id") or ""),
        _series_index(snapshot),
        str(snapshot.get("series_identity") or ""),
        _channel_index(snapshot),
        _time_index(snapshot),
        _position_index(snapshot),
        selector_key,
        _target_units(snapshot),
        str(snapshot.get("source_checksum") or ""),
        str(snapshot.get("geometry_checksum") or ""),
        local_key,
        provenance_key,
        metadata_key,
    )


def _deduplicated_candidates(candidates):
    result = []
    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        key = (
            candidate.family,
            candidate.semantic_kind,
            candidate.source_label,
            candidate.raw_position_xyz,
            candidate.source_units_xyz,
            candidate.series_index,
            candidate.series_identity,
            candidate.scene_identity,
            candidate.tile_identity,
            candidate.channel_index,
            candidate.channel_identity,
            candidate.time_index,
            candidate.time_identity,
            candidate.position_index,
            candidate.status,
            candidate.rejection_reason,
        )
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return tuple(result)


def _observation_mapping(value) -> Mapping[str, Any] | None:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return value if isinstance(value, Mapping) else None


def _observation_identity_issue(observation, snapshot) -> str:
    comparisons = (
        ("series index", observation.get("series_index"), _series_index(snapshot)),
        (
            "series identity",
            observation.get("series_identity"),
            snapshot.get("series_identity"),
        ),
        (
            "scene identity",
            observation.get("scene_identity"),
            snapshot.get("scene_identity"),
        ),
        (
            "tile identity",
            observation.get("tile_identity"),
            snapshot.get("tile_identity"),
        ),
        ("channel index", observation.get("channel_index"), _channel_index(snapshot)),
        (
            "channel identity",
            observation.get("channel_identity"),
            snapshot.get("channel_identity"),
        ),
        ("time index", observation.get("time_index"), _time_index(snapshot)),
        (
            "time identity",
            observation.get("time_identity"),
            snapshot.get("time_identity"),
        ),
        (
            "position index",
            observation.get("scene_index")
            if observation.get("scene_index") is not None
            else observation.get("tile_index"),
            _position_index(snapshot),
        ),
    )
    for label, observed, selected in comparisons:
        if observed in (None, "") or selected in (None, ""):
            continue
        if str(observed) != str(selected):
            return (
                f"Persisted position {label} {observed!r} does not match the "
                f"selected source {label} {selected!r}."
            )
    return ""


def _fingerprint_comparison(source_fields):
    observed = _fingerprints(source_fields.get("source_fingerprints"))
    if not observed:
        return (), (), "not-recorded", ()
    current = tuple(_file_fingerprint(value.path) for value in observed)
    warnings = []
    state = "verified"
    for prior, actual in zip(observed, current):
        if actual.size is None or actual.mtime_ns is None:
            state = "unavailable" if state == "verified" else state
            warnings.append(
                f"Position-evidence source fingerprint is unavailable: {prior.path}."
            )
        elif prior.size != actual.size or prior.mtime_ns != actual.mtime_ns:
            state = "mismatch"
            warnings.append(
                f"Position-evidence source fingerprint changed after cataloging: {prior.path}."
            )
    return observed, current, state, tuple(warnings)


def _checksum_comparison(source_fields, snapshot):
    observed = source_fields.get("source_checksum")
    current = snapshot.get("source_checksum")
    if observed and current:
        state = "verified" if str(observed) == str(current) else "mismatch"
    elif observed:
        state = str(source_fields.get("checksum_state") or "unavailable")
    else:
        state = "not-recorded"
    warnings = ()
    if state == "mismatch":
        warnings = (
            "Persisted position evidence and the current backing source report "
            "different checksums.",
        )
    return observed, current, state, warnings


def _candidate_from_observation(observation, snapshot) -> PositionCandidate:
    payload = dict(observation)
    source_fields = payload.get("source_fields") or {}
    if not isinstance(source_fields, Mapping):
        source_fields = {}
    family = str(source_fields.get("family") or "").strip()
    source_label = str(
        source_fields.get("source_label")
        or payload.get("reader_backend")
        or "persisted acquisition position"
    )
    semantic_kind = str(payload.get("semantic_meaning") or "unknown")
    (
        observed_fingerprints,
        current_fingerprints,
        fingerprint_state,
        fingerprint_warnings,
    ) = _fingerprint_comparison(source_fields)
    source_checksum, current_checksum, checksum_state, checksum_warnings = (
        _checksum_comparison(source_fields, snapshot)
    )
    warnings = _unique_text(
        (
            *tuple(payload.get("warnings") or ()),
            *fingerprint_warnings,
            *checksum_warnings,
        )
    )
    rejection_reason = str(source_fields.get("rejection_reason") or "")
    identity_issue = _observation_identity_issue(payload, snapshot)
    if not family:
        identity_issue = identity_issue or (
            "Persisted position evidence predates position-family ownership; "
            "explicit source recovery enrichment or re-import is required."
        )
    if identity_issue:
        return PositionCandidate(
            family=family or "legacy-unresolved",
            semantic_kind=semantic_kind,
            position_xyz=(None, None, None),
            units_xyz=(None, None, None),
            source_label=source_label,
            source_fields=tuple(
                str(value) for value in (source_fields.get("fields") or ())
            ),
            raw_position_xyz=payload.get("raw_position_xyz"),
            source_units_xyz=payload.get("raw_units_xyz"),
            series_index=_optional_index(payload.get("series_index")),
            series_identity=payload.get("series_identity"),
            scene_identity=payload.get("scene_identity"),
            tile_identity=payload.get("tile_identity"),
            channel_index=_optional_index(payload.get("channel_index")),
            channel_identity=payload.get("channel_identity"),
            time_index=_optional_index(payload.get("time_index")),
            time_identity=payload.get("time_identity"),
            position_index=_optional_index(
                payload.get("scene_index")
                if payload.get("scene_index") is not None
                else payload.get("tile_index")
            ),
            coordinate_frame=payload.get("coordinate_frame"),
            status="ambiguous" if family else "invalid",
            warnings=warnings,
            rejection_reason=identity_issue,
            source_fingerprint=observed_fingerprints,
            current_source_fingerprint=current_fingerprints,
            fingerprint_state=fingerprint_state,
            source_checksum=source_checksum,
            current_source_checksum=current_checksum,
            checksum_state=checksum_state,
            persisted_observation=payload,
        )

    declared_status = str(
        source_fields.get("candidate_status")
        or {
            "interpreted": "usable",
            "partial": "partial",
            "ambiguous": "ambiguous",
            "conflicting": "invalid",
            "unsupported-units": "ambiguous",
            "uninterpreted": "absent",
        }.get(str(payload.get("interpretation_status") or ""), "ambiguous")
    )
    if semantic_kind == "filename_grid":
        candidate = PositionCandidate(
            family=family,
            semantic_kind=semantic_kind,
            position_xyz=payload.get("normalized_position_xyz"),
            units_xyz=payload.get("normalized_units_xyz"),
            source_label=source_label,
            source_fields=tuple(
                str(value) for value in (source_fields.get("fields") or ())
            ),
            raw_position_xyz=payload.get("raw_position_xyz"),
            source_units_xyz=payload.get("raw_units_xyz"),
            conversion_factors=source_fields.get("conversion_factors"),
            series_index=_optional_index(payload.get("series_index")),
            series_identity=payload.get("series_identity"),
            scene_identity=payload.get("scene_identity"),
            tile_identity=payload.get("tile_identity"),
            channel_index=_optional_index(payload.get("channel_index")),
            channel_identity=payload.get("channel_identity"),
            time_index=_optional_index(payload.get("time_index")),
            time_identity=payload.get("time_identity"),
            position_index=_optional_index(
                payload.get("scene_index")
                if payload.get("scene_index") is not None
                else payload.get("tile_index")
            ),
            coordinate_frame=payload.get("coordinate_frame"),
            status=declared_status,
            warnings=warnings,
            rejection_reason=rejection_reason,
            source_fingerprint=observed_fingerprints,
            current_source_fingerprint=current_fingerprints,
            fingerprint_state=fingerprint_state,
            source_checksum=source_checksum,
            current_source_checksum=current_checksum,
            checksum_state=checksum_state,
            persisted_observation=payload,
        )
    else:
        candidate = _physical_candidate(
            family=family,
            semantic_kind=semantic_kind,
            raw_values=payload.get("raw_position_xyz"),
            source_units=payload.get("raw_units_xyz"),
            target_units=_target_units(snapshot),
            source_label=source_label,
            source_fields=tuple(
                str(value) for value in (source_fields.get("fields") or ())
            ),
            series_index=payload.get("series_index"),
            series_identity=payload.get("series_identity"),
            scene_identity=payload.get("scene_identity"),
            tile_identity=payload.get("tile_identity"),
            channel_index=payload.get("channel_index"),
            channel_identity=payload.get("channel_identity"),
            time_index=payload.get("time_index"),
            time_identity=payload.get("time_identity"),
            position_index=(
                payload.get("scene_index")
                if payload.get("scene_index") is not None
                else payload.get("tile_index")
            ),
            coordinate_frame=payload.get("coordinate_frame"),
            warnings=warnings,
            rejection_reason=rejection_reason,
            status=declared_status,
            source_fingerprint=observed_fingerprints,
        )
        candidate = replace(
            candidate,
            current_source_fingerprint=current_fingerprints,
            fingerprint_state=fingerprint_state,
            source_checksum=source_checksum,
            current_source_checksum=current_checksum,
            checksum_state=checksum_state,
            persisted_observation=payload,
        )
    if candidate.status == "usable" and (
        fingerprint_state in {"mismatch", "unavailable"}
        or checksum_state == "mismatch"
    ):
        candidate = replace(candidate, status="partial")
    return candidate


def resolve_position_candidates(
    snapshot: Mapping[str, Any],
) -> tuple[PositionCandidate, ...]:
    """Resolve only model-owned position observations for one stitching snapshot."""
    key = _cache_key(snapshot)
    cached = _POSITION_CACHE.get(key)
    if cached is not None:
        return cached
    candidates: list[PositionCandidate] = []
    for raw_observation in snapshot.get("stage_position_observations") or ():
        observation = _observation_mapping(raw_observation)
        if observation is None:
            candidates.append(
                _diagnostic_candidate(
                    "persisted",
                    "unknown",
                    "persisted acquisition position",
                    "invalid",
                    "Persisted position evidence is not a typed observation mapping.",
                    snapshot=snapshot,
                )
            )
            continue
        candidates.append(_candidate_from_observation(observation, snapshot))
    result = _deduplicated_candidates(candidates)
    _POSITION_CACHE[key] = result
    return result


def _matrix_tuple(matrix) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row)
        for row in np.asarray(matrix, dtype=float)
    )


def _world_origin(snapshot) -> np.ndarray:
    value = snapshot.get("world_affine")
    if value is None:
        value = snapshot.get("world_index_affine")
    if value is None:
        value = np.eye(4)
    return np.asarray(value, dtype=float).reshape(4, 4)[:3, 3].copy()


def _position_evidence_geometry_revision(snapshot) -> str:
    return str(
        snapshot.get("position_evidence_geometry_revision")
        or snapshot.get("channel_local_geometry_revision")
        or ""
    )


def _best_family_candidate(candidates, family):
    matches = [
        candidate
        for candidate in candidates
        if candidate.family == family
        and candidate.status in {"usable", "partial"}
        and candidate.usable_axes
    ]
    matches.sort(key=lambda value: 0 if value.status == "usable" else 1)
    return matches[0] if matches else None


def _assess_family(candidates_by_tile, family):
    selected = tuple(
        _best_family_candidate(candidates, family)
        for candidates in candidates_by_tile
    )
    available = [index for index, candidate in enumerate(selected) if candidate]
    if len(available) < 2:
        return None, "fewer than two usable tile positions"
    common = set(_AXES)
    for index in available:
        common.intersection_update(selected[index].usable_axes)
    ordered_common = tuple(axis for axis in _AXES if axis in common)
    if not ordered_common:
        return None, "no coordinate axis is usable across the candidate family"
    for axis in ordered_common:
        units = {
            selected[index].units_xyz[_AXES.index(axis)]
            for index in available
            if selected[index].semantic_kind != "filename_grid"
        }
        if len(units) > 1:
            return None, f"{axis} target working units differ across tiles"
    frames = {
        selected[index].coordinate_frame
        for index in available
        if selected[index].coordinate_frame
    }
    if len(frames) > 1:
        return None, "candidate coordinate-frame identifiers do not match"
    positions = np.vstack(
        [selected[index].position_array() for index in available]
    )
    axes = [_AXES.index(axis) for axis in ordered_common]
    if not np.any(np.ptp(positions[:, axes], axis=0) > 1e-9):
        return None, "tile positions do not vary on a common usable axis"
    return (selected, tuple(available), ordered_common), ""


def _diagnostic_layout(
    snapshots,
    settings,
    candidates_by_tile,
    *,
    status,
    warning,
):
    identity = _matrix_tuple(np.eye(4))
    anchor_index = next(
        (index for index, snapshot in enumerate(snapshots) if snapshot.get("anchor")),
        0,
    )
    records = []
    for snapshot, candidates in zip(snapshots, candidates_by_tile):
        candidate = next(
            (
                value
                for value in candidates
                if value.status in {"ambiguous", "invalid", "partial", "usable"}
            ),
            candidates[0] if candidates else None,
        )
        records.append(
            InitialPlacementRecord(
                tile_id=str(snapshot["tile_id"]),
                display_name=str(snapshot.get("display_name") or snapshot["tile_id"]),
                candidate=candidate,
                target_world_origin=tuple(float(value) for value in _world_origin(snapshot)),
                kept_current_pose=True,
                warnings=(warning,),
                geometry_revision=_position_evidence_geometry_revision(snapshot),
            )
        )
    return InitialPlacementResult(
        mode=str(settings.mode),
        label="no coherent automatic layout",
        selected_family=None,
        semantic_kind=None,
        status=status,
        requires_review=True,
        placement_deltas={
            str(snapshot["tile_id"]): identity for snapshot in snapshots
        },
        common_axes=(),
        anchor_tile_id=str(snapshots[anchor_index]["tile_id"]),
        anchor_display_name=str(
            snapshots[anchor_index].get("display_name")
            or snapshots[anchor_index]["tile_id"]
        ),
        records=tuple(records),
        warnings=(warning,),
        assumptions=(),
        stage_axis_mapping=settings.stage_axis_mapping(),
        grid_overlap_percent=None,
        grid_step=None,
        summary=warning,
    )


def _mapped_stage_delta(delta, common_axes, settings):
    source = np.asarray(delta, dtype=float).copy()
    for index, invert in enumerate(
        (
            settings.invert_stage_x,
            settings.invert_stage_y,
            settings.invert_stage_z,
        )
    ):
        if invert:
            source[index] *= -1.0
    mapped = source.copy()
    mapped_axes = list(common_axes)
    if settings.stage_xy_mapping == "swap_xy":
        mapped[0], mapped[1] = source[1], source[0]
        mapped_axes = [
            {"X": "Y", "Y": "X"}.get(axis, axis) for axis in mapped_axes
        ]
    elif settings.stage_xy_mapping != "identity":
        raise ValueError(
            "Stage XY mapping must be 'identity' or 'swap_xy'."
        )
    return mapped, tuple(axis for axis in _AXES if axis in set(mapped_axes))


def infer_initial_layout(
    snapshots: Sequence[Mapping[str, Any]],
    settings: InitialPlacementSettings,
) -> InitialPlacementResult:
    """Resolve one coherent family and return translation-only placement deltas."""
    snapshots = tuple(snapshots)
    if not snapshots:
        raise RuntimeError("At least one stitching tile is required.")
    mode = str(settings.mode or "current").strip().lower()
    if mode not in {"current", "metadata", "filename", "auto"}:
        raise ValueError(f"Unsupported initial-placement mode {mode!r}.")
    overlap = float(settings.grid_overlap_percent)
    if not math.isfinite(overlap) or not 0 <= overlap < 100:
        raise ValueError("Filename grid overlap must be finite and in [0, 100).")
    if settings.stage_xy_mapping not in {"identity", "swap_xy"}:
        raise ValueError("Stage XY mapping must be 'identity' or 'swap_xy'.")
    identity = _matrix_tuple(np.eye(4))
    anchor_index = next(
        (index for index, snapshot in enumerate(snapshots) if snapshot.get("anchor")),
        0,
    )
    if mode == "current":
        records = tuple(
            InitialPlacementRecord(
                tile_id=str(snapshot["tile_id"]),
                display_name=str(snapshot.get("display_name") or snapshot["tile_id"]),
                candidate=None,
                target_world_origin=tuple(float(value) for value in _world_origin(snapshot)),
                kept_current_pose=True,
                applied_axes=(),
                geometry_revision=_position_evidence_geometry_revision(snapshot),
            )
            for snapshot in snapshots
        )
        return InitialPlacementResult(
            mode="current",
            label="current MADI3D poses",
            selected_family=None,
            semantic_kind=None,
            status="usable",
            requires_review=False,
            placement_deltas={
                str(snapshot["tile_id"]): identity for snapshot in snapshots
            },
            common_axes=("X", "Y", "Z"),
            anchor_tile_id=str(snapshots[anchor_index]["tile_id"]),
            anchor_display_name=str(
                snapshots[anchor_index].get("display_name")
                or snapshots[anchor_index]["tile_id"]
            ),
            records=records,
            warnings=(),
            assumptions=(),
            stage_axis_mapping=settings.stage_axis_mapping(),
            grid_overlap_percent=None,
            grid_step=None,
            summary="Current MADI3D poses used as the initial layout.",
        )

    metadata_candidates = tuple(
        resolve_position_candidates(snapshot) for snapshot in snapshots
    )
    chosen = None
    family_issues = []
    if mode in {"metadata", "auto"}:
        for family in _FAMILY_PRECEDENCE:
            assessment, issue = _assess_family(metadata_candidates, family)
            if assessment is not None:
                chosen = (family, *assessment)
                break
            if any(
                candidate.family == family
                for candidates in metadata_candidates
                for candidate in candidates
            ):
                family_issues.append(f"{family}: {issue}")
    candidates_by_tile = metadata_candidates
    if chosen is None and mode in {"filename", "auto"}:
        filename_candidates = tuple(
            tuple(
                candidate
                for candidate in candidates
                if candidate.family == "filename"
            )
            for candidates in metadata_candidates
        )
        assessment, issue = _assess_family(filename_candidates, "filename")
        if assessment is not None:
            chosen = ("filename", *assessment)
            candidates_by_tile = filename_candidates
        elif any(filename_candidates):
            family_issues.append(f"filename: {issue}")
            if mode == "filename":
                candidates_by_tile = filename_candidates
    if chosen is None:
        has_diagnostics = any(
            candidates for candidates in candidates_by_tile
        )
        issue_text = "; ".join(family_issues)
        warning = (
            "No coherent position family provides at least two varying tile "
            "positions on a shared usable axis."
            + (f" {issue_text}" if issue_text else "")
        )
        return _diagnostic_layout(
            snapshots,
            settings,
            candidates_by_tile,
            status="ambiguous" if has_diagnostics else "absent",
            warning=warning,
        )

    family, selected, available, source_common_axes = chosen
    available_set = set(available)
    reference_index = (
        anchor_index if anchor_index in available_set else available[0]
    )
    reference_candidate = selected[reference_index]
    reference_position = reference_candidate.position_array()
    reference_origin = _world_origin(snapshots[reference_index])
    semantic_kind = reference_candidate.semantic_kind
    stage_semantic = semantic_kind in _STAGE_SEMANTICS
    common_axes = source_common_axes
    if stage_semantic:
        _ignored, common_axes = _mapped_stage_delta(
            np.zeros(3), source_common_axes, settings
        )

    grid_step = np.ones(3, dtype=float)
    overlap_percent = None
    assumptions: list[str] = []
    if semantic_kind == "filename_grid":
        sizes = []
        for index in available:
            snapshot = snapshots[index]
            world = snapshot.get("world_affine")
            if world is None:
                world = snapshot.get("world_index_affine")
            linear = np.asarray(world, dtype=float).reshape(4, 4)[:3, :3]
            dims = np.asarray(_triple(snapshot.get("dims"), 1.0), dtype=float)
            sizes.append(np.linalg.norm(linear, axis=0) * dims)
        overlap_percent = overlap
        grid_step = np.maximum(
            1e-9,
            np.median(np.vstack(sizes), axis=0)
            * (1.0 - overlap_percent / 100.0),
        )
        assumptions.append(
            f"Grid indices use median tile extent with {overlap_percent:.6g}% overlap."
        )

    deltas = {}
    records = []
    warnings: list[str] = []
    missing_names = []
    for index, snapshot in enumerate(snapshots):
        tile_id = str(snapshot["tile_id"])
        display_name = str(snapshot.get("display_name") or tile_id)
        current_origin = _world_origin(snapshot)
        candidate = selected[index]
        if index not in available_set or candidate is None:
            deltas[tile_id] = identity
            missing_names.append(display_name)
            records.append(
                InitialPlacementRecord(
                    tile_id=tile_id,
                    display_name=display_name,
                    candidate=None,
                    target_world_origin=tuple(float(value) for value in current_origin),
                    kept_current_pose=True,
                    applied_axes=(),
                    warnings=(
                        f"No usable {family} position; current pose retained.",
                    ),
                    geometry_revision=_position_evidence_geometry_revision(snapshot),
                )
            )
            continue
        source_delta = candidate.position_array() - reference_position
        if semantic_kind == "filename_grid":
            source_delta = source_delta * grid_step
        if stage_semantic:
            evidence_delta, applied_axes = _mapped_stage_delta(
                source_delta, source_common_axes, settings
            )
        else:
            evidence_delta = source_delta
            applied_axes = source_common_axes
        current_relative = current_origin - reference_origin
        target_relative = current_relative.copy()
        for axis in applied_axes:
            axis_index = _AXES.index(axis)
            target_relative[axis_index] = evidence_delta[axis_index]
        target_origin = reference_origin + target_relative
        placement = np.eye(4, dtype=float)
        placement[:3, 3] = target_origin - current_origin
        if index == anchor_index:
            placement = np.eye(4, dtype=float)
            target_origin = current_origin
        deltas[tile_id] = _matrix_tuple(placement)
        record_warnings = _unique_text(
            (
                *candidate.warnings,
                candidate.rejection_reason,
            )
        )
        warnings.extend(record_warnings)
        records.append(
            InitialPlacementRecord(
                tile_id=tile_id,
                display_name=display_name,
                candidate=candidate,
                target_world_origin=tuple(float(value) for value in target_origin),
                kept_current_pose=False,
                applied_axes=applied_axes,
                warnings=record_warnings,
                geometry_revision=_position_evidence_geometry_revision(snapshot),
            )
        )

    if missing_names:
        warnings.append(
            f"{len(missing_names)} tile(s) had no usable {family} position and "
            "kept their current MADI3D poses: "
            + ", ".join(missing_names)
        )
    selected_candidates = [
        candidate for candidate in selected if candidate is not None
    ]
    partial = bool(missing_names) or any(
        candidate.status != "usable" or candidate.warnings
        for candidate in selected_candidates
    )
    status = "partial" if partial else "usable"
    requires_review = status != "usable" or bool(warnings)
    anchor_name = str(
        snapshots[anchor_index].get("display_name")
        or snapshots[anchor_index]["tile_id"]
    )
    summary = (
        f"Initial placement selected one coherent {family} family "
        f"({semantic_kind}); applied axes: {', '.join(common_axes)}; "
        f"anchor tile: {anchor_name}."
    )
    if reference_index != anchor_index:
        summary += (
            " The anchor has no usable position, so another positioned tile "
            "defines the metadata-difference reference while the anchor remains fixed."
        )
    if overlap_percent is not None:
        summary += f" Filename/grid overlap: {overlap_percent:.2f}%."
    if missing_names:
        summary += f" {len(missing_names)} tile(s) retained their current poses."
    return InitialPlacementResult(
        mode=mode,
        label=family,
        selected_family=family,
        semantic_kind=semantic_kind,
        status=status,
        requires_review=requires_review,
        placement_deltas=deltas,
        common_axes=common_axes,
        anchor_tile_id=str(snapshots[anchor_index]["tile_id"]),
        anchor_display_name=anchor_name,
        records=tuple(records),
        warnings=tuple(warnings),
        assumptions=tuple(assumptions),
        stage_axis_mapping=settings.stage_axis_mapping(),
        grid_overlap_percent=overlap_percent,
        grid_step=(
            tuple(float(value) for value in grid_step)
            if overlap_percent is not None
            else None
        ),
        summary=summary,
    )


__all__ = [
    "InitialPlacementRecord",
    "InitialPlacementResult",
    "InitialPlacementSettings",
    "PositionCandidate",
    "SourceFingerprint",
    "catalog_position_candidates",
    "clear_position_cache",
    "filename_position_candidate",
    "h5j_position_candidate",
    "imagej_position_candidate",
    "infer_initial_layout",
    "micromanager_position_candidates",
    "ome_position_candidates",
    "resolve_position_candidates",
]
