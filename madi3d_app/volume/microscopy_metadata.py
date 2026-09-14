"""Typed, GUI-independent microscopy source and acquisition evidence.

The parent volume records retain normalized operational identity and routing
facts. ``reported_*`` fields here preserve reader evidence and may legitimately
differ from those normalized facts when the source is incomplete or conflicting.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Optional


class MicroscopyMetadataError(ValueError):
    """Raised when microscopy evidence cannot be represented without loss."""


CHECKSUM_STATES = frozenset(
    {
        "unknown",
        "not-computed",
        "computed",
        "verified",
        "mismatch",
        "unavailable",
    }
)

STAGE_INTERPRETATION_STATUSES = frozenset(
    {
        "uninterpreted",
        "partial",
        "interpreted",
        "ambiguous",
        "conflicting",
        "unsupported-units",
    }
)


def _json_safe(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return copy.deepcopy(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MicroscopyMetadataError(
                f"{path} contains a non-finite number; microscopy metadata must "
                "contain finite JSON values."
            )
        return float(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise MicroscopyMetadataError(
            f"{path} contains binary data ({type(value).__name__}); exclude binary "
            "pixel payloads or attachments before storing microscopy metadata."
        )
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise MicroscopyMetadataError(
                    f"{path} contains a non-string object key {key!r}; JSON object "
                    "keys must be strings."
                )
            result[key] = _json_safe(item, f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [
            _json_safe(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise MicroscopyMetadataError(
        f"{path} contains unsupported value type {type(value).__name__}; microscopy "
        "metadata must use JSON null, booleans, finite numbers, strings, arrays, "
        "and objects."
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MicroscopyMetadataError(f"{label} must be a JSON object.")
    result = _json_safe(value, label)
    if not isinstance(result, dict):  # pragma: no cover - established above
        raise MicroscopyMetadataError(f"{label} must be a JSON object.")
    return result


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MicroscopyMetadataError(f"{label} must be a non-empty string.")
    return value.strip()


def _optional_string(value: Any, label: str) -> Optional[str]:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise MicroscopyMetadataError(f"{label} must be a string or null.")
    result = value.strip()
    return result or None


def _optional_index(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MicroscopyMetadataError(
            f"{label} must be a non-negative integer or null."
        )
    return int(value)


def _optional_positive_number(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MicroscopyMetadataError(
            f"{label} must be a positive finite number or null."
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise MicroscopyMetadataError(
            f"{label} must be a positive finite number or null."
        )
    return result


def _warnings(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise MicroscopyMetadataError(f"{label} must be an array of strings.")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise MicroscopyMetadataError(
                f"{label} must contain only non-empty strings."
            )
        result.append(item.strip())
    return tuple(result)


def _xyz_values(value: Any, label: str) -> tuple[Optional[float], ...]:
    if value is None:
        return (None, None, None)
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 3:
        raise MicroscopyMetadataError(
            f"{label} must contain exactly three numeric or null axis values."
        )
    result = []
    for index, item in enumerate(value):
        if item is None:
            result.append(None)
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise MicroscopyMetadataError(
                f"{label}[{index}] must be a finite number or null."
            )
        number = float(item)
        if not math.isfinite(number):
            raise MicroscopyMetadataError(
                f"{label}[{index}] must be a finite number or null."
            )
        result.append(number)
    return tuple(result)


def _xyz_units(value: Any, label: str) -> tuple[Optional[str], ...]:
    if value is None:
        return (None, None, None)
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 3:
        raise MicroscopyMetadataError(
            f"{label} must contain exactly three string or null axis units."
        )
    result = []
    for index, item in enumerate(value):
        if item is None or item == "":
            result.append(None)
            continue
        if not isinstance(item, str):
            raise MicroscopyMetadataError(
                f"{label}[{index}] must be a string or null."
            )
        result.append(item.strip() or None)
    return tuple(result)


def _record_payload(
    value: Mapping[str, Any],
    *,
    record_name: str,
    known_fields: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MicroscopyMetadataError(f"{record_name} must be a JSON object.")
    payload = copy.deepcopy(dict(value))
    explicit_additional = payload.pop("additional_fields", {})
    additional = _mapping(
        explicit_additional, f"{record_name}.additional_fields"
    )
    for key in list(payload):
        if key not in known_fields:
            if key in additional:
                raise MicroscopyMetadataError(
                    f"{record_name} repeats unknown field {key!r}."
                )
            additional[key] = payload.pop(key)
    payload["additional_fields"] = additional
    return payload


def _serialized_record(
    known: Mapping[str, Any],
    additional_fields: Mapping[str, Any],
    *,
    record_name: str,
) -> dict[str, Any]:
    additional = _mapping(additional_fields, f"{record_name}.additional_fields")
    conflicts = sorted(set(known).intersection(additional))
    if conflicts:
        raise MicroscopyMetadataError(
            f"{record_name}.additional_fields conflicts with typed fields: "
            + ", ".join(conflicts)
            + "."
        )
    result = additional
    result.update(copy.deepcopy(dict(known)))
    normalized = _json_safe(result, record_name)
    if not isinstance(normalized, dict):  # pragma: no cover - constructed above
        raise MicroscopyMetadataError(f"{record_name} must be a JSON object.")
    return normalized


def _additional_fields(
    value: Any,
    *,
    record_name: str,
    known_fields: frozenset[str],
) -> dict[str, Any]:
    result = _mapping(value, f"{record_name}.additional_fields")
    conflicts = sorted(set(result).intersection(known_fields))
    if conflicts:
        raise MicroscopyMetadataError(
            f"{record_name}.additional_fields conflicts with typed fields: "
            + ", ".join(conflicts)
            + "."
        )
    return result


class _MicroscopyRecord:
    @classmethod
    def _known_fields(cls) -> frozenset[str]:
        return frozenset(
            item.name for item in fields(cls) if item.name != "additional_fields"
        )

    @staticmethod
    def _serialized_value(value: Any) -> Any:
        if isinstance(value, _MicroscopyRecord):
            return value.to_dict()
        if isinstance(value, Mapping):
            return {
                key: _MicroscopyRecord._serialized_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                _MicroscopyRecord._serialized_value(item) for item in value
            ]
        return copy.deepcopy(value)

    def to_dict(self) -> dict[str, Any]:
        known = {
            item.name: self._serialized_value(getattr(self, item.name))
            for item in fields(self)
            if item.name != "additional_fields"
        }
        return _serialized_record(
            known,
            self.additional_fields,
            record_name=type(self).__name__,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        return cls(
            **_record_payload(
                value,
                record_name=cls.__name__,
                known_fields=cls._known_fields(),
            )
        )


@dataclass(frozen=True)
class SourceMemberRecord(_MicroscopyRecord):
    """One member of a single logical multi-file microscopy source bundle."""

    member_id: str
    path: str = ""
    role: str = "unspecified"
    media_type: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    checksum_state: str = "unknown"
    structured_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    additional_fields: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self, "member_id", _required_string(self.member_id, "member_id")
        )
        object.__setattr__(self, "path", _optional_string(self.path, "path") or "")
        object.__setattr__(self, "role", _required_string(self.role, "role"))
        object.__setattr__(
            self, "media_type", _optional_string(self.media_type, "media_type")
        )
        if self.size_bytes is not None:
            object.__setattr__(
                self,
                "size_bytes",
                _optional_index(self.size_bytes, "size_bytes"),
            )
        object.__setattr__(
            self, "checksum", _optional_string(self.checksum, "checksum")
        )
        checksum_state = _required_string(self.checksum_state, "checksum_state")
        if checksum_state not in CHECKSUM_STATES:
            raise MicroscopyMetadataError(
                f"Unsupported checksum_state {checksum_state!r}; expected one of "
                f"{sorted(CHECKSUM_STATES)!r}."
            )
        object.__setattr__(self, "checksum_state", checksum_state)
        object.__setattr__(
            self,
            "structured_metadata",
            _mapping(self.structured_metadata, "structured_metadata"),
        )
        object.__setattr__(self, "warnings", _warnings(self.warnings, "warnings"))
        object.__setattr__(
            self,
            "additional_fields",
            _additional_fields(
                self.additional_fields,
                record_name="SourceMemberRecord",
                known_fields=self._known_fields(),
            ),
        )


@dataclass(frozen=True)
class StagePositionObservation(_MicroscopyRecord):
    """Raw stage evidence; never an image origin or scene pose by implication."""

    raw_position_xyz: tuple[Optional[float], ...] = (None, None, None)
    raw_units_xyz: tuple[Optional[str], ...] = (None, None, None)
    normalized_position_xyz: tuple[Optional[float], ...] = (None, None, None)
    normalized_units_xyz: tuple[Optional[str], ...] = (None, None, None)
    semantic_meaning: str = "unknown"
    coordinate_frame: str = "unknown"
    source_fields: dict[str, Any] = field(default_factory=dict)
    series_identity: Optional[str] = None
    series_index: Optional[int] = None
    scene_identity: Optional[str] = None
    scene_index: Optional[int] = None
    tile_identity: Optional[str] = None
    tile_index: Optional[int] = None
    channel_identity: Optional[str] = None
    channel_index: Optional[int] = None
    time_identity: Optional[str] = None
    time_index: Optional[int] = None
    interpretation_status: str = "uninterpreted"
    warnings: tuple[str, ...] = ()
    reader_backend: Optional[str] = None
    reader_version: Optional[str] = None
    source_path: Optional[str] = None
    source_member_id: Optional[str] = None
    additional_fields: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "raw_position_xyz",
            _xyz_values(self.raw_position_xyz, "raw_position_xyz"),
        )
        object.__setattr__(
            self, "raw_units_xyz", _xyz_units(self.raw_units_xyz, "raw_units_xyz")
        )
        object.__setattr__(
            self,
            "normalized_position_xyz",
            _xyz_values(self.normalized_position_xyz, "normalized_position_xyz"),
        )
        object.__setattr__(
            self,
            "normalized_units_xyz",
            _xyz_units(self.normalized_units_xyz, "normalized_units_xyz"),
        )
        object.__setattr__(
            self,
            "semantic_meaning",
            _required_string(self.semantic_meaning, "semantic_meaning"),
        )
        object.__setattr__(
            self,
            "coordinate_frame",
            _required_string(self.coordinate_frame, "coordinate_frame"),
        )
        object.__setattr__(
            self, "source_fields", _mapping(self.source_fields, "source_fields")
        )
        for field_name in (
            "series_identity",
            "scene_identity",
            "tile_identity",
            "channel_identity",
            "time_identity",
            "reader_backend",
            "reader_version",
            "source_path",
            "source_member_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_string(getattr(self, field_name), field_name),
            )
        for field_name in (
            "series_index",
            "scene_index",
            "tile_index",
            "channel_index",
            "time_index",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_index(getattr(self, field_name), field_name),
            )
        status = _required_string(
            self.interpretation_status, "interpretation_status"
        )
        if status not in STAGE_INTERPRETATION_STATUSES:
            raise MicroscopyMetadataError(
                f"Unsupported interpretation_status {status!r}; expected one of "
                f"{sorted(STAGE_INTERPRETATION_STATUSES)!r}."
            )
        object.__setattr__(self, "interpretation_status", status)
        object.__setattr__(self, "warnings", _warnings(self.warnings, "warnings"))
        object.__setattr__(
            self,
            "additional_fields",
            _additional_fields(
                self.additional_fields,
                record_name="StagePositionObservation",
                known_fields=self._known_fields(),
            ),
        )


@dataclass(frozen=True)
class MicroscopySourceMetadata(_MicroscopyRecord):
    """Reader and container evidence shared by every channel using one source."""

    reader_backend: str
    reader_version: Optional[str] = None
    reported_vendor_format: Optional[str] = None
    reported_primary_source_path: Optional[str] = None
    source_members: tuple[SourceMemberRecord, ...] = ()
    reported_source_checksum: Optional[str] = None
    checksum_state: str = "unknown"
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    source_series_evidence: Any = field(default_factory=dict)
    source_axis_evidence: Any = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    additional_fields: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "reader_backend",
            _required_string(self.reader_backend, "reader_backend"),
        )
        for field_name in (
            "reader_version",
            "reported_vendor_format",
            "reported_primary_source_path",
            "reported_source_checksum",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_string(getattr(self, field_name), field_name),
            )
        members = tuple(
            item
            if isinstance(item, SourceMemberRecord)
            else SourceMemberRecord.from_dict(item)
            for item in (self.source_members or ())
        )
        member_ids = [item.member_id for item in members]
        if len(member_ids) != len(set(member_ids)):
            raise MicroscopyMetadataError(
                "source_members must contain unique member_id values."
            )
        object.__setattr__(self, "source_members", members)
        checksum_state = _required_string(self.checksum_state, "checksum_state")
        if checksum_state not in CHECKSUM_STATES:
            raise MicroscopyMetadataError(
                f"Unsupported checksum_state {checksum_state!r}; expected one of "
                f"{sorted(CHECKSUM_STATES)!r}."
            )
        object.__setattr__(self, "checksum_state", checksum_state)
        object.__setattr__(
            self, "raw_metadata", _mapping(self.raw_metadata, "raw_metadata")
        )
        object.__setattr__(
            self,
            "source_series_evidence",
            _json_safe(self.source_series_evidence, "source_series_evidence"),
        )
        object.__setattr__(
            self,
            "source_axis_evidence",
            _json_safe(self.source_axis_evidence, "source_axis_evidence"),
        )
        object.__setattr__(self, "warnings", _warnings(self.warnings, "warnings"))
        object.__setattr__(
            self,
            "additional_fields",
            _additional_fields(
                self.additional_fields,
                record_name="MicroscopySourceMetadata",
                known_fields=self._known_fields(),
            ),
        )


@dataclass(frozen=True)
class MicroscopyAcquisitionMetadata(_MicroscopyRecord):
    """Series, instrument, stage, and explicit import-decision evidence."""

    acquisition_timestamp: Optional[str] = None
    microscope_vendor: Optional[str] = None
    microscope_model: Optional[str] = None
    acquisition_software: Optional[str] = None
    reported_acquisition_software_version: Optional[str] = None
    objective_magnification: Optional[float] = None
    objective_numerical_aperture: Optional[float] = None
    objective_immersion: Optional[str] = None
    objective_model: Optional[str] = None
    series_identity: Optional[str] = None
    scene_identity: Optional[str] = None
    tile_identity: Optional[str] = None
    reported_scan_direction: Optional[str] = None
    stage_positions: tuple[StagePositionObservation, ...] = ()
    normalized_metadata: dict[str, Any] = field(default_factory=dict)
    import_decisions: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    additional_fields: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for field_name in (
            "acquisition_timestamp",
            "microscope_vendor",
            "microscope_model",
            "acquisition_software",
            "reported_acquisition_software_version",
            "objective_immersion",
            "objective_model",
            "series_identity",
            "scene_identity",
            "tile_identity",
            "reported_scan_direction",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_string(getattr(self, field_name), field_name),
            )
        for field_name in (
            "objective_magnification",
            "objective_numerical_aperture",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_positive_number(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "stage_positions",
            tuple(
                item
                if isinstance(item, StagePositionObservation)
                else StagePositionObservation.from_dict(item)
                for item in (self.stage_positions or ())
            ),
        )
        object.__setattr__(
            self,
            "normalized_metadata",
            _mapping(self.normalized_metadata, "normalized_metadata"),
        )
        object.__setattr__(
            self,
            "import_decisions",
            _mapping(self.import_decisions, "import_decisions"),
        )
        object.__setattr__(self, "warnings", _warnings(self.warnings, "warnings"))
        object.__setattr__(
            self,
            "additional_fields",
            _additional_fields(
                self.additional_fields,
                record_name="MicroscopyAcquisitionMetadata",
                known_fields=self._known_fields(),
            ),
        )


@dataclass(frozen=True)
class MicroscopyChannelMetadata(_MicroscopyRecord):
    """Source channel evidence, distinct from display and decode routing state."""

    source_channel_identifier: str
    source_channel_name: Optional[str] = None
    excitation_wavelength: Optional[float] = None
    excitation_wavelength_units: Optional[str] = None
    emission_wavelength: Optional[float] = None
    emission_wavelength_units: Optional[str] = None
    detector_settings: dict[str, Any] = field(default_factory=dict)
    source_color: Any = None
    reported_scientific_role: Optional[str] = None
    normalized_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    additional_fields: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "source_channel_identifier",
            _required_string(
                self.source_channel_identifier, "source_channel_identifier"
            ),
        )
        for field_name in (
            "source_channel_name",
            "excitation_wavelength_units",
            "emission_wavelength_units",
            "reported_scientific_role",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_string(getattr(self, field_name), field_name),
            )
        for field_name in ("excitation_wavelength", "emission_wavelength"):
            object.__setattr__(
                self,
                field_name,
                _optional_positive_number(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "detector_settings",
            _mapping(self.detector_settings, "detector_settings"),
        )
        object.__setattr__(
            self, "source_color", _json_safe(self.source_color, "source_color")
        )
        object.__setattr__(
            self,
            "normalized_metadata",
            _mapping(self.normalized_metadata, "normalized_metadata"),
        )
        object.__setattr__(self, "warnings", _warnings(self.warnings, "warnings"))
        object.__setattr__(
            self,
            "additional_fields",
            _additional_fields(
                self.additional_fields,
                record_name="MicroscopyChannelMetadata",
                known_fields=self._known_fields(),
            ),
        )


__all__ = [
    "CHECKSUM_STATES",
    "MicroscopyAcquisitionMetadata",
    "MicroscopyChannelMetadata",
    "MicroscopyMetadataError",
    "MicroscopySourceMetadata",
    "SourceMemberRecord",
    "STAGE_INTERPRETATION_STATUSES",
    "StagePositionObservation",
]
