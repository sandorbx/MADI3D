"""Qt-independent inspection and frame-wise decoding of Leica LIF sources."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .microscopy_metadata import (
    MicroscopyAcquisitionMetadata,
    MicroscopyChannelMetadata,
    MicroscopySourceMetadata,
    SourceMemberRecord,
    StagePositionObservation,
)


LEICA_UNREADABLE_SOURCE = "unreadable-source"
LEICA_MALFORMED_METADATA = "malformed-metadata"
LEICA_UNSUPPORTED_VARIANT = "unsupported-variant"
LEICA_LIF_READER_VERSION = "2026.2.16"

_DIMENSION_IDS = {
    1: "X",
    2: "Y",
    3: "Z",
    4: "T",
    5: "λ",
    6: "A",
    7: "N",
    8: "Q",
    9: "Λ",
    10: "M",
    11: "L",
}
_CHANNEL_TAGS = {0: "Gray", 1: "Red", 2: "Green", 3: "Blue"}
_UNIT_FACTORS_TO_MICRON = {
    "m": 1_000_000.0,
    "meter": 1_000_000.0,
    "meters": 1_000_000.0,
    "metre": 1_000_000.0,
    "metres": 1_000_000.0,
    "mm": 1_000.0,
    "millimeter": 1_000.0,
    "millimeters": 1_000.0,
    "millimetre": 1_000.0,
    "millimetres": 1_000.0,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "micrometer": 1.0,
    "micrometers": 1.0,
    "micrometre": 1.0,
    "micrometres": 1.0,
    "nm": 0.001,
    "nanometer": 0.001,
    "nanometers": 0.001,
    "nanometre": 0.001,
    "nanometres": 0.001,
}
_OMIT = object()


class LeicaInspectionError(ValueError):
    """Categorized failure while inspecting or decoding a Leica LIF source."""

    def __init__(self, category: str, detail: str):
        self.category = str(category)
        self.detail = str(detail)
        super().__init__(f"Leica LIF {self.category}: {self.detail}")


@dataclass(frozen=True)
class LeicaSeriesInspection:
    """Detached metadata for one independently selectable LIF image."""

    index: int
    identity: str
    name: str
    source_path_name: str
    unique_id: Optional[str]
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    scalar_dtype: str
    scalar_bit_depth: Optional[int]
    channel_names: tuple[str, ...]
    channel_metadata: tuple[MicroscopyChannelMetadata, ...]
    channel_scalar_dtypes: tuple[str, ...]
    channel_scalar_bit_depths: tuple[Optional[int], ...]
    spacing: Optional[tuple[float, float, float]]
    space_units: Optional[tuple[str, str, str]]
    time_count: int
    time_interval: float
    time_units: str
    acquisition_metadata: MicroscopyAcquisitionMetadata
    missing_fields: tuple[str, ...]
    physical_geometry_diagnostics: tuple[str, ...]
    warnings: tuple[str, ...]
    raw_fields: dict[str, Any]


@dataclass(frozen=True)
class LeicaSourceInspection:
    """Detached metadata for one Leica LIF container and all its images."""

    source_path: str
    container_format: str
    source_metadata: MicroscopySourceMetadata
    series: tuple[LeicaSeriesInspection, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LeicaChannelPixels:
    """One complete channel plus the source contract used to decode it."""

    channel_index: Optional[int]
    array: np.ndarray
    source_axes: tuple[str, ...]
    source_shape: tuple[int, ...]
    scalar_dtype: str
    scalar_bit_depth: Optional[int]


def require_liffile():
    try:
        import liffile
    except ImportError as exc:  # pragma: no cover - production build dependency
        raise RuntimeError(
            "Leica LIF inspection requires liffile==2026.2.16."
        ) from exc
    version = str(getattr(liffile, "__version__", "") or "").strip()
    if version != LEICA_LIF_READER_VERSION:
        raise RuntimeError(
            "Leica LIF inspection requires the validated "
            f"liffile=={LEICA_LIF_READER_VERSION} API, but found "
            f"{version or 'an unknown version'}."
        )
    return liffile


def _check_cancel(cancel_check: Optional[Callable[[], bool]]) -> None:
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Leica LIF operation was cancelled.")


def _unique_text(values) -> tuple[str, ...]:
    result = []
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive_number(value: Any) -> Optional[float]:
    result = _number(value)
    return result if result is not None and result > 0.0 else None


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    try:
        exact = float(value) == float(result)
    except (TypeError, ValueError):
        exact = False
    return result if result > 0 and exact else None


def _nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    try:
        exact = float(value) == float(result)
    except (TypeError, ValueError):
        exact = False
    return result if result >= 0 and exact else None


def _safe_metadata_value(value: Any, path: str, exclusions: list[dict[str, str]]):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str, int)):
        return copy.deepcopy(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        exclusions.append({"path": path, "reason": "non-finite numeric metadata"})
        return _OMIT
    if isinstance(value, (datetime, np.datetime64)):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            exclusions.append({"path": path, "reason": "binary metadata attachment"})
            return _OMIT
        if "\x00" in text:
            exclusions.append({"path": path, "reason": "binary metadata attachment"})
            return _OMIT
        return text
    if isinstance(value, Mapping):
        result = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = str(raw_key)
            child = _safe_metadata_value(value[raw_key], f"{path}.{key}", exclusions)
            if child is not _OMIT:
                result[key] = child
        return result
    if isinstance(value, Sequence):
        result = []
        for index, item in enumerate(value):
            child = _safe_metadata_value(item, f"{path}[{index}]", exclusions)
            result.append(None if child is _OMIT else child)
        return result
    exclusions.append(
        {"path": path, "reason": f"unsupported metadata type {type(value).__name__}"}
    )
    return _OMIT


def _safe_metadata(value: Any, path: str, exclusions: list[dict[str, str]]):
    result = _safe_metadata_value(value, path, exclusions)
    return None if result is _OMIT else result


def _xml_mapping(element: Any) -> dict[str, Any]:
    """Preserve textual Leica XML without serializing pixel memory blocks."""
    result: dict[str, Any] = {
        "tag": str(getattr(element, "tag", "")),
        "attributes": {
            str(key): str(value)
            for key, value in sorted(
                dict(getattr(element, "attrib", {}) or {}).items(),
                key=lambda item: str(item[0]),
            )
        },
    }
    text = str(getattr(element, "text", "") or "").strip()
    if text:
        result["text"] = text
    children = [_xml_mapping(child) for child in list(element)]
    if children:
        result["children"] = children
    return result


def _find_elements(root: Any, tag: str) -> list[Any]:
    if root is None:
        return []
    return [item for item in root.iter() if str(item.tag).rsplit("}", 1)[-1] == tag]


def _first_attribute(root: Any, names: set[str]) -> Optional[str]:
    wanted = {_normalized_key(name) for name in names}
    for element in root.iter() if root is not None else ():
        for key, value in (getattr(element, "attrib", {}) or {}).items():
            if _normalized_key(key) in wanted and str(value or "").strip():
                return str(value).strip()
    return None


def _dimension_records(image: Any) -> dict[str, dict[str, Any]]:
    root = getattr(image, "xml_element", None)
    records: dict[str, dict[str, Any]] = {}
    duplicate_counts: dict[str, int] = {}
    for element in _find_elements(root, "DimensionDescription"):
        attrs = dict(getattr(element, "attrib", {}) or {})
        try:
            dim_id = int(attrs.get("DimID"))
        except (TypeError, ValueError):
            continue
        label = _DIMENSION_IDS.get(dim_id, "?")
        count = duplicate_counts.get(label, 0)
        duplicate_counts[label] = count + 1
        if count:
            label = f"{label}{count}"
        records[label] = {str(key): str(value) for key, value in attrs.items()}
    return records


def _channel_elements(image: Any) -> tuple[Any, ...]:
    root = getattr(image, "xml_element", None)
    elements = _find_elements(root, "ChannelDescription")

    def key(element):
        try:
            return int(element.attrib.get("BytesInc", 0))
        except (TypeError, ValueError):
            return 0

    return tuple(sorted(elements, key=key))


def _channel_dtype(element: Any) -> tuple[str, Optional[int]]:
    attrs = getattr(element, "attrib", {}) or {}
    resolution = _positive_int(attrs.get("Resolution"))
    data_type = _nonnegative_int(attrs.get("DataType"))
    if resolution is None or resolution > 64 or data_type not in {0, 1}:
        raise ValueError("Leica channel scalar metadata is invalid.")
    itemsize = 1 if resolution <= 8 else 2 if resolution <= 16 else 4 if resolution <= 32 else 8
    if data_type == 1 and itemsize == 1:
        raise ValueError("Leica floating-point channel resolution is invalid.")
    return np.dtype(f"<{('u' if data_type == 0 else 'f')}{itemsize}").name, resolution


def _channel_properties(element: Any) -> dict[str, Any]:
    result = {}
    for child in list(element):
        if str(child.tag).rsplit("}", 1)[-1] != "ChannelProperty":
            continue
        key = child.findtext("Key") or child.attrib.get("Key")
        value = child.findtext("Value") or child.attrib.get("Value")
        if str(key or "").strip():
            result[str(key).strip()] = value
    return result


def _hardware_dye_names(image: Any, channel_count: int) -> tuple[str, ...]:
    root = getattr(image, "xml_element", None)
    names = []
    for element in _find_elements(root, "MultiBand"):
        value = str(element.attrib.get("DyeName") or "").strip()
        if value:
            names.append(value.removeprefix("Leica/"))
    return tuple(names) if len(names) == channel_count else ()


def _channel_metadata(
    image: Any,
    channel_count: int,
    fallback_dtype: str,
) -> tuple[
    tuple[str, ...],
    tuple[MicroscopyChannelMetadata, ...],
    tuple[str, ...],
    tuple[Optional[int], ...],
    tuple[str, ...],
]:
    elements = _channel_elements(image)
    hardware_names = _hardware_dye_names(image, channel_count)
    names = []
    metadata = []
    dtypes = []
    bit_depths = []
    warnings = []
    for index in range(channel_count):
        element = elements[index] if index < len(elements) else None
        attrs = dict(getattr(element, "attrib", {}) or {})
        properties = _channel_properties(element) if element is not None else {}
        try:
            dtype, bit_depth = (
                _channel_dtype(element)
                if element is not None
                else (np.dtype(fallback_dtype).name, np.dtype(fallback_dtype).itemsize * 8)
            )
        except (TypeError, ValueError):
            dtype = np.dtype(fallback_dtype).name
            bit_depth = np.dtype(dtype).itemsize * 8
            warnings.append(
                f"Leica channel {index + 1} scalar metadata was invalid; "
                "the reader array dtype is used operationally."
            )
        dye_name = str(properties.get("DyeName") or "").strip().removeprefix("Leica/")
        explicit_name = dye_name or (hardware_names[index] if hardware_names else "")
        channel_tag = _nonnegative_int(attrs.get("ChannelTag"))
        if not explicit_name:
            explicit_name = str(attrs.get("LUTName") or "").strip()
        if not explicit_name:
            explicit_name = _CHANNEL_TAGS.get(
                channel_tag if channel_tag is not None else -1, ""
            )
        if not explicit_name:
            measured = str(attrs.get("NameOfMeasuredQuantity") or "").strip()
            if measured.casefold() not in {"", "intensity"}:
                explicit_name = measured
        fallback_used = not explicit_name
        name = explicit_name or f"ch{index + 1}"
        if name in names:
            name = f"{name} [{index + 1}]"
        names.append(name)
        channel_warnings = ()
        if fallback_used:
            channel_warnings = (
                "Leica metadata provides no explicit channel name; "
                f"{name!r} is an operational fallback.",
            )
            warnings.extend(channel_warnings)

        def property_number(*keys):
            by_key = {_normalized_key(key): value for key, value in properties.items()}
            for key in keys:
                value = _positive_number(by_key.get(_normalized_key(key)))
                if value is not None:
                    return value
            return None

        excitation = property_number("ExcitationWavelength", "ExcitationWaveLength")
        emission = property_number("EmissionWavelength", "EmissionWaveLength")
        source_identifier = str(
            attrs.get("ChannelTag", properties.get("ChannelID", index))
        )
        raw_channel = {
            "channel_description": {str(key): str(value) for key, value in attrs.items()},
            "channel_properties": copy.deepcopy(properties),
        }
        metadata.append(
            MicroscopyChannelMetadata(
                source_channel_identifier=source_identifier,
                source_channel_name=name,
                excitation_wavelength=excitation,
                excitation_wavelength_units="nm" if excitation is not None else None,
                emission_wavelength=emission,
                emission_wavelength_units="nm" if emission is not None else None,
                detector_settings=raw_channel,
                source_color=(
                    str(attrs.get("LUTName") or "").strip()
                    or _CHANNEL_TAGS.get(
                        channel_tag if channel_tag is not None else -1
                    )
                ),
                reported_scientific_role=None,
                normalized_metadata={
                    "leica_channel_index": index,
                    "scalar_dtype": dtype,
                    "significant_bit_depth": bit_depth,
                    "name_source": (
                        "ChannelProperty.DyeName"
                        if dye_name
                        else "HardwareSetting.MultiBand.DyeName"
                        if hardware_names
                        else "ChannelDescription"
                        if explicit_name
                        else "operational-fallback"
                    ),
                },
                warnings=channel_warnings,
            )
        )
        dtypes.append(dtype)
        bit_depths.append(bit_depth)
    if len(elements) not in {0, channel_count}:
        warnings.append(
            "Leica channel-description count does not match the biological channel axis."
        )
    return (
        tuple(names),
        tuple(metadata),
        tuple(dtypes),
        tuple(bit_depths),
        _unique_text(warnings),
    )


def _series_identity(
    index: int,
    image: Any,
    *,
    container_uuid: Optional[str],
) -> str:
    unique_id = str(getattr(image, "uuid", "") or "").strip()
    if unique_id:
        return f"leica-unique-id:{unique_id}"
    path = str(getattr(image, "path", "") or getattr(image, "name", "") or "").strip()
    payload = json.dumps(
        {"container_uuid": container_uuid, "image_index": index, "image_path": path},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"leica-series-{index}:" + hashlib.sha256(payload).hexdigest()[:16]


def _axis_spacing(
    image: Any,
    axes: tuple[str, ...],
    shape: tuple[int, ...],
    dimension_records: Mapping[str, Mapping[str, Any]],
):
    raw_evidence = {}
    normalized = []
    missing = []
    diagnostics = []
    coords = dict(getattr(image, "coords", {}) or {})
    for axis in "XYZ":
        record = dict(dimension_records.get(axis) or {})
        size = int(shape[axes.index(axis)]) if axis in axes else 1
        unit = str(record.get("Unit") or "").strip() or None
        factor = _UNIT_FACTORS_TO_MICRON.get(str(unit or "").casefold())
        step = None
        values = np.asanyarray(coords.get(axis, ()))
        if size > 1 and values.size == size:
            try:
                differences = np.diff(values.astype(np.float64, copy=False))
                if differences.size and np.all(np.isfinite(differences)):
                    candidate = abs(float(differences[0]))
                    if candidate > 0.0 and np.allclose(
                        np.abs(differences), candidate, rtol=1e-6, atol=0.0
                    ):
                        step = candidate
                    else:
                        diagnostics.append(
                            f"Leica {axis} coordinates are non-uniform or non-positive."
                        )
            except (TypeError, ValueError):
                pass
        raw_evidence[axis] = {
            "dimension_description": record,
            "coordinate_step": step,
            "coordinate_unit": unit,
        }
        if size <= 1:
            missing.append(f"spacing_{axis.lower()}")
            normalized.append(None)
        elif step is None:
            missing.append(f"spacing_{axis.lower()}")
            normalized.append(None)
        elif factor is None:
            missing.append(f"spacing_{axis.lower()}")
            normalized.append(None)
            diagnostics.append(
                f"Leica {axis} spacing unit {unit!r} is missing or unsupported."
            )
        else:
            normalized.append(step * factor)
    spacing = (
        tuple(float(value) for value in normalized)
        if all(value is not None for value in normalized)
        else None
    )
    units = ("micron", "micron", "micron") if spacing is not None else None
    if units is None:
        missing.append("space_units")
    return spacing, units, tuple(dict.fromkeys(missing)), _unique_text(diagnostics), raw_evidence


def _time_evidence(
    image: Any,
    axes: tuple[str, ...],
    shape: tuple[int, ...],
    dimension_records: Mapping[str, Mapping[str, Any]],
):
    count = int(shape[axes.index("T")]) if "T" in axes else 1
    if count <= 1:
        return 1, 1.0, "frame", ()
    record = dict(dimension_records.get("T") or {})
    unit = str(record.get("Unit") or "").strip()
    coords = np.asanyarray((getattr(image, "coords", {}) or {}).get("T", ()))
    interval = None
    if coords.size == count:
        try:
            differences = np.diff(coords.astype(np.float64, copy=False))
            candidate = abs(float(differences[0])) if differences.size else 0.0
            if candidate > 0.0 and np.all(np.isfinite(differences)) and np.allclose(
                np.abs(differences), candidate, rtol=1e-6, atol=0.0
            ):
                interval = candidate
        except (TypeError, ValueError):
            pass
    if interval is None or not unit:
        return (
            count,
            1.0,
            "frame",
            ("Leica time interval or unit is missing; timepoints remain distinct frames.",),
        )
    return count, float(interval), unit, ()


def _stage_units(element: Any) -> tuple[Optional[str], Optional[str], Optional[str]]:
    attrs = dict(getattr(element, "attrib", {}) or {})
    common = str(attrs.get("Unit") or attrs.get("PositionUnit") or "").strip() or None
    result = []
    for axis in "XYZ":
        value = next(
            (
                str(attrs[key]).strip()
                for key in (f"Pos{axis}Unit", f"Position{axis}Unit", f"Unit{axis}")
                if str(attrs.get(key) or "").strip()
            ),
            common,
        )
        result.append(value)
    return tuple(result)


def _stage_observation(
    values,
    units,
    *,
    identity: str,
    series_index: int,
    source_path: str,
    reader_version: Optional[str],
    source_fields: dict[str, Any],
    tile_identity: Optional[str] = None,
    tile_index: Optional[int] = None,
) -> StagePositionObservation:
    raw_values = tuple(_number(value) for value in values)
    raw_units = tuple(str(unit).strip() if unit not in (None, "") else None for unit in units)
    normalized = []
    normalized_units = []
    warnings = []
    for axis, value, unit in zip("XYZ", raw_values, raw_units):
        factor = _UNIT_FACTORS_TO_MICRON.get(str(unit or "").casefold())
        if value is not None and factor is None:
            warnings.append(
                f"Leica stage {axis} position has no explicit supported physical unit."
            )
        normalized.append(value * factor if value is not None and factor is not None else None)
        normalized_units.append("micron" if normalized[-1] is not None else None)
    present = [index for index, value in enumerate(raw_values) if value is not None]
    if present and all(normalized[index] is not None for index in present):
        status = "interpreted" if len(present) == 3 else "partial"
    elif present:
        status = "unsupported-units"
    else:
        status = "uninterpreted"
    if len(present) < 3:
        warnings.append("Leica stage position reports only a subset of XYZ axes.")
    source_fields = copy.deepcopy(source_fields)
    source_fields.update(
        {
            "family": "leica-lif",
            "candidate_status": (
                "usable"
                if status == "interpreted"
                else "partial"
                if status == "partial"
                else "ambiguous"
            ),
            "rejection_reason": (
                ""
                if status in {"interpreted", "partial"}
                else "Stage units are not explicit or supported."
            ),
        }
    )
    return StagePositionObservation(
        raw_position_xyz=raw_values,
        raw_units_xyz=raw_units,
        normalized_position_xyz=tuple(normalized),
        normalized_units_xyz=tuple(normalized_units),
        semantic_meaning=(
            "reported-tile-position" if tile_identity else "reported-stage-position"
        ),
        coordinate_frame="leica-instrument-stage-unverified",
        source_fields=source_fields,
        series_identity=identity,
        series_index=series_index,
        scene_identity=identity,
        tile_identity=tile_identity,
        tile_index=tile_index,
        interpretation_status=status,
        warnings=_unique_text(warnings),
        reader_backend="liffile",
        reader_version=reader_version,
        source_path=source_path,
        source_member_id="primary",
    )


def _stage_positions(
    image: Any,
    *,
    identity: str,
    series_index: int,
    source_path: str,
    reader_version: Optional[str],
) -> tuple[StagePositionObservation, ...]:
    result = []
    root = getattr(image, "xml_element", None)
    scanner_values: dict[str, float] = {}
    scanner_units: dict[str, Optional[str]] = {}
    scanner_fields = []
    for element in root.iter() if root is not None else ():
        attrs = dict(getattr(element, "attrib", {}) or {})
        identifier = _normalized_key(attrs.get("Identifier") or attrs.get("Name"))
        axis = {
            "xpos": "X",
            "ypos": "Y",
            "zpos": "Z",
            "stageposx": "X",
            "stageposy": "Y",
            "stageposz": "Z",
            "positionx": "X",
            "positiony": "Y",
            "positionz": "Z",
        }.get(identifier)
        if axis is None or axis in scanner_values:
            continue
        value = _number(attrs.get("Variant") or attrs.get("Value"))
        if value is None:
            continue
        scanner_values[axis] = value
        scanner_units[axis] = str(attrs.get("Unit") or "").strip() or None
        scanner_fields.append(
            {"axis": axis, "tag": str(element.tag), "attributes": attrs}
        )
    if scanner_values:
        result.append(
            _stage_observation(
                tuple(scanner_values.get(axis) for axis in "XYZ"),
                tuple(scanner_units.get(axis) for axis in "XYZ"),
                identity=identity,
                series_index=series_index,
                source_path=source_path,
                reader_version=reader_version,
                source_fields={
                    "source_label": "Leica scanner stage-position records",
                    "fields": [
                        f"ScannerSettingRecord.{axis}Pos"
                        for axis in "XYZ"
                        if axis in scanner_values
                    ],
                    "leica_scanner_records": scanner_fields,
                },
            )
        )

    tilescan = getattr(image, "tilescan", None)
    if tilescan is not None:
        attachments = [
            item
            for item in _find_elements(root, "Attachment")
            if str(item.attrib.get("Name") or "") == "TileScanInfo"
        ]
        attachment = attachments[0] if attachments else None
        units = _stage_units(attachment) if attachment is not None else (None, None, None)
        data = getattr(tilescan, "tiles", ())
        tile_elements = _find_elements(attachment, "Tile")
        for index, row in enumerate(data):
            try:
                values = (row["pos_x"], row["pos_y"], row["pos_z"])
                field_x, field_y = int(row["field_x"]), int(row["field_y"])
            except (IndexError, KeyError, TypeError, ValueError):
                continue
            tile_identity = f"{identity}:tile:{field_y}:{field_x}:{index}"
            result.append(
                _stage_observation(
                    values,
                    units,
                    identity=identity,
                    series_index=series_index,
                    source_path=source_path,
                    reader_version=reader_version,
                    source_fields={
                        "source_label": "Leica TileScanInfo",
                        "fields": ["PosX", "PosY", "PosZ"],
                        "field_x": field_x,
                        "field_y": field_y,
                        "flip_x": bool(getattr(tilescan, "flip_x", False)),
                        "flip_y": bool(getattr(tilescan, "flip_y", False)),
                        "swap_xy": bool(getattr(tilescan, "swap_xy", False)),
                        "tile_attributes": (
                            dict(tile_elements[index].attrib)
                            if index < len(tile_elements)
                            else {}
                        ),
                    },
                    tile_identity=tile_identity,
                    tile_index=index,
                )
            )
    deduplicated = []
    seen = set()
    for item in result:
        key = json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)
    return tuple(deduplicated)


def _objective_fields(root: Any):
    magnification = _positive_number(
        _first_attribute(root, {"Magnification", "ObjectiveMagnification"})
    )
    numerical_aperture = _positive_number(
        _first_attribute(root, {"NumericalAperture", "ObjectiveNA"})
    )
    immersion = _first_attribute(root, {"Immersion", "ImmersionType"})
    model = _first_attribute(root, {"ObjectiveName", "ObjectiveModel", "Objective"})
    return magnification, numerical_aperture, immersion, model


def inspect_leica_lif_source(
    path: os.PathLike[str] | str,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    reader_module=None,
) -> LeicaSourceInspection:
    """Inspect all images and metadata without reading a voxel frame."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".lif":
        raise LeicaInspectionError(
            LEICA_UNSUPPORTED_VARIANT,
            f"unsupported Leica suffix {source.suffix or '<none>'}; only .lif is supported",
        )
    _check_cancel(cancel_check)
    if not source.is_file():
        raise LeicaInspectionError(
            LEICA_UNREADABLE_SOURCE, f"primary source file is missing: {source.name}"
        )
    reader_module = reader_module or require_liffile()
    try:
        reader = reader_module.LifFile(source, squeeze=False)
    except Exception as exc:
        raise LeicaInspectionError(
            LEICA_UNREADABLE_SOURCE, f"cannot open {source.name}: {exc}"
        ) from exc

    try:
        _check_cancel(cancel_check)
        type_name = str(getattr(getattr(reader, "type", None), "name", "") or "").upper()
        if type_name and type_name != "LIF":
            raise LeicaInspectionError(
                LEICA_UNSUPPORTED_VARIANT,
                f"reader identified {type_name}, not a Leica LIF container",
            )
        reader_version = str(getattr(reader_module, "__version__", "") or "").strip() or None
        container_uuid = str(getattr(reader, "uuid", "") or "").strip() or None
        root = getattr(reader, "xml_element", None)
        source_exclusions: list[dict[str, str]] = []
        file_datetime = _safe_metadata(
            getattr(reader, "datetime", None), "leica_container.datetime", source_exclusions
        )
        container_metadata = {
            "name": str(getattr(reader, "name", "") or "") or None,
            "uuid": container_uuid,
            "version": getattr(reader, "version", None),
            "type": type_name or "LIF",
            "datetime": file_datetime,
            "root_tag": str(getattr(root, "tag", "") or ""),
            "root_attributes": dict(getattr(root, "attrib", {}) or {}),
        }
        series_records = []
        series_evidence = []
        axis_evidence = []
        for index, image in enumerate(tuple(reader.images)):
            _check_cancel(cancel_check)
            try:
                sizes = dict(image.sizes)
                axes = tuple(str(axis) for axis in sizes)
                shape = tuple(int(size) for size in sizes.values())
                if len(axes) != len(shape) or not axes or any(size <= 0 for size in shape):
                    raise ValueError("source axes and positive dimensions are inconsistent")
                if axes.count("X") != 1 or axes.count("Y") != 1:
                    raise ValueError("source image must declare exactly one X and one Y axis")
                fallback_dtype = np.dtype(image.dtype).name
            except Exception as exc:
                raise LeicaInspectionError(
                    LEICA_MALFORMED_METADATA,
                    f"image {index} axis, shape, or scalar metadata is invalid: {exc}",
                ) from exc
            identity = _series_identity(index, image, container_uuid=container_uuid)
            image_path = str(getattr(image, "path", "") or getattr(image, "name", "") or "")
            name = str(getattr(image, "name", "") or image_path or f"Leica image {index + 1}")
            unique_id = str(getattr(image, "uuid", "") or "").strip() or None
            dimensions = _dimension_records(image)
            channel_count = int(shape[axes.index("C")]) if "C" in axes else 1
            (
                channel_names,
                channel_metadata,
                channel_dtypes,
                channel_bits,
                channel_warnings,
            ) = _channel_metadata(image, channel_count, fallback_dtype)
            scalar_dtype = np.result_type(*(np.dtype(item) for item in channel_dtypes)).name
            scalar_bit_depth = (
                channel_bits[0]
                if channel_bits and len(set(channel_bits)) == 1
                else max((value or 0 for value in channel_bits), default=0) or None
            )
            spacing, units, missing, geometry_diagnostics, spacing_raw = _axis_spacing(
                image, axes, shape, dimensions
            )
            time_count, time_interval, time_units, time_warnings = _time_evidence(
                image, axes, shape, dimensions
            )
            stage_positions = _stage_positions(
                image,
                identity=identity,
                series_index=index,
                source_path=str(source),
                reader_version=reader_version,
            )
            timestamps = getattr(image, "timestamps", None)
            acquisition_timestamp = None
            if timestamps is not None:
                flattened = np.asanyarray(timestamps).reshape(-1)
                if flattened.size and not np.isnat(flattened[0]):
                    acquisition_timestamp = str(flattened[0])
            objective = _objective_fields(getattr(image, "xml_element", None))
            image_exclusions: list[dict[str, str]] = []
            attrs = _safe_metadata(
                dict(getattr(image, "attrs", {}) or {}),
                f"leica_series[{index}].attrs",
                image_exclusions,
            )
            raw_image_metadata = {
                "leica_image_attrs": attrs,
                "leica_image_xml": _xml_mapping(
                    getattr(image, "xml_element", ET.Element("Element"))
                ),
            }
            if image_exclusions:
                raw_image_metadata["excluded_metadata"] = image_exclusions
            warnings = _unique_text(
                (
                    *channel_warnings,
                    *time_warnings,
                    *(warning for position in stage_positions for warning in position.warnings),
                    *(f"Excluded {item['path']}: {item['reason']}." for item in image_exclusions),
                )
            )
            acquisition = MicroscopyAcquisitionMetadata(
                acquisition_timestamp=acquisition_timestamp,
                microscope_vendor="Leica Microsystems",
                microscope_model=_first_attribute(
                    getattr(image, "xml_element", None),
                    {"MicroscopeModel", "SystemName", "Microscope"},
                ),
                acquisition_software=_first_attribute(
                    getattr(image, "xml_element", None),
                    {"ApplicationName", "AcquisitionSoftware", "SoftwareName"},
                ),
                reported_acquisition_software_version=_first_attribute(
                    getattr(image, "xml_element", None),
                    {"ApplicationVersion", "SystemVersion", "SoftwareVersion"},
                ),
                objective_magnification=objective[0],
                objective_numerical_aperture=objective[1],
                objective_immersion=objective[2],
                objective_model=objective[3],
                series_identity=identity,
                scene_identity=identity,
                reported_scan_direction=_first_attribute(
                    getattr(image, "xml_element", None), {"ScanDirection"}
                ),
                stage_positions=stage_positions,
                normalized_metadata={
                    "leica_series_index": index,
                    "leica_image_path": image_path,
                    "leica_unique_id": unique_id,
                    "source_axes": list(axes),
                    "source_shape": list(shape),
                    "time_point_count": time_count,
                    "time_interval": time_interval,
                    "time_units": time_units,
                    "scalar_dtype": scalar_dtype,
                    "significant_bit_depth": scalar_bit_depth,
                },
                warnings=warnings,
                additional_fields={"raw_metadata": raw_image_metadata},
            )
            raw_fields = {
                "leica_image_path": image_path,
                "leica_unique_id": unique_id,
                "is_flim": bool(getattr(image, "is_flim", False)),
                "leica_dimension_descriptions": copy.deepcopy(dimensions),
                "leica_spacing_evidence": spacing_raw,
                "time_point_count": time_count,
                "time_interval": time_interval,
                "time_units": time_units,
                "channel_scalar_dtypes": list(channel_dtypes),
                "channel_scalar_bit_depths": list(channel_bits),
                "tilescan_present": getattr(image, "tilescan", None) is not None,
            }
            series_records.append(
                LeicaSeriesInspection(
                    index=index,
                    identity=identity,
                    name=name,
                    source_path_name=image_path,
                    unique_id=unique_id,
                    axes=axes,
                    shape=shape,
                    scalar_dtype=scalar_dtype,
                    scalar_bit_depth=scalar_bit_depth,
                    channel_names=channel_names,
                    channel_metadata=channel_metadata,
                    channel_scalar_dtypes=channel_dtypes,
                    channel_scalar_bit_depths=channel_bits,
                    spacing=spacing,
                    space_units=units,
                    time_count=time_count,
                    time_interval=time_interval,
                    time_units=time_units,
                    acquisition_metadata=acquisition,
                    missing_fields=tuple(dict.fromkeys((*missing, "origin", "direction"))),
                    physical_geometry_diagnostics=geometry_diagnostics,
                    warnings=warnings,
                    raw_fields=raw_fields,
                )
            )
            series_evidence.append(
                {
                    "series_index": index,
                    "series_identity": identity,
                    "name": name,
                    "path": image_path,
                    "UniqueID": unique_id,
                }
            )
            axis_evidence.append(
                {"series_index": index, "axes": list(axes), "shape": list(shape)}
            )
        source_warnings = _unique_text(
            f"Excluded {item['path']}: {item['reason']}." for item in source_exclusions
        )
        raw_metadata = {"leica_container": container_metadata}
        if source_exclusions:
            raw_metadata["excluded_metadata"] = source_exclusions
        source_metadata = MicroscopySourceMetadata(
            reader_backend="liffile",
            reader_version=reader_version,
            reported_vendor_format="leica-lif",
            reported_primary_source_path=str(source),
            source_members=(
                SourceMemberRecord(
                    member_id="primary",
                    path=source.name,
                    role="compound-container",
                    size_bytes=source.stat().st_size,
                    checksum_state="not-computed",
                    structured_metadata={"container_type": "Leica LIF"},
                ),
            ),
            checksum_state="not-computed",
            raw_metadata=raw_metadata,
            source_series_evidence=series_evidence,
            source_axis_evidence=axis_evidence,
            warnings=source_warnings,
            additional_fields={
                "source_bundle": {
                    "kind": "single-file",
                    "container_uuid": container_uuid,
                    "member_count": 1,
                    "member_checksums": "not-computed",
                }
            },
        )
        return LeicaSourceInspection(
            source_path=str(source),
            container_format="leica-lif",
            source_metadata=source_metadata,
            series=tuple(series_records),
            warnings=source_warnings,
        )
    except InterruptedError:
        raise
    except LeicaInspectionError:
        raise
    except Exception as exc:
        raise LeicaInspectionError(LEICA_MALFORMED_METADATA, str(exc)) from exc
    finally:
        reader.close()


def _decode_channel_contract(image: Any, axes: tuple[str, ...], shape: tuple[int, ...]):
    channel_count = int(shape[axes.index("C")]) if "C" in axes else 1
    fallback_dtype = np.dtype(image.dtype).name
    return _channel_metadata(image, channel_count, fallback_dtype)[2:4]


def decode_leica_lif_series_channels(
    path: os.PathLike[str] | str,
    series_index: int,
    channel_indices,
    *,
    expected_series_identity: str,
    expected_axes: tuple[str, ...],
    expected_shape: tuple[int, ...],
    cancel_check: Optional[Callable[[], bool]] = None,
    reader_module=None,
):
    """Decode selected channels frame-by-frame with no partial publications."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".lif":
        raise LeicaInspectionError(
            LEICA_UNSUPPORTED_VARIANT, "only .lif sources can use the Leica decoder"
        )
    requested = tuple(channel_indices)
    if not requested:
        return []
    _check_cancel(cancel_check)
    reader_module = reader_module or require_liffile()
    try:
        reader = reader_module.LifFile(source, squeeze=False)
    except Exception as exc:
        raise LeicaInspectionError(
            LEICA_UNREADABLE_SOURCE, f"cannot open {source.name}: {exc}"
        ) from exc
    try:
        _check_cancel(cancel_check)
        images = tuple(reader.images)
        selected_index = int(series_index)
        if not 0 <= selected_index < len(images):
            raise LeicaInspectionError(
                LEICA_UNSUPPORTED_VARIANT,
                f"series index {selected_index} is unavailable during decode",
            )
        image = images[selected_index]
        axes = tuple(str(value) for value in dict(image.sizes))
        shape = tuple(int(value) for value in dict(image.sizes).values())
        identity = _series_identity(
            selected_index,
            image,
            container_uuid=str(getattr(reader, "uuid", "") or "").strip() or None,
        )
        if identity != expected_series_identity:
            raise LeicaInspectionError(
                LEICA_MALFORMED_METADATA,
                "persisted series identity does not match the selected source image",
            )
        if axes != tuple(expected_axes) or shape != tuple(expected_shape):
            raise LeicaInspectionError(
                LEICA_MALFORMED_METADATA,
                "persisted source axes or dimensions no longer match the selected image",
            )
        channel_count = int(shape[axes.index("C")]) if "C" in axes else 1
        normalized = []
        for value in requested:
            if value is None and channel_count == 1:
                normalized.append(None)
                continue
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("Leica channel selector must be an integer.")
            index = int(value)
            if index != value or not 0 <= index < channel_count:
                raise ValueError(
                    f"Leica channel {value!r} is outside 0..{channel_count - 1}."
                )
            normalized.append(index)
        requested = tuple(normalized)
        channel_dtypes, channel_bits = _decode_channel_contract(image, axes, shape)
        frame_dims = tuple(str(value) for value in image.frames.frame_dims)
        frame_shape = tuple(int(value) for value in image.frames.frame_shape)
        if "X" not in frame_dims or "Y" not in frame_dims:
            raise LeicaInspectionError(
                LEICA_UNSUPPORTED_VARIANT,
                f"frame layout {frame_dims!r} does not expose complete X/Y planes",
            )
        if any(
            axis not in {"X", "Y"} and shape[axes.index(axis)] > 1
            for axis in frame_dims
        ):
            raise LeicaInspectionError(
                LEICA_UNSUPPORTED_VARIANT,
                f"frame layout {frame_dims!r} contains unsupported components",
            )

        results = []
        for channel in requested:
            _check_cancel(cancel_check)
            try:
                source_channel = 0 if channel is None else int(channel)
                dtype = np.dtype(channel_dtypes[source_channel])
                bit_depth = channel_bits[source_channel]
                selected_shape = list(shape)
                if "C" in axes:
                    selected_shape[axes.index("C")] = 1
                output = np.empty(tuple(selected_shape), dtype=dtype)
                iteration_axes = [axis for axis in axes if axis not in frame_dims and axis != "C"]
                for axis in iteration_axes:
                    if axis not in {"T", "Z"} and shape[axes.index(axis)] > 1:
                        raise LeicaInspectionError(
                            LEICA_UNSUPPORTED_VARIANT,
                            f"axis {axis!r} cannot be decoded as a biological volume",
                        )
                ranges = [range(shape[axes.index(axis)]) for axis in iteration_axes]
                for indices in itertools.product(*ranges):
                    _check_cancel(cancel_check)
                    source_indices = dict(zip(iteration_axes, indices, strict=True))
                    if "C" in axes:
                        source_indices["C"] = source_channel
                    frame = np.asanyarray(image.frame(**source_indices))
                    if tuple(frame.shape) != frame_shape:
                        raise ValueError(
                            f"Leica frame shape {frame.shape!r} changed from {frame_shape!r}."
                        )
                    if np.dtype(frame.dtype) != dtype:
                        raise ValueError(
                            f"Leica frame dtype {frame.dtype!r} changed from {dtype.name!r}."
                        )
                    target = []
                    by_axis = dict(zip(iteration_axes, indices, strict=True))
                    for axis in axes:
                        if axis in frame_dims:
                            target.append(slice(None))
                        elif axis == "C":
                            target.append(0)
                        else:
                            target.append(by_axis[axis])
                    output[tuple(target)] = frame
                    _check_cancel(cancel_check)
                results.append(
                    LeicaChannelPixels(
                        channel_index=channel,
                        array=output,
                        source_axes=axes,
                        source_shape=shape,
                        scalar_dtype=dtype.name,
                        scalar_bit_depth=bit_depth,
                    )
                )
            except InterruptedError:
                raise
            except Exception as exc:
                results.append(exc)
        return results
    finally:
        reader.close()


__all__ = [
    "decode_leica_lif_series_channels",
    "inspect_leica_lif_source",
    "LeicaChannelPixels",
    "LeicaInspectionError",
    "LEICA_LIF_READER_VERSION",
    "LeicaSeriesInspection",
    "LeicaSourceInspection",
    "require_liffile",
]
