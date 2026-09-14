"""Canonical, data-only provenance descriptors for scientific volumes."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping

import numpy as np


_H5J_POSITION_ATTRIBUTE_KEYS = frozenset(
    {
        "stagex",
        "stagey",
        "stagez",
        "xposition",
        "yposition",
        "zposition",
        "positionx",
        "positiony",
        "positionz",
        "xpositionum",
        "ypositionum",
        "zpositionum",
    }
)
_H5J_UNIT_ATTRIBUTE_KEYS = frozenset(
    {
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
    }
)
_H5J_FRAME_ATTRIBUTE_KEYS = frozenset(
    {"CoordinateFrame", "coordinate_frame"}
)


def _normalized_h5j_position_key(value: Any) -> str:
    return re.sub(r"[\s_\-]+", "", str(value or "")).lower()


def _h5j_scalar_attribute(value: Any) -> str | int | float | None:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(()).item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        return value.strip() or None
    return None


def h5j_position_attribute_subset(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Return only strict scalar H5J stage position/unit/frame evidence."""

    if not isinstance(attributes, Mapping):
        return {}
    retained: dict[str, Any] = {}
    for raw_key in sorted(attributes, key=lambda item: str(item)):
        key = str(raw_key)
        normalized_key = _normalized_h5j_position_key(key)
        is_position = normalized_key in _H5J_POSITION_ATTRIBUTE_KEYS
        is_unit = key in _H5J_UNIT_ATTRIBUTE_KEYS
        is_frame = key in _H5J_FRAME_ATTRIBUTE_KEYS
        if not (is_position or is_unit or is_frame):
            continue
        value = _h5j_scalar_attribute(attributes[raw_key])
        if value is None:
            continue
        if (is_unit or is_frame) and not isinstance(value, str):
            continue
        retained[key] = value
    return retained


def canonical_h5j_position_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and detach the persisted strict H5J stage-evidence schema."""

    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("H5J position metadata must be an object.")
    unknown_sections = set(value).difference({"root_attrs", "channel_attrs"})
    if unknown_sections:
        raise ValueError(
            "H5J position metadata contains unsupported sections: "
            + ", ".join(sorted(str(key) for key in unknown_sections))
        )

    root_value = value.get("root_attrs") or {}
    if not isinstance(root_value, Mapping):
        raise ValueError("H5J root position attributes must be an object.")
    root_attrs = h5j_position_attribute_subset(root_value)
    if set(root_attrs) != {str(key) for key in root_value}:
        raise ValueError(
            "H5J root position attributes contain unsupported keys or values."
        )

    raw_channels = value.get("channel_attrs") or {}
    if not isinstance(raw_channels, Mapping):
        raise ValueError("H5J channel position attributes must be an object.")
    channel_attrs: dict[str, dict[str, Any]] = {}
    for raw_channel in sorted(raw_channels, key=lambda item: str(item)):
        if not isinstance(raw_channel, str) or not raw_channel.strip():
            raise ValueError(
                "H5J channel position attributes require non-empty channel names."
            )
        attributes = raw_channels[raw_channel]
        if not isinstance(attributes, Mapping):
            raise ValueError(
                "H5J channel position attributes must contain attribute objects."
            )
        retained = h5j_position_attribute_subset(attributes)
        if set(retained) != {str(key) for key in attributes}:
            raise ValueError(
                f"H5J channel {raw_channel!r} position attributes contain "
                "unsupported keys or values."
            )
        if retained:
            channel_attrs[raw_channel] = retained

    result: dict[str, Any] = {}
    if root_attrs:
        result["root_attrs"] = root_attrs
    if channel_attrs:
        result["channel_attrs"] = channel_attrs
    return result


def stable_scientific_operation_record(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a detached operation payload with a stable explicit identity."""

    record = copy.deepcopy(dict(value or {}))
    operation_type = str(
        record.get("operation_type") or record.get("operation") or ""
    ).strip()
    if not operation_type:
        raise ValueError("Scientific operation type is required.")
    record["operation_type"] = operation_type
    record.pop("operation", None)
    operation_id = str(record.get("operation_id") or "").strip()
    if not operation_id:
        identity = copy.deepcopy(record)
        identity.pop("operation_id", None)
        encoded = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        operation_id = "sha256:" + hashlib.sha256(encoded).hexdigest()
    record["operation_id"] = operation_id
    return record


def generated_voxel_provenance_projection(
    snapshot: Mapping[str, Any],
    operation_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make a generated-data operation the producer of one detached output."""

    scientific = copy.deepcopy(
        dict(snapshot.get("scientific_provenance") or {})
    )
    previous_id = str(
        scientific.get("producing_operation_id") or ""
    ).strip()
    previous_producer = scientific.get("producing_operation")
    if not previous_id or not isinstance(previous_producer, Mapping):
        raise ValueError(
            "Generated voxel provenance requires the exact previous producer."
        )

    record = copy.deepcopy(dict(operation_record or {}))
    record["input_operation_ids"] = list(
        dict.fromkeys(
            [
                *list(record.get("input_operation_ids") or ()),
                previous_id,
            ]
        )
    )
    source_id = str(snapshot.get("source_id") or "").strip()
    channel_id = str(snapshot.get("channel_id") or "").strip()
    if source_id:
        record.setdefault("input_acquisition_ids", [source_id])
        record.setdefault("output_acquisition_id", source_id)
    if channel_id:
        record.setdefault("input_channel_ids", [channel_id])
        record.setdefault("output_channel_id", channel_id)
    record = stable_scientific_operation_record(record)

    supporting_by_id: dict[str, dict[str, Any]] = {}
    supporting_order: list[str] = []
    for raw_record in (
        previous_producer,
        *list(scientific.get("supporting_operations") or ()),
    ):
        support = stable_scientific_operation_record(raw_record)
        operation_id = support["operation_id"]
        prior = supporting_by_id.get(operation_id)
        if prior is not None and prior != support:
            raise ValueError(
                f"Scientific operation {operation_id!r} has conflicting payloads."
            )
        supporting_by_id[operation_id] = support
        if operation_id not in supporting_order:
            supporting_order.append(operation_id)
    if record["operation_id"] in supporting_by_id:
        raise ValueError(
            "Generated voxel producer identity conflicts with an input operation."
        )

    operation_ids = list(
        dict.fromkeys(
            [
                *list(scientific.get("operation_ids") or ()),
                record["operation_id"],
            ]
        )
    )
    history_ids = list(
        dict.fromkeys(
            [
                *list(scientific.get("history_operation_ids") or ()),
                record["operation_id"],
            ]
        )
    )
    projection = {
        **scientific,
        "producing_operation_id": record["operation_id"],
        "producing_operation": copy.deepcopy(record),
        "supporting_operations": [
            copy.deepcopy(supporting_by_id[operation_id])
            for operation_id in supporting_order
        ],
        "operation_ids": operation_ids,
        "history_operation_ids": history_ids,
    }
    return record, projection


def scalar_descriptor(dtype) -> tuple[str | None, int | None]:
    """Return normalized scalar dtype and storage bit depth, or explicit unknowns."""
    if dtype in (None, ""):
        return None, None
    try:
        normalized = np.dtype(dtype)
    except Exception:
        return None, None
    return normalized.name, int(normalized.itemsize * 8)


def explicit_text(value) -> str | None:
    """Normalize source-provided text while preserving missing values as unknown."""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    value = str(value).strip()
    return value or None


def first_explicit_text(metadata: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = explicit_text(metadata.get(key))
        if value is not None:
            return value
    return None


def axis_contract(axes: Iterable[Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return source-order labels and semantic names without inventing labels."""
    axes = tuple(axes or ())
    return (
        tuple(str(getattr(axis, "label", "") or "") for axis in axes),
        tuple(str(getattr(axis, "semantic", "unknown") or "unknown") for axis in axes),
    )


def geometry_checksum(
    *,
    dimensions,
    spacing,
    spatial_units,
    origin,
    direction,
    time_point_count=1,
    time_interval=1.0,
    time_units="frame",
) -> str | None:
    """Hash complete authoritative geometry; incomplete geometry remains unknown."""
    if any(value is None for value in (dimensions, spacing, spatial_units, origin, direction)):
        return None
    try:
        dims = tuple(int(value) for value in dimensions)
        spacing_values = tuple(float(value) for value in spacing)
        origin_values = tuple(float(value) for value in origin)
        direction_values = tuple(
            tuple(float(value) for value in row) for row in direction
        )
        units = tuple(str(value) for value in spatial_units)
        time_count = int(time_point_count)
        time_step = float(time_interval)
        if (
            len(dims) != 3
            or any(value <= 0 for value in dims)
            or len(spacing_values) != 3
            or any(not math.isfinite(value) or value <= 0.0 for value in spacing_values)
            or len(origin_values) != 3
            or any(not math.isfinite(value) for value in origin_values)
            or len(direction_values) != 3
            or any(len(row) != 3 for row in direction_values)
            or not np.all(np.isfinite(np.asarray(direction_values, dtype=float)))
            or len(units) != 3
            or any(not value.strip() or value.strip().lower() == "unknown" for value in units)
            or time_count < 1
            or not math.isfinite(time_step)
            or time_step <= 0.0
        ):
            return None
    except Exception:
        return None

    payload = {
        "dimensions": dims,
        "spacing": spacing_values,
        "spatial_units": units,
        "origin": origin_values,
        "direction": direction_values,
        "time_point_count": time_count,
        "time_interval": time_step,
        "time_units": str(time_units or "frame"),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def decoded_payload_descriptor(
    array,
    *,
    decoded_array=None,
    dimensions=(),
    time_point_count=1,
    axis_order=(),
    axis_semantics=(),
):
    """Describe source and decoded scalar contracts without interpreting geometry."""
    source_dtype, source_bit_depth = scalar_descriptor(getattr(array, "dtype", None))
    decoded = array if decoded_array is None else decoded_array
    decoded_dtype, decoded_bit_depth = scalar_descriptor(
        getattr(decoded, "dtype", None)
    )
    return {
        "source_scalar_dtype": source_dtype,
        "source_scalar_bit_depth": source_bit_depth,
        "decoded_scalar_dtype": decoded_dtype,
        "decoded_scalar_bit_depth": decoded_bit_depth,
        "decoded_dimensions": tuple(int(value) for value in dimensions),
        "decoded_time_point_count": int(time_point_count),
        "source_axis_order": tuple(str(value) for value in axis_order),
        "source_axis_semantics": tuple(str(value) for value in axis_semantics),
    }


__all__ = [
    "axis_contract",
    "canonical_h5j_position_metadata",
    "decoded_payload_descriptor",
    "explicit_text",
    "first_explicit_text",
    "generated_voxel_provenance_projection",
    "geometry_checksum",
    "h5j_position_attribute_subset",
    "scalar_descriptor",
    "stable_scientific_operation_record",
]
