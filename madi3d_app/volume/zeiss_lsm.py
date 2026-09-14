"""Shared Zeiss LSM metadata interpretation used by probe and catalog."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Optional


def lsm_scan_information(metadata) -> dict[str, Any]:
    value = dict(metadata or {}).get("ScanInformation")
    return dict(value) if isinstance(value, Mapping) else {}


def lsm_channel_names(
    metadata, channel_count: int
) -> tuple[tuple[str, ...], bool]:
    colors = dict(metadata or {}).get("ChannelColors")
    names = colors.get("ColorNames") if isinstance(colors, Mapping) else None
    result = [str(value or "").strip() for value in (names or ())]
    order_verified = len(result) == channel_count and all(result)
    if len(result) != channel_count:
        result = []
    if not result:
        scan = lsm_scan_information(metadata)
        for track_index, track in enumerate(scan.get("Tracks") or ()):
            if not isinstance(track, Mapping):
                continue
            for channel in track.get("DataChannels") or ():
                if not isinstance(channel, Mapping):
                    continue
                name = str(channel.get("Name") or "").strip()
                if name:
                    result.append(f"{name}-T{track_index + 1}")
                if len(result) == channel_count:
                    break
            if len(result) == channel_count:
                break
        # Track records label channels but do not prove stored C-axis order.
        order_verified = False
    normalized = tuple(
        result[index] if index < len(result) and result[index] else f"ch{index + 1}"
        for index in range(channel_count)
    )
    if len(set(normalized)) != len(normalized):
        normalized = tuple(
            f"{name} [channel {index + 1}]"
            for index, name in enumerate(normalized)
        )
    return normalized, order_verified


def finite_lsm_triplet(metadata, names, *, positive=False, scale=1.0):
    values = []
    for name in names:
        try:
            value = float(dict(metadata or {}).get(name))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or (positive and value <= 0):
            return None
        values.append(value * scale)
    return tuple(values)


def lsm_direction(scan_information):
    values = []
    for name in ("Rotation", "Nutation", "Precession"):
        try:
            value = float(dict(scan_information or {}).get(name))
        except (TypeError, ValueError):
            return None, ()
        if not math.isfinite(value):
            return None, ()
        values.append(value)
    if all(math.isclose(value, 0.0, abs_tol=1e-12) for value in values):
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ), ()
    return None, (
        "Zeiss LSM orientation angles were retained as evidence but cannot be "
        "normalized to a direction matrix without an explicit coordinate convention.",
    )


def lsm_acquisition_timestamp(scan_information) -> Optional[str]:
    try:
        microsoft_days = float(dict(scan_information or {}).get("Sample0time"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(microsoft_days) or microsoft_days <= 0:
        return None
    try:
        value = datetime(1899, 12, 30) + timedelta(days=microsoft_days)
    except (OverflowError, ValueError):
        return None
    return value.isoformat(timespec="seconds")


def lsm_objective_fields(value):
    text = str(value or "").strip()
    magnification = None
    numerical_aperture = None
    immersion = None
    match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[xX](?!\w)", text)
    if match:
        magnification = float(match.group(1))
    match = re.search(r"/(\d+(?:\.\d+)?)", text)
    if match:
        numerical_aperture = float(match.group(1))
    for candidate in ("oil", "water", "glycerol", "air"):
        if re.search(rf"\b{candidate}\b", text, flags=re.IGNORECASE):
            immersion = candidate
            break
    return text or None, magnification, numerical_aperture, immersion


def lsm_track_channel_records(scan_information, channel_names):
    tracks = [
        dict(value)
        for value in (dict(scan_information or {}).get("Tracks") or ())
        if isinstance(value, Mapping)
    ]
    result = []
    for source_name in channel_names:
        track_index = None
        match = re.search(r"-T(\d+)$", str(source_name), flags=re.IGNORECASE)
        if match:
            track_index = int(match.group(1)) - 1
        base_name = re.sub(r"-T\d+$", "", str(source_name), flags=re.IGNORECASE)
        if track_index is not None and 0 <= track_index < len(tracks):
            track = tracks[track_index]
        else:
            track = next(
                (
                    candidate
                    for candidate in tracks
                    if any(
                        str(record.get("Name") or "").strip().casefold()
                        == base_name.casefold()
                        for record in candidate.get("DataChannels") or ()
                        if isinstance(record, Mapping)
                    )
                ),
                {},
            )

        def matching(records, *names):
            values = [
                dict(value) for value in records or () if isinstance(value, Mapping)
            ]
            for record in values:
                candidate = next(
                    (
                        str(record.get(name) or "").strip()
                        for name in names
                        if record.get(name)
                    ),
                    "",
                )
                if candidate.casefold() == base_name.casefold():
                    return record
            acquired = [
                record
                for record in values
                if record.get("Acquire", record.get("Aquire"))
                not in (None, 0, False)
            ]
            return acquired[0] if acquired else (values[0] if values else {})

        data_channel = matching(track.get("DataChannels"), "Name")
        detection = matching(
            track.get("DetectionChannels"), "ChannelName", "PointDetectorName"
        )
        illumination = matching(
            track.get("IlluminationChannels"), "DetchannelName", "Name"
        )
        result.append((data_channel, detection, illumination))
    return tuple(result)


__all__ = [
    "finite_lsm_triplet",
    "lsm_acquisition_timestamp",
    "lsm_channel_names",
    "lsm_direction",
    "lsm_objective_fields",
    "lsm_scan_information",
    "lsm_track_channel_records",
]
