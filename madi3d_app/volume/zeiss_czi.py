"""Read-only, Qt-independent interpretation of scalar ZEISS CZI volumes.

Only fully covered, unmasked layer-0 planes are accepted.  The audited
pylibCZIrw build exposes bounded subblock-header inspection; ordinary upstream
wheels do not provide enough evidence to distinguish missing pixels from zero.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .microscopy_metadata import (
    MicroscopyAcquisitionMetadata,
    MicroscopyChannelMetadata,
    MicroscopySourceMetadata,
)


CZI_VERSION = "6.1.0"
CZI_UNREADABLE_SOURCE = "unreadable-source"
CZI_MALFORMED_METADATA = "malformed-metadata"
CZI_UNSUPPORTED_LAYOUT = "unsupported-layout"
CZI_UNSUPPORTED_BACKEND = "unsupported-backend"
_PIXEL_TYPES = {
    "Gray8": ("uint8", 8),
    "Gray16": ("uint16", 16),
    "Gray32Float": ("float32", 32),
}
_AXES = ("T", "Z", "C", "Y", "X")
_MAX_LAYER0_SUBBLOCKS = 100_000
_MAX_COVERAGE_INTERVALS = 2_000_000
_MAX_SUBBLOCK_METADATA_BYTES = 65_536


class CziInspectionError(ValueError):
    """Categorized source or scientific-interpretation failure."""

    def __init__(self, category: str, detail: str):
        self.category = str(category)
        self.detail = str(detail)
        super().__init__(f"ZEISS CZI {self.category}: {self.detail}")


@dataclass(frozen=True)
class CziSeriesInspection:
    """Detached scientific contract for one scene, without subblock inventory."""

    index: int
    identity: str
    name: str
    scene_index: int | None
    roi: tuple[int, int, int, int]
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    t_indices: tuple[int, ...]
    z_indices: tuple[int, ...]
    channel_indices: tuple[int, ...]
    fixed_indices: tuple[tuple[str, int], ...]
    scalar_dtype: str
    scalar_bit_depth: int
    channel_names: tuple[str, ...]
    channel_metadata: tuple[MicroscopyChannelMetadata, ...]
    channel_scalar_dtypes: tuple[str, ...]
    channel_scalar_bit_depths: tuple[int, ...]
    channel_coordinates: tuple[tuple[tuple[str, int], ...], ...]
    view_index: int | None
    spacing: tuple[float, float, float] | None
    space_units: tuple[str, str, str] | None
    time_count: int
    time_interval: float
    time_units: str
    acquisition_metadata: MicroscopyAcquisitionMetadata
    missing_fields: tuple[str, ...]
    physical_geometry_diagnostics: tuple[str, ...]
    warnings: tuple[str, ...]
    raw_fields: dict[str, Any]

    def source_selection(self) -> dict[str, Any]:
        """Small, persistent selector validated again before decoding."""
        selection = {
            "scene_index": self.scene_index,
            "roi": list(self.roi),
            "t_start": self.t_indices[0],
            "z_start": self.z_indices[0],
            "channel_indices": list(self.channel_indices),
            "fixed_indices": dict(self.fixed_indices),
        }
        if self.view_index is not None:
            selection["view_index"] = self.view_index
        if self.view_index is not None or any("I" in dict(item) for item in self.channel_coordinates):
            selection["channel_coordinates"] = [dict(item) for item in self.channel_coordinates]
        return selection


@dataclass(frozen=True)
class CziSourceInspection:
    source_path: str
    container_format: str
    source_metadata: MicroscopySourceMetadata
    series: tuple[CziSeriesInspection, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CziChannelPixels:
    channel_index: int | None
    array: np.ndarray  # complete T,Z,Y,X scalar volume
    source_axes: tuple[str, ...]
    source_shape: tuple[int, ...]
    scalar_dtype: str
    scalar_bit_depth: int
    time_count: int
    time_interval: float = 1.0
    time_units: str = "frame"


def require_pylibczirw():
    """Require the exact reader and the audited validity-inspection binding."""
    try:
        from importlib import metadata
        from pylibCZIrw import czi
    except ImportError as exc:
        raise CziInspectionError(
            CZI_UNSUPPORTED_BACKEND, "the audited pylibCZIrw 6.1.0 reader is not installed"
        ) from exc
    try:
        version = metadata.version("pylibCZIrw")
    except metadata.PackageNotFoundError as exc:
        raise CziInspectionError(
            CZI_UNSUPPORTED_BACKEND, "pylibCZIrw distribution metadata is missing"
        ) from exc
    if version != CZI_VERSION or not callable(getattr(czi.CziReader, "inspect_subblock", None)):
        raise CziInspectionError(
            CZI_UNSUPPORTED_BACKEND,
            "MADI3D requires its audited pylibCZIrw 6.1.0 wheel with "
            "bounded subblock validity inspection",
        )
    return czi


def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise InterruptedError("ZEISS CZI operation was cancelled.")


def _integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise CziInspectionError(CZI_MALFORMED_METADATA, f"{label} must be an integer")
    return int(value)


def _rect(value: Any) -> tuple[int, int, int, int]:
    result = tuple(_integer(getattr(value, name), f"subblock {name}") for name in ("x", "y", "w", "h"))
    if result[2] <= 0 or result[3] <= 0:
        raise CziInspectionError(CZI_MALFORMED_METADATA, "layer-0 subblock has nonpositive XY size")
    return result


def _subblock(index: int, info: Any, reader: Any) -> dict[str, Any]:
    """Detach one verified layer-0 header; never read pixels or attachments."""
    raw = reader.inspect_subblock(index, max_metadata_bytes=_MAX_SUBBLOCK_METADATA_BYTES)
    if not isinstance(raw, Mapping):
        raise CziInspectionError(CZI_UNSUPPORTED_BACKEND, "subblock inspection returned no header evidence")
    required = {
        "data_size", "metadata_size", "attachment_size",
        "attachment_format_present", "metadata_xml_valid", "directory_header_match",
        "pixel_type", "logical_rect", "coordinate",
    }
    if not required <= raw.keys():
        raise CziInspectionError(CZI_UNSUPPORTED_BACKEND, "subblock validity evidence is incomplete")
    for key in ("data_size", "metadata_size", "attachment_size"):
        if _integer(raw[key], key) < 0:
            raise CziInspectionError(CZI_MALFORMED_METADATA, f"subblock {key} is negative")
    if raw["data_size"] == 0:
        raise CziInspectionError(CZI_UNSUPPORTED_LAYOUT, "subblock has no source pixel payload")
    if raw["directory_header_match"] is not True:
        raise CziInspectionError(CZI_MALFORMED_METADATA, "subblock header disagrees with its directory")
    if raw["metadata_xml_valid"] is not True:
        raise CziInspectionError(CZI_UNSUPPORTED_LAYOUT, "subblock metadata cannot prove valid-pixel semantics")
    if raw["attachment_format_present"] is not False or raw["attachment_size"] != 0:
        raise CziInspectionError(
            CZI_UNSUPPORTED_LAYOUT,
            "subblock has an attachment or mask declaration; pixel validity is not exposed by the reader",
        )
    rect = _rect(info.logicalRect)
    physical = (
        _integer(info.physicalSize.w, "physical width"),
        _integer(info.physicalSize.h, "physical height"),
    )
    if physical != rect[2:] or not math.isclose(float(info.get_zoom()), 1.0, abs_tol=1e-9):
        raise CziInspectionError(
            CZI_UNSUPPORTED_LAYOUT, "only native-size layer-0 subblocks are supported"
        )
    coordinates = info.coordinate.to_dict()
    if not isinstance(coordinates, Mapping):
        raise CziInspectionError(CZI_MALFORMED_METADATA, "subblock coordinates are unavailable")
    coords = {str(axis): _integer(value, f"{axis} coordinate") for axis, value in coordinates.items()}
    pixel_type = getattr(info.pixelType, "name", None)
    if (
        tuple(raw["logical_rect"]) != rect
        or not isinstance(raw["pixel_type"], str)
        or not isinstance(pixel_type, str)
        or raw["pixel_type"].casefold() != pixel_type.casefold()
        or raw["coordinate"] != coords
    ):
        raise CziInspectionError(
            CZI_MALFORMED_METADATA, "subblock image header disagrees with its directory"
        )
    if pixel_type not in _PIXEL_TYPES:
        raise CziInspectionError(
            CZI_UNSUPPORTED_LAYOUT,
            f"native pixel type {pixel_type or 'unknown'} is not a supported grayscale scalar type",
        )
    return {"rect": rect, "coords": coords, "pixel_type": pixel_type}


def _cover_exactly(roi: tuple[int, int, int, int], rectangles: Sequence[tuple[int, int, int, int]]) -> None:
    """Prove a rectangular tiling without holes or ambiguous overlaps."""
    rx, ry, rw, rh = roi
    if rw <= 0 or rh <= 0 or not rectangles:
        raise CziInspectionError(CZI_UNSUPPORTED_LAYOUT, "source plane has no full-resolution pixels")
    events: dict[int, list[tuple[tuple[int, int], int]]] = defaultdict(list)
    for x, y, width, height in rectangles:
        if x < rx or y < ry or x + width > rx + rw or y + height > ry + rh:
            raise CziInspectionError(CZI_UNSUPPORTED_LAYOUT, "source tile extends outside the scene ROI")
        interval = (y, y + height)
        events[x].append((interval, 1))
        events[x + width].append((interval, -1))
    if rx not in events or rx + rw not in events:
        raise CziInspectionError(CZI_UNSUPPORTED_LAYOUT, "source tiles do not span the scene ROI")
    active: dict[tuple[int, int], int] = {}
    positions = sorted(events)
    examined = 0
    for offset, x in enumerate(positions):
        for interval, change in events[x]:
            count = active.get(interval, 0) + change
            if count < 0:
                raise CziInspectionError(CZI_MALFORMED_METADATA, "source tile edges are inconsistent")
            if count:
                active[interval] = count
            else:
                active.pop(interval, None)
        if offset == len(positions) - 1:
            break
        next_x = positions[offset + 1]
        if x < rx or next_x > rx + rw or next_x <= x:
            raise CziInspectionError(CZI_MALFORMED_METADATA, "source tile X coordinates are inconsistent")
        examined += len(active)
        if examined > _MAX_COVERAGE_INTERVALS:
            raise CziInspectionError(
                CZI_UNSUPPORTED_LAYOUT, "tile arrangement exceeds bounded coverage inspection"
            )
        cursor = ry
        for (top, bottom), count in sorted(active.items()):
            if count != 1 or top != cursor:
                raise CziInspectionError(
                    CZI_UNSUPPORTED_LAYOUT, "source plane has missing or overlapping valid-pixel coverage"
                )
            cursor = bottom
        if cursor != ry + rh:
            raise CziInspectionError(
                CZI_UNSUPPORTED_LAYOUT, "source plane has missing valid-pixel coverage"
            )
    if active:
        raise CziInspectionError(CZI_MALFORMED_METADATA, "source tile edges remain open")


def _metadata_root(reader: Any) -> Mapping[str, Any]:
    metadata = reader.metadata
    if not isinstance(metadata, Mapping):
        return {}
    root = metadata.get("ImageDocument", {})
    if not isinstance(root, Mapping):
        return {}
    value = root.get("Metadata", {})
    return value if isinstance(value, Mapping) else {}


def _path(root: Mapping[str, Any], *keys: str) -> Any:
    value: Any = root
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _spacing(root: Mapping[str, Any]) -> tuple[float, float, float] | None:
    # The CZI Scaling/Distance Value is in metres, per the ZEISS CZI writer API.
    values = _path(root, "Scaling", "Items", "Distance")
    if isinstance(values, Mapping):
        values = [values]
    if not isinstance(values, list):
        return None
    parsed = {}
    for item in values:
        if isinstance(item, Mapping) and item.get("@Id") in {"X", "Y", "Z"}:
            axis = item["@Id"]
            if axis in parsed:
                return None
            unit = item.get("DefaultUnitFormat") or item.get("Unit") or "m"
            if str(unit).strip().lower() not in {"m", "meter", "metre"}:
                return None
            parsed[axis] = _positive(item.get("Value"))
    if any(parsed.get(axis) is None for axis in "XYZ"):
        return None
    return tuple(parsed[axis] * 1_000_000.0 for axis in "XYZ")


def _channel_records(root: Mapping[str, Any], indices: tuple[int, ...]):
    raw = _path(root, "Information", "Image", "Dimensions", "Channels", "Channel")
    raw_items = [raw] if isinstance(raw, Mapping) else raw if isinstance(raw, list) else []
    by_index: dict[int, Mapping[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        identifier = str(item.get("@Id", ""))
        try:
            index = int(identifier.rsplit(":", 1)[-1])
        except ValueError:
            continue
        if index not in by_index:
            by_index[index] = item
    if len(indices) == 1 and indices[0] not in by_index and len(raw_items) == 1:
        item = raw_items[0]
        if isinstance(item, Mapping):
            by_index[indices[0]] = item
    names = []
    metadata = []
    for index in indices:
        item = by_index.get(index, {})
        name = str(item.get("@Name") or f"CZI channel {index}").strip()[:160]
        if not name:
            name = f"CZI channel {index}"
        names.append(name)
        # The scalar XML value alone does not establish a wavelength unit.
        # Retain bounded source evidence without publishing an inferred nm value.
        raw_wavelengths = {
            key: _positive(item.get(key))
            for key in ("ExcitationWavelength", "EmissionWavelength")
            if _positive(item.get(key)) is not None
        }
        metadata.append(MicroscopyChannelMetadata(
            source_channel_identifier=f"C:{index}",
            source_channel_name=name,
            source_color=str(item.get("Color"))[:80] if item.get("Color") else None,
            normalized_metadata={"source_channel_index": index, **raw_wavelengths},
        ))
    return tuple(names), tuple(metadata)


def _series(
    index: int, scene_index: int | None, view_index: int | None,
    blocks: Sequence[Mapping[str, Any]],
    metadata_root: Mapping[str, Any],
) -> CziSeriesInspection:
    identity = f"zeiss-czi:scene-{index}" if scene_index is None else f"zeiss-czi:scene-{scene_index}"
    if view_index is not None:
        identity += f":view-{view_index}"
    min_x = min(block["rect"][0] for block in blocks)
    min_y = min(block["rect"][1] for block in blocks)
    max_x = max(block["rect"][0] + block["rect"][2] for block in blocks)
    max_y = max(block["rect"][1] + block["rect"][3] for block in blocks)
    roi = (min_x, min_y, max_x - min_x, max_y - min_y)
    dimension_keys = {tuple(sorted(block["coords"])) for block in blocks}
    if len(dimension_keys) != 1:
        raise CziInspectionError(
            CZI_UNSUPPORTED_LAYOUT, f"scene {identity} has inconsistent dimension declarations"
        )
    axis_values: dict[str, set[int]] = defaultdict(set)
    for block in blocks:
        for axis, value in block["coords"].items():
            if axis != "S":
                axis_values[axis].add(value)
    illumination_indices = tuple(sorted(axis_values.get("I", ())))
    illumination_channels = len(illumination_indices) > 1
    extra = {
        axis: values for axis, values in axis_values.items()
        if axis not in ({"C", "T", "Z", "I"} if illumination_channels else {"C", "T", "Z"})
    }
    if any(len(values) != 1 for values in extra.values()):
        raise CziInspectionError(
            CZI_UNSUPPORTED_LAYOUT,
            f"scene {identity} has a non-singleton unsupported dimension; select a conventional C/T/Z volume",
        )
    fixed = tuple(sorted((axis, next(iter(values))) for axis, values in extra.items()))
    t_indices = tuple(sorted(axis_values.get("T", {0})))
    z_indices = tuple(sorted(axis_values.get("Z", {0})))
    for axis, values in (("T", t_indices), ("Z", z_indices)):
        if values != tuple(range(values[0], values[0] + len(values))):
            raise CziInspectionError(
                CZI_UNSUPPORTED_LAYOUT,
                f"scene {identity} has missing {axis} source indices",
            )
    channel_axes = ("C", "I") if illumination_channels else ("C",)
    channel_coordinates = tuple(
        sorted({
            tuple((axis, block["coords"].get(axis, 0)) for axis in channel_axes)
            for block in blocks
        })
    )
    planes: dict[tuple[tuple[tuple[str, int], ...], int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    types: dict[tuple[tuple[str, int], ...], set[str]] = defaultdict(set)
    for block in blocks:
        coordinates = block["coords"]
        channel = tuple((axis, coordinates.get(axis, 0)) for axis in channel_axes)
        planes[(channel, coordinates.get("T", 0), coordinates.get("Z", 0))].append(block["rect"])
        types[channel].add(block["pixel_type"])
    dtypes = []
    bits = []
    for channel in channel_coordinates:
        if len(types[channel]) != 1:
            raise CziInspectionError(
                CZI_UNSUPPORTED_LAYOUT, f"scene {identity} channel {dict(channel)} mixes native pixel types"
            )
        dtype, bit_depth = _PIXEL_TYPES[next(iter(types[channel]))]
        dtypes.append(dtype)
        bits.append(bit_depth)
        for time in t_indices:
            for z in z_indices:
                _cover_exactly(roi, planes[(channel, time, z)])
    source_c_indices = tuple(dict(channel)["C"] for channel in channel_coordinates)
    source_c_indices = tuple(sorted(set(source_c_indices)))
    names, channel_metadata = _channel_records(metadata_root, source_c_indices)
    if illumination_channels:
        metadata_by_c = dict(zip(tuple(sorted(set(source_c_indices))), channel_metadata))
        names = tuple(
            f"{metadata_by_c[channel[0][1]].source_channel_name} illumination {dict(channel)['I']}"
            for channel in channel_coordinates
        )
        channel_metadata = tuple(
            MicroscopyChannelMetadata(
                source_channel_identifier=f"C:{dict(channel)['C']};I:{dict(channel)['I']}",
                source_channel_name=name,
                source_color=metadata_by_c[dict(channel)["C"]].source_color,
                excitation_wavelength=metadata_by_c[dict(channel)["C"]].excitation_wavelength,
                excitation_wavelength_units=metadata_by_c[dict(channel)["C"]].excitation_wavelength_units,
                emission_wavelength=metadata_by_c[dict(channel)["C"]].emission_wavelength,
                emission_wavelength_units=metadata_by_c[dict(channel)["C"]].emission_wavelength_units,
                normalized_metadata={
                    **dict(metadata_by_c[dict(channel)["C"]].normalized_metadata),
                    "source_channel_coordinates": dict(channel),
                    "illumination_index": dict(channel)["I"],
                },
            )
            for channel, name in zip(channel_coordinates, names)
        )
    spacing = _spacing(metadata_root)
    acquisition = MicroscopyAcquisitionMetadata(
        microscope_vendor="ZEISS",
        series_identity=identity,
        scene_identity=identity,
        normalized_metadata={
            "source_scene_index": scene_index,
            **({"source_view_index": view_index} if view_index is not None else {}),
            "source_roi": list(roi),
            "source_t_start": t_indices[0],
            "source_z_start": z_indices[0],
            "time_point_count": len(t_indices),
        },
    )
    if view_index is not None:
        series_name = (
            f"ZEISS scene {scene_index + 1}, view {view_index}"
            if scene_index is not None else f"ZEISS acquisition, view {view_index}"
        )
    else:
        series_name = (
            f"ZEISS scene {scene_index + 1}"
            if scene_index is not None else "ZEISS acquisition"
        )
    return CziSeriesInspection(
        index=index, identity=identity,
        name=series_name,
        scene_index=scene_index, roi=roi, axes=_AXES,
        shape=(len(t_indices), len(z_indices), len(channel_coordinates), roi[3], roi[2]),
        t_indices=t_indices, z_indices=z_indices,
        channel_indices=tuple(dict(channel)["C"] for channel in channel_coordinates),
        fixed_indices=fixed, scalar_dtype=dtypes[0], scalar_bit_depth=bits[0],
        channel_names=names, channel_metadata=channel_metadata,
        channel_scalar_dtypes=tuple(dtypes), channel_scalar_bit_depths=tuple(bits),
        channel_coordinates=channel_coordinates,
        view_index=view_index,
        spacing=spacing, space_units=("micron",) * 3 if spacing else None,
        time_count=len(t_indices), time_interval=1.0, time_units="frame",
        acquisition_metadata=acquisition,
        missing_fields=("origin", "direction") if spacing else ("spacing", "origin", "direction"),
        physical_geometry_diagnostics=() if spacing else ("CZI has no complete validated XYZ pixel spacing.",),
        warnings=() if spacing else ("CZI has no complete validated XYZ pixel spacing.",),
        raw_fields={
            "source_scene_index": scene_index, "source_roi": list(roi),
            "source_t_start": t_indices[0], "source_z_start": z_indices[0],
            "source_channel_indices": list(dict.fromkeys(
                dict(channel)["C"] for channel in channel_coordinates
            )),
            "source_fixed_indices": dict(fixed),
            **({"source_view_index": view_index} if view_index is not None else {}),
            **({"source_channel_coordinates": [dict(channel) for channel in channel_coordinates]}
               if illumination_channels else {}),
        },
    )


def _inspect_open_reader(source: Path, reader: Any, cancel_check) -> CziSourceInspection:
    """Inspect validity on the same read-only handle later used for pixel reads."""
    _check_cancel(cancel_check)
    blocks: list[dict[str, Any]] = []

    def collect(index: int, info: Any) -> bool:
        _check_cancel(cancel_check)
        if len(blocks) >= _MAX_LAYER0_SUBBLOCKS:
            raise CziInspectionError(
                CZI_UNSUPPORTED_LAYOUT, "source has too many layer-0 subblocks to inspect safely"
            )
        blocks.append(_subblock(index, info, reader))
        return True

    reader.enumerate_subblocks_subset(collect, only_layer0=True)
    if not blocks:
        raise CziInspectionError(CZI_UNSUPPORTED_LAYOUT, "source has no layer-0 image subblocks")
    has_s = {"S" in block["coords"] for block in blocks}
    if len(has_s) != 1:
        raise CziInspectionError(
            CZI_UNSUPPORTED_LAYOUT, "source mixes scene-tagged and untagged subblocks"
        )
    grouped: dict[tuple[int | None, int | None], list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        scene = block["coords"].get("S") if True in has_s else None
        grouped[(scene, block["coords"].get("V"))].append(block)
    root = _metadata_root(reader)
    views_by_scene: dict[int | None, set[int | None]] = defaultdict(set)
    for scene, view in grouped:
        views_by_scene[scene].add(view)
    varying_scenes = {scene for scene, views in views_by_scene.items() if len(views) > 1}
    series = tuple(
        _series(index, scene, view if scene in varying_scenes else None, grouped[(scene, view)], root)
        for index, (scene, view) in enumerate(sorted(
            grouped, key=lambda value: (-1 if value[0] is None else value[0], -1 if value[1] is None else value[1])
        ))
    )
    source_metadata = MicroscopySourceMetadata(
        reader_backend="pylibCZIrw", reader_version=CZI_VERSION,
        reported_vendor_format="zeiss-czi",
        reported_primary_source_path=str(source),
        checksum_state="not-computed",
        source_series_evidence={"scene_count": len(series)},
        source_axis_evidence={"source_dimensions": list(_AXES)},
    )
    _check_cancel(cancel_check)
    return CziSourceInspection(str(source), "zeiss-czi", source_metadata, series)


def inspect_zeiss_czi_source(
    path: os.PathLike[str] | str, *,
    cancel_check: Callable[[], bool] | None = None,
    reader_module=None,
) -> CziSourceInspection:
    """Inspect header geometry and validity without decoding source pixels."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".czi":
        raise CziInspectionError(CZI_UNSUPPORTED_LAYOUT, "only .czi sources are supported")
    if not source.is_file():
        raise CziInspectionError(CZI_UNREADABLE_SOURCE, f"source file is missing: {source.name}")
    _check_cancel(cancel_check)
    backend = reader_module or require_pylibczirw()
    try:
        options = backend.ReaderOptions(enable_mask_awareness=True, lax_subblock_coordinate_checks=False)
        with backend.open_czi(str(source), reader_options=options) as reader:
            return _inspect_open_reader(source, reader, cancel_check)
    except (InterruptedError, CziInspectionError):
        raise
    except OSError as exc:
        raise CziInspectionError(CZI_UNREADABLE_SOURCE, f"cannot read {source.name}: {exc}") from exc
    except Exception as exc:
        raise CziInspectionError(CZI_MALFORMED_METADATA, f"cannot inspect {source.name}: {exc}") from exc


def decode_zeiss_czi_series_channels(
    path: os.PathLike[str] | str,
    series_index: int,
    channel_indices: Sequence[int | None],
    *,
    expected_series_identity: str,
    expected_axes: tuple[str, ...],
    expected_shape: tuple[int, ...],
    expected_selection: Mapping[str, Any],
    cancel_check: Callable[[], bool] | None = None,
    reader_module=None,
) -> list[CziChannelPixels]:
    """Revalidate the scene, then read native scalar T/Z planes into final arrays."""
    requested = tuple(channel_indices)
    if not requested:
        return []
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".czi" or not source.is_file():
        raise CziInspectionError(CZI_UNREADABLE_SOURCE, f"CZI source is unavailable: {source.name}")
    _check_cancel(cancel_check)
    backend = reader_module or require_pylibczirw()
    options = backend.ReaderOptions(enable_mask_awareness=True, lax_subblock_coordinate_checks=False)
    try:
        with backend.open_czi(str(source), reader_options=options) as reader:
            inspection = _inspect_open_reader(source, reader, cancel_check)
            if isinstance(series_index, bool) or not isinstance(series_index, int):
                raise CziInspectionError(CZI_MALFORMED_METADATA, "persisted scene selector must be an integer")
            selected = next((item for item in inspection.series if item.index == series_index), None)
            if selected is None or selected.identity != expected_series_identity:
                raise CziInspectionError(CZI_MALFORMED_METADATA, "persisted scene identity is unavailable")
            if selected.axes != tuple(expected_axes) or selected.shape != tuple(expected_shape):
                raise CziInspectionError(CZI_MALFORMED_METADATA, "persisted source dimensions changed")
            if selected.source_selection() != dict(expected_selection):
                raise CziInspectionError(
                    CZI_MALFORMED_METADATA, "persisted scene, source indices or geometry changed"
                )
            normalized = []
            for channel in requested:
                if channel is None and len(selected.channel_indices) == 1:
                    normalized.append(0)
                elif isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel < len(selected.channel_indices):
                    raise CziInspectionError(CZI_MALFORMED_METADATA, f"channel selector {channel!r} is unavailable")
                else:
                    normalized.append(channel)
            if len(set(normalized)) != len(normalized):
                raise CziInspectionError(CZI_MALFORMED_METADATA, "requested channels must be unique")
            outputs = [
                np.empty(
                    (len(selected.t_indices), len(selected.z_indices), selected.roi[3], selected.roi[2]),
                    dtype=np.dtype(selected.channel_scalar_dtypes[channel]),
                )
                for channel in normalized
            ]
            _check_cancel(cancel_check)
            for output, channel in zip(outputs, normalized):
                for t, source_t in enumerate(selected.t_indices):
                    for z, source_z in enumerate(selected.z_indices):
                        _check_cancel(cancel_check)
                        plane = {
                            **dict(selected.channel_coordinates[channel]),
                            "T": source_t, "Z": source_z,
                        }
                        plane.update(dict(selected.fixed_indices))
                        image = np.asarray(reader.read(
                            roi=selected.roi, plane=plane, scene=selected.scene_index,
                            zoom=1,
                        ))
                        if image.shape == (selected.roi[3], selected.roi[2], 1):
                            image = image[:, :, 0]
                        if image.shape != (selected.roi[3], selected.roi[2]) or image.dtype != output.dtype:
                            raise CziInspectionError(
                                CZI_MALFORMED_METADATA, "native scalar plane shape or type changed during decode"
                            )
                        output[t, z] = image
                        _check_cancel(cancel_check)
    except (InterruptedError, CziInspectionError):
        raise
    except Exception as exc:
        raise CziInspectionError(CZI_UNREADABLE_SOURCE, f"cannot decode {source.name}: {exc}") from exc
    _check_cancel(cancel_check)
    return [
        CziChannelPixels(
            channel_index=None if channel_indices[index] is None else channel,
            array=output, source_axes=selected.axes, source_shape=selected.shape,
            scalar_dtype=selected.channel_scalar_dtypes[channel],
            scalar_bit_depth=selected.channel_scalar_bit_depths[channel],
            time_count=selected.time_count,
        )
        for index, (channel, output) in enumerate(zip(normalized, outputs))
    ]


__all__ = [
    "CziInspectionError", "CziSeriesInspection", "CziSourceInspection",
    "CziChannelPixels", "require_pylibczirw", "inspect_zeiss_czi_source",
    "decode_zeiss_czi_series_channels",
]
