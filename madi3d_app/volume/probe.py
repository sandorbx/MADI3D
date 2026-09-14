# -*- coding: utf-8 -*-
"""Header-only inspection of supported scientific volume containers."""
from __future__ import annotations

import copy
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from .geometry import (
    canonical_space_units,
    direction_matrix3,
    finite_tuple3,
    grid_components_from_affine,
    json_safe_source_value,
)
from .microscopy_metadata import (
    MicroscopyAcquisitionMetadata,
    MicroscopyChannelMetadata,
    MicroscopySourceMetadata,
)
from .leica_lif import (
    inspect_leica_lif_source,
    LeicaInspectionError,
    LeicaSourceInspection,
)
from .olympus import (
    inspect_olympus_source,
    OlympusInspectionError,
    OlympusSourceInspection,
)
from .provenance import (
    axis_contract,
    first_explicit_text,
    geometry_checksum,
    h5j_position_attribute_subset,
    scalar_descriptor,
)
from .source_formats import (
    source_error_container_format,
    tiff_container_format,
    volume_reader_mode,
    volume_source_format,
)
from .zeiss_lsm import (
    finite_lsm_triplet,
    lsm_channel_names,
    lsm_direction,
    lsm_scan_information,
)


@dataclass(frozen=True)
class AxisSemantic:
    index: int
    size: int
    semantic: str
    label: str = ""


@dataclass(frozen=True)
class SeriesCandidate:
    index: int
    identity: str
    axes: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class VolumeSourceProbe:
    container_format: str
    series_identity: str = ""
    series_index: Optional[int] = None
    series_candidates: tuple[SeriesCandidate, ...] = ()
    axis_semantics: tuple[AxisSemantic, ...] = ()
    channel_count: int = 1
    channel_selectors: tuple[Any, ...] = (None,)
    channel_names: tuple[str, ...] = ()
    dimensions: tuple[int, int, int] = ()
    spacing: Optional[tuple[float, float, float]] = None
    space_units: Optional[tuple[str, str, str]] = None
    origin: Optional[tuple[float, float, float]] = None
    direction: Optional[tuple[tuple[float, float, float], ...]] = None
    time_count: int = 1
    time_interval: float = 1.0
    time_units: str = "frame"
    scalar_dtype: Optional[str] = None
    scalar_bit_depth: Optional[int] = None
    channel_scalar_dtypes: tuple[str, ...] = ()
    channel_scalar_bit_depths: tuple[Optional[int], ...] = ()
    scan_direction: Optional[str] = None
    source_checksum: Optional[str] = None
    h5j_position_metadata: dict[str, Any] = field(default_factory=dict)
    acquisition_software_version: Optional[str] = None
    ambiguities: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    raw_fields: dict[str, Any] = field(default_factory=dict)
    physical_geometry_diagnostics: tuple[str, ...] = ()
    resolution_required: bool = False
    resolution_provenance: tuple[dict[str, Any], ...] = ()
    madi3d_geometry_provenance: dict[str, Any] = field(default_factory=dict)
    microscopy_source_metadata: Optional[MicroscopySourceMetadata] = None
    microscopy_acquisition_metadata: Optional[
        MicroscopyAcquisitionMetadata
    ] = None
    microscopy_channel_metadata: tuple[MicroscopyChannelMetadata, ...] = ()

    @property
    def is_multichannel(self) -> bool:
        return self.channel_count > 1

    @property
    def can_create_multichannel(self) -> bool:
        return (
            self.is_multichannel
            and not self.errors
            and not self.requires_axis_resolution
        )

    @property
    def requires_axis_resolution(self) -> bool:
        return self.resolution_required or any(
            axis.size > 1 and axis.semantic in {"component", "unknown"}
            for axis in self.axis_semantics
        )

    @property
    def requires_series_selection(self) -> bool:
        return self.series_index is None and len(self.series_candidates) > 1

    @property
    def has_complete_physical_geometry(self) -> bool:
        return not self.missing_fields and not self.physical_geometry_diagnostics and all(
            value is not None
            for value in (
                self.spacing,
                self.space_units,
                self.origin,
                self.direction,
            )
        )

    @property
    def source_axis_order(self) -> tuple[str, ...]:
        return axis_contract(self.axis_semantics)[0]

    @property
    def source_axis_semantics(self) -> tuple[str, ...]:
        return axis_contract(self.axis_semantics)[1]

    @property
    def geometry_checksum(self) -> Optional[str]:
        return geometry_checksum(
            dimensions=self.dimensions,
            spacing=self.spacing,
            spatial_units=self.space_units,
            origin=self.origin,
            direction=self.direction,
            time_point_count=self.time_count,
            time_interval=self.time_interval,
            time_units=self.time_units,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update({
            "source_axis_order": self.source_axis_order,
            "source_axis_semantics": self.source_axis_semantics,
            "geometry_checksum": self.geometry_checksum,
        })
        return result


def _decoded(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value or "")


def _madi3d_geometry_provenance(value: Any) -> tuple[dict[str, Any], str]:
    """Decode one exporter-authored geometry record without guessing its meaning."""
    if value is None:
        return {}, ""
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8", "replace")
    if isinstance(value, str):
        if not value.strip():
            return {}, ""
        try:
            value = json.loads(value)
        except Exception as exc:
            return {}, f"MADI3D geometry provenance is not valid JSON: {exc}"
    if not isinstance(value, dict):
        return {}, "MADI3D geometry provenance must be a JSON object."
    try:
        return json_safe_source_value(value), ""
    except ValueError as exc:
        return {}, f"MADI3D geometry provenance is invalid: {exc}"


def _nifti_madi3d_geometry_provenance(header) -> tuple[dict[str, Any], str]:
    for extension in tuple(getattr(header, "extensions", ()) or ()):
        try:
            content = extension.get_content()
        except Exception:
            continue
        record, error = _madi3d_geometry_provenance(content)
        if error:
            continue
        if "madi3d_geometry_provenance" in record:
            return _madi3d_geometry_provenance(
                record.get("madi3d_geometry_provenance")
            )
    return {}, ""


def _names_metadata(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = [part for part in value.split(",") if part.strip()]
    try:
        return tuple(_decoded(item).strip().strip('"') for item in value)
    except Exception:
        return ()


def _units3(value: Any, default: str = "unknown") -> tuple[str, str, str]:
    if isinstance(value, str) or value is None:
        values = [value or default] * 3
    else:
        try:
            values = list(value)
        except Exception:
            values = [default] * 3
        if len(values) == 1:
            values *= 3
        while len(values) < 3:
            values.append(default)
    return tuple(_decoded(item).strip().strip('"') or default for item in values[:3])


def _optional_vector3(
    value: Any, *, positive: bool = False
) -> Optional[tuple[float, float, float]]:
    try:
        return finite_tuple3(value, "Geometry vector", positive=positive)
    except ValueError:
        return None


def _optional_matrix3(
    value: Any,
) -> Optional[tuple[tuple[float, float, float], ...]]:
    try:
        result = direction_matrix3(value)
    except (TypeError, ValueError):
        return None
    return tuple(tuple(float(item) for item in row) for row in result)


def _optional_units3(value: Any) -> Optional[tuple[str, str, str]]:
    if value is None:
        return None
    try:
        return canonical_space_units(_units3(value, "unknown"))
    except ValueError:
        return None


def _raw_fields(**values: Any) -> dict[str, Any]:
    return json_safe_source_value(values)


def _raw_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        return not _decoded(value).strip()
    try:
        return np.asarray(value).size == 0
    except Exception:
        return False


def _scalar_type_error(dtype: Optional[str], container: str) -> Optional[str]:
    if dtype is None:
        return None
    try:
        kind = np.dtype(dtype).kind
    except Exception:
        kind = ""
    if kind not in {"b", "i", "u", "f"}:
        return f"{container} scalar type {dtype!r} is unsupported."
    return None


def _spatial_dimensions(shape: tuple[int, ...], semantics: list[str]) -> tuple[int, int, int]:
    result = []
    for name in ("space-x", "space-y", "space-z"):
        result.append(int(shape[semantics.index(name)]) if name in semantics else 1)
    return tuple(result)


def _axis_labels(value: Any, ndim: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        text = value.strip()
        try:
            decoded = json.loads(text)
        except Exception:
            decoded = None
        if isinstance(decoded, (list, tuple)):
            labels = tuple(str(item).strip().upper() for item in decoded)
        else:
            labels = tuple(
                character.upper()
                for character in text
                if character not in {" ", ",", ";", "|"}
            )
    else:
        try:
            labels = tuple(_decoded(item).strip().upper() for item in value)
        except Exception:
            return ()
    return labels if len(labels) == int(ndim) else ()


def _indexed_h5j_channel_order(group) -> tuple[str, ...]:
    indexed = []
    for name in (str(value) for value in group.keys()):
        match = re.fullmatch(r"Channel_(0|[1-9][0-9]*)", name)
        if match is None:
            return ()
        indexed.append((int(match.group(1)), name))
    indexed.sort()
    if [index for index, _name in indexed] != list(range(len(indexed))):
        return ()
    return tuple(name for _index, name in indexed)


def _h5j_channel_order(handle, group):
    keys = tuple(str(name) for name in group.keys())
    declared = ()
    for owner in (group, handle):
        for key in (
            "channel_order",
            "madi3d_channel_order",
            "channel_names",
            "madi3d_channel_names",
        ):
            declared = _names_metadata(owner.attrs.get(key))
            if declared:
                break
        if declared:
            break
    if not declared:
        indexed = _indexed_h5j_channel_order(group)
        if indexed:
            return indexed, True, ""
        return tuple(sorted(keys)), False, ""
    if len(set(declared)) != len(declared):
        return (), True, "H5J channel-order metadata contains duplicate names."
    if set(declared) != set(keys):
        return (), True, (
            "H5J channel-order metadata does not match the /Channels datasets."
        )
    return tuple(declared), True, ""


def janelia_h5j_profile(handle, group=None) -> bool:
    """Return whether the container matches Janelia's indexed H5J profile."""
    group = group if group is not None else handle.get("/Channels")
    if group is None or not hasattr(group, "keys"):
        return False
    indexed_order = _indexed_h5j_channel_order(group)
    if not indexed_order:
        return False

    channel_spec = _decoded(handle.attrs.get("channel_spec")).strip().lower()
    if len(channel_spec) != len(indexed_order) or any(
        code not in {"s", "r"} for code in channel_spec
    ):
        return False
    try:
        image_size = finite_tuple3(
            np.asarray(handle.attrs["image_size"]).reshape(-1),
            "H5J image size",
            positive=True,
            integer=True,
        )
        finite_tuple3(
            np.asarray(handle.attrs["voxel_size"]).reshape(-1),
            "H5J voxel size",
            positive=True,
        )
        group_size = finite_tuple3(
            (
                np.asarray(group.attrs["width"]).reshape(-1)[0],
                np.asarray(group.attrs["height"]).reshape(-1)[0],
                np.asarray(group.attrs["frames"]).reshape(-1)[0],
            ),
            "H5J encoded size",
            positive=True,
            integer=True,
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return False
    unit = _decoded(handle.attrs.get("unit")).strip()
    return image_size == group_size and bool(unit)


def nrrd_writer_local_origin(path: os.PathLike[str] | str, header=None):
    """Return the Nrrd_Writer producer profile's local zero-origin convention."""
    if header is not None and header.get("space origin") is not None:
        return None
    try:
        with open(path, "rb") as stream:
            prefix = stream.read(256 * 1024).replace(b"\r\n", b"\n")
    except OSError:
        return None
    header_text = prefix.split(b"\n\n", 1)[0].decode("ascii", "replace")
    if any(
        line.startswith("# Created by Nrrd_Writer ")
        for line in header_text.splitlines()
    ):
        return (0.0, 0.0, 0.0)
    return None


def _h5j_dataset_axes(handle, group, dataset):
    for owner in (dataset, group, handle):
        for key in ("axes", "axis_order", "dimension_order"):
            labels = _axis_labels(owner.attrs.get(key), dataset.ndim)
            if labels:
                return labels, True
    if dataset.ndim == 3:
        return ("Z", "Y", "X"), False
    return (), False


def _probe_h5j(
    path: str,
    *,
    channel_order: Optional[tuple[str, ...]] = None,
    resolution_source: str = "explicit-import-plan",
    h5_file=None,
) -> VolumeSourceProbe:
    import h5py

    errors = []
    physical_geometry_diagnostics = []
    ambiguities = []
    resolution_required = False
    manager = nullcontext(h5_file) if h5_file is not None else h5py.File(path, "r")
    with manager as handle:
        group = handle.get("/Channels")
        if group is None or not hasattr(group, "keys"):
            return VolumeSourceProbe(
                container_format="h5j",
                errors=("H5J source does not contain a /Channels group.",),
            )
        names, explicit_order, order_error = _h5j_channel_order(handle, group)
        if channel_order is not None:
            requested_order = tuple(str(name) for name in channel_order)
            keys = {str(name) for name in group.keys()}
            if len(set(requested_order)) != len(requested_order) or set(requested_order) != keys:
                order_error = (
                    "Explicit H5J channel order does not match the /Channels datasets."
                )
                names = ()
            else:
                names = requested_order
                explicit_order = True
                order_error = ""
        if order_error:
            errors.append(order_error)
        if not names:
            errors.append("H5J /Channels group contains no channels.")
        elif len(names) > 1 and not explicit_order:
            ambiguities.append(
                "H5J channel order is not explicit; choose and persist the intended channel order before import."
            )
            resolution_required = True

        shapes = [tuple(int(value) for value in group[name].shape) for name in names]
        dimensions = ()
        time_count = 1
        logical_axes = ()
        axis_semantics = []
        per_channel_geometry = []
        for name, shape in zip(names, shapes):
            dataset = group[name]
            labels, explicit_axes = _h5j_dataset_axes(handle, group, dataset)
            if labels:
                semantics = [
                    {
                        "X": "space-x",
                        "Y": "space-y",
                        "Z": "space-z",
                        "T": "time",
                        "C": "channel",
                    }.get(label, "unknown")
                    for label in labels
                ]
                if any(labels.count(axis) != 1 for axis in ("X", "Y", "Z")):
                    errors.append(
                        f"H5J channel {name!r} must declare exactly one X, Y, and Z axis."
                    )
                    continue
                nested_channels = [
                    shape[index]
                    for index, semantic in enumerate(semantics)
                    if semantic == "channel" and shape[index] > 1
                ]
                if nested_channels:
                    errors.append(
                        f"H5J channel {name!r} contains an additional channel axis; /Channels already owns channel identity."
                    )
                unsupported = [
                    (labels[index], shape[index])
                    for index, semantic in enumerate(semantics)
                    if semantic == "unknown" and shape[index] > 1
                ]
                if unsupported:
                    ambiguities.append(
                        f"H5J channel {name!r} contains unsupported nonspatial axes {unsupported}."
                    )
                    resolution_required = True
                channel_time = (
                    int(shape[semantics.index("time")])
                    if "time" in semantics
                    else 1
                )
                if channel_time > 1:
                    ambiguities.append(
                        f"H5J channel {name!r} contains a time axis with {channel_time} points; time remains separate from channels but this H5J payload requires explicit time handling."
                    )
                    resolution_required = True
                channel_dimensions = _spatial_dimensions(shape, semantics)
                per_channel_geometry.append(
                    (channel_dimensions, channel_time, tuple(labels), tuple(semantics))
                )
                if not explicit_axes and dataset.ndim != 3:
                    resolution_required = True
            elif dataset.ndim > 1:
                errors.append(
                    f"H5J channel {name!r} has shape {shape} without explicit axis metadata; spatial axes are not inferred from the final three dimensions."
                )

        if per_channel_geometry:
            first_geometry = per_channel_geometry[0]
            if any(value[:2] != first_geometry[:2] for value in per_channel_geometry[1:]):
                errors.append("H5J channels report incompatible spatial or time dimensions.")
            dimensions, time_count, logical_axes, logical_semantics = first_geometry
            axis_semantics = [
                AxisSemantic(index + 1, int(size), semantic, label)
                for index, (size, semantic, label) in enumerate(
                    zip(shapes[0], logical_semantics, logical_axes)
                )
            ]
        else:
            for key in ("dimensions", "volume_size", "image_size"):
                raw = handle.attrs.get(key)
                try:
                    values = tuple(int(value) for value in np.asarray(raw).reshape(-1)[:3])
                except Exception:
                    values = ()
                if len(values) == 3 and all(value > 0 for value in values):
                    dimensions = values
                    logical_axes = ("Z", "Y", "X")
                    axis_semantics = [
                        AxisSemantic(1, dimensions[2], "space-z", "Z"),
                        AxisSemantic(2, dimensions[1], "space-y", "Y"),
                        AxisSemantic(3, dimensions[0], "space-x", "X"),
                    ]
                    break
            if names and not dimensions:
                missing_dimensions = (
                    "H5J encoded channel dimensions are not present in container metadata."
                )
                errors.append(missing_dimensions)

        reported_dimensions = dimensions
        if dimensions:
            try:
                dimensions = finite_tuple3(
                    dimensions, "H5J dimensions", positive=True, integer=True
                )
            except ValueError as exc:
                errors.append(str(exc))
                dimensions = ()

        is_janelia_profile = janelia_h5j_profile(handle, group)
        raw_spacing = handle.attrs.get("voxel_size")
        reported_origin = handle.attrs.get("origin")
        reported_direction = handle.attrs.get("direction")
        raw_origin = reported_origin
        raw_direction = reported_direction
        used_local_origin = is_janelia_profile and reported_origin is None
        used_local_direction = is_janelia_profile and reported_direction is None
        if used_local_origin:
            raw_origin = (0.0, 0.0, 0.0)
        if used_local_direction:
            raw_direction = np.eye(3, dtype=float)
        spacing = _optional_vector3(raw_spacing, positive=True)
        origin = _optional_vector3(raw_origin)
        direction = _optional_matrix3(raw_direction)
        if raw_spacing is not None and spacing is None:
            physical_geometry_diagnostics.append(
                "H5J spacing must contain three finite positive values."
            )
        if raw_origin is not None and origin is None:
            physical_geometry_diagnostics.append(
                "H5J origin must contain three finite values."
            )
        if raw_direction is not None and direction is None:
            physical_geometry_diagnostics.append(
                "H5J direction must be finite and nonsingular."
            )
        reported_space_units = handle.attrs.get("space_units")
        profile_unit = handle.attrs.get("unit")
        raw_space_units = reported_space_units
        used_profile_unit = is_janelia_profile and reported_space_units is None
        if used_profile_unit:
            raw_space_units = _decoded(profile_unit).strip()
        space_units = _optional_units3(raw_space_units)
        if not _raw_missing(raw_space_units) and space_units is None:
            physical_geometry_diagnostics.append(
                "H5J spatial units are unsupported, ambiguous, or inconsistent."
            )
        declared_dtype = first_explicit_text(
            handle.attrs, "scalar_dtype", "pixel_type", "data_type"
        )
        if declared_dtype is None:
            spatial_dtypes = {
                np.dtype(group[name].dtype).name
                for name, shape in zip(names, shapes)
                if len(shape) >= 3
            }
            declared_dtype = (
                next(iter(spatial_dtypes)) if len(spatial_dtypes) == 1 else None
            )
        scalar_dtype, scalar_bit_depth = scalar_descriptor(declared_dtype)
        scalar_error = _scalar_type_error(scalar_dtype, "H5J")
        if scalar_error:
            errors.append(scalar_error)
        scan_direction = first_explicit_text(
            handle.attrs, "scan_direction", "madi3d_scan_direction"
        )
        source_checksum = first_explicit_text(
            handle.attrs, "source_checksum", "madi3d_source_checksum", "checksum"
        )
        acquisition_software_version = first_explicit_text(
            handle.attrs,
            "acquisition_software_version",
            "acquisition_software",
            "software_version",
        )
        madi3d_geometry, madi3d_geometry_error = _madi3d_geometry_provenance(
            handle.attrs.get("madi3d_geometry_provenance")
        )
        if madi3d_geometry_error:
            errors.append(madi3d_geometry_error)
        raw_fields = _raw_fields(
            dimensions=reported_dimensions,
            spacing=raw_spacing,
            origin=reported_origin,
            direction=reported_direction,
            space_units=reported_space_units,
            unit=profile_unit,
            time_point_count=time_count,
        )
        root_position_attrs = h5j_position_attribute_subset(handle.attrs)
        channel_position_attrs = {
            name: retained
            for name in names
            if (retained := h5j_position_attribute_subset(group[name].attrs))
        }
        h5j_position_metadata = {}
        if root_position_attrs:
            h5j_position_metadata["root_attrs"] = root_position_attrs
        if channel_position_attrs:
            h5j_position_metadata["channel_attrs"] = channel_position_attrs

    missing_fields = []
    if not reported_dimensions:
        missing_fields.append("dimensions")
    if _raw_missing(raw_spacing):
        missing_fields.append("spacing")
    if _raw_missing(reported_origin) and not used_local_origin:
        missing_fields.append("origin")
    if _raw_missing(reported_direction) and not used_local_direction:
        missing_fields.append("direction")
    if _raw_missing(reported_space_units) and not (
        used_profile_unit and not _raw_missing(profile_unit)
    ):
        missing_fields.append("space_units")

    axes = [AxisSemantic(0, len(names), "channel", "Channels")]
    axes.extend(axis_semantics)
    resolution_provenance = []
    if explicit_order and names:
        resolution_provenance.append(
            {
                "decision": "channel-order",
                "source": (
                    str(resolution_source)
                    if channel_order is not None
                    else "container-metadata"
                ),
                "order": list(names),
            }
        )
    local_grid_fields = []
    if used_local_origin:
        local_grid_fields.append("origin")
    if used_local_direction:
        local_grid_fields.append("direction")
    if used_profile_unit:
        local_grid_fields.append("space_units")
    if local_grid_fields:
        resolution_provenance.append(
            {
                "decision": "format-local-grid",
                "source": "janelia-h5j-profile",
                "fields": local_grid_fields,
            }
        )

    return VolumeSourceProbe(
        container_format="h5j",
        series_identity="/Channels",
        axis_semantics=tuple(axes),
        channel_count=max(1, len(names)),
        channel_selectors=names if len(names) > 1 else ((names[0],) if names else (None,)),
        channel_names=names,
        dimensions=dimensions,
        spacing=spacing,
        space_units=space_units,
        origin=origin,
        direction=direction,
        time_count=time_count,
        scalar_dtype=scalar_dtype,
        scalar_bit_depth=scalar_bit_depth,
        scan_direction=scan_direction,
        source_checksum=source_checksum,
        h5j_position_metadata=h5j_position_metadata,
        acquisition_software_version=acquisition_software_version,
        ambiguities=tuple(ambiguities),
        errors=tuple(errors),
        missing_fields=tuple(missing_fields),
        raw_fields=raw_fields,
        physical_geometry_diagnostics=tuple(physical_geometry_diagnostics),
        resolution_required=resolution_required,
        resolution_provenance=tuple(resolution_provenance),
        madi3d_geometry_provenance=madi3d_geometry,
    )


def ome_series_metadata(
    ome_xml: Optional[str], series_index: int = 0
) -> tuple[tuple[str, ...], dict[str, Any], str]:
    if not ome_xml:
        return (), {}, ""
    try:
        root = ET.fromstring(ome_xml)
        images = [
            node
            for node in root.iter()
            if node.tag.rsplit("}", 1)[-1] == "Image"
        ]
        image = images[int(series_index)]
        pixels = next(
            node
            for node in image
            if node.tag.rsplit("}", 1)[-1] == "Pixels"
        )
        names = tuple(
            str(node.attrib.get("Name") or node.attrib.get("ID") or "")
            for node in pixels
            if node.tag.rsplit("}", 1)[-1] == "Channel"
        )
        identity = str(
            image.attrib.get("Name") or image.attrib.get("ID") or ""
        )
        return names, dict(pixels.attrib), identity
    except Exception:
        return (), {}, ""


def _resolution_spacing(
    tf: Any, page: Any = None
) -> tuple[Optional[float], Optional[float]]:
    spacing = [None, None]
    try:
        page = page or tf.pages[0]
        for index, tag_name in enumerate(("XResolution", "YResolution")):
            raw = page.tags[tag_name].value
            value = float(raw[0]) / float(raw[1]) if isinstance(raw, tuple) else float(raw)
            if math.isfinite(value) and value > 0:
                spacing[index] = 1.0 / value
    except Exception:
        pass
    return tuple(spacing)


def _tiff_series_candidates(tf) -> tuple[SeriesCandidate, ...]:
    candidates = []
    for index, series in enumerate(tf.series):
        axes = str(series.axes or "").upper()
        shape = tuple(int(value) for value in series.shape)
        if (
            len(axes) == len(shape)
            and "X" in axes
            and "Y" in axes
            and all(value > 0 for value in shape)
        ):
            candidates.append(
                SeriesCandidate(
                    index=index,
                    identity=(
                        _lsm_series_base_identity(tf, index, series)
                        if bool(getattr(tf, "is_lsm", False))
                        else str(
                            getattr(series, "name", "") or f"series-{index}"
                        )
                    ),
                    axes=axes,
                    shape=shape,
                )
            )
    return tuple(candidates)


def _lsm_series_base_identity(tf, index: int, series) -> str:
    """Return a stable tifffile-series identity without using display names."""
    reduced = False
    try:
        reduced = bool(series.pages[0].is_reduced)
    except Exception:
        pass
    if reduced:
        parent_index = _lsm_parent_series_index(tf, index)
        return (
            f"lsm-series-{parent_index}:thumbnail"
            if parent_index is not None
            else f"lsm-series-{index}:auxiliary"
        )
    return f"lsm-series-{index}"


def _lsm_parent_series_index(tf, series_index: int) -> Optional[int]:
    """Return the nearest preceding non-reduced LSM series, when explicit."""
    for index in range(int(series_index) - 1, -1, -1):
        try:
            if not bool(tf.series[index].pages[0].is_reduced):
                return index
        except Exception:
            continue
    return None


def tiff_series_subselections(series) -> tuple[dict[str, int], ...]:
    """Enumerate exact LSM position/mosaic selections exposed as scene axes."""
    axes = str(series.axes or "").upper()
    shape = tuple(int(value) for value in series.shape)
    scene_axes = [
        (label, shape[index])
        for index, label in enumerate(axes)
        if label in {"P", "M"}
    ]
    if not scene_axes:
        return ({},)
    selections = [{}]
    for label, size in scene_axes:
        selections = [
            {**selection, label: value}
            for selection in selections
            for value in range(size)
        ]
    return tuple(selections)


def _tiff_scene_identity_suffix(source_subselection) -> str:
    normalized = {
        str(key).upper(): int(value)
        for key, value in dict(source_subselection or {}).items()
    }
    if not normalized:
        return ""
    return ":scene:" + ",".join(
        f"{key}={normalized[key]}" for key in sorted(normalized)
    )


def tiff_series_identity(
    tf,
    series_index: int,
    *,
    pyramid_level: int = 0,
    source_subselection=None,
) -> str:
    """Return the exact logical identity shared by probe, catalog, and decode."""
    candidates = {candidate.index: candidate for candidate in _tiff_series_candidates(tf)}
    candidate = candidates[int(series_index)]
    if bool(getattr(tf, "is_lsm", False)):
        identity = candidate.identity
    else:
        _names, _pixels, ome_identity = ome_series_metadata(
            getattr(tf, "ome_metadata", None), int(series_index)
        )
        identity = ome_identity or candidate.identity
    if int(pyramid_level):
        identity = f"{identity}:pyramid-level:{int(pyramid_level)}"
    return identity + _tiff_scene_identity_suffix(source_subselection)


def _selected_tiff_scene_axes(labels, shape, source_subselection):
    scene_axes = [
        (index, label, shape[index])
        for index, label in enumerate(labels)
        if label in {"P", "M"}
    ]
    supplied = {
        str(key).upper(): value
        for key, value in dict(source_subselection or {}).items()
    }
    unknown = sorted(set(supplied) - {label for _index, label, _size in scene_axes})
    if unknown:
        raise ValueError(f"TIFF scene selection contains unavailable axes {unknown!r}.")
    if any(size > 1 and label not in supplied for _index, label, size in scene_axes):
        return tuple(labels), tuple(shape), {}, False
    normalized = {}
    for _index, label, size in scene_axes:
        value = supplied.get(label, 0)
        if isinstance(value, bool):
            raise ValueError("TIFF scene indices must be non-negative integers.")
        index = int(value)
        if index < 0 or index != value or index >= size:
            raise ValueError(
                f"TIFF scene index {value!r} is unavailable for axis {label!r}."
            )
        normalized[label] = index
    kept = [index for index, label in enumerate(labels) if label not in {"P", "M"}]
    return (
        tuple(labels[index] for index in kept),
        tuple(shape[index] for index in kept),
        normalized,
        True,
    )


def _resolved_tiff_labels(labels, shape, decisions):
    labels = list(labels)
    applied = []
    decisions = dict(decisions or {})
    semantic_to_label = {
        "space-z": "Z",
        "time": "T",
        "channel": "C",
        "component": "S",
        "unknown": "",
    }
    for index, label in enumerate(tuple(labels)):
        if label in {"X", "Y", "Z", "T", "C", "S"} or shape[index] <= 1:
            continue
        decision = decisions.get(index, decisions.get(str(index), decisions.get(label)))
        if decision is None:
            continue
        normalized = str(decision).strip().lower()
        if normalized.upper() in {"Z", "T", "C", "S"}:
            replacement = normalized.upper()
            semantic = {
                "Z": "space-z",
                "T": "time",
                "C": "channel",
                "S": "component",
            }[replacement]
        else:
            semantic = normalized
            replacement = semantic_to_label.get(semantic)
        if replacement is None:
            raise ValueError(
                f"Unsupported TIFF axis-resolution decision {decision!r} for axis {index}."
            )
        labels[index] = replacement or label
        applied.append(
            {"axis_index": index, "source_label": label, "semantic": semantic}
        )
    return labels, tuple(applied)


def _probe_tiff(
    path: str,
    *,
    series_index: Optional[int] = None,
    pyramid_level: int = 0,
    source_subselection: Optional[dict[str, int]] = None,
    axis_resolution: Optional[dict[Any, str]] = None,
    channel_order: Optional[tuple[str, ...]] = None,
    resolution_source: str = "explicit-import-plan",
    tiff_file=None,
) -> VolumeSourceProbe:
    import tifffile

    ambiguities = []
    errors = []
    physical_geometry_diagnostics = []
    resolution_provenance = []
    manager = (
        nullcontext(tiff_file)
        if tiff_file is not None
        else tifffile.TiffFile(path)
    )
    with manager as tf:
        is_lsm = bool(getattr(tf, "is_lsm", False))
        path_format = volume_source_format(path)
        if path_format is not None and path_format.key == "lsm" and not is_lsm:
            return VolumeSourceProbe(
                container_format="zeiss-lsm",
                errors=("The .lsm source does not contain Zeiss LSM metadata.",),
            )
        container_format = tiff_container_format(tf)
        if not tf.series:
            return VolumeSourceProbe(
                container_format=container_format,
                errors=("TIFF contains no image series.",),
            )
        candidates = _tiff_series_candidates(tf)
        if not candidates:
            return VolumeSourceProbe(
                container_format=container_format,
                errors=("TIFF contains no plausible spatial image series.",),
            )
        candidate_by_index = {candidate.index: candidate for candidate in candidates}
        if series_index is None and len(candidates) > 1:
            return VolumeSourceProbe(
                container_format=container_format,
                series_candidates=candidates,
                ambiguities=(
                    "TIFF contains several plausible image series; select one series explicitly before import.",
                ),
                resolution_required=True,
            )
        selected_index = candidates[0].index if series_index is None else int(series_index)
        if selected_index not in candidate_by_index:
            return VolumeSourceProbe(
                container_format=container_format,
                series_candidates=candidates,
                errors=(f"TIFF series index {selected_index} is not a plausible spatial image series.",),
            )
        candidate = candidate_by_index[selected_index]
        parent_series = tf.series[selected_index]
        levels = tuple(getattr(parent_series, "levels", ()) or (parent_series,))
        try:
            selected_level = int(pyramid_level)
        except (TypeError, ValueError):
            selected_level = -1
        if not 0 <= selected_level < len(levels):
            return VolumeSourceProbe(
                container_format=container_format,
                series_candidates=candidates,
                errors=(
                    f"TIFF pyramid level {pyramid_level!r} is unavailable for "
                    f"series index {selected_index}.",
                ),
            )
        series = levels[selected_level]
        labels = list(str(series.axes or "").upper())
        shape = tuple(int(value) for value in series.shape)
        if len(labels) != len(shape):
            return VolumeSourceProbe(
                container_format=container_format,
                errors=("TIFF series axes do not match its reported dimensions.",),
            )

        try:
            labels, shape, selected_scene, _scene_resolved = _selected_tiff_scene_axes(
                labels, shape, source_subselection
            )
        except ValueError as exc:
            errors.append(str(exc))
            selected_scene = {}
            _scene_resolved = False
        if selected_scene:
            resolution_provenance.append(
                {
                    "decision": "scene-selection",
                    "source": str(resolution_source),
                    "series_index": selected_index,
                    "source_subselection": dict(selected_scene),
                }
            )

        try:
            labels, applied_axis_resolution = _resolved_tiff_labels(
                labels, shape, axis_resolution
            )
        except ValueError as exc:
            errors.append(str(exc))
            applied_axis_resolution = ()
        if applied_axis_resolution:
            resolution_provenance.append(
                {
                    "decision": "axis-resolution",
                    "source": str(resolution_source),
                    "series_index": selected_index,
                    "axes": list(applied_axis_resolution),
                }
            )
        if series_index is not None:
            series_decision = {
                "decision": "series-selection",
                "source": str(resolution_source),
                "series_index": selected_index,
                "series_identity": tiff_series_identity(
                    tf,
                    selected_index,
                    pyramid_level=selected_level,
                    source_subselection=selected_scene,
                ),
            }
            if selected_level:
                series_decision["pyramid_level"] = selected_level
            resolution_provenance.append(series_decision)

        unknown = [
            index for index, label in enumerate(labels)
            if label not in {"X", "Y", "Z", "T", "C", "S"}
            and shape[index] > 1
        ]
        for required_label in ("X", "Y"):
            if labels.count(required_label) != 1:
                errors.append(
                    f"TIFF series must declare exactly one {required_label} axis."
                )
        for optional_label in ("Z", "T", "C", "S"):
            if labels.count(optional_label) > 1:
                errors.append(
                    f"TIFF series declares duplicate {optional_label} axes."
                )

        semantics = []
        for index, label in enumerate(labels):
            semantic = {
                "X": "space-x",
                "Y": "space-y",
                "Z": "space-z",
                "T": "time",
                "C": "channel",
                "S": "component",
            }.get(label, "unknown")
            semantics.append(semantic)
            if semantic == "component" and shape[index] > 1:
                ambiguities.append(
                    f"TIFF sample/component axis {index} has size {shape[index]} and is not a biological channel axis."
                )
            elif semantic == "unknown" and shape[index] > 1:
                ambiguities.append(
                    f"TIFF axis {label or index!r} has unknown semantics and size {shape[index]}."
                )

        imagej = dict(tf.imagej_metadata or {})
        lsm_metadata = dict(getattr(tf, "lsm_metadata", None) or {})
        ome_names, ome_pixels, _ome_identity = ome_series_metadata(
            tf.ome_metadata, selected_index
        )
        selected_series_identity = tiff_series_identity(
            tf,
            selected_index,
            pyramid_level=selected_level,
            source_subselection=selected_scene,
        )
        scalar_dtype, scalar_bit_depth = scalar_descriptor(series.dtype)
        scalar_error = _scalar_type_error(
            scalar_dtype, "Zeiss LSM" if is_lsm else "TIFF"
        )
        if scalar_error:
            errors.append(scalar_error)
        lsm_scan_info = lsm_scan_information(lsm_metadata)
        scan_direction = (
            str(lsm_scan_info.get("ScanDirection"))
            if is_lsm and lsm_scan_info.get("ScanDirection") is not None
            else first_explicit_text(
                imagej, "scan_direction", "madi3d_scan_direction"
            )
        )
        source_checksum = first_explicit_text(
            imagej, "source_checksum", "madi3d_source_checksum", "checksum"
        )
        acquisition_software_version = first_explicit_text(
            imagej,
            "acquisition_software_version",
            "acquisition_software",
            "software_version",
        )
        madi3d_geometry, madi3d_geometry_error = _madi3d_geometry_provenance(
            imagej.get("madi3d_geometry_provenance")
        )
        if madi3d_geometry_error:
            errors.append(madi3d_geometry_error)
        channel_count = int(shape[semantics.index("channel")]) if "channel" in semantics else 1
        if is_lsm:
            names, channel_order_verified = lsm_channel_names(
                lsm_metadata, channel_count
            )
        else:
            names = (
                _names_metadata(imagej.get("madi3d_channel_names")) or ome_names
            )
            channel_order_verified = True
        if len(names) != channel_count:
            names = tuple(
                names[index]
                if index < len(names) and names[index]
                else f"ch{index + 1}"
                for index in range(channel_count)
            )
        channel_indices = list(range(channel_count))
        channel_order_ambiguous = False
        if channel_count > 1 and channel_order is not None:
            requested_order = tuple(str(value) for value in channel_order)
            if (
                len(set(requested_order)) != len(requested_order)
                or set(requested_order) != set(names)
            ):
                errors.append(
                    "Explicit TIFF channel order does not match the selected series channels."
                )
            else:
                channel_indices = [names.index(value) for value in requested_order]
                names = requested_order
                channel_order_verified = True
                resolution_provenance.append(
                    {
                        "decision": "channel-order",
                        "source": str(resolution_source),
                        "order": list(names),
                    }
                )
        elif is_lsm and channel_count > 1:
            if channel_order_verified:
                resolution_provenance.append(
                    {
                        "decision": "channel-order",
                        "source": "zeiss-lsm-metadata",
                        "order": list(names),
                    }
                )
            else:
                channel_order_ambiguous = True
                ambiguities.append(
                    "Zeiss LSM channel order could not be verified from source metadata; "
                    "confirm the channel order before import."
                )
        resolved_axes = "".join(labels)
        if channel_count > 1:
            selectors = tuple(
                {
                    "series_index": selected_index,
                    "series_identity": selected_series_identity,
                    **(
                        {"source_subselection": dict(selected_scene)}
                        if selected_scene
                        else {}
                    ),
                    **(
                        {"pyramid_level": selected_level}
                        if selected_level
                        else {}
                    ),
                    "channel": source_channel_index,
                    "axes": resolved_axes,
                }
                for source_channel_index in channel_indices
            )
        else:
            selectors = ({
                "series_index": selected_index,
                "series_identity": selected_series_identity,
                **(
                    {"source_subselection": dict(selected_scene)}
                    if selected_scene
                    else {}
                ),
                **(
                    {"pyramid_level": selected_level}
                    if selected_level
                    else {}
                ),
                "channel": None,
                "axes": resolved_axes,
            },)

        page = series.pages[0]
        raw_bit_depth = getattr(page, "bitspersample", None)
        try:
            bit_values = tuple(
                int(value)
                for value in np.asarray(raw_bit_depth).reshape(-1)
            )
        except Exception:
            bit_values = ()
        if (
            bit_values
            and len(set(bit_values)) == 1
            and bit_values[0] > 0
            and (
                scalar_bit_depth is None
                or bit_values[0] <= scalar_bit_depth
            )
        ):
            scalar_bit_depth = bit_values[0]
        if is_lsm:
            lsm_channel_bits = []
            for track in lsm_scan_info.get("Tracks") or ():
                if not isinstance(track, dict):
                    continue
                for record in track.get("DataChannels") or ():
                    if not isinstance(record, dict):
                        continue
                    try:
                        value = int(record.get("BitsPerSample"))
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        lsm_channel_bits.append(value)
            if (
                lsm_channel_bits
                and len(set(lsm_channel_bits)) == 1
                and (
                    scalar_bit_depth is None
                    or lsm_channel_bits[0] <= scalar_bit_depth
                )
            ):
                scalar_bit_depth = lsm_channel_bits[0]
        raw_resolution = {}
        for tag_name in ("XResolution", "YResolution"):
            try:
                raw_resolution[tag_name] = page.tags[tag_name].value
            except Exception:
                raw_resolution[tag_name] = None
        sx, sy = _resolution_spacing(tf, page)
        sz = None
        raw_z_spacing = imagej.get("spacing")
        if raw_z_spacing is not None and not (
            isinstance(raw_z_spacing, (str, bytes)) and not raw_z_spacing
        ):
            try:
                z_spacing_candidate = float(raw_z_spacing)
            except Exception:
                z_spacing_candidate = math.nan
            if math.isfinite(z_spacing_candidate) and z_spacing_candidate > 0:
                sz = z_spacing_candidate
            else:
                physical_geometry_diagnostics.append(
                    "TIFF ImageJ Z spacing must be finite and positive."
                )
        ome_spacing = []
        ome_spacing_present = False
        for axis in ("X", "Y", "Z"):
            raw_value = ome_pixels.get(f"PhysicalSize{axis}")
            ome_spacing_present = ome_spacing_present or raw_value not in (None, "")
            try:
                value = float(raw_value)
            except Exception:
                value = math.nan
            ome_spacing.append(value)
        if all(math.isfinite(value) and value > 0 for value in ome_spacing):
            sx, sy, sz = ome_spacing
        elif ome_spacing_present:
            physical_geometry_diagnostics.append(
                "TIFF OME physical sizes must contain three finite positive values."
            )

        imagej_unit = str(imagej.get("unit") or "").strip()
        ome_units = tuple(
            str(ome_pixels.get(f"PhysicalSize{axis}Unit") or "").strip()
            for axis in ("X", "Y", "Z")
        )
        unit_values = None
        if all(ome_units):
            try:
                canonical_ome_units = canonical_space_units(ome_units)
                unit_values = ome_units
                if imagej_unit and canonical_space_units(imagej_unit) != canonical_ome_units:
                    physical_geometry_diagnostics.append(
                        "TIFF ImageJ and OME metadata declare conflicting physical units."
                    )
            except ValueError as exc:
                physical_geometry_diagnostics.append(
                    f"TIFF OME metadata declares conflicting physical units: {exc}"
                )
        elif any(ome_units):
            physical_geometry_diagnostics.append(
                "TIFF OME metadata declares physical units for only some spatial axes."
            )
        elif imagej_unit:
            unit_values = (imagej_unit,) * 3
        origin = None
        direction = None
        raw_affine = imagej.get("madi3d_affine")
        if raw_affine is not None and not (
            isinstance(raw_affine, (str, bytes)) and not raw_affine
        ):
            try:
                if isinstance(raw_affine, bytes):
                    raw_affine = raw_affine.decode("utf-8", "replace")
                affine = np.asarray(json.loads(str(raw_affine)), dtype=float).reshape(4, 4)
                parts = grid_components_from_affine(affine)
                if ome_spacing_present and all(
                    value is not None for value in (sx, sy, sz)
                ) and not np.allclose(
                    (sx, sy, sz), parts["spacing"], rtol=1e-6, atol=1e-9
                ):
                    physical_geometry_diagnostics.append(
                        "TIFF OME sizes and MADI3D affine declare conflicting spacing."
                    )
                sx, sy, sz = parts["spacing"]
                origin = parts["origin"]
                direction = tuple(
                    tuple(float(value) for value in row)
                    for row in parts["direction"]
                )
            except Exception as exc:
                physical_geometry_diagnostics.append(
                    f"TIFF MADI3D affine metadata is invalid: {exc}"
                )
        raw_madi_units = imagej.get("madi3d_space_units")
        madi_units_explicit = False
        if raw_madi_units is not None and not (
            isinstance(raw_madi_units, (str, bytes)) and not raw_madi_units
        ):
            try:
                if isinstance(raw_madi_units, bytes):
                    raw_madi_units = raw_madi_units.decode("utf-8", "replace")
                decoded_madi_units = json.loads(str(raw_madi_units))
                if decoded_madi_units is not None:
                    madi_units = canonical_space_units(
                        tuple(decoded_madi_units)
                    )
                    madi_units_explicit = True
                    if (
                        unit_values is not None
                        and canonical_space_units(unit_values) != madi_units
                    ):
                        physical_geometry_diagnostics.append(
                            "TIFF metadata and MADI3D metadata declare conflicting physical units."
                        )
                    unit_values = madi_units
            except Exception as exc:
                physical_geometry_diagnostics.append(
                    f"TIFF MADI3D physical-unit metadata is invalid: {exc}"
                )
        lsm_spacing_present = False
        lsm_origin_present = False
        if is_lsm:
            raw_lsm_spacing = tuple(
                lsm_metadata.get(f"VoxelSize{axis}") for axis in ("X", "Y", "Z")
            )
            raw_lsm_origin = tuple(
                lsm_metadata.get(f"Origin{axis}") for axis in ("X", "Y", "Z")
            )
            lsm_spacing_present = any(value is not None for value in raw_lsm_spacing)
            lsm_origin_present = any(value is not None for value in raw_lsm_origin)
            lsm_spacing = finite_lsm_triplet(
                lsm_metadata,
                ("VoxelSizeX", "VoxelSizeY", "VoxelSizeZ"),
                positive=True,
                scale=1_000_000.0,
            )
            if lsm_spacing is not None:
                sx, sy, sz = lsm_spacing
            elif lsm_spacing_present:
                sx = sy = sz = None
                physical_geometry_diagnostics.append(
                    "Zeiss LSM voxel sizes must contain three finite positive values."
                )
            lsm_origin = finite_lsm_triplet(
                lsm_metadata,
                ("OriginX", "OriginY", "OriginZ"),
                scale=1_000_000.0,
            )
            if lsm_origin is not None:
                origin = lsm_origin
            elif lsm_origin_present:
                physical_geometry_diagnostics.append(
                    "Zeiss LSM origin must contain three finite values."
                )
            direction, direction_diagnostics = lsm_direction(
                lsm_scan_info
            )
            physical_geometry_diagnostics.extend(direction_diagnostics)
            unit_values = ("micron", "micron", "micron")

            reduced = False
            try:
                reduced = bool(series.pages[0].is_reduced)
            except Exception:
                pass
            if reduced and tf.series:
                parent_index = _lsm_parent_series_index(tf, selected_index)
                main = (
                    tf.series[parent_index]
                    if parent_index is not None
                    else None
                )
            else:
                parent_index = None
                main = None
            if main is not None:
                main_axes = str(main.axes or "").upper()
                main_shape = tuple(int(value) for value in main.shape)
                scale_xyz = [1.0, 1.0, 1.0]
                for output, axis in enumerate(("X", "Y", "Z")):
                    if axis in labels and axis in main_axes:
                        scale_xyz[output] = (
                            float(main_shape[main_axes.index(axis)])
                            / float(shape[labels.index(axis)])
                        )
                if sx is not None:
                    sx *= scale_xyz[0]
                if sy is not None:
                    sy *= scale_xyz[1]
                if sz is not None:
                    sz *= scale_xyz[2]
                resolution_provenance.append(
                    {
                        "decision": "auxiliary-series-grid",
                        "source": "zeiss-lsm-reduced-series",
                        "parent_series_index": parent_index,
                        "series_index": selected_index,
                        "spacing_scale_xyz": scale_xyz,
                    }
                )
        if selected_level:
            base_labels = list(str(parent_series.axes or "").upper())
            base_shape = tuple(int(value) for value in parent_series.shape)
            pyramid_scale = [1.0, 1.0, 1.0]
            for output, axis in enumerate(("X", "Y", "Z")):
                if axis not in labels or axis not in base_labels:
                    continue
                base_size = int(base_shape[base_labels.index(axis)])
                level_size = int(shape[labels.index(axis)])
                pyramid_scale[output] = float(base_size) / float(level_size)
            base_sx, base_sy = _resolution_spacing(
                tf, parent_series.pages[0]
            )
            ome_spacing_valid = all(
                math.isfinite(value) and value > 0 for value in ome_spacing
            )
            if sx is not None and (
                ome_spacing_valid
                or not (
                    base_sx is not None
                    and not math.isclose(sx, base_sx, rel_tol=1e-9, abs_tol=1e-12)
                )
            ):
                sx *= pyramid_scale[0]
            if sy is not None and (
                ome_spacing_valid
                or not (
                    base_sy is not None
                    and not math.isclose(sy, base_sy, rel_tol=1e-9, abs_tol=1e-12)
                )
            ):
                sy *= pyramid_scale[1]
            if sz is not None:
                sz *= pyramid_scale[2]
            resolution_provenance.append(
                {
                    "decision": "pyramid-level-grid",
                    "source": "tiff-subifd-pyramid",
                    "parent_series_index": selected_index,
                    "pyramid_level": selected_level,
                    "spacing_scale_xyz": pyramid_scale,
                }
            )
        time_count = int(shape[semantics.index("time")]) if "time" in semantics else 1
        time_interval = 1.0
        for key in ("finterval", "timeincrement", "TimeIncrement"):
            try:
                value = float(imagej.get(key))
                if math.isfinite(value) and value > 0:
                    time_interval = value
                    break
            except Exception:
                pass
        if time_interval == 1.0:
            try:
                value = float(ome_pixels.get("TimeIncrement"))
                if math.isfinite(value) and value > 0:
                    time_interval = value
            except Exception:
                pass
        if is_lsm:
            try:
                lsm_interval = float(lsm_metadata.get("TimeIntervall"))
            except (TypeError, ValueError):
                lsm_interval = math.nan
            if math.isfinite(lsm_interval) and lsm_interval > 0:
                time_interval = lsm_interval
            time_units = "s"
        else:
            time_units = str(
                imagej.get("tunit")
                or imagej.get("timeunit")
                or ome_pixels.get("TimeIncrementUnit")
                or "frame"
            )
        raw_field_values = dict(
            dimensions=_spatial_dimensions(shape, semantics),
            resolution=raw_resolution,
            imagej_z_spacing=raw_z_spacing,
            ome_physical_sizes={
                axis: ome_pixels.get(f"PhysicalSize{axis}")
                for axis in ("X", "Y", "Z")
            },
            imagej_unit=imagej_unit or None,
            ome_space_units=ome_units,
            madi3d_affine=raw_affine,
            madi3d_space_units=raw_madi_units,
            time_point_count=time_count,
            time_interval=time_interval,
            time_units=time_units,
        )
        if is_lsm:
            raw_field_values.update(
                {
                    "lsm_voxel_size_m": {
                        axis: lsm_metadata.get(f"VoxelSize{axis}")
                        for axis in ("X", "Y", "Z")
                    },
                    "lsm_origin_m": {
                        axis: lsm_metadata.get(f"Origin{axis}")
                        for axis in ("X", "Y", "Z")
                    },
                    "lsm_scan_type": lsm_metadata.get("ScanType"),
                    "lsm_scan_direction": lsm_scan_info.get(
                        "ScanDirection"
                    ),
                    "lsm_orientation_angles": {
                        name: lsm_scan_info.get(name)
                        for name in ("Rotation", "Nutation", "Precession")
                    },
                    "lsm_time_stamps": lsm_metadata.get("TimeStamps"),
                    "source_subselection": dict(selected_scene),
                }
            )
        if selected_level:
            raw_field_values["pyramid_level"] = selected_level
        raw_fields = _raw_fields(**raw_field_values)
        container_format = tiff_container_format(
            tf, ome_pixels=ome_pixels, imagej=imagej
        )

    spacing = (
        (float(sx), float(sy), float(sz))
        if all(value is not None for value in (sx, sy, sz))
        else None
    )
    space_units = _optional_units3(unit_values)
    missing_fields = []
    affine_present = not _raw_missing(raw_affine)
    ome_spacing_presence = tuple(
        not _raw_missing(ome_pixels.get(f"PhysicalSize{axis}"))
        for axis in ("X", "Y", "Z")
    )
    if is_lsm:
        if sx is None:
            missing_fields.append("spacing_x")
        if sy is None:
            missing_fields.append("spacing_y")
        if sz is None:
            missing_fields.append("spacing_z")
    else:
        if sx is None and _raw_missing(raw_resolution["XResolution"]) and not ome_spacing_presence[0] and not affine_present:
            missing_fields.append("spacing_x")
        if sy is None and _raw_missing(raw_resolution["YResolution"]) and not ome_spacing_presence[1] and not affine_present:
            missing_fields.append("spacing_y")
        if sz is None and _raw_missing(raw_z_spacing) and not ome_spacing_presence[2] and not affine_present:
            missing_fields.append("spacing_z")
    units_present = is_lsm or bool(imagej_unit) or any(ome_units) or madi_units_explicit
    if space_units is None and not units_present:
        missing_fields.append("space_units")
    if origin is None and not affine_present:
        missing_fields.append("origin")
    if direction is None and not affine_present:
        missing_fields.append("direction")

    return VolumeSourceProbe(
        container_format=container_format,
        series_identity=selected_series_identity,
        series_index=selected_index,
        series_candidates=candidates,
        axis_semantics=tuple(
            AxisSemantic(index, shape[index], semantics[index], labels[index])
            for index in range(len(shape))
        ),
        channel_count=channel_count,
        channel_selectors=selectors,
        channel_names=names,
        dimensions=_spatial_dimensions(shape, semantics),
        spacing=spacing,
        space_units=space_units,
        origin=origin,
        direction=direction,
        time_count=time_count,
        time_interval=time_interval,
        time_units=time_units,
        scalar_dtype=scalar_dtype,
        scalar_bit_depth=scalar_bit_depth,
        scan_direction=scan_direction,
        source_checksum=source_checksum,
        acquisition_software_version=acquisition_software_version,
        ambiguities=tuple(ambiguities),
        errors=tuple(errors),
        missing_fields=tuple(missing_fields),
        raw_fields=raw_fields,
        physical_geometry_diagnostics=tuple(physical_geometry_diagnostics),
        resolution_required=bool(unknown) or channel_order_ambiguous,
        resolution_provenance=tuple(resolution_provenance),
        madi3d_geometry_provenance=madi3d_geometry,
    )


def _nrrd_list(header: dict[str, Any], key: str, count: int) -> list[Any]:
    value = header.get(key)
    if value is None:
        return [None] * count
    try:
        result = list(value)
    except Exception:
        result = [value]
    return (result + [None] * count)[:count]


def _nrrd_scalar_descriptor(value):
    aliases = {
        "signed char": "int8",
        "int8_t": "int8",
        "uchar": "uint8",
        "unsigned char": "uint8",
        "uint8_t": "uint8",
        "short": "int16",
        "short int": "int16",
        "signed short": "int16",
        "signed short int": "int16",
        "int16_t": "int16",
        "ushort": "uint16",
        "unsigned short": "uint16",
        "unsigned short int": "uint16",
        "uint16_t": "uint16",
        "int": "int32",
        "signed int": "int32",
        "int32_t": "int32",
        "uint": "uint32",
        "unsigned int": "uint32",
        "uint32_t": "uint32",
        "longlong": "int64",
        "long long": "int64",
        "int64_t": "int64",
        "ulonglong": "uint64",
        "unsigned long long": "uint64",
        "uint64_t": "uint64",
        "float": "float32",
        "double": "float64",
    }
    raw = _decoded(value).strip().lower()
    return scalar_descriptor(aliases.get(raw, raw))


def _probe_nrrd(
    path: str,
    *,
    axis_resolution: Optional[dict[Any, str]] = None,
    resolution_source: str = "explicit-import-plan",
    header=None,
) -> VolumeSourceProbe:
    import nrrd

    header = header if header is not None else nrrd.read_header(path)
    scalar_dtype, scalar_bit_depth = _nrrd_scalar_descriptor(header.get("type"))
    scan_direction = first_explicit_text(
        header, "scan direction", "scan_direction", "madi3d_scan_direction"
    )
    source_checksum = first_explicit_text(
        header, "source checksum", "source_checksum", "madi3d_source_checksum"
    )
    acquisition_software_version = first_explicit_text(
        header,
        "acquisition software version",
        "acquisition_software_version",
        "software_version",
    )
    madi3d_geometry, madi3d_geometry_error = _madi3d_geometry_provenance(
        header.get("madi3d_geometry_provenance")
    )
    try:
        shape = tuple(int(value) for value in header.get("sizes", ()))
    except Exception:
        shape = ()
    if not shape or any(value <= 0 for value in shape):
        return VolumeSourceProbe(
            container_format="nrrd",
            errors=("NRRD header does not contain valid positive axis sizes.",),
        )

    count = len(shape)
    kinds = [_decoded(value).strip().lower() for value in _nrrd_list(header, "kinds", count)]
    labels = [_decoded(value).strip().strip('"').lower() for value in _nrrd_list(header, "labels", count)]
    directions = _nrrd_list(header, "space directions", count)
    names = (
        _names_metadata(header.get("madi3d_channel_names"))
        or _names_metadata(header.get("channel names"))
        or _names_metadata(header.get("channel_names"))
    )

    explicit_channel = [
        index for index, label in enumerate(labels)
        if label in {"c", "channel", "channels"}
    ]
    nonspatial_kinds = {"list", "vector", "color", "rgb-color", "rgba-color", "3-color", "4-color"}
    component_candidates = [
        index for index, kind in enumerate(kinds) if kind in nonspatial_kinds
    ]
    kind_only_channels = [
        index for index, kind in enumerate(kinds) if kind == "list"
    ]
    if not explicit_channel and len(kind_only_channels) == 1:
        explicit_channel = kind_only_channels
    if not explicit_channel and names:
        matching = [index for index in component_candidates if shape[index] == len(names)]
        if len(matching) == 1:
            explicit_channel = matching

    errors = []
    physical_geometry_diagnostics = []
    ambiguities = []
    if madi3d_geometry_error:
        errors.append(madi3d_geometry_error)
    scalar_error = _scalar_type_error(scalar_dtype, "NRRD")
    if scalar_error or scalar_dtype is None:
        errors.append(
            scalar_error or "NRRD declares an unsupported or unreadable scalar type."
        )
    if len(explicit_channel) > 1:
        errors.append("NRRD declares more than one channel axis.")
    channel_axis = explicit_channel[0] if len(explicit_channel) == 1 else None

    semantics = ["unknown"] * count
    if channel_axis is not None:
        semantics[channel_axis] = "channel"
    for index, kind in enumerate(kinds):
        if kind == "time":
            semantics[index] = "time"
        elif index != channel_axis and kind in nonspatial_kinds:
            semantics[index] = "component"

    spatial = []
    for index, value in enumerate(directions):
        if semantics[index] != "unknown" or value is None:
            continue
        try:
            vector = np.asarray(value, dtype=float).reshape(-1)[:3]
            if vector.size == 3 and np.all(np.isfinite(vector)):
                spatial.append(index)
        except Exception:
            pass
    for index, label in enumerate(labels):
        if semantics[index] != "unknown":
            continue
        if label in {"x", "space-x"}:
            semantics[index] = "space-x"
        elif label in {"y", "space-y"}:
            semantics[index] = "space-y"
        elif label in {"z", "space-z"}:
            semantics[index] = "space-z"
    assigned_spatial = {value for value in semantics if value.startswith("space-")}
    for index in spatial + list(range(count)):
        if semantics[index] != "unknown":
            continue
        missing = next(
            (name for name in ("space-x", "space-y", "space-z") if name not in assigned_spatial),
            None,
        )
        if missing is None:
            break
        if kinds[index] in {"domain", "space", ""} or index in spatial:
            semantics[index] = missing
            assigned_spatial.add(missing)

    applied_axis_resolution = []
    decisions = dict(axis_resolution or {})
    aliases = {
        "x": "space-x",
        "y": "space-y",
        "z": "space-z",
        "c": "channel",
        "t": "time",
        "s": "component",
    }
    for index, semantic in enumerate(tuple(semantics)):
        if semantic not in {"component", "unknown"}:
            continue
        decision = decisions.get(
            index,
            decisions.get(
                str(index),
                decisions.get(labels[index], decisions.get(kinds[index])),
            ),
        )
        if decision in (None, ""):
            continue
        resolved = aliases.get(
            str(decision).strip().lower(), str(decision).strip().lower()
        )
        if resolved not in {
            "channel",
            "component",
            "time",
            "space-x",
            "space-y",
            "space-z",
        }:
            errors.append(
                f"Unsupported NRRD axis-resolution decision {decision!r} "
                f"for axis {index}."
            )
            continue
        semantics[index] = resolved
        applied_axis_resolution.append(
            {
                "axis_index": index,
                "source_label": labels[index] or kinds[index],
                "semantic": resolved,
            }
        )

    for semantic, label in (
        ("channel", "channel"),
        ("time", "time"),
        ("space-x", "X"),
        ("space-y", "Y"),
        ("space-z", "Z"),
    ):
        if semantics.count(semantic) > 1:
            errors.append(f"NRRD declares duplicate {label} axes.")

    channel_axis = (
        semantics.index("channel") if semantics.count("channel") == 1 else None
    )

    for index, semantic in enumerate(semantics):
        if shape[index] <= 1:
            continue
        if semantic == "component":
            ambiguities.append(
                f"NRRD {kinds[index] or 'non-spatial'} axis {index} has size {shape[index]} and is not explicitly a biological channel axis."
            )
        elif semantic == "unknown":
            ambiguities.append(
                f"NRRD axis {index} has unknown semantics and size {shape[index]}."
            )

    spacing = [None, None, None]
    direction = np.full((3, 3), np.nan, dtype=float)
    raw_spacings = _nrrd_list(header, "spacings", count)
    raw_spatial_directions = [None, None, None]
    raw_spatial_spacings = [None, None, None]
    for output, semantic in enumerate(("space-x", "space-y", "space-z")):
        if semantic not in semantics:
            continue
        source = semantics.index(semantic)
        raw_spatial_directions[output] = directions[source]
        raw_spatial_spacings[output] = raw_spacings[source]
        try:
            vector = np.asarray(directions[source], dtype=float).reshape(-1)[:3]
            norm = float(np.linalg.norm(vector))
            if vector.size == 3 and math.isfinite(norm) and norm > 1e-12:
                spacing[output] = norm
                direction[:, output] = vector / norm
                continue
        except Exception:
            pass
        if not _raw_missing(directions[source]):
            physical_geometry_diagnostics.append(
                f"NRRD {semantic} space direction must be finite and nonzero."
            )
        try:
            value = float(raw_spacings[source])
            if math.isfinite(value) and value > 1e-12:
                spacing[output] = value
            elif not _raw_missing(raw_spacings[source]):
                physical_geometry_diagnostics.append(
                    f"NRRD {semantic} spacing must be finite and positive."
                )
        except Exception:
            if not _raw_missing(raw_spacings[source]):
                physical_geometry_diagnostics.append(
                    f"NRRD {semantic} spacing must be finite and positive."
                )

    channel_count = int(shape[channel_axis]) if channel_axis is not None else 1
    selectors = tuple(range(channel_count)) if channel_count > 1 else (None,)
    if channel_count > 1 and len(names) != channel_count:
        names = tuple(
            names[index] if index < len(names) and names[index] else f"ch{index + 1}"
            for index in range(channel_count)
        )
    time_count = int(shape[semantics.index("time")]) if "time" in semantics else 1
    time_interval = 1.0
    time_units = "frame"
    if "time" in semantics:
        source = semantics.index("time")
        try:
            value = float(_nrrd_list(header, "spacings", count)[source])
            if math.isfinite(value) and value > 0:
                time_interval = value
        except Exception:
            pass
        raw_unit = _nrrd_list(header, "units", count)[source]
        if raw_unit not in (None, ""):
            time_units = _decoded(raw_unit).strip().strip('"') or "frame"

    spacing_value = (
        tuple(float(value) for value in spacing)
        if all(value is not None for value in spacing)
        else None
    )
    direction_value = (
        _optional_matrix3(direction)
        if np.all(np.isfinite(direction))
        else None
    )
    if np.all(np.isfinite(direction)) and direction_value is None:
        physical_geometry_diagnostics.append(
            "NRRD space directions must define a finite nonsingular grid."
        )
    raw_origin = header.get("space origin")
    origin = _optional_vector3(raw_origin)
    if not _raw_missing(raw_origin) and origin is None:
        physical_geometry_diagnostics.append(
            "NRRD space origin must contain three finite values."
        )
    used_writer_origin = False
    if origin is None:
        writer_origin = nrrd_writer_local_origin(path, header)
        if writer_origin is not None:
            origin = writer_origin
            used_writer_origin = True
    raw_space_units = header.get("space units")
    space_units = _optional_units3(raw_space_units)
    if not _raw_missing(raw_space_units) and space_units is None:
        physical_geometry_diagnostics.append(
            "NRRD spatial units are unsupported, ambiguous, or inconsistent."
        )
    missing_fields = []
    for axis, raw_direction, raw_spacing in zip(
        ("x", "y", "z"), raw_spatial_directions, raw_spatial_spacings
    ):
        if _raw_missing(raw_direction) and _raw_missing(raw_spacing):
            missing_fields.append(f"spacing_{axis}")
    if direction_value is None and any(
        _raw_missing(value) for value in raw_spatial_directions
    ):
        missing_fields.append("direction")
    if origin is None and _raw_missing(raw_origin) and not used_writer_origin:
        missing_fields.append("origin")
    if space_units is None and _raw_missing(raw_space_units):
        missing_fields.append("space_units")
    raw_fields = _raw_fields(
        dimensions=_spatial_dimensions(shape, semantics),
        space_directions=raw_spatial_directions,
        spacings=raw_spatial_spacings,
        origin=raw_origin,
        space_units=raw_space_units,
        time_point_count=time_count,
        time_interval=time_interval,
        time_units=time_units,
    )

    return VolumeSourceProbe(
        container_format="nhdr" if path.lower().endswith(".nhdr") else "nrrd",
        series_identity="header",
        axis_semantics=tuple(
            AxisSemantic(index, shape[index], semantics[index], labels[index] or kinds[index])
            for index in range(count)
        ),
        channel_count=channel_count,
        channel_selectors=selectors,
        channel_names=names,
        dimensions=_spatial_dimensions(shape, semantics),
        spacing=spacing_value,
        space_units=space_units,
        origin=origin,
        direction=direction_value,
        time_count=time_count,
        time_interval=time_interval,
        time_units=time_units,
        scalar_dtype=scalar_dtype,
        scalar_bit_depth=scalar_bit_depth,
        scan_direction=scan_direction,
        source_checksum=source_checksum,
        acquisition_software_version=acquisition_software_version,
        ambiguities=tuple(ambiguities),
        errors=tuple(errors),
        missing_fields=tuple(missing_fields),
        raw_fields=raw_fields,
        physical_geometry_diagnostics=tuple(physical_geometry_diagnostics),
        resolution_provenance=(
            (
                {
                    "decision": "axis-resolution",
                    "source": str(resolution_source),
                    "axes": applied_axis_resolution,
                },
            )
            if applied_axis_resolution
            else ()
        )
        + (
            ({
                "decision": "format-local-origin",
                "source": "nrrd-writer-profile",
                "origin": list(origin),
            },)
            if used_writer_origin
            else ()
        ),
        madi3d_geometry_provenance=madi3d_geometry,
    )


def _probe_nifti(path: str, *, image=None) -> VolumeSourceProbe:
    import nibabel

    image = image if image is not None else nibabel.load(path, mmap=True)
    header = image.header
    madi3d_geometry, madi3d_geometry_error = (
        _nifti_madi3d_geometry_provenance(header)
    )
    shape = tuple(int(value) for value in image.shape)
    scalar_dtype, scalar_bit_depth = scalar_descriptor(image.get_data_dtype())
    if len(shape) < 2 or any(value <= 0 for value in shape):
        return VolumeSourceProbe(
            container_format="nifti",
            errors=(f"NIfTI has unsupported shape {shape}.",),
        )
    try:
        spatial_unit, time_unit = header.get_xyzt_units()
    except Exception:
        spatial_unit, time_unit = "unknown", "unknown"
    spatial_unit = str(spatial_unit or "unknown")
    time_unit = str(time_unit or "unknown")
    known_time = time_unit.lower() not in {"", "unknown", "none"}

    semantics = ["space-x", "space-y"]
    if len(shape) >= 3:
        semantics.append("space-z")
    ambiguities = []
    for index in range(3, len(shape)):
        # NIfTI reserves dimension 4 for time.  The xyzt_units field describes
        # the interval's unit; it does not determine the axis semantic.
        if index == 3:
            semantics.append("time")
        elif shape[index] == 1:
            semantics.append("component")
        else:
            semantics.append("component")
            ambiguities.append(
                f"NIfTI component dimension {index + 1} has size {shape[index]} and is not a biological channel axis."
            )

    affine = np.asarray(image.affine, dtype=float)
    sform_code = int(header["sform_code"])
    qform_code = int(header["qform_code"])
    try:
        sform_affine = np.asarray(header.get_sform(), dtype=float)
    except Exception:
        sform_affine = None
    try:
        qform_affine = np.asarray(header.get_qform(), dtype=float)
    except Exception:
        qform_affine = None
    selected_affine_source = (
        "sform" if sform_code != 0 else "qform" if qform_code != 0 else "base"
    )
    errors = []
    physical_geometry_diagnostics = []
    nifti_affine_diagnostics = []
    if madi3d_geometry_error:
        errors.append(madi3d_geometry_error)
    scalar_error = _scalar_type_error(scalar_dtype, "NIfTI")
    if scalar_error:
        errors.append(scalar_error)
    if (
        sform_code != 0
        and qform_code != 0
        and sform_affine is not None
        and qform_affine is not None
        # Sform rows and qform quaternion fields are stored with finite header
        # precision. Ignore representation-level roundoff while retaining
        # materially different coded transforms as explicit source conflict.
        and not np.allclose(
            sform_affine, qform_affine, rtol=1e-5, atol=1e-8
        )
    ):
        nifti_affine_diagnostics.append(
            "NIfTI coded sform and qform affines conflict; the loader selected "
            "the sform affine."
        )
    spacing_value = None
    direction_value = None
    origin = None
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        physical_geometry_diagnostics.append(
            "NIfTI voxel-to-world affine is not a finite 4 x 4 matrix."
        )
    else:
        origin = _optional_vector3(affine[:3, 3])
        linear = affine[:3, :3]
        spacing = np.linalg.norm(linear, axis=0)
        if np.any(~np.isfinite(spacing)) or np.any(spacing <= 1e-12):
            physical_geometry_diagnostics.append(
                "NIfTI voxel-to-world affine has invalid spatial spacing."
            )
        else:
            direction = linear / spacing
            if abs(float(np.linalg.det(direction))) < 1e-12:
                physical_geometry_diagnostics.append(
                    "NIfTI voxel-to-world direction is singular."
                )
            else:
                spacing_value = tuple(float(value) for value in spacing)
                direction_value = _optional_matrix3(direction)
                if direction_value is None:
                    physical_geometry_diagnostics.append(
                        "NIfTI voxel-to-world direction is invalid."
                    )

    space_units = _optional_units3(spatial_unit)
    missing_fields = []
    affine_valid = affine.shape == (4, 4) and np.all(np.isfinite(affine))
    if not affine_valid:
        pass
    elif spacing_value is None and not physical_geometry_diagnostics:
        missing_fields.append("spacing")
    if direction_value is None and not affine_valid:
        pass
    elif direction_value is None and not physical_geometry_diagnostics:
        missing_fields.append("direction")
    if origin is None and not affine_valid:
        pass
    elif origin is None:
        missing_fields.append("origin")
    if space_units is None and spatial_unit.lower() in {"", "unknown", "none"}:
        missing_fields.append("space_units")

    zooms = tuple(float(value) for value in header.get_zooms())
    time_count = int(shape[3]) if len(shape) > 3 and semantics[3] == "time" else 1
    time_interval = (
        float(zooms[3])
        if time_count > 1 and len(zooms) > 3 and math.isfinite(zooms[3]) and zooms[3] > 0
        else 1.0
    )
    raw_fields = _raw_fields(
        dimensions=tuple((list(shape[:3]) + [1, 1, 1])[:3]),
        affine=affine,
        nifti_sform_affine=sform_affine,
        nifti_sform_code=sform_code,
        nifti_qform_affine=qform_affine,
        nifti_qform_code=qform_code,
        nifti_selected_affine_source=selected_affine_source,
        nifti_affine_diagnostics=tuple(nifti_affine_diagnostics),
        space_units=spatial_unit,
        time_point_count=time_count,
        time_interval=(zooms[3] if len(zooms) > 3 else None),
        time_units=time_unit,
    )
    return VolumeSourceProbe(
        container_format="nifti",
        series_identity="primary-image",
        axis_semantics=tuple(
            AxisSemantic(index, shape[index], semantics[index], f"dim{index + 1}")
            for index in range(len(shape))
        ),
        channel_count=1,
        channel_selectors=(None,),
        dimensions=tuple((list(shape[:3]) + [1, 1, 1])[:3]),
        spacing=spacing_value,
        space_units=space_units,
        origin=origin,
        direction=direction_value,
        time_count=time_count,
        time_interval=time_interval,
        time_units=time_unit if known_time else "frame",
        scalar_dtype=scalar_dtype,
        scalar_bit_depth=scalar_bit_depth,
        ambiguities=tuple(ambiguities),
        errors=tuple(errors),
        missing_fields=tuple(missing_fields),
        raw_fields=raw_fields,
        physical_geometry_diagnostics=tuple(physical_geometry_diagnostics),
        madi3d_geometry_provenance=madi3d_geometry,
    )


def _probe_leica(
    path: str,
    *,
    series_index: Optional[int] = None,
    axis_resolution: Optional[dict[Any, str]] = None,
    channel_order: Optional[tuple[str, ...]] = None,
    resolution_source: str = "explicit-import-plan",
    cancel_check=None,
    inspection: Optional[LeicaSourceInspection] = None,
) -> VolumeSourceProbe:
    """Resolve one metadata-only Leica image through the common probe contract."""
    container_format = source_error_container_format(path)
    try:
        source = inspection or inspect_leica_lif_source(
            path, cancel_check=cancel_check
        )
    except LeicaInspectionError as exc:
        return VolumeSourceProbe(
            container_format=container_format,
            errors=(str(exc),),
        )
    candidates = tuple(
        SeriesCandidate(
            index=item.index,
            identity=item.identity,
            axes="".join(item.axes),
            shape=item.shape,
        )
        for item in source.series
    )
    if not candidates:
        return VolumeSourceProbe(
            container_format=source.container_format,
            errors=("Leica LIF source contains no inspectable image series.",),
            microscopy_source_metadata=source.source_metadata,
        )
    if series_index is None and len(candidates) > 1:
        return VolumeSourceProbe(
            container_format=source.container_format,
            series_candidates=candidates,
            ambiguities=(
                "Leica LIF source contains several independent images; "
                "select one image explicitly before import.",
            ),
            resolution_required=True,
            microscopy_source_metadata=source.source_metadata,
        )
    selected_index = candidates[0].index if series_index is None else int(series_index)
    selected = next(
        (item for item in source.series if item.index == selected_index), None
    )
    if selected is None:
        return VolumeSourceProbe(
            container_format=source.container_format,
            series_candidates=candidates,
            errors=(f"Leica LIF series index {selected_index} is unavailable.",),
            microscopy_source_metadata=source.source_metadata,
        )

    original_labels = list(selected.axes)
    labels = list(original_labels)
    shape = selected.shape
    errors = []
    ambiguities = []
    resolution_provenance = []
    if axis_resolution:
        errors.append(
            "Leica LIF nonstandard axes cannot be remapped in this importer; "
            "the source layout remains unsupported."
        )
    if series_index is not None:
        resolution_provenance.append(
            {
                "decision": "series-selection",
                "source": str(resolution_source),
                "series_index": selected_index,
                "series_identity": selected.identity,
            }
        )
    for required_label in ("X", "Y"):
        if labels.count(required_label) != 1:
            errors.append(
                f"Leica LIF image must declare exactly one {required_label} axis."
            )
    for optional_label in ("Z", "T", "C", "S"):
        if labels.count(optional_label) > 1:
            errors.append(
                f"Leica LIF image declares duplicate {optional_label} axes."
            )
    if selected.raw_fields.get("is_flim"):
        errors.append(
            "Leica LIF FLIM/TCSPC histogram images are not supported by the "
            "biological-volume decoder."
        )

    semantics = []
    for index, label in enumerate(labels):
        semantic = {
            "X": "space-x",
            "Y": "space-y",
            "Z": "space-z",
            "T": "time",
            "C": "channel",
            "S": "component",
        }.get(label, "unknown")
        semantics.append(semantic)
        if shape[index] <= 1:
            continue
        original_label = original_labels[index]
        if original_label == "M":
            errors.append(
                "Leica LIF mosaic/tile axis M is not flattened or reinterpreted; "
                "this image layout is unsupported."
            )
        elif semantic == "component":
            ambiguities.append(
                f"Leica LIF component axis {index} has size {shape[index]} and "
                "is not a biological channel axis."
            )
        elif semantic == "unknown":
            ambiguities.append(
                f"Leica LIF axis {original_label or index!r} has unknown semantics "
                f"and size {shape[index]}."
            )

    for dtype in selected.channel_scalar_dtypes or (selected.scalar_dtype,):
        scalar_error = _scalar_type_error(dtype, "Leica LIF")
        if scalar_error:
            errors.append(scalar_error)
    channel_count = (
        int(shape[semantics.index("channel")]) if "channel" in semantics else 1
    )
    names = tuple(selected.channel_names)
    channel_metadata = tuple(selected.channel_metadata)
    channel_dtypes = tuple(selected.channel_scalar_dtypes)
    channel_bits = tuple(selected.channel_scalar_bit_depths)
    if len(names) != channel_count:
        names = tuple(
            names[index] if index < len(names) and names[index] else f"ch{index + 1}"
            for index in range(channel_count)
        )
    channel_indices = list(range(channel_count))
    if channel_count > 1 and channel_order is not None:
        requested_order = tuple(str(value) for value in channel_order)
        if (
            len(set(requested_order)) != len(requested_order)
            or len(set(names)) != len(names)
            or set(requested_order) != set(names)
        ):
            errors.append(
                "Explicit Leica LIF channel order does not match the selected image channels."
            )
        else:
            channel_indices = [names.index(value) for value in requested_order]
            names = requested_order
            channel_metadata = tuple(channel_metadata[index] for index in channel_indices)
            channel_dtypes = tuple(channel_dtypes[index] for index in channel_indices)
            channel_bits = tuple(channel_bits[index] for index in channel_indices)
            resolution_provenance.append(
                {
                    "decision": "channel-order",
                    "source": str(resolution_source),
                    "order": list(names),
                }
            )
    elif channel_count > 1:
        resolution_provenance.append(
            {
                "decision": "channel-order",
                "source": "leica-channel-description-bytes-inc",
                "order": list(names),
            }
        )

    selectors = tuple(
        {
            "series_index": selected_index,
            "series_identity": selected.identity,
            "channel": source_channel_index if channel_count > 1 else None,
            "axes": list(labels),
        }
        for source_channel_index in channel_indices
    )
    unresolved_axes = any(
        semantic in {"component", "unknown"} and shape[index] > 1
        for index, semantic in enumerate(semantics)
    )
    return VolumeSourceProbe(
        container_format=source.container_format,
        series_identity=selected.identity,
        series_index=selected_index,
        series_candidates=candidates,
        axis_semantics=tuple(
            AxisSemantic(index, shape[index], semantics[index], labels[index])
            for index in range(len(shape))
        ),
        channel_count=channel_count,
        channel_selectors=selectors,
        channel_names=names,
        dimensions=_spatial_dimensions(shape, semantics),
        spacing=selected.spacing,
        space_units=selected.space_units,
        origin=None,
        direction=None,
        time_count=selected.time_count,
        time_interval=selected.time_interval,
        time_units=selected.time_units,
        scalar_dtype=selected.scalar_dtype,
        scalar_bit_depth=selected.scalar_bit_depth,
        channel_scalar_dtypes=channel_dtypes,
        channel_scalar_bit_depths=channel_bits,
        ambiguities=tuple(ambiguities),
        errors=tuple(errors),
        missing_fields=selected.missing_fields,
        raw_fields=copy.deepcopy(selected.raw_fields),
        physical_geometry_diagnostics=selected.physical_geometry_diagnostics,
        resolution_required=unresolved_axes,
        resolution_provenance=tuple(resolution_provenance),
        microscopy_source_metadata=copy.deepcopy(source.source_metadata),
        microscopy_acquisition_metadata=copy.deepcopy(
            selected.acquisition_metadata
        ),
        microscopy_channel_metadata=copy.deepcopy(channel_metadata),
    )


def _probe_olympus(
    path: str,
    *,
    series_index: Optional[int] = None,
    axis_resolution: Optional[dict[Any, str]] = None,
    channel_order: Optional[tuple[str, ...]] = None,
    resolution_source: str = "explicit-import-plan",
    cancel_check=None,
    inspection: Optional[OlympusSourceInspection] = None,
) -> VolumeSourceProbe:
    """Resolve one metadata-only Olympus series through the common probe contract."""
    container_format = source_error_container_format(path)
    try:
        source = inspection or inspect_olympus_source(
            path, cancel_check=cancel_check
        )
    except OlympusInspectionError as exc:
        return VolumeSourceProbe(
            container_format=container_format,
            errors=(str(exc),),
        )
    candidates = tuple(
        SeriesCandidate(
            index=item.index,
            identity=item.identity,
            axes="".join(item.axes),
            shape=item.shape,
        )
        for item in source.series
    )
    if not candidates:
        return VolumeSourceProbe(
            container_format=source.container_format,
            errors=("Olympus source contains no inspectable image series.",),
            microscopy_source_metadata=source.source_metadata,
        )
    if series_index is None and len(candidates) > 1:
        return VolumeSourceProbe(
            container_format=source.container_format,
            series_candidates=candidates,
            ambiguities=(
                "Olympus source contains several independent image series; "
                "select one series explicitly before import.",
            ),
            resolution_required=True,
            microscopy_source_metadata=source.source_metadata,
        )
    selected_index = candidates[0].index if series_index is None else int(series_index)
    selected = next(
        (item for item in source.series if item.index == selected_index), None
    )
    if selected is None:
        return VolumeSourceProbe(
            container_format=source.container_format,
            series_candidates=candidates,
            errors=(f"Olympus series index {selected_index} is unavailable.",),
            microscopy_source_metadata=source.source_metadata,
        )

    labels = list(selected.axes)
    shape = selected.shape
    errors = []
    ambiguities = []
    resolution_provenance = []
    try:
        labels, applied_axis_resolution = _resolved_tiff_labels(
            labels, shape, axis_resolution
        )
    except ValueError as exc:
        errors.append(str(exc))
        applied_axis_resolution = ()
    if applied_axis_resolution:
        resolution_provenance.append(
            {
                "decision": "axis-resolution",
                "source": str(resolution_source),
                "series_index": selected_index,
                "axes": list(applied_axis_resolution),
            }
        )
    if series_index is not None:
        resolution_provenance.append(
            {
                "decision": "series-selection",
                "source": str(resolution_source),
                "series_index": selected_index,
                "series_identity": selected.identity,
            }
        )
    for required_label in ("X", "Y"):
        if labels.count(required_label) != 1:
            errors.append(
                f"Olympus series must declare exactly one {required_label} axis."
            )
    for optional_label in ("Z", "T", "C", "S"):
        if labels.count(optional_label) > 1:
            errors.append(f"Olympus series declares duplicate {optional_label} axes.")
    semantics = []
    for index, label in enumerate(labels):
        semantic = {
            "X": "space-x",
            "Y": "space-y",
            "Z": "space-z",
            "T": "time",
            "C": "channel",
            "S": "component",
        }.get(label, "unknown")
        semantics.append(semantic)
        if shape[index] <= 1:
            continue
        if semantic == "component":
            ambiguities.append(
                f"Olympus component axis {index} has size {shape[index]} and "
                "is not a biological channel axis."
            )
        elif semantic == "unknown":
            ambiguities.append(
                f"Olympus axis {label or index!r} has unknown semantics and "
                f"size {shape[index]}."
            )

    scalar_error = _scalar_type_error(selected.scalar_dtype, "Olympus OIF/OIB")
    if scalar_error:
        errors.append(scalar_error)
    channel_count = int(shape[semantics.index("channel")]) if "channel" in semantics else 1
    names = tuple(selected.channel_names)
    if len(names) != channel_count:
        names = tuple(
            names[index] if index < len(names) and names[index] else f"ch{index + 1}"
            for index in range(channel_count)
        )
    channel_indices = list(range(channel_count))
    channel_metadata = tuple(selected.channel_metadata)
    channel_order_ambiguous = False
    if channel_count > 1 and channel_order is not None:
        requested_order = tuple(str(value) for value in channel_order)
        if (
            len(set(requested_order)) != len(requested_order)
            or len(set(names)) != len(names)
            or set(requested_order) != set(names)
        ):
            errors.append(
                "Explicit Olympus channel order does not match the selected series channels."
            )
        else:
            channel_indices = [names.index(value) for value in requested_order]
            names = requested_order
            channel_metadata = tuple(
                channel_metadata[index]
                for index in channel_indices
                if index < len(channel_metadata)
            )
            resolution_provenance.append(
                {
                    "decision": "channel-order",
                    "source": str(resolution_source),
                    "order": list(names),
                }
            )
    elif channel_count > 1 and not selected.channel_order_verified:
        channel_order_ambiguous = True
        ambiguities.append(
            "Olympus channel order could not be verified from source metadata; "
            "confirm the channel order before import."
        )
    elif channel_count > 1:
        resolution_provenance.append(
            {
                "decision": "channel-order",
                "source": "olympus-channel-settings",
                "order": list(names),
            }
        )

    resolved_axes = "".join(labels)
    selectors = tuple(
        {
            "series_index": selected_index,
            "series_identity": selected.identity,
            "channel": source_channel_index if channel_count > 1 else None,
            "axes": resolved_axes,
        }
        for source_channel_index in channel_indices
    )
    unknown = any(
        semantic in {"component", "unknown"} and shape[index] > 1
        for index, semantic in enumerate(semantics)
    )
    return VolumeSourceProbe(
        container_format=source.container_format,
        series_identity=selected.identity,
        series_index=selected_index,
        series_candidates=candidates,
        axis_semantics=tuple(
            AxisSemantic(index, shape[index], semantics[index], labels[index])
            for index in range(len(shape))
        ),
        channel_count=channel_count,
        channel_selectors=selectors,
        channel_names=names,
        dimensions=_spatial_dimensions(shape, semantics),
        spacing=selected.spacing,
        space_units=selected.space_units,
        origin=None,
        direction=None,
        time_count=selected.time_count,
        time_interval=selected.time_interval,
        time_units=selected.time_units,
        scalar_dtype=selected.scalar_dtype,
        scalar_bit_depth=selected.scalar_bit_depth,
        scan_direction=selected.scan_direction,
        acquisition_software_version=selected.acquisition_software_version,
        ambiguities=tuple(ambiguities),
        errors=tuple(errors),
        missing_fields=selected.missing_fields,
        raw_fields=copy.deepcopy(selected.raw_fields),
        physical_geometry_diagnostics=selected.physical_geometry_diagnostics,
        resolution_required=unknown or channel_order_ambiguous,
        resolution_provenance=tuple(resolution_provenance),
        microscopy_source_metadata=copy.deepcopy(source.source_metadata),
        microscopy_acquisition_metadata=copy.deepcopy(
            selected.acquisition_metadata
        ),
        microscopy_channel_metadata=copy.deepcopy(channel_metadata),
    )


def probe_volume_source(
    path: os.PathLike[str] | str,
    *,
    series_index: Optional[int] = None,
    pyramid_level: int = 0,
    source_subselection: Optional[dict[str, int]] = None,
    axis_resolution: Optional[dict[Any, str]] = None,
    channel_order: Optional[tuple[str, ...]] = None,
    resolution_source: str = "explicit-import-plan",
    cancel_check=None,
) -> VolumeSourceProbe:
    """Inspect one source without decoding or materializing its voxel payload."""
    source = os.path.abspath(os.fspath(path))
    reader_mode = volume_reader_mode(source)
    try:
        if reader_mode == "h5j":
            return _probe_h5j(
                source,
                channel_order=channel_order,
                resolution_source=resolution_source,
            )
        if reader_mode == "tiff":
            return _probe_tiff(
                source,
                series_index=series_index,
                pyramid_level=pyramid_level,
                source_subselection=source_subselection,
                axis_resolution=axis_resolution,
                channel_order=channel_order,
                resolution_source=resolution_source,
            )
        if reader_mode == "olympus":
            return _probe_olympus(
                source,
                series_index=series_index,
                axis_resolution=axis_resolution,
                channel_order=channel_order,
                resolution_source=resolution_source,
                cancel_check=cancel_check,
            )
        if reader_mode == "leica":
            return _probe_leica(
                source,
                series_index=series_index,
                axis_resolution=axis_resolution,
                channel_order=channel_order,
                resolution_source=resolution_source,
                cancel_check=cancel_check,
            )
        if reader_mode == "nrrd":
            return _probe_nrrd(
                source,
                axis_resolution=axis_resolution,
                resolution_source=resolution_source,
            )
        if reader_mode == "nifti":
            return _probe_nifti(source)
        return VolumeSourceProbe(
            container_format="unknown",
            errors=(f"Unsupported volume container: {os.path.basename(source)}",),
        )
    except InterruptedError:
        raise
    except Exception as exc:
        return VolumeSourceProbe(
            container_format=source_error_container_format(source),
            errors=(f"Could not inspect volume header: {exc}",),
        )
