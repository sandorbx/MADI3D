"""GUI-independent decoding of supported scientific volume payloads."""

from __future__ import annotations

from madi3d_app.io_utils import check_resources, file_revision, resolve_file_locator

import errno
import glob
import json
import math
import os
import shutil
import tempfile
import weakref
from collections import Counter
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from . import nrrd_stream
from .leica_lif import (
    decode_leica_lif_series_channels,
    inspect_leica_lif_source,
    LeicaChannelPixels,
)
from .nikon_nd2 import NikonChannelPixels, decode_nikon_nd2_series_channels
from .zeiss_czi import CziChannelPixels, decode_zeiss_czi_series_channels
from .olympus import (
    OlympusChannelPixels,
    decode_olympus_series_channels,
    inspect_olympus_source,
)
from .provenance import decoded_payload_descriptor
from .probe import (
    _h5j_channel_order,
    _nrrd_scalar_descriptor,
    ome_series_metadata,
    open_tiff_source,
    tiff_series_identity,
    tiff_time_metadata,
)
from .geometry import finite_tuple3
from .source_formats import nifti_runtime_time_unit, volume_reader_mode
from .scalars import prepare_volume_scalar_array, volume_runtime_dtype


_nrrd = None
_tifffile = None
_nibabel = None


def require_nrrd():
    global _nrrd
    if _nrrd is None:
        try:
            import nrrd as module
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency: pynrrd. Install with: pip install pynrrd"
            ) from exc
        _nrrd = module
    return _nrrd


def _known_shape_bytes(shape, dtype):
    """Return an array's byte size when both shape and scalar type are known."""
    if dtype is None:
        return None
    try:
        return math.prod(int(value) for value in shape) * np.dtype(dtype).itemsize
    except (TypeError, ValueError):
        return None


def require_tifffile():
    global _tifffile
    if _tifffile is None:
        try:
            import imagecodecs  # noqa: F401
        except Exception:
            pass
        try:
            import tifffile as module
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency: tifffile. Install with: pip install tifffile"
            ) from exc
        _tifffile = module
    return _tifffile


def require_nibabel():
    global _nibabel
    if _nibabel is None:
        try:
            import nibabel as module
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency: nibabel. Install with: pip install nibabel"
            ) from exc
        _nibabel = module
    return _nibabel


def safe_scalar_range(array):
    try:
        low = float(np.nanmin(array))
        high = float(np.nanmax(array))
        if math.isfinite(low) and math.isfinite(high):
            return low, high
    except Exception:
        pass
    return None


def normalize_volume_to_tzyx(data, axes, channel=None, *, reserve_memory=None):
    """Normalize named axes to T,Z,Y,X, retaining T even when singleton."""
    array = np.asanyarray(data)
    labels = [str(axis).upper() for axis in axes]
    if len(labels) != array.ndim:
        raise ValueError(
            f"Axis description {axes!r} does not match array shape {array.shape}."
        )

    def remove_axis(index, take=None):
        nonlocal array, labels
        if take is None:
            array = np.squeeze(array, axis=index)
        else:
            if reserve_memory is not None:
                reserve_memory(array.nbytes // array.shape[index], (array,))
            array = np.take(array, int(take), axis=index)
        labels.pop(index)

    for channel_label in ("C", "S"):
        while channel_label in labels:
            index = labels.index(channel_label)
            size = int(array.shape[index])
            if size == 1:
                remove_axis(index)
                continue
            if channel is None:
                raise ValueError(
                    f"Volume has {size} {channel_label}-components; select a channel first."
                )
            if not isinstance(channel, (int, np.integer)):
                raise ValueError(
                    f"Numeric channel index required for axis {channel_label}; got {channel!r}."
                )
            if not 0 <= int(channel) < size:
                raise IndexError(f"Channel {channel} is outside 0..{size - 1}.")
            remove_axis(index, take=int(channel))
            channel = None

    index = 0
    while index < len(labels):
        if labels[index] in ("X", "Y", "Z", "T"):
            index += 1
            continue
        if int(array.shape[index]) == 1:
            remove_axis(index)
            continue
        raise ValueError(
            f"Unsupported non-spatial axis {labels[index]!r} with size {array.shape[index]}."
        )
    if labels.count("T") > 1:
        raise ValueError("Multiple time axes are not supported.")
    if "X" not in labels or "Y" not in labels:
        raise ValueError(f"Volume must contain X and Y axes; got {''.join(labels)}.")
    if "Z" not in labels:
        array = np.expand_dims(array, axis=array.ndim)
        labels.append("Z")
    if "T" in labels:
        order = [labels.index(axis) for axis in ("T", "Z", "Y", "X")]
        return np.transpose(array, order)
    order = [labels.index(axis) for axis in ("Z", "Y", "X")]
    return np.transpose(array, order)[np.newaxis, ...]


def nrrd_time_metadata(header, axes):
    if "T" not in axes:
        return 1.0, "frame"
    source_axis = axes.index("T")
    spacing = 1.0
    try:
        values = header.get("spacings")
        if values is not None:
            candidate = float(list(values)[source_axis])
            if math.isfinite(candidate) and candidate > 0:
                spacing = candidate
    except Exception:
        pass
    units = "frame"
    try:
        values = header.get("units")
        if values is not None and source_axis < len(values):
            raw = values[source_axis]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            units = str(raw or "frame").strip('"') or "frame"
    except Exception:
        pass
    return spacing, units


def h5j_expected_dimensions(handle, channel_group):
    keys = ("frames", "height", "width")
    if all(key in channel_group.attrs for key in keys):
        return finite_tuple3(
            [np.asarray(channel_group.attrs[key]).reshape(-1)[0] for key in keys],
            "H5J dimensions", positive=True, integer=True,
        )
    if "image_size" in handle.attrs:
        width, height, frames = finite_tuple3(
            np.asarray(handle.attrs["image_size"]).reshape(-1),
            "H5J image size", positive=True, integer=True,
        )
        return frames, height, width
    return None


def crop_h5j_codec_padding(array, expected_dimensions):
    result = np.asanyarray(array)
    if expected_dimensions is None:
        return result
    expected = tuple(int(value) for value in expected_dimensions)
    if len(expected) != 3 or any(value <= 0 for value in expected):
        raise ValueError("H5J metadata contains invalid spatial dimensions.")
    if result.ndim != 3:
        raise ValueError(f"Expected decoded H5J Z,Y,X data, got shape {result.shape}.")
    if any(actual < wanted for actual, wanted in zip(result.shape, expected)):
        raise ValueError(
            "Decoded H5J data is smaller than its authoritative container dimensions: "
            f"decoded {result.shape}, expected {expected}."
        )
    return np.ascontiguousarray(
        result[: expected[0], : expected[1], : expected[2]]
    )


def _h5j_ffmpeg_input_options(header):
    """Preserve every frame in legacy H5J channels stored as MP4.

    Only the payload's first few bytes are required for the MP4-vs-HEVC
    transport detection; callers pass that bounded header, never a full read.
    """
    header = bytes(memoryview(header)[:8])
    if len(header) == 8 and header[4:8] == b"ftyp":
        return ("-ignore_editlist", "1")
    return ()


_H5J_TRANSPORT_CHUNK_BYTES = 1024 * 1024


def _h5j_encoded_payload_bytes(source):
    """Return the encoded byte count of an HDF5 dataset handle or a buffer."""
    import h5py

    if isinstance(source, h5py.Dataset):
        return int(np.prod(source.shape, dtype=np.int64)) * int(source.dtype.itemsize)
    view = memoryview(source)
    if view.ndim != 1 or view.format != "B":
        view = memoryview(np.ascontiguousarray(source)).cast("B")
    return len(view)


def _write_h5j_encoded_payload(source, destination_path, *, cancel_check=None):
    """Write one encoded H5J payload to FFmpeg's input file in fixed slabs.

    HDF5 dataset handles transfer with bounded <=1 MiB slices so import never
    needs the complete encoded channel in memory. Buffers stream through
    memoryview slices of the same bound. Returns the payload's first bytes for
    the legacy MP4 transport detection.
    """
    import h5py

    from madi3d_app.io_utils import check_cancelled

    dataset = isinstance(source, h5py.Dataset)
    if dataset:
        header_source = np.ascontiguousarray(source[:8])
        data = None
    else:
        data = memoryview(source)
        if data.ndim != 1 or data.format != "B":
            data = memoryview(np.ascontiguousarray(source)).cast("B")
        header_source = data[:8]
    header = memoryview(header_source).cast("B")[:8].tobytes()
    with open(destination_path, "wb") as stream:
        if dataset:
            itemsize = max(1, int(source.dtype.itemsize))
            block = max(1, _H5J_TRANSPORT_CHUNK_BYTES // itemsize)
            for offset in range(0, int(source.size), block):
                if offset:
                    check_cancelled(cancel_check)  # Cancellation between slabs.
                stream.write(
                    memoryview(
                        np.ascontiguousarray(source[offset:offset + block])
                    ).cast("B")
                )
        else:
            for offset in range(0, len(data), _H5J_TRANSPORT_CHUNK_BYTES):
                if offset:
                    check_cancelled(cancel_check)  # Cancellation between slabs.
                stream.write(data[offset:offset + _H5J_TRANSPORT_CHUNK_BYTES])
    return header


def run_h5j_ffmpeg(executable, args, *, cancel_check=None, operation="H5J"):
    from madi3d_app.integrations.ffmpeg.backend import (
        FFmpegProcessCancelled,
        FFmpegProcessTimeout,
        run_ffmpeg_process,
    )

    try:
        return run_ffmpeg_process(executable, args, cancel_check=cancel_check,
                                  operation=operation)
    except FFmpegProcessCancelled as exc:
        raise InterruptedError("H5J FFmpeg operation cancelled.") from exc
    except FFmpegProcessTimeout as exc:
        raise RuntimeError(str(exc)) from exc


@dataclass(frozen=True)
class VolumeDecodeContract:
    """Probe-owned instructions for interpreting one source payload."""

    container_format: str
    series_identity: str
    series_index: int | None
    source_axis_sizes: tuple[int, ...]
    source_axis_order: tuple[str, ...]
    source_axis_semantics: tuple[str, ...]
    selector: Any = None
    resolution_provenance: tuple[dict[str, Any], ...] = ()
    source_bundle_identity: str = ""

    def __post_init__(self):
        container_format = str(self.container_format or "").strip().lower()
        if not container_format:
            raise ValueError("Decode contract container format must be explicit.")
        object.__setattr__(self, "container_format", container_format)
        object.__setattr__(
            self, "series_identity", str(self.series_identity or "").strip()
        )
        if self.series_index in (None, ""):
            object.__setattr__(self, "series_index", None)
        else:
            if isinstance(self.series_index, bool):
                raise ValueError(
                    "Decode contract series index must be a non-negative integer."
                )
            series_index = int(self.series_index)
            if series_index < 0 or series_index != self.series_index:
                raise ValueError(
                    "Decode contract series index must be a non-negative integer."
                )
            object.__setattr__(self, "series_index", series_index)

        sizes = []
        for value in self.source_axis_sizes:
            if isinstance(value, bool):
                raise ValueError(
                    "Decode contract source-axis sizes must be positive integers."
                )
            size = int(value)
            if size <= 0 or size != value:
                raise ValueError(
                    "Decode contract source-axis sizes must be positive integers."
                )
            sizes.append(size)
        sizes = tuple(sizes)
        if not sizes:
            raise ValueError(
                "Decode contract source-axis sizes must be positive integers."
            )
        order = tuple(str(value or "") for value in self.source_axis_order)
        semantics = tuple(
            str(value or "unknown").strip().lower()
            for value in self.source_axis_semantics
        )
        if len({len(sizes), len(order), len(semantics)}) != 1:
            raise ValueError(
                "Decode contract axis sizes, labels, and semantics must have equal lengths."
            )
        supported = {
            "channel",
            "component",
            "time",
            "space-x",
            "space-y",
            "space-z",
            "unknown",
        }
        unsupported = sorted(set(semantics) - supported)
        if unsupported:
            raise ValueError(
                f"Decode contract contains unsupported axis semantics {unsupported!r}."
            )
        for semantic in ("channel", "time", "space-x", "space-y", "space-z"):
            if semantics.count(semantic) > 1:
                raise ValueError(
                    f"Decode contract declares duplicate {semantic!r} axes."
                )
        for semantic in ("space-x", "space-y"):
            if semantic not in semantics:
                raise ValueError(
                    f"Decode contract is missing its {semantic!r} axis."
                )
        object.__setattr__(self, "source_axis_sizes", sizes)
        object.__setattr__(self, "source_axis_order", order)
        object.__setattr__(self, "source_axis_semantics", semantics)
        object.__setattr__(
            self,
            "resolution_provenance",
            tuple(dict(value) for value in (self.resolution_provenance or ())),
        )
        object.__setattr__(
            self, "source_bundle_identity", str(self.source_bundle_identity or "").strip()
        )

    @property
    def channel_axis(self) -> int | None:
        return (
            self.source_axis_semantics.index("channel")
            if "channel" in self.source_axis_semantics
            else None
        )

    @property
    def spatial_axes(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (semantic, self.source_axis_semantics.index(semantic))
            for semantic in ("space-x", "space-y", "space-z")
            if semantic in self.source_axis_semantics
        )

    @property
    def time_axis(self) -> int | None:
        return (
            self.source_axis_semantics.index("time")
            if "time" in self.source_axis_semantics
            else None
        )

    @property
    def normalization_axes(self) -> tuple[str, ...]:
        labels = {
            "channel": "C",
            "component": "S",
            "time": "T",
            "space-x": "X",
            "space-y": "Y",
            "space-z": "Z",
        }
        return tuple(
            labels.get(semantic, f"U{index}")
            for index, semantic in enumerate(self.source_axis_semantics)
        )

    @classmethod
    def from_probe(cls, probe, *, selector=None) -> "VolumeDecodeContract":
        metadata = getattr(probe, "microscopy_source_metadata", None)
        additional = getattr(metadata, "additional_fields", {}) if metadata else {}
        bundle = additional.get("source_bundle", {}) if isinstance(additional, dict) else {}
        return cls(
            container_format=probe.container_format,
            series_identity=probe.series_identity,
            series_index=probe.series_index,
            source_axis_sizes=tuple(axis.size for axis in probe.axis_semantics),
            source_axis_order=probe.source_axis_order,
            source_axis_semantics=probe.source_axis_semantics,
            selector=selector,
            resolution_provenance=probe.resolution_provenance,
            source_bundle_identity=(
                bundle.get("manifest_identity", "") if isinstance(bundle, dict) else ""
            ),
        )

    @classmethod
    def from_backing_source(
        cls, backing_source, *, selector=None
    ) -> "VolumeDecodeContract":
        metadata = getattr(backing_source, "microscopy_source_metadata", None)
        additional = getattr(metadata, "additional_fields", {}) if metadata else {}
        bundle = additional.get("source_bundle", {}) if isinstance(additional, dict) else {}
        return cls(
            container_format=backing_source.format,
            series_identity=backing_source.series_identity,
            series_index=backing_source.series_index,
            source_axis_sizes=backing_source.source_axis_sizes,
            source_axis_order=backing_source.source_axis_order,
            source_axis_semantics=backing_source.source_axis_semantics,
            selector=selector,
            resolution_provenance=(
                backing_source.physical_grid_observation.resolution_provenance
            ),
            source_bundle_identity=(
                bundle.get("manifest_identity", "") if isinstance(bundle, dict) else ""
            ),
        )


def _validate_microscopy_series_request(selector, contract, inspection, *, source_name):
    """Match a typed, container-checked request to its currently inspected series."""
    if selector != contract.selector:
        raise ValueError(
            f"{source_name} selector does not match the probe-owned decode contract."
        )
    if not isinstance(selector, dict):
        raise TypeError(
            f"{source_name} selector must be a structured series/channel record."
        )
    try:
        selector_series_index = int(selector["series_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source_name} selector is missing its series index.") from exc
    if contract.series_index is None or selector_series_index != contract.series_index:
        raise ValueError(
            f"{source_name} selector series does not match the persisted decode contract."
        )
    selected = next(
        (item for item in inspection.series if item.index == contract.series_index),
        None,
    )
    if selected is None:
        raise ValueError(
            f"Persisted {source_name} series index {contract.series_index} is unavailable."
        )
    selector_identity = str(selector.get("series_identity") or "")
    if (
        not contract.series_identity
        or selector_identity != contract.series_identity
        or selected.identity != contract.series_identity
    ):
        raise ValueError(
            f"Persisted {source_name} series identity does not match the selected source series."
        )
    if tuple(selected.shape) != contract.source_axis_sizes:
        raise ValueError(
            f"{source_name} source shape {selected.shape!r} does not match the "
            f"probe-owned axis contract {contract.source_axis_sizes!r}."
        )
    return selected


@dataclass(frozen=True)
class DecodedVolumePayload:
    frame_zyx: np.ndarray
    scalar_range: tuple[float, float]
    time_info: dict[str, Any] | None
    source_descriptor: dict[str, Any] = field(default_factory=dict)
    backing_revisions: tuple[tuple[str, tuple | None], ...] = ()


_UNKNOWN_BACKING_PATH = "<unknown>"


def _resolved_backing_path(path, base_directory):
    resolved = resolve_file_locator(path, base_directory)
    return os.path.normcase(resolved) if os.path.isabs(resolved) else resolved


def _capture_backing_revisions(paths=(), *, unknown=False, base_directory=None):
    revisions = []
    seen = set()
    for path in paths:
        try:
            resolved = _resolved_backing_path(path, base_directory or os.curdir)
        except (TypeError, ValueError, OSError):
            unknown = True
            continue
        key = os.path.normcase(resolved)
        if key in seen:
            continue
        seen.add(key)
        try:
            revision = file_revision(resolved) if os.path.isabs(resolved) else None
        except (OSError, TypeError, ValueError):
            revision = None
        revisions.append((resolved, revision))
    if unknown:
        revisions.append((_UNKNOWN_BACKING_PATH, None))
    return tuple(revisions)


def verify_backing_revisions(revisions):
    """Reject changes to known pixel files; unknown evidence remains unknown."""
    for path, revision in revisions:
        if revision is None:
            continue
        try:
            current = file_revision(path)
        except OSError as exc:
            raise RuntimeError(
                f"External volume backing changed during decoding: {path}"
            ) from exc
        if current != revision:
            raise RuntimeError(
                f"External volume backing changed during decoding: {path}"
            )


def _nrrd_external_backing(header, descriptor_path):
    value = header.get("datafile", header.get("data file"))
    if value is None:
        return ()
    if not isinstance(value, (str, os.PathLike)):
        return ((_UNKNOWN_BACKING_PATH, None),)
    value = os.fspath(value)
    if not value:
        return ((_UNKNOWN_BACKING_PATH, None),)
    return _capture_backing_revisions(
        (value,), base_directory=os.path.dirname(os.path.abspath(descriptor_path))
    )


def _tiff_external_backing_paths(series, descriptor_path):
    """Find external files represented in the selected series' pages."""
    root_path = os.path.normcase(os.path.abspath(descriptor_path))
    paths = []
    unknown = False
    try:
        pages = series.pages
        for page in pages:
            parent = getattr(page, "parent", None)
            handle = getattr(parent, "filehandle", None)
            member_path = getattr(handle, "path", None)
            if member_path:
                resolved = _resolved_backing_path(
                    member_path, os.path.dirname(root_path)
                )
                if resolved != root_path:
                    paths.append(resolved)
        if not pages:
            unknown = True
    except Exception:
        unknown = True

    return tuple(paths), unknown


def _ome_external_backing_paths(ome_xml, series_index, descriptor_path):
    paths = []
    unknown = False
    root_path = os.path.normcase(os.path.abspath(descriptor_path))
    base = os.path.dirname(root_path)
    if ome_xml:
        try:
            root = ET.fromstring(ome_xml)
            images = [
                node for node in root.iter()
                if node.tag.rsplit("}", 1)[-1] == "Image"
            ]
            image = images[int(series_index)]
            pixels = next(
                node for node in image
                if node.tag.rsplit("}", 1)[-1] == "Pixels"
            )
            for tiff_data in pixels:
                if tiff_data.tag.rsplit("}", 1)[-1] != "TiffData":
                    continue
                for uuid_node in tiff_data:
                    if uuid_node.tag.rsplit("}", 1)[-1] != "UUID":
                        continue
                    if uuid_node.text and uuid_node.text == root.get("UUID"):
                        continue
                    filename = uuid_node.attrib.get("FileName")
                    if filename:
                        resolved = _resolved_backing_path(filename, base)
                        if resolved != root_path:
                            paths.append(resolved)
        except Exception:
            unknown = True
    return tuple(paths), unknown


class VolumePayloadDecoder:
    """Decode arrays and plain metadata without creating Qt or VTK objects."""

    def __init__(
        self,
        *,
        ffmpeg_executable=None,
        cancel_check: Callable[[], bool] | None = None,
        reserve_memory=None,
    ):
        self.ffmpeg_executable = (
            os.fspath(ffmpeg_executable) if ffmpeg_executable else None
        )
        self.cancel_check = cancel_check or (lambda: False)
        self._memory_reserved = False
        self.reserve_memory = reserve_memory

    def _check_resources(self, *, memory_bytes=0, disk_bytes=0, directory=None):
        """Keep disk checks while avoiding a duplicate RAM gate after admission."""
        check_resources(
            memory_bytes=0 if self._memory_reserved else memory_bytes,
            disk_bytes=disk_bytes,
            directory=directory,
        )

    @staticmethod
    def reader_mode(path):
        return volume_reader_mode(path)

    @staticmethod
    def _time_info(tzyx, spacing, units):
        count = int(tzyx.shape[0])
        return {
            "data": tzyx if count > 1 else None,
            "count": count,
            "spacing": float(spacing) if spacing and spacing > 0 else 1.0,
            "units": str(units or "frame"),
        }

    @classmethod
    def _payload(
        cls,
        tzyx,
        time_spacing=1.0,
        time_units="frame",
        *,
        source_axis_order=(),
        source_axis_semantics=(),
        backing_revisions=(),
        reserve_memory=None,
    ):
        source_array = np.asanyarray(tzyx)
        if reserve_memory is not None:
            dtype = source_array.dtype
            runtime = volume_runtime_dtype(dtype)
            normalization_bytes = (source_array.nbytes if not dtype.isnative else 0)
            if runtime != dtype.newbyteorder("="):
                normalization_bytes += source_array.size * runtime.itemsize
            if normalization_bytes:
                reserve_memory(normalization_bytes, (source_array,))
        tzyx = prepare_volume_scalar_array(tzyx)
        scalar_range = safe_scalar_range(tzyx) or (0.0, 1.0)
        if reserve_memory is not None and not tzyx[0].flags.c_contiguous:
            reserve_memory(tzyx[0].nbytes, (source_array, tzyx))
        frame = np.ascontiguousarray(tzyx[0])
        source_axis_order = tuple(str(value) for value in source_axis_order)
        if source_axis_semantics:
            source_axis_semantics = tuple(
                str(value) for value in source_axis_semantics
            )
        else:
            semantics = {
                "X": "space-x",
                "Y": "space-y",
                "Z": "space-z",
                "T": "time",
                "C": "channel",
                "S": "component",
                "CHANNELS": "channel",
            }
            source_axis_semantics = tuple(
                semantics.get(value.upper(), "unknown")
                for value in source_axis_order
            )
        dimensions = (
            int(tzyx.shape[3]),
            int(tzyx.shape[2]),
            int(tzyx.shape[1]),
        )
        return DecodedVolumePayload(
            frame_zyx=frame,
            scalar_range=scalar_range,
            time_info=cls._time_info(tzyx, time_spacing, time_units),
            source_descriptor=decoded_payload_descriptor(
                source_array,
                decoded_array=tzyx,
                dimensions=dimensions,
                time_point_count=int(tzyx.shape[0]),
                axis_order=source_axis_order,
                axis_semantics=source_axis_semantics,
            ),
            backing_revisions=tuple(backing_revisions),
        )

    @staticmethod
    def _tiff_axes(series, explicit_axes=None):
        return VolumePayloadDecoder._validated_tiff_axes(
            series.axes, series.shape, explicit_axes
        )

    @staticmethod
    def _validated_tiff_axes(source_axes, source_shape, explicit_axes=None):
        axes = str(source_axes).upper()
        shape = tuple(int(value) for value in source_shape)
        if explicit_axes not in (None, ""):
            candidate = str(explicit_axes).upper()
            if len(candidate) != len(shape):
                raise ValueError(
                    "Persisted TIFF axis resolution does not match the selected series shape."
                )
            axes = candidate
        unknown = [
            (index, label, shape[index])
            for index, label in enumerate(axes)
            if label not in ("X", "Y", "Z", "T", "C", "S")
            and shape[index] > 1
        ]
        if unknown:
            raise ValueError(
                f"TIFF contains unresolved axes {unknown}; spatial axes are not inferred from shape."
            )
        return axes, shape

    @staticmethod
    def _select_tiff_source_subspace(data, axes, source_subselection, *, reserve_memory=None):
        array = np.asanyarray(data)
        labels = [str(label).upper() for label in axes]
        supplied = {
            str(key).upper(): value
            for key, value in dict(source_subselection or {}).items()
        }
        scene_labels = {label for label in labels if label in {"P", "M"}}
        unknown = sorted(set(supplied) - scene_labels)
        if unknown:
            raise ValueError(
                f"Persisted TIFF scene selection contains unavailable axes {unknown!r}."
            )
        for axis_index in range(len(labels) - 1, -1, -1):
            label = labels[axis_index]
            if label not in {"P", "M"}:
                continue
            size = int(array.shape[axis_index])
            if size > 1 and label not in supplied:
                raise ValueError(
                    f"TIFF scene axis {label!r} requires an exact source selection."
                )
            value = supplied.get(label, 0)
            if isinstance(value, bool):
                raise ValueError("Persisted TIFF scene indices must be integers.")
            index = int(value)
            if index < 0 or index != value or index >= size:
                raise ValueError(
                    f"Persisted TIFF scene index {value!r} is unavailable for axis {label!r}."
                )
            if reserve_memory is not None:
                reserve_memory(array.nbytes // size, (array,))
            array = np.take(array, index, axis=axis_index)
            labels.pop(axis_index)
        return array, "".join(labels)

    def _decode_tiff_many(self, path, selectors):
        module = require_tifffile()
        parsed = []
        for selector in selectors:
            if isinstance(selector, dict):
                parsed.append(
                    (
                        int(selector.get("series_index", 0)),
                        str(selector.get("series_identity") or ""),
                        int(selector.get("pyramid_level", 0)),
                        dict(selector.get("source_subselection") or {}),
                        selector.get("channel"),
                        selector.get("axes"),
                    )
                )
            else:
                parsed.append((0, "", 0, {}, selector, None))
        source_revisions = _capture_backing_revisions((path,))
        with open_tiff_source(module.TiffFile, path) as (tiff, _):
            verify_backing_revisions(source_revisions)
            cache = {}
            ome_backings = {}
            remaining = Counter((entry[0], entry[2]) for entry in parsed)
            retained_outputs = []
            data = None

            def reserve(future_bytes, arrays=()):
                if self.cancel_check():
                    raise InterruptedError("Volume decoding was cancelled.")
                roots = []
                live_outputs = [ref for ref in retained_outputs if ref() is not None]
                retained_outputs[:] = live_outputs
                backing = [entry[0] for entry in cache.values()]
                current = () if data is None else (data,)
                for array in (*backing, *(ref() for ref in live_outputs), *current, *arrays):
                    if array is None:
                        continue
                    while isinstance(array.base, np.ndarray):
                        array = array.base
                    if not any(array is root for root in roots):
                        roots.append(array)
                if self.reserve_memory is not None:
                    self.reserve_memory(future_bytes, tuple(roots))
                else:
                    # A bare memory_reserved flag is not evidence that late
                    # header facts were admitted by an operation token.
                    check_resources(memory_bytes=future_bytes)
                if self.cancel_check():
                    raise InterruptedError("Volume decoding was cancelled.")

            for (
                series_index,
                series_identity,
                pyramid_level,
                source_subselection,
                channel,
                explicit_axes,
            ) in parsed:
                parent_key = (series_index, pyramid_level)
                payload = selected = None
                if self.cancel_check():
                    raise InterruptedError("Volume decoding was cancelled.")
                try:
                    if series_index not in ome_backings:
                        declared_paths, unknown_declared = _ome_external_backing_paths(
                            getattr(tiff, "ome_metadata", None), series_index, path
                        )
                        declared_revisions = _capture_backing_revisions(declared_paths)
                        # tifffile may silently zero planes for missing companions.
                        for member, revision in declared_revisions:
                            if revision is None or not os.path.isfile(member):
                                raise FileNotFoundError(
                                    errno.ENOENT, "Missing OME-TIFF pixel backing", member
                                )
                        ome_backings[series_index] = declared_revisions, unknown_declared
                    declared_revisions, unknown_declared = ome_backings[series_index]
                    if not 0 <= series_index < len(tiff.series):
                        raise ValueError(
                            f"Persisted TIFF series index {series_index} is unavailable."
                        )
                    parent_series = tiff.series[series_index]
                    levels = tuple(
                        getattr(parent_series, "levels", ()) or (parent_series,)
                    )
                    if not 0 <= pyramid_level < len(levels):
                        raise ValueError(
                            f"Persisted TIFF pyramid level {pyramid_level} is unavailable."
                        )
                    cache_key = (
                        source_revisions,
                        series_index,
                        pyramid_level,
                    )
                    series = levels[pyramid_level]
                    actual_identity = tiff_series_identity(
                        tiff,
                        series_index,
                        pyramid_level=pyramid_level,
                        source_subselection=source_subselection,
                    )
                    if series_identity and series_identity != actual_identity:
                        raise ValueError(
                            "Persisted TIFF series identity does not match the selected source series."
                        )
                    if cache_key not in cache:
                        member_paths, unknown_members = _tiff_external_backing_paths(
                            series, path
                        )
                        declared_members = dict(declared_revisions)
                        page_revisions = _capture_backing_revisions(
                            (member for member in member_paths if member not in declared_members),
                            unknown=unknown_declared or unknown_members,
                        )
                        backing_revisions = source_revisions + declared_revisions + page_revisions
                        source_dtype = np.dtype(series.dtype)
                        prepare_volume_scalar_array(np.empty(0, dtype=source_dtype))
                        reserve(math.prod(int(v) for v in series.shape) * source_dtype.itemsize)
                        verify_backing_revisions(backing_revisions)
                        data = series.asarray()
                        verify_backing_revisions(backing_revisions)
                        if self.cancel_check():
                            raise InterruptedError("Volume decoding was cancelled.")
                        cache[cache_key] = (data, backing_revisions)
                        data = None
                    data, backing_revisions = cache[cache_key]
                    verify_backing_revisions(backing_revisions)
                    data, source_axes = self._select_tiff_source_subspace(
                        data, str(series.axes).upper(), source_subselection,
                        reserve_memory=reserve,
                    )
                    axes, _shape = self._validated_tiff_axes(source_axes, data.shape, explicit_axes)
                    time_spacing, time_units = tiff_time_metadata(
                        dict(getattr(tiff, "imagej_metadata", None) or {}),
                        dict(getattr(tiff, "lsm_metadata", None) or {}),
                        ome_series_metadata(tiff.ome_metadata, series_index)[1],
                        is_lsm=bool(getattr(tiff, "is_lsm", False)),
                        time_axis_length=(
                            int(_shape[axes.index("T")]) if "T" in axes else 1
                        ),
                    )
                    selected = normalize_volume_to_tzyx(
                        data, axes, channel=channel, reserve_memory=reserve,
                    )
                    data = None
                    if remaining[parent_key] == 1:
                        cache.pop(cache_key, None)
                    payload = self._payload(
                        selected,
                        time_spacing,
                        time_units,
                        source_axis_order=axes,
                        backing_revisions=backing_revisions,
                        reserve_memory=reserve,
                    )
                    selected = None
                    # Only the returned payload and still-needed parents remain
                    # when byte staging and VTK publication begin.
                    if self.reserve_memory is not None:
                        reserve(2 * payload.frame_zyx.nbytes,
                                (payload.frame_zyx, payload.time_info.get("data")))
                    if self.cancel_check():
                        raise InterruptedError("Volume decoding was cancelled.")
                    verify_backing_revisions(backing_revisions)
                    for array in (payload.frame_zyx, payload.time_info.get("data")):
                        if array is not None:
                            while isinstance(array.base, np.ndarray):
                                array = array.base
                            retained_outputs.append(weakref.ref(array))
                    array = data = None
                    remaining[parent_key] -= 1
                    if not remaining[parent_key]:
                        cache.pop(cache_key, None)
                    yield payload
                    del payload
                except InterruptedError:
                    raise
                except Exception as exc:
                    # Do not retain failed allocation frames (and their arrays)
                    # in the exception yielded to a caller.
                    exc.__traceback__ = None
                    payload = selected = data = None
                    remaining[parent_key] -= 1
                    if not remaining[parent_key]:
                        cache.pop((source_revisions, *parent_key), None)
                    yield exc

    def _decode_nrrd_many(self, path, selectors, decode_contracts=None):
        selectors = list(selectors)
        if decode_contracts is None:
            from .probe import probe_volume_source

            probe = probe_volume_source(path)
            if probe.errors:
                raise ValueError("; ".join(probe.errors))
            if probe.requires_axis_resolution or probe.requires_series_selection:
                raise ValueError(
                    "; ".join(probe.ambiguities)
                    or "NRRD source axes require an explicit import decision."
                )
            decode_contracts = [
                VolumeDecodeContract.from_probe(probe, selector=selector)
                for selector in selectors
            ]
        else:
            decode_contracts = list(decode_contracts)
        if len(decode_contracts) != len(selectors):
            raise ValueError(
                "NRRD decode contracts must correspond one-to-one with selectors."
            )
        sizes = [contract.source_axis_sizes for contract in decode_contracts
                 if isinstance(contract, VolumeDecodeContract)]
        nrrd = require_nrrd()
        descriptor_path = os.fspath(path)
        header_before = nrrd.read_header(descriptor_path)
        source_dtype, _bit_depth = _nrrd_scalar_descriptor(header_before.get("type"))
        if sizes and source_dtype is not None:
            self._check_resources(
                memory_bytes=max(math.prod(size) for size in sizes) * np.dtype(source_dtype).itemsize
            )
        backing_revisions = _nrrd_external_backing(header_before, path)
        array, header = nrrd_stream.read_nrrd(
            descriptor_path, index_order="F", cancel_check=self.cancel_check
        )
        verify_backing_revisions(backing_revisions)
        for selector, contract in zip(selectors, decode_contracts):
            if self.cancel_check():
                raise InterruptedError("Volume decoding was cancelled.")
            try:
                if not isinstance(contract, VolumeDecodeContract):
                    raise TypeError("NRRD decoding requires a typed decode contract.")
                if contract.container_format not in {"nrrd", "nhdr"}:
                    raise ValueError(
                        "NRRD decode contract declares a different container format."
                    )
                if tuple(int(value) for value in array.shape) != (
                    contract.source_axis_sizes
                ):
                    raise ValueError(
                        f"NRRD source shape {array.shape!r} does not match the "
                        f"probe-owned axis contract {contract.source_axis_sizes!r}."
                    )
                if selector != contract.selector:
                    raise ValueError(
                        "NRRD selector does not match the probe-owned decode contract."
                    )
                axes = contract.normalization_axes
                time_spacing, time_units = nrrd_time_metadata(header, axes)
                yield self._payload(
                    normalize_volume_to_tzyx(array, axes, channel=selector),
                    time_spacing,
                    time_units,
                    source_axis_order=contract.source_axis_order,
                    source_axis_semantics=contract.source_axis_semantics,
                    backing_revisions=backing_revisions,
                )
            except Exception as exc:
                yield exc

    def _decode_nifti(self, path):
        nibabel = require_nibabel()
        image = nibabel.load(path, mmap=True)
        shape = getattr(image, "shape", None) or getattr(image.dataobj, "shape", ())
        source_dtype = getattr(image.dataobj, "dtype", None)
        # Nonidentity ArrayProxy scaling can widen the result. Stored scalar
        # width alone is not evidence of that materialized allocation.
        unscaled = (getattr(image.dataobj, "slope", None) == 1
                    and getattr(image.dataobj, "inter", None) == 0)
        allocation_bytes = _known_shape_bytes(shape, source_dtype) if unscaled else None
        if allocation_bytes is not None:
            self._check_resources(memory_bytes=allocation_bytes)
        data = np.asanyarray(image.dataobj)
        if data.ndim < 2:
            raise ValueError(f"NIfTI volume has unsupported shape {data.shape}.")
        if data.ndim == 2:
            axes = "XY"
        elif data.ndim == 3:
            axes = "XYZ"
        else:
            axes = list("XYZT")
            if data.ndim >= 5:
                axes.append("C")
            axes.extend(f"U{index}" for index in range(max(0, data.ndim - len(axes))))
        tzyx = normalize_volume_to_tzyx(data, axes)
        try:
            zooms = tuple(float(value) for value in image.header.get_zooms())
        except Exception:
            zooms = ()
        try:
            _spatial_units, time_units = image.header.get_xyzt_units()
        except Exception:
            time_units = None
        time_spacing = 1.0
        if len(zooms) >= 4 and math.isfinite(zooms[3]) and zooms[3] > 0:
            time_spacing = float(zooms[3])
        return self._payload(
            tzyx,
            time_spacing,
            nifti_runtime_time_unit(time_units),
            source_axis_order=axes,
        )

    def _decode_h5j_stream(self, source, expected_dimensions):
        if not self.ffmpeg_executable:
            raise RuntimeError(
                "H5J loading started without a prepared FFmpeg executable."
            )
        from madi3d_app.io_utils import check_cancelled
        from madi3d_app.integrations.ffmpeg.backend import (
            H5J_DECODER_FILTER, read_grayscale_y4m_header,
        )
        # The encoded copy is known; decoder scratch and output depth are not.
        self._check_resources(disk_bytes=_h5j_encoded_payload_bytes(source))
        temporary = tempfile.mkdtemp(prefix="h5j_")
        try:
            encoded = os.path.join(temporary, "data.h265")
            header = _write_h5j_encoded_payload(
                source, encoded, cancel_check=self.cancel_check
            )
            input_options = _h5j_ffmpeg_input_options(header)
            decoded_path = os.path.join(temporary, "decoded.y4m")
            process = run_h5j_ffmpeg(
                self.ffmpeg_executable,
                [
                    "-y", "-v", "error", *input_options, "-i", encoded,
                    # Negotiate native 8/12-bit gray without TIFF's 16-bit
                    # rescaling. FFmpeg also handles limited-range YUV input.
                    "-vf", H5J_DECODER_FILTER, "-strict", "-1",
                    "-fps_mode", "passthrough", "-f", "yuv4mpegpipe", decoded_path,
                ],
                cancel_check=self.cancel_check,
                operation="H5J decode",
            )
            if process.returncode != 0:
                stderr = (process.stderr or b"").decode("utf-8", "replace")[-4000:]
                raise RuntimeError(f"FFmpeg H5J decoding failed:\n{stderr}")
            with open(decoded_path, "rb") as stream:
                width, height, bit_depth = read_grayscale_y4m_header(stream)
                dtype = np.dtype(np.uint8 if bit_depth == 8 else "<u2")
                frame_bytes = width * height * dtype.itemsize
                # FFmpeg emits a fixed FRAME marker without per-frame tags.
                count, remainder = divmod(os.path.getsize(decoded_path) - stream.tell(), frame_bytes + 6)
                if count == 0 or remainder:
                    raise ValueError("H5J decoding produced missing or incomplete frames.")
                memory_bytes = count * frame_bytes
                if expected_dimensions is not None:
                    z, y, x = expected_dimensions
                    if y < height or x < width:
                        memory_bytes += z * y * x * dtype.itemsize
                self._check_resources(memory_bytes=memory_bytes)
                array = np.empty((count, height, width), dtype=dtype)
                for index in range(count):
                    check_cancelled(self.cancel_check)
                    if stream.read(6) != b"FRAME\n":
                        raise ValueError("Invalid decoded H5J frame marker.")
                    if stream.readinto(memoryview(array[index]).cast("B")) != frame_bytes:
                        raise ValueError("H5J decoding produced an incomplete frame.")
            return crop_h5j_codec_padding(array, expected_dimensions)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def _decode_h5j_many(self, path, selectors):
        import h5py

        with h5py.File(path, "r") as handle:
            group = handle["/Channels"]
            expected = h5j_expected_dimensions(handle, group)
            keys, explicit_order, order_error = _h5j_channel_order(handle, group)
            available = {str(key) for key in group.keys()}
            if order_error:
                raise ValueError(order_error)
            for selector in selectors:
                if self.cancel_check():
                    raise InterruptedError("Volume decoding was cancelled.")
                try:
                    if isinstance(selector, str):
                        key = selector
                    elif not explicit_order and len(keys) != 1:
                        raise ValueError(
                            "H5J channel order is unresolved; select channels by explicit name."
                        )
                    else:
                        key = keys[0 if selector is None else int(selector)]
                    if key not in available:
                        raise ValueError(f"H5J channel {key!r} is unavailable.")
                    # The dataset handle streams to FFmpeg's input file in
                    # bounded slabs; no complete encoded channel is read here.
                    array = self._decode_h5j_stream(group[key], expected)
                    payload = self._payload(
                        array[np.newaxis, ...],
                        source_axis_order=("Channels", "Z", "Y", "X"),
                        source_axis_semantics=(
                            "channel", "space-z", "space-y", "space-x"
                        ),
                    )
                    yield DecodedVolumePayload(
                        frame_zyx=payload.frame_zyx,
                        scalar_range=payload.scalar_range,
                        time_info=None,
                        source_descriptor=payload.source_descriptor,
                    )
                except InterruptedError:
                    raise
                except Exception as exc:
                    yield exc

    @staticmethod
    def _resolved_olympus_axes(source_axes, contract):
        labels = list(source_axes)
        semantic_labels = {
            "space-z": "Z",
            "time": "T",
            "channel": "C",
            "component": "S",
        }
        for record in contract.resolution_provenance:
            if record.get("decision") != "axis-resolution":
                continue
            for decision in record.get("axes") or ():
                index = int(decision.get("axis_index", -1))
                if not 0 <= index < len(labels):
                    raise ValueError(
                        "Olympus axis-resolution provenance references an unavailable axis."
                    )
                source_label = str(decision.get("source_label") or "").upper()
                if source_label and labels[index] != source_label:
                    raise ValueError(
                        "Olympus axis-resolution provenance does not match the decoded source axis."
                    )
                semantic = str(decision.get("semantic") or "").strip().lower()
                replacement = semantic_labels.get(semantic)
                if replacement is not None:
                    labels[index] = replacement
        return tuple(labels)

    @staticmethod
    def _align_olympus_array_to_probe_axes(decoded, selected):
        """Restore only singleton axes omitted by the TIFF file sequence.

        Olympus metadata retains axes such as T=1 and Z=1 even when those
        tokens are absent from TIFF member names. Filename tokens can also use
        a different order than the metadata. ``TiffSequence`` therefore returns
        a compact or permuted array. Only uniquely named axes are transposed;
        unknown or missing non-singleton axes remain errors.
        """
        array = np.asanyarray(decoded.array)
        actual_axes = list(decoded.source_axes)
        actual_shape = list(decoded.source_shape)
        expected_axes = tuple(selected.axes)
        expected_shape = tuple(selected.shape)
        if len(actual_axes) != len(actual_shape):
            raise ValueError(
                "Decoded Olympus axes and source dimensions are inconsistent."
            )
        if len(actual_axes) != array.ndim:
            raise ValueError(
                "Decoded Olympus axes do not match the selected channel array."
            )
        if len(expected_axes) != len(expected_shape):
            raise ValueError(
                "Probed Olympus axes and source dimensions are inconsistent."
            )
        axes_are_duplicated = (
            len(set(actual_axes)) != len(actual_axes)
            or len(set(expected_axes)) != len(expected_axes)
        )
        if axes_are_duplicated:
            raise ValueError("Olympus source axes must be uniquely identified.")

        for index in range(len(actual_axes) - 1, -1, -1):
            if actual_axes[index] in expected_axes:
                continue
            if actual_shape[index] != 1 or array.shape[index] != 1:
                raise ValueError(
                    "Decoded Olympus data contains an unexpected non-singleton axis."
                )
            array = np.squeeze(array, axis=index)
            actual_axes.pop(index)
            actual_shape.pop(index)

        expected_sizes = dict(zip(expected_axes, expected_shape))
        for axis, size in zip(actual_axes, actual_shape):
            if size != expected_sizes[axis]:
                raise ValueError(
                    f"Decoded Olympus axis {axis!r} has size {size}, not "
                    f"the probed size {expected_sizes[axis]}."
                )

        shared_expected = [
            axis for axis in expected_axes if axis in actual_axes
        ]
        permutation = [actual_axes.index(axis) for axis in shared_expected]
        if permutation != list(range(len(actual_axes))):
            array = np.transpose(array, permutation)
            actual_shape = [actual_shape[index] for index in permutation]
            actual_axes = list(shared_expected)

        for index, (axis, size) in enumerate(zip(expected_axes, expected_shape)):
            if axis in actual_axes:
                continue
            if size != 1:
                raise ValueError(
                    f"Decoded Olympus data omits non-singleton axis {axis!r}."
                )
            array = np.expand_dims(array, axis=index)
            actual_axes.insert(index, axis)
            actual_shape.insert(index, 1)

        selected_shape = list(expected_shape)
        if "C" in expected_axes:
            selected_shape[expected_axes.index("C")] = 1
        if (
            tuple(actual_axes) != expected_axes
            or tuple(actual_shape) != expected_shape
        ):
            raise ValueError(
                "Decoded Olympus axes could not be reconciled with the probe contract."
            )
        if tuple(array.shape) != tuple(selected_shape):
            raise ValueError(
                "Decoded Olympus selected-channel dimensions do not match the probe contract."
            )
        return array

    @classmethod
    def _validate_olympus_decode_request(cls, selector, contract, inspection):
        if not isinstance(contract, VolumeDecodeContract):
            raise TypeError("Olympus decoding requires a typed decode contract.")
        if contract.container_format not in {"olympus-oif", "olympus-oib"}:
            raise ValueError(
                "Olympus decode contract declares a different container format."
            )
        if contract.container_format != inspection.container_format:
            raise ValueError(
                "Olympus source container does not match the persisted decode contract."
            )
        current_metadata = inspection.source_metadata
        current_bundle = getattr(current_metadata, "additional_fields", {}).get(
            "source_bundle", {}
        )
        current_identity = (
            current_bundle.get("manifest_identity", "")
            if isinstance(current_bundle, dict) else ""
        )
        if (
            contract.source_bundle_identity
            and contract.source_bundle_identity != current_identity
        ):
            raise ValueError(
                "Olympus source family changed since this request was prepared; "
                "retry loading or relink the source."
            )
        selected = _validate_microscopy_series_request(
            selector, contract, inspection, source_name="Olympus"
        )
        resolved_axes = cls._resolved_olympus_axes(selected.axes, contract)
        if resolved_axes != contract.source_axis_order:
            raise ValueError(
                f"Olympus source axes {resolved_axes!r} do not match the "
                f"probe-owned axis order {contract.source_axis_order!r}."
            )
        if str(selector.get("axes") or "").upper() != "".join(resolved_axes):
            raise ValueError(
                "Olympus selector axes do not match the probe-owned decode contract."
            )
        semantic_by_label = {
            "C": "channel",
            "S": "component",
            "T": "time",
            "X": "space-x",
            "Y": "space-y",
            "Z": "space-z",
        }
        actual_semantics = tuple(
            semantic_by_label.get(label, "unknown") for label in resolved_axes
        )
        if actual_semantics != contract.source_axis_semantics:
            raise ValueError(
                "Olympus resolved axes do not match the persisted axis semantics."
            )
        channel = selector.get("channel")
        if contract.channel_axis is None:
            if channel is not None:
                raise ValueError(
                    "Olympus selector declares a channel for a series without a channel axis."
                )
        else:
            channel_count = contract.source_axis_sizes[contract.channel_axis]
            if channel is None and channel_count == 1:
                return selected, None
            if isinstance(channel, (bool, np.bool_)):
                raise ValueError("Olympus channel selector must be an integer.")
            try:
                channel_index = int(channel)
            except (TypeError, ValueError) as exc:
                raise ValueError("Olympus channel selector must be an integer.") from exc
            if channel_index != channel or not 0 <= channel_index < channel_count:
                raise ValueError(
                    f"Olympus channel {channel!r} is outside 0..{channel_count - 1}."
                )
            channel = channel_index
        return selected, channel

    def _decode_olympus_many(self, path, selectors, decode_contracts=None):
        selectors = list(selectors)
        if decode_contracts is None:
            decode_contracts = [None] * len(selectors)
        else:
            decode_contracts = list(decode_contracts)
        if len(decode_contracts) != len(selectors):
            raise ValueError(
                "Olympus decode contracts must correspond one-to-one with selectors."
            )
        inspection = inspect_olympus_source(path, cancel_check=self.cancel_check)
        results = [None] * len(selectors)
        groups = {}
        for index, (selector, contract) in enumerate(
            zip(selectors, decode_contracts)
        ):
            try:
                selected, channel = self._validate_olympus_decode_request(
                    selector, contract, inspection
                )
                groups.setdefault(selected.index, []).append(
                    (index, selected, channel, contract)
                )
            except Exception as exc:
                results[index] = exc

        for series_index, requests in groups.items():
            if self.cancel_check():
                raise InterruptedError("Volume decoding was cancelled.")
            selected_series = requests[0][1]
            if inspection.container_format == "olympus-oif":
                member_paths = tuple(selected_series.member_paths)
                backing_revisions = _capture_backing_revisions(
                    member_paths,
                    unknown=not bool(member_paths),
                    base_directory=os.path.dirname(os.path.abspath(path)),
                )
            else:
                backing_revisions = ()
            pixels = decode_olympus_series_channels(
                path,
                series_index,
                [request[2] for request in requests],
                cancel_check=self.cancel_check,
            )
            try:
                verify_backing_revisions(backing_revisions)
            except Exception as exc:
                for request in requests:
                    results[request[0]] = exc
                continue
            if len(pixels) != len(requests):
                raise RuntimeError(
                    "Olympus pixel reader returned the wrong number of channel results."
                )
            for request, decoded in zip(requests, pixels):
                result_index, selected, channel, contract = request
                if isinstance(decoded, Exception):
                    results[result_index] = decoded
                    continue
                try:
                    if not isinstance(decoded, OlympusChannelPixels):
                        raise TypeError(
                            "Olympus pixel reader returned an invalid channel payload."
                        )
                    if tuple(decoded.member_paths) != tuple(selected.member_paths):
                        raise ValueError(
                            "Decoded Olympus series members changed after source inspection."
                        )
                    source_array = self._align_olympus_array_to_probe_axes(
                        decoded, selected
                    )
                    resolved_axes = self._resolved_olympus_axes(selected.axes, contract)
                    if resolved_axes != contract.source_axis_order:
                        raise ValueError(
                            "Decoded Olympus source axes do not match the typed contract."
                        )
                    if decoded.scalar_dtype != selected.scalar_dtype:
                        raise ValueError(
                            f"Decoded Olympus scalar dtype {decoded.scalar_dtype!r} "
                            f"does not match probed dtype {selected.scalar_dtype!r}."
                        )
                    normalized = normalize_volume_to_tzyx(
                        source_array,
                        contract.normalization_axes,
                        channel=0 if contract.channel_axis is not None else None,
                    )
                    payload = self._payload(
                        normalized,
                        selected.time_interval,
                        selected.time_units,
                        source_axis_order=contract.source_axis_order,
                        source_axis_semantics=contract.source_axis_semantics,
                        backing_revisions=backing_revisions,
                    )
                    expected_dimensions = tuple(
                        contract.source_axis_sizes[
                            contract.source_axis_semantics.index(semantic)
                        ]
                        if semantic in contract.source_axis_semantics
                        else 1
                        for semantic in ("space-x", "space-y", "space-z")
                    )
                    descriptor = dict(payload.source_descriptor)
                    if tuple(descriptor["decoded_dimensions"]) != expected_dimensions:
                        raise ValueError(
                            "Decoded Olympus spatial dimensions do not match the probe contract."
                        )
                    if int(descriptor["decoded_time_point_count"]) != int(
                        selected.time_count
                    ):
                        raise ValueError(
                            "Decoded Olympus time count does not match the probe contract."
                        )
                    descriptor["source_scalar_dtype"] = selected.scalar_dtype
                    descriptor["source_scalar_bit_depth"] = selected.scalar_bit_depth
                    results[result_index] = DecodedVolumePayload(
                        frame_zyx=payload.frame_zyx,
                        scalar_range=payload.scalar_range,
                        time_info=payload.time_info,
                        source_descriptor=descriptor,
                        backing_revisions=backing_revisions,
                    )
                except Exception as exc:
                    results[result_index] = exc
        return results

    @staticmethod
    def _validate_leica_decode_request(selector, contract, inspection):
        if not isinstance(contract, VolumeDecodeContract):
            raise TypeError("Leica LIF decoding requires a typed decode contract.")
        if contract.container_format != "leica-lif":
            raise ValueError(
                "Leica LIF decode contract declares a different container format."
            )
        if inspection.container_format != contract.container_format:
            raise ValueError(
                "Leica LIF source container does not match the persisted decode contract."
            )
        selected = _validate_microscopy_series_request(
            selector, contract, inspection, source_name="Leica LIF"
        )
        if tuple(selector.get("axes") or ()) != contract.source_axis_order:
            raise ValueError(
                "Leica LIF selector axes do not match the probe-owned decode contract."
            )
        semantic_by_label = {
            "C": "channel",
            "S": "component",
            "T": "time",
            "X": "space-x",
            "Y": "space-y",
            "Z": "space-z",
        }
        actual_semantics = tuple(
            semantic_by_label.get(label, "unknown")
            for label in contract.source_axis_order
        )
        if actual_semantics != contract.source_axis_semantics:
            raise ValueError(
                "Leica LIF resolved axes do not match the persisted axis semantics."
            )
        channel = selector.get("channel")
        if contract.channel_axis is None:
            if channel is not None:
                raise ValueError(
                    "Leica LIF selector declares a channel for an image without a channel axis."
                )
            return selected, None
        channel_count = contract.source_axis_sizes[contract.channel_axis]
        if isinstance(channel, (bool, np.bool_)):
            raise ValueError("Leica LIF channel selector must be an integer.")
        try:
            channel_index = int(channel)
        except (TypeError, ValueError) as exc:
            raise ValueError("Leica LIF channel selector must be an integer.") from exc
        if channel_index != channel or not 0 <= channel_index < channel_count:
            raise ValueError(
                f"Leica LIF channel {channel!r} is outside 0..{channel_count - 1}."
            )
        return selected, channel_index

    def _decode_leica_many(self, path, selectors, decode_contracts=None):
        selectors = list(selectors)
        contracts = (
            [None] * len(selectors)
            if decode_contracts is None
            else list(decode_contracts)
        )
        if len(contracts) != len(selectors):
            raise ValueError(
                "Leica LIF decode contracts must correspond one-to-one with selectors."
            )
        inspection = inspect_leica_lif_source(
            path, cancel_check=self.cancel_check
        )
        results = [None] * len(selectors)
        groups = {}
        for result_index, (selector, contract) in enumerate(
            zip(selectors, contracts)
        ):
            try:
                selected, channel = self._validate_leica_decode_request(
                    selector, contract, inspection
                )
                groups.setdefault(selected.index, []).append(
                    (result_index, selected, channel, contract)
                )
            except Exception as exc:
                results[result_index] = exc

        for series_index, requests in groups.items():
            if self.cancel_check():
                raise InterruptedError("Volume decoding was cancelled.")
            selected = requests[0][1]
            decoded_channels = decode_leica_lif_series_channels(
                path,
                series_index,
                [request[2] for request in requests],
                expected_series_identity=selected.identity,
                expected_axes=selected.axes,
                expected_shape=selected.shape,
                cancel_check=self.cancel_check,
            )
            if len(decoded_channels) != len(requests):
                raise RuntimeError(
                    "Leica LIF reader returned the wrong number of channel results."
                )
            for request, decoded in zip(requests, decoded_channels):
                result_index, selected, channel, contract = request
                if isinstance(decoded, Exception):
                    results[result_index] = decoded
                    continue
                try:
                    if not isinstance(decoded, LeicaChannelPixels):
                        raise TypeError(
                            "Leica LIF reader returned an invalid channel payload."
                        )
                    if (
                        decoded.source_axes != selected.axes
                        or decoded.source_shape != selected.shape
                    ):
                        raise ValueError(
                            "Decoded Leica LIF axes or dimensions changed after inspection."
                        )
                    metadata_index = 0 if channel is None else int(channel)
                    expected_dtype = selected.channel_scalar_dtypes[metadata_index]
                    expected_bits = selected.channel_scalar_bit_depths[metadata_index]
                    if decoded.scalar_dtype != expected_dtype:
                        raise ValueError(
                            f"Decoded Leica LIF scalar dtype {decoded.scalar_dtype!r} "
                            f"does not match probed dtype {expected_dtype!r}."
                        )
                    if decoded.scalar_bit_depth != expected_bits:
                        raise ValueError(
                            "Decoded Leica LIF significant bit depth does not match inspection."
                        )
                    normalized = normalize_volume_to_tzyx(
                        decoded.array,
                        contract.normalization_axes,
                        channel=0 if contract.channel_axis is not None else None,
                    )
                    payload = self._payload(
                        normalized,
                        selected.time_interval,
                        selected.time_units,
                        source_axis_order=contract.source_axis_order,
                        source_axis_semantics=contract.source_axis_semantics,
                    )
                    expected_dimensions = tuple(
                        contract.source_axis_sizes[
                            contract.source_axis_semantics.index(semantic)
                        ]
                        if semantic in contract.source_axis_semantics
                        else 1
                        for semantic in ("space-x", "space-y", "space-z")
                    )
                    descriptor = dict(payload.source_descriptor)
                    if tuple(descriptor["decoded_dimensions"]) != expected_dimensions:
                        raise ValueError(
                            "Decoded Leica LIF dimensions do not match the probe contract."
                        )
                    if int(descriptor["decoded_time_point_count"]) != selected.time_count:
                        raise ValueError(
                            "Decoded Leica LIF time count does not match the probe contract."
                        )
                    descriptor["source_scalar_dtype"] = expected_dtype
                    descriptor["source_scalar_bit_depth"] = expected_bits
                    results[result_index] = DecodedVolumePayload(
                        frame_zyx=payload.frame_zyx,
                        scalar_range=payload.scalar_range,
                        time_info=payload.time_info,
                        source_descriptor=descriptor,
                    )
                except Exception as exc:
                    results[result_index] = exc
        return results

    @staticmethod
    def _validate_nikon_decode_request(selector, contract):
        if not isinstance(contract, VolumeDecodeContract):
            raise TypeError("Nikon ND2 decoding requires a typed decode contract.")
        if contract.container_format != "nikon-nd2":
            raise ValueError("Nikon ND2 decode contract declares a different container.")
        if selector != contract.selector or not isinstance(selector, dict):
            raise ValueError("Nikon ND2 selector does not match its decode contract.")
        if contract.series_index is None or not contract.series_identity:
            raise ValueError("Nikon ND2 position identity is missing from its contract.")
        if (
            selector.get("series_index") != contract.series_index
            or selector.get("series_identity") != contract.series_identity
        ):
            raise ValueError("Nikon ND2 position identity does not match its contract.")
        source_subselection = selector.get("source_subselection")
        if not isinstance(source_subselection, dict) or set(source_subselection) - {"P"}:
            raise ValueError("Nikon ND2 position selection is missing or invalid.")
        position = source_subselection.get("P")
        if position is not None and (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position != contract.series_index
        ):
            raise ValueError("Nikon ND2 P selection does not match its position.")
        if tuple(selector.get("axes") or ()) != contract.source_axis_order:
            raise ValueError("Nikon ND2 selector axes do not match its source contract.")
        semantics = tuple(
            {"X": "space-x", "Y": "space-y", "Z": "space-z",
             "T": "time", "C": "channel", "S": "component"}.get(axis, "unknown")
            for axis in contract.source_axis_order
        )
        if semantics != contract.source_axis_semantics or "P" in contract.source_axis_order:
            raise ValueError("Nikon ND2 axis semantics do not match its source contract.")
        if any(
            semantic in {"component", "unknown"} and size > 1
            for semantic, size in zip(semantics, contract.source_axis_sizes)
        ):
            raise ValueError("Nikon ND2 contains an unsupported component or custom axis.")
        channel = selector.get("channel")
        channel_count = (
            contract.source_axis_sizes[contract.channel_axis]
            if contract.channel_axis is not None else 1
        )
        if channel is None and channel_count == 1:
            return None
        if (
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not 0 <= channel < channel_count
        ):
            raise ValueError("Nikon ND2 channel selector is outside the source channels.")
        return channel

    def _decode_nikon_many(self, path, selectors, decode_contracts=None):
        selectors = list(selectors)
        contracts = list(decode_contracts or ())
        if len(contracts) != len(selectors):
            raise ValueError("Nikon ND2 requires one typed contract per channel.")
        results = [None] * len(selectors)
        groups = {}
        for result_index, (selector, contract) in enumerate(zip(selectors, contracts)):
            try:
                channel = self._validate_nikon_decode_request(selector, contract)
                groups.setdefault(contract.series_index, []).append(
                    (result_index, channel, selector, contract)
                )
            except Exception as exc:
                results[result_index] = exc

        for position_index, requests in groups.items():
            if self.cancel_check():
                raise InterruptedError("Nikon ND2 decoding was cancelled.")
            first_contract = requests[0][3]
            if any(
                contract.source_axis_order != first_contract.source_axis_order
                or contract.source_axis_sizes != first_contract.source_axis_sizes
                or contract.series_identity != first_contract.series_identity
                or selector.get("source_subselection")
                != requests[0][2].get("source_subselection")
                for _, _, selector, contract in requests
            ):
                error = ValueError("Nikon ND2 channels disagree on their source position contract.")
                for result_index, *_ in requests:
                    results[result_index] = error
                continue
            allocation_bytes = _known_shape_bytes(
                first_contract.source_axis_sizes,
                requests[0][2].get("scalar_dtype"),
            )
            if allocation_bytes is not None:
                self._check_resources(memory_bytes=allocation_bytes)
            try:
                decoded_channels = decode_nikon_nd2_series_channels(
                    path,
                    position_index,
                    [channel for _, channel, _, _ in requests],
                    expected_series_identity=first_contract.series_identity,
                    expected_axes=first_contract.source_axis_order,
                    expected_shape=first_contract.source_axis_sizes,
                    expected_subselection=requests[0][2]["source_subselection"],
                    cancel_check=self.cancel_check,
                )
                if len(decoded_channels) != len(requests):
                    raise RuntimeError("Nikon ND2 returned the wrong channel count.")
                pending = []
                for (_, channel, selector, contract), decoded in zip(
                    requests, decoded_channels
                ):
                    if not isinstance(decoded, NikonChannelPixels):
                        raise TypeError("Nikon ND2 returned an invalid channel payload.")
                    if (
                        decoded.source_axes != contract.source_axis_order
                        or decoded.source_shape != contract.source_axis_sizes
                        or decoded.channel_index != channel
                    ):
                        raise ValueError("Nikon ND2 source axes or channel changed during decode.")
                    if (
                        decoded.scalar_dtype != selector.get("scalar_dtype")
                        or decoded.scalar_bit_depth != selector.get("significant_bit_depth")
                    ):
                        raise ValueError("Nikon ND2 scalar type or bit depth changed since import.")
                    normalized = normalize_volume_to_tzyx(
                        decoded.array,
                        contract.normalization_axes,
                        channel=0 if contract.channel_axis is not None else None,
                    )
                    expected_dimensions = tuple(
                        contract.source_axis_sizes[
                            contract.source_axis_semantics.index(semantic)
                        ] if semantic in contract.source_axis_semantics else 1
                        for semantic in ("space-x", "space-y", "space-z")
                    )
                    if (
                        tuple(normalized.shape[1:][::-1]) != expected_dimensions
                        or normalized.shape[0] != selector.get("time_count")
                        or normalized.shape[0] != decoded.time_count
                    ):
                        raise ValueError("Nikon ND2 decoded dimensions or time count changed.")
                    payload = self._payload(
                        normalized,
                        decoded.time_interval,
                        decoded.time_units,
                        source_axis_order=contract.source_axis_order,
                        source_axis_semantics=contract.source_axis_semantics,
                    )
                    descriptor = dict(payload.source_descriptor)
                    descriptor["source_scalar_dtype"] = decoded.scalar_dtype
                    descriptor["source_scalar_bit_depth"] = decoded.scalar_bit_depth
                    pending.append(DecodedVolumePayload(
                        frame_zyx=payload.frame_zyx,
                        scalar_range=payload.scalar_range,
                        time_info=payload.time_info,
                        source_descriptor=descriptor,
                    ))
                if self.cancel_check():
                    raise InterruptedError("Nikon ND2 decoding was cancelled.")
                for request, payload in zip(requests, pending):
                    results[request[0]] = payload
            except InterruptedError:
                raise
            except Exception as exc:
                for result_index, *_ in requests:
                    results[result_index] = exc
        return results

    @staticmethod
    def _validate_czi_decode_request(selector, contract):
        if not isinstance(contract, VolumeDecodeContract):
            raise TypeError("ZEISS CZI decoding requires a typed decode contract.")
        if contract.container_format != "zeiss-czi":
            raise ValueError("ZEISS CZI decode contract declares a different container.")
        if selector != contract.selector or not isinstance(selector, dict):
            raise ValueError("ZEISS CZI selector does not match its decode contract.")
        if contract.series_index is None or not contract.series_identity:
            raise ValueError("ZEISS CZI scene identity is missing from its contract.")
        if (
            selector.get("series_index") != contract.series_index
            or selector.get("series_identity") != contract.series_identity
            or tuple(selector.get("axes") or ()) != contract.source_axis_order
            or contract.source_axis_order != ("T", "Z", "C", "Y", "X")
            or contract.source_axis_semantics != (
                "time", "space-z", "channel", "space-y", "space-x"
            )
        ):
            raise ValueError("ZEISS CZI scene or axes do not match their source contract.")
        selection = selector.get("source_selection")
        ordinary_selection_keys = {
            "scene_index", "roi", "t_start", "z_start", "channel_indices", "fixed_indices"
        }
        illumination_selection_keys = ordinary_selection_keys | {"channel_coordinates"}
        variant_selection_keys = illumination_selection_keys | {"view_index"}
        if not isinstance(selection, dict) or frozenset(selection) not in {
            frozenset(ordinary_selection_keys), frozenset(illumination_selection_keys),
            frozenset(variant_selection_keys),
        }:
            raise ValueError("ZEISS CZI source-coordinate selection is incomplete.")
        scene = selection["scene_index"]
        if scene is not None and (isinstance(scene, bool) or not isinstance(scene, int)):
            raise ValueError("ZEISS CZI scene tag is invalid.")
        subselection = selector.get("source_subselection")
        expected_subselection = {"S": scene} if scene is not None else {}
        if "view_index" in selection:
            view = selection["view_index"]
            if isinstance(view, bool) or not isinstance(view, int) or view < 0:
                raise ValueError("ZEISS CZI view selector is invalid.")
            expected_subselection["V"] = view
        if subselection != expected_subselection:
            raise ValueError("ZEISS CZI scene tag disagrees with its source selection.")
        roi = selection["roi"]
        if (
            not isinstance(roi, (list, tuple)) or len(roi) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in roi)
            or tuple(roi[2:]) != (
                contract.source_axis_sizes[4], contract.source_axis_sizes[3]
            )
        ):
            raise ValueError("ZEISS CZI full-resolution ROI is invalid.")
        for axis, position in (("T", 0), ("Z", 1)):
            value = selection[f"{axis.lower()}_start"]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"ZEISS CZI {axis} source start is invalid.")
        source_channels = selection["channel_indices"]
        if (
            not isinstance(source_channels, (list, tuple))
            or len(source_channels) != contract.source_axis_sizes[2]
            or any(isinstance(value, bool) or not isinstance(value, int) for value in source_channels)
            or not isinstance(selection["fixed_indices"], dict)
        ):
            raise ValueError("ZEISS CZI source channel or fixed indices are invalid.")
        coordinates = selection.get("channel_coordinates")
        if coordinates is None:
            if len(set(source_channels)) != len(source_channels):
                raise ValueError("ZEISS CZI source channel or fixed indices are invalid.")
        elif (
            not isinstance(coordinates, (list, tuple))
            or len(coordinates) != len(source_channels)
            or any(
                not isinstance(item, dict)
                or set(item) not in ({"C"}, {"C", "I"})
                or any(isinstance(value, bool) or not isinstance(value, int) for value in item.values())
                for item in coordinates
            )
            or [item["C"] for item in coordinates] != list(source_channels)
        ):
            raise ValueError("ZEISS CZI native channel coordinates are invalid.")
        channel = selector.get("channel")
        if channel is None and len(source_channels) == 1:
            channel = 0
        if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel < len(source_channels):
            raise ValueError("ZEISS CZI channel selector is outside the source channels.")
        if selector.get("channel_source_index") != source_channels[channel]:
            raise ValueError("ZEISS CZI source channel index changed.")
        if coordinates is not None and selector.get("channel_source_coordinates") != coordinates[channel]:
            raise ValueError("ZEISS CZI native channel coordinates changed.")
        return channel

    def _decode_czi_many(self, path, selectors, decode_contracts=None):
        selectors = list(selectors)
        contracts = list(decode_contracts or ())
        if len(contracts) != len(selectors):
            raise ValueError("ZEISS CZI requires one typed contract per channel.")
        results = [None] * len(selectors)
        groups = {}
        for result_index, (selector, contract) in enumerate(zip(selectors, contracts)):
            try:
                channel = self._validate_czi_decode_request(selector, contract)
                groups.setdefault(contract.series_index, []).append(
                    (result_index, channel, selector, contract)
                )
            except Exception as exc:
                results[result_index] = exc

        for series_index, requests in groups.items():
            if self.cancel_check():
                raise InterruptedError("ZEISS CZI decoding was cancelled.")
            first_selector, first_contract = requests[0][2:]
            if any(
                contract.source_axis_sizes != first_contract.source_axis_sizes
                or contract.series_identity != first_contract.series_identity
                or selector.get("source_selection") != first_selector["source_selection"]
                for _, _, selector, contract in requests
            ):
                error = ValueError("ZEISS CZI channels disagree on their source scene contract.")
                for result_index, *_ in requests:
                    results[result_index] = error
                continue
            allocation_bytes = _known_shape_bytes(
                first_contract.source_axis_sizes,
                requests[0][2].get("scalar_dtype"),
            )
            if allocation_bytes is not None:
                self._check_resources(memory_bytes=allocation_bytes)
            try:
                decoded_channels = decode_zeiss_czi_series_channels(
                    path, series_index,
                    [None if selector["channel"] is None else channel
                     for _, channel, selector, _ in requests],
                    expected_series_identity=first_contract.series_identity,
                    expected_axes=first_contract.source_axis_order,
                    expected_shape=first_contract.source_axis_sizes,
                    expected_selection=first_selector["source_selection"],
                    cancel_check=self.cancel_check,
                )
                if len(decoded_channels) != len(requests):
                    raise RuntimeError("ZEISS CZI returned the wrong channel count.")
                pending = []
                for (_, channel, selector, contract), decoded in zip(requests, decoded_channels):
                    if not isinstance(decoded, CziChannelPixels):
                        raise TypeError("ZEISS CZI returned an invalid channel payload.")
                    if (
                        decoded.source_axes != contract.source_axis_order
                        or decoded.source_shape != contract.source_axis_sizes
                        or decoded.channel_index != selector["channel"]
                        or decoded.scalar_dtype != selector.get("scalar_dtype")
                        or decoded.scalar_bit_depth != selector.get("significant_bit_depth")
                        or decoded.array.shape != (
                            contract.source_axis_sizes[0], contract.source_axis_sizes[1],
                            contract.source_axis_sizes[3], contract.source_axis_sizes[4],
                        )
                        or decoded.time_count != selector.get("time_count")
                    ):
                        raise ValueError("ZEISS CZI source scalar contract changed during decode.")
                    payload = self._payload(
                        decoded.array, decoded.time_interval, decoded.time_units,
                        source_axis_order=contract.source_axis_order,
                        source_axis_semantics=contract.source_axis_semantics,
                    )
                    descriptor = dict(payload.source_descriptor)
                    descriptor["source_scalar_dtype"] = decoded.scalar_dtype
                    descriptor["source_scalar_bit_depth"] = decoded.scalar_bit_depth
                    pending.append(DecodedVolumePayload(
                        frame_zyx=payload.frame_zyx, scalar_range=payload.scalar_range,
                        time_info=payload.time_info, source_descriptor=descriptor,
                    ))
                if self.cancel_check():
                    raise InterruptedError("ZEISS CZI decoding was cancelled.")
                for request, payload in zip(requests, pending):
                    results[request[0]] = payload
            except InterruptedError:
                raise
            except Exception as exc:
                for result_index, *_ in requests:
                    results[result_index] = exc
        return results

    def decode_iter_many(
        self,
        path,
        selectors: Iterable[Any],
        *,
        decode_contracts: Iterable[VolumeDecodeContract | None] | None = None,
        memory_reserved=False,
    ):
        """Yield one result per selector while keeping one source open.

        TIFF, NRRD and H5J produce channels incrementally. Native microscopy
        readers currently return a complete source-group result; that one
        unavoidable group owner is drained as each result is consumed.
        """
        selectors = list(selectors)
        mode = self.reader_mode(path)
        previous_memory_reserved = self._memory_reserved
        self._memory_reserved = bool(memory_reserved)
        try:
            if mode == "tiff":
                yield from self._decode_tiff_many(path, selectors)
                return
            if mode == "nrrd":
                yield from self._decode_nrrd_many(path, selectors, decode_contracts)
                return
            if mode == "h5j":
                yield from self._decode_h5j_many(path, selectors)
                return
            if mode == "nifti":
                if any(selector not in (None, "") for selector in selectors):
                    raise ValueError("NIfTI channel selectors are not supported.")
                if self.cancel_check():
                    raise InterruptedError("Volume decoding was cancelled.")
                if not selectors:
                    return
                payload = self._decode_nifti(path)
                # Immutable source pixels may back distinct scene identities.
                payload.frame_zyx.setflags(write=False)
                if payload.time_info.get("data") is not None:
                    payload.time_info["data"].setflags(write=False)
                for _selector in selectors:
                    yield payload
                return
            if mode in {"olympus", "leica", "nikon", "zeiss-czi"}:
                if mode == "olympus":
                    values = self._decode_olympus_many(path, selectors, decode_contracts)
                elif mode == "leica":
                    values = self._decode_leica_many(path, selectors, decode_contracts)
                elif mode == "nikon":
                    values = self._decode_nikon_many(path, selectors, decode_contracts)
                else:
                    values = self._decode_czi_many(path, selectors, decode_contracts)
                # These reader APIs allocate a complete source group. Keep it
                # under the single source-group reservation and release each
                # result reference as soon as the consumer advances.
                values.reverse()
                while values:
                    yield values.pop()
                return
            raise RuntimeError(f"Unsupported volume type: {path}")
        finally:
            self._memory_reserved = previous_memory_reserved

    def decode_many(
        self,
        path,
        selectors: Iterable[Any],
        *,
        decode_contracts: Iterable[VolumeDecodeContract | None] | None = None,
    ):
        """Decode all selected payloads for callers retaining the complete result."""
        return list(self.decode_iter_many(
            path, selectors, decode_contracts=decode_contracts
        ))

    def decode(self, path, selector=None, *, decode_contract=None):
        contracts = None if decode_contract is None else [decode_contract]
        result = self.decode_many(
            path, [selector], decode_contracts=contracts
        )[0]
        if isinstance(result, Exception):
            raise result
        return result


__all__ = [
    "DecodedVolumePayload",
    "VolumeDecodeContract",
    "VolumePayloadDecoder",
    "crop_h5j_codec_padding",
    "h5j_expected_dimensions",
    "normalize_volume_to_tzyx",
    "nrrd_time_metadata",
    "require_nibabel",
    "require_nrrd",
    "require_tifffile",
    "safe_scalar_range",
]
