"""GUI-independent decoding of supported scientific volume payloads."""

from __future__ import annotations

from madi3d_app.io_utils import check_resources

import glob
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .leica_lif import (
    decode_leica_lif_series_channels,
    inspect_leica_lif_source,
    LeicaChannelPixels,
)
from .olympus import (
    OlympusChannelPixels,
    decode_olympus_series_channels,
    inspect_olympus_source,
)
from .provenance import decoded_payload_descriptor
from .probe import tiff_series_identity
from .source_formats import volume_reader_mode


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


def prepare_decoded_array(array):
    """Return a native-endian scalar array using the established load dtypes."""
    array = np.asanyarray(array)
    if np.iscomplexobj(array):
        raise TypeError("Complex-valued volumes are not supported.")
    if not array.dtype.isnative:
        array = array.byteswap().view(array.dtype.newbyteorder("="))
    dtype = array.dtype
    if dtype == np.bool_:
        array = array.astype(np.uint8, copy=False)
    elif dtype == np.int8:
        array = array.astype(np.int16)
    elif dtype in (np.uint8, np.int16, np.uint16, np.float32, np.float64):
        pass
    elif np.issubdtype(dtype, np.integer) or dtype == np.float16:
        array = array.astype(np.float32)
    elif np.issubdtype(dtype, np.floating):
        array = array.astype(np.float32)
    else:
        raise TypeError(f"Unsupported volume dtype: {dtype}")
    return array


def safe_scalar_range(array):
    try:
        low = float(np.nanmin(array))
        high = float(np.nanmax(array))
        if math.isfinite(low) and math.isfinite(high):
            return low, high
    except Exception:
        pass
    return None


def normalize_volume_to_tzyx(data, axes, channel=None):
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
    values = []
    for key in ("frames", "height", "width"):
        try:
            raw = np.asarray(channel_group.attrs.get(key)).reshape(-1)
            value = int(raw[0]) if raw.size else 0
        except Exception:
            value = 0
        values.append(value)
    if all(value > 0 for value in values):
        return tuple(values)
    try:
        image_size = tuple(
            int(value)
            for value in np.asarray(handle.attrs.get("image_size")).reshape(-1)[:3]
        )
    except Exception:
        image_size = ()
    if len(image_size) == 3 and all(value > 0 for value in image_size):
        width, height, frames = image_size
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


def run_h5j_ffmpeg(executable, args, *, cancel_check=None):
    from madi3d_app.integrations.ffmpeg.backend import (
        FFmpegProcessCancelled,
        FFmpegProcessTimeout,
        run_ffmpeg_process,
    )

    try:
        return run_ffmpeg_process(executable, args, cancel_check=cancel_check)
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
        return cls(
            container_format=probe.container_format,
            series_identity=probe.series_identity,
            series_index=probe.series_index,
            source_axis_sizes=tuple(axis.size for axis in probe.axis_semantics),
            source_axis_order=probe.source_axis_order,
            source_axis_semantics=probe.source_axis_semantics,
            selector=selector,
            resolution_provenance=probe.resolution_provenance,
        )

    @classmethod
    def from_backing_source(
        cls, backing_source, *, selector=None
    ) -> "VolumeDecodeContract":
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
        )


@dataclass(frozen=True)
class DecodedVolumePayload:
    frame_zyx: np.ndarray
    scalar_range: tuple[float, float]
    time_info: dict[str, Any] | None
    source_descriptor: dict[str, Any] = field(default_factory=dict)


class VolumePayloadDecoder:
    """Decode arrays and plain metadata without creating Qt or VTK objects."""

    def __init__(
        self,
        *,
        ffmpeg_executable=None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        self.ffmpeg_executable = (
            os.fspath(ffmpeg_executable) if ffmpeg_executable else None
        )
        self.cancel_check = cancel_check or (lambda: False)

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
    ):
        source_array = np.asanyarray(tzyx)
        tzyx = prepare_decoded_array(tzyx)
        scalar_range = safe_scalar_range(tzyx) or (0.0, 1.0)
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
    def _tiff_time_metadata(tiff, axes):
        imagej = getattr(tiff, "imagej_metadata", None) or {}
        time_spacing = 1.0
        time_units = "frame"
        if "T" in axes:
            if bool(getattr(tiff, "is_lsm", False)):
                try:
                    candidate = float(
                        dict(getattr(tiff, "lsm_metadata", None) or {}).get(
                            "TimeIntervall"
                        )
                    )
                    if math.isfinite(candidate) and candidate > 0:
                        return candidate, "s"
                except (TypeError, ValueError):
                    pass
            for key in ("finterval", "timeincrement", "TimeIncrement"):
                try:
                    candidate = float(imagej.get(key))
                    if math.isfinite(candidate) and candidate > 0:
                        time_spacing = candidate
                        break
                except Exception:
                    pass
            time_units = str(
                imagej.get("tunit") or imagej.get("timeunit") or "frame"
            )
        return time_spacing, time_units

    @staticmethod
    def _select_tiff_source_subspace(data, axes, source_subselection):
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
        with module.TiffFile(path) as tiff:
            cache = {}
            results = []
            for (
                series_index,
                series_identity,
                pyramid_level,
                source_subselection,
                channel,
                explicit_axes,
            ) in parsed:
                if self.cancel_check():
                    raise InterruptedError("Volume decoding was cancelled.")
                try:
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
                        series_index,
                        pyramid_level,
                        tuple(sorted(source_subselection.items())),
                        str(explicit_axes or ""),
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
                        source_dtype = np.dtype(series.dtype)
                        runtime_dtype = prepare_decoded_array(
                            np.empty(0, dtype=source_dtype)
                        ).dtype
                        # Cover source pixels, normalized channels and a temporary
                        # copy during VTK import. Account for scalar promotion
                        # without assuming every TIFF sample occupies 16 bytes.
                        check_resources(
                            memory_bytes=math.prod(int(v) for v in series.shape)
                            * max(source_dtype.itemsize, runtime_dtype.itemsize) * 3
                        )
                        data = series.asarray()
                        if self.cancel_check():
                            raise InterruptedError("Volume decoding was cancelled.")
                        data, source_axes = self._select_tiff_source_subspace(
                            data,
                            str(series.axes).upper(),
                            source_subselection,
                        )
                        axes, _shape = self._validated_tiff_axes(
                            source_axes,
                            data.shape,
                            explicit_axes,
                        )
                        time_spacing, time_units = self._tiff_time_metadata(
                            tiff, axes
                        )
                        cache[cache_key] = (
                            data,
                            axes,
                            time_spacing,
                            time_units,
                        )
                    data, axes, time_spacing, time_units = cache[cache_key]
                    results.append(
                        self._payload(
                            normalize_volume_to_tzyx(data, axes, channel=channel),
                            time_spacing,
                            time_units,
                            source_axis_order=axes,
                        )
                    )
                except Exception as exc:
                    results.append(exc)
        return results

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
        sizes = [math.prod(contract.source_axis_sizes) for contract in decode_contracts
                 if isinstance(contract, VolumeDecodeContract)]
        if sizes:
            check_resources(memory_bytes=max(sizes) * 16)
        array, header = require_nrrd().read(path)
        results = []
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
                results.append(
                    self._payload(
                        normalize_volume_to_tzyx(array, axes, channel=selector),
                        time_spacing,
                        time_units,
                        source_axis_order=contract.source_axis_order,
                        source_axis_semantics=contract.source_axis_semantics,
                    )
                )
            except Exception as exc:
                results.append(exc)
        return results

    def _decode_nifti(self, path):
        nibabel = require_nibabel()
        image = nibabel.load(path, mmap=True)
        shape = getattr(image, "shape", None) or getattr(image.dataobj, "shape", ())
        if shape:
            check_resources(memory_bytes=math.prod(int(v) for v in shape) * 16)
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
            str(time_units or "frame"),
            source_axis_order=axes,
        )

    def _decode_h5j_stream(self, raw, expected_dimensions):
        if not self.ffmpeg_executable:
            raise RuntimeError(
                "H5J loading started without a prepared FFmpeg executable."
            )
        from madi3d_app.io_utils import check_resources, check_cancelled
        if expected_dimensions is not None:
            z, y, x = (int(v) for v in expected_dimensions)
            # Include codec padding, TIFF overhead and the encoded input.
            padded = max(64, ((x + 31) // 32) * 32) * max(64, ((y + 31) // 32) * 32) * max(1, z)
            check_resources(disk_bytes=padded * 8 + np.asarray(raw).nbytes)
        temporary = tempfile.mkdtemp(prefix="h5j_")
        try:
            encoded = os.path.join(temporary, "data.h265")
            with open(encoded, "wb") as stream:
                data = memoryview(np.ascontiguousarray(raw)).cast('B')
                for offset in range(0, len(data), 1024 * 1024):
                    check_cancelled(self.cancel_check)
                    stream.write(data[offset:offset + 1024 * 1024])
            pattern = os.path.join(temporary, "frame_%06d.tif")
            process = run_h5j_ffmpeg(
                self.ffmpeg_executable,
                [
                    "-y", "-v", "error", "-i", encoded,
                    "-compression_algo", "raw", pattern,
                ],
                cancel_check=self.cancel_check,
            )
            if process.returncode != 0:
                stderr = (process.stderr or b"").decode("utf-8", "replace")[-4000:]
                raise RuntimeError(f"FFmpeg H5J decoding failed:\n{stderr}")
            frames = sorted(glob.glob(os.path.join(temporary, "frame_*.tif")))
            if not frames:
                raise RuntimeError("No frames decoded from H5J channel")
            first = require_tifffile().imread(frames[0])
            shape = (len(frames), *first.shape)
            check_resources(memory_bytes=int(np.prod(shape)) * first.dtype.itemsize)
            array = np.empty(shape, dtype=first.dtype)
            array[0] = first
            os.remove(frames[0])
            for index, frame_path in enumerate(frames[1:], start=1):
                check_cancelled(self.cancel_check)
                frame = require_tifffile().imread(frame_path)
                if frame.shape != first.shape or frame.dtype != first.dtype:
                    raise ValueError("H5J decoded frames have inconsistent shape or dtype.")
                array[index] = frame
                os.remove(frame_path)
            return crop_h5j_codec_padding(array, expected_dimensions)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def _decode_h5j_many(self, path, selectors):
        import h5py

        results = []
        with h5py.File(path, "r") as handle:
            group = handle["/Channels"]
            expected = h5j_expected_dimensions(handle, group)
            raw_order = group.attrs.get(
                "channel_order", handle.attrs.get("channel_order")
            )
            keys = []
            if raw_order is not None:
                if isinstance(raw_order, (bytes, str)):
                    try:
                        raw_order = json.loads(
                            raw_order.decode("utf-8", "replace")
                            if isinstance(raw_order, bytes)
                            else raw_order
                        )
                    except Exception:
                        raw_order = [raw_order]
                keys = [
                    value.decode("utf-8", "replace")
                    if isinstance(value, bytes)
                    else str(value)
                    for value in np.asarray(raw_order).reshape(-1)
                ]
            available = {str(key) for key in group.keys()}
            if keys and (
                len(set(keys)) != len(keys) or set(keys) != available
            ):
                raise ValueError(
                    "H5J channel-order metadata does not match the /Channels datasets."
                )
            for selector in selectors:
                if self.cancel_check():
                    raise InterruptedError("Volume decoding was cancelled.")
                try:
                    if isinstance(selector, str):
                        key = selector
                    elif not keys:
                        raise ValueError(
                            "H5J channel order is unresolved; select channels by explicit name."
                        )
                    else:
                        key = keys[0 if selector is None else int(selector)]
                    if key not in available:
                        raise ValueError(f"H5J channel {key!r} is unavailable.")
                    check_resources(memory_bytes=int(group[key].size) * int(group[key].dtype.itemsize)
                                    + (math.prod(expected) * 8 if expected is not None else 0))
                    raw = group[key][:]
                    array = self._decode_h5j_stream(raw, expected)
                    del raw
                    payload = self._payload(
                        array[np.newaxis, ...],
                        source_axis_order=("Channels", "Z", "Y", "X"),
                        source_axis_semantics=(
                            "channel", "space-z", "space-y", "space-x"
                        ),
                    )
                    results.append(
                        DecodedVolumePayload(
                            frame_zyx=payload.frame_zyx,
                            scalar_range=payload.scalar_range,
                            time_info=None,
                            source_descriptor=payload.source_descriptor,
                        )
                    )
                except InterruptedError:
                    raise
                except Exception as exc:
                    results.append(exc)
        return results

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
        if selector != contract.selector:
            raise ValueError(
                "Olympus selector does not match the probe-owned decode contract."
            )
        if not isinstance(selector, dict):
            raise TypeError("Olympus selector must be a structured series/channel record.")
        try:
            selector_series_index = int(selector["series_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Olympus selector is missing its series index.") from exc
        if (
            contract.series_index is None
            or selector_series_index != contract.series_index
        ):
            raise ValueError(
                "Olympus selector series does not match the persisted decode contract."
            )
        selected = next(
            (
                item
                for item in inspection.series
                if item.index == contract.series_index
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                f"Persisted Olympus series index {contract.series_index} is unavailable."
            )
        selector_identity = str(selector.get("series_identity") or "")
        if (
            not contract.series_identity
            or selector_identity != contract.series_identity
            or selected.identity != contract.series_identity
        ):
            raise ValueError(
                "Persisted Olympus series identity does not match the selected source series."
            )
        if tuple(selected.shape) != contract.source_axis_sizes:
            raise ValueError(
                f"Olympus source shape {selected.shape!r} does not match the "
                f"probe-owned axis contract {contract.source_axis_sizes!r}."
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
            pixels = decode_olympus_series_channels(
                path,
                series_index,
                [request[2] for request in requests],
                cancel_check=self.cancel_check,
            )
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
        if selector != contract.selector:
            raise ValueError(
                "Leica LIF selector does not match the probe-owned decode contract."
            )
        if not isinstance(selector, dict):
            raise TypeError(
                "Leica LIF selector must be a structured image/channel record."
            )
        try:
            selector_index = int(selector["series_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Leica LIF selector is missing its series index."
            ) from exc
        if contract.series_index is None or selector_index != contract.series_index:
            raise ValueError(
                "Leica LIF selector series does not match the persisted decode contract."
            )
        selected = next(
            (item for item in inspection.series if item.index == contract.series_index),
            None,
        )
        if selected is None:
            raise ValueError(
                f"Persisted Leica LIF series index {contract.series_index} is unavailable."
            )
        selector_identity = str(selector.get("series_identity") or "")
        if (
            not contract.series_identity
            or selector_identity != contract.series_identity
            or selected.identity != contract.series_identity
        ):
            raise ValueError(
                "Persisted Leica LIF series identity does not match the selected source image."
            )
        if tuple(selected.shape) != contract.source_axis_sizes:
            raise ValueError(
                f"Leica LIF source shape {selected.shape!r} does not match the "
                f"probe-owned axis contract {contract.source_axis_sizes!r}."
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

    def decode_many(
        self,
        path,
        selectors: Iterable[Any],
        *,
        decode_contracts: Iterable[VolumeDecodeContract | None] | None = None,
    ):
        selectors = list(selectors)
        mode = self.reader_mode(path)
        if mode == "tiff":
            return self._decode_tiff_many(path, selectors)
        if mode == "nrrd":
            return self._decode_nrrd_many(path, selectors, decode_contracts)
        if mode == "h5j":
            return self._decode_h5j_many(path, selectors)
        if mode == "nifti":
            if any(selector not in (None, "") for selector in selectors):
                raise ValueError("NIfTI channel selectors are not supported.")
            if self.cancel_check():
                raise InterruptedError("Volume decoding was cancelled.")
            if not selectors:
                return []
            payload = self._decode_nifti(path)
            # Immutable source pixels may back distinct scene identities. GUI
            # publication creates independent runtime objects for each request.
            payload.frame_zyx.setflags(write=False)
            if payload.time_info.get("data") is not None:
                payload.time_info["data"].setflags(write=False)
            return [payload for _selector in selectors]
        if mode == "olympus":
            return self._decode_olympus_many(
                path, selectors, decode_contracts
            )
        if mode == "leica":
            return self._decode_leica_many(
                path, selectors, decode_contracts
            )
        raise RuntimeError(f"Unsupported volume type: {path}")

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
    "prepare_decoded_array",
    "require_nibabel",
    "require_nrrd",
    "require_tifffile",
    "safe_scalar_range",
]
