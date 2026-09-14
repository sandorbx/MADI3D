"""Olympus OIF/OIB source inspection and cancellable pixel access.

OIF is a multi-file source bundle. OIB is one compound backing file whose
internal stream identities are retained as evidence. Inspection remains
metadata-only; the decode helper reads selected series/channel members for the
GUI-independent volume decode layer.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
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


OLYMPUS_REQUIRED_MEMBER_MISSING = "required-member-missing"
OLYMPUS_MALFORMED_METADATA = "malformed-metadata"
OLYMPUS_UNSUPPORTED_VARIANT = "unsupported-acquisition-variant"
OLYMPUS_UNREADABLE_SOURCE = "unreadable-source"

_MEMBER_EXTENSIONS = {
    ".bmp",
    ".lut",
    ".oif",
    ".pty",
    ".roi",
    ".tif",
    ".tiff",
    ".txt",
}
_OPTIONAL_SETTINGS_EXTENSIONS = {".lut", ".pty", ".roi", ".txt"}
_PHYSICAL_UNIT_FACTORS_TO_MICRON = {
    "nm": 0.001,
    "nanometer": 0.001,
    "nanometers": 0.001,
    "nanometre": 0.001,
    "nanometres": 0.001,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "micrometer": 1.0,
    "micrometers": 1.0,
    "micrometre": 1.0,
    "micrometres": 1.0,
    "mm": 1000.0,
    "millimeter": 1000.0,
    "millimeters": 1000.0,
    "millimetre": 1000.0,
    "millimetres": 1000.0,
    "cm": 10_000.0,
    "m": 1_000_000.0,
}


class OlympusInspectionError(ValueError):
    """Categorized source-inspection failure suitable for user diagnostics."""

    def __init__(self, category: str, message: str) -> None:
        self.category = str(category)
        self.message = str(message).strip()
        super().__init__(f"Olympus {self.category}: {self.message}")


@dataclass(frozen=True)
class OlympusSeriesInspection:
    index: int
    identity: str
    name: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    scalar_dtype: str
    scalar_bit_depth: Optional[int]
    channel_names: tuple[str, ...]
    channel_metadata: tuple[MicroscopyChannelMetadata, ...]
    channel_order_verified: bool
    spacing: Optional[tuple[float, float, float]]
    space_units: Optional[tuple[str, str, str]]
    time_count: int
    time_interval: float
    time_units: str
    scan_direction: Optional[str]
    acquisition_software_version: Optional[str]
    acquisition_metadata: MicroscopyAcquisitionMetadata
    missing_fields: tuple[str, ...]
    physical_geometry_diagnostics: tuple[str, ...]
    warnings: tuple[str, ...]
    raw_fields: dict[str, Any]
    member_paths: tuple[str, ...]


@dataclass(frozen=True)
class OlympusSourceInspection:
    source_path: str
    container_format: str
    source_metadata: MicroscopySourceMetadata
    series: tuple[OlympusSeriesInspection, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class OlympusChannelPixels:
    """One complete selected channel plus its unchanged source-array contract."""

    channel_index: Optional[int]
    array: np.ndarray
    member_paths: tuple[str, ...]
    source_axes: tuple[str, ...]
    source_shape: tuple[int, ...]
    scalar_dtype: str


@dataclass(frozen=True)
class _MemberEntry:
    path: str
    role: str
    requirement: str
    location_kind: str
    exists: bool
    size_bytes: Optional[int]
    reader_path: Optional[str] = None
    filesystem_path: Optional[str] = None
    warning: Optional[str] = None

    def to_record(self, *, member_id: Optional[str] = None) -> SourceMemberRecord:
        return SourceMemberRecord(
            member_id=member_id or f"{self.location_kind}:{self.path}",
            path=self.path,
            role=self.role,
            size_bytes=self.size_bytes,
            checksum_state="not-computed" if self.exists else "unavailable",
            structured_metadata={
                "requirement": self.requirement,
                "location_kind": self.location_kind,
                "exists": self.exists,
            },
            warnings=(self.warning,) if self.warning else (),
        )


def require_oiffile():
    try:
        import oiffile
    except ImportError as exc:  # pragma: no cover - build dependency in production
        raise RuntimeError(
            "Olympus OIF/OIB inspection requires the 'oiffile' package."
        ) from exc
    return oiffile


def _check_cancel(cancel_check: Optional[Callable[[], bool]]) -> None:
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Olympus source operation was cancelled.")


def _read_olympus_tiff_member(reader, member_path: str):
    """Read one TIFF member without routing through a whole-series materializer."""
    import tifffile

    with reader.open_file(member_path) as handle:
        with tifffile.TiffFile(handle, name=member_path) as tiff:
            if not tiff.series:
                raise ValueError("TIFF member contains no image series.")
            series = tiff.series[0]
            axes = tuple(str(series.axes or "").upper())
            array = np.asanyarray(series.asarray())
    if len(axes) != array.ndim:
        raise ValueError(
            f"TIFF member axes {''.join(axes)!r} do not match shape {array.shape!r}."
        )
    return array, axes


def _flat_sequence_index(index, shape) -> int:
    if not shape:
        if tuple(index) != ():
            raise ValueError("Scalar TIFF sequence contains a non-scalar index.")
        return 0
    return int(np.ravel_multi_index(tuple(index), tuple(shape)))


def decode_olympus_series_channels(
    path: os.PathLike[str] | str,
    series_index: int,
    channel_indices,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    reader_module=None,
):
    """Read complete selected channels with cancellation between TIFF members.

    Results correspond one-to-one with ``channel_indices``. A corrupt member is
    returned as an exception only for the selected channels that depend on it;
    no incomplete array is returned. Structural source failures abort the group.
    """
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix not in {".oif", ".oib"}:
        raise OlympusInspectionError(
            OLYMPUS_UNSUPPORTED_VARIANT,
            f"unsupported Olympus suffix {suffix or '<none>'}",
        )
    requested = tuple(channel_indices)
    normalized = []
    for value in requested:
        if value is None:
            normalized.append(None)
            continue
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                "Olympus channel selector must be a non-negative integer."
            )
        index = int(value)
        if index < 0 or index != value:
            raise ValueError(
                "Olympus channel selector must be a non-negative integer."
            )
        normalized.append(index)
    requested = tuple(normalized)
    if not requested:
        return []

    _check_cancel(cancel_check)
    reader_module = reader_module or require_oiffile()
    try:
        reader = reader_module.OifFile(source)
    except Exception as exc:
        raise OlympusInspectionError(
            OLYMPUS_UNREADABLE_SOURCE, f"cannot open {source.name}: {exc}"
        ) from exc

    try:
        _check_cancel(cancel_check)
        sequences = tuple(reader.series)
        selected_index = int(series_index)
        if not 0 <= selected_index < len(sequences):
            raise OlympusInspectionError(
                OLYMPUS_UNSUPPORTED_VARIANT,
                f"series index {selected_index} is unavailable during decode",
            )
        sequence = sequences[selected_index]
        sequence_axes = tuple(str(getattr(sequence, "axes", "") or "").upper())
        sequence_shape = tuple(int(value) for value in getattr(sequence, "shape", ()))
        files = tuple(getattr(sequence, "_files", ()))
        member_paths = _series_files(sequence)
        indices = tuple(
            tuple(int(value) for value in item) for item in sequence.indices
        )
        if len(sequence_axes) != len(sequence_shape) or any(
            value <= 0 for value in sequence_shape
        ):
            raise OlympusInspectionError(
                OLYMPUS_MALFORMED_METADATA,
                "TIFF sequence axes and positive dimensions are inconsistent",
            )
        expected_file_count = math.prod(sequence_shape)
        if len(files) != expected_file_count or len(indices) != len(files):
            raise OlympusInspectionError(
                OLYMPUS_REQUIRED_MEMBER_MISSING,
                f"series {selected_index} requires {expected_file_count} TIFF members "
                f"but exposes {len(files)} files and {len(indices)} indices",
            )
        flat_indices = tuple(
            _flat_sequence_index(index, sequence_shape) for index in indices
        )
        if len(set(flat_indices)) != expected_file_count:
            raise OlympusInspectionError(
                OLYMPUS_MALFORMED_METADATA,
                f"series {selected_index} TIFF member indices are incomplete or duplicated",
            )

        requested_keys = tuple(dict.fromkeys(requested))
        sequence_channel_axis = (
            sequence_axes.index("C") if "C" in sequence_axes else None
        )
        if sequence_channel_axis is not None:
            channel_count = sequence_shape[sequence_channel_axis]
            invalid = [
                value
                for value in requested_keys
                if (value is None and channel_count != 1)
                or (
                    value is not None
                    and not 0 <= int(value) < channel_count
                )
            ]
            if invalid:
                raise ValueError(
                    f"Olympus channel selectors {invalid!r} are outside 0..{channel_count - 1}."
                )

        arrays: dict[Optional[int], np.ndarray] = {}
        seen: dict[Optional[int], set[int]] = {
            value: set() for value in requested_keys
        }
        errors: dict[Optional[int], Exception] = {}
        source_axes = None
        source_shape = None
        source_dtype = None
        plane_axes = None
        plane_shape = None

        for member_path, sequence_index in zip(files, indices):
            _check_cancel(cancel_check)
            if sequence_channel_axis is None:
                affected = requested_keys
            else:
                source_channel = sequence_index[sequence_channel_axis]
                affected = tuple(
                    value
                    for value in requested_keys
                    if value == source_channel
                    or (value is None and source_channel == 0)
                )
            if not affected:
                continue
            try:
                member, member_axes = _read_olympus_tiff_member(
                    reader, os.fspath(member_path)
                )
                _check_cancel(cancel_check)
                member_shape = tuple(int(value) for value in member.shape)
                member_dtype = np.dtype(member.dtype).name
                if plane_axes is None:
                    plane_axes = member_axes
                    plane_shape = member_shape
                    source_axes = sequence_axes + member_axes
                    source_shape = sequence_shape + member_shape
                    source_dtype = member_dtype
                elif (
                    member_axes != plane_axes
                    or member_shape != plane_shape
                    or member_dtype != source_dtype
                ):
                    raise ValueError(
                        "TIFF member axes, shape, or scalar dtype differs within the selected series."
                    )
                assert source_axes is not None
                assert source_shape is not None
                source_channel_axis = (
                    source_axes.index("C") if "C" in source_axes else None
                )
                source_channel_count = (
                    source_shape[source_channel_axis]
                    if source_channel_axis is not None
                    else 1
                )
                for key in affected:
                    if key in errors:
                        continue
                    if source_channel_axis is None:
                        if key is not None:
                            errors[key] = ValueError(
                                "Olympus selector declares a channel but the selected series has no channel axis."
                            )
                            continue
                    elif key is None and source_channel_count == 1:
                        pass
                    elif key is None or not 0 <= int(key) < source_channel_count:
                        errors[key] = ValueError(
                            f"Olympus channel {key!r} is outside 0..{source_channel_count - 1}."
                        )
                        continue

                    if key not in arrays:
                        selected_shape = list(source_shape)
                        if source_channel_axis is not None:
                            selected_shape[source_channel_axis] = 1
                        arrays[key] = np.empty(
                            tuple(selected_shape), dtype=member.dtype
                        )

                    target_sequence_index = list(sequence_index)
                    if sequence_channel_axis is not None:
                        target_sequence_index[sequence_channel_axis] = 0
                    selected_member = member
                    if "C" in member_axes:
                        member_channel_axis = member_axes.index("C")
                        selected_channel = 0 if key is None else int(key)
                        slicer = [slice(None)] * member.ndim
                        slicer[member_channel_axis] = slice(
                            selected_channel, selected_channel + 1
                        )
                        selected_member = member[tuple(slicer)]
                    target = arrays[key]
                    if target_sequence_index:
                        target[tuple(target_sequence_index)] = selected_member
                    else:
                        target[...] = selected_member
                    target_sequence_shape = list(sequence_shape)
                    if sequence_channel_axis is not None:
                        target_sequence_shape[sequence_channel_axis] = 1
                    seen[key].add(
                        _flat_sequence_index(
                            target_sequence_index, tuple(target_sequence_shape)
                        )
                    )
            except InterruptedError:
                raise
            except Exception as exc:
                diagnostic = OlympusInspectionError(
                    OLYMPUS_UNREADABLE_SOURCE,
                    f"could not decode TIFF member {member_path!r}: {exc}",
                )
                for key in affected:
                    errors.setdefault(key, diagnostic)

        results = []
        for key in requested:
            if key in errors:
                results.append(errors[key])
                continue
            if source_axes is None or source_shape is None or source_dtype is None:
                results.append(
                    OlympusInspectionError(
                        OLYMPUS_UNREADABLE_SOURCE,
                        f"series {selected_index} contains no readable selected TIFF members",
                    )
                )
                continue
            target_sequence_shape = list(sequence_shape)
            if sequence_channel_axis is not None:
                target_sequence_shape[sequence_channel_axis] = 1
            expected_selected_members = math.prod(target_sequence_shape)
            if key not in arrays or len(seen[key]) != expected_selected_members:
                results.append(
                    OlympusInspectionError(
                        OLYMPUS_REQUIRED_MEMBER_MISSING,
                        f"selected channel {key!r} decoded {len(seen[key])} of "
                        f"{expected_selected_members} required TIFF members",
                    )
                )
                continue
            results.append(
                OlympusChannelPixels(
                    channel_index=key,
                    array=arrays[key],
                    member_paths=member_paths,
                    source_axes=source_axes,
                    source_shape=source_shape,
                    scalar_dtype=source_dtype,
                )
            )
        return results
    finally:
        reader.close()


def _unique_text(values) -> tuple[str, ...]:
    result = []
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _normalized_member_path(value: Any) -> str:
    text = os.fspath(value).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    parts = [part for part in text.split("/") if part not in {"", "."}]
    if not parts or ".." in parts:
        return ""
    return "/".join(parts)


def _member_role(path: str, *, primary=False, directory=False, oib=False) -> str:
    if primary:
        return "compound-container" if oib else "primary-settings"
    if directory:
        return "bundle-directory"
    suffix = Path(path).suffix.lower()
    return {
        ".oif": "internal-main-settings",
        ".tif": "pixel-plane",
        ".tiff": "pixel-plane",
        ".pty": "plane-settings",
        ".txt": "settings",
        ".lut": "lookup-table",
        ".roi": "roi",
        ".bmp": "thumbnail",
    }.get(suffix, "auxiliary")


def _member_requirement(path: str, *, primary=False, directory=False) -> str:
    if primary or directory or Path(path).suffix.lower() in {".oif", ".tif", ".tiff"}:
        return "required"
    return "optional"


def _metadata_references(value: Any) -> tuple[str, ...]:
    result = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if not isinstance(item, str) or "%" in item:
            return
        normalized = _normalized_member_path(item.strip().strip('"'))
        if normalized and Path(normalized).suffix.lower() in _MEMBER_EXTENSIONS:
            result.append(normalized)

    visit(value)
    return tuple(dict.fromkeys(result))


def _safe_metadata(value: Any, path: str, exclusions: list[dict[str, str]]) -> Any:
    leaf = path.rsplit(".", 1)[-1].casefold()
    if leaf == "colorlutdata":
        exclusions.append({"path": path, "reason": "binary LUT attachment"})
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return copy.deepcopy(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        exclusions.append({"path": path, "reason": "non-finite numeric metadata"})
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            exclusions.append({"path": path, "reason": "binary metadata attachment"})
            return None
        if "\x00" in text:
            exclusions.append({"path": path, "reason": "binary metadata attachment"})
            return None
        return text
    if isinstance(value, dict):
        return {
            str(key): _safe_metadata(item, f"{path}.{key}", exclusions)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _safe_metadata(item, f"{path}[{index}]", exclusions)
            for index, item in enumerate(value)
        ]
    exclusions.append(
        {"path": path, "reason": f"unsupported metadata type {type(value).__name__}"}
    )
    return None


def _file_size(path: Path) -> Optional[int]:
    try:
        return int(path.stat().st_size)
    except OSError:
        return None


def _reference_match(reference: str, actual_paths: tuple[str, ...]) -> Optional[str]:
    normalized = _normalized_member_path(reference)
    if normalized in actual_paths:
        return normalized
    full_matches = [
        value for value in actual_paths if value.casefold() == normalized.casefold()
    ]
    if len(full_matches) == 1:
        return full_matches[0]
    basename = Path(normalized).name.casefold()
    matches = [value for value in actual_paths if Path(value).name.casefold() == basename]
    return matches[0] if len(matches) == 1 else None


def _discover_oif_members(
    source: Path,
    reader,
    references: tuple[str, ...],
    cancel_check: Optional[Callable[[], bool]],
) -> tuple[tuple[_MemberEntry, ...], tuple[str, ...]]:
    entries: dict[str, _MemberEntry] = {}
    warnings = []
    entries[f"primary:{source.name}"] = _MemberEntry(
        path=source.name,
        role=_member_role(source.name, primary=True),
        requirement="required",
        location_kind="filesystem",
        exists=source.is_file(),
        size_bytes=_file_size(source),
        reader_path=source.name,
        filesystem_path=str(source),
    )

    directories = {Path(str(source) + ".files")}
    for reference in references:
        first = reference.split("/", 1)[0]
        if first.casefold().endswith(".files"):
            directories.add(source.parent / first)

    for directory in sorted(directories, key=lambda item: str(item).casefold()):
        _check_cancel(cancel_check)
        if not directory.is_dir():
            if directory.name.casefold() == (source.name + ".files").casefold():
                raise OlympusInspectionError(
                    OLYMPUS_REQUIRED_MEMBER_MISSING,
                    f"associated OIF directory is missing: {directory.name}",
                )
            continue
        relative_directory = _normalized_member_path(
            os.path.relpath(directory, source.parent)
        )
        entries[f"directory:{relative_directory}"] = _MemberEntry(
            path=relative_directory,
            role=_member_role(relative_directory, directory=True),
            requirement="required",
            location_kind="filesystem-directory",
            exists=True,
            size_bytes=None,
            filesystem_path=str(directory),
        )
        for root, child_directories, files in os.walk(directory, followlinks=False):
            child_directories.sort(key=str.casefold)
            files.sort(key=str.casefold)
            _check_cancel(cancel_check)
            for name in files:
                physical = Path(root) / name
                relative = _normalized_member_path(
                    os.path.relpath(physical, source.parent)
                )
                entries[f"file:{relative}"] = _MemberEntry(
                    path=relative,
                    role=_member_role(relative),
                    requirement=_member_requirement(relative),
                    location_kind="filesystem",
                    exists=True,
                    size_bytes=_file_size(physical),
                    reader_path=relative,
                    filesystem_path=str(physical),
                )

    reader_paths = tuple(
        _normalized_member_path(value) for value in reader.filesystem.files()
    )
    for reader_path in reader_paths:
        if not reader_path or reader_path.casefold() == source.name.casefold():
            continue
        if not any(
            item.path.casefold() == reader_path.casefold() for item in entries.values()
        ):
            physical = source.parent / Path(reader_path.replace("/", os.path.sep))
            exists = physical.is_file()
            entry = _MemberEntry(
                path=reader_path,
                role=_member_role(reader_path),
                requirement=_member_requirement(reader_path),
                location_kind="filesystem",
                exists=exists,
                size_bytes=_file_size(physical),
                reader_path=reader_path,
                filesystem_path=str(physical),
            )
            entries[f"reader:{reader_path}"] = entry
            if not exists and entry.requirement == "required":
                raise OlympusInspectionError(
                    OLYMPUS_REQUIRED_MEMBER_MISSING,
                    f"required OIF member is missing: {reader_path}",
                )

    actual_paths = tuple(item.path for item in entries.values())
    for reference in references:
        if _reference_match(reference, actual_paths) is not None:
            continue
        requirement = _member_requirement(reference)
        warning = (
            f"Optional Olympus metadata member is missing: {reference}."
            if requirement == "optional"
            else None
        )
        if requirement == "required":
            raise OlympusInspectionError(
                OLYMPUS_REQUIRED_MEMBER_MISSING,
                f"required OIF member is missing: {reference}",
            )
        warnings.append(warning)
        entries[f"missing:{reference}"] = _MemberEntry(
            path=reference,
            role=_member_role(reference),
            requirement=requirement,
            location_kind="filesystem",
            exists=False,
            size_bytes=None,
            warning=warning,
        )

    result = tuple(
        sorted(
            entries.values(),
            key=lambda item: (
                0 if item.path.casefold() == source.name.casefold() else 1,
                0 if item.location_kind == "filesystem-directory" else 1,
                item.path.casefold(),
                item.path,
            ),
        )
    )
    return result, _unique_text(warnings)


def _discover_oib_members(
    source: Path,
    reader,
    references: tuple[str, ...],
) -> tuple[tuple[_MemberEntry, ...], tuple[str, ...]]:
    entries = [
        _MemberEntry(
            path=source.name,
            role=_member_role(source.name, primary=True, oib=True),
            requirement="required",
            location_kind="filesystem",
            exists=source.is_file(),
            size_bytes=_file_size(source),
            filesystem_path=str(source),
        )
    ]
    internal_paths = tuple(
        sorted(
            {
                _normalized_member_path(value)
                for value in reader.filesystem.files()
                if _normalized_member_path(value)
            },
            key=lambda value: (value.casefold(), value),
        )
    )
    for path in internal_paths:
        entries.append(
            _MemberEntry(
                path=path,
                role=_member_role(path),
                requirement=_member_requirement(path),
                location_kind="compound-member",
                exists=True,
                size_bytes=None,
                reader_path=path,
            )
        )
    warnings = []
    for reference in references:
        if _reference_match(reference, internal_paths) is not None:
            continue
        requirement = _member_requirement(reference)
        if requirement == "required":
            raise OlympusInspectionError(
                OLYMPUS_REQUIRED_MEMBER_MISSING,
                f"required OIB stream is missing: {reference}",
            )
        warning = f"Optional Olympus metadata stream is missing: {reference}."
        warnings.append(warning)
        entries.append(
            _MemberEntry(
                path=reference,
                role=_member_role(reference),
                requirement="optional",
                location_kind="compound-member",
                exists=False,
                size_bytes=None,
                warning=warning,
            )
        )
    return (
        tuple(
            sorted(
                entries,
                key=lambda item: (
                    0 if item.location_kind == "filesystem" else 1,
                    item.path.casefold(),
                    item.path,
                ),
            )
        ),
        _unique_text(warnings),
    )


def _read_auxiliary_settings(
    reader,
    reader_module,
    entries: tuple[_MemberEntry, ...],
    cancel_check: Optional[Callable[[], bool]],
) -> tuple[dict[str, Any], tuple[str, ...], tuple[dict[str, str], ...]]:
    result = {}
    warnings = []
    exclusions: list[dict[str, str]] = []
    mainfile = _normalized_member_path(getattr(reader.filesystem, "mainfile", ""))
    for entry in entries:
        _check_cancel(cancel_check)
        if (
            not entry.exists
            or not entry.reader_path
            or Path(entry.path).suffix.lower() not in _OPTIONAL_SETTINGS_EXTENSIONS
            or entry.path.casefold() == mainfile.casefold()
        ):
            continue
        try:
            settings = reader_module.SettingsFile(
                reader.open_file(entry.reader_path), entry.path
            )
        except FileNotFoundError:
            warnings.append(f"Optional Olympus metadata member is missing: {entry.path}.")
            continue
        except Exception as exc:
            warnings.append(
                f"Malformed optional Olympus metadata member {entry.path}: {exc}"
            )
            continue
        result[entry.path] = _safe_metadata(
            dict(settings), f"olympus_auxiliary_settings.{entry.path}", exclusions
        )
    return result, _unique_text(warnings), tuple(exclusions)


def _manifest_identity(entries: tuple[_MemberEntry, ...]) -> str:
    payload = [
        {
            "path": entry.path,
            "role": entry.role,
            "requirement": entry.requirement,
            "location_kind": entry.location_kind,
            "exists": entry.exists,
            "size_bytes": entry.size_bytes,
        }
        for entry in entries
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive_number(value: Any) -> Optional[float]:
    result = _number(value)
    return result if result is not None and result > 0.0 else None


def _axis_sections(main_settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for section_name, section in main_settings.items():
        if not re.fullmatch(r"Axis \d+ Parameters Common", str(section_name)):
            continue
        if not isinstance(section, dict):
            continue
        code = str(section.get("AxisCode") or "").strip().upper()
        if code and code not in result:
            result[code] = section
    return result


def _unit_factor(value: Any) -> Optional[float]:
    key = str(value or "").strip().strip('"').casefold()
    return _PHYSICAL_UNIT_FACTORS_TO_MICRON.get(key)


def _axis_physical_unit(
    axis: str, section: dict[str, Any], diagnostics: list[str]
) -> Optional[str]:
    declared = [
        str(section.get(key) or "").strip()
        for key in ("UnitName", "PixUnit")
        if str(section.get(key) or "").strip()
    ]
    physical = [(value, _unit_factor(value)) for value in declared]
    factors = {factor for _value, factor in physical if factor is not None}
    if len(factors) > 1:
        diagnostics.append(
            f"Olympus {axis} axis declares conflicting physical units: "
            + ", ".join(declared)
            + "."
        )
        return None
    return next((value for value, factor in physical if factor is not None), None)


def _spacing_evidence(
    main_settings: dict[str, Any], axes: tuple[str, ...], shape: tuple[int, ...]
):
    diagnostics = []
    missing = []
    axis_sections = _axis_sections(main_settings)
    reference = main_settings.get("Reference Image Parameter") or {}
    candidates_by_axis: dict[str, list[dict[str, Any]]] = {axis: [] for axis in "XYZ"}

    for axis, value_key, unit_key in (
        ("X", "WidthConvertValue", "WidthUnit"),
        ("Y", "HeightConvertValue", "HeightUnit"),
    ):
        value = _positive_number(reference.get(value_key))
        unit = str(reference.get(unit_key) or "").strip()
        factor = _unit_factor(unit)
        if value is not None and factor is not None:
            candidates_by_axis[axis].append(
                {
                    "source": f"Reference Image Parameter.{value_key}",
                    "raw_value": value,
                    "raw_unit": unit,
                    "microns": value * factor,
                }
            )
        elif reference.get(value_key) not in (None, ""):
            diagnostics.append(
                f"Olympus {axis} spacing has an unsupported or missing unit {unit!r}."
            )

    for axis in "XYZ":
        section = axis_sections.get(axis) or {}
        unit = _axis_physical_unit(axis, section, diagnostics)
        factor = _unit_factor(unit)
        interval = _positive_number(section.get("Interval"))
        if interval is not None and factor is not None:
            candidates_by_axis[axis].append(
                {
                    "source": f"Axis {axis}.Interval",
                    "raw_value": interval,
                    "raw_unit": unit,
                    "microns": interval * factor,
                }
            )
        start = _number(section.get("StartPosition"))
        end = _number(section.get("EndPosition"))
        size = int(shape[axes.index(axis)]) if axis in axes else 1
        if start is not None and end is not None and size > 1 and factor is not None:
            step = abs(end - start) / float(size - 1)
            if step > 0.0:
                candidates_by_axis[axis].append(
                    {
                        "source": f"Axis {axis}.StartPosition/EndPosition",
                        "raw_value": step,
                        "raw_unit": unit,
                        "microns": step * factor,
                    }
                )

    resolved = []
    for axis in "XYZ":
        candidates = candidates_by_axis[axis]
        if not candidates:
            resolved.append(None)
            missing.append(f"spacing_{axis.lower()}")
            continue
        first = float(candidates[0]["microns"])
        conflicts = [
            item
            for item in candidates[1:]
            if not math.isclose(
                first, float(item["microns"]), rel_tol=1e-4, abs_tol=1e-9
            )
        ]
        if conflicts:
            diagnostics.append(
                f"Olympus {axis} spacing fields conflict after unit normalization."
            )
            resolved.append(None)
            missing.append(f"spacing_{axis.lower()}")
        else:
            resolved.append(first)

    recognized_units = any(candidates_by_axis[axis] for axis in "XYZ")
    units = ("micron", "micron", "micron") if recognized_units else None
    if units is None:
        missing.append("space_units")
    spacing = (
        tuple(float(value) for value in resolved)
        if all(value is not None for value in resolved)
        else None
    )
    return (
        spacing,
        units,
        tuple(missing),
        _unique_text(diagnostics),
        candidates_by_axis,
    )


def _stage_position(
    main_settings: dict[str, Any],
    *,
    source_path: str,
    source_member_id: str,
    reader_version: Optional[str],
    series_identity: str,
    series_index: int,
) -> Optional[StagePositionObservation]:
    axis_sections = _axis_sections(main_settings)
    raw_values = []
    raw_units = []
    normalized = []
    normalized_units = []
    source_fields = {}
    warnings = []
    conflicting = False
    unsupported = False

    for axis in "XYZ":
        section = axis_sections.get(axis) or {}
        absolute_fields = {
            str(key): copy.deepcopy(value)
            for key, value in section.items()
            if "absolute" in str(key).casefold()
            and "position" in str(key).casefold()
        }
        value = _number(section.get("AbsolutePosition"))
        if value is None:
            value = next(
                (_number(item) for item in absolute_fields.values() if _number(item) is not None),
                None,
            )
        declared_units = tuple(
            str(section.get(key) or "").strip()
            for key in ("UnitName", "PixUnit")
            if str(section.get(key) or "").strip()
        )
        factors = {
            _unit_factor(unit) for unit in declared_units if _unit_factor(unit) is not None
        }
        unit = next((item for item in declared_units if _unit_factor(item) is not None), None)
        if len(factors) > 1:
            conflicting = True
            warnings.append(
                f"Olympus stage {axis} axis declares conflicting units: "
                + ", ".join(declared_units)
                + "."
            )
            factor = None
        else:
            factor = _unit_factor(unit)
        if value is not None and factor is None:
            unsupported = True
            warnings.append(
                f"Olympus stage {axis} absolute position has no supported physical unit."
            )
        raw_values.append(value)
        raw_units.append(unit)
        normalized.append(value * factor if value is not None and factor is not None else None)
        normalized_units.append("micron" if normalized[-1] is not None else None)
        source_fields[axis] = {
            "absolute_position_fields": absolute_fields,
            "UnitName": section.get("UnitName"),
            "PixUnit": section.get("PixUnit"),
        }

    if all(value is None for value in raw_values):
        return None
    if conflicting:
        status = "conflicting"
    elif unsupported:
        status = "unsupported-units"
    elif any(value is None for value in raw_values) or any(
        unit is None for unit in raw_units
    ):
        status = "partial"
        warnings.append("Olympus stage position reports only a subset of XYZ axes or units.")
    else:
        status = "interpreted"
    return StagePositionObservation(
        raw_position_xyz=tuple(raw_values),
        raw_units_xyz=tuple(raw_units),
        normalized_position_xyz=tuple(normalized),
        normalized_units_xyz=tuple(normalized_units),
        semantic_meaning="reported-stage-position",
        coordinate_frame="olympus-instrument-stage-unverified",
        source_fields={
            "family": "olympus",
            "source_label": "Olympus absolute stage position",
            "fields": [
                f"{axis}.AbsolutePosition"
                for index, axis in enumerate("XYZ")
                if raw_values[index] is not None
            ],
            "conversion_factors": [
                (
                    normalized[index] / raw_values[index]
                    if raw_values[index] not in (None, 0.0)
                    and normalized[index] is not None
                    else 1.0
                    if raw_values[index] == 0.0 and normalized[index] == 0.0
                    else None
                )
                for index in range(3)
            ],
            "candidate_status": {
                "interpreted": "usable",
                "partial": "partial",
                "conflicting": "invalid",
                "unsupported-units": "ambiguous",
            }.get(status, "ambiguous"),
            "rejection_reason": "",
            "olympus_axis_evidence": source_fields,
        },
        series_identity=series_identity,
        series_index=series_index,
        interpretation_status=status,
        warnings=_unique_text(warnings),
        reader_backend="oiffile",
        reader_version=reader_version,
        source_path=source_path,
        source_member_id=source_member_id,
    )


def _flatten_settings(main_settings: dict[str, Any]):
    for section_name, section in main_settings.items():
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            yield str(section_name), str(key), normalized, value


def _first_setting(main_settings: dict[str, Any], names) -> Any:
    expected = {re.sub(r"[^a-z0-9]", "", str(name).casefold()) for name in names}
    for _section, _key, normalized, value in _flatten_settings(main_settings):
        if normalized in expected and value not in (None, ""):
            return value
    return None


def _objective_fields(main_settings: dict[str, Any]):
    model = _first_setting(
        main_settings,
        ("ObjectiveName", "ObjectiveModel", "ObjectiveLens", "Objective"),
    )
    magnification = _positive_number(
        _first_setting(main_settings, ("ObjectiveMagnification", "ObjectiveMag"))
    )
    numerical_aperture = _positive_number(
        _first_setting(
            main_settings,
            ("ObjectiveNumericalAperture", "NumericalAperture", "ObjectiveNA"),
        )
    )
    immersion = _first_setting(main_settings, ("ObjectiveImmersion", "Immersion"))
    if model:
        text = str(model)
        if magnification is None:
            match = re.search(r"(?<![0-9.])(\d+(?:\.\d+)?)\s*[xX]", text)
            if match:
                magnification = _positive_number(match.group(1))
        if numerical_aperture is None:
            match = re.search(r"/\s*(\d+(?:\.\d+)?)", text)
            if match:
                numerical_aperture = _positive_number(match.group(1))
        if immersion is None:
            match = re.search(r"\b(oil|water|air|glycerol|silicone)\b", text, re.I)
            if match:
                immersion = match.group(1)
    return (
        magnification,
        numerical_aperture,
        str(immersion).strip() if immersion else None,
        str(model).strip() if model else None,
    )


def _channel_metadata(
    main_settings: dict[str, Any], channel_count: int
) -> tuple[tuple[str, ...], tuple[MicroscopyChannelMetadata, ...], bool]:
    result = []
    names = []
    verified = True
    for index in range(channel_count):
        section = main_settings.get(f"Channel {index + 1} Parameters")
        if not isinstance(section, dict):
            section = main_settings.get(f"GUI Channel {index + 1} Parameters")
        if not isinstance(section, dict):
            section = {}
            verified = False
        raw_name = str(section.get("CH Name") or "").strip()
        dye_name = str(section.get("DyeName") or "").strip()
        if dye_name.casefold() in {"none", "unknown"}:
            dye_name = ""
        name = dye_name or raw_name or f"ch{index + 1}"
        if name in names:
            name = f"{name} [{index + 1}]"
        names.append(name)
        excitation = _positive_number(section.get("ExcitationWavelength"))
        emission = _positive_number(section.get("EmissionWavelength"))
        result.append(
            MicroscopyChannelMetadata(
                source_channel_identifier=str(
                    section.get("Physical CH Number")
                    or section.get("Device CH Number")
                    or index + 1
                ),
                source_channel_name=name,
                excitation_wavelength=excitation,
                excitation_wavelength_units="nm" if excitation is not None else None,
                emission_wavelength=emission,
                emission_wavelength_units="nm" if emission is not None else None,
                detector_settings=copy.deepcopy(section),
                reported_scientific_role=dye_name or None,
                normalized_metadata={
                    "olympus_channel_index": index,
                    "raw_channel_name": raw_name or None,
                    "raw_dye_name": dye_name or None,
                },
            )
        )
    return tuple(names), tuple(result), verified


def _time_evidence(
    main_settings: dict[str, Any], axes: tuple[str, ...], shape: tuple[int, ...]
):
    count = int(shape[axes.index("T")]) if "T" in axes else 1
    if count <= 1:
        return 1, 1.0, "frame", ()
    section = _axis_sections(main_settings).get("T") or {}
    interval = _positive_number(section.get("Interval"))
    unit = str(section.get("UnitName") or section.get("PixUnit") or "").strip()
    if interval is None:
        start = _number(section.get("StartPosition"))
        end = _number(section.get("EndPosition"))
        if start is not None and end is not None and count > 1:
            interval = abs(end - start) / float(count - 1)
    if interval is None or interval <= 0.0 or not unit:
        return (
            count,
            1.0,
            "frame",
            ("Olympus time interval or unit is missing; timepoints remain distinct frames.",),
        )
    return count, float(interval), unit, ()


def _series_files(series) -> tuple[str, ...]:
    raw = getattr(series, "_files", ())
    return tuple(
        value
        for value in (_normalized_member_path(item) for item in raw)
        if value
    )


def _series_identity(index: int, paths: tuple[str, ...], axes, shape) -> str:
    payload = json.dumps(
        {
            "index": index,
            "members": list(paths),
            "axes": list(axes),
            "shape": list(shape),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"olympus-series-{index}:" + hashlib.sha256(payload).hexdigest()[:16]


def _primary_shape(reader) -> tuple[tuple[str, ...], tuple[int, ...], str, Optional[int]]:
    try:
        axes = tuple(str(reader.axes or "").upper())
        shape = tuple(int(value) for value in reader.shape)
        dtype = np.dtype(reader.dtype).name
    except Exception as exc:
        raise OlympusInspectionError(
            OLYMPUS_MALFORMED_METADATA,
            f"axis, shape, or scalar metadata is invalid: {exc}",
        ) from exc
    if len(axes) != len(shape) or not axes or any(value <= 0 for value in shape):
        raise OlympusInspectionError(
            OLYMPUS_MALFORMED_METADATA,
            "source axes and positive dimensions must be explicit and have equal lengths",
        )
    if "X" not in axes or "Y" not in axes:
        raise OlympusInspectionError(
            OLYMPUS_UNSUPPORTED_VARIANT,
            f"source axes {''.join(axes)!r} do not declare both X and Y",
        )
    bit_depth = None
    try:
        bit_depth = int(reader.mainfile["Reference Image Parameter"]["ValidBitCounts"])
    except (KeyError, TypeError, ValueError):
        bit_depth = int(np.dtype(dtype).itemsize * 8)
    return axes, shape, dtype, bit_depth


def _secondary_shape(
    sequence,
    *,
    reader,
    member_paths: tuple[str, ...],
    primary_axes: tuple[str, ...],
    primary_shape: tuple[int, ...],
    is_oib: bool,
):
    sequence_axes = tuple(str(getattr(sequence, "axes", "") or "").upper())
    sequence_shape = tuple(int(value) for value in getattr(sequence, "shape", ()))
    if len(sequence_axes) != len(sequence_shape):
        raise OlympusInspectionError(
            OLYMPUS_MALFORMED_METADATA,
            "an Olympus TIFF sequence reports mismatched axes and dimensions",
        )
    spatial_axes = tuple(axis for axis in primary_axes if axis in {"Y", "X"})
    spatial_shape = tuple(
        primary_shape[primary_axes.index(axis)] for axis in spatial_axes
    )
    dtype = np.dtype(reader.dtype).name
    try:
        bit_depth = int(
            reader.mainfile["Reference Image Parameter"]["ValidBitCounts"]
        )
    except (KeyError, TypeError, ValueError):
        bit_depth = int(np.dtype(dtype).itemsize * 8)
    warnings = []
    if not is_oib and member_paths:
        import tifffile

        try:
            with reader.open_file(member_paths[0]) as handle:
                with tifffile.TiffFile(handle, name=member_paths[0]) as tiff:
                    plane = tiff.series[0]
                    spatial_axes = tuple(str(plane.axes or "").upper())
                    spatial_shape = tuple(int(value) for value in plane.shape)
                    dtype = np.dtype(plane.dtype).name
                    raw_bits = getattr(plane.pages[0], "bitspersample", None)
                    values = tuple(int(value) for value in np.asarray(raw_bits).reshape(-1))
                    if values and len(set(values)) == 1 and values[0] > 0:
                        bit_depth = values[0]
        except Exception as exc:
            raise OlympusInspectionError(
                OLYMPUS_UNREADABLE_SOURCE,
                f"could not inspect TIFF member {member_paths[0]}: {exc}",
            ) from exc
    elif is_oib:
        warnings.append(
            "Secondary OIB series plane dimensions use common OIF metadata; "
            "pixel streams were not opened during catalog inspection."
        )
    axes = sequence_axes + spatial_axes
    shape = sequence_shape + spatial_shape
    return axes, shape, dtype, bit_depth, tuple(warnings)


def _validate_pixel_members(
    raw_series,
    *,
    reader,
    is_oib: bool,
    primary_axes: tuple[str, ...],
    primary_shape: tuple[int, ...],
) -> None:
    if not raw_series:
        raise OlympusInspectionError(
            OLYMPUS_REQUIRED_MEMBER_MISSING,
            "no referenced TIFF pixel members are available",
        )
    for index, series in enumerate(raw_series):
        files = _series_files(series)
        if not files:
            raise OlympusInspectionError(
                OLYMPUS_REQUIRED_MEMBER_MISSING,
                f"series {index} contains no TIFF pixel members",
            )
        sequence_shape = tuple(int(value) for value in getattr(series, "shape", ()))
        if sequence_shape and math.prod(sequence_shape) > len(files):
            raise OlympusInspectionError(
                OLYMPUS_REQUIRED_MEMBER_MISSING,
                f"series {index} references {math.prod(sequence_shape)} TIFF "
                f"planes but only {len(files)} are present",
            )
    expected_primary = math.prod(
        size
        for axis, size in zip(primary_axes, primary_shape)
        if axis not in {"X", "Y", "S"}
    )
    primary_files = len(_series_files(raw_series[0]))
    plane_capacity = 1
    if not is_oib:
        import tifffile

        first_member = _series_files(raw_series[0])[0]
        try:
            with reader.open_file(first_member) as handle:
                with tifffile.TiffFile(handle, name=first_member) as tiff:
                    plane = tiff.series[0]
                    plane_axes = tuple(str(plane.axes or "").upper())
                    plane_shape = tuple(int(value) for value in plane.shape)
                    plane_capacity = math.prod(
                        size
                        for axis, size in zip(plane_axes, plane_shape)
                        if axis not in {"X", "Y", "S"}
                    )
        except Exception as exc:
            raise OlympusInspectionError(
                OLYMPUS_UNREADABLE_SOURCE,
                f"could not inspect required TIFF member {first_member}: {exc}",
            ) from exc
    available_primary = primary_files * max(1, plane_capacity)
    if expected_primary > available_primary:
        raise OlympusInspectionError(
            OLYMPUS_REQUIRED_MEMBER_MISSING,
            f"primary series requires {expected_primary} TIFF planes but only "
            f"{available_primary} are present",
        )


def inspect_olympus_source(
    path: os.PathLike[str] | str,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    reader_module=None,
) -> OlympusSourceInspection:
    """Inspect OIF/OIB settings and members without materializing voxel arrays."""
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    container_format = "olympus-oif" if suffix == ".oif" else "olympus-oib"
    if suffix not in {".oif", ".oib"}:
        raise OlympusInspectionError(
            OLYMPUS_UNSUPPORTED_VARIANT,
            f"unsupported Olympus suffix {suffix or '<none>'}",
        )
    _check_cancel(cancel_check)
    if not source.is_file():
        raise OlympusInspectionError(
            OLYMPUS_REQUIRED_MEMBER_MISSING,
            f"primary source file is missing: {source.name}",
        )
    reader_module = reader_module or require_oiffile()
    try:
        reader = reader_module.OifFile(source)
    except PermissionError as exc:
        raise OlympusInspectionError(
            OLYMPUS_UNREADABLE_SOURCE, f"cannot read {source.name}: {exc}"
        ) from exc
    except FileNotFoundError as exc:
        raise OlympusInspectionError(
            OLYMPUS_REQUIRED_MEMBER_MISSING, str(exc)
        ) from exc
    except NotImplementedError as exc:
        raise OlympusInspectionError(
            OLYMPUS_UNSUPPORTED_VARIANT, str(exc)
        ) from exc
    except OSError as exc:
        text = str(exc)
        category = (
            OLYMPUS_REQUIRED_MEMBER_MISSING
            if suffix == ".oif" and "storage path not found" in text.casefold()
            else OLYMPUS_UNREADABLE_SOURCE
        )
        raise OlympusInspectionError(category, text) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise OlympusInspectionError(
            OLYMPUS_MALFORMED_METADATA, str(exc)
        ) from exc

    try:
        _check_cancel(cancel_check)
        is_oib = bool(reader.is_oib)
        if is_oib != (suffix == ".oib"):
            raise OlympusInspectionError(
                OLYMPUS_MALFORMED_METADATA,
                "reader container identity does not match the source suffix",
            )
        exclusions: list[dict[str, str]] = []
        main_settings = _safe_metadata(
            dict(reader.mainfile), "olympus_main_settings", exclusions
        )
        if not isinstance(main_settings, dict):
            raise OlympusInspectionError(
                OLYMPUS_MALFORMED_METADATA,
                "main settings are not a structured metadata object",
            )
        references = _metadata_references(main_settings)
        if is_oib:
            member_entries, member_warnings = _discover_oib_members(
                source, reader, references
            )
        else:
            member_entries, member_warnings = _discover_oif_members(
                source, reader, references, cancel_check
            )
        auxiliary, auxiliary_warnings, auxiliary_exclusions = _read_auxiliary_settings(
            reader, reader_module, member_entries, cancel_check
        )
        exclusions.extend(auxiliary_exclusions)
        source_warnings = _unique_text((*member_warnings, *auxiliary_warnings))

        primary_axes, primary_shape, primary_dtype, primary_bits = _primary_shape(reader)
        raw_series = tuple(reader.series)
        _validate_pixel_members(
            raw_series,
            reader=reader,
            is_oib=is_oib,
            primary_axes=primary_axes,
            primary_shape=primary_shape,
        )
        spacing, space_units, spacing_missing, spacing_diagnostics, spacing_raw = (
            _spacing_evidence(main_settings, primary_axes, primary_shape)
        )
        acquisition_common = main_settings.get("Acquisition Parameters Common") or {}
        scan_direction = str(acquisition_common.get("ScanDirection") or "").strip() or None
        version_info = main_settings.get("Version Info") or {}
        profile_info = main_settings.get("ProfileSaveInfo") or {}
        file_info = main_settings.get("File Info") or {}
        reader_version = str(getattr(reader_module, "__version__", "") or "").strip() or None
        software_version = str(
            version_info.get("SystemVersion") or profile_info.get("Version") or ""
        ).strip() or None
        objective = _objective_fields(main_settings)
        series_records = []
        series_evidence = []
        axis_evidence = []
        stage_source_member = (
            f"compound-member:{_normalized_member_path(reader.filesystem.mainfile)}"
            if is_oib
            else "primary"
        )

        for index, sequence in enumerate(raw_series):
            _check_cancel(cancel_check)
            member_paths = _series_files(sequence)
            if index == 0:
                axes = primary_axes
                shape = primary_shape
                dtype = primary_dtype
                bit_depth = primary_bits
                shape_warnings = ()
            else:
                axes, shape, dtype, bit_depth, shape_warnings = _secondary_shape(
                    sequence,
                    reader=reader,
                    member_paths=member_paths,
                    primary_axes=primary_axes,
                    primary_shape=primary_shape,
                    is_oib=is_oib,
                )
            if len(axes) != len(shape) or any(value <= 0 for value in shape):
                raise OlympusInspectionError(
                    OLYMPUS_MALFORMED_METADATA,
                    f"series {index} axes and positive dimensions are inconsistent",
                )
            identity = _series_identity(index, member_paths, axes, shape)
            channel_count = int(shape[axes.index("C")]) if "C" in axes else 1
            names, channel_metadata, channel_order_verified = _channel_metadata(
                main_settings, channel_count
            )
            time_count, time_interval, time_units, time_warnings = _time_evidence(
                main_settings, axes, shape
            )
            stage = _stage_position(
                main_settings,
                source_path=str(source),
                source_member_id=stage_source_member,
                reader_version=reader_version,
                series_identity=identity,
                series_index=index,
            )
            warnings = _unique_text(
                (
                    *shape_warnings,
                    *time_warnings,
                    *(stage.warnings if stage is not None else ()),
                )
            )
            acquisition = MicroscopyAcquisitionMetadata(
                acquisition_timestamp=(
                    str(
                        acquisition_common.get("ImageCaputreDate")
                        or acquisition_common.get("ImageCaptureDate")
                        or ""
                    ).strip()
                    or None
                ),
                microscope_vendor="Olympus",
                microscope_model=(
                    str(
                        version_info.get("SystemName")
                        or acquisition_common.get("Acquisition Device")
                        or ""
                    ).strip()
                    or None
                ),
                acquisition_software="Olympus FluoView",
                reported_acquisition_software_version=software_version,
                objective_magnification=objective[0],
                objective_numerical_aperture=objective[1],
                objective_immersion=objective[2],
                objective_model=objective[3],
                series_identity=identity,
                scene_identity=identity,
                reported_scan_direction=scan_direction,
                stage_positions=(stage,) if stage is not None else (),
                normalized_metadata={
                    "olympus_series_index": index,
                    "time_point_count": time_count,
                    "time_interval": time_interval,
                    "time_units": time_units,
                },
                warnings=warnings,
            )
            series_name = str(
                getattr(sequence, "name", "")
                or (file_info.get("DataName") if index == 0 else "")
                or f"Olympus series {index + 1}"
            ).strip()
            raw_fields = {
                "olympus_axis_sections": copy.deepcopy(_axis_sections(main_settings)),
                "olympus_spacing_evidence": copy.deepcopy(spacing_raw),
                "source_member_paths": list(member_paths),
                "time_point_count": time_count,
                "time_interval": time_interval,
                "time_units": time_units,
            }
            series_records.append(
                OlympusSeriesInspection(
                    index=index,
                    identity=identity,
                    name=series_name,
                    axes=axes,
                    shape=shape,
                    scalar_dtype=dtype,
                    scalar_bit_depth=bit_depth,
                    channel_names=names,
                    channel_metadata=channel_metadata,
                    channel_order_verified=channel_order_verified,
                    spacing=spacing,
                    space_units=space_units,
                    time_count=time_count,
                    time_interval=time_interval,
                    time_units=time_units,
                    scan_direction=scan_direction,
                    acquisition_software_version=software_version,
                    acquisition_metadata=acquisition,
                    missing_fields=tuple(dict.fromkeys((*spacing_missing, "origin", "direction"))),
                    physical_geometry_diagnostics=spacing_diagnostics,
                    warnings=warnings,
                    raw_fields=raw_fields,
                    member_paths=member_paths,
                )
            )
            series_evidence.append(
                {
                    "series_index": index,
                    "series_identity": identity,
                    "name": series_name,
                    "member_paths": list(member_paths),
                }
            )
            axis_evidence.append(
                {
                    "series_index": index,
                    "axes": list(axes),
                    "shape": list(shape),
                }
            )

        records = []
        for entry in member_entries:
            member_id = (
                "primary"
                if entry.location_kind == "filesystem"
                and entry.path.casefold() == source.name.casefold()
                else None
            )
            records.append(entry.to_record(member_id=member_id))
        raw_metadata = {
            "olympus_main_settings": main_settings,
            "olympus_auxiliary_settings": auxiliary,
        }
        if exclusions:
            raw_metadata["excluded_metadata"] = exclusions
            source_warnings = _unique_text(
                (
                    *source_warnings,
                    *(
                        f"Excluded {item['path']}: {item['reason']}."
                        for item in exclusions
                    ),
                )
            )
        manifest_identity = _manifest_identity(member_entries)
        source_metadata = MicroscopySourceMetadata(
            reader_backend="oiffile",
            reader_version=reader_version,
            reported_vendor_format=container_format,
            reported_primary_source_path=str(source),
            source_members=tuple(records),
            checksum_state="not-computed",
            raw_metadata=raw_metadata,
            source_series_evidence=series_evidence,
            source_axis_evidence=axis_evidence,
            warnings=source_warnings,
            additional_fields={
                "source_bundle": {
                    "kind": "compound-file" if is_oib else "multi-file",
                    "manifest_identity": manifest_identity,
                    "member_count": len(records),
                    "member_checksums": "not-computed",
                }
            },
        )
        return OlympusSourceInspection(
            source_path=str(source),
            container_format=container_format,
            source_metadata=source_metadata,
            series=tuple(series_records),
            warnings=source_warnings,
        )
    finally:
        reader.close()


__all__ = [
    "decode_olympus_series_channels",
    "inspect_olympus_source",
    "OlympusChannelPixels",
    "OlympusInspectionError",
    "OlympusSeriesInspection",
    "OlympusSourceInspection",
    "OLYMPUS_MALFORMED_METADATA",
    "OLYMPUS_REQUIRED_MEMBER_MISSING",
    "OLYMPUS_UNREADABLE_SOURCE",
    "OLYMPUS_UNSUPPORTED_VARIANT",
    "require_oiffile",
]
