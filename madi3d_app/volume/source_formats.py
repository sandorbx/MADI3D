"""Authoritative registry for readable scientific volume source formats."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SourceBundleError(ValueError):
    """A backing directory cannot be imported as an independent source."""


def oif_companion_source(directory) -> str | None:
    """Resolve an OIF backing directory to its exact sibling descriptor.

    Suffix recognition is case insensitive, but sibling lookup follows the
    filesystem's own case rules. Never infer a descriptor from bundle members.
    """
    path = Path(directory)
    if not path.name.casefold().endswith(".oif.files") or not path.is_dir():
        return None
    descriptor = path.with_name(path.name[:-len(".files")])
    if not descriptor.is_file():
        raise SourceBundleError(
            f"{path.name} contains backing files for an Olympus OIF source, "
            f"but its sibling {descriptor.name} was not found. "
            "Select or drop the .oif file together with its companion directory."
        )
    return os.fspath(descriptor)


@dataclass(frozen=True)
class VolumeSourceFormat:
    """One source-container family and its cross-module handling policy."""

    key: str
    suffixes: tuple[str, ...]
    reader_mode: str
    container_formats: tuple[str, ...]
    reader_dependencies: tuple[str, ...]
    decode_memory_factor: float
    decode_memory_floor_bytes: int
    source_replacement_supported: bool = True
    source_replacement_checks_channels: bool = False
    source_replacement_reason: str = ""
    prefer_model_decode_contract: bool = False


_MIB = 1024**2

VOLUME_SOURCE_FORMATS = (
    VolumeSourceFormat(
        key="nrrd",
        suffixes=(".nrrd",),
        reader_mode="nrrd",
        container_formats=("nrrd",),
        reader_dependencies=("nrrd",),
        decode_memory_factor=8.0,
        decode_memory_floor_bytes=512 * _MIB,
        source_replacement_checks_channels=True,
        prefer_model_decode_contract=True,
    ),
    VolumeSourceFormat(
        key="nhdr",
        suffixes=(".nhdr",),
        reader_mode="nrrd",
        container_formats=("nrrd",),
        reader_dependencies=("nrrd",),
        decode_memory_factor=8.0,
        decode_memory_floor_bytes=512 * _MIB,
        source_replacement_supported=False,
        source_replacement_reason=(
            "Detached NRRD (.nhdr) sources require export to a new file; atomic "
            "source replacement is not supported."
        ),
        prefer_model_decode_contract=True,
    ),
    VolumeSourceFormat(
        key="tiff",
        suffixes=(".tif", ".tiff"),
        reader_mode="tiff",
        container_formats=("tiff", "imagej-tiff", "ome-tiff", "zeiss-lsm"),
        reader_dependencies=("tifffile", "imagecodecs"),
        decode_memory_factor=4.0,
        decode_memory_floor_bytes=384 * _MIB,
        source_replacement_checks_channels=True,
    ),
    VolumeSourceFormat(
        key="lsm",
        suffixes=(".lsm",),
        reader_mode="tiff",
        container_formats=("zeiss-lsm",),
        reader_dependencies=("tifffile", "imagecodecs"),
        decode_memory_factor=4.0,
        decode_memory_floor_bytes=384 * _MIB,
        source_replacement_supported=False,
        source_replacement_reason=(
            "Zeiss LSM sources are read-only; writing or source replacement is "
            "unsupported."
        ),
    ),
    VolumeSourceFormat(
        key="oif",
        suffixes=(".oif",),
        reader_mode="olympus",
        container_formats=("olympus-oif",),
        reader_dependencies=("oiffile", "tifffile", "imagecodecs"),
        decode_memory_factor=4.0,
        decode_memory_floor_bytes=384 * _MIB,
        source_replacement_supported=False,
        source_replacement_reason=(
            "Olympus OIF/OIB sources are read-only; writing or source replacement "
            "is unsupported."
        ),
        prefer_model_decode_contract=True,
    ),
    VolumeSourceFormat(
        key="oib",
        suffixes=(".oib",),
        reader_mode="olympus",
        container_formats=("olympus-oib",),
        reader_dependencies=("oiffile", "tifffile", "imagecodecs"),
        decode_memory_factor=4.0,
        decode_memory_floor_bytes=384 * _MIB,
        source_replacement_supported=False,
        source_replacement_reason=(
            "Olympus OIF/OIB sources are read-only; writing or source replacement "
            "is unsupported."
        ),
        prefer_model_decode_contract=True,
    ),
    VolumeSourceFormat(
        key="lif",
        suffixes=(".lif",),
        reader_mode="leica",
        container_formats=("leica-lif",),
        reader_dependencies=("liffile",),
        decode_memory_factor=4.0,
        decode_memory_floor_bytes=384 * _MIB,
        source_replacement_supported=False,
        source_replacement_reason=(
            "Leica LIF sources are read-only; import or Save As to a supported "
            "MADI3D output format instead of replacing the source file."
        ),
        prefer_model_decode_contract=True,
    ),
    VolumeSourceFormat(
        key="h5j",
        suffixes=(".h5j",),
        reader_mode="h5j",
        container_formats=("h5j",),
        reader_dependencies=("h5py",),
        decode_memory_factor=16.0,
        decode_memory_floor_bytes=1536 * _MIB,
        source_replacement_checks_channels=True,
    ),
    VolumeSourceFormat(
        key="nifti",
        suffixes=(".nii",),
        reader_mode="nifti",
        container_formats=("nifti",),
        reader_dependencies=("nibabel",),
        decode_memory_factor=3.0,
        decode_memory_floor_bytes=384 * _MIB,
    ),
    VolumeSourceFormat(
        key="nifti-gzip",
        suffixes=(".nii.gz",),
        reader_mode="nifti",
        container_formats=("nifti",),
        reader_dependencies=("nibabel",),
        decode_memory_factor=12.0,
        decode_memory_floor_bytes=768 * _MIB,
    ),
)

SUPPORTED_VOLUME_SUFFIXES = tuple(
    suffix for source_format in VOLUME_SOURCE_FORMATS for suffix in source_format.suffixes
)


def volume_source_format(
    path: os.PathLike[str] | str | None,
) -> VolumeSourceFormat | None:
    lower = os.fspath(path).lower() if path not in (None, "") else ""
    for source_format in sorted(
        VOLUME_SOURCE_FORMATS,
        key=lambda value: max(len(suffix) for suffix in value.suffixes),
        reverse=True,
    ):
        if lower.endswith(source_format.suffixes):
            return source_format
    return None


def volume_source_format_for_container(
    container_format: str | None,
) -> VolumeSourceFormat | None:
    normalized = str(container_format or "").strip().lower()
    if not normalized:
        return None
    for source_format in VOLUME_SOURCE_FORMATS:
        if normalized in source_format.container_formats:
            if normalized == "zeiss-lsm" and source_format.key == "tiff":
                continue
            return source_format
    return None


def is_supported_volume_path(path: os.PathLike[str] | str | None) -> bool:
    return volume_source_format(path) is not None


def volume_path_suffix(path: os.PathLike[str] | str | None) -> str:
    lower = os.fspath(path).lower() if path not in (None, "") else ""
    source_format = volume_source_format(lower)
    if source_format is not None:
        return next(
            suffix
            for suffix in sorted(source_format.suffixes, key=len, reverse=True)
            if lower.endswith(suffix)
        )
    return os.path.splitext(lower)[1]


def volume_file_globs() -> tuple[str, ...]:
    return tuple(f"*{suffix}" for suffix in SUPPORTED_VOLUME_SUFFIXES)


def volume_file_dialog_filter(label: str = "Volumes") -> str:
    return f"{str(label or 'Volumes')} ({' '.join(volume_file_globs())})"


def source_error_container_format(path: os.PathLike[str] | str | None) -> str:
    source_format = volume_source_format(path)
    if source_format is not None:
        return source_format.container_formats[0]
    suffix = volume_path_suffix(path).lstrip(".")
    return suffix or "unknown"


def volume_reader_mode(path: os.PathLike[str] | str | None) -> str:
    source_format = volume_source_format(path)
    return source_format.reader_mode if source_format is not None else ""


def volume_reader_dependency_modules() -> tuple[str, ...]:
    modules = []
    for source_format in VOLUME_SOURCE_FORMATS:
        for module in source_format.reader_dependencies:
            if module not in modules:
                modules.append(module)
    return tuple(modules)


def volume_prefers_model_decode_contract(
    path: os.PathLike[str] | str | None,
) -> bool:
    source_format = volume_source_format(path)
    return bool(
        source_format is not None and source_format.prefer_model_decode_contract
    )


def estimated_volume_decode_bytes(
    path: os.PathLike[str] | str,
    *,
    file_size_bytes: int | None = None,
) -> int:
    if file_size_bytes is None:
        try:
            file_size_bytes = os.path.getsize(path)
        except OSError:
            file_size_bytes = 256 * _MIB
    file_size_bytes = max(1, int(file_size_bytes))
    source_format = volume_source_format(path)
    if source_format is None:
        factor, floor = 6.0, 512 * _MIB
    else:
        factor = source_format.decode_memory_factor
        floor = source_format.decode_memory_floor_bytes
    return max(int(floor), int(file_size_bytes * factor))


def source_replacement_format(
    path: os.PathLike[str] | str,
    *,
    container_format: str | None = None,
) -> VolumeSourceFormat | None:
    persisted = volume_source_format_for_container(container_format)
    path_format = volume_source_format(path)
    if persisted is not None and not persisted.source_replacement_supported:
        return persisted
    normalized = str(container_format or "").strip().lower()
    if path_format is not None and (
        not normalized or normalized in path_format.container_formats
    ):
        return path_format
    return persisted or path_format


def recovery_path_supports_container(
    path: os.PathLike[str] | str,
    container_format: str | None,
) -> bool:
    expected = str(container_format or "").strip().lower()
    if not expected:
        return is_supported_volume_path(path)
    candidate = volume_source_format(path)
    return candidate is not None and expected in candidate.container_formats


def tiff_container_format(
    tiff_file: Any,
    *,
    ome_pixels: Any = None,
    imagej: Any = None,
) -> str:
    if bool(getattr(tiff_file, "is_lsm", False)):
        return "zeiss-lsm"
    if ome_pixels or (
        ome_pixels is None and bool(getattr(tiff_file, "ome_metadata", None))
    ):
        return "ome-tiff"
    if imagej or (
        imagej is None and bool(getattr(tiff_file, "imagej_metadata", None))
    ):
        return "imagej-tiff"
    return "tiff"


__all__ = [
    "SUPPORTED_VOLUME_SUFFIXES",
    "VOLUME_SOURCE_FORMATS",
    "VolumeSourceFormat",
    "SourceBundleError",
    "estimated_volume_decode_bytes",
    "is_supported_volume_path",
    "oif_companion_source",
    "recovery_path_supports_container",
    "source_error_container_format",
    "source_replacement_format",
    "tiff_container_format",
    "volume_file_dialog_filter",
    "volume_file_globs",
    "volume_path_suffix",
    "volume_prefers_model_decode_contract",
    "volume_reader_dependency_modules",
    "volume_reader_mode",
    "volume_source_format",
    "volume_source_format_for_container",
]


def nifti_time_unit(value):
    unit = str(value or "").strip().lower()
    aliases = {
        "s": "sec", "second": "sec", "seconds": "sec", "sec": "sec",
        "ms": "msec", "millisecond": "msec", "milliseconds": "msec", "msec": "msec",
        "us": "usec", "µs": "usec", "microsecond": "usec", "microseconds": "usec", "usec": "usec",
        "hz": "hz", "ppm": "ppm", "rad/s": "rads", "rads": "rads",
        "frame": "unknown", "frames": "unknown",
    }
    return aliases.get(unit, "unknown")
