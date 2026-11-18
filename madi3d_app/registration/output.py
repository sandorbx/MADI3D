# -*- coding: utf-8 -*-
"""Durable result-bundle helpers for MADI3D registration.

This module is independent from widgets, rendering, ITK, and CMTK execution.
It owns output-policy validation, Reformat resource estimates, deterministic result-folder layout,
manifest bookkeeping, small QC summaries, and ordinary file copies.  Image and
mesh serialization stays with MADI3D's central Save As writers.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from madi3d_app.io_utils import atomic_replace, check_cancelled, check_resources, copy_file
from madi3d_app.operation_status import execution_succeeded
from madi3d_app.volume.h5j_quantization import h5j_quantization_scratch_bytes

_LOG = logging.getLogger(__name__)


RESULT_LAYOUTS = {"separate", "common"}
VOLUME_FORMAT_EXTENSIONS = {
    "nrrd": ".nrrd",
    "nifti": ".nii.gz",
    "tiff": ".tif",
    "h5j": ".h5j",
}
MESH_FORMAT_EXTENSIONS = {
    "vtp": ".vtp",
    "obj": ".obj",
    "vtk": ".vtk",
}

# Keep registration paths usable by Windows APIs and third-party tools that do
# not reliably support extended-length paths. Count UTF-16 code units because
# that is the relevant Windows representation even when tests run elsewhere.
WINDOWS_SAFE_PATH_BUDGET = 240
WINDOWS_FILESYSTEM_COMPONENT_BUDGET = 255
_PATH_HASH_LENGTH = 12
_PATH_HASH_SEPARATOR = "--"
_MIN_COMPACTED_COMPONENT_UNITS = (
    1 + len(_PATH_HASH_SEPARATOR) + _PATH_HASH_LENGTH
)
_TEMPORARY_SUFFIX_RESERVE = ".partial"
_TRANSFORM_EXTENSION = ".registration.json"
# Transform export replaces only the final .json suffix. CMTK can write a
# directory archive whose registration.gz is deeper than the JSON sidecar.
_CMTK_TRANSFORM_TAIL = ".registration.cmtk/warp.xform/registration.gz"
_SEPARATE_CATEGORY_RESERVE = max(
    len("transforms"),
    len("reformatted"),
    len("qc"),
    len("logs"),
    _MIN_COMPACTED_COMPONENT_UNITS,
)

DEFAULT_RESULT_OUTPUT = {
    "enabled": False,
    "root_dir": "",
    "layout": "separate",
    "write_transforms": True,
    "write_reformatted": True,
    "write_qc_summary": True,
    "write_qc_volumes": False,
    "write_log": True,
    "volume_format": "nrrd",
    "mesh_format": "vtp",
}


class RegistrationOutputError(RuntimeError):
    pass


def prepare_reformatted_mesh(payload, *, metadata=None, prepare_scene=True, cancel_check=None):
    """Prepare detached scene geometry and atomically publish optional world files."""
    import numpy as np
    from madi3d_app.mesh.processing import reformatted_polydata
    from madi3d_app.mesh.export import write_polydata
    from madi3d_app.volume.geometry import index_to_world_points
    from madi3d_app.io_utils import publish_file_pair
    from madi3d_app.project.scientific_records import object_metadata_sidecar_path, write_object_metadata_sidecar

    captured = payload["mesh_input"]
    points = np.asarray(payload["points_local"], dtype=np.float64).reshape(-1, 3)
    reference = np.asarray(payload["reference_actor_matrix"], dtype=float)
    world_matrix = np.asarray(payload["mesh_world_matrix"], dtype=float)
    if prepare_scene:
        prepared = reformatted_polydata(captured["polydata"], points,
            np.linalg.inv(reference) @ world_matrix, cancel_check=cancel_check)
        from madi3d_app.mesh.ply import is_ply_polydata
        reflected = np.linalg.det(reference[:3, :3]) < 0 and not is_ply_polydata(prepared)
        if reflected:
            import vtk
            from madi3d_app.mesh.processing import update_algorithm
            # Match MeshData's render-only reversal after canonical normals.
            reverse = vtk.vtkReverseSense()
            reverse.SetInputData(prepared)
            reverse.ReverseCellsOn()
            reverse.ReverseNormalsOff()
            update_algorithm(reverse, cancel_check=cancel_check)
            prepared = vtk.vtkPolyData()
            prepared.ShallowCopy(reverse.GetOutput())
        payload["prepared_polydata"] = prepared
        payload["prepared_winding_reversed"] = bool(reflected)
    check_cancelled(cancel_check)
    output_path = payload.get("disk_output_path")
    if not output_path:
        return []
    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = partial_output_path(final_path)
    if temporary_path.exists() or temporary_path.is_symlink():
        raise FileExistsError(f"Reformat staging file already exists: {temporary_path}")
    try:
        output_polydata = reformatted_polydata(captured["polydata"], index_to_world_points(points, reference),
            world_matrix, cancel_check=cancel_check)
        if not write_polydata(output_polydata, str(temporary_path), cancel_check=cancel_check):
            raise RuntimeError(f"Could not write reformatted mesh: {final_path}")
        sidecar = write_object_metadata_sidecar(temporary_path, metadata, cancel_check=cancel_check)
        publish_file_pair(temporary_path, final_path, sidecar,
            object_metadata_sidecar_path(final_path), fail_if_exists=True, cancel_check=cancel_check)
        payload["written_path"] = str(final_path)
        return [final_path, Path(object_metadata_sidecar_path(final_path))]
    finally:
        temporary_path.unlink(missing_ok=True)
        Path(object_metadata_sidecar_path(temporary_path)).unlink(missing_ok=True)


def estimate_reformat_resources(tasks, options):
    """Return the existing Reformat RAM and per-device disk peak estimate."""
    dtype_mode = str((options or {}).get("dtype", "preserve"))
    retained_bytes, peak_frame_bytes, captured_bytes = 0, 0, 0
    published, disk_peaks, disk_locations = {}, {}, {}
    unknown_scopes = set()

    def disk_location(path):
        folder = Path(path).absolute()
        while not folder.exists() and folder != folder.parent:
            folder = folder.parent
        device = folder.stat().st_dev
        disk_locations.setdefault(device, folder)
        return device

    temp_device = None
    for task in tasks:
        unknown_scopes.update(("main", "host"))
        if str(task.get("kind") or "volume").lower() != "volume":
            continue
        data = task.get("data")
        if getattr(data, "ndim", 0) not in (3, 4):
            continue  # The execution owner reports the actual input failure.
        frames = int(data.shape[0]) if data.ndim == 4 else 1
        if not task.get("_borrowed_scene_inputs"):
            captured_bytes += int(data.nbytes)
        geometry = dict(task.get("reference_geometry") or {})
        dims = geometry.get("dims_xyz", geometry.get("dimensions"))
        if dims is None:
            dims = data.shape[-3:]
        voxels = math.prod(int(n) for n in dims)
        itemsize = 4 if dtype_mode == "float32" else int(data.dtype.itemsize)
        frame_bytes = voxels * itemsize
        output_bytes = frames * frame_bytes
        disk_path = str(task.get("disk_output_path") or "").strip()
        disk_only = bool(disk_path and not task.get("emit_data", True))
        scratch_bytes = output_bytes if disk_only and frames > 1 else 0
        cmtk_disk_bytes = 0
        if str(task.get("engine") or "itk").lower() == "cmtk":
            input_frame_bytes = (
                math.prod(int(n) for n in data.shape[-3:])
                * int(data.dtype.itemsize)
            )
            # CMTK stages one raw floating NRRD and one resampled output NRRD
            # in its per-frame workspace. Native scratch beyond those files is
            # unknown; this is disk demand only.
            cmtk_disk_bytes = input_frame_bytes + frame_bytes
        if not disk_only:
            retained_bytes += output_bytes
        h5j_bytes = 0
        quantization_bytes = 0
        if disk_path and str(task.get("disk_output_format") or "").lower() == "h5j":
            # Quantization uses the bounded slabs documented by its writer. The
            # encoded payload and native FFmpeg demand remain unquantified.
            quantization_dtype = (
                "float32" if dtype_mode == "float32" else data.dtype
            )
            quantization_bytes = h5j_quantization_scratch_bytes(
                voxels, quantization_dtype
            )
            h5j_bytes = voxels * 2
        # Multi-frame execution retains the full output while a separate
        # resampled frame is copied into it. Disk-only output also holds the
        # active frame while writing it to its memmap. A single in-memory frame
        # aliases the returned output array and needs no second frame allowance.
        stage_bytes = (
            frame_bytes if disk_only or frames > 1 else 0
        ) + quantization_bytes
        peak_frame_bytes = max(peak_frame_bytes, stage_bytes)
        output_device = disk_location(Path(disk_path).parent) if disk_path else None
        # Publication renames the one serialized file; it never adds a copy.
        # Completed files stay until the batch succeeds or is rolled back.
        for encoding in (False, True):
            transient = {}
            temp_bytes = scratch_bytes + (
                h5j_bytes if encoding else cmtk_disk_bytes
            )
            if temp_bytes:
                if temp_device is None:
                    temp_device = disk_location(tempfile.gettempdir())
                transient[temp_device] = temp_bytes
            if encoding and output_device is not None:
                transient[output_device] = transient.get(output_device, 0) + output_bytes
            for device in published.keys() | transient.keys():
                demand = published.get(device, 0) + transient.get(device, 0)
                disk_peaks[device] = max(disk_peaks.get(device, 0), demand)
        if output_device is not None:
            published[output_device] = published.get(output_device, 0) + output_bytes
    return {
        "captured_bytes": int(captured_bytes),
        "retained_bytes": int(retained_bytes),
        "peak_frame_bytes": int(peak_frame_bytes),
        "memory_bytes": int(retained_bytes + peak_frame_bytes),
        "unknown_scopes": tuple(sorted(unknown_scopes)),
        "disk_peaks": tuple(
            (disk_locations[device], int(byte_count))
            for device, byte_count in disk_peaks.items() if byte_count
        ),
    }


def check_reformat_resources(tasks, options, *, memory_reserved=False):
    """Check captured output RAM and sequential staging peaks before dispatch."""
    estimate = estimate_reformat_resources(tasks, options)
    if estimate["memory_bytes"] and not memory_reserved:
        check_resources(memory_bytes=estimate["memory_bytes"])
    for directory, byte_count in estimate["disk_peaks"]:
        check_resources(disk_bytes=byte_count, directory=directory)


def safe_output_stem(value: Any, fallback: str = "registration") -> str:
    text = str(value or "").strip()
    text = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", text)
    text = re.sub(r"\s+", "_", text).strip(" ._")
    return text or str(fallback or "registration")


_REFORMAT_FORBIDDEN_NAME_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_DEVICE_NAME = re.compile(
    r"^(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)", re.IGNORECASE
)


def normalize_reformat_name(name: str) -> str:
    """Make one portable Reformat filename component without compacting it.

    Valid spaces, case, underscores, and Unicode are retained. Invalid Windows
    filename/control characters become underscores; trailing spaces and dots
    are removed. The caller supplies a meaningful Volume/Mesh fallback for a
    blank source Name before adding any configured suffix.
    """
    text = "" if name is None else str(name)
    text = "".join(
        "_" if character in _REFORMAT_FORBIDDEN_NAME_CHARS
        or unicodedata.category(character) in {"Cc", "Cs"}
        else character
        for character in text
    ).rstrip(" .")
    if not text:
        return "Reformatted"
    if _WINDOWS_RESERVED_DEVICE_NAME.match(text):
        text = "_" + text
    return text


def reformat_name_key(name: str) -> str:
    """Return the portable NFC/case-insensitive Reformat reservation key."""
    return unicodedata.normalize("NFC", normalize_reformat_name(name)).casefold()


def allocate_reformat_name(name: str, occupied_names: Iterable[str]) -> str:
    """Allocate a normalized Reformat name, adding ``_2``, ``_3``, etc."""
    candidate = normalize_reformat_name(name)
    occupied = {reformat_name_key(value) for value in occupied_names}
    if reformat_name_key(candidate) not in occupied:
        return candidate
    index = 2
    while True:
        allocated = f"{candidate}_{index}"
        if reformat_name_key(allocated) not in occupied:
            return allocated
        index += 1


def _windows_text_units(value: Any) -> int:
    return len(str(value).encode("utf-16-le")) // 2


def _truncate_portable_component(
    value: str, max_units: int, max_bytes: int = WINDOWS_FILESYSTEM_COMPONENT_BUDGET
) -> str:
    """Keep a prefix within both Windows UTF-16 and portable UTF-8 limits."""
    kept = []
    units = 0
    byte_count = 0
    for character in str(value):
        character_units = _windows_text_units(character)
        character_bytes = len(character.encode("utf-8"))
        if units + character_units > int(max_units) or byte_count + character_bytes > int(max_bytes):
            break
        kept.append(character)
        units += character_units
        byte_count += character_bytes
    return "".join(kept)


def bounded_output_component(
    value: Any,
    max_units: int,
    fallback: str = "registration",
    *,
    hash_source: Any | None = None,
) -> str:
    """Return a sanitized component bounded for deterministic Windows paths."""
    original = str(value or "")
    sanitized = safe_output_stem(original, fallback)
    max_units = min(int(max_units), WINDOWS_FILESYSTEM_COMPONENT_BUDGET)
    if (
        _windows_text_units(sanitized) <= max_units
        and len(sanitized.encode("utf-8")) <= WINDOWS_FILESYSTEM_COMPONENT_BUDGET
    ):
        return sanitized

    digest_source = original if hash_source is None else str(hash_source)
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[
        :_PATH_HASH_LENGTH
    ]
    hashed_suffix = _PATH_HASH_SEPARATOR + digest
    prefix_units = max_units - _windows_text_units(hashed_suffix)
    prefix_bytes = WINDOWS_FILESYSTEM_COMPONENT_BUDGET - len(hashed_suffix.encode("utf-8"))
    if prefix_units < 1:
        raise RegistrationOutputError(
            "The available filesystem-component budget is too small for a "
            "collision-safe registration output name."
        )
    prefix = _truncate_portable_component(sanitized, prefix_units, prefix_bytes).rstrip(" ._")
    if not prefix:
        prefix = _truncate_portable_component(safe_output_stem(fallback), prefix_units, prefix_bytes)
    if not prefix:
        raise RegistrationOutputError(
            "The available filesystem-component budget is too small for a "
            "readable registration output name."
        )
    return prefix + hashed_suffix


def _bounded_label_component(value: Any, max_units: int, *, hash_source: Any) -> str:
    """Bound a portable folder label while retaining valid spaces and case."""
    sanitized = normalize_reformat_name(str(value or "registration"))
    max_units = min(int(max_units), WINDOWS_FILESYSTEM_COMPONENT_BUDGET)
    if (
        _windows_text_units(sanitized) <= max_units
        and len(sanitized.encode("utf-8")) <= WINDOWS_FILESYSTEM_COMPONENT_BUDGET
    ):
        return sanitized
    digest = hashlib.sha256(str(hash_source).encode("utf-8")).hexdigest()[
        :_PATH_HASH_LENGTH
    ]
    suffix = _PATH_HASH_SEPARATOR + digest
    prefix_units = max_units - _windows_text_units(suffix)
    prefix_bytes = WINDOWS_FILESYSTEM_COMPONENT_BUDGET - len(suffix.encode("utf-8"))
    if prefix_units < 1:
        raise RegistrationOutputError(
            "The available filesystem-component budget is too small for a "
            "collision-safe registration output name."
        )
    prefix = _truncate_portable_component(sanitized, prefix_units, prefix_bytes).rstrip(" .")
    if not prefix:
        raise RegistrationOutputError(
            "The available filesystem-component budget is too small for a "
            "readable registration output name."
        )
    return prefix + suffix


def _short_result_root_error(root: Path) -> RegistrationOutputError:
    return RegistrationOutputError(
        "The registration result directory is too long to create Windows-compatible "
        f"result paths within the {WINDOWS_SAFE_PATH_BUDGET}-character budget. "
        f"Choose a shorter registration result directory. Current directory: {root}"
    )


def _artifact_tail_reserve(policy: dict) -> int:
    ordinary_tail = _windows_text_units(_TRANSFORM_EXTENSION + _TEMPORARY_SUFFIX_RESERVE)
    if policy["write_transforms"]:
        return max(ordinary_tail, _windows_text_units(_CMTK_TRANSFORM_TAIL))
    return ordinary_tail


def _bundle_component_capacity(parent: Path, policy: dict) -> int:
    """Budget for a run or result-group name beneath its parent directory."""
    trailing_units = (
        1
        + _MIN_COMPACTED_COMPONENT_UNITS
        + _artifact_tail_reserve(policy)
    )
    if policy["layout"] == "separate":
        trailing_units += 1 + _SEPARATE_CATEGORY_RESERVE
    return min(
        WINDOWS_FILESYSTEM_COMPONENT_BUDGET,
        WINDOWS_SAFE_PATH_BUDGET
        - _windows_text_units(parent)
        - 1
        - trailing_units,
    )


def _require_bundle_root_capacity(root: Path, policy: dict) -> int:
    capacity = _bundle_component_capacity(root, policy)
    if capacity < _MIN_COMPACTED_COMPONENT_UNITS:
        raise _short_result_root_error(root)
    return capacity


def _assert_windows_safe_path(path: Path, *, extra_suffix: str = "") -> None:
    resolved = Path(path).resolve()
    if (
        _windows_text_units(resolved) + _windows_text_units(extra_suffix)
        > WINDOWS_SAFE_PATH_BUDGET
    ):
        raise _short_result_root_error(resolved.parent)


def _assert_reformat_path(path: Path, name: str) -> None:
    """Validate a complete Reformat path without shortening its basename."""
    try:
        _assert_windows_safe_path(path)
        for component in path.parts[1:] if path.anchor else path.parts:
            utf16_units = _windows_text_units(component)
            utf8_bytes = len(component.encode("utf-8"))
            if (
                utf16_units > WINDOWS_FILESYSTEM_COMPONENT_BUDGET
                or utf8_bytes > WINDOWS_FILESYSTEM_COMPONENT_BUDGET
            ):
                raise RegistrationOutputError(
                    "the filename component exceeds the portable 255-unit/byte limit"
                )
    except (RegistrationOutputError, UnicodeEncodeError) as exc:
        raise RegistrationOutputError(
            f"Cannot publish Reformat output {name!r}: {exc}. Choose a shorter "
            f"source Name or output root. Path: {path}"
        ) from exc


def normalize_result_output_policy(
    value: dict | None,
    *,
    force_enabled: bool = False,
    require_root: bool = False,
    default_root: bool = False,
) -> dict:
    """Normalize output choices, optionally resolving a blank root to durable user data."""
    raw = dict(DEFAULT_RESULT_OUTPUT)
    raw.update(dict(value or {}))
    if default_root and not str(raw.get("root_dir") or "").strip():
        from madi3d_storage import data_dir
        raw["root_dir"] = str(data_dir() / "Registration")
    policy = {
        "enabled": bool(raw.get("enabled") or force_enabled),
        "root_dir": os.path.abspath(os.path.expanduser(str(raw.get("root_dir") or "").strip()))
        if str(raw.get("root_dir") or "").strip() else "",
        "layout": str(raw.get("layout") or "separate").lower(),
        "write_transforms": bool(raw.get("write_transforms", True)),
        "write_reformatted": bool(raw.get("write_reformatted", True)),
        "write_qc_summary": bool(raw.get("write_qc_summary", True)),
        "write_qc_volumes": bool(raw.get("write_qc_volumes", False)),
        "write_log": bool(raw.get("write_log", True)),
        "volume_format": str(raw.get("volume_format") or "nrrd").lower(),
        "mesh_format": str(raw.get("mesh_format") or "vtp").lower(),
    }
    if policy["layout"] not in RESULT_LAYOUTS:
        raise RegistrationOutputError(
            f"Unknown registration result-folder layout: {policy['layout']!r}."
        )
    if policy["volume_format"] not in VOLUME_FORMAT_EXTENSIONS:
        raise RegistrationOutputError(
            f"Unsupported registration volume output format: {policy['volume_format']!r}."
        )
    if policy["mesh_format"] not in MESH_FORMAT_EXTENSIONS:
        raise RegistrationOutputError(
            f"Unsupported registration mesh output format: {policy['mesh_format']!r}."
        )
    if policy["enabled"] and not any(
        policy[key]
        for key in (
            "write_transforms", "write_reformatted", "write_qc_summary",
            "write_qc_volumes", "write_log",
        )
    ):
        raise RegistrationOutputError(
            "Result writing is enabled, but no result artifact is selected."
        )
    if (require_root or policy["enabled"]) and not policy["root_dir"]:
        raise RegistrationOutputError("Choose a registration result output folder first.")
    return policy


def ensure_output_root_writable(policy: dict) -> Path:
    """Create and verify the configured output root without creating a run bundle."""
    normalized = normalize_result_output_policy(policy, require_root=True)
    root = Path(normalized["root_dir"]).resolve()
    _require_bundle_root_capacity(root, normalized)
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise RegistrationOutputError(f"Registration output path is not a directory: {root}")
    probe = None
    try:
        handle = tempfile.NamedTemporaryFile(
            prefix=".madi3d-registration-write-", suffix=".tmp",
            dir=root, delete=False,
        )
        probe = Path(handle.name)
        handle.write(b"MADI3D registration output write test\n")
        handle.close()
    except Exception as exc:
        raise RegistrationOutputError(
            f"Registration output folder is not writable: {root}\n{exc}"
        ) from exc
    finally:
        if probe is not None:
            try:
                probe.unlink(missing_ok=True)
            except Exception:
                pass
    return root


def partial_output_path(path: os.PathLike | str) -> Path:
    """Return a same-format temporary path suitable for atomic replacement."""
    path = Path(path)
    name = path.name
    if name.lower().endswith(".nii.gz"):
        temporary = path.with_name(name[:-7] + ".partial.nii.gz")
    else:
        temporary = path.with_name(path.stem + ".partial" + path.suffix)
    _assert_windows_safe_path(temporary)
    return temporary


def volume_extension(output_format: str) -> str:
    try:
        return VOLUME_FORMAT_EXTENSIONS[str(output_format).lower()]
    except KeyError as exc:
        raise RegistrationOutputError(
            f"Unsupported registration volume output format: {output_format!r}."
        ) from exc


def mesh_extension(output_format: str) -> str:
    try:
        return MESH_FORMAT_EXTENSIONS[str(output_format).lower()]
    except KeyError as exc:
        raise RegistrationOutputError(
            f"Unsupported registration mesh output format: {output_format!r}."
        ) from exc


def _reformat_extension(kind: str, output_format: str) -> str:
    kind = str(kind or "").lower()
    if kind == "volume":
        return volume_extension(output_format)
    if kind == "mesh":
        return mesh_extension(output_format)
    raise RegistrationOutputError(
        f"Unsupported Reformat artifact kind: {kind!r}; expected 'volume' or 'mesh'."
    )


def _validate_reformat_paths(path: Path, name: str) -> None:
    from madi3d_app.project.scientific_records import object_metadata_sidecar_path

    if not name or name != normalize_reformat_name(name):
        raise RegistrationOutputError(
            f"Reformat output name {name!r} is not normalized. Normalize the "
            "source Name before starting Reformat."
        )
    try:
        temporary_path = partial_output_path(path)
    except RegistrationOutputError as exc:
        raise RegistrationOutputError(
            f"Cannot publish Reformat output {name!r}: {exc}. Choose a shorter "
            f"source Name or output root. Path: {path}"
        ) from exc
    for candidate in (
        path,
        temporary_path,
        Path(object_metadata_sidecar_path(path)),
        Path(object_metadata_sidecar_path(temporary_path)),
        Path(object_metadata_sidecar_path(path) + ".publishing"),
    ):
        _assert_reformat_path(candidate, name)


def registration_qc_summary(chain_payload: dict) -> dict:
    chain = copy.deepcopy(dict(chain_payload or {}))
    summaries = []
    warnings = []
    for stage in chain.get("stages") or []:
        stage = dict(stage or {})
        details = dict(stage.get("details") or {})
        summary = {
            "name": str(stage.get("name") or "Stage"),
            "kind": str(stage.get("kind") or "unknown"),
            "execution_status": str(stage.get("execution_status") or "pending"),
            "qc_status": str(stage.get("qc_status") or "not-evaluated"),
            "user_decision": str(stage.get("user_decision") or "unapplied"),
            "metric_value": stage.get("metric_value"),
            "ncc": stage.get("ncc"),
            "iterations": int(stage.get("iterations") or 0),
            "stop_condition": str(stage.get("stop_condition") or ""),
        }
        for key in (
            "support_overlap_fraction",
            "requested_support_overlap_fraction",
            "automatic_center_fallback",
            "linear_sanity",
            "deformation_qc",
            "qc_thresholds",
            "qc_evidence",
            "qc_failures",
            "quality_warnings",
            "deterministic_similarity_before",
            "deterministic_similarity_after",
        ):
            if key in details:
                summary[key] = copy.deepcopy(details.get(key))
        if stage.get("qc"):
            summary["qc"] = copy.deepcopy(stage["qc"])
        execution_error = str(details.get("execution_error") or details.get("artifact_error") or "").strip()
        if execution_error:
            summary["execution_error"] = execution_error
            warnings.append(f"{summary['name']}: {execution_error}")
        for item in details.get("qc_failures") or []:
            text = str(item or "").strip()
            if text:
                warnings.append(f"{summary['name']}: {text}")
        for item in details.get("quality_warnings") or []:
            text = str(item or "").strip()
            if text:
                warnings.append(f"{summary['name']}: {text}")
        deformation_qc = dict(details.get("deformation_qc") or {})
        for item in deformation_qc.get("warnings") or []:
            text = str(item or "").strip()
            if text:
                warnings.append(text)
        summaries.append(summary)
    # Preserve order while removing repeated warnings from stage/model mirrors.
    unique_warnings = list(dict.fromkeys(warnings))
    return {
        "format": "MADI3D Registration QC Summary",
        "format_version": 2,
        "registration_id": str(chain.get("registration_id") or ""),
        "source_name": str(chain.get("source_name") or ""),
        "target_name": str(chain.get("target_name") or ""),
        "created": str(chain.get("created") or ""),
        "has_deformation": bool(chain.get("has_deformation", False)),
        "execution_status": str(chain.get("execution_status") or "pending"),
        "qc_status": str(chain.get("qc_status") or "not-evaluated"),
        "qc_thresholds": copy.deepcopy(dict(chain.get("settings") or {}).get("qc_thresholds") or {}),
        "registration_algorithm_version": chain.get("registration_algorithm_version"),
        "user_decision": str(chain.get("user_decision") or "unapplied"),
        "stage_path": [
            str(stage.get("kind") or "unknown")
            for stage in chain.get("stages") or []
            if str(stage.get("execution_status") or "pending") == "succeeded"
        ],
        "stages": summaries,
        "warnings": unique_warnings,
    }


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    _assert_windows_safe_path(path)
    _assert_windows_safe_path(temporary)
    try:
        temporary.write_bytes(data)
        atomic_replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            _LOG.warning("Could not remove unpublished staging file %s", temporary, exc_info=True)
        raise


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))


@dataclass(frozen=True)
class PublishedTemplateImage:
    image: object
    path: Path
    sha256: str
    dtype: str
    working_grid: object
    generation: str
    file_signature: tuple[int, int]


@dataclass(frozen=True)
class PublishedTemplateRound:
    """Verified physical publication; canonical adoption is a separate step."""

    record: object
    directory: Path
    images: tuple[PublishedTemplateImage, ...]
    native_files: dict


def publish_template_round(bundle, record, *, retain_auxiliary_images=True, cancel_check=None):
    """Publish a complete detached round atomically, without touching the model.

    Native images use spacing-only CMTK coordinates. Re-serialize each scalar
    through the volume writer with its authoritative world lattice before
    assigning independent file backing. Transform members remain byte exact.
    """
    from dataclasses import replace
    import numpy as np
    import nrrd
    from madi3d_app.integrations.cmtk.registration import read_nrrd_zyx
    from madi3d_app.volume.export import write_baked_nrrd, registration_output_grid_records
    from madi3d_app.volume.nrrd_stream import compare_nrrd_payload

    check_cancelled(cancel_check)
    destination = bundle.run_dir / f"round-{record.index:04d}"
    _assert_windows_safe_path(destination / "volume-0000.nrrd", extra_suffix=".partial")
    if destination.exists():
        raise RegistrationOutputError("This round is already published; retry its retained finalization.")
    staging = Path(tempfile.mkdtemp(prefix=".round-", dir=bundle.run_dir))
    def digest(path):
        result = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                check_cancelled(cancel_check)
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                result.update(chunk)
        return result.hexdigest()
    try:
        for artifact in record.artifacts:
            check_cancelled(cancel_check)
            if artifact.path.stat().st_size != artifact.size_bytes or digest(artifact.path) != artifact.sha256:
                raise RegistrationOutputError("A completed template artifact changed before publication.")
        native_files, relocated = {}, {}
        for artifact in record.artifacts:
            check_cancelled(cancel_check)
            if artifact.role != "mapping-transform":
                continue
            artifact.path.resolve().relative_to(record.directory.resolve())
            relative = f".madi3d-transform-{artifact.sha256}.bin"
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_file(artifact.path, target, cancel_check=cancel_check)
            if digest(target) != artifact.sha256:
                raise RegistrationOutputError("Copied template transform checksum differs.")
            native_files[relative] = {"sha256": artifact.sha256, "size_bytes": artifact.size_bytes,
                                      "resource_key": "registration_transform:" + artifact.sha256}
            relocated[artifact.path] = destination / relative
        images = []
        selected_images = (record.registration, *(record.outputs if retain_auxiliary_images else ()))
        for index, image in enumerate(selected_images):
            check_cancelled(cancel_check)
            array, _ = read_nrrd_zyx(image.image)
            check_cancelled(cancel_check)
            lattice = image.lattice
            if array.shape != lattice.dims_xyz[::-1] or array.dtype.kind not in "uif":
                raise RegistrationOutputError("Template output is not a scalar on its declared grid.")
            # Numerical working geometry is authoritative; units alone do not
            # upgrade source calibration to verified physical evidence.
            working, _ = registration_output_grid_records(dimensions=lattice.dims_xyz,
                affine=lattice.index_to_world, space_units=lattice.units,
                coordinate_space_id="template:" + bundle.run_id, source_time={"count": 1},
                registration_provenance={"target_grid_state": "unresolved", "target_working_grid": {
                    "coordinate_mode": "working-grid", "geometry_basis": "partially-assumed"},
                    "warnings": ["Population output retains its inputs' calibration uncertainty."]})
            name = f"volume-{index:04d}.nrrd"
            path = staging / name
            from madi3d_app.volume.geometry import working_grid_geometry_status
            generation = f"template:{bundle.run_id}:{record.index}:{index}"
            write_baked_nrrd(str(path), array[None], {"working_grid": working.to_dict(),
                "grid_state": working.physical_grid_state, "geometry_status": working_grid_geometry_status(working)},
                {"affine": lattice.index_to_world, "dims": lattice.dims_xyz,
                 "coordinate_space_id": working.source_coordinate_space_id, "space_units": lattice.units},
                {"portable_companion": True, "serialization_generation": generation,
                 "allow_unknown_space_units": lattice.units is None}, cancel_check=cancel_check)
            check_cancelled(cancel_check)
            header = nrrd.read_header(str(path))
            check_cancelled(cancel_check)
            if (not compare_nrrd_payload(str(path), array, cancel_check=cancel_check)
                    or not np.allclose(header["space origin"], lattice.origin, rtol=0, atol=1e-12)
                    or not np.allclose(np.asarray(header["space directions"]).T,
                                       lattice.index_to_world[:3, :3], rtol=0, atol=1e-12)):
                raise RegistrationOutputError("Template scalar backing verification failed.")
            info = path.stat()
            images.append(PublishedTemplateImage(image, destination / name, digest(path), str(array.dtype), working, generation,
                                                 (info.st_size, info.st_mtime_ns)))
        mappings = tuple(replace(mapping, native_forward=tuple((relocated[path], inverse)
            for path, inverse in mapping.native_forward)) for mapping in record.mappings)
        durable = replace(record, directory=destination, registration=replace(record.registration,
            image=images[0].path), outputs=tuple(replace(item.image, image=item.path) for item in images[1:]),
            mappings=mappings, artifacts=())
        _atomic_json(staging / "publication.json", {"round": record.index,
            "images": [{"path": item.path.name, "sha256": item.sha256, "dtype": item.dtype} for item in images],
            "native_files": native_files})
        check_cancelled(cancel_check)
        os.replace(staging, destination)
        return PublishedTemplateRound(durable, destination, tuple(images), native_files)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_template_images(publication, *, cancel_check=None):
    """Recheck changed backing files at canonical publication/finalization."""
    from madi3d_app.project.scientific_records import _file_checksum
    signatures = {}
    for image in publication.images:
        check_cancelled(cancel_check)
        info = image.path.stat()
        if (info.st_size, info.st_mtime_ns) != image.file_signature and _file_checksum(image.path, cancel_check) != image.sha256:
            raise RegistrationOutputError("A published template image changed; its original result evidence was retained.")
        signatures[image.path] = (info.st_size, info.st_mtime_ns)
    return signatures


def prepare_template_companions(model, publication, *, run_id):
    """Capture detached companion content from the canonical model on its thread."""
    from madi3d_app.volume.provenance import volume_companion_projection
    from madi3d_app.project.scientific_records import history_bundle
    from .evidence import restore_template_result
    import numpy as np
    operation_id = f"template-round:{run_id}:{publication.record.index}"
    completed = restore_template_result(model, operation_id)
    references = {reference["object_id"]: reference for reference in completed.output_references}
    prepared = []
    for index, image in enumerate(publication.images):
        channel_id = operation_id + f":channel:{index}"
        channel = model.channels.get(channel_id)
        snapshot = {"channel_id": channel_id, "actor_matrix": np.eye(4).tolist(), "time_index": 0,
            "time_count": 1, "display_name": channel.display_name if channel else image.image.channel_id or "Registration template",
            "channel_role": channel.role if channel else "other", "volume_revision_reference": references[channel_id],
            "coordinate_space_id": image.working_grid.source_coordinate_space_id,
            "dtype": image.dtype, "space_units": image.image.lattice.units}
        projection = volume_companion_projection(model, [snapshot])
        projection["scientific_history"] = history_bundle(model, [*projection["references"], *completed.mapping_references])
        prepared.append((image, snapshot, projection))
    return prepared


def write_prepared_template_companions(publication, prepared, *, resources, cancel_check=None):
    """Write detached companions, sharing transform verification across channels."""
    from madi3d_app.volume.probe import probe_volume_source
    from madi3d_app.volume.provenance import volume_companion_content, VOLUME_COMPANION_SCHEMA
    from madi3d_app.project.scientific_records import write_artifact_companion
    from .transform_resources import publish_transform_resources, _requirements
    signatures = verify_template_images(publication, cancel_check=cancel_check)
    # Every channel includes the same completed round's mapping references.
    # Publish that union once, then project only each channel's requirements.
    history = {"scientific_records": [record for _, _, projection in prepared
        for record in projection["scientific_history"]["scientific_records"]]}
    transport = publish_transform_resources(publication.directory,
        {"scientific_history": history}, resources, cancel_check=cancel_check)
    for image, snapshot, projection in prepared:
        check_cancelled(cancel_check)
        probe = probe_volume_source(str(image.path))
        content = volume_companion_content(probe, [snapshot], {}, {}, projection)
        required = _requirements({"scientific_history": projection["scientific_history"]})
        content["transform_resources"] = {key: transport[key] for key in required}
        info = image.path.stat()
        if (info.st_size, info.st_mtime_ns) != signatures[image.path]:
            raise RegistrationOutputError("A template image changed during companion preparation.")
        check_cancelled(cancel_check)
        write_artifact_companion(image.path, content, schema=VOLUME_COMPANION_SCHEMA,
            scope="volume_lineage", output_format="nrrd", serialization_generation=image.generation,
            verified_checksum=image.sha256, cancel_check=cancel_check)


def write_template_companions(model, publication, *, run_id, resources, cancel_check=None):
    """Project canonical producer evidence into checksum-bound volume companions.

    Retrying this storage finalization never creates another scientific result.
    Transform resources use their existing content-addressed companion route.
    """
    prepared = prepare_template_companions(model, publication, run_id=run_id)
    write_prepared_template_companions(publication, prepared, resources=resources, cancel_check=cancel_check)


@dataclass
class RegistrationOutputBundle:
    run_dir: Path
    policy: dict
    run_id: str
    label: str
    queued: bool = False
    job_id: str = ""
    registration_algorithm_version: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        policy: dict,
        *,
        label: str,
        run_id: str,
        queued: bool = False,
        job_id: str = "",
        registration_algorithm_version: str = "",
    ) -> "RegistrationOutputBundle":
        normalized = normalize_result_output_policy(
            policy, force_enabled=bool(queued), require_root=True
        )
        root = Path(normalized["root_dir"]).resolve()
        run_capacity = _require_bundle_root_capacity(root, normalized)
        root.mkdir(parents=True, exist_ok=True)
        token = safe_output_stem(run_id, "run")[:12]
        folder_candidate = f"{normalize_reformat_name(label)}--{token}"
        folder_name = _bounded_label_component(
            folder_candidate,
            run_capacity,
            hash_source=f"{str(label or 'Registration')}--{str(run_id)}",
        )
        run_dir = root / folder_name
        if run_dir.exists():
            raise RegistrationOutputError(
                f"Registration result folder already exists; refusing to overwrite it: {run_dir}"
            )
        run_dir.mkdir(parents=False, exist_ok=False)
        bundle = cls(
            run_dir=run_dir,
            policy=normalized,
            run_id=str(run_id),
            label=str(label or "Registration"),
            queued=bool(queued),
            job_id=str(job_id or ""),
            registration_algorithm_version=str(registration_algorithm_version or ""),
        )
        bundle.manifest = {
            "format": "MADI3D Registration Result Bundle",
            "format_version": 1,
            "registration_algorithm_version": bundle.registration_algorithm_version,
            "run_id": bundle.run_id,
            "job_id": bundle.job_id or None,
            "queued": bundle.queued,
            "label": bundle.label,
            "created": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
            "policy": copy.deepcopy(normalized),
            "artifacts": [],
            "warnings": [],
            "errors": [],
        }
        # Store the output root only as contextual metadata; all artifact paths
        # remain relative so the completed bundle can be moved as one directory.
        bundle.manifest["policy"]["root_dir"] = "."
        bundle.write_manifest()
        return bundle

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def register_result_group(
        self, group_key: str, label: str, *, single_pair: bool = False
    ) -> Path:
        """Freeze one result group's display label and bundle-relative route.

        A single pair keeps the historical bundle-root layout. Multi-pair
        bundles get one collision-safe child directory per opaque result key.
        """
        key = str(group_key or "").strip()
        if not key:
            raise RegistrationOutputError("A result group requires an output identity.")
        display_label = str(label) if str(label or "") else "Registration"
        routes = self.manifest.get("result_groups", {})
        existing = routes.get(key)
        if existing is not None:
            if str(existing.get("label") or "") != display_label:
                raise RegistrationOutputError(
                    f"Result group {key} already has a different frozen label."
                )
            return self.run_dir / str(existing.get("path") or ".")

        normalized_label = normalize_reformat_name(display_label)
        if single_pair:
            relative = "."
        else:
            key_digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            capacity = _bundle_component_capacity(self.run_dir.resolve(), self.policy)
            if capacity < _MIN_COMPACTED_COMPONENT_UNITS:
                root_capacity = (
                    _windows_text_units(self.run_dir.name)
                    - (_MIN_COMPACTED_COMPONENT_UNITS - capacity)
                )
                if root_capacity < _MIN_COMPACTED_COMPONENT_UNITS:
                    raise _short_result_root_error(Path(self.policy["root_dir"]))
                compacted_root = self.run_dir.with_name(
                    _bounded_label_component(
                        self.run_dir.name,
                        root_capacity,
                        hash_source=f"{self.label}--{self.run_id}--pair-routes",
                    )
                )
                if compacted_root.exists() or compacted_root.is_symlink():
                    raise RegistrationOutputError(
                        "Registration result folder already exists; refusing to overwrite it: "
                        f"{compacted_root}"
                    )
                self.run_dir.rename(compacted_root)
                self.run_dir = compacted_root
                capacity = _bundle_component_capacity(self.run_dir.resolve(), self.policy)
                if capacity < _MIN_COMPACTED_COMPONENT_UNITS:
                    raise _short_result_root_error(Path(self.policy["root_dir"]))
            candidate = None
            for token_length in (8, 12, 16, 24, 32, 64):
                suffix = "--" + key_digest[:token_length]
                component = _bounded_label_component(
                    normalized_label + suffix,
                    capacity,
                    hash_source=f"{display_label}--{key}--{token_length}",
                )
                proposed = self.run_dir / component
                if not proposed.exists():
                    candidate = proposed
                    break
            if candidate is None:
                raise RegistrationOutputError(
                    "Registration result folder already exists; refusing to overwrite it: "
                    f"{self.run_dir / component}"
                )
            relative = candidate.name

        route = self.run_dir / relative
        if relative != ".":
            route.mkdir(parents=False, exist_ok=False)
        candidate = copy.deepcopy(self.manifest)
        candidate.setdefault("result_groups", {})[key] = {
            "label": display_label,
            "path": relative,
            "sanitized": normalized_label != display_label,
            "shortened": relative != "." and not Path(relative).name.startswith(
                normalized_label + "--"
            ),
        }
        try:
            self.write_manifest(manifest=candidate)
        except BaseException:
            if relative != ".":
                try:
                    route.rmdir()
                except OSError:
                    _LOG.warning("Could not remove an unpublished result-group directory %s", route)
            raise
        return route

    def result_group_path(self, group_key: str | None = None) -> Path:
        if group_key is None:
            return self.run_dir
        route = self.manifest.get("result_groups", {}).get(str(group_key))
        if route is None:
            raise RegistrationOutputError(
                f"Result group {group_key} has no frozen output route."
            )
        return self.run_dir / str(route.get("path") or ".")

    def directory(self, category: str, *, group_key: str | None = None) -> Path:
        group_dir = self.result_group_path(group_key)
        if self.policy["layout"] == "separate":
            trailing_units = (
                1
                + _MIN_COMPACTED_COMPONENT_UNITS
                + _artifact_tail_reserve(self.policy)
            )
            category_capacity = min(
                WINDOWS_FILESYSTEM_COMPONENT_BUDGET,
                WINDOWS_SAFE_PATH_BUDGET
                - _windows_text_units(group_dir.resolve())
                - 1
                - trailing_units,
            )
            if category_capacity < _MIN_COMPACTED_COMPONENT_UNITS:
                raise _short_result_root_error(Path(self.policy["root_dir"]))
            category_candidate = safe_output_stem(category, "results").lower()
            component = bounded_output_component(
                category_candidate,
                category_capacity,
                "results",
                hash_source=category,
            )
            path = group_dir / component
        else:
            path = group_dir
        _assert_windows_safe_path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_path(
        self, category: str, stem: str, extension: str, *, group_key: str | None = None
    ) -> Path:
        extension = str(extension or "")
        if extension and not extension.startswith("."):
            extension = "." + extension
        parent = self.directory(category, group_key=group_key)
        tail_units = _windows_text_units(extension + _TEMPORARY_SUFFIX_RESERVE)
        if extension == _TRANSFORM_EXTENSION:
            tail_units = max(tail_units, _windows_text_units(_CMTK_TRANSFORM_TAIL))
        stem_capacity = min(
            WINDOWS_FILESYSTEM_COMPONENT_BUDGET
            - _windows_text_units(extension)
            - _windows_text_units(_TEMPORARY_SUFFIX_RESERVE),
            WINDOWS_SAFE_PATH_BUDGET
            - _windows_text_units(parent.resolve())
            - 1
            - tail_units,
        )
        if stem_capacity < _MIN_COMPACTED_COMPONENT_UNITS:
            raise _short_result_root_error(Path(self.policy["root_dir"]))
        filename = bounded_output_component(stem, stem_capacity) + extension
        path = parent / filename
        _assert_windows_safe_path(path, extra_suffix=_TEMPORARY_SUFFIX_RESERVE)
        return path

    def prepare_reformat_paths(
        self,
        artifacts: Iterable[tuple],
        *,
        group_key: str | None = None,
    ) -> None:
        """Preflight all Reformat paths, then compact generated routes as one transaction.

        Three-tuples use ``group_key``. Four-tuples contain the key first and
        allow one invocation to preflight every pair before moving any folder.
        """
        from madi3d_app.project.scientific_records import object_metadata_sidecar_path

        by_group = {}
        for artifact in artifacts:
            if len(artifact) == 4:
                key, name, kind, output_format = artifact
                key = str(key or "") or None
            elif len(artifact) == 3:
                name, kind, output_format = artifact
                key = group_key
            else:
                raise RegistrationOutputError("A Reformat path plan must have three or four fields.")
            by_group.setdefault(key, []).append(
                (str(name), str(name) + _reformat_extension(kind, output_format))
            )
        if not by_group:
            return
        category = Path("reformatted") if self.policy["layout"] == "separate" else Path()
        plans = []
        for key, filenames in by_group.items():
            route = self.result_group_path(key).resolve()
            excess = max(
                _windows_text_units(
                    object_metadata_sidecar_path(route / category / filename) + ".publishing"
                ) - WINDOWS_SAFE_PATH_BUDGET
                for _name, filename in filenames
            )
            destination = route
            if excess > 0:
                capacity = _windows_text_units(route.name) - excess
                if capacity < _MIN_COMPACTED_COMPONENT_UNITS:
                    raise _short_result_root_error(route.parent)
                destination = route.with_name(
                    _bounded_label_component(
                        route.name,
                        capacity,
                        hash_source=f"{self.label}--{self.run_id}--{key or 'root'}",
                    )
                )
            for name, filename in filenames:
                _validate_reformat_paths(destination / category / filename, name)
            if destination != route and (destination.exists() or destination.is_symlink()):
                raise RegistrationOutputError(
                    f"Registration result folder already exists; refusing to overwrite it: {destination}"
                )
            plans.append((key, route, destination))

        def portable_path_key(path):
            return os.path.abspath(os.fspath(path)).replace("\\\\", "/").casefold()

        destination_keys = [portable_path_key(destination) for _key, _route, destination in plans]
        if len(destination_keys) != len(set(destination_keys)):
            raise RegistrationOutputError("Reformat compaction produced colliding result folder paths.")
        source_keys = {
            portable_path_key(route): key for key, route, _destination in plans
        }
        for key, _route, destination in plans:
            destination_key = portable_path_key(destination)
            if destination_key in source_keys and source_keys[destination_key] != key:
                raise RegistrationOutputError(
                    "Reformat compaction would overwrite another result group folder."
                )

        moves = [(key, route, destination) for key, route, destination in plans if route != destination]
        if not moves:
            return
        if any(key is None or str(
            self.manifest.get("result_groups", {}).get(str(key), {}).get("path") or "."
        ) == "." for key, _route, _destination in moves):
            if len(moves) != 1 or len(plans) != 1:
                raise RegistrationOutputError("A root result bundle cannot be compacted with pair folders in one invocation.")
            _key, route, destination = moves[0]
            route.rename(destination)
            self.run_dir = destination
            return
        self._move_result_groups_for_compaction(moves)

    def reformat_artifact_path(
        self,
        name: str,
        kind: str,
        format: str,
        *,
        group_key: str | None = None,
    ) -> Path:
        """Return a Reformat artifact path without hashing or truncating ``name``.

        ``name`` must already be the final normalized display name allocated
        for this invocation. Final, same-format partial, and provenance
        companion paths are checked before a worker starts.
        """
        name = str(name)
        extension = _reformat_extension(kind, format)

        parent = (
            self.result_group_path(group_key) / "reformatted"
            if self.policy["layout"] == "separate"
            else self.result_group_path(group_key)
        )
        path = parent / f"{name}{extension}"
        _validate_reformat_paths(path, name)
        if self.policy["layout"] == "separate":
            parent.mkdir(parents=True, exist_ok=True)
        return path

    def _move_result_groups_for_compaction(self, moves) -> None:
        """Move complete pair directories and publish all route updates together."""
        root = self.run_dir.resolve()
        old_manifest = copy.deepcopy(self.manifest)
        candidate = copy.deepcopy(old_manifest)
        normalized_moves = []
        for group_key, source, destination in moves:
            source_relative = source.resolve().relative_to(root).as_posix()
            destination_relative = destination.resolve().relative_to(root).as_posix()
            candidate["result_groups"][group_key]["path"] = destination_relative
            candidate["result_groups"][group_key]["shortened"] = True
            prefix = source_relative.rstrip("/") + "/"
            for artifact in candidate.get("artifacts") or []:
                artifact_path = str(artifact.get("path") or "")
                if artifact_path == source_relative:
                    artifact["path"] = destination_relative
                elif artifact_path.startswith(prefix):
                    artifact["path"] = destination_relative + artifact_path[len(source_relative):]
            normalized_moves.append((source, destination))
        moved = []
        try:
            for source, destination in normalized_moves:
                source.rename(destination)
                moved.append((source, destination))
            self.write_manifest(manifest=candidate)
        except BaseException:
            for source, destination in reversed(moved):
                try:
                    destination.rename(source)
                except OSError:
                    _LOG.exception("Could not restore result directory after manifest publication failed")
            # Atomic publication normally leaves the prior file untouched. If
            # it did publish before an exception, restore the prior authority.
            try:
                _atomic_json(self.manifest_path, old_manifest)
            except OSError:
                _LOG.exception("Could not restore the prior registration result manifest")
            self.manifest = old_manifest
            raise

    def relative(self, path: os.PathLike | str) -> str:
        candidate = Path(path).resolve()
        root = self.run_dir.resolve()
        if candidate != root and root not in candidate.parents:
            raise RegistrationOutputError(
                f"Result artifact is outside its registration bundle: {candidate}"
            )
        return candidate.relative_to(root).as_posix()

    def _artifact_entry(self, kind, path, *, group_key=None, **metadata) -> dict:
        entry = {"kind": str(kind or "artifact"), "path": self.relative(path)}
        if group_key is not None:
            entry["output_group_key"] = str(group_key)
        entry.update({str(k): copy.deepcopy(v) for k, v in metadata.items() if v is not None})
        return entry

    @staticmethod
    def _merge_artifacts(manifest: dict, entries: Iterable[dict]) -> None:
        artifacts = manifest.setdefault("artifacts", [])
        # One transient index per publication, including replacement on retry.
        positions = {(entry["kind"], entry["path"]): index
                     for index, entry in enumerate(artifacts)}
        for entry in entries:
            key = (entry["kind"], entry["path"])
            index = positions.get(key)
            if index is None:
                positions[key] = len(artifacts)
                artifacts.append(entry)
            else:
                artifacts[index] = entry

    def record_artifacts(self, artifacts: Iterable[dict]) -> None:
        """Publish completed artifact descriptions together, then adopt them.

        Descriptions contain kind, path, optional group_key and additional
        metadata fields. One detached candidate and transient lookup cover all
        additions and replacements in this publication.
        """
        entries = [self._artifact_entry(**artifact) for artifact in artifacts]
        if not entries:
            return
        candidate = copy.deepcopy(self.manifest)
        self._merge_artifacts(candidate, entries)
        self._publish_manifest(candidate)

    def record_artifact(
        self,
        kind: str,
        path: os.PathLike | str,
        *,
        group_key: str | None = None,
        **metadata,
    ) -> None:
        self.record_artifacts([dict(kind=kind, path=path, group_key=group_key, **metadata)])

    def add_warning(self, message: str) -> None:
        text = str(message or "").strip()
        if text and text not in self.manifest.get("warnings", []):
            candidate = copy.deepcopy(self.manifest)
            candidate.setdefault("warnings", []).append(text)
            self.write_manifest(manifest=candidate)

    def add_error(self, message: str) -> None:
        text = str(message or "").strip()
        if text:
            candidate = copy.deepcopy(self.manifest)
            candidate.setdefault("errors", []).append(text)
            self.write_manifest(manifest=candidate)

    def write_manifest(self, status: str | None = None, *, manifest: dict | None = None) -> None:
        """Adopt a manifest update only after its atomic publication succeeds."""
        candidate = copy.deepcopy(self.manifest if manifest is None else manifest)
        if status:
            candidate["status"] = str(status)
        self._publish_manifest(candidate)

    def _publish_manifest(self, candidate: dict) -> None:
        candidate["updated"] = datetime.now().isoformat(timespec="seconds")
        _atomic_json(self.manifest_path, candidate)
        self.manifest = candidate

    def write_qc_summary(
        self, chain_payload: dict, stem: str, *, group_key: str | None = None,
        artifacts: list[dict] | None = None,
    ) -> Path:
        path = self.artifact_path("qc", stem, ".qc.json", group_key=group_key)
        _atomic_json(path, registration_qc_summary(chain_payload))
        artifact = dict(kind="qc_summary", path=path, group_key=group_key,
                        registration_id=str((chain_payload or {}).get("registration_id") or ""))
        if artifacts is None:
            self.record_artifacts([artifact])
        else:
            artifacts.append(artifact)
        return path

    def copy_qc_volume(
        self,
        source: os.PathLike | str,
        stem: str,
        role: str,
        *,
        group_key: str | None = None,
        artifacts: list[dict] | None = None,
    ) -> Path:
        source_path = Path(source)
        if not source_path.is_file():
            raise RegistrationOutputError(f"Registration QC volume is missing: {source_path}")
        destination = self.artifact_path("qc", stem, ".nrrd", group_key=group_key)
        shutil.copy2(source_path, destination)
        artifact = dict(kind="qc_volume", path=destination, group_key=group_key,
                        role=str(role or "qc"), coordinate_space="cmtk_working_grid")
        if artifacts is None:
            self.record_artifacts([artifact])
        else:
            artifacts.append(artifact)
        return destination

    def _write_log_artifact(self, text: str) -> tuple[Path, dict]:
        path = self.artifact_path("logs", "registration", ".log")
        data = str(text or "").encode("utf-8")
        limit = 256 * 1024
        _atomic_bytes(path, data[:limit])
        return path, dict(truncated=len(data) > limit,
                          original_size_bytes=len(data), retained_size_bytes=min(len(data), limit))

    def write_log(self, text: str) -> Path:
        path, metadata = self._write_log_artifact(text)
        self.record_artifact("run_log", path, **metadata)
        return path

    def finalize(self, status: str = "complete", *, error: str = "", log_text: str | None = None) -> Path:
        """Publish terminal status, diagnostics and artifacts in one update.

        The log is optional diagnostic output. Its failure is disclosed in the
        final manifest without preventing that authoritative publication.
        """
        candidate = copy.deepcopy(self.manifest)
        text = str(error or "").strip()
        if text:
            candidate.setdefault("errors", []).append(text)
        if log_text is not None:
            try:
                path, metadata = self._write_log_artifact(log_text)
            except Exception as exc:
                warning = f"Registration diagnostic log could not be written: {exc}"
                candidate.setdefault("warnings", []).append(warning)
                _LOG.warning(warning)
            else:
                self._merge_artifacts(candidate, [self._artifact_entry("run_log", path, **metadata)])
        if status:
            candidate["status"] = str(status)
        self._publish_manifest(candidate)
        return self.run_dir


def cmtk_linear_stage_labels(dof_sequence):
    """Display the executed CMTK linear degrees-of-freedom sequence."""
    labels = {6: "Rigid", 7: "Similarity", 9: "Scale", 12: "Affine"}
    return [labels[int(dof)] for dof in tuple(dof_sequence or ()) if int(dof) in labels]


def registration_stage_path(stages):
    """Return the successfully executed transform path in compact terms."""
    labels = {
        "rigid": "Rigid",
        "similarity": "Similarity",
        "scaleversor": "Scale",
        "affine": "Affine",
        "cmtk_warp": "Warp",
    }
    path = []
    for stage in list(stages or []):
        status = stage.get("execution_status", "pending") if isinstance(stage, dict) else getattr(stage, "execution_status", "pending")
        kind = stage.get("kind", "") if isinstance(stage, dict) else getattr(stage, "kind", "")
        details = stage.get("details", {}) if isinstance(stage, dict) else getattr(stage, "details", {})
        if not execution_succeeded(status):
            continue
        if str(dict(details or {}).get("engine") or "").lower() == "cmtk":
            for cmtk_label in cmtk_linear_stage_labels(dict(details or {}).get("dof_sequence") or ()):
                if not path or path[-1] != cmtk_label:
                    path.append(cmtk_label)
            if dict(details or {}).get("dof_sequence"):
                continue
        label = labels.get(str(kind or "").lower())
        if not label:
            continue
        if not path or path[-1] != label:
            path.append(label)
    if path and path[0] == "Warp":
        path.insert(0, "Initial")
    return " → ".join(path) if path else "Initial only"


def linear_qc_text(qc, precision=4):
    """Format decomposed polar scale and shear values without hiding components."""
    qc = dict(qc or {})
    try:
        digits = max(1, int(precision))
    except Exception:
        digits = 4
    parts = []
    scale = list(qc.get("scale_xyz") or qc.get("axis_stretch_xyz") or ())
    if len(scale) == 3:
        parts.append("scale XYZ=" + "/".join(f"{float(v):.{digits}f}" for v in scale))
    shears = [qc.get("shear_xy"), qc.get("shear_xz"), qc.get("shear_yz")]
    if all(value is not None for value in shears):
        parts.append(
            "shear XY/XZ/YZ="
            + "/".join(f"{float(v):+.{digits}f}" for v in shears)
        )
        if qc.get("max_abs_shear") is not None:
            parts.append(f"max |shear|={float(qc['max_abs_shear']):.{digits}f}")
    return "; ".join(parts)


__all__ = [
    "DEFAULT_RESULT_OUTPUT",
    "MESH_FORMAT_EXTENSIONS",
    "RESULT_LAYOUTS",
    "RegistrationOutputBundle",
    "RegistrationOutputError",
    "VOLUME_FORMAT_EXTENSIONS",
    "WINDOWS_SAFE_PATH_BUDGET",
    "allocate_reformat_name",
    "bounded_output_component",
    "ensure_output_root_writable",
    "mesh_extension",
    "normalize_reformat_name",
    "normalize_result_output_policy",
    "partial_output_path",
    "registration_qc_summary",
    "reformat_name_key",
    "cmtk_linear_stage_labels",
    "linear_qc_text",
    "registration_stage_path",
    "safe_output_stem",
    "volume_extension",
]
