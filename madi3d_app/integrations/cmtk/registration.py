# -*- coding: utf-8 -*-
"""CMTK registration core for MADI3D.

This module is independent from Qt and VTK. It owns CMTK command construction
for global linear registration, nonlinear warp, and image reformatting. MADI3D
passes moving->reference affine matrices at the Python boundary; the existing
CMTK transform bridge performs the required reference->floating inversion and
converts optimized affine output back into the same MADI matrix contract.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from madi3d_app.integrations.cmtk.backend import CMTKError, process_error
from madi3d_app.integrations.cmtk.process import CMTKProcessResult, run_cmtk_streaming
from madi3d_app.integrations.cmtk.xform import (
    AffineRoundTripQC,
    matrix4,
    read_affine_xform,
    write_affine_xform,
)
from madi3d_app.volume.geometry import direction_matrix3, finite_tuple3


_METRIC_FLAGS = {
    "nmi": "--nmi",
    "mi": "--mi",
    "cr": "--cr",
    "msd": "--msd",
    "ncc": "--ncc",
}
_LINEAR_DOF_SEQUENCES = {
    6: (6,),
    7: (6, 7),
    9: (6, 9),
    12: (6, 9, 12),
}
_SMALL_FILE_HASH_LIMIT = 16 * 1024 * 1024
_FINGERPRINT_SAMPLE = 1024 * 1024
_LINEAR_ARTIFACT_NAMES = {
    "linear-initial.xform",
    "linear.pending.list",
    "linear.xform",
    "cmtk-linear-stdout.log",
    "cmtk-linear-stderr.log",
}
_ARTIFACT_NAMES = {
    "affine.xform",
    "affine.pending.xform",
    "warp.xform",
    "warp.pending.xform",
    "manifest.json",
    "manifest.json.tmp",
    "cmtk-stdout.log",
    "cmtk-stderr.log",
}

_NRRD_TYPES = {
    np.dtype(np.uint8): "uchar",
    np.dtype(np.int16): "short",
    np.dtype(np.uint16): "ushort",
    np.dtype(np.int32): "int",
    np.dtype(np.uint32): "uint",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
}
_CMTK_OUTPUT_TYPES = {
    np.dtype(np.uint8): "byte",
    np.dtype(np.int16): "short",
    np.dtype(np.uint16): "ushort",
    np.dtype(np.int32): "int",
    np.dtype(np.uint32): "uint",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
}
_REFORMAT_INTERPOLATION = {
    "nearest": "nn",
    "nn": "nn",
    "linear": "linear",
    "cubic": "cubic",
}


def _finite_float(name: str, value, *, positive=False, nonnegative=False) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"CMTK {name} must be finite.")
    if positive and number <= 0:
        raise ValueError(f"CMTK {name} must be > 0.")
    if nonnegative and number < 0:
        raise ValueError(f"CMTK {name} cannot be negative.")
    return number


def _exact_int(name: str, value, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"CMTK {name} must be an integer.")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"CMTK {name} must be an integer.")
    result = int(number)
    if result < minimum:
        raise ValueError(f"CMTK {name} must be at least {minimum}.")
    return result


@dataclass(frozen=True)
class CMTKLinearSettings:
    """Validated settings for CMTK ``registration`` global alignment."""

    metric: str = "nmi"
    final_dof: int = 9
    exploration: float = 16.0
    accuracy: float = 0.8
    coarsest: float = 8.0
    threads: int = 1

    def validated(self) -> "CMTKLinearSettings":
        metric = str(self.metric or "").lower()
        if metric not in _METRIC_FLAGS:
            raise ValueError(f"Unsupported CMTK linear metric: {self.metric}")
        final_dof = _exact_int("final linear DOF", self.final_dof, minimum=1)
        if final_dof not in _LINEAR_DOF_SEQUENCES:
            allowed = ", ".join(str(v) for v in sorted(_LINEAR_DOF_SEQUENCES))
            raise ValueError(
                f"Unsupported CMTK final linear DOF: {self.final_dof}. "
                f"Expected one of {allowed}."
            )
        _finite_float("linear exploration", self.exploration, positive=True)
        _finite_float("linear accuracy", self.accuracy, positive=True)
        _finite_float("linear coarsest resolution", self.coarsest, positive=True)
        _exact_int("linear thread count", self.threads, minimum=1)
        return self

    @property
    def dof_sequence(self) -> tuple[int, ...]:
        validated = self.validated()
        return _LINEAR_DOF_SEQUENCES[int(validated.final_dof)]


@dataclass(frozen=True)
class CMTKWarpSettings:
    metric: str = "nmi"
    exploration: float = 26.0
    accuracy: float = 0.8
    coarsest: float = 8.0
    grid_spacing: float = 80.0
    refine: int = 4
    mode: str = "fast"
    threads: int = 1
    energy_weight: float = 0.1
    jacobian_weight: float = 0.0
    inverse_consistency_weight: float = 0.0
    omit_original_data: bool = False
    match_histograms: bool = False

    def validated(self) -> "CMTKWarpSettings":
        metric = str(self.metric or "").lower()
        if metric not in _METRIC_FLAGS:
            raise ValueError(f"Unsupported CMTK warp metric: {self.metric}")
        mode = str(self.mode or "").lower()
        if mode not in {"fast", "accurate"}:
            raise ValueError("CMTK warp mode must be 'fast' or 'accurate'.")
        _finite_float("exploration", self.exploration, positive=True)
        _finite_float("accuracy", self.accuracy, positive=True)
        _finite_float("coarsest resolution", self.coarsest, positive=True)
        _finite_float("grid spacing", self.grid_spacing, positive=True)
        _exact_int("refine count", self.refine, minimum=0)
        _exact_int("thread count", self.threads, minimum=1)
        _finite_float("energy weight", self.energy_weight, nonnegative=True)
        _finite_float("Jacobian weight", self.jacobian_weight, nonnegative=True)
        _finite_float(
            "inverse-consistency weight",
            self.inverse_consistency_weight,
            nonnegative=True,
        )
        return self


@dataclass(frozen=True)
class CMTKLinearRegistrationResult:
    workspace: Path
    reference_image: Path
    floating_image: Path
    initial_xform: Path
    output_xform: Path
    stdout_log: Path
    stderr_log: Path
    moving_to_reference: np.ndarray = field(repr=False)
    settings: CMTKLinearSettings = field(default_factory=CMTKLinearSettings)
    dof_sequence: tuple[int, ...] = field(default_factory=tuple)
    cmtk_version: str = ""
    describe: str = ""
    command: tuple[str, ...] = field(default_factory=tuple)
    stdout: str = ""
    stderr: str = ""
    affine_serialization_qc: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CMTKRegistrationArtifacts:
    workspace: Path
    reference_image: Path
    floating_image: Path
    affine_xform: Path
    warp_xform: Path
    manifest_path: Path
    stdout_log: Path
    stderr_log: Path
    cmtk_version: str = ""
    command: tuple[str, ...] = field(default_factory=tuple)
    stdout: str = ""
    stderr: str = ""
    affine_serialization_qc: dict = field(default_factory=dict)


def _number(value) -> str:
    return f"{float(value):.12g}"


def _validated_grid(array_zyx, grid: dict):
    array = np.asarray(array_zyx)
    if array.ndim != 3:
        raise ValueError(
            f"CMTK NRRD staging requires a 3-D array; got {array.shape}."
        )
    dims_value = grid.get("dims_xyz")
    if dims_value is None:
        raise ValueError("CMTK image grid is missing explicit XYZ dimensions.")
    dims_xyz = finite_tuple3(
        dims_value, "CMTK image dimensions", positive=True, integer=True
    )
    expected = (dims_xyz[2], dims_xyz[1], dims_xyz[0])
    if tuple(array.shape) != expected:
        raise ValueError(
            "CMTK image grid/array mismatch: "
            f"grid XYZ={dims_xyz}, array ZYX={tuple(array.shape)}."
        )
    return array, dims_xyz, _validated_geometry(grid)


def _validated_geometry(grid: dict):
    origin_value = grid.get("origin")
    spacing_value = grid.get("spacing")
    direction_value = grid.get("direction")
    missing = [
        name
        for name, value in (
            ("origin", origin_value),
            ("spacing", spacing_value),
            ("direction", direction_value),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "CMTK image grid is missing explicit " + ", ".join(missing) + "."
        )
    origin = np.asarray(
        finite_tuple3(origin_value, "CMTK image origin"), dtype=np.float64
    )
    spacing = np.asarray(
        finite_tuple3(
            spacing_value, "CMTK image spacing", positive=True
        ),
        dtype=np.float64,
    )
    try:
        serialized_direction = np.asarray(direction_value, dtype=np.float64)
        if serialized_direction.size != 9:
            raise ValueError
        serialized_direction = serialized_direction.reshape(3, 3)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "CMTK image direction must contain exactly nine numeric values "
            "for a 3 x 3 matrix."
        ) from exc
    direction = direction_matrix3(serialized_direction).astype(
        np.float64, copy=False
    )
    return origin, spacing, direction


def canonical_cmtk_grid(grid: dict) -> dict:
    """Return a CMTK-safe image lattice with spacing only and zero pose.

    CMTK's oriented NRRD reader normalizes anatomical RAS/LAS/LPS geometry.
    MADI scene/world pose must therefore not be encoded as NRRD origin or
    direction. The array dimensions and physical voxel spacing are intrinsic
    image properties and remain in the staged lattice; pose is carried by
    explicit transform matrices instead.
    """
    value = dict(grid or {})
    dims = finite_tuple3(
        value.get("dims_xyz"),
        "CMTK image dimensions",
        positive=True,
        integer=True,
    )
    _origin, spacing, _direction = _validated_geometry(value)
    return {
        "dims_xyz": dims,
        "origin": (0.0, 0.0, 0.0),
        "spacing": tuple(float(v) for v in spacing),
        "direction": np.eye(3, dtype=np.float64).tolist(),
    }


def cmtk_working_moving_to_reference(
    moving_to_reference_world,
    reference_origin_world,
    floating_origin_world,
) -> np.ndarray:
    """Convert a MADI world correction into zero-origin CMTK working frames.

    MADI/ITK working arrays are sampled on compact world-axis-aligned grids.
    If CMTK stages those same arrays with zero NRRD origins, CMTK coordinates
    are q_ref = world_ref - O_ref and q_flt = world_flt - O_flt. Therefore
    the moving->reference correction in CMTK working coordinates is

        T(-O_ref) @ M_world @ T(O_flt).
    """
    world = matrix4(moving_to_reference_world)
    reference_origin = np.asarray(reference_origin_world, dtype=np.float64).reshape(3)
    floating_origin = np.asarray(floating_origin_world, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(reference_origin)) or not np.all(np.isfinite(floating_origin)):
        raise ValueError("CMTK working-grid origins must be finite.")
    to_reference_local = np.eye(4, dtype=np.float64)
    to_reference_local[:3, 3] = -reference_origin
    from_floating_local = np.eye(4, dtype=np.float64)
    from_floating_local[:3, 3] = floating_origin
    return matrix4(to_reference_local @ world @ from_floating_local)


def cmtk_neutral_to_working_matrix(
    native_geometry: dict,
    actor_matrix,
    working_origin_world,
) -> np.ndarray:
    """Map a spacing-only native CMTK image into a registration working frame.

    A neutral native NRRD uses the captured voxel spacing but zero origin and
    identity direction. Its physical coordinates already include spacing, so
    the captured image direction and origin plus the MADI actor matrix supply
    only the remaining pose. The destination working frame is world-axis
    aligned with its captured world origin mapped to CMTK coordinate zero.
    """
    origin, _spacing, direction = _validated_geometry(dict(native_geometry or {}))
    actor = matrix4(actor_matrix)
    neutral_to_local = np.eye(4, dtype=np.float64)
    neutral_to_local[:3, :3] = direction
    neutral_to_local[:3, 3] = origin
    neutral_to_world = actor @ neutral_to_local
    working_origin = np.asarray(working_origin_world, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(working_origin)):
        raise ValueError("CMTK working-grid origin must be finite.")
    world_to_working = np.eye(4, dtype=np.float64)
    world_to_working[:3, 3] = -working_origin
    return matrix4(world_to_working @ neutral_to_world)


def _nrrd_scalar_array(array_zyx):
    array = np.asarray(array_zyx)
    if np.iscomplexobj(array):
        raise TypeError("Complex-valued CMTK staging images are not supported.")
    if array.dtype == np.bool_:
        array = array.astype(np.uint8)
    elif array.dtype == np.int8:
        array = array.astype(np.int16)
    elif array.dtype == np.float16:
        array = array.astype(np.float32)
    elif array.dtype not in _NRRD_TYPES:
        if np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32)
        else:
            raise TypeError(f"Unsupported CMTK staging dtype: {array.dtype}")
    dtype = np.dtype(array.dtype).newbyteorder("<")
    return np.ascontiguousarray(array, dtype=dtype)


def _nrrd_header(dims_xyz, geometry, *, nrrd_type: str, encoding: str) -> bytes:
    origin, spacing, direction = geometry

    def vector_text(values) -> str:
        return "(" + ",".join(f"{float(v):.17g}" for v in values) + ")"

    space_directions = [
        direction[:, axis] * spacing[axis]
        for axis in range(3)
    ]
    return "\n".join([
        "NRRD0005",
        "# MADI3D CMTK image",
        f"type: {nrrd_type}",
        "dimension: 3",
        "space: right-anterior-superior",
        f"sizes: {dims_xyz[0]} {dims_xyz[1]} {dims_xyz[2]}",
        "space directions: " + " ".join(
            vector_text(value) for value in space_directions
        ),
        "kinds: domain domain domain",
        "endian: little",
        f"encoding: {encoding}",
        "space origin: " + vector_text(origin),
        "",
        "",
    ]).encode("ascii")


def write_volume_nrrd(
    path: os.PathLike | str,
    array_zyx,
    grid: dict,
    *,
    force_float32: bool = False,
) -> Path:
    """Stage one scalar Z,Y,X image with exact MADI physical geometry."""
    output = Path(path)
    source = np.asarray(array_zyx)
    if force_float32:
        source = np.asarray(source, dtype=np.float32)
    array = _nrrd_scalar_array(source)
    array, dims_xyz, geometry = _validated_grid(array, grid)
    header = _nrrd_header(
        dims_xyz, geometry, nrrd_type=_NRRD_TYPES[np.dtype(array.dtype).newbyteorder("=")],
        encoding="raw",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(header)
            handle.write(array.tobytes(order="C"))
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def write_reference_grid_nrrd(path: os.PathLike | str, grid: dict) -> Path:
    """Write a memory-light zero-valued NRRD carrying only a target lattice.

    ``reformatx`` uses the reference image to define the output lattice. A gzip
    stream of zeros keeps this target image tiny on disk even for very large
    volumes while preserving dimensions, origin and the complete direction map.
    """
    output = Path(path)
    dims_value = grid.get("dims_xyz")
    dims_xyz = tuple(int(v) for v in (dims_value or ()))
    if len(dims_xyz) != 3 or any(v < 1 for v in dims_xyz):
        raise ValueError(f"Invalid CMTK reference dimensions: {dims_xyz}.")
    geometry = _validated_geometry(grid)
    header = _nrrd_header(dims_xyz, geometry, nrrd_type="uchar", encoding="gzip")
    total = int(dims_xyz[0]) * int(dims_xyz[1]) * int(dims_xyz[2])
    zero_chunk = bytes(1024 * 1024)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(header)
            with gzip.GzipFile(fileobj=handle, mode="wb", compresslevel=1) as compressed:
                remaining = total
                while remaining:
                    count = min(remaining, len(zero_chunk))
                    compressed.write(zero_chunk[:count])
                    remaining -= count
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def write_working_nrrd(
    path: os.PathLike | str,
    array_zyx,
    grid: dict,
) -> Path:
    """Stage one float32 registration working image without another resampling."""
    return write_volume_nrrd(path, array_zyx, grid, force_float32=True)


def read_nrrd_zyx(path: os.PathLike | str):
    """Read a scalar NRRD as MADI's Z,Y,X array order."""
    try:
        import nrrd
    except Exception as exc:
        raise RuntimeError(
            "CMTK Reformat requires pynrrd to read the generated NRRD output."
        ) from exc
    array, header = nrrd.read(os.fspath(path), index_order="C")
    array = np.asarray(array)
    if array.ndim != 3:
        raise CMTKError(
            f"CMTK reformat output is not a scalar 3-D image: {array.shape}."
        )
    return np.ascontiguousarray(array), dict(header or {})


def cmtk_output_type_for_dtype(dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        dtype = np.dtype(np.uint8)
    elif dtype == np.int8:
        dtype = np.dtype(np.int16)
    elif dtype == np.float16:
        dtype = np.dtype(np.float32)
    elif dtype not in _CMTK_OUTPUT_TYPES:
        if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating):
            dtype = np.dtype(np.float32)
        else:
            raise TypeError(f"Unsupported CMTK output dtype: {dtype}")
    return _CMTK_OUTPUT_TYPES[dtype]


def build_reformatx_args(
    backend,
    reference_image: os.PathLike | str,
    floating_image: os.PathLike | str,
    xforms,
    output_image: os.PathLike | str,
    *,
    interpolation: str = "linear",
    output_type: Optional[str] = None,
    pad_out: float = 0.0,
) -> list[str]:
    """Build argv for CMTK ``reformatx`` using an ordered transform chain."""
    interpolation_key = str(interpolation or "linear").lower()
    if interpolation_key not in _REFORMAT_INTERPOLATION:
        raise ValueError(f"Unsupported CMTK reformat interpolation: {interpolation}")
    transform_values = (
        [xforms]
        if isinstance(xforms, (str, os.PathLike))
        else list(xforms or ())
    )
    if not transform_values:
        raise ValueError("CMTK reformat requires at least one transform.")
    args = [
        "--interpolation", _REFORMAT_INTERPOLATION[interpolation_key],
        "--pad-out", _number(pad_out),
    ]
    if output_type:
        output_value = str(output_type).lower()
        allowed = {"char", "byte", "short", "ushort", "int", "uint", "float", "double"}
        if output_value not in allowed:
            raise ValueError(f"Unsupported CMTK reformat output type: {output_type}")
        args += ["--outputtype", output_value]
    args += [
        "--outfile", backend.translate_path(os.fspath(output_image)),
        "--floating", backend.translate_path(os.fspath(floating_image)),
        backend.translate_path(os.fspath(reference_image)),
    ]
    args += [
        backend.translate_path(os.fspath(value))
        for value in transform_values
    ]
    return args



def build_jacobian_args(
    backend,
    reference_image: os.PathLike | str,
    xforms,
    output_image: os.PathLike | str,
    *,
    correct_global: bool = True,
    threads: int = 1,
) -> list[str]:
    """Build ``reformatx`` argv for a Jacobian determinant map.

    CMTK's Jacobian mode does not require a floating intensity image. The
    reference image defines the target grid and transformations follow the
    explicit ``--jacobian`` separator. ``--jacobian-correct-global`` removes
    the affine seed's global scale so nonlinear QC is not double-counted with
    MADI's separate linear determinant/scale diagnostics.
    """
    transform_values = (
        [xforms]
        if isinstance(xforms, (str, os.PathLike))
        else list(xforms or ())
    )
    if not transform_values:
        raise ValueError("CMTK Jacobian QC requires at least one transform.")
    args = [
        "--threads", str(_exact_int("Jacobian thread count", threads, minimum=1)),
        "--outputtype", "float",
        "--outfile", backend.translate_path(os.fspath(output_image)),
    ]
    if correct_global:
        args.append("--jacobian-correct-global")
    args += [
        backend.translate_path(os.fspath(reference_image)),
        "--jacobian",
    ]
    args += [backend.translate_path(os.fspath(value)) for value in transform_values]
    return args


def summarize_jacobian(array) -> dict[str, object]:
    """Return deterministic, scientist-facing Jacobian determinant QC.

    Non-positive determinants indicate local folding. They are deliberately
    reported rather than rejected here: MADI's registration policy is
    warn-and-keep for a completed CMTK warp while preserving the evidence needed
    to judge whether the deformation is biologically plausible.
    """
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    total = int(values.size)
    finite = values[np.isfinite(values)]
    if total <= 0 or finite.size <= 0:
        raise CMTKError("CMTK Jacobian QC produced no finite determinant samples.")
    percentiles = np.percentile(finite, [1.0, 5.0, 50.0, 95.0, 99.0])
    finite_count = int(finite.size)
    negative = int(np.count_nonzero(finite < 0.0))
    nonpositive = int(np.count_nonzero(finite <= 0.0))
    severe_compression = int(np.count_nonzero((finite > 0.0) & (finite < 0.2)))
    severe_expansion = int(np.count_nonzero(finite > 5.0))
    warnings = []
    if finite_count < total:
        warnings.append(
            f"Jacobian map contains {total - finite_count} non-finite voxel(s) "
            f"({100.0 * (total - finite_count) / max(1, total):.4g}%)"
        )
    if nonpositive:
        warnings.append(
            f"local folding detected: {nonpositive}/{finite_count} finite voxel(s) "
            f"have Jacobian <= 0 ({100.0 * nonpositive / finite_count:.4g}%)"
        )
    if severe_compression:
        warnings.append(
            f"strong local compression: {severe_compression}/{finite_count} finite voxel(s) "
            f"have 0 < Jacobian < 0.2 ({100.0 * severe_compression / finite_count:.4g}%)"
        )
    if severe_expansion:
        warnings.append(
            f"strong local expansion: {severe_expansion}/{finite_count} finite voxel(s) "
            f"have Jacobian > 5 ({100.0 * severe_expansion / finite_count:.4g}%)"
        )
    return {
        "sample_count": total,
        "finite_count": finite_count,
        "finite_fraction": float(finite_count / total),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "median": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "negative_count": negative,
        "negative_fraction": float(negative / finite_count),
        "nonpositive_count": nonpositive,
        "nonpositive_fraction": float(nonpositive / finite_count),
        "severe_compression_count": severe_compression,
        "severe_compression_fraction": float(severe_compression / finite_count),
        "severe_expansion_count": severe_expansion,
        "severe_expansion_fraction": float(severe_expansion / finite_count),
        "warnings": warnings,
    }


def update_artifact_bundle_qc(
    bundle_dir: os.PathLike | str,
    deformation_qc: dict,
) -> dict:
    """Atomically attach completed QC metadata to a verified CMTK bundle."""
    verified = verify_artifact_bundle(bundle_dir)
    manifest_path = Path(verified["manifest_path"])
    manifest = dict(verified.get("manifest") or {})
    manifest["format_version"] = max(3, int(manifest.get("format_version") or 0))
    manifest["deformation_qc"] = json.loads(json.dumps(dict(deformation_qc or {})))
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return verify_artifact_bundle(bundle_dir)



def build_linear_registration_args(
    backend,
    reference_image: os.PathLike | str,
    floating_image: os.PathLike | str,
    initial_xform: os.PathLike | str,
    output_list: os.PathLike | str,
    settings: CMTKLinearSettings,
) -> list[str]:
    """Build argv for CMTK ``registration`` with an explicit MADI initializer.

    ``--no-switch`` keeps reference/floating roles fixed, so the transform
    direction remains the same reference->floating convention used by the
    existing CMTK bridge. Successive ``--dofs`` options deliberately mirror
    CMTK's staged optimizer semantics. ``--outlist`` writes a CMTK StudyList
    archive directory; ``run_linear`` extracts its canonical affine member.
    """
    s = settings.validated()
    args = [
        "--echo",
        "--verbose-level", "1",
        "--threads", str(_exact_int("linear thread count", s.threads, minimum=1)),
        "--no-switch",
        _METRIC_FLAGS[str(s.metric).lower()],
        "--exploration", _number(s.exploration),
        "--accuracy", _number(s.accuracy),
        "--coarsest", _number(s.coarsest),
    ]
    for dof in s.dof_sequence:
        args += ["--dofs", str(int(dof))]
    args += [
        "--outlist", backend.translate_path(os.fspath(output_list)),
        "--initial", backend.translate_path(os.fspath(initial_xform)),
        backend.translate_path(os.fspath(reference_image)),
        backend.translate_path(os.fspath(floating_image)),
    ]
    return args


def build_warp_args(
    backend,
    reference_image: os.PathLike | str,
    floating_image: os.PathLike | str,
    initial_xform: os.PathLike | str,
    output_xform: os.PathLike | str,
    settings: CMTKWarpSettings,
    *,
    output_xform_backend_path: Optional[str] = None,
) -> list[str]:
    """Build argv for the documented CMTK 3.3.x ``warp`` interface."""
    s = settings.validated()
    args = [
        "--echo",
        "--verbose-level", "1",
        "--threads", str(_exact_int("thread count", s.threads, minimum=1)),
        _METRIC_FLAGS[str(s.metric).lower()],
        "--exploration", _number(s.exploration),
        "--accuracy", _number(s.accuracy),
        "--grid-spacing", _number(s.grid_spacing),
        "--refine", str(_exact_int("refine count", s.refine, minimum=0)),
        "--coarsest", _number(s.coarsest),
        "--energy-weight", _number(s.energy_weight),
        "--jacobian-weight", _number(s.jacobian_weight),
        "--ic-weight", _number(s.inverse_consistency_weight),
        "--fast" if str(s.mode).lower() == "fast" else "--accurate",
    ]
    if s.omit_original_data:
        args.append("--omit-original-data")
    if s.match_histograms:
        args.append("--match-histograms")
    # Use the explicit initializer option rather than relying on the optional
    # third positional InitialXform.  This is unambiguous for the standalone
    # affine written by mat2dof and keeps the two image operands explicit.
    output_path = (
        str(output_xform_backend_path)
        if output_xform_backend_path is not None
        else backend.translate_path(os.fspath(output_xform))
    )
    args += [
        "--outlist", output_path,
        "--initial", backend.translate_path(os.fspath(initial_xform)),
        backend.translate_path(os.fspath(reference_image)),
        backend.translate_path(os.fspath(floating_image)),
    ]
    return args


def _clear_linear_artifact(workspace: Path, name: str) -> None:
    if name not in _LINEAR_ARTIFACT_NAMES:
        raise CMTKError(f"Refusing to remove unknown linear registration artifact: {name}")
    path = workspace / name
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _clear_artifact(workspace: Path, name: str) -> None:
    if name not in _ARTIFACT_NAMES:
        raise CMTKError(f"Refusing to remove unknown registration artifact: {name}")
    path = workspace / name
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _sha256_file(path: Path, digest=None) -> str:
    hasher = digest or hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        return _sha256_file(path, digest)
    if not path.is_dir():
        raise FileNotFoundError(path)
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with child.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def _input_identity(path: Path) -> dict:
    stat = path.stat()
    payload = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if stat.st_size <= _SMALL_FILE_HASH_LIMIT:
        payload["sha256"] = _sha256_file(path)
        payload["hash_mode"] = "full"
        return payload

    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(_FINGERPRINT_SAMPLE))
        handle.seek(max(0, stat.st_size - _FINGERPRINT_SAMPLE))
        digest.update(handle.read(_FINGERPRINT_SAMPLE))
    payload["sampled_sha256"] = digest.hexdigest()
    payload["hash_mode"] = "size+first-last-1MiB"
    return payload


def _bundle_child(root: Path, relative: str, label: str) -> Path:
    value = str(relative or "").strip()
    if not value:
        raise CMTKError(f"CMTK artifact manifest is missing {label} path.")
    candidate = (root / value).resolve()
    if candidate != root and root not in candidate.parents:
        raise CMTKError(f"CMTK artifact {label} escapes its bundle: {value}")
    return candidate


def verify_artifact_bundle(bundle_dir: os.PathLike | str) -> dict:
    """Validate a persisted/runtime CMTK artifact bundle and its checksums."""
    root = Path(bundle_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise CMTKError(f"CMTK artifact bundle is missing manifest.json: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CMTKError(f"Could not read CMTK artifact manifest: {manifest_path}") from exc
    if manifest.get("format") != "MADI3D CMTK Registration Artifacts":
        raise CMTKError(f"Not a MADI3D CMTK artifact bundle: {root}")

    affine_meta = dict(manifest.get("affine_xform") or {})
    warp_meta = dict(manifest.get("warp_xform") or {})
    affine = _bundle_child(root, affine_meta.get("path", "affine.xform"), "affine")
    warp = _bundle_child(root, warp_meta.get("path", "warp.xform"), "warp")
    if not affine.is_file():
        raise CMTKError(f"CMTK artifact bundle is missing affine transform: {affine}")
    if not warp.exists():
        raise CMTKError(f"CMTK artifact bundle is missing nonlinear transform: {warp}")
    expected_affine = str(affine_meta.get("sha256") or "").lower()
    expected_warp = str(warp_meta.get("sha256") or "").lower()
    if not expected_affine or _artifact_sha256(affine).lower() != expected_affine:
        raise CMTKError(f"CMTK affine transform checksum mismatch: {affine}")
    if not expected_warp or _artifact_sha256(warp).lower() != expected_warp:
        raise CMTKError(f"CMTK warp transform checksum mismatch: {warp}")

    stdout_log = root / str(manifest.get("stdout_log") or "cmtk-stdout.log")
    stderr_log = root / str(manifest.get("stderr_log") or "cmtk-stderr.log")
    return {
        "bundle_dir": root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "affine_xform": affine,
        "warp_xform": warp,
        "stdout_log": stdout_log if stdout_log.is_file() else None,
        "stderr_log": stderr_log if stderr_log.is_file() else None,
    }


def persist_artifact_bundle(
    source_bundle: os.PathLike | str,
    destination_dir: os.PathLike | str,
) -> dict:
    """Copy only portable CMTK transform artifacts into project-owned storage."""
    source = verify_artifact_bundle(source_bundle)
    destination = Path(destination_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=destination.name + ".tmp-", dir=os.fspath(destination.parent)
    ))
    try:
        shutil.copy2(source["affine_xform"], temporary / "affine.xform")
        warp_target = temporary / "warp.xform"
        if source["warp_xform"].is_dir():
            shutil.copytree(source["warp_xform"], warp_target)
        else:
            shutil.copy2(source["warp_xform"], warp_target)
        shutil.copy2(source["manifest_path"], temporary / "manifest.json")
        for key, name in (("stdout_log", "cmtk-stdout.log"), ("stderr_log", "cmtk-stderr.log")):
            path = source.get(key)
            if path is not None and Path(path).is_file():
                shutil.copy2(path, temporary / name)
        verified = verify_artifact_bundle(temporary)
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                raise CMTKError(
                    f"Refusing to replace non-directory CMTK artifact destination: {destination}"
                )
        # Sync clients and scanners can briefly hold the newly copied bundle
        # open on Windows. Retry only sharing violations, without recopying or
        # weakening the verified-directory publication boundary (3.1 s total).
        for attempt in range(6):
            try:
                os.replace(temporary, destination)
                break
            except OSError as exc:
                if getattr(exc, "winerror", None) != 32 or attempt == 5:
                    raise
                time.sleep(0.1 * (2 ** attempt))
        return verify_artifact_bundle(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _first_output_line(proc) -> str:
    text = process_error(proc)
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


class CMTKRegistrationRunner:
    """Run CMTK global linear, nonlinear, and reformat operations for one pair."""

    REQUIRED_TOOLS = ("mat2dof", "dof2mat", "warp", "describe", "reformatx")
    LINEAR_REQUIRED_TOOLS = ("mat2dof", "dof2mat", "registration", "describe")
    REFORMAT_REQUIRED_TOOLS = ("reformatx",)

    def __init__(self, backend, *, executor=run_cmtk_streaming):
        self.backend = backend
        self.executor = executor
        self._validated_tools = set(getattr(backend, "validated_tools", set()) or set())
        self._cmtk_version = str(getattr(backend, "cmtk_version", "") or "")
        self._persisted_ready = bool(getattr(backend, "_madi_persisted_ready", False))

    @staticmethod
    def create_workspace(root: os.PathLike | str, registration_id: str) -> Path:
        root_path = Path(root).resolve()
        token = str(registration_id or "registration").strip() or "registration"
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in token)
        if safe in {"", ".", ".."}:
            safe = "registration"
        workspace = root_path / safe
        workspace.mkdir(parents=True, exist_ok=False)
        return workspace

    def _validate_tools(self, tools=None) -> tuple[list[str], str]:
        requested = tuple(dict.fromkeys(tools or self.REQUIRED_TOOLS))
        missing = tuple(tool for tool in requested if tool not in self._validated_tools)
        details = []
        if missing:
            if self._persisted_ready:
                # Normal application use trusts the persisted dependency state.
                # A genuine command-not-found error invalidates that state at the
                # backend/process boundary; do not preflight every registration.
                self._validated_tools.update(missing)
            else:
                probe = getattr(self.backend, "probe", None)
                if not callable(probe):
                    raise CMTKError("Selected CMTK backend cannot validate required tools.")
                ok, details = probe(missing)
                if not ok:
                    detail = "\n".join(str(v) for v in details if str(v).strip())
                    raise CMTKError(
                        "Required CMTK tools are not available."
                        + (f"\n{detail}" if detail else "")
                    )
                self._validated_tools.update(missing)
        if not self._cmtk_version and not self._persisted_ready:
            if "warp" in requested:
                version_tool = "warp"
            elif "registration" in requested:
                version_tool = "registration"
            else:
                version_tool = requested[0]
            proc = self.backend.run(
                version_tool,
                ["--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
            if getattr(proc, "returncode", 1) != 0:
                raise CMTKError(process_error(proc) or "Could not read CMTK version.")
            self._cmtk_version = _first_output_line(proc)
        return list(details), self._cmtk_version

    def _validate_transform(self, xform: Path) -> str:
        proc = self.backend.run(
            "describe",
            [self.backend.translate_path(os.fspath(xform))],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if getattr(proc, "returncode", 1) != 0:
            raise CMTKError(
                process_error(proc)
                or f"CMTK could not read generated transform: {xform}"
            )
        return process_error(proc)

    def run_linear(
        self,
        *,
        reference_image: os.PathLike | str,
        floating_image: os.PathLike | str,
        moving_to_reference,
        reference_grid: dict,
        floating_grid: dict,
        workspace: os.PathLike | str,
        settings: CMTKLinearSettings,
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
    ) -> CMTKLinearRegistrationResult:
        """Refine a MADI moving->reference affine with CMTK ``registration``.

        The initializer is written through ``write_affine_xform`` so CMTK sees
        its native reference->floating direction. The optimized transform is
        read back through ``read_affine_xform`` and returned as the canonical
        MADI moving->reference 4x4 matrix.
        """
        settings = settings.validated()
        reference = Path(reference_image).resolve()
        floating = Path(floating_image).resolve()
        work = Path(workspace).resolve()
        if not reference.is_file():
            raise FileNotFoundError(f"CMTK reference image does not exist: {reference}")
        if not floating.is_file():
            raise FileNotFoundError(f"CMTK floating image does not exist: {floating}")
        work.mkdir(parents=True, exist_ok=True)

        _tool_details, cmtk_version = self._validate_tools(self.LINEAR_REQUIRED_TOOLS)
        initial_xform = work / "linear-initial.xform"
        output_archive = work / "linear.pending.list"
        output_xform = work / "linear.xform"
        stdout_log = work / "cmtk-linear-stdout.log"
        stderr_log = work / "cmtk-linear-stderr.log"
        for name in _LINEAR_ARTIFACT_NAMES:
            _clear_linear_artifact(work, name)

        result = None
        args = None
        describe_text = ""
        optimized_moving_to_reference = None
        serialization_qc = {}

        def capture_serialization_qc(qc: AffineRoundTripQC):
            serialization_qc.update(asdict(qc))
            if qc.status in {"warning", "failed"} and on_stderr is not None:
                on_stderr(
                    f"CMTK affine serialization QC {qc.status}:\n" + qc.diagnostic()
                )

        try:
            write_affine_xform(
                self.backend,
                moving_to_reference,
                initial_xform,
                verify=True,
                reference_grid=reference_grid,
                floating_grid=floating_grid,
                on_qc=capture_serialization_qc,
            )
            args = build_linear_registration_args(
                self.backend,
                reference,
                floating,
                initial_xform,
                output_archive,
                settings,
            )
            with stdout_log.open("w", encoding="utf-8") as stdout_handle, \
                    stderr_log.open("w", encoding="utf-8") as stderr_handle:

                def stdout_tee(text):
                    stdout_handle.write(str(text))
                    stdout_handle.flush()
                    if on_stdout is not None:
                        on_stdout(text)

                def stderr_tee(text):
                    stderr_handle.write(str(text))
                    stderr_handle.flush()
                    if on_stderr is not None:
                        on_stderr(text)

                result = self.executor(
                    self.backend,
                    "registration",
                    args,
                    on_stdout=stdout_tee,
                    on_stderr=stderr_tee,
                    cancel_check=cancel_check,
                    timeout=timeout,
                    check=True,
                )

            # CMTK ``registration --outlist`` writes a StudyList archive, not a
            # standalone affine file.  For linear registration the archive's
            # canonical affine member is the uncompressed ``registration``
            # typedstream file beside ``studylist``.  Copy that affine into
            # MADI's standalone result path before using the existing transform
            # bridge, then discard the temporary archive.
            archive_affine = output_archive / "registration"
            studylist = output_archive / "studylist"
            if (
                not output_archive.is_dir()
                or not studylist.is_file()
                or not archive_affine.is_file()
                or archive_affine.stat().st_size <= 0
            ):
                command_text = json.dumps(
                    list(getattr(result, "command", ()) or ()), ensure_ascii=False
                )
                entries = (
                    ", ".join(sorted(path.name for path in output_archive.iterdir()))
                    if output_archive.is_dir() else "<archive not created>"
                )
                raise CMTKError(
                    "CMTK registration returned without writing a valid affine study-list archive.\n"
                    f"Requested archive: {output_archive}\n"
                    f"Archive entries: {entries or '<empty>'}\n"
                    f"Return code: {getattr(result, 'returncode', None)}\n"
                    f"Command argv: {command_text}"
                )

            shutil.copy2(archive_affine, output_xform)
            describe_text = self._validate_transform(output_xform)
            optimized_moving_to_reference = matrix4(
                read_affine_xform(
                    self.backend, output_xform, direction="madi", timeout=60
                )
            )
            _clear_linear_artifact(work, "linear.pending.list")
        except BaseException:
            for name in ("linear-initial.xform", "linear.pending.list", "linear.xform"):
                try:
                    _clear_linear_artifact(work, name)
                except Exception:
                    pass
            raise

        return CMTKLinearRegistrationResult(
            workspace=work,
            reference_image=reference,
            floating_image=floating,
            initial_xform=initial_xform,
            output_xform=output_xform,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            moving_to_reference=np.asarray(optimized_moving_to_reference, dtype=np.float64).copy(),
            settings=settings,
            dof_sequence=tuple(int(v) for v in settings.dof_sequence),
            cmtk_version=cmtk_version,
            describe=describe_text,
            command=tuple(result.command),
            stdout=result.stdout,
            stderr=result.stderr,
            affine_serialization_qc=dict(serialization_qc),
        )


    def run_jacobian_qc(
        self,
        *,
        reference_image: os.PathLike | str,
        xforms,
        output_image: os.PathLike | str,
        correct_global: bool = True,
        threads: int = 1,
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
    ) -> dict[str, object]:
        """Generate and summarize a CMTK Jacobian determinant map."""
        reference = Path(reference_image).resolve()
        output = Path(output_image).resolve()
        transform_values = (
            [xforms]
            if isinstance(xforms, (str, os.PathLike))
            else list(xforms or ())
        )
        transforms = [Path(value).resolve() for value in transform_values]
        if not reference.is_file():
            raise FileNotFoundError(f"CMTK Jacobian reference image does not exist: {reference}")
        if not transforms:
            raise ValueError("CMTK Jacobian QC requires at least one transform.")
        for transform in transforms:
            if not transform.exists():
                raise FileNotFoundError(f"CMTK Jacobian transform does not exist: {transform}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if output.is_dir():
                raise CMTKError(f"Refusing to replace directory with Jacobian map: {output}")
            output.unlink()
        self._validate_tools(self.REFORMAT_REQUIRED_TOOLS)
        args = build_jacobian_args(
            self.backend,
            reference,
            transforms,
            output,
            correct_global=correct_global,
            threads=threads,
        )
        result = self.executor(
            self.backend,
            "reformatx",
            args,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            cancel_check=cancel_check,
            timeout=timeout,
            check=True,
        )
        if not output.is_file() or output.stat().st_size <= 0:
            raise CMTKError(
                "CMTK reformatx returned without writing the Jacobian QC map.\n"
                f"Requested output: {output}\n"
                f"Return code: {getattr(result, 'returncode', None)}"
            )
        array, _header = read_nrrd_zyx(output)
        summary = summarize_jacobian(array)
        summary["global_scale_corrected"] = bool(correct_global)
        summary["correct_global_requested"] = bool(correct_global)
        summary["command"] = list(getattr(result, "command", ()) or ())
        return summary


    def run_reformat(
        self,
        *,
        reference_image: os.PathLike | str,
        floating_image: os.PathLike | str,
        xforms,
        output_image: os.PathLike | str,
        interpolation: str = "linear",
        output_type: Optional[str] = None,
        pad_out: float = 0.0,
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
    ) -> CMTKProcessResult:
        reference = Path(reference_image).resolve()
        floating = Path(floating_image).resolve()
        transform_values = (
            [xforms]
            if isinstance(xforms, (str, os.PathLike))
            else list(xforms or ())
        )
        transforms = [Path(value).resolve() for value in transform_values]
        output = Path(output_image).resolve()
        if not reference.is_file():
            raise FileNotFoundError(f"CMTK reference image does not exist: {reference}")
        if not floating.is_file():
            raise FileNotFoundError(f"CMTK floating image does not exist: {floating}")
        if not transforms:
            raise ValueError("CMTK reformat requires at least one transform.")
        for transform in transforms:
            if not transform.exists():
                raise FileNotFoundError(f"CMTK transform does not exist: {transform}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if output.is_dir():
                raise CMTKError(f"Refusing to replace directory with reformat output: {output}")
            output.unlink()
        self._validate_tools(self.REFORMAT_REQUIRED_TOOLS)
        args = build_reformatx_args(
            self.backend, reference, floating, transforms, output,
            interpolation=interpolation, output_type=output_type, pad_out=pad_out,
        )
        result = self.executor(
            self.backend,
            "reformatx",
            args,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            cancel_check=cancel_check,
            timeout=timeout,
            check=True,
        )
        if not output.is_file():
            raise CMTKError(
                "CMTK reformatx exited successfully but did not create the requested "
                f"output: {output}"
            )
        return result

    def run_warp(
        self,
        *,
        reference_image: os.PathLike | str,
        floating_image: os.PathLike | str,
        moving_to_reference,
        reference_grid: dict,
        floating_grid: dict,
        workspace: os.PathLike | str,
        settings: CMTKWarpSettings,
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
    ) -> CMTKRegistrationArtifacts:
        settings = settings.validated()
        reference = Path(reference_image).resolve()
        floating = Path(floating_image).resolve()
        work = Path(workspace).resolve()
        if not reference.is_file():
            raise FileNotFoundError(f"CMTK reference image does not exist: {reference}")
        if not floating.is_file():
            raise FileNotFoundError(f"CMTK floating image does not exist: {floating}")
        work.mkdir(parents=True, exist_ok=True)

        tool_details, cmtk_version = self._validate_tools()

        affine = work / "affine.xform"
        affine_pending = work / "affine.pending.xform"
        warp = work / "warp.xform"
        warp_pending = work / "warp.pending.xform"
        manifest = work / "manifest.json"
        stdout_log = work / "cmtk-stdout.log"
        stderr_log = work / "cmtk-stderr.log"

        for name in _ARTIFACT_NAMES:
            _clear_artifact(work, name)

        reference_identity = _input_identity(reference)
        floating_identity = _input_identity(floating)
        started = time.time()
        result = None
        args = None
        execution_warp_path = None
        serialization_qc = {}

        def capture_serialization_qc(qc: AffineRoundTripQC):
            serialization_qc.update(asdict(qc))
            if qc.status in {"warning", "failed"} and on_stderr is not None:
                on_stderr(
                    f"CMTK affine serialization QC {qc.status}:\n" + qc.diagnostic()
                )

        try:
            write_affine_xform(
                self.backend,
                moving_to_reference,
                affine_pending,
                verify=True,
                reference_grid=reference_grid,
                floating_grid=floating_grid,
                on_qc=capture_serialization_qc,
            )

            # ``warp --outlist`` is a CMTK typed-stream directory archive.
            # CMTK 3.3.x can read ordinary files from a Windows DrvFS mount,
            # but its multi-file archive writer can silently leave an empty
            # directory there and still return code 0. Backends that provide a
            # native execution workspace (managed WSL) therefore write the
            # archive on their native filesystem first; the completed bundle is
            # copied back atomically at the artifact boundary. Native backends
            # continue to write directly into the requested local workspace.
            execution_warp_path = self.backend.create_execution_temp_dir(
                "madi3d-cmtk-warp-"
            )
            if execution_warp_path is None:
                warp_pending.mkdir(parents=False, exist_ok=False)

            args = build_warp_args(
                self.backend,
                reference,
                floating,
                affine_pending,
                warp_pending,
                settings,
                output_xform_backend_path=execution_warp_path,
            )

            with stdout_log.open("w", encoding="utf-8") as stdout_handle, \
                    stderr_log.open("w", encoding="utf-8") as stderr_handle:

                def stdout_tee(text):
                    stdout_handle.write(str(text))
                    stdout_handle.flush()
                    if on_stdout is not None:
                        on_stdout(text)

                def stderr_tee(text):
                    stderr_handle.write(str(text))
                    stderr_handle.flush()
                    if on_stderr is not None:
                        on_stderr(text)

                result: CMTKProcessResult = self.executor(
                    self.backend,
                    "warp",
                    args,
                    on_stdout=stdout_tee,
                    on_stderr=stderr_tee,
                    cancel_check=cancel_check,
                    timeout=timeout,
                    check=True,
                )

            if execution_warp_path is not None:
                self.backend.copy_execution_tree_to_local(
                    execution_warp_path, warp_pending
                )

            payload_candidates = (
                warp_pending / "registration",
                warp_pending / "registration.gz",
            )
            transform_payload = next(
                (
                    candidate for candidate in payload_candidates
                    if candidate.is_file() and candidate.stat().st_size > 0
                ),
                None,
            )
            if transform_payload is None:
                try:
                    workspace_entries = sorted(
                        child.name + ("/" if child.is_dir() else "")
                        for child in work.iterdir()
                    )
                except Exception as exc:
                    workspace_entries = [f"<could not list workspace: {exc}>"]
                try:
                    archive_entries = sorted(
                        child.name + ("/" if child.is_dir() else "")
                        for child in warp_pending.iterdir()
                    )
                except Exception as exc:
                    archive_entries = [f"<could not list archive: {exc}>"]
                command_text = json.dumps(
                    list(getattr(result, "command", ()) or ()),
                    ensure_ascii=False,
                )
                stdout_tail = str(getattr(result, "stdout", "") or "").strip()
                stderr_tail = str(getattr(result, "stderr", "") or "").strip()
                diagnostics = [
                    "CMTK warp returned without writing a valid transform archive.",
                    f"Requested transform: {warp_pending}",
                    f"Return code: {getattr(result, 'returncode', None)}",
                    f"Command argv: {command_text}",
                    "Workspace entries: " + (", ".join(workspace_entries) if workspace_entries else "<empty>"),
                    "Transform archive entries: " + (", ".join(archive_entries) if archive_entries else "<empty>"),
                ]
                if stdout_tail:
                    diagnostics.append("CMTK stdout tail:\n" + stdout_tail)
                if stderr_tail:
                    diagnostics.append("CMTK stderr tail:\n" + stderr_tail)
                raise CMTKError("\n".join(diagnostics))

            describe_text = self._validate_transform(warp_pending)
            os.replace(affine_pending, affine)
            os.replace(warp_pending, warp)

            payload = {
                "format": "MADI3D CMTK Registration Artifacts",
                "format_version": 2,
                "direction": (
                    "input moving_to_reference; CMTK transform reference_to_floating"
                ),
                "backend": getattr(
                    self.backend,
                    "label",
                    getattr(self.backend, "kind", "CMTK"),
                ),
                "cmtk_version": cmtk_version,
                "validated_tools": tool_details,
                "reference_image": reference_identity,
                "floating_image": floating_identity,
                "affine_xform": {
                    "path": affine.name,
                    "sha256": _artifact_sha256(affine),
                },
                "warp_xform": {
                    "path": warp.name,
                    "sha256": _artifact_sha256(warp),
                    "describe": describe_text,
                },
                "stdout_log": stdout_log.name,
                "stderr_log": stderr_log.name,
                "settings": asdict(settings),
                "cmtk_args": list(args),
                "command": list(result.command),
                "returncode": int(result.returncode),
                "affine_serialization_qc": dict(serialization_qc),
                "started_unix": started,
                "finished_unix": time.time(),
            }
            tmp = work / "manifest.json.tmp"
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, manifest)

        except BaseException:
            for name in (
                "affine.pending.xform",
                "warp.pending.xform",
                "affine.xform",
                "warp.xform",
                "manifest.json",
                "manifest.json.tmp",
            ):
                try:
                    _clear_artifact(work, name)
                except Exception:
                    pass
            raise
        finally:
            if execution_warp_path is not None:
                try:
                    self.backend.remove_execution_temp_dir(execution_warp_path)
                except Exception as cleanup_exc:
                    warning = (
                        "WARNING: could not remove native CMTK execution workspace "
                        f"{execution_warp_path}: {cleanup_exc}\n"
                    )
                    try:
                        with stderr_log.open("a", encoding="utf-8") as handle:
                            handle.write(warning)
                    except Exception:
                        pass
                    if on_stderr is not None:
                        try:
                            on_stderr(warning)
                        except Exception:
                            pass

        return CMTKRegistrationArtifacts(
            workspace=work,
            reference_image=reference,
            floating_image=floating,
            affine_xform=affine,
            warp_xform=warp,
            manifest_path=manifest,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            cmtk_version=cmtk_version,
            command=tuple(result.command),
            stdout=result.stdout,
            stderr=result.stderr,
            affine_serialization_qc=dict(serialization_qc),
        )


__all__ = [
    "CMTKLinearRegistrationResult",
    "CMTKLinearSettings",
    "CMTKRegistrationArtifacts",
    "CMTKRegistrationRunner",
    "CMTKWarpSettings",
    "build_jacobian_args",
    "build_linear_registration_args",
    "build_reformatx_args",
    "build_warp_args",
    "canonical_cmtk_grid",
    "cmtk_neutral_to_working_matrix",
    "cmtk_output_type_for_dtype",
    "cmtk_working_moving_to_reference",
    "persist_artifact_bundle",
    "summarize_jacobian",
    "update_artifact_bundle_qc",
    "read_nrrd_zyx",
    "verify_artifact_bundle",
    "write_reference_grid_nrrd",
    "write_volume_nrrd",
    "write_working_nrrd",
]
