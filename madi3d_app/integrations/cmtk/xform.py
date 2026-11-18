# -*- coding: utf-8 -*-
"""Canonical affine conversion between MADI3D and CMTK.

MADI3D stores affine registration results as moving/floating -> reference
matrices. CMTK transformations consumed by reformatx map continuous reference
coordinates -> continuous floating coordinates, so the two directions are exact
inverses.

CMTK's Matrix4x4 text boundary is column-first. MADI3D therefore always uses the
same explicit ``--transpose`` option for both ``mat2dof`` input and ``dof2mat``
output. There is no runtime convention probing.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from madi3d_app.integrations.cmtk.backend import CMTKError, decode_console


MADI_DIRECTION = "moving_to_reference"
CMTK_DIRECTION = "reference_to_floating"
MAT2DOF_INPUT_TRANSPOSE = True
DOF2MAT_OUTPUT_TRANSPOSE = True

# Serialization QC limits. These are deliberately much tighter than biological
# registration tolerances: the CMTK text/DOF boundary must add negligible error.
AFFINE_ROUNDTRIP_SILENT_VOXELS = 0.05
AFFINE_ROUNDTRIP_MAX_VOXELS = 0.25
AFFINE_ROUNDTRIP_MAX_WORKING_UNITS = 0.1
AFFINE_CONDITION_WARNING = 1.0e8
AFFINE_CONDITION_FAILURE = 1.0e12

# One bounded CMTK stream inside a single-line affine serialization failure.
# Failures reach the user through the activity log and dialog, so the complete
# subprocess text must not displace the identity of the failed operation.
FAILURE_DETAIL_LIMIT = 900

DEFAULT_SERIALIZATION_OPERATION = "affine transform"
DEFAULT_READ_OPERATION = "affine transform read"


@dataclass(frozen=True)
class AffineRoundTripQC:
    max_matrix_element_error: float
    matrix_frobenius_error: float
    max_forward_displacement_working_units: float
    max_forward_voxel_displacement: float
    max_inverse_displacement_working_units: float
    max_inverse_voxel_displacement: float
    affine_condition_number: float
    thresholds: dict[str, float]
    status: str

    def diagnostic(self) -> str:
        return "\n".join((
            f"max matrix element error:        {self.max_matrix_element_error:.9e}",
            f"matrix Frobenius error:          {self.matrix_frobenius_error:.9e}",
            f"max forward displacement:        {self.max_forward_displacement_working_units:.9e} working units",
            f"max forward voxel displacement:  {self.max_forward_voxel_displacement:.9e} voxel",
            f"max inverse displacement:        {self.max_inverse_displacement_working_units:.9e} working units",
            f"max inverse voxel displacement:  {self.max_inverse_voxel_displacement:.9e} voxel",
            f"affine condition number:         {self.affine_condition_number:.9e}",
            f"status:                          {self.status}",
        ))


def matrix4(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != 16:
        raise ValueError("Expected a 4x4 affine matrix.")
    arr = arr.reshape(4, 4)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Affine matrix contains non-finite values.")
    if not np.allclose(arr[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise ValueError("Affine matrix must have homogeneous last row [0, 0, 0, 1].")
    try:
        np.linalg.inv(arr)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Affine matrix is singular.") from exc
    return arr


def matrix_text(value) -> str:
    arr = matrix4(value)
    return "\n".join(
        " ".join(f"{arr[row, col]:.17g}" for col in range(4))
        for row in range(4)
    ) + "\n"


def affine_diagnostics(value) -> dict:
    """Measure the numeric properties that decide whether CMTK can use a matrix.

    Real CMTK rejects a matrix whose 3x3 block its own constructor cannot invert,
    which happens for finite matrices that remain numerically invertible in
    double precision and are therefore valid under ``matrix4()``. These values are
    failure diagnostics only: they never accept, reject, clamp or repair a
    transform, and MADI3D's validity gate stays ``matrix4()``.
    """
    arr = np.asarray(value, dtype=np.float64)
    arr = arr.reshape(4, 4) if arr.size == 16 else np.full((4, 4), np.nan)
    with np.errstate(all="ignore"):
        try:
            determinant = float(np.linalg.det(arr[:3, :3]))
            singular = sorted(
                float(v) for v in np.linalg.svd(arr[:3, :3], compute_uv=False)
            )
        except np.linalg.LinAlgError:
            determinant = float("nan")
            singular = [float("nan")] * 3
    smallest, largest = singular[0], singular[-1]
    condition = float(largest / smallest) if smallest > 0.0 else float("inf")
    return {
        "determinant": determinant,
        "singular_values": singular,
        "condition_number": condition,
        "translation_magnitude": float(np.linalg.norm(arr[:3, 3])),
        "finite": bool(np.all(np.isfinite(arr))),
        "invertible": bool(math.isfinite(determinant) and determinant != 0.0),
    }


def parse_matrix(text: str) -> np.ndarray:
    values = []
    for token in str(text or "").replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    if len(values) < 16:
        raise CMTKError("Could not parse a 4x4 matrix from CMTK dof2mat output.")
    return matrix4(np.asarray(values[:16], dtype=np.float64).reshape(4, 4))


def madi_to_cmtk_matrix(moving_to_reference) -> np.ndarray:
    """Convert MADI moving->reference to CMTK reference->floating."""
    return np.linalg.inv(matrix4(moving_to_reference))


def cmtk_to_madi_matrix(reference_to_floating) -> np.ndarray:
    """Convert CMTK reference->floating to MADI moving->reference."""
    return np.linalg.inv(matrix4(reference_to_floating))


def _clear_file_output(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        raise CMTKError(f"Refusing to replace directory with affine file: {path}")
    path.unlink()


def _transpose_args(enabled: bool) -> list[str]:
    return ["--transpose"] if enabled else []


def _bounded_stream(value) -> str:
    """Collapse one CMTK stream to one bounded line without dropping its content."""
    text = " ".join(decode_console(value).split())
    if len(text) > FAILURE_DETAIL_LIMIT:
        text = text[:FAILURE_DETAIL_LIMIT].rstrip() + " ...[truncated]"
    return text


def _matrix_line(value) -> str:
    """Render one complete 4x4 affine on a single line at full double precision."""
    arr = np.asarray(value, dtype=np.float64)
    arr = arr.reshape(4, 4) if arr.size == 16 else np.full((4, 4), np.nan)
    return "[" + "; ".join(
        " ".join(f"{arr[row, col]:.17g}" for col in range(4)) for row in range(4)
    ) + "]"


def _failure(
    tool: str,
    proc,
    backend,
    *,
    operation: str,
    extra: tuple[str, ...] = (),
    cause: str = "",
) -> CMTKError:
    """Compose one diagnosable single-line failure for a failed CMTK affine call.

    The identity of the operation, the executable, its exit code, the backend, and
    whatever the process wrote are all reported so a missing tool, an unusable
    affine, a process exit failure, an output-path failure, and a round-trip
    verification failure remain distinguishable without source-level debugging.
    """
    stderr = _bounded_stream(getattr(proc, "stderr", ""))
    stdout = _bounded_stream(getattr(proc, "stdout", ""))
    if stderr and stdout:
        output = f"stderr: {stderr} | stdout: {stdout}"
    elif stderr:
        output = f"stderr: {stderr}"
    elif stdout:
        output = f"stdout: {stdout}"
    else:
        output = "no output captured (CMTK wrote nothing to stdout or stderr)"
    head = (
        f"CMTK {tool} failed for {operation} "
        f"(exit code {getattr(proc, 'returncode', None)})"
    )
    if cause:
        head = f"{head}: {cause}"
    backend_label = str(
        getattr(backend, "label", "") or getattr(backend, "kind", "") or "CMTK"
    )
    return CMTKError(" | ".join((
        head,
        f"backend: {backend_label}",
        *(str(v) for v in extra),
        output,
    )))


def _serialization_failure(
    tool: str,
    proc,
    backend,
    *,
    operation: str,
    output: Path,
    translated_output: str,
    matrix: np.ndarray,
) -> CMTKError:
    """Add the rejected affine and its numeric condition to a serialization failure."""
    diagnostics = affine_diagnostics(matrix)
    return _failure(
        tool,
        proc,
        backend,
        operation=operation,
        extra=(
            f"output: {output}",
            f"CMTK output path: {translated_output}",
            f"matrix: {_matrix_line(matrix)}",
            "linear determinant: "
            f"{diagnostics['determinant']:.6g}",
            "singular values: ["
            + ", ".join(f"{v:.6g}" for v in diagnostics["singular_values"]) + "]",
            f"condition number: {diagnostics['condition_number']:.6g}",
            f"translation magnitude: {diagnostics['translation_magnitude']:.6g}",
            f"finite and invertible: {diagnostics['finite'] and diagnostics['invertible']}",
        ),
    )


def _grid_geometry(grid, label: str):
    if not isinstance(grid, dict):
        raise ValueError(f"{label} grid metadata is required for affine round-trip QC.")
    dims = np.asarray(grid.get("dims_xyz", ()), dtype=np.int64)
    origin = np.asarray(grid.get("origin", ()), dtype=np.float64)
    spacing = np.asarray(grid.get("spacing", ()), dtype=np.float64)
    direction = np.asarray(grid.get("direction", np.eye(3)), dtype=np.float64)
    if dims.shape != (3,) or np.any(dims < 1):
        raise ValueError(f"Invalid {label} grid dimensions: {grid.get('dims_xyz')!r}.")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError(f"Invalid {label} grid origin.")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise ValueError(f"Invalid {label} grid spacing.")
    if direction.shape != (3, 3) or not np.all(np.isfinite(direction)):
        raise ValueError(f"Invalid {label} grid direction.")
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6, rtol=1e-6):
        raise ValueError(f"{label.capitalize()} grid direction must be orthonormal.")
    return dims, origin, spacing, direction


def _physical_corners(geometry) -> np.ndarray:
    dims, origin, spacing, direction = geometry
    indices = np.asarray(
        list(itertools.product(*((0.0, float(size - 1)) for size in dims))),
        dtype=np.float64,
    )
    return origin + (direction @ (indices * spacing).T).T


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (matrix[:3, :3] @ points.T).T + matrix[:3, 3]


def _max_displacements(
    first: np.ndarray,
    second: np.ndarray,
    output_geometry,
) -> tuple[float, float]:
    delta = second - first
    physical = np.linalg.norm(delta, axis=1)
    _dims, _origin, spacing, direction = output_geometry
    voxel_delta = (direction.T @ delta.T).T / spacing
    voxel = np.linalg.norm(voxel_delta, axis=1)
    return float(np.max(physical)), float(np.max(voxel))


def evaluate_affine_roundtrip(
    original,
    recovered,
    *,
    reference_grid: dict,
    floating_grid: dict,
) -> AffineRoundTripQC:
    """Measure CMTK reference->floating serialization error over both domains."""
    native = matrix4(original)
    restored = matrix4(recovered)
    reference = _grid_geometry(reference_grid, "reference")
    floating = _grid_geometry(floating_grid, "floating")

    reference_corners = _physical_corners(reference)
    forward_working, forward_voxels = _max_displacements(
        _transform_points(native, reference_corners),
        _transform_points(restored, reference_corners),
        floating,
    )
    native_inverse = np.linalg.inv(native)
    restored_inverse = np.linalg.inv(restored)
    floating_corners = _physical_corners(floating)
    inverse_working, inverse_voxels = _max_displacements(
        _transform_points(native_inverse, floating_corners),
        _transform_points(restored_inverse, floating_corners),
        reference,
    )
    delta = restored - native
    condition = max(
        float(np.linalg.cond(native[:3, :3])),
        float(np.linalg.cond(restored[:3, :3])),
    )
    failed = (
        max(forward_working, inverse_working) > AFFINE_ROUNDTRIP_MAX_WORKING_UNITS
        or max(forward_voxels, inverse_voxels) > AFFINE_ROUNDTRIP_MAX_VOXELS
        or not np.isfinite(condition)
        or condition >= AFFINE_CONDITION_FAILURE
    )
    warning = (
        max(forward_voxels, inverse_voxels) > AFFINE_ROUNDTRIP_SILENT_VOXELS
        or condition >= AFFINE_CONDITION_WARNING
    )
    return AffineRoundTripQC(
        max_matrix_element_error=float(np.max(np.abs(delta))),
        matrix_frobenius_error=float(np.linalg.norm(delta)),
        max_forward_displacement_working_units=forward_working,
        max_forward_voxel_displacement=forward_voxels,
        max_inverse_displacement_working_units=inverse_working,
        max_inverse_voxel_displacement=inverse_voxels,
        affine_condition_number=condition,
        thresholds={
            "silent_max_voxel_displacement": AFFINE_ROUNDTRIP_SILENT_VOXELS,
            "failure_max_voxel_displacement": AFFINE_ROUNDTRIP_MAX_VOXELS,
            "failure_max_working_unit_displacement": AFFINE_ROUNDTRIP_MAX_WORKING_UNITS,
            "condition_warning": AFFINE_CONDITION_WARNING,
            "condition_failure": AFFINE_CONDITION_FAILURE,
        },
        status="failed" if failed else ("warning" if warning else "passed"),
    )


def write_cmtk_matrix_xform(
    backend,
    cmtk_matrix,
    output_path: os.PathLike | str,
    *,
    operation: str = DEFAULT_SERIALIZATION_OPERATION,
    verify: bool = True,
    reference_grid: Optional[dict] = None,
    floating_grid: Optional[dict] = None,
    on_qc: Optional[Callable[[AffineRoundTripQC], None]] = None,
    timeout: float = 60.0,
) -> Path:
    """Write one explicit CMTK-space affine matrix with ``mat2dof``.

    ``operation`` names the transform being serialized, such as one Reformat frame
    bridge, so a failure identifies which bridge could not be written.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _clear_file_output(output)

    native_matrix = matrix4(cmtk_matrix)
    translated_output = backend.translate_path(str(output))
    args = _transpose_args(MAT2DOF_INPUT_TRANSPOSE) + ["--output", translated_output]
    proc = backend.run(
        "mat2dof",
        args,
        input=matrix_text(native_matrix),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if getattr(proc, "returncode", 1) != 0:
        if output.is_file():
            output.unlink()
        raise _serialization_failure(
            "mat2dof",
            proc,
            backend,
            operation=operation,
            output=output,
            translated_output=translated_output,
            matrix=native_matrix,
        )
    if not output.is_file():
        raise CMTKError(
            f"CMTK mat2dof exited successfully for {operation} but did not create "
            f"{output} (CMTK output path: {translated_output}); the CMTK backend "
            "could not write the requested affine file."
        )

    if verify:
        try:
            recovered = read_affine_xform(
                backend,
                output,
                timeout=timeout,
                direction="cmtk",
                operation=f"{operation} round-trip",
            )
            if reference_grid is None or floating_grid is None:
                raise CMTKError(
                    f"CMTK affine round-trip validation for {operation} requires both "
                    "reference and floating grid metadata."
                )
            qc = evaluate_affine_roundtrip(
                native_matrix,
                recovered,
                reference_grid=reference_grid,
                floating_grid=floating_grid,
            )
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        if on_qc is not None:
            try:
                on_qc(qc)
            except BaseException:
                output.unlink(missing_ok=True)
                raise
    return output


def write_affine_xform(
    backend,
    moving_to_reference,
    output_path: os.PathLike | str,
    *,
    operation: str = DEFAULT_SERIALIZATION_OPERATION,
    verify: bool = True,
    reference_grid: Optional[dict] = None,
    floating_grid: Optional[dict] = None,
    on_qc: Optional[Callable[[AffineRoundTripQC], None]] = None,
    timeout: float = 60.0,
) -> Path:
    """Write MADI moving->reference as CMTK reference->floating."""
    return write_cmtk_matrix_xform(
        backend,
        madi_to_cmtk_matrix(moving_to_reference),
        output_path,
        operation=operation,
        verify=verify,
        reference_grid=reference_grid,
        floating_grid=floating_grid,
        on_qc=on_qc,
        timeout=timeout,
    )


def read_affine_xform(
    backend,
    xform_path: os.PathLike | str,
    *,
    timeout: float = 60.0,
    direction: str = "madi",
    operation: str = DEFAULT_READ_OPERATION,
) -> np.ndarray:
    """Read a CMTK affine file into MADI3D's ordinary row-major convention."""
    path = Path(xform_path)
    if not path.is_file():
        raise FileNotFoundError(f"CMTK affine transform does not exist: {path}")

    args = _transpose_args(DOF2MAT_OUTPUT_TRANSPOSE) + [
        backend.translate_path(os.fspath(path))
    ]
    proc = backend.run(
        "dof2mat",
        args,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if getattr(proc, "returncode", 1) != 0:
        raise _failure("dof2mat", proc, backend, operation=operation)

    try:
        native = parse_matrix(getattr(proc, "stdout", ""))
    except CMTKError as exc:
        raise _failure(
            "dof2mat", proc, backend, operation=operation, cause=str(exc)
        ) from exc
    mode = str(direction or "madi").lower()
    if mode in {"cmtk", "reference_to_floating", "native"}:
        return native
    if mode in {"madi", "moving_to_reference"}:
        return cmtk_to_madi_matrix(native)
    raise ValueError(f"Unknown affine direction: {direction}")


__all__ = [
    "AFFINE_CONDITION_FAILURE",
    "AFFINE_CONDITION_WARNING",
    "AFFINE_ROUNDTRIP_MAX_WORKING_UNITS",
    "AFFINE_ROUNDTRIP_MAX_VOXELS",
    "AFFINE_ROUNDTRIP_SILENT_VOXELS",
    "AffineRoundTripQC",
    "CMTK_DIRECTION",
    "DOF2MAT_OUTPUT_TRANSPOSE",
    "MADI_DIRECTION",
    "MAT2DOF_INPUT_TRANSPOSE",
    "affine_diagnostics",
    "cmtk_to_madi_matrix",
    "evaluate_affine_roundtrip",
    "madi_to_cmtk_matrix",
    "matrix4",
    "matrix_text",
    "parse_matrix",
    "read_affine_xform",
    "write_affine_xform",
    "write_cmtk_matrix_xform",
]
