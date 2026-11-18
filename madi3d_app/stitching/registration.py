"""Stitching pair estimation, pose-graph solving and registration execution."""
from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from madi3d_app.stitching.models import StitchingRegistrationResult, StitchingRejection
from madi3d_app.stitching.correlation import (
    DEFAULT_PHASE_PEAK_COUNT,
    check_cancelled,
    evaluate_translations,
)
from madi3d_app.volume.geometry import (
    affine_support_bounds as _support_bounds,
    grid_affine_from_components,
    invertible_affine4,
)

from . import common


DEFAULT_STITCHING_MINIMUM_NCC = 0.05
DEFAULT_STITCHING_MAX_ANGLE_DEG = 3.0
DEFAULT_STITCHING_MAX_SCALE_PERCENT = 5.0
DEFAULT_STITCHING_MAX_SHEAR = 0.03


TRANSLATION_STITCHING_ACCEPTANCE_CONTRACT = {
    "contract_version": "translation-stitching-v1",
    "supported_fixture_accuracy_registration_voxels": 0.5,
    "default_minimum_pair_ncc": DEFAULT_STITCHING_MINIMUM_NCC,
    "pair_acceptance": (
        "Phase-correlation confidence must be numerically valid and final pair "
        "NCC must meet the configured minimum. Ambiguous but valid phase peaks "
        "remain reviewable rather than being treated as universally invalid."
    ),
    "global_inconsistency": (
        "Only cycle-redundant translation edges whose registration-grid residual "
        "is a unique outlier above max(0.5 voxel, median + 3 normalized MAD) "
        "are rejected. Rejection is bounded to eight edges and cannot disconnect "
        "the accepted graph; non-identifiable disagreement remains explicit."
    ),
    "preprocessing": (
        "Masked 1st-to-99.5th percentile normalization, Gaussian background "
        "subtraction, and renormalization. FFT phase-correlation inputs receive a "
        "cosine distance taper at the support boundary while NCC remains untapered. "
        "The deterministic boundary audit reduced mean supported-fixture translation "
        "error relative to a hard mask while retaining the 0.5-voxel worst-case "
        "accuracy contract."
    ),
    "scope": (
        "The 0.5-voxel value is an algorithmic synthetic-fixture accuracy and "
        "global-consistency threshold, not a universal microscopy or biological "
        "quality threshold."
    ),
}


def _robust_normalize(volume, mask=None):
    x = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(x)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    values = x[finite]
    if values.size < 32:
        return np.zeros_like(x, dtype=np.float32)
    positive = values[values > 0]
    if positive.size >= 256:
        values = positive
    lo, hi = np.percentile(values, (1.0, 99.5))
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    out = np.clip((x - float(lo)) / float(hi - lo), 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32, copy=False)


def _registration_preprocess(volume, mask):
    try:
        from scipy import ndimage
    except Exception as exc:
        raise RuntimeError("3-D stitching requires SciPy.") from exc

    x = _robust_normalize(volume, mask)
    if min(x.shape) >= 8:
        background = ndimage.gaussian_filter(x, sigma=2.0)
        x = x - background
        x = _robust_normalize(x, mask)
    x *= np.asarray(mask, dtype=np.float32)
    return np.ascontiguousarray(x)


def _registration_support_taper(volume, mask):
    """Condition support boundaries for FFT phase correlation only."""
    try:
        from scipy import ndimage
    except Exception as exc:
        raise RuntimeError("3-D stitching requires SciPy.") from exc

    x = np.asarray(volume, dtype=np.float32).copy()
    support = np.asarray(mask, dtype=bool)
    padded_support = np.pad(support, 1, constant_values=False)
    distance = ndimage.distance_transform_edt(padded_support)[1:-1, 1:-1, 1:-1]
    taper_width = max(2.0, min(6.0, 0.125 * min(x.shape)))
    support_taper = np.sin(
        0.5 * np.pi * np.clip(distance / taper_width, 0.0, 1.0)
    ) ** 2
    x *= support_taper.astype(np.float32)
    return np.ascontiguousarray(x)


def _ncc(a, b, mask=None):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if np.count_nonzero(valid) < 128:
        return -1.0
    av = a[valid]
    bv = b[valid]
    # Keep registration focused on signal while retaining enough samples.
    signal = (av > np.percentile(av, 40.0)) | (bv > np.percentile(bv, 40.0))
    if np.count_nonzero(signal) >= 128:
        av = av[signal]
        bv = bv[signal]
    av = av - float(av.mean())
    bv = bv - float(bv.mean())
    denom = math.sqrt(float(np.dot(av, av)) * float(np.dot(bv, bv)))
    if denom <= 1e-12:
        return -1.0
    return float(np.dot(av, bv) / denom)


def _phase_correlation_shift(fixed, moving, max_shift_zyx=None):
    """Periodic single-peak diagnostic retained for the boundary regression audit.

    Production registration evaluates linearly padded hypotheses on common
    support. This diagnostic uses the same FFT/peak implementation with periodic
    input semantics, as required by its circular-shift characterization tests.
    """
    from madi3d_app.stitching.correlation import phase_confidence, phase_peaks

    fixed = np.asarray(fixed, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)
    if fixed.shape != moving.shape:
        raise ValueError("Phase-correlation arrays must have the same shape.")
    peaks, _shape = phase_peaks(fixed, moving, count=1, linear=False,
                                max_shift=max_shift_zyx)
    return dict(phase_confidence(peaks, 0),
                shift_zyx=np.array(peaks[0]["shift_zyx"] if peaks else [0., 0., 0.]))


def _translation_matrix_xyz(translation_xyz):
    out = np.eye(4, dtype=float)
    out[:3, 3] = np.asarray(translation_xyz, dtype=float)
    return out


def _rigid_matrix(params, center_xyz=(0.0, 0.0, 0.0)):
    """World correction matrix from tx,ty,tz,rx,ry,rz (degrees)."""
    try:
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        raise RuntimeError("Rigid stitching requires scipy.spatial.transform.") from exc

    p = np.asarray(params, dtype=float)
    t = p[:3]
    rot = Rotation.from_euler("xyz", p[3:6], degrees=True).as_matrix()
    center = np.asarray(center_xyz, dtype=float)
    out = np.eye(4, dtype=float)
    out[:3, :3] = rot
    out[:3, 3] = t + center - rot @ center
    return out


def _proper_rotation_from_linear(matrix3):
    """Closest proper rotation to a rigid or affine 3x3 linear matrix."""
    linear = np.asarray(matrix3, dtype=float).reshape(3, 3)
    u, _s, vh = np.linalg.svd(linear)
    rot = u @ vh
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vh
    return rot


def _rotation_vector_degrees(matrix3):
    try:
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        raise RuntimeError("Rigid stitching requires scipy.spatial.transform.") from exc
    rot = _proper_rotation_from_linear(matrix3)
    return Rotation.from_matrix(rot).as_rotvec() * (180.0 / math.pi)


def _affine_matrix(params, center_xyz=(0.0, 0.0, 0.0)):
    """World correction from 12 parameters.

    Parameters are translation XYZ, Euler rotation XYZ in degrees, logarithmic
    scale XYZ, and upper-triangular shear XY/XZ/YZ. This is a complete
    orientation-preserving 12-DOF affine model without projective terms.
    """
    try:
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        raise RuntimeError("Affine stitching requires scipy.spatial.transform.") from exc

    p = np.asarray(params, dtype=float).reshape(12)
    translation = p[:3]
    rotation = Rotation.from_euler("xyz", p[3:6], degrees=True).as_matrix()
    scales = np.exp(np.clip(p[6:9], -6.0, 6.0))
    shear = np.eye(3, dtype=float)
    shear[0, 1] = p[9]
    shear[0, 2] = p[10]
    shear[1, 2] = p[11]
    linear = rotation @ shear @ np.diag(scales)
    center = np.asarray(center_xyz, dtype=float)
    out = np.eye(4, dtype=float)
    out[:3, :3] = linear
    out[:3, 3] = translation + center - linear @ center
    return out


def _affine_linear_metrics(matrix3):
    """Return approximate rotation, principal scales, and shear magnitudes."""
    linear = np.asarray(matrix3, dtype=float).reshape(3, 3)
    rotation = _proper_rotation_from_linear(linear)
    stretch = rotation.T @ linear
    scales = np.diag(stretch).copy()
    safe = scales.copy()
    safe[np.abs(safe) < 1e-9] = 1.0
    normalized = stretch @ np.diag(1.0 / safe)
    shear = np.array([normalized[0, 1], normalized[0, 2], normalized[1, 2]], dtype=float)
    return rotation, scales, shear


def _candidate_pairs(tiles, search_margin, minimum_overlap_fraction, *, cancelled=None):
    """Find exact expanded-AABB overlaps, in original tile-pair order.

    A sweep excludes disjoint intervals before the existing three-dimensional
    predicate. Dense projected overlap still costs quadratic work; every eligible
    pair remains available, including margin-only overlaps.
    """
    check_cancelled(cancelled)
    tiles = common._prepared_tiles(tiles, cancelled=cancelled)
    margin = max(0.0, float(search_margin))
    min_fraction = max(0.0, float(minimum_overlap_fraction))
    bounds = []
    for tile in tiles:
        check_cancelled(cancelled)
        bounds.append(_support_bounds(tile["world_affine"], tile["dims"]))
    if len(tiles) < 2:
        return []
    lows = np.array([lo for lo, _hi in bounds]) - margin
    highs = np.array([hi for _lo, hi in bounds]) + margin
    # Prefer the axis that separates centers most relative to interval width.
    centers = (lows + highs) * 0.5
    axis = int(np.argmax(np.ptp(centers, axis=0) / np.maximum(
        np.mean(highs - lows, axis=0), 1e-12)))

    def sweep_key(index):
        check_cancelled(cancelled)
        return float(lows[index, axis]), index

    order = sorted(range(len(tiles)), key=sweep_key)
    active = []
    by_first = [[] for _tile in tiles]
    for current in order:
        check_cancelled(cancelled)
        remaining = []
        for previous in active:
            check_cancelled(cancelled)
            if highs[previous, axis] <= lows[current, axis]:
                continue
            remaining.append(previous)
            i, j = (previous, current) if previous < current else (current, previous)
            alo, ahi = bounds[i]
            blo, bhi = bounds[j]
            search_lo = np.maximum(alo - margin, blo - margin)
            search_hi = np.minimum(ahi + margin, bhi + margin)
            if np.any(search_hi <= search_lo):
                continue
            actual_extent = np.maximum(0.0, np.minimum(ahi, bhi) - np.maximum(alo, blo))
            actual_volume = float(np.prod(actual_extent))
            min_volume = max(1e-12, min(float(np.prod(ahi - alo)), float(np.prod(bhi - blo))))
            overlap_fraction = actual_volume / min_volume
            if actual_volume > 0.0 and overlap_fraction < min_fraction:
                continue
            by_first[i].append({
                "i": i, "j": j,
                "fixed_id": tiles[i]["tile_id"], "moving_id": tiles[j]["tile_id"],
                "actual_overlap_fraction": overlap_fraction,
                "search_lo": search_lo, "search_hi": search_hi,
            })
        remaining.append(current)
        active = remaining

    def pair_key(pair):
        check_cancelled(cancelled)
        return pair["j"]

    pairs = []
    for bucket in by_first:
        check_cancelled(cancelled)
        bucket.sort(key=pair_key)
        for pair in bucket:
            check_cancelled(cancelled)
            pairs.append(pair)
    return pairs


def _pair_registration_grid(tile_a, tile_b, pair, coarse_max_dim):
    lo = np.asarray(pair["search_lo"], dtype=float)
    hi = np.asarray(pair["search_hi"], dtype=float)
    raw_extent = hi - lo
    if lo.shape != (3,) or hi.shape != (3,) or not np.all(np.isfinite(raw_extent)) or np.any(raw_extent <= 0.0):
        raise ValueError("A registration pair has invalid physical search bounds.")
    extent = np.maximum(raw_extent, 1e-6)

    spacing_a = np.asarray(tile_a["spacing"], dtype=float)
    spacing_b = np.asarray(tile_b["spacing"], dtype=float)
    base_spacing = np.maximum(np.minimum(spacing_a, spacing_b), 1e-6)
    dims = np.ceil(extent / base_spacing).astype(int) + 1
    max_dim = max(16, int(coarse_max_dim))
    scale = max(1.0, float(np.max(dims)) / float(max_dim))
    spacing = base_spacing * scale
    dims = np.maximum(8, np.ceil(extent / spacing).astype(int) + 1)
    origin = lo + 0.5 * spacing
    affine = grid_affine_from_components(origin, spacing, np.eye(3, dtype=float))
    shape_zyx = (int(dims[2]), int(dims[1]), int(dims[0]))
    return affine, shape_zyx, spacing, lo, hi


def _estimate_registration_pair_bytes(tile_a, tile_b, pair, settings, mode):
    """Known simultaneously live grid arrays and explicit phase FFT outputs.

    Native FFT/optimizer scratch and data-dependent indexed samples are not
    calibrated here. This is partial allocation coverage, not a total peak.
    """
    _affine, shape_zyx, _spacing, _lo, _hi = _pair_registration_grid(
        tile_a, tile_b, pair, settings["coarse_max_dim"]
    )
    del mode  # All models execute the same phase-correlation stage.
    voxels = math.prod(shape_zyx)
    # Original, preprocessed and tapered fixed/moving float32 images plus
    # fixed_mask, moving_mask and initial_common remain live during the FFT.
    grid_bytes = voxels * (6 * np.dtype(np.float32).itemsize + 3 * np.dtype(bool).itemsize)
    return grid_bytes + _phase_fft_bytes(shape_zyx, shape_zyx)


def _phase_fft_bytes(fixed_shape, moving_shape):
    from scipy import fft

    shape = tuple(fft.next_fast_len(a + b - 1) for a, b in zip(fixed_shape, moving_shape))
    spectrum = math.prod((*shape[:-1], shape[-1] // 2 + 1)) * np.dtype(np.complex64).itemsize
    # f, m and conj(m) coexist; inverse output coexists with f, after m retires.
    return max(3 * spectrum, spectrum + math.prod(shape) * np.dtype(np.float32).itemsize)


def _register_pair(tile_a, tile_b, pair, settings, mode, *, cancelled=None):
    tile_a, tile_b = common._prepared_tiles([tile_a, tile_b])
    try:
        from scipy import ndimage, optimize
    except Exception as exc:
        raise RuntimeError("3-D stitching requires SciPy.") from exc

    mode = str(mode or "translation").lower()
    if mode not in {"translation", "rigid", "affine"}:
        raise ValueError(f"Unsupported stitching registration mode: {mode}")

    grid_affine, shape_zyx, spacing_xyz, lo, hi = _pair_registration_grid(
        tile_a, tile_b, pair, settings["coarse_max_dim"]
    )

    fixed = common._resample_zyx(
        tile_a["data"], tile_a["world_affine"], grid_affine, shape_zyx,
        order=1, output_dtype=np.float32,
    )
    check_cancelled(cancelled)
    moving = common._resample_zyx(
        tile_b["data"], tile_b["world_affine"], grid_affine, shape_zyx,
        order=1, output_dtype=np.float32,
    )
    fixed_mask = common._resample_zyx(
        common._tile_support_mask(tile_a),
        tile_a["world_affine"], grid_affine, shape_zyx,
        order=0, output_dtype=np.uint8,
    ) > 0
    moving_mask = common._resample_zyx(
        common._tile_support_mask(tile_b),
        tile_b["world_affine"], grid_affine, shape_zyx,
        order=0, output_dtype=np.uint8,
    ) > 0
    check_cancelled(cancelled)
    fixed_mask &= np.isfinite(fixed)
    moving_mask &= np.isfinite(moving)

    fixed_p = _registration_preprocess(fixed, fixed_mask)
    moving_p = _registration_preprocess(moving, moving_mask)
    initial_common = fixed_mask & moving_mask
    initial_score = _ncc(fixed_p, moving_p, initial_common)

    search_margin = max(0.0, float(settings["search_margin"]))
    spacing_zyx = np.asarray(spacing_xyz[::-1], dtype=float)
    max_shift_zyx = np.ceil(search_margin / np.maximum(spacing_zyx, 1e-9)).astype(int)
    fixed_phase = _registration_support_taper(fixed_p, fixed_mask)
    moving_phase = _registration_support_taper(moving_p, moving_mask)
    evaluation = evaluate_translations(
        fixed_p, moving_p, fixed_mask, moving_mask, fixed_phase, moving_phase,
        ncc=_ncc, settings=settings, max_shift=max_shift_zyx, cancelled=cancelled,
    )
    phase_correlation = evaluation["phase_correlation"]
    selected_candidate = evaluation["selected"]
    shift_zyx = np.asarray(
        selected_candidate["translation_zyx"] if selected_candidate else (0., 0., 0.), dtype=float
    )

    moved_translation = ndimage.shift(
        moving_p,
        shift=shift_zyx,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    moved_mask_translation = ndimage.shift(
        moving_mask.astype(np.uint8),
        shift=shift_zyx,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ) > 0
    translation_common = fixed_mask & moved_mask_translation
    translation_score = _ncc(fixed_p, moved_translation, translation_common)

    translation_xyz = shift_zyx[::-1] * np.asarray(spacing_xyz, dtype=float)
    correction = _translation_matrix_xyz(translation_xyz)
    final_score = float(translation_score)
    final_common = translation_common
    rigid_params = np.r_[translation_xyz, [0.0, 0.0, 0.0]]
    affine_params = np.r_[rigid_params, np.zeros(6, dtype=float)]
    selected_model = "translation"
    center_xyz = 0.5 * (lo + hi)
    grid_inv = np.linalg.inv(grid_affine)

    def transform_moving_matrix(corr, order=1, mask=False):
        # The already-resampled moving image lives on grid_affine. Moving it by
        # corr requires output world points to sample corr^-1 world points.
        output_to_input = grid_inv @ np.linalg.inv(corr) @ grid_affine
        mapping = common._XYZ_ZYX_PERMUTATION @ output_to_input @ common._XYZ_ZYX_PERMUTATION
        source = moving_mask.astype(np.uint8) if mask else moving_p
        return ndimage.affine_transform(
            source,
            matrix=mapping[:3, :3],
            offset=mapping[:3, 3],
            output_shape=shape_zyx,
            output=np.uint8 if mask else np.float32,
            order=0 if mask else order,
            mode="constant",
            cval=0,
            prefilter=False,
        )

    def evaluated_correction(corr):
        check_cancelled(cancelled)
        moved = transform_moving_matrix(corr)
        moved_mask = transform_moving_matrix(corr, mask=True) > 0
        valid = fixed_mask & moved_mask
        return float(_ncc(fixed_p, moved, valid)), valid

    model_selection = {
        "requested_model": mode,
        "selected_model": selected_model,
        "candidates": {
            "translation": {
                "score": float(final_score),
                "selected": True,
                "source": "phase-correlation",
                "bound_hits": [],
            }
        },
    }

    max_angle = max(
        0.0,
        float(settings.get("max_angle_deg", DEFAULT_STITCHING_MAX_ANGLE_DEG)),
    )
    translation_radius = max(
        float(np.max(spacing_xyz)) * 2.0,
        search_margin * 0.35,
    )

    if mode in {"rigid", "affine"}:
        translation_lower = translation_xyz - translation_radius
        translation_upper = translation_xyz + translation_radius

        def translation_objective(params):
            score, _valid = evaluated_correction(_translation_matrix_xyz(params))
            return -score

        translation_result = optimize.minimize(
            translation_objective,
            translation_xyz,
            method="Powell",
            bounds=list(zip(translation_lower, translation_upper)),
            options={
                "maxiter": int(settings.get("translation_refinement_iterations", 40)),
                "xtol": 1e-3,
                "ftol": 1e-4,
                "disp": False,
            },
        )
        refined_translation = np.asarray(translation_result.x, dtype=float)
        refined_correction = _translation_matrix_xyz(refined_translation)
        refined_score, refined_common = evaluated_correction(refined_correction)
        if refined_score > final_score + 1e-8:
            translation_xyz = refined_translation
            correction = refined_correction
            final_score = refined_score
            final_common = refined_common
        rigid_params = np.r_[translation_xyz, [0.0, 0.0, 0.0]]
        affine_params = np.r_[rigid_params, np.zeros(6, dtype=float)]
        model_selection["candidates"]["translation"] = {
            "score": float(final_score),
            "selected": True,
            "source": "bounded-continuous-refinement",
            "phase_score": float(translation_score),
            "bound_hits": [],
        }

        lower = np.r_[translation_xyz - translation_radius, [-max_angle] * 3]
        upper = np.r_[translation_xyz + translation_radius, [max_angle] * 3]

        def rigid_objective(params):
            corr = _rigid_matrix(params, center_xyz)
            score, _valid = evaluated_correction(corr)
            angular_penalty = 1e-5 * float(np.dot(params[3:6], params[3:6]))
            return -score + angular_penalty

        result = optimize.minimize(
            rigid_objective,
            rigid_params,
            method="Powell",
            bounds=list(zip(lower, upper)),
            options={
                "maxiter": int(settings.get("rigid_iterations", 60)),
                "xtol": 1e-3,
                "ftol": 1e-4,
                "disp": False,
            },
        )
        rigid_candidate_params = np.asarray(result.x, dtype=float)
        rigid_candidate_corr = _rigid_matrix(rigid_candidate_params, center_xyz)
        rigid_candidate_score, rigid_candidate_common = evaluated_correction(
            rigid_candidate_corr
        )
        rigid_gain = float(rigid_candidate_score - final_score)
        rigid_bound_hits = []
        if max_angle > 0.0:
            bound_tolerance = max(1e-6, max_angle * 1e-3)
            for axis, value in zip("XYZ", rigid_candidate_params[3:6]):
                if abs(abs(float(value)) - max_angle) <= bound_tolerance:
                    rigid_bound_hits.append(f"rotation_{axis.lower()}")
        model_selection["candidates"]["rigid"] = {
            "score": float(rigid_candidate_score),
            "gain_over_translation": rigid_gain,
            "selected": True,
            "bound_hits": rigid_bound_hits,
        }
        selected_model = "rigid"
        rigid_params = rigid_candidate_params
        correction = rigid_candidate_corr
        final_score = float(rigid_candidate_score)
        final_common = rigid_candidate_common

    if mode == "affine":
        affine_seed = np.r_[rigid_candidate_params, np.zeros(6, dtype=float)]
        max_scale_fraction = max(
            0.0,
            min(
                0.95,
                float(
                    settings.get(
                        "max_scale_percent", DEFAULT_STITCHING_MAX_SCALE_PERCENT
                    )
                )
                / 100.0,
            ),
        )
        minimum_scale = max(0.05, 1.0 - max_scale_fraction)
        maximum_scale = max(minimum_scale + 1e-6, 1.0 + max_scale_fraction)
        log_scale_lower = math.log(minimum_scale)
        log_scale_upper = math.log(maximum_scale)
        max_shear = max(
            0.0, float(settings.get("max_shear", DEFAULT_STITCHING_MAX_SHEAR))
        )
        lower = np.r_[
            rigid_candidate_params[:3] - translation_radius,
            [-max_angle] * 3,
            [log_scale_lower] * 3,
            [-max_shear] * 3,
        ]
        upper = np.r_[
            rigid_candidate_params[:3] + translation_radius,
            [max_angle] * 3,
            [log_scale_upper] * 3,
            [max_shear] * 3,
        ]

        def affine_objective(params):
            corr = _affine_matrix(params, center_xyz)
            determinant = float(np.linalg.det(corr[:3, :3]))
            if not math.isfinite(determinant) or determinant <= 1e-5:
                return 10.0 + abs(determinant)
            moved = transform_moving_matrix(corr)
            moved_mask = transform_moving_matrix(corr, mask=True) > 0
            valid = fixed_mask & moved_mask
            score = _ncc(fixed_p, moved, valid)
            regularizer = (
                1e-5 * float(np.dot(params[3:6], params[3:6]))
                + 5e-4 * float(np.dot(params[6:9], params[6:9]))
                + 5e-4 * float(np.dot(params[9:12], params[9:12]))
            )
            return -score + regularizer

        result = optimize.minimize(
            affine_objective,
            affine_seed,
            method="Powell",
            bounds=list(zip(lower, upper)),
            options={
                "maxiter": int(settings.get("affine_iterations", 90)),
                "xtol": 5e-4,
                "ftol": 5e-5,
                "disp": False,
            },
        )
        affine_candidate_params = np.asarray(result.x, dtype=float)
        affine_candidate_corr = _affine_matrix(affine_candidate_params, center_xyz)
        affine_candidate_score, affine_candidate_common = evaluated_correction(
            affine_candidate_corr
        )
        best_simpler_score = max(
            float(model_selection["candidates"]["translation"]["score"]),
            float(model_selection["candidates"]["rigid"]["score"]),
        )
        affine_gain = float(affine_candidate_score - best_simpler_score)
        affine_bound_hits = []
        if max_angle > 0.0:
            angle_tolerance = max(1e-6, max_angle * 1e-3)
            for axis, value in zip("XYZ", affine_candidate_params[3:6]):
                if abs(abs(float(value)) - max_angle) <= angle_tolerance:
                    affine_bound_hits.append(f"rotation_{axis.lower()}")
        scale_tolerance = max(1e-8, (log_scale_upper - log_scale_lower) * 1e-3)
        for axis, value in zip("XYZ", affine_candidate_params[6:9]):
            if abs(float(value) - log_scale_lower) <= scale_tolerance:
                affine_bound_hits.append(f"scale_{axis.lower()}_minimum")
            elif abs(float(value) - log_scale_upper) <= scale_tolerance:
                affine_bound_hits.append(f"scale_{axis.lower()}_maximum")
        if max_shear > 0.0:
            shear_tolerance = max(1e-8, max_shear * 1e-3)
            for plane, value in zip(("xy", "xz", "yz"), affine_candidate_params[9:12]):
                if abs(abs(float(value)) - max_shear) <= shear_tolerance:
                    affine_bound_hits.append(f"shear_{plane}")
        model_selection["candidates"]["affine"] = {
            "score": float(affine_candidate_score),
            "gain_over_best_simpler_model": affine_gain,
            "best_simpler_score": best_simpler_score,
            "selected": True,
            "bound_hits": affine_bound_hits,
        }
        selected_model = "affine"
        rigid_params = affine_candidate_params[:6].copy()
        affine_params = affine_candidate_params
        correction = affine_candidate_corr
        final_score = float(affine_candidate_score)
        final_common = affine_candidate_common

    for candidate_name, candidate_record in model_selection["candidates"].items():
        candidate_record["selected"] = candidate_name == selected_model
    model_selection["selected_model"] = selected_model

    _rot, scales, shear = _affine_linear_metrics(correction[:3, :3])
    physical_translation = _transform_displacement_at_point(correction, center_xyz)
    overlap_voxels = int(np.count_nonzero(final_common))
    weight = max(1e-4, (max(-1.0, min(1.0, final_score)) + 1.0) * 0.5)
    weight *= max(1.0, math.sqrt(max(1, overlap_voxels)))

    return {
        "fixed_id": tile_a["tile_id"],
        "moving_id": tile_b["tile_id"],
        "fixed_name": tile_a["display_name"],
        "moving_name": tile_b["display_name"],
        "mode": selected_model,
        "requested_mode": mode,
        "selected_model": selected_model,
        "model_selection": model_selection,
        "correction": correction,
        # For rigid/affine corrections the homogeneous translation column
        # includes pivot compensation and changes when the world origin moves.
        # Report the physical displacement at the registration overlap center.
        "translation_xyz": physical_translation,
        "overlap_center_xyz": np.asarray(center_xyz, dtype=float).copy(),
        "phase_translation_xyz": np.asarray(translation_xyz, dtype=float).copy(),
        "rigid_params": np.asarray(rigid_params, dtype=float),
        "affine_params": np.asarray(affine_params, dtype=float),
        "scale_xyz": np.asarray(scales, dtype=float),
        "shear_xyz": np.asarray(shear, dtype=float),
        "score": float(final_score),
        "initial_score": float(initial_score),
        "translation_score": float(translation_score),
        "weight": float(weight),
        "overlap_voxels": overlap_voxels,
        "overlap_fraction": float(pair["actual_overlap_fraction"]),
        "grid_shape_zyx": list(shape_zyx),
        "grid_spacing_xyz": np.asarray(spacing_xyz, dtype=float).tolist(),
        "grid_affine": common._matrix_to_json(grid_affine),
        "translation_grid_voxels_xyz": np.linalg.solve(
            np.asarray(grid_affine, dtype=float)[:3, :3], physical_translation
        ),
        "preprocessing": {
            "normalization_percentiles": [1.0, 99.5],
            "gaussian_background_sigma_registration_voxels": 2.0,
            "support_boundary": "cosine-distance-taper",
            "support_taper_width_registration_voxels": max(
                2.0, min(6.0, 0.125 * min(shape_zyx))
            ),
            "boundary_audit_decision": "cosine-taper-adopted",
        },
        "phase_correlation": {
            key: value
            for key, value in phase_correlation.items()
            if key != "shift_zyx"
        },
        "candidate_evaluation": evaluation,
        "measured_overlap": _measured_overlap(tile_a, tile_b, correction, grid_affine, overlap_voxels),
    }


def _measured_overlap(tile_a, tile_b, correction, grid_affine, overlap_voxels):
    """Geometric extent and sampled support after the measured correction."""
    alo, ahi = _support_bounds(tile_a["world_affine"], tile_a["dims"])
    blo, bhi = _support_bounds(correction @ tile_b["world_affine"], tile_b["dims"])
    cell_volume = abs(float(np.linalg.det(np.asarray(grid_affine)[:3, :3])))
    valid_volume = overlap_voxels * cell_volume
    volumes = [abs(float(np.linalg.det(t["world_affine"][:3, :3]))) * math.prod(t["dims"])
               for t in (tile_a, tile_b)]
    return dict(
        minimum_xyz=np.maximum(alo, blo).tolist(), maximum_xyz=np.minimum(ahi, bhi).tolist(),
        extent_xyz=np.maximum(0., np.minimum(ahi, bhi)-np.maximum(alo, blo)).tolist(),
        valid_registration_samples=overlap_voxels, valid_volume=valid_volume,
        fraction_of_fixed=min(1., valid_volume/volumes[0]),
        fraction_of_moving=min(1., valid_volume/volumes[1]),
        measurement="sampled common support, including zero-valued background; extent is its world-axis bounding box")


def _connected_components(tile_ids, edges, *, cancelled=None):
    tile_ids = list(tile_ids)
    adjacency = {tile_id: set() for tile_id in tile_ids}
    for edge in edges:
        check_cancelled(cancelled)
        adjacency[edge["fixed_id"]].add(edge["moving_id"])
        adjacency[edge["moving_id"]].add(edge["fixed_id"])
    components = []
    unseen = set(tile_ids)
    component_by_tile = {}
    for start in tile_ids:
        if start not in unseen:
            continue
        unseen.remove(start)
        component_index = len(components)
        components.append([])
        component_by_tile[start] = component_index
        stack = [start]
        while stack:
            check_cancelled(cancelled)
            cur = stack.pop()
            for nxt in adjacency[cur]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    component_by_tile[nxt] = component_index
                    stack.append(nxt)
    # Preserve tile order and anchors without rescanning all tiles per component.
    for tile_id in tile_ids:
        components[component_by_tile[tile_id]].append(tile_id)
    return components


def _edge_overlap_center(edge):
    center = np.asarray(edge.get("overlap_center_xyz", (0.0, 0.0, 0.0)), dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("A stitching edge has an invalid physical overlap center.")
    return center


def _transform_displacement_at_point(transform, point_xyz):
    """Return T(point) - point without using an origin-dependent column alone."""
    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    point = np.asarray(point_xyz, dtype=float).reshape(3)
    return (matrix[:3, :3] - np.eye(3, dtype=float)) @ point + matrix[:3, 3]


def _transform_rebased_to_origin(transform, origin_xyz):
    """Conjugate a world transform into coordinates centered at origin_xyz."""
    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    rebased = matrix.copy()
    rebased[:3, 3] = _transform_displacement_at_point(matrix, origin_xyz)
    return rebased


def _transform_rebased_to_world(transform, origin_xyz):
    """Undo _transform_rebased_to_origin without forming large translations."""
    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    origin = np.asarray(origin_xyz, dtype=float).reshape(3)
    restored = matrix.copy()
    restored[:3, 3] = (
        matrix[:3, 3] + origin - matrix[:3, :3] @ origin
    )
    return restored


def _edge_rebased_to_origin(edge, origin_xyz):
    rebased = dict(edge)
    rebased["correction"] = _transform_rebased_to_origin(
        edge["correction"], origin_xyz
    )
    rebased["overlap_center_xyz"] = (
        _edge_overlap_center(edge) - np.asarray(origin_xyz, dtype=float)
    )
    return rebased


def _edge_measured_displacement(edge):
    return _transform_displacement_at_point(
        edge["correction"], _edge_overlap_center(edge)
    )


def _edge_translation_error_vector(edge, fixed_correction, moving_correction):
    """Physical predicted-minus-measured displacement at the pair pivot."""
    predicted = np.linalg.inv(np.asarray(fixed_correction, dtype=float)) @ np.asarray(
        moving_correction, dtype=float
    )
    center = _edge_overlap_center(edge)
    return _transform_displacement_at_point(
        predicted, center
    ) - _transform_displacement_at_point(edge["correction"], center)


def _check_pose_graph_budget(
    tile_ids, edges, mode, memory_budget_bytes, cancelled, *, memory_reserved=False,
):
    """Budget sparse assembly, finite differences, solver vectors and local poses."""
    check_cancelled(cancelled)
    estimated = _pose_graph_bytes(tile_ids, edges, mode)
    available = common._available_memory_bytes()
    budget = int(memory_budget_bytes)
    if available > 0 and not memory_reserved:
        budget = min(budget, available)
    if estimated > budget:
        raise MemoryError(
            f"The known {mode} pose-graph arrays need {estimated / 1024**2:.1f} MiB "
            f"of working memory, exceeding the {budget / 1024**2:.1f} MiB stitching budget. "
            "Reduce the graph size or increase registration working memory."
        )
    return estimated


def _pose_graph_bytes(tile_ids, edges, mode):
    """Identifiable sparse pattern/vector storage; native solver scratch unknown."""
    parameters, residuals = {
        "translation": (3, 3), "rigid": (6, 6), "affine": (12, 15),
    }[mode]
    nodes, pairs = len(tile_ids), len(edges)
    # Every free pose needs an edge to its anchored component. Isolated tiles
    # have no solver parameters or regularization rows.
    variables = min(max(0, nodes - 1), pairs)
    rows = pairs * residuals + (13 * variables if mode == "affine" else 0)
    nonzeros = 2 * pairs * residuals * parameters
    if mode == "affine":
        nonzeros += 13 * variables * parameters
    # Count the identifiable sparse pattern and numerical vectors. Optimizer,
    # SciPy and Python-object scratch remain unmeasured and do not become a
    # guessed hard rejection threshold.
    estimated = (
        nonzeros * (np.dtype(bool).itemsize + np.dtype(np.intp).itemsize)
        + (rows + 1) * np.dtype(np.intp).itemsize
        + (rows + variables * parameters) * np.dtype(np.float64).itemsize
    )
    return estimated


def _pose_graph_incidence(index, edges, cancelled):
    """Sparse signed edge/node incidence with the fixed anchor column omitted."""
    from scipy import sparse

    rows = np.empty(2 * len(edges), dtype=np.int64)
    columns = np.empty_like(rows)
    values = np.empty(rows.size, dtype=float)
    used = 0
    for row, edge in enumerate(edges):
        check_cancelled(cancelled)
        for endpoint, sign in (("fixed_id", -1.0), ("moving_id", 1.0)):
            column = index.get(edge[endpoint])
            if column is not None:
                rows[used], columns[used], values[used] = row, column, sign
                used += 1
    return sparse.csr_matrix(
        (values[:used], (rows[:used], columns[:used])),
        shape=(len(edges), len(index)),
    )


def _pose_graph_jacobian_pattern(incidence, parameters, residuals, *, regularization_rows=0):
    from scipy import sparse

    pattern = sparse.kron(
        abs(incidence), np.ones((residuals, parameters), dtype=bool), format="csr",
    )
    if regularization_rows:
        local = sparse.kron(
            sparse.eye(incidence.shape[1], format="csr"),
            np.ones((regularization_rows, parameters), dtype=bool), format="csr",
        )
        pattern = sparse.vstack((pattern, local), format="csr")
    return pattern


def _solve_translation_pose_graph(
    tile_ids, edges, *, cancelled=None, memory_budget_bytes=1024**3,
    memory_reserved=False,
):
    tile_ids = list(tile_ids)
    _check_pose_graph_budget(
        tile_ids, edges, "translation", memory_budget_bytes, cancelled,
        memory_reserved=memory_reserved,
    )
    corrections = {tile_id: np.eye(4, dtype=float) for tile_id in tile_ids}
    components = _connected_components(tile_ids, edges, cancelled=cancelled)
    if not edges:
        return corrections, components
    try:
        from scipy import optimize, sparse
        from scipy.sparse.linalg import LinearOperator, lsmr
    except Exception as exc:
        raise RuntimeError("Translation pose-graph optimization requires SciPy.") from exc

    for component in components:
        check_cancelled(cancelled)
        if len(component) <= 1:
            continue
        anchor = component[0]
        variables = [tile_id for tile_id in component if tile_id != anchor]
        index = {tile_id: i for i, tile_id in enumerate(variables)}
        members = set(component)
        component_edges = [
            edge for edge in edges
            if edge["fixed_id"] in members and edge["moving_id"] in members
        ]
        if not component_edges:
            continue

        raw_weights = np.asarray(
            [max(1e-8, float(edge["weight"])) for edge in component_edges],
            dtype=float,
        )
        weight_reference = max(1e-8, float(np.median(raw_weights)))

        incidence = _pose_graph_incidence(index, component_edges, cancelled)
        weights = np.sqrt(raw_weights / weight_reference)
        measured = np.empty((len(component_edges), 3), dtype=float)
        for row, edge in enumerate(component_edges):
            check_cancelled(cancelled)
            measured[row] = _edge_measured_displacement(edge)
        matrix = sparse.kron(incidence.multiply(weights[:, None]), sparse.eye(3), format="csr")
        vector = (weights[:, None] * measured).ravel()

        def matvec(params):
            check_cancelled(cancelled)
            return matrix @ params

        def rmatvec(values):
            check_cancelled(cancelled)
            return matrix.T @ values

        operator = LinearOperator(matrix.shape, matvec=matvec, rmatvec=rmatvec, dtype=float)
        x0 = lsmr(
            operator, vector, atol=1e-12, btol=1e-12,
            maxiter=max(100, 3 * len(variables)),
        )[0]
        measured_norms = np.linalg.norm(measured, axis=1)
        residual_scale = max(1.0, float(np.median(measured_norms)))
        jacobian = matrix / residual_scale

        def residual(params):
            check_cancelled(cancelled)
            return (matrix @ params - vector) / residual_scale

        robust = optimize.least_squares(
            residual,
            x0,
            # SciPy scales a returned sparse Jacobian in-place for robust loss.
            jac=lambda _params: jacobian.copy(),
            tr_solver="lsmr",
            tr_options={"maxiter": 100, "atol": 1e-10, "btol": 1e-10},
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=150,
        )
        solution = robust.x
        check_cancelled(cancelled)
        for tile_id in variables:
            corrections[tile_id] = _translation_matrix_xyz(
                solution[3 * index[tile_id] : 3 * index[tile_id] + 3]
            )
    return corrections, components


def _solve_rigid_pose_graph(
    tile_ids, edges, *, cancelled=None, memory_budget_bytes=1024**3,
    memory_reserved=False,
):
    tile_ids = list(tile_ids)
    _check_pose_graph_budget(
        tile_ids, edges, "rigid", memory_budget_bytes, cancelled,
        memory_reserved=memory_reserved,
    )
    translation_start, components = _solve_translation_pose_graph(
        tile_ids, edges, cancelled=cancelled, memory_budget_bytes=memory_budget_bytes,
        memory_reserved=memory_reserved,
    )
    corrections = {tile_id: np.eye(4, dtype=float) for tile_id in tile_ids}
    if not edges:
        return corrections, components
    try:
        from scipy import optimize
    except Exception as exc:
        raise RuntimeError("Rigid pose-graph optimization requires SciPy.") from exc

    for component in components:
        check_cancelled(cancelled)
        if len(component) <= 1:
            continue
        anchor = component[0]
        variables = [tile_id for tile_id in component if tile_id != anchor]
        index = {tile_id: i for i, tile_id in enumerate(variables)}
        members = set(component)
        component_edges = [
            edge for edge in edges
            if edge["fixed_id"] in members and edge["moving_id"] in members
        ]
        pattern = _pose_graph_jacobian_pattern(
            _pose_graph_incidence(index, component_edges, cancelled), 6, 6,
        )
        component_center = np.median(
            np.vstack([_edge_overlap_center(edge) for edge in component_edges]),
            axis=0,
        )
        local_edges = [
            _edge_rebased_to_origin(edge, component_center)
            for edge in component_edges
        ]
        x0 = np.zeros(6 * len(variables), dtype=float)
        for tile_id in variables:
            x0[6 * index[tile_id] : 6 * index[tile_id] + 3] = (
                _transform_displacement_at_point(
                    translation_start[tile_id], component_center
                )
            )
        translation_scale = np.median(
            [
                max(1e-6, np.linalg.norm(_edge_measured_displacement(edge)))
                for edge in component_edges
            ]
            or [1.0]
        )
        translation_scale = max(1.0, float(translation_scale))
        weight_reference = max(
            1e-8,
            float(np.median([max(1e-8, float(edge["weight"])) for edge in component_edges])),
        )

        def matrices(params):
            out = {anchor: np.eye(4, dtype=float)}
            for tile_id in variables:
                check_cancelled(cancelled)
                p = params[6 * index[tile_id] : 6 * index[tile_id] + 6]
                out[tile_id] = _rigid_matrix(p)
            return out

        def residual(params):
            check_cancelled(cancelled)
            mats = matrices(params)
            values = []
            for edge in local_edges:
                check_cancelled(cancelled)
                ci = mats[edge["fixed_id"]]
                cj = mats[edge["moving_id"]]
                measured = np.asarray(edge["correction"], dtype=float)
                relative = np.linalg.inv(ci) @ cj
                weight = math.sqrt(
                    max(1e-8, float(edge["weight"])) / weight_reference
                )
                translation_error = _edge_translation_error_vector(edge, ci, cj)
                values.extend((weight * translation_error / translation_scale).tolist())
                linear_error = np.linalg.inv(measured[:3, :3]) @ relative[:3, :3]
                rot_deg = _rotation_vector_degrees(linear_error)
                values.extend((weight * rot_deg / 5.0).tolist())
            return np.asarray(values, dtype=float)

        result = optimize.least_squares(
            residual,
            x0,
            jac_sparsity=pattern,
            tr_solver="lsmr",
            tr_options={"maxiter": 100, "atol": 1e-10, "btol": 1e-10},
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=250,
        )
        solved = matrices(result.x)
        corrections.update({
            tile_id: _transform_rebased_to_world(matrix, component_center)
            for tile_id, matrix in solved.items()
        })

    return corrections, components


def _solve_affine_pose_graph(
    tile_ids, edges, *, cancelled=None, memory_budget_bytes=1024**3,
    memory_reserved=False,
):
    """Solve a robust global 12-DOF affine correction graph.

    Pairwise affine edges are already bounded during image registration. The
    global stage uses direct 3x3 linear matrices plus translation for each
    non-anchor tile and a soft determinant barrier to avoid singular solutions.
    """
    tile_ids = list(tile_ids)
    _check_pose_graph_budget(
        tile_ids, edges, "affine", memory_budget_bytes, cancelled,
        memory_reserved=memory_reserved,
    )
    rigid_start, components = _solve_rigid_pose_graph(
        tile_ids, edges, cancelled=cancelled, memory_budget_bytes=memory_budget_bytes,
        memory_reserved=memory_reserved,
    )
    corrections = {tile_id: np.eye(4, dtype=float) for tile_id in tile_ids}
    if not edges:
        return corrections, components
    try:
        from scipy import optimize
    except Exception as exc:
        raise RuntimeError("Affine pose-graph optimization requires SciPy.") from exc

    for component in components:
        check_cancelled(cancelled)
        if len(component) <= 1:
            continue
        anchor = component[0]
        variables = [tile_id for tile_id in component if tile_id != anchor]
        index = {tile_id: i for i, tile_id in enumerate(variables)}
        members = set(component)
        component_edges = [
            edge for edge in edges
            if edge["fixed_id"] in members and edge["moving_id"] in members
        ]
        if not component_edges:
            continue
        pattern = _pose_graph_jacobian_pattern(
            _pose_graph_incidence(index, component_edges, cancelled), 12, 15,
            regularization_rows=13,
        )

        translation_scale = max(
            1.0,
            float(np.median([
                max(1e-6, np.linalg.norm(_edge_measured_displacement(edge)))
                for edge in component_edges
            ] or [1.0])),
        )
        weight_reference = max(
            1e-8,
            float(np.median([
                max(1e-8, float(edge["weight"])) for edge in component_edges
            ])),
        )
        component_center = np.median(
            np.vstack([_edge_overlap_center(edge) for edge in component_edges]),
            axis=0,
        )
        local_edges = [
            _edge_rebased_to_origin(edge, component_center)
            for edge in component_edges
        ]
        x0 = np.zeros(12 * len(variables), dtype=float)
        for tile_id in variables:
            matrix = _transform_rebased_to_origin(
                rigid_start.get(tile_id, np.eye(4)), component_center
            )
            offset = 12 * index[tile_id]
            x0[offset : offset + 3] = matrix[:3, 3]
            x0[offset + 3 : offset + 12] = (
                matrix[:3, :3] - np.eye(3)
            ).reshape(-1)

        def matrices(params):
            out = {anchor: np.eye(4, dtype=float)}
            for tile_id in variables:
                check_cancelled(cancelled)
                offset = 12 * index[tile_id]
                matrix = np.eye(4, dtype=float)
                matrix[:3, :3] = np.eye(3) + params[offset + 3 : offset + 12].reshape(3, 3)
                matrix[:3, 3] = params[offset : offset + 3]
                out[tile_id] = matrix
            return out

        def residual(params):
            check_cancelled(cancelled)
            mats = matrices(params)
            values = []
            for edge in local_edges:
                check_cancelled(cancelled)
                ci = mats[edge["fixed_id"]]
                cj = mats[edge["moving_id"]]
                measured = np.asarray(edge["correction"], dtype=float)
                try:
                    relative = np.linalg.inv(ci) @ cj
                    linear_error = np.linalg.inv(measured[:3, :3]) @ relative[:3, :3]
                except np.linalg.LinAlgError:
                    values.extend([100.0] * 15)
                    continue
                weight = math.sqrt(
                    max(1e-8, float(edge["weight"])) / weight_reference
                )
                translation_error = _edge_translation_error_vector(edge, ci, cj)
                values.extend((weight * translation_error / translation_scale).tolist())
                rotation = _proper_rotation_from_linear(linear_error)
                rot_deg = _rotation_vector_degrees(rotation)
                values.extend((weight * rot_deg / 5.0).tolist())
                stretch = rotation.T @ linear_error
                values.extend((weight * (stretch - np.eye(3)).reshape(-1) / 0.05).tolist())

            # Constant-length soft barrier and weak regularization keep node
            # matrices invertible without suppressing genuine scale/shear.
            for tile_id in variables:
                check_cancelled(cancelled)
                matrix = mats[tile_id]
                determinant = float(np.linalg.det(matrix[:3, :3]))
                values.append(max(0.0, 0.10 - determinant) * 50.0)
                start = np.asarray(rigid_start.get(tile_id, np.eye(4)), dtype=float)
                start = _transform_rebased_to_origin(start, component_center)
                action_delta = matrix[:3, 3] - start[:3, 3]
                values.extend((0.002 * action_delta).tolist())
                values.extend(
                    (0.002 * (matrix[:3, :3] - start[:3, :3]).reshape(-1)).tolist()
                )
            return np.asarray(values, dtype=float)

        result = optimize.least_squares(
            residual,
            x0,
            jac_sparsity=pattern,
            tr_solver="lsmr",
            tr_options={"maxiter": 100, "atol": 1e-10, "btol": 1e-10},
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=350,
        )
        solved = {
            tile_id: _transform_rebased_to_world(matrix, component_center)
            for tile_id, matrix in matrices(result.x).items()
        }
        for tile_id, matrix in solved.items():
            if tile_id == anchor:
                continue
            determinant = float(np.linalg.det(matrix[:3, :3]))
            if math.isfinite(determinant) and determinant > 1e-5:
                corrections[tile_id] = matrix
            else:
                corrections[tile_id] = rigid_start.get(tile_id, np.eye(4, dtype=float))

    return corrections, components


def _edge_residuals(edges, corrections, *, cancelled=None):
    for edge in edges:
        check_cancelled(cancelled)
        ci = corrections[edge["fixed_id"]]
        cj = corrections[edge["moving_id"]]
        measured = np.asarray(edge["correction"], dtype=float)
        relative = np.linalg.inv(ci) @ cj
        linear_error = np.linalg.inv(measured[:3, :3]) @ relative[:3, :3]
        translation_error = _edge_translation_error_vector(edge, ci, cj)
        edge["global_translation_residual_xyz"] = translation_error.copy()
        edge["global_translation_residual"] = float(np.linalg.norm(translation_error))
        edge["global_rotation_residual_deg"] = float(
            np.linalg.norm(_rotation_vector_degrees(linear_error))
        )
        rotation = _proper_rotation_from_linear(linear_error)
        stretch = rotation.T @ linear_error
        edge["global_affine_residual"] = float(
            np.linalg.norm(stretch - np.eye(3), ord="fro")
        )


_POSE_GRAPH_QC_THRESHOLDS = {
    # These are review thresholds, not rejection thresholds. Half a registration
    # voxel catches loop disagreement at the scale of the established synthetic
    # accuracy contract without claiming a microscopy-specific acceptance limit.
    "translation_voxels": 0.5,
    "rotation_degrees": 0.5,
    "affine_frobenius": 0.01,
}


def _pose_graph_qc_summary(edges):
    """Summarize accepted-edge global consistency and phase ambiguity."""
    edges = list(edges)
    for edge in edges:
        residual_vector = np.asarray(
            edge.get("global_translation_residual_xyz", ()), dtype=float
        )
        grid_affine = np.asarray(edge.get("grid_affine", ()), dtype=float)
        spacing = np.asarray(edge.get("grid_spacing_xyz") or (), dtype=float)
        if residual_vector.shape == (3,) and np.all(np.isfinite(residual_vector)):
            if grid_affine.shape == (4, 4):
                grid_linear = grid_affine[:3, :3]
            elif spacing.shape == (3,) and np.all(np.isfinite(spacing)) and np.all(spacing > 0.0):
                grid_linear = np.diag(spacing)
            else:
                grid_linear = np.eye(3, dtype=float)
            try:
                residual_index = np.linalg.solve(grid_linear, residual_vector)
            except np.linalg.LinAlgError:
                residual_index = residual_vector
            edge["global_translation_residual_voxels_xyz"] = residual_index.copy()
            edge["global_translation_residual_voxels"] = float(
                np.linalg.norm(residual_index)
            )
        else:
            # Historical payloads retain only a scalar world residual and cannot
            # recover its physical direction for anisotropic conversion.
            finite_spacing = spacing[np.isfinite(spacing) & (spacing > 0.0)]
            voxel_scale = float(np.min(finite_spacing)) if finite_spacing.size else 1.0
            edge["global_translation_residual_voxels"] = float(
                edge.get("global_translation_residual", 0.0)
            ) / voxel_scale

    metric_specs = {
        "translation_voxels": (
            "global_translation_residual_voxels",
            _POSE_GRAPH_QC_THRESHOLDS["translation_voxels"],
        ),
        "rotation_degrees": (
            "global_rotation_residual_deg",
            _POSE_GRAPH_QC_THRESHOLDS["rotation_degrees"],
        ),
        "affine_frobenius": (
            "global_affine_residual",
            _POSE_GRAPH_QC_THRESHOLDS["affine_frobenius"],
        ),
    }
    metrics = {}
    residual_warning_edges = []
    for name, (field, threshold) in metric_specs.items():
        values = np.asarray([float(edge.get(field, 0.0)) for edge in edges])
        maximum = float(np.max(values)) if values.size else 0.0
        largest = [
            {
                "fixed_id": str(edge.get("fixed_id") or ""),
                "moving_id": str(edge.get("moving_id") or ""),
                "fixed_name": str(edge.get("fixed_name") or edge.get("fixed_id") or ""),
                "moving_name": str(edge.get("moving_name") or edge.get("moving_id") or ""),
                "value": float(value),
            }
            for edge, value in zip(edges, values)
            if values.size and np.isclose(value, maximum, rtol=1e-9, atol=1e-12)
        ]
        metrics[name] = {
            "count": int(values.size),
            "mean": float(np.mean(values)) if values.size else 0.0,
            "median": float(np.median(values)) if values.size else 0.0,
            "p95": float(np.percentile(values, 95.0)) if values.size else 0.0,
            "maximum": maximum,
            "warning_threshold": float(threshold),
            "largest_edges": largest,
        }
        if maximum > threshold:
            residual_warning_edges.extend(
                f"{record['fixed_id']}->{record['moving_id']}" for record in largest
            )

    ambiguous_edges = []
    evaluated_phase_edges = 0
    for edge in edges:
        phase = edge.get("phase_correlation")
        if not isinstance(phase, dict):
            continue
        evaluated_phase_edges += 1
        if bool(phase.get("ambiguous", True)):
            ambiguous_edges.append(
                {
                    "fixed_id": str(edge.get("fixed_id") or ""),
                    "moving_id": str(edge.get("moving_id") or ""),
                    "fixed_name": str(edge.get("fixed_name") or edge.get("fixed_id") or ""),
                    "moving_name": str(edge.get("moving_name") or edge.get("moving_id") or ""),
                    "valid": bool(phase.get("valid", False)),
                    "peak_ratio": phase.get("peak_ratio"),
                    "peak_prominence": phase.get("peak_prominence"),
                }
            )

    warning = bool(residual_warning_edges or ambiguous_edges)
    return {
        "status": "warning" if warning else "passed" if edges else "not-evaluated",
        "accepted_edge_count": len(edges),
        "metrics": metrics,
        "residual_warning_edges": sorted(set(residual_warning_edges)),
        "phase_correlation": {
            "evaluated_edge_count": int(evaluated_phase_edges),
            "ambiguous_edge_count": len(ambiguous_edges),
            "ambiguous_edges": ambiguous_edges,
            "warning_policy": (
                "Peak ratio below 1.5, relative prominence below 0.25, or "
                "invalid confidence triggers review but does not reject an edge."
            ),
        },
    }


def _translation_pose_graph_with_global_rejection(
    tile_ids, edges, *, cancelled=None, memory_budget_bytes=1024**3,
    memory_reserved=False,
):
    """Solve translation constraints and conservatively reject unique outliers.

    Global residuals can identify disagreement but cannot always identify which
    member of a loop is wrong. Remove only a unique robust residual outlier whose
    removal preserves the graph components. Tied or otherwise non-identifiable
    disagreement remains explicit QC.
    """
    tile_ids = list(tile_ids)
    _check_pose_graph_budget(
        tile_ids, edges, "translation", memory_budget_bytes, cancelled,
        memory_reserved=memory_reserved,
    )
    active_edges = list(edges)
    rejected = []
    iterations = []
    ambiguities = []
    check_cancelled(cancelled)
    initial_components = _connected_components(tile_ids, active_edges, cancelled=cancelled)
    initial_cycle_rank = max(
        0, len(active_edges) - len(tile_ids) + len(initial_components)
    )
    maximum_rejections = min(8, initial_cycle_rank)
    numerical_threshold = float(
        TRANSLATION_STITCHING_ACCEPTANCE_CONTRACT[
            "supported_fixture_accuracy_registration_voxels"
        ]
    )

    for iteration_index in range(maximum_rejections):
        check_cancelled(cancelled)
        corrections, components = _solve_translation_pose_graph(
            tile_ids, active_edges, cancelled=cancelled, memory_budget_bytes=memory_budget_bytes,
            memory_reserved=memory_reserved,
        )
        _edge_residuals(active_edges, corrections, cancelled=cancelled)
        _pose_graph_qc_summary(active_edges)

        removable = []
        component_count = len(components)
        for edge_index, edge in enumerate(active_edges):
            check_cancelled(cancelled)
            remaining = active_edges[:edge_index] + active_edges[edge_index + 1 :]
            if len(_connected_components(tile_ids, remaining, cancelled=cancelled)) != component_count:
                continue
            removable.append(
                (
                    edge_index,
                    edge,
                    float(edge.get("global_translation_residual_voxels", 0.0)),
                )
            )

        values = np.asarray([record[2] for record in removable], dtype=float)
        median = float(np.median(values)) if values.size else 0.0
        mad = (
            float(np.median(np.abs(values - median))) if values.size else 0.0
        )
        normalized_mad = 1.4826 * mad
        cutoff = max(numerical_threshold, median + 3.0 * normalized_mad)
        outliers = [
            record for record in removable if record[2] > cutoff + 1e-9
        ]
        residual_warnings = [
            record for record in removable if record[2] > numerical_threshold
        ]
        audit = {
            "iteration": iteration_index + 1,
            "accepted_edge_count_before": len(active_edges),
            "component_count_before": component_count,
            "removable_edge_count": len(removable),
            "residual_voxels": {
                "median": median,
                "normalized_mad": normalized_mad,
                "outlier_cutoff": cutoff,
                "maximum": float(np.max(values)) if values.size else 0.0,
            },
            "candidate_edges": [
                {
                    "fixed_id": str(edge.get("fixed_id") or ""),
                    "moving_id": str(edge.get("moving_id") or ""),
                    "residual_voxels": residual,
                }
                for _index, edge, residual in outliers
            ],
        }
        iterations.append(audit)

        if not outliers:
            if residual_warnings:
                ambiguities.append(
                    {
                        "code": "global-inconsistency-not-identifiable",
                        "reason": (
                            "Translation constraints exceed the 0.5 registration-"
                            "voxel consistency threshold, but no unique robust "
                            "cycle outlier can be identified."
                        ),
                        "edges": [
                            f"{edge.get('fixed_id', '')}->{edge.get('moving_id', '')}"
                            for _index, edge, _residual in residual_warnings
                        ],
                    }
                )
            break

        maximum = max(record[2] for record in outliers)
        strongest = [
            record
            for record in outliers
            if math.isclose(record[2], maximum, rel_tol=1e-9, abs_tol=1e-9)
        ]
        if len(strongest) != 1:
            ambiguities.append(
                {
                    "code": "global-inconsistency-tied-outliers",
                    "reason": (
                        "Multiple translation constraints are tied as the largest "
                        "global residual; rejecting one would assign blame without "
                        "independent evidence."
                    ),
                    "edges": [
                        f"{edge.get('fixed_id', '')}->{edge.get('moving_id', '')}"
                        for _index, edge, _residual in strongest
                    ],
                }
            )
            break

        edge_index, rejected_edge, residual_voxels = strongest[0]
        rejected.append(
            {
                "iteration": iteration_index + 1,
                "criterion": (
                    "unique cycle-redundant residual > max(0.5 registration voxel, "
                    "median + 3 normalized MAD)"
                ),
                "outlier_cutoff_voxels": cutoff,
                "residual_voxels": residual_voxels,
                "constraint": copy.deepcopy(rejected_edge),
            }
        )
        audit["rejected_edge"] = {
            "fixed_id": str(rejected_edge.get("fixed_id") or ""),
            "moving_id": str(rejected_edge.get("moving_id") or ""),
            "residual_voxels": residual_voxels,
        }
        del active_edges[edge_index]

    check_cancelled(cancelled)
    corrections, components = _solve_translation_pose_graph(
        tile_ids, active_edges, cancelled=cancelled, memory_budget_bytes=memory_budget_bytes,
        memory_reserved=memory_reserved,
    )
    _edge_residuals(active_edges, corrections, cancelled=cancelled)
    pose_graph_qc = _pose_graph_qc_summary(active_edges)
    final_cycle_rank = max(
        0, len(active_edges) - len(tile_ids) + len(components)
    )
    if (
        len(rejected) >= maximum_rejections
        and maximum_rejections > 0
        and pose_graph_qc["residual_warning_edges"]
    ):
        ambiguities.append(
            {
                "code": "global-inconsistency-rejection-bound-reached",
                "reason": (
                    "The bounded global inconsistency pass reached its rejection "
                    "limit while residual disagreement remained."
                ),
                "edges": list(pose_graph_qc["residual_warning_edges"]),
            }
        )

    global_qc = {
        "policy": TRANSLATION_STITCHING_ACCEPTANCE_CONTRACT[
            "global_inconsistency"
        ],
        "numerical_threshold_registration_voxels": numerical_threshold,
        "maximum_rejections": maximum_rejections,
        "iteration_count": len(iterations),
        "iterations": iterations,
        "rejected_edge_count": len(rejected),
        "rejected_edges": [
            {
                "fixed_id": str(item["constraint"].get("fixed_id") or ""),
                "moving_id": str(item["constraint"].get("moving_id") or ""),
                "iteration": item["iteration"],
                "residual_voxels": item["residual_voxels"],
                "outlier_cutoff_voxels": item["outlier_cutoff_voxels"],
            }
            for item in rejected
        ],
        "ambiguity_count": len(ambiguities),
        "ambiguities": ambiguities,
        "initial_component_count": len(initial_components),
        "final_component_count": len(components),
        "initial_cycle_rank": initial_cycle_rank,
        "final_cycle_rank": final_cycle_rank,
    }
    pose_graph_qc["component_count"] = len(components)
    pose_graph_qc["cycle_rank"] = final_cycle_rank
    pose_graph_qc["global_inconsistency"] = global_qc
    if rejected or ambiguities:
        pose_graph_qc["status"] = "warning"
    return corrections, components, active_edges, rejected, pose_graph_qc


def _source_corrections_from_mosaic(mosaic_corrections, preparation):
    conversions = {}
    for record in preparation.provenance.unit_conversions:
        tile_id = str(record.get("tile_id") or "")
        conversions.setdefault(
            tile_id,
            invertible_affine4(
                record.get("conversion_matrix"),
                f"Unit conversion for stitching tile {tile_id!r}",
            ),
        )
    source_corrections = {}
    for tile_id, correction in mosaic_corrections.items():
        conversion = conversions.get(str(tile_id), np.eye(4, dtype=float))
        source_corrections[str(tile_id)] = invertible_affine4(
            np.linalg.inv(conversion) @ np.asarray(correction, dtype=float) @ conversion,
            f"Source-frame stitching correction for tile {tile_id!r}",
        ).copy()
    return source_corrections


def _registration_geometry_provenance(
    preparation, mosaic_corrections, source_corrections
):
    payload = preparation.provenance.to_dict()
    for record in payload["source_tiles"]:
        tile_id = str(record.get("tile_id") or "")
        mosaic_correction = np.asarray(
            mosaic_corrections.get(tile_id, np.eye(4)), dtype=float
        )
        source_correction = np.asarray(
            source_corrections.get(tile_id, np.eye(4)), dtype=float
        )
        record["solved_correction_mosaic"] = common._matrix_to_json(mosaic_correction)
        record["solved_correction_source"] = common._matrix_to_json(source_correction)
        initial_mapping = np.asarray(record["initial_mosaic_mapping"], dtype=float)
        record["mosaic_space_mapping"] = common._matrix_to_json(
            mosaic_correction @ initial_mapping
        )
    return payload


def registration_settings_from_saved_panel(settings):
    """Convert captured interactive or queued settings to the numerical contract."""
    s = dict(settings or {})
    values = dict(search_margin=s.get("search_margin", 25.),
        minimum_overlap_fraction=float(s.get("minimum_overlap", .25))/100.,
        minimum_score=s.get("minimum_score", DEFAULT_STITCHING_MINIMUM_NCC),
        coarse_max_dim=s.get("coarse_size", 112), max_angle_deg=s.get("max_angle", DEFAULT_STITCHING_MAX_ANGLE_DEG),
        max_scale_percent=s.get("max_scale", DEFAULT_STITCHING_MAX_SCALE_PERCENT),
        max_shear=s.get("max_shear", DEFAULT_STITCHING_MAX_SHEAR), worker_count=s.get("worker_count", 0),
        initial_pose_source=s.get("initial_pose", "current"), overlap_source=s.get("overlap_source", "madi"),
        filename_grid_overlap_percent=s.get("filename_overlap", 10.0),
        rigid_iterations=s.get("rigid_iterations", 60),
        affine_iterations=s.get("affine_iterations", 90),
        registration_channel_key=s.get("registration_channel"),
        registration_channel_label=s.get("registration_channel_label"))
    for key in ("topology", "phase_peak_count", "discovery_max_pairs", "discovery_exhaustive_tiles", "registration_memory_mb",
                "stage_xy_mapping", "invert_stage_x", "invert_stage_y", "invert_stage_z"):
        if key in s:
            values[key] = copy.deepcopy(s[key])
    return validated_registration_settings(values)


def validated_registration_settings(settings, subject="Stitching registration"):
    """Validate open-ended persisted settings at the registration boundary."""
    values = dict(settings or {})
    defaults = dict(
        phase_peak_count=DEFAULT_PHASE_PEAK_COUNT, discovery_preview_dim=64,
        discovery_minimum_support=512, discovery_minimum_informative=128,
        discovery_minimum_ncc=0.4, candidate_ambiguity_ncc=0.03,
        discovery_exhaustive_tiles=12, discovery_max_pairs=256,
        discovery_neighbors=4, preview_cache_mb=64, registration_memory_mb=1024,
    )
    for key, default in defaults.items():
        values.setdefault(key, default)
        number = float(values[key])
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{subject} setting {key!r} must be positive and finite.")
        if isinstance(default, int):
            if int(number) != number:
                raise ValueError(f"{subject} setting {key!r} must be integral.")
            values[key] = int(number)
        else:
            values[key] = number
    if values["phase_peak_count"] > 32 or values["discovery_preview_dim"] > 256:
        raise ValueError("Phase peaks are limited to 32 and discovery previews to 256 samples per axis.")
    if values["discovery_neighbors"] > 32:
        raise ValueError("Coarse image scheduling is limited to 32 neighbors per tile; use exhaustive search to broaden it further.")
    if values["discovery_minimum_ncc"] > 1 or values["candidate_ambiguity_ncc"] > 1:
        raise ValueError("Discovery NCC settings must be within 0..1.")
    from madi3d_app.stitching.topology import validate_topology
    values["topology"] = validate_topology(values.get("topology"))
    required = {
        "search_margin",
        "minimum_overlap_fraction",
        "minimum_score",
        "coarse_max_dim",
        "max_angle_deg",
        "max_scale_percent",
        "max_shear",
        "worker_count",
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"{subject} is missing settings: {', '.join(missing)}.")
    for key in (
        "search_margin",
        "minimum_overlap_fraction",
        "minimum_score",
        "max_angle_deg",
        "max_scale_percent",
        "max_shear",
    ):
        values[key] = float(values[key])
        if not math.isfinite(values[key]):
            raise ValueError(f"{subject} setting {key!r} must be finite.")
    if values["search_margin"] < 0.0:
        raise ValueError(f"{subject} search margin cannot be negative.")
    if not 0.0 <= values["minimum_overlap_fraction"] <= 1.0:
        raise ValueError(f"{subject} minimum overlap fraction must be within 0..1.")
    if not -1.0 <= values["minimum_score"] <= 1.0:
        raise ValueError(f"{subject} minimum NCC score must be within -1..1.")
    for key in ("max_angle_deg", "max_scale_percent", "max_shear"):
        if values[key] < 0.0:
            raise ValueError(f"{subject} setting {key!r} cannot be negative.")
    for key, minimum in (
        ("coarse_max_dim", 1),
        ("worker_count", 0),
        ("rigid_iterations", 1),
        ("affine_iterations", 1),
    ):
        if key not in values:
            continue
        numeric = float(values[key])
        if not math.isfinite(numeric):
            raise ValueError(f"{subject} setting {key!r} must be finite.")
        integer = int(numeric)
        if numeric != integer or integer < minimum:
            raise ValueError(
                f"{subject} setting {key!r} must be an integer of at least {minimum}."
            )
        values[key] = integer
    return values


class StitchRegistrationOperation:
    """GUI-independent registration operation driven by explicit callbacks."""

    def __init__(
        self,
        tiles,
        settings,
        mode,
        *,
        progress_callback=None,
        cancelled=None,
        completed_callback=None,
        failed_callback=None,
        reuse_result=None,
        reserve_memory=None,
    ):
        self.tiles = tiles
        self.settings = dict(settings)
        self.memory_reserved = bool(self.settings.pop("memory_reserved", False))
        self.mode = str(mode)
        self._progress_callback = progress_callback or (lambda _value, _text: None)
        self._cancelled = cancelled or (lambda: False)
        self._completed_callback = completed_callback or (lambda _result: None)
        self._failed_callback = failed_callback or (lambda _message: None)
        self.reuse_result = reuse_result
        self._reserve_memory = reserve_memory

    def _admit_working_memory(self, known_bytes, retained_bytes=0):
        check_cancelled(self._cancelled)
        budget = int(self.settings["registration_memory_mb"] * 1024**2)
        if self._reserve_memory is not None:
            self._reserve_memory(known_bytes, budget, retained_bytes)
        elif known_bytes + retained_bytes > budget:
            raise MemoryError(
                f"Known registration buffers need {(known_bytes + retained_bytes) / 1024**2:.1f} MiB, "
                f"exceeding the {budget / 1024**2:.1f} MiB working-memory budget. "
                "Reduce registration size or increase the budget."
            )
        check_cancelled(self._cancelled)

    def run(self):
        try:
            self.settings = validated_registration_settings(self.settings)
            if self.mode not in {"translation", "rigid", "affine"}:
                raise ValueError(
                    f"Unsupported stitching registration mode {self.mode!r}."
                )
            check_cancelled(self._cancelled)
            automatic = self.settings.get("initial_pose_source") == "auto"
            placement_evidence = copy.deepcopy(self.settings.get("initial_layout") or {})
            if automatic and not placement_evidence:
                from madi3d_app.stitching.stitching_positioning import (
                    InitialPlacementSettings, registration_initial_layout,
                )
                placement_evidence = registration_initial_layout(
                    self.tiles, InitialPlacementSettings(mode="auto",
                        **{key: self.settings[key] for key in ("stage_xy_mapping", "invert_stage_x", "invert_stage_y", "invert_stage_z") if key in self.settings}),
                    topology=self.settings.get("topology"),
                )
                placed_tiles = []
                for original in self.tiles:
                    tile = dict(original)
                    delta = np.asarray(placement_evidence["placement_deltas"][tile["tile_id"]])
                    tile["world_affine"] = delta @ tile["world_affine"]
                    if tile.get("actor_matrix") is not None:
                        tile["actor_matrix"] = delta @ tile["actor_matrix"]
                    placed_tiles.append(tile)
                self.tiles = placed_tiles
            preparation = common._prepare_stitching_geometry(self.tiles, cancelled=self._cancelled)
            self.tiles = preparation.prepared_tiles
            registration_inputs = []
            for tile in self.tiles:
                check_cancelled(self._cancelled)
                support_min, support_max = _support_bounds(
                    tile["world_affine"], tile["dims"]
                )
                registration_inputs.append(
                    {
                        "tile_id": tile["tile_id"],
                        "display_name": tile["display_name"],
                        **{key: copy.deepcopy(tile.get(key)) for key in (
                            "source_id", "channel_id", "backing_source_id",
                            "channel_selector", "series_index", "series_identity",
                            "channel_index", "channel_identity", "time_index",
                            "position_index", "geometry_revision",
                            "acquisition_geometry_revision", "dependency_revision",
                            "source_operation_ids",
                        )},
                        "channel": copy.deepcopy(tile.get("channel_selector")),
                        "world_affine": common._matrix_to_json(tile["world_affine"]),
                        "support_bounds": {
                            "minimum_xyz": np.asarray(
                                support_min, dtype=float
                            ).tolist(),
                            "maximum_xyz": np.asarray(
                                support_max, dtype=float
                            ).tolist(),
                        },
                    }
                )
            discovery = None
            search_coverage = {}
            input_fingerprints = {}
            reuse_signature = None
            if automatic:
                from madi3d_app.stitching.discovery import DiscoveryContext, pixel_fingerprint
                discovery = DiscoveryContext(self.tiles, self.settings, cancelled=self._cancelled,
                                             progress=self._progress_callback, placement_evidence=placement_evidence,
                                             reserve_memory=self._admit_working_memory)
                for tile in self.tiles:
                    self._progress_callback(0, f"Checking registration pixels: {tile['display_name']}")
                    input_fingerprints[tile["tile_id"]] = pixel_fingerprint(tile, self._cancelled)
                    tile["_registration_pixel_fingerprint"] = input_fingerprints[tile["tile_id"]]
                    discovery.metrics["source_hash_bytes"] += np.asarray(tile["data"]).nbytes
                signature_settings = {k: v for k, v in self.settings.items() if k not in {"resolved_worker_count", "worker_resolution", "initial_layout"}}
                signature_payload = dict(algorithm=common.STITCHING_ALGORITHM_VERSION, mode=self.mode,
                    settings=signature_settings, pixels=input_fingerprints,
                    placement_evidence=common._json_geometry_value(placement_evidence),
                    geometry=preparation.provenance.to_dict(),
                    tiles=[dict(tile_id=t["tile_id"], affine=t["world_affine"].tolist(), dims=list(t["dims"]),
                                source_id=t.get("source_id"), reference=t.get("reference_enabled", True)) for t in self.tiles])
                reuse_signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
                if self.reuse_result is not None and self.reuse_result.get("reuse_signature") == reuse_signature:
                    reused = copy.deepcopy(self.reuse_result)
                    reused["reused_calculation"] = True
                    check_cancelled(self._cancelled)
                    self._progress_callback(100, "Reused registration after verifying pixels, channel, geometry and settings")
                    self._completed_callback(reused)
                    return
                known_ids = {r["tile_id"] for r in placement_evidence.get("records", [])
                             if r.get("applied_axes") and not r.get("kept_current_pose")}
                known_tiles = [tile for tile in self.tiles if tile["tile_id"] in known_ids]
                # For larger collections, physical proximity only schedules bounded
                # comparisons; it does not require an all-pairs bounds allocation.
                if len(known_tiles) > self.settings["discovery_exhaustive_tiles"]:
                    from scipy.spatial import cKDTree
                    centers = np.array([t["world_affine"][:3, 3] for t in known_tiles])
                    nearest = cKDTree(centers).query(centers, k=min(len(centers), self.settings["discovery_neighbors"]+1))[1]
                    subset_pairs = sorted({tuple(sorted((i, int(j)))) for i, row in enumerate(nearest) for j in np.atleast_1d(row) if i != j})
                else:
                    from itertools import combinations
                    subset_pairs = list(combinations(range(len(known_tiles)), 2))
                indices = {t["tile_id"]: i for i, t in enumerate(self.tiles)}
                guided = []
                for i, j in subset_pairs:
                    for pair in _candidate_pairs(
                        [known_tiles[i], known_tiles[j]], self.settings["search_margin"],
                        self.settings["minimum_overlap_fraction"], cancelled=self._cancelled,
                    ):
                        pair["i"], pair["j"] = indices[pair["fixed_id"]], indices[pair["moving_id"]]
                        guided.append(pair)
                pairs, search_coverage = discovery.schedule(guided, known_ids)
            else:
                pairs = _candidate_pairs(
                    self.tiles, self.settings["search_margin"],
                    self.settings["minimum_overlap_fraction"], cancelled=self._cancelled,
                )
            # Reference-enabled tiles can constrain neighbors. Fusion-only tiles
            # remain in the project/output but do not create edges solely between
            # one another. If only the second tile is a reference, orient the edge
            # so registration treats it as fixed.
            filtered_pairs = []
            for pair in pairs:
                check_cancelled(self._cancelled)
                a = self.tiles[pair["i"]]
                b = self.tiles[pair["j"]]
                a_ref = bool(a.get("reference_enabled", True))
                b_ref = bool(b.get("reference_enabled", True))
                if not (a_ref or b_ref):
                    continue
                pair = dict(pair)
                if b_ref and not a_ref:
                    pair["i"], pair["j"] = pair["j"], pair["i"]
                    pair["fixed_id"], pair["moving_id"] = pair["moving_id"], pair["fixed_id"]
                filtered_pairs.append(pair)
            pairs = filtered_pairs
            # Build the immutable broadcast support masks before pair tasks start,
            # so worker threads only read tile dictionaries.
            for tile in self.tiles:
                check_cancelled(self._cancelled)
                common._tile_support_mask(tile)

            total = len(pairs)
            # The executor accepts at most 32 workers. Retain only the grids
            # that can contribute to its largest actual concurrent demand.
            pair_bytes = []
            for pair in pairs:
                check_cancelled(self._cancelled)
                if pair.get("search_domain") != "unknown-position":
                    size = _estimate_registration_pair_bytes(
                        self.tiles[pair["i"]], self.tiles[pair["j"]], pair, self.settings, self.mode,
                    )
                    heapq.heappush(pair_bytes, size)
                    if len(pair_bytes) > 32:
                        heapq.heappop(pair_bytes)
            estimated_pair_bytes = max(pair_bytes, default=0)
            discovery_bytes = discovery.metrics["estimated_working_bytes"] if discovery else 0
            worker_resolution = {}
            working_budget = int(self.settings["registration_memory_mb"] * 1024**2)
            requested_workers = int(self.settings.get("worker_count", 0)) or 8
            worker_count = common._resolved_worker_count(
                1 if discovery else requested_workers,
                total,
                auto_cap=8,
                estimated_bytes_per_job=estimated_pair_bytes,
                diagnostics=worker_resolution,
                memory_reserved=self.memory_reserved,
            )
            # Any subset of this many tasks can overlap in the executor. Sum
            # the largest actual grids, rather than charging every pair as the
            # largest one or relying on detached input size.
            ordered_bytes = sorted(pair_bytes, reverse=True)
            requested_concurrency = worker_count
            while worker_count > 1 and sum(ordered_bytes[:worker_count]) > working_budget:
                worker_count -= 1
            if worker_count < requested_concurrency:
                worker_resolution["memory_limited"] = True
                worker_resolution["reductions"].append(
                    f"registration working-memory budget limited concurrency to {worker_count} worker(s)")
            concurrent_bytes = max(estimated_pair_bytes, discovery_bytes) if discovery else sum(ordered_bytes[:worker_count])
            self._admit_working_memory(concurrent_bytes)
            worker_resolution["resolved_workers"] = worker_count
            worker_resolution["known_concurrent_bytes"] = concurrent_bytes
            self.settings["resolved_worker_count"] = int(worker_count)
            self.settings["worker_resolution"] = worker_resolution
            self._progress_callback(
                0,
                f"Registering {total} candidate pair(s) with {worker_count} CPU worker(s)",
            )

            edges = []
            pair_failures = []
            rejections = []
            pair_evaluations = []
            score_rejections = 0
            invalid_pair_rejections = 0
            minimum_score = float(self.settings["minimum_score"])
            recoverable_pair_errors = (
                np.linalg.LinAlgError,
                FloatingPointError,
                OverflowError,
                ValueError,
                RuntimeError,
            )

            def register_pair(pair):
                if self._cancelled():
                    raise InterruptedError
                a = self.tiles[pair["i"]]
                b = self.tiles[pair["j"]]
                try:
                    if discovery and pair.get("search_domain") == "unknown-position":
                        edge = discovery.register(a, b, pair, self.settings, self.mode)
                    else:
                        # Retain the guided path for usable placement evidence.
                        if discovery:
                            self._admit_working_memory(_estimate_registration_pair_bytes(
                                a, b, pair, self.settings, self.mode), discovery.cache_bytes)
                        edge = _register_pair(a, b, pair, self.settings, self.mode, cancelled=self._cancelled)
                        if discovery and (edge["score"] < minimum_score or not edge["phase_correlation"]["valid"]):
                            guided_evaluation = copy.deepcopy(edge)
                            edge = discovery.register(a, b, pair, self.settings, self.mode)
                            edge["guided_attempt"] = guided_evaluation
                except recoverable_pair_errors as exc:
                    # Retire numerical locals before this executor slot can be
                    # reused. A Future otherwise retains failed grids until its
                    # result is consumed, beyond the planned active-task set.
                    traceback.clear_frames(exc.__traceback__)
                    raise
                return pair, edge

            def pair_evaluation(pair, outcome, *, edge=None, error=None):
                a = self.tiles[pair["i"]]
                b = self.tiles[pair["j"]]
                record = {
                    "fixed_id": a["tile_id"],
                    "moving_id": b["tile_id"],
                    "fixed_name": a["display_name"],
                    "moving_name": b["display_name"],
                    "actual_overlap_fraction": float(
                        pair["actual_overlap_fraction"]
                    ),
                    "search_bounds": {
                        "minimum_xyz": np.asarray(
                            pair["search_lo"], dtype=float
                        ).tolist(),
                        "maximum_xyz": np.asarray(
                            pair["search_hi"], dtype=float
                        ).tolist(),
                    },
                    "minimum_accepted_ncc": minimum_score,
                    "outcome": str(outcome),
                    "search_domain": pair.get("search_domain", "guided"),
                }
                if edge is not None:
                    record.update(
                        {
                            "phase_correlation_translation_xyz": np.asarray(
                                edge.get(
                                    "phase_translation_xyz",
                                    edge.get("translation_xyz", (0.0, 0.0, 0.0)),
                                ),
                                dtype=float,
                            ).tolist(),
                            "final_translation_xyz": np.asarray(
                                edge.get("translation_xyz", (0.0, 0.0, 0.0)),
                                dtype=float,
                            ).tolist(),
                            "initial_ncc": float(
                                edge.get("initial_score", edge["translation_score"])
                            ),
                            "translation_ncc": float(edge["translation_score"]),
                            "final_ncc": float(edge["score"]),
                            "translation_grid_voxels_xyz": np.asarray(
                                edge.get(
                                    "translation_grid_voxels_xyz",
                                    (0.0, 0.0, 0.0),
                                ),
                                dtype=float,
                            ).tolist(),
                            "overlap_voxels": int(edge["overlap_voxels"]),
                            "preprocessing": copy.deepcopy(
                                edge.get("preprocessing") or {}
                            ),
                            "phase_correlation": copy.deepcopy(
                                edge.get("phase_correlation") or {}
                            ),
                            "candidate_evaluation": copy.deepcopy(edge.get("candidate_evaluation") or {}),
                            "coarse_candidate_evaluation": copy.deepcopy(edge.get("coarse_candidate_evaluation") or {}),
                            "measured_overlap": copy.deepcopy(edge.get("measured_overlap") or {}),
                            "discovery": copy.deepcopy(edge.get("discovery") or {}),
                        }
                    )
                if error is not None:
                    record["error_type"] = type(error).__name__
                    record["error"] = str(error)
                pair_evaluations.append(record)
                return record

            def record_pair_failure(pair, exc, completed_count):
                a = self.tiles[pair["i"]]
                b = self.tiles[pair["j"]]
                failure = {
                    "fixed_id": a["tile_id"],
                    "moving_id": b["tile_id"],
                    "fixed_name": a["display_name"],
                    "moving_name": b["display_name"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                pair_failures.append(failure)
                pair_evaluation(pair, "numerical_failure", error=exc)
                rejections.append(
                    StitchingRejection(
                        code="numerical_failure",
                        reason=f"{type(exc).__name__}: {exc}",
                        fixed_id=a["tile_id"],
                        moving_id=b["tile_id"],
                        fixed_name=a["display_name"],
                        moving_name=b["display_name"],
                    )
                )
                self._progress_callback(
                    int(completed_count * 75 / total),
                    f"Rejected {a['display_name']} ↔ {b['display_name']} after "
                    f"{type(exc).__name__}: {exc}",
                )

            def record_score_rejection(pair, edge):
                nonlocal score_rejections
                a = self.tiles[pair["i"]]
                b = self.tiles[pair["j"]]
                score_rejections += 1
                evaluation = pair_evaluation(
                    pair, "below_minimum_score", edge=edge
                )
                rejections.append(
                    StitchingRejection(
                        code="below_minimum_score",
                        reason=(
                            f"Registration score {edge['score']:.6g} is below "
                            f"the required minimum {minimum_score:.6g}."
                        ),
                        fixed_id=a["tile_id"],
                        moving_id=b["tile_id"],
                        fixed_name=a["display_name"],
                        moving_name=b["display_name"],
                        score=edge["score"],
                        details={
                            "constraint": copy.deepcopy(edge),
                            "pair_evaluation": copy.deepcopy(evaluation),
                        },
                    )
                )

            def record_invalid_pair_rejection(pair, edge):
                nonlocal invalid_pair_rejections
                a = self.tiles[pair["i"]]
                b = self.tiles[pair["j"]]
                invalid_pair_rejections += 1
                evaluation = pair_evaluation(
                    pair, "invalid_phase_correlation", edge=edge
                )
                rejections.append(
                    StitchingRejection(
                        code="invalid_phase_correlation",
                        reason=(
                            "Phase correlation did not produce numerically valid "
                            "confidence for this pair."
                        ),
                        fixed_id=a["tile_id"],
                        moving_id=b["tile_id"],
                        fixed_name=a["display_name"],
                        moving_name=b["display_name"],
                        score=edge["score"],
                        details={
                            "constraint": copy.deepcopy(edge),
                            "phase_correlation": copy.deepcopy(
                                edge.get("phase_correlation") or {}
                            ),
                            "pair_evaluation": copy.deepcopy(evaluation),
                        },
                    )
                )

            def accept_or_reject_pair(pair, edge):
                phase_valid = (edge.get("phase_correlation") or {}).get("valid")
                discovery_reasons = (edge.get("discovery") or {}).get("rejection_reasons")
                if discovery_reasons:
                    evaluation = pair_evaluation(pair, "searched-without-accepted-match", edge=edge)
                    rejections.append(StitchingRejection(
                        code="discovery-unresolved", reason=", ".join(discovery_reasons),
                        fixed_id=edge["fixed_id"], moving_id=edge["moving_id"],
                        fixed_name=edge["fixed_name"], moving_name=edge["moving_name"], score=edge["score"],
                        details={"constraint": copy.deepcopy(edge), "pair_evaluation": evaluation},
                    ))
                elif phase_valid is False:
                    record_invalid_pair_rejection(pair, edge)
                elif edge["score"] >= minimum_score:
                    edges.append(edge)
                    pair_evaluation(pair, "accepted", edge=edge)
                else:
                    record_score_rejection(pair, edge)

            with common._limit_native_threadpools(worker_count > 1):
                if worker_count == 1:
                    for completed_count, pair in enumerate(pairs, 1):
                        if self._cancelled():
                            return
                        a = self.tiles[pair["i"]]
                        b = self.tiles[pair["j"]]
                        self._progress_callback(
                            int((completed_count - 1) * 75 / total),
                            f"Registering {a['display_name']} ↔ {b['display_name']}",
                        )
                        try:
                            _pair, edge = register_pair(pair)
                        except recoverable_pair_errors as exc:
                            record_pair_failure(pair, exc, completed_count)
                            continue
                        accept_or_reject_pair(pair, edge)
                else:
                    executor = ThreadPoolExecutor(
                        max_workers=worker_count,
                        thread_name_prefix="MADI3D-Stitch-Reg",
                    )
                    futures = {
                        executor.submit(register_pair, pair): pair
                        for pair in pairs
                    }
                    try:
                        completed_count = 0
                        for future in as_completed(futures):
                            if self._cancelled():
                                for pending in futures:
                                    pending.cancel()
                                return
                            pair = futures.pop(future)
                            try:
                                pair, edge = future.result()
                            except recoverable_pair_errors as exc:
                                completed_count += 1
                                record_pair_failure(pair, exc, completed_count)
                                continue
                            completed_count += 1
                            accept_or_reject_pair(pair, edge)
                            a = self.tiles[pair["i"]]
                            b = self.tiles[pair["j"]]
                            self._progress_callback(
                                int(completed_count * 75 / total),
                                f"Completed {completed_count}/{total}: "
                                f"{a['display_name']} ↔ {b['display_name']}",
                            )
                    finally:
                        executor.shutdown(wait=True, cancel_futures=True)

            check_cancelled(self._cancelled)
            # Parallel completion order is nondeterministic; keep pose-graph
            # residual ordering and project JSON stable across runs.
            edges.sort(key=lambda edge: (
                str(edge.get("fixed_id", "")),
                str(edge.get("moving_id", "")),
            ))
            pair_evaluations.sort(
                key=lambda record: (
                    str(record.get("fixed_id", "")),
                    str(record.get("moving_id", "")),
                )
            )

            candidate_pairs = []
            for pair in pairs:
                a = self.tiles[pair["i"]]
                b = self.tiles[pair["j"]]
                candidate_pairs.append(
                    {
                        "fixed_id": a["tile_id"],
                        "moving_id": b["tile_id"],
                        "fixed_name": a["display_name"],
                        "moving_name": b["display_name"],
                        "actual_overlap_fraction": float(
                            pair["actual_overlap_fraction"]
                        ),
                        "search_bounds": {
                            "minimum_xyz": np.asarray(
                                pair["search_lo"], dtype=float
                            ).tolist(),
                            "maximum_xyz": np.asarray(
                                pair["search_hi"], dtype=float
                            ).tolist(),
                        },
                    }
                )

            tile_ids = [tile["tile_id"] for tile in self.tiles]
            if discovery is not None:
                discovery.cache.clear()
                discovery.cache_bytes = 0
            graph_bytes = _pose_graph_bytes(tile_ids, edges, self.mode)
            self._admit_working_memory(graph_bytes)
            # A configured-ceiling override covers this exact graph estimate.
            working_budget = max(working_budget, graph_bytes)
            self._progress_callback(80, "Optimizing the global tile layout")
            globally_rejected = []
            global_mode = self.mode
            if global_mode == "affine":
                corrections, components = _solve_affine_pose_graph(
                    tile_ids, edges, cancelled=self._cancelled,
                    memory_budget_bytes=working_budget,
                    memory_reserved=self.memory_reserved,
                )
            elif global_mode == "rigid":
                corrections, components = _solve_rigid_pose_graph(
                    tile_ids, edges, cancelled=self._cancelled,
                    memory_budget_bytes=working_budget,
                    memory_reserved=self.memory_reserved,
                )
            else:
                (
                    corrections,
                    components,
                    edges,
                    globally_rejected,
                    pose_graph_qc,
                ) = _translation_pose_graph_with_global_rejection(
                    tile_ids, edges, cancelled=self._cancelled,
                    memory_budget_bytes=working_budget,
                    memory_reserved=self.memory_reserved,
                )
            if global_mode != "translation":
                _edge_residuals(edges, corrections, cancelled=self._cancelled)
                pose_graph_qc = _pose_graph_qc_summary(edges)

            evaluation_by_pair = {
                (record["fixed_id"], record["moving_id"]): record
                for record in pair_evaluations
            }
            for rejected_record in globally_rejected:
                constraint = rejected_record["constraint"]
                fixed_id = str(constraint.get("fixed_id") or "")
                moving_id = str(constraint.get("moving_id") or "")
                evaluation = evaluation_by_pair.get((fixed_id, moving_id))
                if evaluation is not None:
                    evaluation["outcome"] = "global_inconsistency_rejected"
                    evaluation["global_rejection"] = {
                        key: copy.deepcopy(value)
                        for key, value in rejected_record.items()
                        if key != "constraint"
                    }
                rejection_details = copy.deepcopy(rejected_record)
                if evaluation is not None:
                    rejection_details["pair_evaluation"] = copy.deepcopy(
                        evaluation
                    )
                rejections.append(
                    StitchingRejection(
                        code="global_inconsistency_outlier",
                        reason=(
                            "Rejected globally inconsistent translation constraint "
                            f"at {rejected_record['residual_voxels']:.6g} "
                            "registration voxels (cutoff "
                            f"{rejected_record['outlier_cutoff_voxels']:.6g})."
                        ),
                        fixed_id=fixed_id,
                        moving_id=moving_id,
                        fixed_name=str(constraint.get("fixed_name") or fixed_id),
                        moving_name=str(constraint.get("moving_name") or moving_id),
                        score=constraint.get("score"),
                        details=rejection_details,
                    )
                )
            mosaic_corrections = corrections
            source_corrections = _source_corrections_from_mosaic(
                mosaic_corrections, preparation
            )

            execution_warnings = list(preparation.warnings)
            if discovery:
                search_coverage.update(searched_pairs=len(pair_evaluations), accepted_pairs=len(edges),
                    searched_without_accepted_match=len(pair_evaluations)-len(edges),
                    not_searched_count=search_coverage["total_possible_pairs"]-len(pair_evaluations))
                if search_coverage["not_searched_count"]:
                    execution_warnings.append(dict(code="discovery-search-incomplete", message=search_coverage["broaden_search"]))
            selected_bound_hits = []
            for edge in edges:
                selection = edge.get("model_selection") or {}
                selected = str(selection.get("selected_model") or "translation")
                candidate = (selection.get("candidates") or {}).get(selected) or {}
                for bound_name in candidate.get("bound_hits") or []:
                    selected_bound_hits.append(
                        f"{edge['fixed_id']}->{edge['moving_id']}:{bound_name}"
                    )
            if selected_bound_hits:
                execution_warnings.append(
                    {
                        "code": "registration-model-bound-hit",
                        "message": (
                            "Accepted advanced registration parameters reached a "
                            "configured deformation bound. Review: "
                            + ", ".join(selected_bound_hits)
                        ),
                        "bound_hits": selected_bound_hits,
                    }
                )
            if not pairs:
                execution_warnings.append(
                    {
                        "code": "no-candidate-registration-edges",
                        "message": (
                            "No candidate overlap was found from the current tile poses "
                            "and configured overlap extent. No alignment was obtained; "
                            "the images were not registered."
                        ),
                    }
                )
            elif not edges:
                if discovery is not None:
                    detail = (
                        f"{len(pair_evaluations)} pair(s) were searched without an accepted match. "
                        "Review overlap support, image ambiguity and rejected candidate details."
                    )
                elif (
                    score_rejections
                    and not pair_failures
                    and not invalid_pair_rejections
                ):
                    rejected_scores = [
                        record["final_ncc"]
                        for record in pair_evaluations
                        if record["outcome"] == "below_minimum_score"
                    ]
                    if len(rejected_scores) == 1:
                        detail = (
                            "1 candidate pair was evaluated, but its NCC "
                            f"{rejected_scores[0]:.6g} was below the required "
                            f"{minimum_score:.6g}."
                        )
                    else:
                        detail = (
                            f"{score_rejections} candidate pairs were below the "
                            f"required NCC {minimum_score:.6g}."
                        )
                elif (
                    pair_failures
                    and not score_rejections
                    and not invalid_pair_rejections
                ):
                    detail = (
                        f"{len(pair_failures)} candidate pair(s) were found, but "
                        "registration failed numerically. Review the rejected pair "
                        "errors."
                    )
                elif (
                    invalid_pair_rejections
                    and not score_rejections
                    and not pair_failures
                ):
                    detail = (
                        f"{invalid_pair_rejections} candidate pair(s) had invalid "
                        "phase-correlation confidence. Review the rejected pair "
                        "diagnostics."
                    )
                else:
                    detail = (
                        f"{score_rejections} pair(s) were below the required NCC and "
                        f"{invalid_pair_rejections} pair(s) had invalid phase "
                        f"confidence; {len(pair_failures)} pair(s) failed numerically."
                    )
                execution_warnings.append(
                    {
                        "code": "no-accepted-registration-edges",
                        "message": (
                            detail
                            + " No alignment was obtained; the images were not "
                            "registered and their existing poses remain unchanged."
                        ),
                    }
                )
            elif len(components) > 1:
                execution_warnings.append(
                    {
                        "code": "disconnected-registration-components",
                        "message": (
                            f"The accepted registration graph has {len(components)} "
                            "components. Relative placement between components is unvalidated; component anchors retained their starting positions."
                        ),
                    }
                )
            if pose_graph_qc["residual_warning_edges"]:
                execution_warnings.append(
                    {
                        "code": "pose-graph-residual-warning",
                        "message": (
                            "The accepted constraints are globally inconsistent at "
                            "or above the conservative review threshold. Review edge(s): "
                            + ", ".join(pose_graph_qc["residual_warning_edges"])
                        ),
                        "edges": pose_graph_qc["residual_warning_edges"],
                    }
                )
            global_inconsistency = pose_graph_qc.get("global_inconsistency") or {}
            globally_rejected_edges = (
                global_inconsistency.get("rejected_edges") or []
            )
            if globally_rejected_edges:
                edge_labels = [
                    f"{edge['fixed_id']}->{edge['moving_id']}"
                    for edge in globally_rejected_edges
                ]
                execution_warnings.append(
                    {
                        "code": "global-inconsistency-edges-rejected",
                        "message": (
                            "Globally inconsistent translation constraint(s) were "
                            "removed and the mosaic was re-solved: "
                            + ", ".join(edge_labels)
                        ),
                        "edges": edge_labels,
                    }
                )
            global_ambiguities = global_inconsistency.get("ambiguities") or []
            if global_ambiguities:
                execution_warnings.append(
                    {
                        "code": "global-inconsistency-ambiguous",
                        "message": (
                            "Global translation disagreement remains ambiguous; "
                            "no constraint was discarded without independent evidence."
                        ),
                        "ambiguities": copy.deepcopy(global_ambiguities),
                    }
                )
            ambiguous_phase = pose_graph_qc["phase_correlation"]["ambiguous_edges"]
            if ambiguous_phase:
                edge_labels = [
                    f"{edge['fixed_id']}->{edge['moving_id']}"
                    for edge in ambiguous_phase
                ]
                execution_warnings.append(
                    {
                        "code": "ambiguous-phase-correlation",
                        "message": (
                            "Phase correlation did not have a unique interpretable "
                            "peak for edge(s): " + ", ".join(edge_labels)
                        ),
                        "edges": edge_labels,
                    }
                )
            review_edge_labels = [
                f"{edge['fixed_id']}->{edge['moving_id']}"
                for edge in edges
                if (edge.get("discovery") or {}).get("review_reasons")
            ]
            manual_review_required = bool(review_edge_labels)
            if manual_review_required:
                review_scores = [
                    float(edge["score"])
                    for edge in edges
                    if (edge.get("discovery") or {}).get("review_reasons")
                ]
                execution_warnings.append(
                    {
                        "code": "discovery-low-confidence-proposal",
                        "message": (
                            "Automatic image discovery found a low-confidence "
                            "alignment proposal. Review the pair evidence before "
                            "applying or using the registered layout: "
                            + ", ".join(review_edge_labels)
                        ),
                        "edges": review_edge_labels,
                        "final_ncc": min(review_scores),
                        "automatic_confidence_threshold": float(
                            self.settings["discovery_minimum_ncc"]
                        ),
                    }
                )

            result = {
                "algorithm_version": common.STITCHING_ALGORITHM_VERSION,
                "execution_status": "succeeded",
                "qc_status": (
                    "warning"
                    if execution_warnings or preparation.assumptions or rejections
                    else "passed"
                ),
                "user_decision": "unapplied",
                "manual_review_required": manual_review_required,
                "mode": global_mode,
                "requested_mode": self.mode,
                "corrections": source_corrections,
                "mosaic_corrections": mosaic_corrections,
                "edges": edges,
                "components": components,
                "pose_graph_qc": pose_graph_qc,
                "translation_acceptance_contract": (
                    copy.deepcopy(TRANSLATION_STITCHING_ACCEPTANCE_CONTRACT)
                    if global_mode == "translation"
                    else None
                ),
                "tile_ids": tile_ids,
                "registration_channel_key": self.settings.get(
                    "registration_channel_key"
                ),
                "registration_channel_label": self.settings.get(
                    "registration_channel_label"
                ),
                "registration_inputs": registration_inputs,
                "placement_evidence": placement_evidence,
                "input_pixel_fingerprints": input_fingerprints,
                "reuse_signature": reuse_signature,
                "reused_calculation": False,
                "search_coverage": search_coverage,
                "discovery_resources": dict(discovery.metrics) if discovery else {},
                "unresolved_component_relationships": {"components": components, "relative_placement_validated": len(components) == 1},
                "candidate_pairs": candidate_pairs,
                "pair_evaluations": pair_evaluations,
                "candidate_pair_count": len(pairs),
                "evaluated_pair_count": len(pair_evaluations),
                "accepted_edge_count": len(edges),
                "below_minimum_score_count": score_rejections,
                "invalid_pair_count": invalid_pair_rejections,
                "global_inconsistency_rejected_count": len(globally_rejected),
                "numerical_failure_count": len(pair_failures),
                "pair_failures": pair_failures,
                "rejections": [item.to_dict() for item in rejections],
                "settings": dict(self.settings),
                "mosaic_geometry": _registration_geometry_provenance(
                    preparation, mosaic_corrections, source_corrections
                ),
                "warnings": execution_warnings,
                "assumptions": preparation.assumptions,
                "completed_with_warnings": bool(
                    execution_warnings or preparation.assumptions or rejections
                ),
            }
            failure_note = (
                f"; {len(pair_failures)} numerical pair failure(s) rejected"
                if pair_failures
                else ""
            )
            self._progress_callback(
                100,
                (
                    "Registration completed with warnings"
                    if result["completed_with_warnings"]
                    else "Registration complete"
                )
                + f" using {worker_count} CPU worker(s){failure_note}",
            )
            check_cancelled(self._cancelled)
            self._completed_callback(StitchingRegistrationResult.from_runtime(result))
        except InterruptedError:
            return
        except Exception:
            self._failed_callback(traceback.format_exc())
