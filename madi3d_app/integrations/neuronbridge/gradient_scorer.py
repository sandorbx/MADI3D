"""Janelia LM->EM two-stage shape scoring (NB-GRAD-06C) plus pinned primitives.

The shape scorer is a native port of the reference characterized by
``tests/oracles/janelia_em_mip_oracle.py``:

    repository : https://github.com/JaneliaSciComp/EM_MIP_search
    commit     : 83ee6f73cee0708f7b013c66d9dbbe11a1b3e7b2 ("Fiji plugin")
    files      : EM_MIP_Mask_Search.java (query masks, per-candidate shape
                 scoring, shortlist budget, normalization, final assembly)

Production never imports the oracle; this module is the executed implementation
of that frozen contract. Java quirks (product wrap mod 65536, ``setSignal1``
keeping raw 1/2 values, the ``> 3`` sum gate, integer ``normalval`` division,
flip-wins-ties, never-selected zero gaps, and the exact ``%.10f`` HALF_UP gap
strings) are reproduced deliberately with reference provenance.

The module also retains the arithmetic primitives (``line_radii``,
``circular_max``, ``gray_no_gamma``, ``slice_gap`` and friends) shared with the
pinned gradient producer, and ``normalize_score``/``GradientParameters``, which
remain the pinned contract for historical ``gradient_execution`` records. Only
fresh row-major filter materialization is used, never a reused Java lazy
iterator. Cohort selection and ranking belong to callers.
"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from functools import cache
import math

import numpy as np
from scipy.ndimage import distance_transform_cdt, maximum_filter1d

from .cdm_lut import LUT_BYTES
from .local_scorer import checkpoint

# Scoring reference revision (EM_MIP_search; the shape-search contract).
REFERENCE_REVISION = "83ee6f73cee0708f7b013c66d9dbbe11a1b3e7b2"
PIXEL_BLOCK = 8192
# EM_MIP_Mask_Search.java L260/L333: "Threshold for mask" default 50.
MASK_THRESHOLD = 50
# EM_MIP_Mask_Search.java L56/L275/L359-361: negative-score region radius
# default 10 (dialog choices {10, 5}; only the pinned default is executed).
NEGATIVE_RADIUS = 10
# EM_MIP_Mask_Search.java L3812: int colorFlux = 40; a Z-gap replacement fires
# at pxGapSlice >= 80 slices and writes (gap - 40).
COLOR_FLUX = 40


class GradientUnavailable(ValueError):
    """Required gradient measurements cannot be computed; never a zero score."""


@dataclass(frozen=True)
class GradientParameters:
    """Retired-method scoring parameters, retained for historical records.

    ``gradient_execution`` records pin exactly this field set; the shape search
    does not consume these values.
    """
    query_threshold: int = 20
    mirror: bool = True
    clear_name_label: bool = True
    clear_color_scale: bool = True
    border_pixels: int = 0

    def __post_init__(self):
        if type(self.query_threshold) is not int or not 0 <= self.query_threshold <= 255:
            raise ValueError("Gradient threshold must be an integer from 0 to 255.")
        for name in ("mirror", "clear_name_label", "clear_color_scale"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean.")
        # These are the two traversals covered by the independent filter oracle.
        if type(self.border_pixels) is not int or self.border_pixels not in (0, 1):
            raise ValueError("Validated gradient borders are 0 and 1 pixels.")


def _rgb(value, shape=None):
    if (not isinstance(value, np.ndarray) or value.dtype != np.uint8
            or value.ndim != 3 or value.shape[-1] != 3 or min(value.shape[:2]) <= 0):
        raise ValueError("CDM inputs require original uint8 RGB pixels.")
    if shape is not None and value.shape[:2] != shape:
        raise ValueError("Shape scoring inputs must share the declared aligned canvas.")
    return value


def gray_no_gamma(rgb):
    """Java's RGB -> GRAY16 branch uses a 255 scale and equal double weights."""
    r, g, b = np.moveaxis(rgb, -1, 0)
    return (r * (1 / 3.) + g * (1 / 3.) + b * (1 / 3.) + .5).astype(np.uint16)


def line_radii(radius):
    if not isinstance(radius, (int, float)) or not math.isfinite(radius) or not 0 <= radius <= 60:
        raise ValueError("Filter radius must be finite and between 0 and 60.")
    if 1.5 <= radius < 1.75:
        radius = 1.75
    elif 2.5 <= radius < 2.85:
        radius = 2.85
    r2 = int(radius * radius) + 1
    k = int(math.sqrt(r2 + 1e-10))
    return tuple(k if dy == 0 else int(math.sqrt(r2 - dy * dy + 1e-10))
                 for dy in range(-k, k + 1))


def circular_max(rgb, radius, cancel=None, *, border_pixels=0):
    """First fresh row-major unsafeMaxFilter materialization, per RGB channel.

    Decompose the exact disk into horizontal runs. Each distinct run width
    needs one linear-time 1D maximum, shared by its row offsets. Accumulate
    shifted views into one output, with zero outside the canvas. Two additional
    RGB buffers coexist here; no footprint tensor or per-pixel Python loop.
    Java's later reused lazy traversals are intentionally not this interface.
    """
    _rgb(rgb)
    return _circular_max_nonnegative(rgb, radius, cancel, border_pixels=border_pixels)


def _circular_max_nonnegative(pixels, radius, cancel=None, *, border_pixels=0):
    """Shared disk kernel for validated scalar or RGB unsigned-byte pixels.

    Zero and nearest-edge padding give the same maximum for this nonnegative,
    centered disk: an extended edge sample is also covered by an in-canvas run
    with at least the same half-width. This lets the ImageJ grayscale producer
    and the shape scorer's dilations share the exact filter, not their roles.
    """
    if type(border_pixels) is not int or border_pixels not in (0, 1):
        raise ValueError("Validated filter borders are 0 and 1 pixels.")
    checkpoint(cancel)
    radii = line_radii(radius)
    k = len(radii) // 2
    h = pixels.shape[0]
    result = np.zeros_like(pixels)
    runs = np.empty_like(pixels)
    for half in sorted(set(radii)):
        checkpoint(cancel)
        maximum_filter1d(pixels, 2 * half + 1, axis=1, mode="constant", cval=0, output=runs)
        for dy, dx in enumerate(radii, -k):
            if dx != half or abs(dy) >= h:
                continue
            if dy < 0:
                np.maximum(result[-dy:], runs[:h + dy], out=result[-dy:])
            else:
                np.maximum(result[:h - dy], runs[dy:], out=result[:h - dy])
    if border_pixels:
        result[[0, -1]] = 0
        result[:, [0, -1]] = 0
    checkpoint(cancel)
    return result


# Sector ranges and palette are exactly those in the reference Z-decoder. Share
# the existing canonical palette, rather than introducing another LUT copy.
_SECTORS = {(0, 1): (171, 212), (0, 2): (213, 255),
            (1, 0): (128, 170), (1, 2): (86, 127),
            (2, 0): (0, 29), (2, 1): (30, 85)}
_LUT = np.array([line.split() for line in LUT_BYTES.splitlines()], dtype=np.float64)


def find_slice_number(maximum_channel, second_channel, ratio):
    """Scalar/vector sector lookup; first strict closest (or exact) LUT entry."""
    start, end = _SECTORS[maximum_channel, second_channel]
    values = np.asarray(ratio, dtype=np.float64)
    best = np.full(values.shape, 1000., dtype=np.float64)
    slices = np.zeros(values.shape, np.uint16)
    for index in range(start, end + 1):
        ordered = np.sort(_LUT[index])
        lut_ratio = ordered[1] / ordered[2] if ordered[0] != ordered[1] and ordered[1] != ordered[2] else 0.
        distance = np.abs(values - lut_ratio)
        closer = distance < best
        slices[closer] = index + 1
        best[closer] = distance[closer]
    return slices


@cache
def _slice_tables():
    # 6 * 256 * 256 * 2 bytes, independent of canvas size. NaN (black) stays 0.
    high = np.arange(256, dtype=np.float64)[:, None]
    low = np.arange(256, dtype=np.float64)[None, :]
    ratio = np.full((256, 256), np.nan)
    np.divide(low, high, out=ratio, where=high != 0)
    return {sector: find_slice_number(*sector, ratio) for sector in _SECTORS}


def depth_slices(rgb):
    """RGB vectors to one-based slices, with Java's red/green/blue tie order."""
    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim < 1 or pixels.shape[-1] != 3:
        raise ValueError("Depth colors require uint8 RGB.")
    r, g, b = np.moveaxis(pixels, -1, 0)
    first = np.where((r >= g) & (r >= b), 0, np.where(g >= b, 1, 2))
    second = np.where(first == 0, np.where(g >= b, 1, 2),
                      np.where(first == 1, np.where(r >= b, 0, 2), np.where(r >= g, 0, 1)))
    result = np.zeros(first.shape, np.uint16)
    for (a, b), table in _slice_tables().items():
        mask = (first == a) & (second == b)
        result[mask] = table[pixels[..., a][mask], pixels[..., b][mask]]
    return result


def slice_gap(rgb_a, rgb_b):
    a, b = depth_slices(rgb_a).astype(np.int32), depth_slices(rgb_b).astype(np.int32)
    return np.where((a == 0) | (b == 0), b, np.abs(a - b))


def _black(rgb):
    """Reference opaque-black test (``== -16777216`` on an Integer pixel)."""
    return (rgb[..., 0] == 0) & (rgb[..., 1] == 0) & (rgb[..., 2] == 0)


def _set_signal_one(gray16):
    """CDM_area_measure ``setSignal1``: pixels > 2 become 1; raw values <= 2
    survive unchanged and multiply into the score products (reference quirk)."""
    value = np.asarray(gray16).copy()
    value[value > 2] = 1
    return value


def _distance_to_signal(dilated_binary):
    """Four-connected distance to the dilated signal (the reference
    ``gradientSlice`` wavefront): 0 inside, d beyond, all-65535 when empty."""
    if not dilated_binary.any():
        return np.full(dilated_binary.shape, 65535, np.uint16)
    return distance_transform_cdt(dilated_binary == 0, metric="taxicab").astype(np.uint16)


def _zgap_replace(score, rgb_mask_side, rgb_data_side):
    """``deleteMatchZ``: where both the dilated and the original CDM are
    non-black and the LUT Z gap is >= 80 slices, the running per-pixel shape
    value is replaced by ``gap - COLOR_FLUX`` (L3838-3840)."""
    region = (~_black(rgb_data_side)) & (~_black(rgb_mask_side))
    if not region.any():
        return score
    gap = slice_gap(rgb_mask_side[region], rgb_data_side[region]).astype(np.int64)
    replace = gap >= 2 * COLOR_FLUX
    if replace.any():
        flat = score.reshape(-1)
        positions = np.flatnonzero(region.reshape(-1))[replace]
        flat[positions] = gap[replace] - COLOR_FLUX
    return score


@dataclass(frozen=True)
class ShapeMeasurement:
    """One candidate's raw bidirectional shape measurements and mirror choice.

    ``selected_value`` is the reference's ``realval`` (the value that enters
    the area-gap normalization); flip trials win ties (L3124).
    """

    mask_to_sample: int
    sample_to_mask: int
    normal_value: int
    flip_value: int | None
    mirrored: bool
    selected_value: int


class ShapeScorer:
    """Query-side precomputation of the reference ``CDM_area_measure`` stage.

    All per-candidate work is a bounded sequence of fresh vectorized
    traversals; the query masks are built once and reused. ``mirror`` selects
    whether flipped trials run (ties go to the flip, L3124). Copies insulate
    reuse from caller mutation; the borrowed RGB input is never modified.
    """

    def __init__(self, query_rgb, *, mirror=True, mask_threshold=MASK_THRESHOLD,
                 negative_radius=NEGATIVE_RADIUS, shape_boundary=None, cancel=None):
        _rgb(query_rgb)
        if type(mirror) is not bool:
            raise ValueError("Shape mirror preference must be boolean.")
        if type(mask_threshold) is not int or not 0 <= mask_threshold <= 255:
            raise ValueError("Mask threshold must be an integer from 0 to 255.")
        if type(negative_radius) is not int or not 0 <= negative_radius <= 60:
            raise ValueError("Negative radius must be an integer from 0 to 60.")
        if shape_boundary is not None and (
                not isinstance(shape_boundary, np.ndarray) or shape_boundary.dtype != np.bool_
                or shape_boundary.shape != query_rgb.shape[:2]):
            raise ValueError("Shape boundary requires boolean pixels on the exact query canvas.")
        checkpoint(cancel)
        self.mirror = mirror
        self.negative_radius = negative_radius
        gray16 = gray_no_gamma(query_rgb)
        # Gradient carrier: round((r+g+b)/3) > thresm in the zero-extended
        # 16-bit domain (L2549-2561).
        binary = np.where(gray16 > mask_threshold, np.uint16(1), np.uint16(0))
        dilated = _circular_max_nonnegative(binary, negative_radius, cancel)
        # RGB overlay carrier: black unless ANY channel > thresm (L2563-2570).
        rgb_mask = query_rgb.copy()
        rgb_mask[~np.any(rgb_mask > mask_threshold, axis=2)] = 0
        rgb_dilated = _circular_max_nonnegative(rgb_mask, negative_radius, cancel)
        self.mask_gradient_flip = self.rgb_mask_flip = self.value1_flip = None
        outside = None if shape_boundary is None else ~shape_boundary
        if mirror:
            # L2631-2650: flip copies first, then clip all four carriers.
            flipped = np.ascontiguousarray(dilated[:, ::-1])
            self.rgb_mask_flip = np.ascontiguousarray(rgb_dilated[:, ::-1])
            if outside is not None:
                dilated[outside] = 0
                flipped[outside] = 0
                rgb_dilated[outside] = 0
                self.rgb_mask_flip[outside] = 0
            self.mask_gradient_flip = _distance_to_signal(flipped)
        # With mirror OFF these two carriers are deliberately not clipped.
        self.mask_gradient = _distance_to_signal(dilated)
        self.rgb_mask_dilated = rgb_dilated
        self.value1 = _set_signal_one(gray16)
        if outside is not None:
            self.value1[outside] = 0
        self.query_rgb = query_rgb.copy()
        if mirror:
            # L2720-2741: clipped ordinary value1 -> flip -> clip again.
            self.value1_flip = np.ascontiguousarray(self.value1[:, ::-1])
            if outside is not None:
                self.value1_flip[outside] = 0
        for array in (self.mask_gradient, self.rgb_mask_dilated, self.value1,
                      self.query_rgb, self.mask_gradient_flip, self.rgb_mask_flip,
                      self.value1_flip):
            if array is not None:
                array.flags.writeable = False
        checkpoint(cancel)

    def measure(self, hit_rgb, candidate_gradient, cancel=None):
        """Score one shortlisted candidate; ``hit_rgb`` is its aligned CDM
        slice and ``candidate_gradient`` the 06B-supplied uint16 gradient."""
        _rgb(hit_rgb, self.query_rgb.shape[:2])
        if (not isinstance(candidate_gradient, np.ndarray)
                or candidate_gradient.dtype != np.uint16
                or candidate_gradient.shape != self.query_rgb.shape[:2]):
            raise ValueError("Candidate gradient requires uint16 pixels on the aligned canvas.")
        checkpoint(cancel)
        signal = _set_signal_one(gray_no_gamma(hit_rgb))
        dilated_em = _circular_max_nonnegative(hit_rgb, self.negative_radius, cancel)

        # MaskToSample: EM brightness x query-mask distance (L2904-2935).
        m2s = self._measure_pair(signal * self.mask_gradient, hit_rgb,
                                 self.rgb_mask_dilated, cancel)
        # SampleToMask: candidate gradient x binarized query mask (L3025-3056).
        s2m = self._measure_pair(candidate_gradient * self.value1,
                                 self.query_rgb, dilated_em, cancel)
        normal = (s2m + m2s) // 2  # L3076 integer division
        value, flip, flipped = normal, None, False
        if self.mirror:
            m2s_flip = self._measure_pair(
                signal * self.mask_gradient_flip, hit_rgb, self.rgb_mask_flip, cancel)
            s2m_flip = self._measure_pair(
                candidate_gradient * self.value1_flip, self.query_rgb,
                np.ascontiguousarray(dilated_em[:, ::-1]), cancel)
            flip = (s2m_flip + m2s_flip) // 2  # L3123 integer division
            if flip <= normal:  # L3124: ties go to the flip
                value, flipped = flip, True
        checkpoint(cancel)
        return ShapeMeasurement(mask_to_sample=m2s, sample_to_mask=s2m,
                                normal_value=normal, flip_value=flip,
                                mirrored=flipped, selected_value=value)

    @staticmethod
    def _measure_pair(product, rgb_mask_side, rgb_data_side, cancel):
        # Products are stored through a ShortProcessor: the uint16 product
        # keeps Java's low-16-bit wrap, and get() reads values back unsigned.
        wrapped = product.astype(np.int64)
        checkpoint(cancel)
        replaced = _zgap_replace(wrapped, rgb_mask_side, rgb_data_side)
        checkpoint(cancel)
        return int(np.sum(replaced[replaced > 3]))


def positive_label(pct, filename):
    """Slice-title construction L690-710 with getZeroFilledNumString L202-212:
    ``Math.round(pct*100*100)/100`` formatted ``%.2f``, zero-filled to three
    integer digits, plus ``_`` + filename (labelmethodE is fixed at 1)."""
    value = pct * 100.0 * 100.0
    rounded = math.floor(value + 0.5)  # Java Math.round: floor(x + 0.5)
    text = "%.2f" % (rounded / 100.0)
    integer_digits = len(text) - 3
    if integer_digits < 3:
        text = "0" * (3 - integer_digits) + text
    return text + "_" + filename


def stage1_cap(n_positive, maxnumber):
    """L1637-1642: the positive-sorted hit list is truncated to 400 before
    shape scoring (or maxnumber+200 when maxnumber > 400)."""
    shortlist = n_positive
    if shortlist > 400:
        shortlist = 400
        if maxnumber > 400:
            shortlist = maxnumber + 200
    return shortlist


def positive_stack_slice(n_shortlist, maxnumber, fl_positive):
    """L2759-2764: the shape-scoring budget is
    min(shortlist length, maxnumber + FLpositive + 50)."""
    count = n_shortlist
    if count > maxnumber + fl_positive + 50:
        count = maxnumber + fl_positive + 50
    return count


def _format_gap(double_gap):
    """Java gap string: zero-pad aligning the integer part to five digits,
    then ``%.10f`` with HALF_UP rounding of the exact double (L3192-3201)."""
    if math.isnan(double_gap):
        return "NaN"
    if double_gap < 10000 and double_gap > 999.999999:
        pad = "0"
    elif double_gap < 1000 and double_gap > 99.999999:
        pad = "00"
    elif double_gap < 100 and double_gap > 9.999999:
        pad = "000"
    elif double_gap < 10:
        pad = "0000"
    else:
        pad = ""
    return pad + str(Decimal(double_gap).quantize(Decimal("0.0000000001"),
                                                  rounding=ROUND_HALF_UP))


def normalize_gaps(scores, areaarray, max_score, max_area_gap):
    """Per-entry normalization against cohort maxima (L3152-3211).

    ``scores`` are positive label floats and ``areaarray`` the selected shape
    values. normArea = area/max * 2 clamped to [0.002, 1]; normScore has no
    clamps; gap = normScore/normArea*100 with the exact reference string form.
    Returns ``(gap_strings, gap_values)`` — the value is Java's ``parseFloat``
    of the padded string, which is what ordering and emission compare. A
    zero-maximum cohort yields NaN exactly as Java's double 0/0 does, and NaN
    gaps never order below a numeric string ("NaN" sorts above digits in
    descending string order and never compares equal as a value).
    """
    gaps, values = [], []
    for score, area in zip(scores, areaarray, strict=True):
        norm_area = math.nan if max_area_gap == 0 else area / max_area_gap
        norm_score = math.nan if max_score == 0 else score / max_score
        norm_area = norm_area * 2.0
        if norm_area > 1:
            norm_area = 1.0
        if norm_area < 0.002:
            norm_area = 0.002
        double_gap = (norm_score / norm_area) * 100.0
        text = _format_gap(double_gap)
        gaps.append(text)
        values.append(float("nan") if text == "NaN" else float(text))
    return gaps, values


def combined_gap(score, area_gap, max_score, max_area_gap):
    """The pinned single-entry normalization used to re-verify one stored
    combined gap against its cohort maxima. Returns ``(gap_text, gap_value)``."""
    gaps, values = normalize_gaps([float(score)], [area_gap], max_score, max_area_gap)
    return gaps[0], values[0]


@dataclass(frozen=True)
class ShapeAssemblySelection:
    reference_index: int
    score_source_index: int
    fl_renamed: bool
    gap_text: str
    mirrored: bool


def assemble_ranking(labels, gaps, gap_values, flips, *, maxnumber):
    """L3216-3605 final assembly: sort gap strings descending, merge ``_FL``
    duplicate bodies, then emit at most ``maxnumber`` entries.

    Returns frozen selections separating the final assembly reference slot
    from the candidate supplying its score and scientific mirror decision.

    Faithful points that must not be silently changed:
      * the FL duplicate-body merge re-runs on EVERY output iteration and
        mutates the working table; an FL-typed entry whose non-FL twin exists
        is consumed ("0 NN"); the twin takes the FL gap only when strictly
        greater and is then flagged fixedFL (ties keep the twin's own gap);
      * selection takes the first array-order entry whose parsed gap equals
        the sorted gap value and is non-zero — zero gaps are never selected;
      * scans A and B both run every iteration (scan B's FL branch has no
        zero guard, matching L3495);
      * emission stops once ``maxnumber`` entries were added; consumed
        duplicates leave holes in the reference's own label numbering — the
        reference index preserves the assembly slot instead.
    """
    n = len(labels)
    # Each row retains (gap_text, parsed_value, name, score_source_index).
    # Consumed rows become the reference's "0 NN" sentinel (parsed value 0.0 participates in the
    # reference's equality comparisons exactly as Java's parseFloat("0") does).
    table = [[gaps[i], gap_values[i], labels[i], i] for i in range(n)]  # L3205-3207
    sorted_gaps = sorted(gaps, reverse=True)  # L3216 reverse string order
    fixed_fl = [-1] * (n + 1)                 # L3222-3223
    added = 0
    emitted = []

    def body_of(name):
        """Reference body extraction: text between the first '_' and the last
        '_FL', plus '.tif' (L3261-3262)."""
        fl = name.rfind("_FL")
        under = name.find("_")
        return name[under + 1:fl] + ".tif"

    for inew in range(n):                     # L3219: Finslice = n
        if added >= maxnumber:                # L3236-3237
            break
        totalscore = float(sorted_gaps[inew]) if sorted_gaps[inew] != "NaN" else math.nan
        slicename = None
        # -- duplicate-body "_FL" merge (re-runs every iteration, L3246-3321) --
        for iscan in range(len(table)):
            arg = table[iscan]
            oriname = arg[2]
            if oriname.rfind("_FL") == -1:
                continue
            truebody = body_of(oriname)
            for ibody in range(len(table)):
                if ibody == iscan:            # the reference's own index guard
                    continue
                target = table[ibody]
                if target[2] == "NN":         # L3270
                    continue
                if target[2].rfind("_FL") != -1:  # L3273
                    continue
                underindextarget = target[2].find("_")
                if target[2][underindextarget + 1:] != truebody:  # L3278
                    continue
                table[iscan] = ["0", 0.0, "NN", -1]  # L3285
                if arg[1] > target[1]:           # L3287 STRICT on parsed value
                    # L3303: the twin takes the FL gap; fixedFL marks the
                    # name rewrite at emission.
                    table[ibody] = [arg[0], arg[1], target[2], arg[3]]
                    fixed_fl[ibody] = ibody      # L3306
                else:
                    table[ibody] = [target[0], target[1], target[2], target[3]]  # L3309
                break
        # -- select the first array-order entry with this exact gap value --
        for iscan in range(len(table)):       # L3324
            arg = table[iscan]
            if arg[1] == totalscore and arg[1] != 0:  # L3334 zero never selected
                slicename = arg[2]
                break
        if slicename == "NN":                 # L3355 guard; None still scans
            continue

        def emit(search_index, gap_text):
            score_source = table[search_index][3]
            emitted.append(ShapeAssemblySelection(
                search_index, score_source, fixed_fl[search_index] != -1,
                gap_text, bool(flips[score_source])))
            table[search_index] = ["0", 0.0, "NN", -1]

        # -- scan A (L3357-3480): name-equality match, FL twins included --
        for search_index in range(n):
            arg = table[search_index]
            name = arg[2]
            if name.rfind("_FL") != -1:
                # L3368-3382: an FL entry whose twin body equals the selected
                # full label emits itself (virtual mode) unless already fixed.
                if body_of(name) == slicename and fixed_fl[search_index] == -1:
                    emit(search_index, sorted_gaps[inew])
                    added += 1                # L3420
                    break
                continue
            if name == slicename:             # L3455
                emit(search_index, sorted_gaps[inew])
                added += 1                    # L3470
                break
        # -- scan B (L3482-3601): runs unconditionally after scan A --
        for search_index in range(n):
            arg = table[search_index]
            name = arg[2]
            if name.rfind("_FL") != -1:
                if totalscore == arg[1]:      # L3491-3495 (no zero guard)
                    if fixed_fl[search_index] == -1:  # L3507
                        emit(search_index, sorted_gaps[inew])  # virtual L3511
                        added += 1            # L3543
                        break
                continue
            if name == slicename:             # L3576
                emit(search_index, sorted_gaps[inew])
                added += 1                    # L3591
                break
    return tuple(emitted)


# ---------------------------------------------------------------------------
# Retired-method normalization contract, retained for historical records.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedGradientScore:
    utility_float64: float
    stored_float32: np.float32


def _signed_integer(value, bits, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not -(1 << (bits - 1)) <= int(value) < (1 << (bits - 1))):
        raise ValueError(f"{name} must be a signed {bits}-bit integer.")
    return int(value)


def normalize_score(pixel_match, shape_score, max_pixel_match, max_shape_score):
    """Retired pinned utility with caller-supplied independent cohort maxima.

    Its negative-shape sentinel fallback is retained for scalar oracle parity.
    Callers must not normalize an unavailable pair as a valid measurement.
    """
    if shape_score is None:
        raise GradientUnavailable("A raw shape measurement is required for normalization.")
    pixel_match = _signed_integer(pixel_match, 32, "pixel_match")
    shape_score = _signed_integer(shape_score, 64, "shape_score")
    max_pixel_match = _signed_integer(max_pixel_match, 64, "max_pixel_match")
    max_shape_score = _signed_integer(max_shape_score, 64, "max_shape_score")
    if pixel_match == 0 or max_pixel_match == 0 or shape_score < 0 or max_shape_score <= 0:
        utility = float(pixel_match)
    else:
        pixel = float(pixel_match) / max_pixel_match
        shape = float(shape_score) / max_shape_score
        bounded = min(max(shape * 2.5, .002), 1.)
        utility = pixel / bounded * 100
    return NormalizedGradientScore(utility, np.float32(utility))
