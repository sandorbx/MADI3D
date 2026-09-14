"""Positive CDS, translated from Janelia's pinned mipsearch.js (BSD-3-Clause).

See docs/licenses/neuronbridge-services.txt. No image conversion, GUI or I/O.
The ratio gaps deliberately retain the reference's signed boundary values and
strict component comparisons, including its treatment of equal components.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .search_profiles import BRAIN, SEARCH_PROFILES, search_profile

REFERENCE_REVISION = "3847c602fcc5e3ae2bae47b7308ee11263ae6cef"
ALGORITHM = "janelia-positive-cds/numpy-v1"
PIXEL_BLOCK = 8192


class SearchCancelled(InterruptedError):
    pass


def checkpoint(cancel):
    if cancel is not None and cancel.is_set():
        raise SearchCancelled("Cancelled. No incomplete ranking was published.")


@dataclass(frozen=True)
class SearchParameters:
    query_threshold: int = 20
    candidate_threshold: int = 20
    color_tolerance_percent: float = 2.0
    xy_shift: int = 0
    mirror: bool = True
    minimum_overlap: float = 0.0
    result_limit: int = 100
    memory_mb: int = 256

    def __post_init__(self):
        for name in ("query_threshold", "candidate_threshold"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 255:
                raise ValueError(f"{name} must be an integer from 0 to 255.")
        for name, maximum in (("color_tolerance_percent", 100), ("minimum_overlap", 1)):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be finite and between 0 and {maximum}.")
        if type(self.xy_shift) is not int or self.xy_shift not in range(0, 21, 2):
            raise ValueError("XY shift must be 0, 2, 4, …, 20 pixels.")
        if type(self.mirror) is not bool:
            raise ValueError("Mirror must be a boolean.")
        if type(self.result_limit) is not int or not 1 <= self.result_limit <= 5000:
            raise ValueError("Result limit must be between 1 and 5000.")
        if type(self.memory_mb) is not int or not 64 <= self.memory_mb <= 4096:
            raise ValueError("Search memory budget must be between 64 and 4096 MiB.")

    def to_dict(self):
        return asdict(self)


def validate_rgb(value, profile=None):
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("Search images must retain their original uint8 RGB pixels.")
    maximum = max(p.search_canvas_yx[0] * p.search_canvas_yx[1] for p in SEARCH_PROFILES.values())
    if profile is not None and value.shape[:2] != profile.search_canvas_yx:
        raise ValueError("Search canvas disagrees with its selected profile.")
    if not 0 < value.shape[0] * value.shape[1] <= maximum:
        raise ValueError("Invalid or oversized search canvas.")
    return value


def _sectors(rgb):
    r, g, b = np.asarray(rgb, dtype=np.float64).T
    sector = np.zeros(r.shape, np.uint8)
    ratio = np.zeros(r.shape, np.float64)
    for number, high, low, condition in (
        (1, b, r, (b > r) & (b > g) & (r > g)),
        (2, b, g, (b > r) & (b > g) & (r <= g)),
        (3, g, b, (g > b) & (g > r) & (b > r)),
        (4, g, r, (g > b) & (g > r) & (b <= r)),
        (5, r, g, (r > b) & (r > g) & (g > b)),
        (6, r, b, (r > b) & (r > g) & (g <= b)),
    ):
        sector[condition] = number
        np.divide(low, high, out=ratio, where=condition)
    return sector, ratio


def _gaps(s1, a, s2, b):
    gap = np.full(a.shape, 10000., np.float64)
    same = (s1 == s2) & (s1 != 0) & (a > 0) & (b > 0)
    gap[same] = np.abs(b[same] - a[same])
    # Boundary conditions and constants are literal upstream values. Gaps
    # may be negative: taking abs here would silently change matching counts.
    for lo, hi, boundary, limit_lo, limit_hi, increasing in (
        (1, 2, .354862745, .44, .54, False),
        (2, 3, .996078431, .8, .8, True),
        (3, 4, .505882353, .7, .7, False),
        (4, 5, .996078431, .8, .8, True),
        (5, 6, .505882353, .7, .7, False),
    ):
        forward, backward = (s1 == lo) & (s2 == hi), (s1 == hi) & (s2 == lo)
        if increasing:
            eligible = ((forward & (a > limit_lo) & (b > limit_hi)) |
                        (backward & (a > limit_hi) & (b > limit_lo)))
            gap[eligible] = (boundary - a[eligible]) + (boundary - b[eligible])
        else:
            eligible = ((forward & (a < limit_lo) & (b < limit_hi)) |
                        (backward & (a < limit_hi) & (b < limit_lo)))
            gap[eligible] = (a[eligible] - boundary) + (b[eligible] - boundary)
    return gap


def pixel_gaps(query_rgb, candidate_rgb):
    """Independent vector interface for the upstream pixel-gap oracle tests."""
    a, b = np.asarray(query_rgb), np.asarray(candidate_rgb)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError("Pixel pairs must be equally sized uint8 RGB arrays.")
    return _gaps(*_sectors(a), *_sectors(b))


def shifts(xy_shift):
    # Janelia includes (0,0) in each radius, with x outermost and y innermost.
    return tuple((x, y) for radius in range(2, xy_shift + 1, 2)
                 for x in (-radius, 0, radius) for y in (-radius, 0, radius)) if xy_shift else ((0, 0),)


@dataclass(frozen=True)
class Score:
    matching_pixels: int
    query_foreground_pixels: int
    overlap_fraction: float
    mirrored: bool
    shift: tuple[int, int] | None


class PreparedQuery:
    """One query plus bounded pixel blocks; never materializes all shifted masks."""

    def __init__(self, rgb, parameters, cancel=None, *, profile_id=None):
        self.profile = None if profile_id is None else search_profile(profile_id)
        validate_rgb(rgb, self.profile)
        if not isinstance(parameters, SearchParameters):
            raise ValueError("Validated search parameters are required.")
        checkpoint(cancel)
        self.parameters, self.shape = parameters, rgb.shape
        height, width = rgb.shape[:2]
        foreground = np.any(rgb > parameters.query_threshold, axis=2)
        for anchor, x, y, w, h in (self.profile or BRAIN).score_exclusions:
            left = x if anchor == "left" else width - x - w
            foreground[max(0, y):min(height, y + h), max(0, left):min(width, left + w)] = False
        self.positions = np.flatnonzero(foreground).astype(np.int32)
        if not len(self.positions):
            raise ValueError("The query has no foreground above its threshold outside the label regions.")
        pixels = rgb.reshape(-1, 3)[self.positions]
        self.sectors, self.ratios = _sectors(pixels)
        self.foreground_count = len(self.positions)

    def score(self, candidate, cancel=None):
        validate_rgb(candidate, self.profile)
        if candidate.shape != self.shape:
            raise ValueError("Candidate and query canvases must agree; automatic resizing is forbidden.")
        flat = candidate.reshape(-1, 3)
        return self.score_pixels(lambda positions: flat[positions], cancel)

    def score_pixels(self, read_pixels, cancel=None):
        """Score a lossless indexed pixel reader at flat, in-bounds positions."""
        parameters = self.parameters
        height, width = self.shape[:2]
        best, chosen, mirrored = 0, None, False
        for mirror in ((False, True) if parameters.mirror else (False,)):
            for dx, dy in shifts(parameters.xy_shift):
                count = 0
                for start in range(0, self.foreground_count, PIXEL_BLOCK):
                    checkpoint(cancel)
                    end = start + PIXEL_BLOCK
                    positions = self.positions[start:end]
                    x, y = positions % width + dx, positions // width + dy
                    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                    if mirror:
                        x = width - 1 - x  # mirror after shifting, with invalid positions excluded
                    pixels = read_pixels((y[valid] * width + x[valid]).astype(np.int32))
                    s, ratios = _sectors(pixels)
                    gaps = _gaps(self.sectors[start:end][valid], self.ratios[start:end][valid], s, ratios)
                    count += int(np.count_nonzero(np.any(pixels > parameters.candidate_threshold, axis=1) &
                                                 (gaps <= parameters.color_tolerance_percent / 100.)))
                if count > best:  # first winning shift, unmirrored on ties
                    best, chosen, mirrored = count, (dx, dy), mirror
        checkpoint(cancel)
        return Score(best, self.foreground_count, best / self.foreground_count, mirrored, chosen)


def score(query, candidate, parameters=None, cancel=None, *, profile_id=None):
    return PreparedQuery(query, parameters or SearchParameters(), cancel, profile_id=profile_id).score(candidate, cancel)
