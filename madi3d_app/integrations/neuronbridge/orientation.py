"""Conservative template-X placement for retained brain CDM results."""

import hashlib
import io

import numpy as np
from PIL import Image

from madi3d_app.volume.geometry import DEFAULT_FLIP_REFERENCES, local_axis_flip_matrix

from .query import query_has_template_placement, query_png
from .search_profiles import BRAIN, query_profile


def _x_side(low, high):
    # Match the existing Flip X plane. Leave one brain-canvas pixel of
    # clearance on either side before treating an extent as unilateral.
    midpoint = DEFAULT_FLIP_REFERENCES["x"] / 2
    clearance = BRAIN.spacing_xyz_um[0]
    if np.isfinite((low, high)).all() and low <= high:
        if high < midpoint - clearance:
            return -1
        if low > midpoint + clearance:
            return 1
    return 0


def query_template_x_side(record, searched_png_sha256):
    """Return a side, 0 for ambiguous foreground, or None without a mapped query."""
    if not record or not searched_png_sha256:
        return None
    if not query_has_template_placement(record) or query_profile(record) != BRAIN:
        return None
    png = query_png(record)
    if hashlib.sha256(png).hexdigest() != searched_png_sha256:
        return None
    with Image.open(io.BytesIO(png)) as image:
        if image.mode != "RGB" or image.size != BRAIN.search_canvas_yx[::-1]:
            return None
        columns = np.flatnonzero(np.any(np.asarray(image), axis=(0, 2)))
    if not columns.size:
        return 0
    # Generated CDMs already place pixels at the mapped template grid indices.
    # The retained Brain grid has +X columns and no horizontal canvas padding.
    x = BRAIN.origin_xyz_um[0] + columns[[0, -1]] * BRAIN.spacing_xyz_um[0]
    return _x_side(*x)


def mesh_needs_template_x_flip(bounds, pose, query_side):
    """Check the X extent before and after the existing local-axis reflection."""
    intervals = np.asarray(bounds, dtype=float).reshape(3, 2)
    if not np.isfinite(intervals).all() or np.any(intervals[:, 0] > intervals[:, 1]):
        return False

    def side(matrix):
        # The transformed bounding box is conservative for oblique poses.
        ends = matrix[0, :3, None] * intervals
        return _x_side(matrix[0, 3] + ends.min(axis=1).sum(),
                       matrix[0, 3] + ends.max(axis=1).sum())

    return (side(pose) == -query_side
            and side(pose @ local_axis_flip_matrix(0, DEFAULT_FLIP_REFERENCES["x"])) == query_side)
