"""Qt/VTK-free SciPy sampling shared by registration and bounded CDM tiles."""
from __future__ import annotations

import numpy as np

from .geometry import affine_matrix4


def affine_pull_mapping_zyx(source_world_affine, output_world_affine):
    """Map output ZYX indices to source ZYX indices using exact XYZ affines."""
    permutation = np.eye(4)[[2, 1, 0, 3]]
    mapping = np.linalg.solve(
        affine_matrix4(source_world_affine), affine_matrix4(output_world_affine)
    )
    return permutation @ mapping @ permutation


def sample_affine_zyx(source_zyx, output_to_source_zyx, output_shape_zyx, *,
                      order=1, output_dtype=np.float32, background=0.0):
    """Pull sample an explicit ZYX mapping; no forced copy of the source array.

    Order 0/1 needs no full-volume prefilter. Callers bound output_shape and
    control cancellation between blocks; order >1 may allocate a prefilter.
    """
    from scipy import ndimage

    mapping = affine_matrix4(output_to_source_zyx, "Output-to-source ZYX mapping")
    return ndimage.affine_transform(
        source_zyx, matrix=mapping[:3, :3], offset=mapping[:3, 3],
        output_shape=tuple(int(v) for v in output_shape_zyx), output=output_dtype,
        order=int(order), mode="constant", cval=float(background), prefilter=(int(order) > 1),
    )
