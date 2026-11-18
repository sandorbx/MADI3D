"""Native Janelia candidate-gradient production; no GUI, search flow or publication.

Candidate gradient production translates EM_MIP_search
83ee6f73cee0708f7b013c66d9dbbe11a1b3e7b2 (Copyright 2023 Howard Hughes
Medical Institute, BSD-3-Clause; LICENSES/EM_MIP_search-LICENSE.txt).
The controlled producer environment is ImageJ 1.53v, equal RGB weights and a
normal LUT. Canvas validation establishes pixel coordinates, never anatomical
alignment.

``candidate_gradient`` produces one candidate's gradient on the same pinned
path the archive's own gradient files were made with. It is the fallback for a
supplied-gradient distribution whose candidate has no supplied gradient, and
the generator for a sparse distribution that publishes none. ``gradient_supply``
decides *when* it runs and where its result is cached; this module decides
only what the producer computes.
"""
import hashlib

import numpy as np
from scipy.ndimage import distance_transform_cdt

from .gradient_scorer import _circular_max_nonnegative
from .local_scorer import checkpoint
from .search_profiles import SearchProfile, search_profile

PRODUCER_REVISION = "83ee6f73cee0708f7b013c66d9dbbe11a1b3e7b2"
IMAGEJ_REVISION = "72c46b115c91cce887793141cb43a8ec31c82f5d"
IMPLEMENTATION = "madi3d-lm-derivatives/1"
# The producer half of the pinned gradient's identity. A generated gradient is
# only interchangeable with a supplied one while this string is unchanged, so it
# is part of the derived-gradient cache key; changing the producer invalidates
# every cached derivative rather than silently mixing two gradient meanings.
GRADIENT_FORMAT = "janelia-threfunction-uint16-four-connected-v1"
# Gradient_conv_multithread.java L66/L71/L176: the "Max radius" stored Prefs
# default, which is what the archive's gradient files were produced with.
PRODUCER_MAXIMUM_RADIUS = 20
# Gradient_conv_multithread.java L161-173 erases the two generator annotation
# boxes with ``set(..., -16777216)``, which lands as 0 on the ByteProcessor. The
# heights deliberately differ: the right box is 100 rows, the left box 105.
ANNOTATION_RIGHT_BOX = (100, 270)
ANNOTATION_LEFT_BOX = (105, 330)


def gradient_pixel_sha256(gradient) -> str:
    """Hash decoded 2-D uint16 pixels in little-endian row-major order."""
    if (not isinstance(gradient, np.ndarray) or gradient.dtype.type is not np.uint16
            or gradient.ndim != 2):
        raise ValueError("Gradient pixels must be a 2-D uint16 array.")
    return hashlib.sha256(np.ascontiguousarray(gradient, dtype="<u2").tobytes(order="C")).hexdigest()


def _gray8(rgb, cancel):
    """ImageJ color-to-byte rounding, retaining double expression order.

    Process rows in bounded strips: the RGB input is borrowed, not copied. Unlike
    integer averaging, this preserves rounding at every actual byte combination.
    """
    gray = np.empty(rgb.shape[:2], np.uint8)
    for start in range(0, len(rgb), 32):
        checkpoint(cancel)
        block = rgb[start:start + 32]
        r, g, b = np.moveaxis(block, -1, 0)
        gray[start:start + 32] = r * (1 / 3.) + g * (1 / 3.) + b * (1 / 3.) + .5
    return gray


def _distance(signal, cancel):
    """Four-connected layer propagation, with the producer's empty sentinel.

    Supported canvases have maximum path length below 65535. The Java sentinel
    therefore cannot collide with a distance layer. SciPy runs bounded native
    passes; cancellation is checked immediately before and after them.
    """
    checkpoint(cancel)
    if not np.any(signal):
        return np.full(signal.shape, 65535, np.uint16)
    background = signal == 0
    distance = distance_transform_cdt(background, metric="taxicab")
    checkpoint(cancel)
    return distance.astype(np.uint16)


def _clear_producer_annotations(gray):
    """Erase the two generator annotation boxes (Gradient_conv_multithread L161-173)."""
    height, width = gray.shape
    rows, columns = ANNOTATION_RIGHT_BOX
    gray[:min(rows, height), max(0, width - columns):width] = 0
    rows, columns = ANNOTATION_LEFT_BOX
    gray[:min(rows, height), 0:min(columns, width)] = 0
    return gray


def _produce_gradient(rgb, *, radius, cancel, clear_annotations):
    """The pinned producer path.

    ``Gradient_conv_multithread.java`` L147-192: RGB to Gray8 with ImageJ's
    nearest-integer mean, the generator annotation clear, the rank maximum,
    zero-extension to 16-bit, the strict ``< 1`` binary inversion and
    four-connected propagation. The borrowed RGB input is never modified.
    """
    gray = _gray8(rgb, cancel)
    if clear_annotations:
        _clear_producer_annotations(gray)
    signal = _circular_max_nonnegative(gray, radius, cancel)
    del gray
    gradient = _distance(signal, cancel)
    del signal
    return gradient


def candidate_gradient(canvas_rgb, *, profile, cancel=None):
    """Produce one candidate's uint16 gradient on the archive producer's path.

    This is the pinned production the archive's own gradient PNGs were made
    with. The generator annotation boxes are always cleared, because the
    producer erases them for every file it converts (L161-173) and a supplied
    gradient therefore already has them erased.

    A caller supplies already-decoded candidate pixels: the exact dense
    reconstruction of a sparse span, or a decoded local TIFF. Choosing between
    a supplied gradient and this production, and caching the result, belongs to
    ``gradient_supply``; the shortlist decision belongs to the search flow.
    """
    if not isinstance(profile, SearchProfile) or profile != search_profile(profile.profile_id):
        raise ValueError("A pinned NeuronBridge search profile is required.")
    if (not isinstance(canvas_rgb, np.ndarray) or canvas_rgb.dtype != np.uint8
            or canvas_rgb.shape != (*profile.search_canvas_yx, 3)):
        raise ValueError("Candidate gradient production requires uint8 RGB on the declared full canvas.")
    checkpoint(cancel)
    gradient = _produce_gradient(canvas_rgb, radius=PRODUCER_MAXIMUM_RADIUS, cancel=cancel,
                                 clear_annotations=True)
    checkpoint(cancel)
    gradient.setflags(write=False)
    return gradient
