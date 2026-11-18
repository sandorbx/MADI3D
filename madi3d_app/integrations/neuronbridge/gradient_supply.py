"""Supply candidate gradients only where an admitted shortlist needs them.

A gradient is not part of a candidate's published identity: it is a derived
raster. The official Hemibrain distribution ships one per candidate, so searching
there reads a supplied file. The sparse FAFB and MANC packages publish RGB spans
only, so a gradient has to be produced — and producing one for every candidate in
a 144,000-member library would be hours of work for pixels nobody scores. This
module therefore supplies gradients *after* shortlist admission, one admitted
candidate at a time, and keeps generated results in the disposable derived cache
owned by ``cache.py``.

Ownership is deliberately split and nothing here re-implements another owner:

* ``archive_catalog`` declares each distribution's gradient policy and names the
  supplied-gradient member for a candidate;
* ``library_store`` reads installed distribution bytes (candidate canvas, supplied
  gradient PNG) and reports installed identity;
* ``gradient_preparation`` owns the pinned producer that computes a gradient;
* ``cache`` owns the derived-gradient root, key derivation, encoding and pruning;
* this module owns only the supply decision — supplied file or generation, whether
  the cache answers, and whether the caller's already reconstructed candidate is
  used instead of a second reconstruction.

It never enumerates a whole library, never publishes a search result, and never
writes project scientific state. A shortlisted candidate is reconstructed at most
once per search: the shape stage already holds the canvas it scores, so it hands
that canvas in through ``candidate_canvas`` and generation uses it instead of
reading the same span again. A borrowed canvas is verified against the row's
retained pixel evidence, never mutated, and never retained.

Computation is synchronous on the calling thread. Callers outside the GUI thread
own scheduling, exactly as the local-search flow does for scoring.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .cache import derived_gradient_key
from .gradient_preparation import (GRADIENT_FORMAT, IMPLEMENTATION, PRODUCER_REVISION,
                                   candidate_gradient, gradient_pixel_sha256)
from .library_store import decode_gradient
from .local_scorer import checkpoint
from .search_profiles import search_profile

GRADIENT_SOURCE_SUPPLIED = "supplied-archive"
GRADIENT_SOURCE_GENERATED = "generated-producer"


@dataclass(frozen=True)
class CandidateGradient:
    """One shortlisted candidate's gradient and where its pixels came from."""

    member_key: str
    source: str
    gradient: np.ndarray
    pixel_sha256: str


def _candidate_identity(row) -> dict:
    """Candidate facts that determine one gradient inside one distribution."""
    identity = {"member_key": row["member_key"]}
    pixel_sha256 = row.get("pixel_sha256")
    if pixel_sha256:
        identity["pixel_sha256"] = pixel_sha256
    return identity


def _cache_key(reader, row) -> str:
    """Key one generated gradient by installed distribution, candidate and producer."""
    return derived_gradient_key(library=reader.identity, candidate=_candidate_identity(row),
                                canvas_yx=reader.canvas_yx, producer_revision=PRODUCER_REVISION,
                                implementation_revision=IMPLEMENTATION,
                                gradient_format=GRADIENT_FORMAT)


def _verify_borrowed(canvas, row) -> None:
    """Refuse a borrowed canvas that is not the row's own retained candidate pixels.

    The borrowed canvas is the only evidence the generation path has that it is
    deriving from the candidate the caller named, and the derived-cache key binds
    the row's digest. Without this check a mismatched canvas could publish another
    neuron's gradient under this row's key, so the check runs before any derivative
    is produced and a mismatch publishes nothing.
    """
    expected = row.get("pixel_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("A borrowed candidate canvas needs the row's retained pixel evidence to be verified.")
    digest = hashlib.sha256(np.ascontiguousarray(canvas).tobytes(order="C")).hexdigest()
    if digest != expected:
        raise ValueError("Borrowed candidate canvas does not match the retained candidate pixels.")


def _generate(reader, row, cache, cancel, candidate_canvas=None):
    """Produce one gradient, or reuse its cached entry; returns ``(pixels, hit)``.

    ``candidate_canvas`` is the caller's exact decoded/reconstructed candidate. When
    it is absent the reconstruction happens here, and the dense canvas is released
    as soon as the gradient exists.
    """
    key = _cache_key(reader, row) if cache is not None else None
    if cache is not None:
        cached = cache.load(key, canvas_yx=reader.canvas_yx)
        if cached is not None:
            return cached, True
    if candidate_canvas is None:
        canvas = reader.dense_canvas(row)
    else:
        _verify_borrowed(candidate_canvas, row)
        canvas = candidate_canvas
    try:
        gradient = candidate_gradient(canvas, profile=search_profile(reader.profile_id), cancel=cancel)
    finally:
        canvas = None
    if cache is not None:
        cache.publish(key, gradient)
    return gradient, False


def _supply(reader, row, cache, cancel, candidate_canvas=None):
    """The single supply route; returns ``(member_key, source, pixels, from_cache)``."""
    member_key = row.get("member_key") if isinstance(row, dict) else None
    if not isinstance(member_key, str):
        raise ValueError("Candidate row does not identify a library member.")
    checkpoint(cancel)
    location = reader.gradient_path(row)
    if location is not None:
        try:
            pixels = decode_gradient(location, search_profile(reader.profile_id))
            source = GRADIENT_SOURCE_SUPPLIED
        except (OSError, ValueError):
            # A supplied gradient that cannot be read is a missing gradient: the
            # candidate keeps its producer-generated equivalent instead of being
            # dropped, and its source says so.
            pixels = None
    else:
        pixels = None
    if pixels is None:
        pixels, from_cache = _generate(reader, row, cache, cancel, candidate_canvas)
        source = GRADIENT_SOURCE_GENERATED
    else:
        from_cache = False
    pixels.setflags(write=False)
    return member_key, source, pixels, from_cache


def supply_candidate_gradient(reader, row, *, candidate_canvas=None, cache=None,
                              cancel=None) -> CandidateGradient:
    """Return one admitted candidate's gradient, using its supplied file when valid.

    A distribution that supplies gradients is read from the archive PNG named for
    that candidate. A candidate whose supplied gradient is missing, undecodable or
    outside the declared pixel contract is not discarded: it is produced on the
    pinned producer and cached like any generated candidate, and its source is
    reported as ``generated-producer`` so the substitution stays visible.

    ``candidate_canvas`` is the caller's already reconstructed RGB candidate for
    exactly this row. Supplying it is what lets a shape search reconstruct a
    shortlisted candidate once and use that canvas both for generation and for
    measurement; a supplied archive gradient never consumes it. It is borrowed —
    not retained, not mutated — and it is verified against the row's retained
    ``pixel_sha256`` before a derivative is produced or published, so a mismatched
    canvas raises instead of poisoning the derived cache.

    ``cache`` is optional; without it a generated gradient is computed and returned
    but never stored, so the caller alone pays for the repeat work.
    """
    member_key, source, pixels, _hit = _supply(reader, row, cache, cancel, candidate_canvas)
    return CandidateGradient(member_key=member_key, source=source, gradient=pixels,
                             pixel_sha256=gradient_pixel_sha256(pixels))


__all__ = [
    "GRADIENT_SOURCE_GENERATED",
    "GRADIENT_SOURCE_SUPPLIED",
    "CandidateGradient",
    "supply_candidate_gradient",
]
