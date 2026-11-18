import hashlib
import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from madi3d_app.integrations.neuronbridge.local_scorer import (
    PreparedQuery, SearchCancelled, SearchParameters, pixel_gaps, score, shifts,
)
from madi3d_app.integrations.neuronbridge.library_store import read_sparse, sparse_pixels

FOLDER = Path(__file__).parent / "fixtures" / "neuronbridge_local"
ORACLE = json.loads((FOLDER / "oracle.json").read_text(encoding="utf-8"))


def canvas(case, key):
    rgb = np.zeros((*case["shape"], 3), np.uint8)
    for x, y, r, g, b in case[key]:
        rgb[y, x] = (r, g, b)
    return rgb


def test_pinned_source_and_independent_pixel_gaps():
    assert hashlib.sha256((FOLDER / "mipsearch.mjs").read_bytes()).hexdigest() == ORACLE["source_sha256"]
    pairs = ORACLE["gaps"]
    actual = pixel_gaps(np.array([p["a"] for p in pairs], np.uint8), np.array([p["b"] for p in pairs], np.uint8))
    # Both implementations use IEEE-754 doubles; allow one final-addition ulp.
    np.testing.assert_allclose(actual, [p["gap"] for p in pairs], rtol=0, atol=1e-15)


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda c: c["name"])
def test_reference_positive_scorer_and_lossless_index(case):
    params = SearchParameters(**case["parameters"])
    query, target = canvas(case, "query"), canvas(case, "target")
    before = query.copy(), target.copy()
    if not case["foreground"]:
        with pytest.raises(ValueError, match="no foreground"):
            score(query, target, params)
        return
    prepared = PreparedQuery(query, params)
    result = prepared.score(target)
    expected = case["expected"]
    assert result.matching_pixels == expected["matchingPixNum"]
    assert result.overlap_fraction == pytest.approx(expected["matchingPixNumToMaskRatio"], abs=1e-15, rel=0)
    assert result.mirrored == expected["bestScoreMirrored"]
    assert result.query_foreground_pixels == case["foreground"]
    sparse = sparse_pixels(target)
    assert prepared.score_pixels(lambda positions: read_sparse(sparse, positions)) == result
    assert prepared.score_sparse(sparse["position"], sparse["rgb"]) == result
    np.testing.assert_array_equal(query, before[0])
    np.testing.assert_array_equal(target, before[1])
    if case["name"] == "shift ordered tie":
        assert result.shift == (-2, -2)
    upstream_offsets = (tuple((x, y) for radius in range(2, params.xy_shift + 1, 2)
        for x in (-radius, 0, radius) for y in (-radius, 0, radius))
        if params.xy_shift else ((0, 0),))
    masks = {}
    for offset, expected_mask in zip(upstream_offsets, case["expected_target_masks"], strict=True):
        masks.setdefault(offset, expected_mask)
    assert shifts(params.xy_shift) == tuple(masks)
    for (dx, dy), expected_mask in masks.items():
        width, height = case["shape"][1], case["shape"][0]
        x, y = prepared.positions % width + dx, prepared.positions // width + dy
        actual = np.where((x >= 0) & (x < width) & (y >= 0) & (y < height), y * width + x, -1)
        np.testing.assert_array_equal(actual, expected_mask)


def test_largest_shift_scores_each_mirror_offset_once():
    case = ORACLE["cases"][0]
    prepared = PreparedQuery(canvas(case, "query"), SearchParameters(xy_shift=20, mirror=True))
    reads = []
    target = canvas(case, "target").reshape(-1, 3)
    def read(positions):
        reads.append(len(positions))
        return target[positions]
    result = prepared.score_pixels(read)
    assert len(shifts(20)) == len(set(shifts(20))) == 81
    assert len(reads) == 162  # One small foreground block per distinct offset/mirror.
    assert result == prepared.score(target.reshape((*case["shape"], 3)))


def test_current_madi3d_generator_matches_oracle_input():
    from generate_neuronbridge_local_oracle import generated_query
    case = next(c for c in ORACLE["cases"] if c["name"] == "MADI3D generated query")
    np.testing.assert_array_equal(generated_query().rgb, canvas(case, "query"))


@pytest.mark.parametrize("changes", [
    {"query_threshold": -1}, {"query_threshold": True}, {"candidate_threshold": 256},
    {"candidate_threshold": 1.5}, {"xy_shift": 1}, {"xy_shift": -2}, {"xy_shift": 22},
    {"color_tolerance_percent": float("nan")}, {"minimum_overlap": float("inf")},
    {"minimum_overlap": -1}, {"mirror": 1}, {"result_limit": 0}, {"memory_mb": 32},
])
def test_invalid_parameters_rejected(changes):
    with pytest.raises(ValueError):
        SearchParameters(**changes)


def test_malformed_images_and_real_cancellation():
    case = ORACLE["cases"][0]
    q = canvas(case, "query")
    for invalid in (q.astype(float), q[:, :, :2], q[:, :, 0], np.zeros((1000, 1000, 3), np.uint8)):
        with pytest.raises(ValueError):
            score(invalid, q)
    with pytest.raises(ValueError, match="canvases"):
        score(q, q[:-1])
    cancel = Event()
    def read(positions):
        cancel.set()
        return np.zeros((len(positions), 3), np.uint8)
    with pytest.raises(SearchCancelled):
        PreparedQuery(q, SearchParameters(xy_shift=2)).score_pixels(read, cancel)


@pytest.mark.parametrize("profile_id", ["nb02-prepared-signal-v1", "nb-vnc-prepared-signal-v1"])
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("xy_shift", [0, 2, 20])
@pytest.mark.parametrize("candidate_threshold", [0, 20])
def test_sparse_candidate_first_exact_profile_parity(profile_id, mirror, xy_shift, candidate_threshold):
    from madi3d_app.integrations.neuronbridge.search_profiles import search_profile
    profile = search_profile(profile_id)
    height, width = profile.search_canvas_yx
    query = np.zeros((height, width, 3), np.uint8)
    # Edges, both label exclusions, and central pixels; ties and clipped shifts.
    for x, y in ((0, 200), (width - 1, 200), (100, 0), (100, height - 1),
                 (200, 200), (202, 202), (10, 10), (width - 10, 10)):
        query[y, x] = (230, 90, 5)
    parameters = SearchParameters(mirror=mirror, xy_shift=xy_shift, candidate_threshold=candidate_threshold)
    prepared = PreparedQuery(query, parameters, profile_id=profile_id)
    for candidate in (query, query[:, ::-1].copy(), np.roll(query, 2, axis=0),
                      query // 16, np.zeros_like(query)):
        pixels = sparse_pixels(candidate)
        assert prepared.score_sparse(pixels["position"], pixels["rgb"]) == prepared.score(candidate)


def test_sparse_blocks_and_shared_prepared_query_are_read_only():
    from concurrent.futures import ThreadPoolExecutor
    from madi3d_app.integrations.neuronbridge.local_scorer import PIXEL_BLOCK
    rng = np.random.default_rng(740)
    query = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    prepared = PreparedQuery(query, SearchParameters(xy_shift=2))
    pixels = sparse_pixels(query)
    assert len(pixels) > PIXEL_BLOCK
    arrays = (prepared.positions, prepared.sectors, prepared.ratios, prepared.query_index)
    before = [array.copy() for array in arrays]
    expected = prepared.score(query)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: prepared.score_sparse(pixels["position"], pixels["rgb"]), range(8)))
    assert results == [expected] * 8
    for actual, original in zip(arrays, before):
        assert not actual.flags.writeable
        np.testing.assert_array_equal(actual, original)
    cancel = Event()
    cancel.set()
    with pytest.raises(SearchCancelled):
        prepared.score_sparse(pixels["position"], pixels["rgb"], cancel)
