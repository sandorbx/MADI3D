"""Pinned scoring primitives against executed-Java observations (plus shape I/O).

The retired ``PreparedLM`` pair scorer was replaced by the two-stage Janelia
shape search (NB-GRAD-06C; parity lives in test_neuronbridge_gradient_search.py
against tests/oracles/janelia_em_mip_oracle.py). What remains pinned here is
what the shape search and the historical ``gradient_execution`` records still
execute: the Z decoder / gray-rounding probes, the fresh row-major disk filter,
the retired normalization contract, and cooperative cancellation.
"""
import json
from pathlib import Path
import threading

import numpy as np
import pytest
from scipy.ndimage import maximum_filter

from madi3d_app.integrations.neuronbridge import gradient_scorer as gs
from madi3d_app.integrations.neuronbridge.local_scorer import SearchCancelled

FIXTURES = Path(__file__).parent / "fixtures/neuronbridge_gradient"
OBSERVED = json.loads((FIXTURES / "observations.json").read_text())
PROBES = OBSERVED["probes"]


def rgb(packed):
    p = np.asarray(packed, dtype=np.int64)
    return np.stack(((p >> 16) & 255, (p >> 8) & 255, p & 255), axis=-1).astype(np.uint8)


def test_depth_and_gray_scalar_oracles():
    pairs = PROBES["sliceGapColorPairs"]
    np.testing.assert_array_equal(gs.slice_gap(rgb([p["rgbA"] for p in pairs]), rgb([p["rgbB"] for p in pairs])),
                                  [p["sliceGap"] for p in pairs])
    channels = {"RED": 0, "GREEN": 1, "BLUE": 2}
    for p in PROBES["sectorLutProbes"]:
        if "sector" in p:
            # Sector-range midpoint probes all sit in the blue->green sector.
            assert gs.find_slice_number(2, 1, p["midpointRatio"]) == p["observedSliceNumber"]
        else:
            assert gs.find_slice_number(channels[p["maximumChannel"]], channels[p["secondChannel"]],
                                        p["ratio"]) == p["sliceNumber"]
    np.testing.assert_array_equal(gs.gray_no_gamma(rgb([p["rgb"] for p in PROBES["grayRounding"]])),
                                  [p["gray16NoGamma"] for p in PROBES["grayRounding"]])


@pytest.mark.parametrize("p", PROBES["unsafeMaxFilter"])
def test_all_fresh_filter_oracles(p):
    source = rgb(p["inputPixels"]).reshape(p["height"], p["width"], 3)
    original = source.copy()
    result = gs.circular_max(source, p["radius"], border_pixels=p["borderPixels"])
    expected = rgb(p["materializedPixels"]).reshape(source.shape)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result, rgb(p["freshPixels"]).reshape(source.shape))
    np.testing.assert_array_equal(result[:, ::-1], rgb(p["freshMirroredPixels"]).reshape(source.shape))
    np.testing.assert_array_equal(source, original)
    # The Java reused/mirrored lazy probes characterize a discarded iterator,
    # not a second algorithm path. Their integrity remains in reference tests.


@pytest.mark.parametrize("radius", [10, 20, 60])
def test_optimized_footprint_parity_at_edges_and_multichannel(radius):
    source = np.zeros((127, 131, 3), np.uint8)
    source[0, 0] = [255, 0, 0]
    source[-1, -1] = [0, 255, 0]
    source[63, 65] = [0, 0, 255]
    source[5, 64] = [127, 211, 37]
    radii = gs.line_radii(radius)
    k = len(radii) // 2
    footprint = np.zeros((2*k+1, 2*k+1, 1), dtype=bool)
    for y, dx in enumerate(radii):
        footprint[y, k-dx:k+dx+1] = True
    expected = maximum_filter(source, footprint=footprint, mode="constant", cval=0)
    np.testing.assert_array_equal(gs.circular_max(source, radius), expected)


def test_normalization_oracles_remain_pinned_for_historical_records():
    for p in PROBES["normalization"] + PROBES["upstreamNormalizationRows"]:
        score = gs.normalize_score(p["pixelMatch"], p["shapeScore"], p["maxPixelMatch"], p["maxShapeScore"])
        assert score.utility_float64 == pytest.approx(p["normalizedDouble"], rel=1e-14, abs=1e-12)
        if "storedFloat32Bits" in p:
            assert score.stored_float32.view(np.int32) == p["storedFloat32Bits"]
            assert float(score.stored_float32) == p["storedFloat32Exact"]
    cohort, collision = PROBES["normalizationCandidates"]
    for letter in ("A", "B"):
        score = gs.normalize_score(*cohort[f"candidate{letter}"], cohort["independentMaximumPixel"], cohort["independentMaximumShape"])
        assert score.utility_float64 == cohort[f"candidate{letter}Normalized"]
        score = gs.normalize_score(*collision[f"inputs{letter}"])
        assert score.utility_float64 == collision[f"utilityDouble{letter}"]
        assert score.stored_float32.view(np.uint32) == collision[f"storedFloatBits{letter}"]


@pytest.mark.parametrize("values", [(1, None, 1, 1), (1, float("nan"), 1, 1),
                                   (1., 0, 1, 1), (True, 0, 1, 1),
                                   (1, 0, 1, 1 << 63), (1 << 31, 0, 1, 1)])
def test_normalization_requires_encoded_integer_measurements_and_maxima(values):
    with pytest.raises(ValueError):
        gs.normalize_score(*values)


@pytest.mark.parametrize("kwargs", [dict(query_threshold=-1), dict(query_threshold=20.), dict(mirror=1),
                                    dict(border_pixels=2), dict(clear_name_label="no")])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        gs.GradientParameters(**kwargs)


def test_shape_scorer_input_contract():
    """The shape scorer accepts exactly aligned uint8 RGB / uint16 gradients."""
    query = np.zeros((12, 16, 3), np.uint8)
    query[4:8, 4:8] = (255, 0, 255)
    gradient = np.zeros((12, 16), np.uint16)
    scorer = gs.ShapeScorer(query)
    measurement = scorer.measure(query.copy(), gradient)
    assert measurement.sample_to_mask >= 0 and measurement.normal_value >= 0
    for bad_query in (query.astype(np.uint16), query[..., :2], query[0], query[:0]):
        with pytest.raises(ValueError):
            gs.ShapeScorer(bad_query)
    with pytest.raises(ValueError):
        gs.ShapeScorer(query, mirror="yes")
    with pytest.raises(ValueError):
        gs.ShapeScorer(query, mask_threshold=256)
    with pytest.raises(ValueError):
        gs.ShapeScorer(query, negative_radius=-1)
    with pytest.raises(ValueError):
        scorer.measure(np.zeros((13, 16, 3), np.uint8), gradient)
    with pytest.raises(ValueError):
        scorer.measure(query.copy(), gradient.astype(np.uint8))
    with pytest.raises(ValueError):
        scorer.measure(query.copy(), np.zeros((13, 16), np.uint16))


def test_query_reuse_is_immune_to_caller_mutation():
    query = np.zeros((12, 16, 3), np.uint8)
    query[2:6, 3:9] = (0, 255, 200)
    scorer = gs.ShapeScorer(query, mirror=True)
    query[8:10, 10:12] = (255, 120, 0)  # mutate the borrowed input after construction
    gradient = np.zeros((12, 16), np.uint16)
    hit = np.zeros((12, 16, 3), np.uint8)
    hit[2:6, 3:9] = (0, 255, 200)
    first = scorer.measure(hit.copy(), gradient)
    second = scorer.measure(hit.copy(), gradient)
    assert first == second


def test_cooperative_cancellation():
    query = np.full((12, 16, 3), 255, np.uint8)
    event = threading.Event()
    event.set()
    with pytest.raises(SearchCancelled):
        gs.circular_max(query, 60, event)
    with pytest.raises(SearchCancelled):
        gs.ShapeScorer(query, cancel=event)
    scorer = gs.ShapeScorer(query)
    with pytest.raises(SearchCancelled):
        scorer.measure(query.copy(), np.zeros((12, 16), np.uint16), event)

    class CancelAfterCheckpoints:
        def __init__(self, count):
            self.remaining = count
        def is_set(self):
            self.remaining -= 1
            return self.remaining <= 0

    with pytest.raises(SearchCancelled):
        gs.circular_max(query, 60, CancelAfterCheckpoints(4))
