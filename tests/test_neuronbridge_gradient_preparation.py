"""Native candidate-gradient production against executed Java producer evidence.

The retired LM-derivative preparation (label policies, Z-gap overlay) was
replaced by the two-stage Janelia shape search (NB-GRAD-06C); the shape
search's shortlist stage now consumes ``candidate_gradient`` through the 06B
supply. What remains pinned here is the producer itself: the manifest of
original Java/ImageJ executions that establishes the pinned revision, and the
exact per-pixel equality of ``candidate_gradient`` with the reference
producer's own gradient outputs on the ``reference-labels`` (annotation
cleared) rows. The single ``no-labels`` manifest row characterized the
retired label-policy variant the producer no longer offers.
"""
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import pytest

from madi3d_app.integrations.neuronbridge import gradient_preparation as gp
from madi3d_app.integrations.neuronbridge.local_scorer import SearchCancelled
from madi3d_app.integrations.neuronbridge.search_profiles import BRAIN, VNC

FIXTURES = Path(__file__).parent / "fixtures/neuronbridge_gradient"
REPOSITORY = FIXTURES.parents[2]
PRODUCER = json.loads((FIXTURES / "producer_manifest.json").read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_rgb(path):
    with Image.open(path) as image:
        assert image.mode == "RGB" and image.n_frames == 1
        return np.asarray(image).copy()


def load_u16(path):
    with Image.open(path) as image:
        assert image.n_frames == 1
        pixels = np.asarray(image)
        assert pixels.dtype.kind == "u" and pixels.dtype.itemsize == 2
        return pixels.astype(np.uint16, copy=True)


def profile_for(row):
    return {"Brain": BRAIN, "VNC": VNC}[row["profile"]]


def fixture_path(path):
    return REPOSITORY / path


def load_input(row):
    input_info = row["input"]
    if row["kind"] != "synthetic":
        path = fixture_path(input_info["path"])
        return load_rgb(path), path
    pixels = np.empty(tuple(row["input_shape"]), dtype=np.uint8)
    recipe = row["recipe"]
    pixels[...] = recipe.get("fill_rgb", recipe.get("background_rgb"))
    for point in recipe.get("points", []):
        x, y = point["xy"]
        pixels[y, x] = point["rgb"]
    return pixels, None


def test_producer_oracle_manifest_pins_executed_sources_and_settings():
    assert PRODUCER["evidence"] == "observed original Java/ImageJ execution"
    assert PRODUCER["producer"]["revision"] == gp.PRODUCER_REVISION
    assert PRODUCER["imagej"]["source_revision"] == gp.IMAGEJ_REVISION
    # The manifest's scoring harness belongs to the retired pair scorer; the
    # shape search's own reference commit is pinned by gradient_scorer and is
    # intentionally not cross-equal here.
    assert PRODUCER["scorer"]["revision"] == "8b85c458f3b9231ec7bff5f1b4262cbf2abf8c3d"
    assert PRODUCER["producer"]["settings"]["radius"] == 20
    assert PRODUCER["scorer"]["original_junit_failures"] == 0
    assert PRODUCER["scorer"]["original_junit_errors"] == 0
    manifest_bytes = (FIXTURES / "manifest.json").read_bytes()
    assert sha(manifest_bytes) == (
        PRODUCER["scorer"]["reference_manifest_sha256"]
    )
    reference = json.loads(manifest_bytes)
    maven_version = reference["execution"]["maven_version"]
    assert "Maven home:" not in maven_version and ", runtime: " not in maven_version
    assert PRODUCER["execution"]["scorer_test_command"][0] in {"mvnw", "mvnw.cmd"}
    observations = FIXTURES / "observations.json"
    assert sha(observations.read_bytes()) == reference["file_sha256"][
        observations.relative_to(REPOSITORY).as_posix()
    ]
    source_hashes = [
        *PRODUCER["producer"]["source_sha256"].values(),
        PRODUCER["imagej"]["jar_sha256"],
        *PRODUCER["scorer"]["reference_harness_sha256"].values(),
    ]
    assert all(len(value) == 64 and all(c in "0123456789abcdef" for c in value)
               for value in source_hashes)
    for name, expected in PRODUCER["execution"]["runner_sha256"].items():
        path = fixture_path(name)
        data = path.read_bytes()
        if path.suffix in (".java", ".py", ".txt"):
            data = data.replace(b"\r\n", b"\n")
        assert sha(data) == expected


def test_java_producer_synthetic_rows_match_exact_candidate_gradient_pixels():
    rows = [row for row in PRODUCER["outputs"]
            if row["label_policy"] == "reference-labels" and row["kind"] == "synthetic"]
    assert len(rows) == 7
    assert {row["profile"] for row in rows} == {"Brain", "VNC"}
    assert_java_producer_pixels(rows)


def test_java_producer_real_rows_match_exact_candidate_gradient_pixels():
    rows = [row for row in PRODUCER["outputs"]
            if row["label_policy"] == "reference-labels" and row["kind"] != "synthetic"]
    assert len(rows) == 6
    assert {(row["profile"], row["kind"]) for row in rows} == {
        ("Brain", "real-lm"), ("VNC", "real-vnc-manc"),
    }
    assert_java_producer_pixels(rows)


def assert_java_producer_pixels(rows):
    """Compare decoded outputs with Java-retained pixels, never Python goldens."""
    for row in rows:
        profile = profile_for(row)
        lm, input_path = load_input(row)
        original = lm.copy()
        if input_path is not None:
            assert sha(input_path.read_bytes()) == row["input"]["file_sha256"]
        else:
            assert row["input_format"] == "RGB uint8; ImageJ ColorProcessor saved as TIFF"
        assert list(lm.shape) == (row["input_shape"] if row["kind"] == "synthetic"
                                  else row["input"]["shape"])
        assert sha(lm.tobytes()) == row["input"]["decoded_sha256"]
        output_path = fixture_path(row["path"])
        assert sha(output_path.read_bytes()) == row["file_sha256"]
        expected = load_u16(output_path)
        assert list(expected.shape) == row["shape"]
        assert sha(expected.astype("<u2", copy=False).tobytes()) == row["decoded_sha256"]

        gradient = gp.candidate_gradient(lm, profile=profile)
        assert gradient.dtype == np.uint16
        assert gradient.shape == profile.search_canvas_yx
        np.testing.assert_array_equal(gradient, expected)
        assert not gradient.flags.writeable
        assert row["radius"] == gp.PRODUCER_MAXIMUM_RADIUS
        assert np.array_equal(lm, original)  # borrowed pixels are never modified


def test_vnc_extra_clear_region_carries_no_retained_manc_signal():
    """The producer's fixed annotation boxes clear more than the VNC positive
    CDS excludes: the difference is 4,080 physical-content pixels. A retained
    MANC candidate with real signal there would mean the shape producer deletes
    pixels the positive stage considers scoreable for a purely brain-origin
    geometry reason — a Tier-1 escalation, never something to tune away."""
    height, width = VNC.search_canvas_yx
    producer = np.zeros((height, width), bool)
    rows, columns = gp.ANNOTATION_RIGHT_BOX
    producer[:rows, width - columns:] = True
    rows, columns = gp.ANNOTATION_LEFT_BOX
    producer[:rows, :columns] = True
    positive = np.zeros((height, width), bool)
    for anchor, x, y, excluded_w, excluded_h in VNC.score_exclusions:
        left = x if anchor == "left" else width - x - excluded_w
        positive[y:y + excluded_h, left:left + excluded_w] = True
    extra_clear = producer & ~positive
    assert int(np.count_nonzero(extra_clear)) == 4080
    geometry = PRODUCER["vnc_annotation_clear"]
    assert geometry["canvas_yx"] == [height, width]
    assert geometry["producer_clear_pixels"] == int(np.count_nonzero(producer))
    assert geometry["positive_exclusion_pixels"] == int(np.count_nonzero(positive))
    assert geometry["extra_clear_pixels"] == int(np.count_nonzero(extra_clear))

    rows = [row for row in PRODUCER["outputs"] if row["kind"] == "real-vnc-manc"]
    assert len(rows) == 3
    for row in rows:
        name = Path(row["input"]["path"]).name
        input_path = fixture_path(row["input"]["path"])
        assert sha(input_path.read_bytes()) == row["input"]["file_sha256"]
        lm, _ = load_input(row)
        assert sha(lm.tobytes()) == row["input"]["decoded_sha256"]
        foreground = np.any(lm != 0, axis=2)
        measured = row["vnc_extra_clear_signal"]
        assert measured["foreground_pixels"] == int(np.count_nonzero(foreground))
        assert measured["nonzero_pixels_in_extra_clear"] == int(
            np.count_nonzero(foreground & extra_clear))
        assert measured["rgb_sum_in_extra_clear"] == int(lm[extra_clear].sum())
        assert measured["rgb_maximum_channel_in_extra_clear"] == int(lm[extra_clear].max())
        assert measured["extra_clear_foreground_fraction"] == float(
            measured["nonzero_pixels_in_extra_clear"]) / measured["foreground_pixels"]
        # No retained real candidate signal may live inside the extra-clear
        # region; any nonzero count here is the escalation condition itself.
        assert measured["nonzero_pixels_in_extra_clear"] == 0
        assert measured["rgb_maximum_channel_in_extra_clear"] == 0


def test_empty_and_full_signals_have_defined_uint16_extrema():
    for profile in (BRAIN, VNC):
        empty = np.zeros((*profile.search_canvas_yx, 3), dtype=np.uint8)
        gradient = gp.candidate_gradient(empty, profile=profile)
        assert gradient.dtype == np.uint16
        assert np.all(gradient == 65535)

        # An all-signal canvas still carries the producer's annotation clear:
        # the boxes hold no signal, and the radius-20 disk reaches 20 rows up
        # from the first surviving row, so the hole depths are exact.
        full = np.full_like(empty, 255)
        gradient = gp.candidate_gradient(full, profile=profile)
        h, w = profile.search_canvas_yx
        assert np.all(gradient[gp.ANNOTATION_LEFT_BOX[0]:, :] == 0)
        assert int(gradient[0, 0]) == gp.ANNOTATION_LEFT_BOX[0] - 20
        assert int(gradient[0, w - 1]) == gp.ANNOTATION_RIGHT_BOX[0] - 20


@pytest.mark.parametrize("profile", [BRAIN, VNC], ids=lambda p: p.anatomical_area.lower())
def test_sparse_signal_reaches_supported_full_canvas_distance(profile):
    rgb = np.zeros((*profile.search_canvas_yx, 3), dtype=np.uint8)
    rgb[-1, -1] = 255  # opposite corner from the annotation boxes
    gradient = gp.candidate_gradient(rgb, profile=profile)
    h, w = profile.search_canvas_yx
    # Radius-20's exact ImageJ disk reaches a maximum Manhattan offset of 28;
    # the far corner is the longest four-connected propagation path.
    assert int(gradient[0, 0]) == h + w - 2 - 28
    assert int(gradient.max()) < 65535


def test_producer_annotation_boxes_are_always_erased():
    """Gradient_conv_multithread L161-173 erases both boxes for every file,
    so pixels that live only inside a box never seed a gradient."""
    rgb = np.zeros((*BRAIN.search_canvas_yx, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 40, 10)     # inside the left box (105 rows x 330 columns)
    rgb[0, -1] = (10, 40, 255)    # inside the right box (100 rows x 270 columns)
    original = rgb.copy()
    gradient = gp.candidate_gradient(rgb, profile=BRAIN)
    assert np.all(gradient == 65535)
    np.testing.assert_array_equal(rgb, original)
    # Signals just outside each box's row extent survive and seed normally.
    surviving = np.zeros((*BRAIN.search_canvas_yx, 3), dtype=np.uint8)
    surviving[gp.ANNOTATION_LEFT_BOX[0], gp.ANNOTATION_LEFT_BOX[1]] = (255, 40, 10)
    surviving[gp.ANNOTATION_RIGHT_BOX[0], BRAIN.search_canvas_yx[1] - gp.ANNOTATION_RIGHT_BOX[1] - 1] = (10, 40, 255)
    gradient = gp.candidate_gradient(surviving, profile=BRAIN)
    assert gradient[gp.ANNOTATION_LEFT_BOX[0], gp.ANNOTATION_LEFT_BOX[1]] == 0
    assert gradient[gp.ANNOTATION_RIGHT_BOX[0],
                    BRAIN.search_canvas_yx[1] - gp.ANNOTATION_RIGHT_BOX[1] - 1] == 0


@pytest.mark.parametrize("bad", [
    np.zeros((566, 1210, 3), dtype=np.uint16),
    np.zeros((566, 1210, 4), dtype=np.uint8),
    np.zeros((565, 1210, 3), dtype=np.uint8),
    np.zeros((566, 1210), dtype=np.uint8),
])
def test_invalid_canvas_or_encoding_rejected(bad):
    with pytest.raises(ValueError):
        gp.candidate_gradient(bad, profile=BRAIN)


def test_pinned_search_profile_is_required():
    valid = np.zeros((*BRAIN.search_canvas_yx, 3), dtype=np.uint8)
    gradient = gp.candidate_gradient(valid, profile=BRAIN)
    assert gradient.shape == BRAIN.search_canvas_yx
    with pytest.raises(ValueError):
        gp.candidate_gradient(valid, profile="Brain")
    with pytest.raises(ValueError):
        gp.candidate_gradient(valid, profile=None)


def test_cancellation_before_and_during_production():
    rgb = np.zeros((*BRAIN.search_canvas_yx, 3), dtype=np.uint8)
    rgb[200, 600] = (255, 80, 20)
    original = rgb.copy()
    already_cancelled = threading.Event()
    already_cancelled.set()
    with pytest.raises(SearchCancelled):
        gp.candidate_gradient(rgb, profile=BRAIN, cancel=already_cancelled)

    class CancelAfterStart:
        started = False

        def is_set(self):
            # Cancellation arrives after the initial poll, independent of
            # machine speed or the number/location of later checkpoints.
            cancelled = self.started
            self.started = True
            return cancelled

    with pytest.raises(SearchCancelled):
        gp.candidate_gradient(rgb, profile=BRAIN, cancel=CancelAfterStart())
    np.testing.assert_array_equal(rgb, original)
