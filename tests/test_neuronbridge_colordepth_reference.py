"""Offline integrity and discriminating assertions for the independent NB-02 oracle.

No Java, network, application generator, Qt, or native CDM implementation runs here.
"""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/neuronbridge_colordepth"
REFERENCE = ROOT / "tests/reference/neuronbridge_colordepth"
SPEC = importlib.util.spec_from_file_location("nb02_reference", REFERENCE / "reproduce.py")
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
CASES = json.loads((FIXTURES / "cases.json").read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rgb(name):
    with Image.open(FIXTURES / "expected" / (name + ".png")) as image:
        assert image.mode == "RGB"
        return np.asarray(image).copy()


def test_corpus_and_reference_identity():
    ORACLE.verify_manifest_identity(MANIFEST)
    assert MANIFEST["cases_sha256"] == sha((FIXTURES / "cases.json").read_bytes())
    assert MANIFEST["reference_lock_sha256"] == sha((FIXTURES / "reference-lock.json").read_bytes())
    for name, key in [("reproduce.py", "runner_sha256"), ("RunReference.java", "launcher_sha256")]:
        assert MANIFEST[key] == sha((REFERENCE / name).read_bytes().replace(b"\r\n", b"\n"))
    assert set(MANIFEST["records"]) == {case["name"] for case in CASES}
    assert {p.stem for p in (FIXTURES / "expected").glob("*.png")} == set(MANIFEST["records"])


@pytest.mark.parametrize("key", ["cases_sha256", "reference_lock_sha256",
                                 "runner_sha256", "launcher_sha256"])
def test_changed_reference_identity_is_rejected(key):
    altered = dict(MANIFEST, **{key: "0" * 64})
    with pytest.raises(AssertionError, match="Reference identity changed"):
        ORACLE.verify_manifest_identity(altered)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_reference_output_and_input_hashes(case):
    record = MANIFEST["records"][case["name"]]
    data = ORACLE.read_input(case)
    source_hash = None if case["kind"] == "binary_mask" else sha(data.tobytes())
    assert record["source_raw_sha256"] == source_hash
    if "mask_zyx" in case:
        mask = np.zeros(data.shape, dtype=np.uint8)
        for z, y, x in case["mask_zyx"]:
            mask[z, y, x] = 1
        assert record["mask_raw_sha256"] == sha(mask.tobytes())
        if case["kind"] == "binary_mask":
            data = mask * np.uint8(255)
    data = ORACLE.place_on_canvas(data, case)
    assert record["input_raw_sha256"] == sha(data.tobytes())
    actual = rgb(case["name"])
    assert list(actual.shape) == record["output_shape"]
    assert sha(actual.tobytes()) == record["output_rgb_sha256"]
    assert sha((FIXTURES / "expected" / (case["name"] + ".png")).read_bytes()) == record["output_file_sha256"]
    assert int(np.any(actual != 0, axis=-1).sum()) == record["foreground_pixels"]


def test_depth_endpoints_low_values_and_intermediate_reference_samples():
    image = rgb("all_depths_and_low_values")
    # Observations from upstream execution, not expected values from a new encoder.
    for z, expected in [(0, [127, 0, 255]), (58, [0, 254, 255]),
                        (87, [129, 255, 126]), (173, [255, 3, 199])]:
        assert image[0, z].tolist() == expected
    assert not image[3:5].any()  # 0 and 1 are both discarded by Z_Code_Stack_HO.
    assert np.all(np.any(image[2] != 0, axis=-1))  # 2 survives at all depths.


def test_overlap_is_neither_independent_rgb_max_nor_winner_voxel():
    image = rgb("overlaps_and_ties")
    # z=0 at 128, z=116 at 200: preserve two components of the brighter colour,
    # allow the older blue component into its third channel.
    assert image[4, 4].tolist() == [199, 200, 128]
    assert image[4, 4].tolist() != [199, 200, 0]
    # Both at 255: earlier depth wins, even though independent maxima differ.
    assert image[0, 4].tolist() == [127, 0, 255]
    assert image[0, 4].tolist() != [254, 255, 255]


def test_binary64_truncation_is_not_integer_multiply_divide():
    # Independently observed in the exhaustive original-plugin sweep: LUT green
    # is 102 at z=23, but trunc((155/255.0)*102) is 61, not integer result 62.
    assert rgb("intensity_depth_matrix")[155, 23, 1] == 61
    assert (155 * 102) // 255 == 62


def test_crop_placement_and_reversed_storage_equivalence():
    crop = rgb("crop_global")
    assert crop.shape == (566, 1210, 3)
    for name in ("crop_full_equivalent", "crop_reversed_storage"):
        np.testing.assert_array_equal(crop, rgb(name))
    assert np.argwhere(np.any(crop, axis=-1)).tolist() == [[503, 1001], [504, 1003], [506, 1006]]
    signal = rgb("crop_signal_mask")
    binary = rgb("crop_binary_mask")
    np.testing.assert_array_equal(np.any(signal, axis=-1), np.any(binary, axis=-1))
    assert signal[506, 1006].max() == 128
    assert binary[506, 1006].max() == 255


def test_threshold_and_binary_mask_have_different_meanings():
    assert np.argwhere(np.any(rgb("signal_plus_mask"), axis=-1)).tolist() == [[0, 5], [0, 6]]
    assert np.argwhere(np.any(rgb("binary_mask"), axis=-1)).tolist() == [[0, x] for x in range(7)]
    assert not rgb("empty").any()


def test_nonzero_display_endpoint_conversion():
    # Original ImageJ/plugin observations: values <= 1023 disappear with
    # display range [1000,5095]; 1024 becomes intensity 2, not intensity 64.
    image = rgb("uint16_nonzero_lower_endpoint")
    assert not image[0, :7].any()
    assert image[0, 7].tolist() == [1, 2, 0]
    assert image[0, 9].tolist() == [64, 128, 63]
    assert image[0, 10:13].tolist() == [[129, 255, 126]] * 3
    assert not image[1].any()


def test_threshold_upper_limits_preserve_amplitude_or_erase_everything():
    image = rgb("threshold_254")
    assert np.argwhere(np.any(image, axis=-1)).tolist() == [[0, 2]]
    assert image[0, 2].tolist() == [129, 255, 126]
    assert not rgb("threshold_255").any()


def test_reversed_option_moves_x_and_z_but_not_y():
    image = rgb("upstream_reversed")
    assert image[0, 6].tolist() == [255, 3, 199]
    assert image[3, 0].tolist() == [127, 0, 255]
    assert np.argwhere(np.any(image, axis=-1)).tolist() == [[0, 6], [1, 4], [2, 1], [3, 0]]


@pytest.mark.parametrize("damage", ["colour", "false_positive", "false_negative"])
def test_foreground_gate_cannot_be_diluted_by_black_canvas(damage):
    expected = rgb("canvas_corners")
    actual = expected.copy()
    if damage == "colour":
        actual[0, 0, 0] -= 1
    elif damage == "false_positive":
        actual[300, 600] = [1, 0, 0]
    else:
        actual[0, 0] = 0
    with pytest.raises(AssertionError, match="foreground"):
        ORACLE.assert_parity(actual, expected)


def test_real_data_provenance_and_template_evidence():
    metadata = json.loads((FIXTURES / "template-provenance.json").read_text())
    assert sha((FIXTURES / "template-header.txt").read_bytes()) == metadata["header_sha256"]
    raw = (FIXTURES / "real_template_crop.u16le.gz").read_bytes()
    assert sha(raw) == metadata["crop"]["file_sha256"]
    assert sha(gzip.decompress(raw)) == metadata["crop"]["sha256"]
    assert metadata["license"] == "CC-BY-4.0"
