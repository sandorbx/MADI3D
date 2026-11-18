"""Native CDM acceptance against independently executed NB-02 plugin outputs."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from madi3d_app.integrations.neuronbridge.cdm import (
    CDMCancelled, array_checksum, generate_brain_cdm,
)
from madi3d_app.integrations.neuronbridge.cdm_export import export_brain_cdm
from madi3d_app.integrations.neuronbridge.cdm_lut import LUT_BYTES
from madi3d_app.integrations.neuronbridge.cdm_records import (
    CDMArtifact, CDMGeometry, CDMMapping, CDMParameters, CDMSelection, TEMPLATE_SHAPE,
)
from madi3d_app.integrations.neuronbridge.records import ChannelSelection, SourceIdentity
from madi3d_app.volume.geometry import VolumeWorkingGrid
from test_neuronbridge_colordepth_reference import CASES, ORACLE, rgb


SCALAR_CASES = [c for c in CASES if c["kind"] != "rgb_projection"]


def test_embedded_lut_matches_reference_bytes():
    assert hashlib.sha256(LUT_BYTES).hexdigest() == (
        "10bdeb45546b89d5c311c423975072ea11a5d32ea6eb18c2b950583605a7389f"
    )


def selection(name="source"):
    return CDMSelection(name, "revision-1", SourceIdentity(kind="lm", library="fixture", library_release="v1"),
                        ChannelSelection("signal", 0), 0, "ZYX")


def geometry(shape):
    grid = VolumeWorkingGrid(
        dimensions=shape[::-1], spacing=(0.5189161, 0.5189161, 1), origin=(0, 0, 0),
        direction=np.eye(3), physical_units=("um",) * 3,
        source_coordinate_space_id="fixture-index-space", coordinate_mode="working-grid",
        geometry_basis="partially-assumed", physical_grid_state="incomplete-geometry",
        assumed_fields=("origin",), warnings=("Fixture numerical origin is assumed.",),
    )
    return CDMGeometry(grid, "geometry-1", np.eye(4))


def arguments(data):
    return dict(selection=selection(), geometry=geometry(data.shape),
                mapping=CDMMapping(np.eye(4), "mapping-1"),
                parameters=CDMParameters(allow_exploratory=True))


def case_input(case):
    data = ORACLE.read_input(case)
    kwargs = arguments(data)
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag(case.get("index_step_xyz", [1, 1, 1]))
    matrix[:3, 3] = case.get("offset_xyz", [0, 0, 0])
    if case.get("upstream_reverse"):
        matrix = np.diag([-1., 1., -1., 1.])
        matrix[:3, 3] = (data.shape[2] - 1, 0, 173)
    kwargs["mapping"] = CDMMapping(matrix, "fixture-mapping")
    mode = "signal"
    if "mask_zyx" in case:
        mask = np.zeros(data.shape, np.uint8)
        for z, y, x in case["mask_zyx"]:
            mask[z, y, x] = 1
        kwargs.update(mask=mask, mask_selection=selection("mask"), mask_revision="mask-1")
        mode = "signal_mask"
        if case["kind"] == "binary_mask":
            mode = "binary_mask"
            kwargs["selection"] = kwargs["mask_selection"]
            data = None
    kwargs["parameters"] = CDMParameters(
        mode=mode, display_range=case.get("display_range"), threshold=case.get("threshold", 0),
        depth_range=tuple(v - 1 for v in case.get("projection_slices_1based", [1, 174])),
        allow_exploratory=True,
    )
    return data, kwargs


@pytest.mark.parametrize("case", SCALAR_CASES, ids=lambda c: c["name"])
def test_native_independent_golden_foreground_and_coordinates(case):
    data, kwargs = case_input(case)
    result = generate_brain_cdm(data, **kwargs)
    expected_patch = rgb(case["name"])
    expected = np.zeros((*TEMPLATE_SHAPE[1:], 3), dtype=np.uint8)
    expected[:expected_patch.shape[0], :expected_patch.shape[1]] = expected_patch
    ORACLE.assert_parity(result.rgb, expected)
    assert result.rgb.shape == (566, 1210, 3) and result.rgb.dtype == np.uint8
    assert result.artifact.rgb_pixel_sha256 == hashlib.sha256(expected.tobytes()).hexdigest()
    assert CDMArtifact.from_json(result.artifact.to_json()) == result.artifact
    assert result.resources.peak_tile_voxels <= 32 * 1210


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.int8, np.complex64])
def test_unsupported_and_nonfinite_signal_rejected(dtype):
    data = np.zeros((2, 2, 2), dtype=dtype)
    if np.issubdtype(dtype, np.inexact):
        data[0, 0, 0] = np.nan
        data[1, 0, 0] = np.inf
    with pytest.raises(ValueError, match="outside NB-02"):
        generate_brain_cdm(data, **arguments(data))


@pytest.mark.parametrize("changes", [{"channel": None}, {"frame_index": None},
                                     {"frame_index": -1}, {"axis_order": "XYZ"}])
def test_unresolved_channel_frame_axes_rejected(changes):
    with pytest.raises(ValueError):
        replace(selection(), **changes)


def test_caller_must_select_a_scalar_frame_and_matching_geometry():
    data = np.full((2, 2, 2), 128, np.uint8)
    kwargs = arguments(data)
    for ambiguous in (data[None], np.stack((data, data), axis=-1)):
        with pytest.raises(ValueError, match="scalar ZYX"):
            generate_brain_cdm(ambiguous, **kwargs)
    with pytest.raises(ValueError, match="geometry does not match"):
        generate_brain_cdm(data[:1], **kwargs)
    chosen = replace(selection(), frame_index=4, channel=ChannelSelection("channel-3", 0))
    result = generate_brain_cdm(data, **dict(kwargs, selection=chosen))
    assert result.artifact.selection == chosen


@pytest.mark.parametrize("bad", [np.full((2, 2, 2), 2, np.uint8),
                                  np.full((2, 2, 2), np.nan), np.zeros((1, 2, 2), np.uint8)])
def test_invalid_masks_rejected(bad):
    data = np.zeros((2, 2, 2), np.uint8)
    kwargs = arguments(data)
    kwargs.update(parameters=CDMParameters(mode="signal_mask", allow_exploratory=True),
                  mask=bad, mask_selection=selection("mask"), mask_revision="mask-1")
    with pytest.raises(ValueError):
        generate_brain_cdm(data, **kwargs)


def test_boolean_mask_and_big_endian_uint16():
    case = next(c for c in SCALAR_CASES if c["name"] == "uint16_nonzero_lower_endpoint")
    data, kwargs = case_input(case)
    little = generate_brain_cdm(data, **kwargs)
    big = generate_brain_cdm(data.astype(">u2"), **kwargs)
    np.testing.assert_array_equal(little.rgb, big.rgb)
    assert little.artifact == big.artifact
    kwargs.update(mask=np.ones(data.shape, bool), mask_selection=selection("mask"), mask_revision="mask-1",
                  parameters=replace(kwargs["parameters"], mode="signal_mask"))
    np.testing.assert_array_equal(generate_brain_cdm(data, **kwargs).rgb, little.rgb)


@pytest.mark.parametrize("mutate", ["singular", "nan", "projective", "fractional", "outside"])
def test_invalid_or_unresolved_transforms_rejected(mutate):
    data = np.zeros((2, 2, 2), np.uint8)
    affine = np.eye(4)
    if mutate == "singular":
        affine[0, 0] = 0
    elif mutate == "nan":
        affine[0, 3] = np.nan
    elif mutate == "projective":
        affine[3, 0] = 0.1
    elif mutate == "fractional":
        affine[0, 3] = 0.5
    else:
        affine[0, 3] = 1210
    with pytest.raises(ValueError):
        generate_brain_cdm(data, **dict(arguments(data), mapping=CDMMapping(affine, "bad")))


def test_signed_axis_permutation_and_pose_consumed_once():
    data = np.zeros((3, 4, 5), np.uint8)
    data[1, 2, 3] = 255
    kwargs = arguments(data)
    affine = np.array([[0, 1, 0, 100], [-1, 0, 0, 200], [0, 0, 1, 86], [0, 0, 0, 1]])
    pose = np.eye(4)
    pose[:3, 3] = (123, 34, 7)
    kwargs.update(mapping=CDMMapping(affine, "permuted"), geometry=replace(kwargs["geometry"], pose=pose))
    result = generate_brain_cdm(data, **kwargs)
    assert np.argwhere(np.any(result.rgb, axis=-1)).tolist() == [[197, 102]]
    assert result.rgb[197, 102].tolist() == [129, 255, 126]  # global z=87
    assert result.artifact.geometry.pose == tuple(map(tuple, pose))


def test_nearest_affine_pull_sampling_and_tile_determinism():
    data = np.zeros((3, 3, 3), np.uint16)
    data[1, 1, 1] = 4095
    kwargs = arguments(data)
    affine = np.eye(4)
    affine[:3, 3] = (10.25, 20.25, 86.25)
    kwargs.update(mapping=CDMMapping(affine, "fractional"),
                  parameters=CDMParameters(display_range=(0, 4095), interpolation="nearest",
                                           depth_range=(87, 88), allow_exploratory=True))
    a = generate_brain_cdm(data, **kwargs)
    b = generate_brain_cdm(data, **kwargs, tile_rows=7)
    assert np.argwhere(np.any(a.rgb, axis=-1)).tolist() == [[21, 11]]
    assert a.rgb[21, 11].tolist() == [129, 255, 126]
    assert a.artifact == b.artifact
    np.testing.assert_array_equal(a.rgb, b.rgb)


def test_exploratory_success_never_claims_compatibility():
    data = np.zeros((2, 2, 2), np.uint8)
    kwargs = arguments(data)
    with pytest.raises(ValueError, match="exploratory"):
        generate_brain_cdm(data, **dict(kwargs, parameters=CDMParameters()))
    exploratory = generate_brain_cdm(data, **kwargs).artifact
    assert exploratory.output_label == "exploratory"
    evidence = replace(kwargs["mapping"], evidence_id="registration-record", evidence_sha256="a" * 64)
    supplied = generate_brain_cdm(data, **dict(kwargs, mapping=evidence, parameters=CDMParameters())).artifact
    assert supplied.execution_status == "completed"
    assert supplied.compatibility_status == "not_validated"
    assert supplied.output_label == "mapping_evidence_supplied"
    with pytest.raises(ValueError, match="compatibility_status"):
        replace(supplied, compatibility_status="validated")


def test_immutable_context_pixels_and_serialization():
    data = np.full((2, 2, 2), 128, np.uint8)
    kwargs = arguments(data)
    a = generate_brain_cdm(data, **kwargs)
    b = generate_brain_cdm(data, **kwargs, tile_rows=1)
    assert a.artifact == b.artifact
    assert a.artifact.context_sha256 == b.artifact.context_sha256
    with pytest.raises(ValueError):
        a.rgb.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        a.artifact.geometry.revision = "later"
    mutable = a.artifact.to_dict()
    mutable["selection"]["source"]["library_release"] = "changed"
    assert a.artifact.selection.source.library_release == "v1"
    restored = CDMArtifact.from_json(a.artifact.to_json())
    assert restored == a.artifact
    assert restored.context_sha256 == a.artifact.context_sha256


@pytest.mark.parametrize("stop_after", [1, 4, 16])
def test_cancellation_during_validation_projection_or_final_hash(stop_after):
    data = np.full((12, 4, 5), 255, np.uint8)
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        return calls >= stop_after
    with pytest.raises(CDMCancelled):
        generate_brain_cdm(data, **arguments(data), cancel=cancel)


def test_detect_input_revision_violation():
    data = np.zeros((12, 4, 5), np.uint8)
    calls = 0
    def mutate():
        nonlocal calls
        calls += 1
        if calls == 8:
            data[0, 0, 0] = 255
        return False
    with pytest.raises(ValueError, match="Source changed"):
        generate_brain_cdm(data, **arguments(data), cancel=mutate)


def test_export_lossless_round_trip_and_no_overwrite(tmp_path):
    data, kwargs = case_input(next(c for c in SCALAR_CASES if c["name"] == "crop_signal_mask"))
    result = generate_brain_cdm(data, **kwargs)
    destination = tmp_path / "artifact"
    exported = export_brain_cdm(result, destination)
    loaded = json.loads((destination / "manifest.json").read_text())
    assert CDMArtifact.from_dict(loaded["artifact"]) == exported
    assert exported.context_sha256 == result.artifact.context_sha256
    assert exported.png_file_sha256 == hashlib.sha256((destination / "cdm.png").read_bytes()).hexdigest()
    assert exported.png_file_sha256 != exported.rgb_pixel_sha256
    with Image.open(destination / "cdm.png") as image:
        np.testing.assert_array_equal(np.asarray(image), result.rgb)
    assert loaded["output_label"] == "exploratory"
    with pytest.raises(FileExistsError):
        export_brain_cdm(result, destination)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["artifact"]


@pytest.mark.parametrize("failure", ["png", "manifest", "fsync", "rename", "cancel"])
def test_export_failures_leave_no_completed_or_pending_bundle(tmp_path, monkeypatch, failure):
    data = np.full((2, 2, 2), 128, np.uint8)
    result = generate_brain_cdm(data, **arguments(data))
    def fail(*args, **kwargs):
        raise OSError("injected write failure")
    if failure == "png":
        monkeypatch.setattr(Image.Image, "save", fail)
    elif failure in ("fsync", "rename"):
        monkeypatch.setattr(os, failure, fail)
    elif failure == "manifest":
        original = Path.open
        def broken(path, *args, **kwargs):
            if path.name == "manifest.json":
                fail()
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "open", broken)
    def cancel():
        return any(p.name == "manifest.json" for p in tmp_path.rglob("*"))
    with pytest.raises(CDMCancelled if failure == "cancel" else OSError):
        export_brain_cdm(result, tmp_path / "artifact", cancel=cancel if failure == "cancel" else None)
    assert list(tmp_path.iterdir()) == []


def test_hash_strided_source_uses_bounded_canonical_chunks():
    data = np.arange(300000, dtype=np.uint16).reshape(30, 100, 100)[:, ::-1, ::-1]
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        return False
    assert array_checksum(data, cancel=check) == hashlib.sha256(data.astype("<u2").tobytes()).hexdigest()
    assert calls >= 5  # Cancellation remains reachable while hashing a large source.


def test_read_only_memmap_and_strided_mask_preserve_inputs(tmp_path):
    backing = tmp_path / "source.raw"
    original = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    backing.write_bytes(original.tobytes())
    data = np.memmap(backing, mode="r", dtype=np.uint8, shape=original.shape)
    mask = np.ones(data.shape, bool)[:, :, ::-1]
    kwargs = arguments(data)
    kwargs.update(mask=mask, mask_selection=selection("mask"), mask_revision="mask-1",
                  parameters=replace(kwargs["parameters"], mode="signal_mask"))
    result = generate_brain_cdm(data, **kwargs)
    assert result.artifact.source_pixel_sha256 == hashlib.sha256(original.tobytes()).hexdigest()
    assert backing.read_bytes() == original.tobytes()
    assert mask.all()
    data._mmap.close()


def test_nearest_mask_is_sampled_in_the_same_mapping():
    data = np.zeros((3, 3, 3), np.uint8)
    data[1, 1, :2] = (128, 255)
    mask = np.zeros(data.shape, bool)
    mask[1, 1, 0] = True
    kwargs = arguments(data)
    affine = np.eye(4)
    affine[:3, 3] = (10, 20, 86)
    kwargs.update(mask=mask, mask_selection=selection("mask"), mask_revision="mask-1",
                  mapping=CDMMapping(affine, "translated"),
                  parameters=CDMParameters(mode="signal_mask", depth_range=(87, 87),
                                           interpolation="nearest", allow_exploratory=True))
    nearest = generate_brain_cdm(data, **kwargs)
    exact = generate_brain_cdm(data, **dict(kwargs, parameters=replace(kwargs["parameters"], interpolation="exact")))
    np.testing.assert_array_equal(nearest.rgb, exact.rgb)
    assert np.argwhere(np.any(nearest.rgb, axis=-1)).tolist() == [[21, 10]]
    assert nearest.rgb[21, 10].max() == 128


def test_export_rejects_mismatched_pixels_without_staging(tmp_path):
    data = np.full((2, 2, 2), 128, np.uint8)
    result = generate_brain_cdm(data, **arguments(data))
    changed = result.rgb.copy()
    changed[0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        export_brain_cdm(replace(result, rgb=changed), tmp_path / "artifact")
    assert list(tmp_path.iterdir()) == []


def test_template_selection_and_serialized_context_are_validated():
    with pytest.raises(ValueError, match="pinned"):
        CDMMapping(np.eye(4), "mapping", template_id="different-grid")
    data = np.zeros((2, 2, 2), np.uint8)
    artifact = generate_brain_cdm(data, **arguments(data)).artifact
    for change in ({"template_shape_zyx": [174, 4, 5]}, {"schema_version": 2},
                   {"source_dtype": "<f4"}, {"png_file_sha256": "a" * 64},
                   {"numpy_version": {}}, {"scipy_version": {}}, {"warnings": "text"}):
        with pytest.raises(ValueError):
            CDMArtifact.from_dict(dict(artifact.to_dict(), **change))


def test_import_and_execution_do_not_load_gui_or_renderer():
    code = """
import sys
from madi3d_app.integrations.neuronbridge.cdm import generate_brain_cdm
from madi3d_app.integrations.neuronbridge.cdm_export import export_brain_cdm
assert not any(k.startswith(('PySide', 'PyQt', 'vtk')) for k in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
