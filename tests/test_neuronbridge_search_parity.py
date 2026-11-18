"""One positive-CDS route; storage representation must not change scientific results.

The same synthetic Brain candidate canvases are installed twice — as an
extracted official dense TIFF directory (Hemibrain distribution) and as a
published sparse RGB u32 package (FlyWire FAFB distribution) — and searched
with the identical retained query and parameter set. Accepted/rejected
candidates, matching-pixel counts, overlap, winning shift, mirror, rank and
tie order must agree exactly, and a completed search must examine every
installed candidate even when every HTTP route is sabotaged.
"""
import hashlib
import http.client
import urllib.request
from unittest import mock

import numpy as np
import pytest
import requests

from madi3d_app.integrations.neuronbridge.archive_catalog import distribution
from madi3d_app.integrations.neuronbridge.library_store import SparseLibraryReader
from madi3d_app.integrations.neuronbridge.local_scorer import PreparedQuery, SearchParameters
from madi3d_app.integrations.neuronbridge.local_search import search_library
from madi3d_app.integrations.neuronbridge.records import NeuronBridgeRecordError, SearchSession
from neuronbridge_local_fixtures import (
    build_installed_sparse_library, member_identity, retained_query, synthetic_populations,
    tiff_bytes, write_installation)

HEMIBRAIN = distribution("hemibrain-v1.2")
FAFB = distribution("flywire-fafb-v783")
RETIRED = ("snapshot_id", "source_snapshot_id", "inventory_sha256",
           "source_inventory_sha256", "manifest_sha256", "membership_sha256")

PARAMETER_SETS = (
    ("default", SearchParameters()),
    ("mirror-off", SearchParameters(mirror=False)),
    ("shift2-mirror-off", SearchParameters(xy_shift=2, mirror=False)),
    ("shift2-mirror-on", SearchParameters(xy_shift=2)),
    ("thresholds-tolerance", SearchParameters(query_threshold=40, candidate_threshold=40,
                                              color_tolerance_percent=5.0)),
    ("minimum-overlap", SearchParameters(minimum_overlap=0.5)),
    ("result-limit", SearchParameters(result_limit=2)),
)


def candidate_canvases(rgb):
    """Seven deterministic variants; ordinals 1 and 2 tie below the exact hit."""
    flat = rgb.reshape(-1, 3)
    foreground = np.flatnonzero(np.any(flat != 0, axis=1))
    drop_last, drop_second = rgb.copy(), rgb.copy()
    drop_last.reshape(-1, 3)[foreground[-1]] = 0
    drop_second.reshape(-1, 3)[foreground[-2]] = 0
    return [
        ("exact", rgb),
        ("drop-last-foreground", drop_last),
        ("drop-penultimate-foreground", drop_second),
        ("mirror", rgb[:, ::-1].copy()),
        ("shifted-two-rows", np.roll(rgb, 2, axis=0)),
        ("dimmed-below-threshold", (rgb // 16).astype(np.uint8)),  # under every candidate threshold
        ("background", np.zeros_like(rgb)),
    ]


def dense_installation(root, canvases):
    """Extract-style official dense library: strict-TIFF candidates plus installation."""
    directory = root / "ColorDepthLibraries" / HEMIBRAIN.library_id
    members = directory / HEMIBRAIN.candidate_directory
    members.mkdir(parents=True)
    for index, (_label, rgb) in enumerate(canvases):
        (members / f"{1000000 + index}-ParityTest-RT_18U.tif").write_bytes(tiff_bytes(rgb))
    write_installation(directory, library_id=HEMIBRAIN.library_id, candidate_count=len(canvases))
    return directory


def sparse_installation(root, canvases):
    """Published sparse RGB u32 library: inventory spans plus installation."""
    prototype = {"type": "EMImage", "gender": "f", "neuronType": "fixture"}
    members = []
    for index, (_label, rgb) in enumerate(canvases):
        token = str(1000000 + index)
        url, metadata, source, evidence = member_identity(FAFB, token, prototype)
        members.append({
            "key": f"{FAFB.alignment_space}/{FAFB.geometry_source_library}"
                   f"/searchable_neurons/{index}/{token}-{FAFB.alignment_space}-CDM.tif",
            "rgb": rgb, "metadata": {"url": url, **metadata},
            "metadata_evidence_default": evidence, "source": source})
    return build_installed_sparse_library(root, library_id=FAFB.library_id, members=members)


@pytest.fixture(scope="module")
def parity(tmp_path_factory):
    root = tmp_path_factory.mktemp("representation-parity")
    rgb, record = retained_query(root / "saved query")
    canvases = candidate_canvases(rgb)
    return record, {
        "dense": dense_installation(root / "dense", canvases),
        "sparse": sparse_installation(root / "sparse", canvases),
    }, canvases


def pixel_hash(rgb):
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def outcome(result):
    """Every scientific field a representation could be tempted to change."""
    return {
        "counts": result.session.context["counts"],
        "truncated": result.session.context["results_truncated"],
        "accepted": sorted(occurrence.input_hashes["decoded_rgb"] for occurrence in result.occurrences),
        "ranked": [(occurrence.rank, occurrence.input_hashes["decoded_rgb"], occurrence.matched_pixels,
                    occurrence.overlap, occurrence.mirrored, tuple(occurrence.shift))
                   for occurrence in sorted(result.occurrences, key=lambda item: item.rank)],
    }


@pytest.mark.parametrize("name", tuple(key for key, _value in PARAMETER_SETS))
def test_positive_results_are_identical_in_both_representations(parity, name):
    record, libraries, canvases = parity
    parameters = dict(PARAMETER_SETS)[name]
    results = {kind: search_library(directory, record, parameters) for kind, directory in libraries.items()}
    dense, sparse = outcome(results["dense"]), outcome(results["sparse"])
    assert dense == sparse
    total = len(canvases)
    for counts in (dense["counts"], sparse["counts"]):
        assert counts["total"] == counts["examined"] == total  # every candidate was examined
        assert counts["failed"] == 0 and counts["retained"] <= counts["matched"]
    assert dense["ranked"], f"{name}: the fixture stopped producing accepted candidates"
    # Acceptance itself is representation-independent and content-correct: the
    # exact copy is accepted; sub-threshold and empty canvases are not.
    assert pixel_hash(canvases[0][1]) in dense["accepted"]
    rejected_by_construction = {pixel_hash(canvases[5][1]), pixel_hash(canvases[6][1])}
    assert not rejected_by_construction & set(dense["accepted"])


def test_tie_order_is_identical_and_ordinal_first(parity):
    record, libraries, _canvases = parity
    runs = {kind: search_library(directory, record) for kind, directory in libraries.items()}
    sequences = {kind: [(occurrence.rank, occurrence.input_hashes["decoded_rgb"], occurrence.matched_pixels,
                         occurrence.candidate_ordinal)
                        for occurrence in sorted(runs[kind].occurrences, key=lambda item: item.rank)]
                 for kind in runs}
    assert sequences["dense"] == sequences["sparse"]
    counts = [entry[2] for entry in sequences["dense"]]
    tied_values = {value for value in counts if counts.count(value) > 1}
    assert tied_values, "the fixture must exercise at least one score tie"
    for value in tied_values:
        group = [entry[3] for entry in sequences["dense"] if entry[2] == value]
        assert group == sorted(group)  # equal scores rank by earlier candidate ordinal first


def test_sparse_positive_scan_never_reconstructs_dense_canvases(parity):
    record, libraries, canvases = parity
    with mock.patch.object(SparseLibraryReader, "dense_canvas",
                           side_effect=AssertionError("A positive scan must consume sparse spans")), \
            mock.patch.object(PreparedQuery, "score_pixels",
                              side_effect=AssertionError("Sparse scoring must be candidate-first")):
        result = search_library(libraries["sparse"], record)
    counts = result.session.context["counts"]
    assert counts["total"] == counts["examined"] == len(canvases)
    assert counts["failed"] == 0 and counts["retained"] == counts["matched"]
    assert pixel_hash(canvases[0][1]) in {occurrence.input_hashes["decoded_rgb"]
                                          for occurrence in result.occurrences}


@pytest.mark.parametrize("kind", ("dense", "sparse"))
def test_search_completes_when_every_http_route_fails(parity, kind):
    record, libraries, canvases = parity
    with mock.patch.object(requests, "Session", side_effect=AssertionError("No HTTP clients")), \
            mock.patch.object(requests, "get", side_effect=AssertionError("No HTTP")), \
            mock.patch.object(requests, "post", side_effect=AssertionError("No HTTP")), \
            mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("No HTTP")), \
            mock.patch.object(http.client, "HTTPConnection", side_effect=AssertionError("No HTTP")), \
            mock.patch.object(http.client, "HTTPSConnection", side_effect=AssertionError("No HTTP")):
        result = search_library(libraries[kind], record)
    counts = result.session.context["counts"]
    assert result.session.context["completion_status"] == "completed"
    assert counts["total"] == counts["examined"] == len(canvases)
    assert counts["failed"] == 0 and counts["retained"] == counts["matched"] > 0


@pytest.mark.parametrize("kind", ("dense", "sparse"))
def test_installed_evidence_names_the_distribution_without_retired_hashes(parity, kind):
    record, libraries, canvases = parity
    context = search_library(libraries[kind], record).session.context
    descriptor = distribution(context["library_id"])
    assert context["schema_version"] == 3
    assert context["backend"] == "madi3d-local"
    assert context["provider"] == descriptor.provider
    assert context["representation"] == descriptor.representation
    assert len(context["archive_sha256"]) == 64
    assert context["search_profile"] == descriptor.profile_id
    assert context["counts"]["total"] == len(canvases)
    assert not any(name in context for name in RETIRED)


def test_new_sessions_refuse_retired_snapshot_and_inventory_fields(parity):
    record, libraries, _canvases = parity
    session = search_library(libraries["sparse"], record).session
    for name, value in (("source_snapshot_id", "snapshot-should-never-return"),
                        ("source_inventory_sha256", "0" * 64),
                        ("membership_sha256", "0" * 64),
                        ("inventory_sha256", "1" * 64)):
        with pytest.raises(NeuronBridgeRecordError, match="retired"):
            SearchSession(session.session_id, "custom_query", session.neuronbridge_data_version,
                          query_reference=session.query_reference, parameters=session.parameters,
                          context=dict(session.context, **{name: value}),
                          external_evidence=session.external_evidence)


@pytest.mark.parametrize("field", ["library_id", "archive_sha256"])
def test_schema3_requires_canonical_local_identity(parity, field):
    from dataclasses import replace
    record, libraries, _ = parity
    session = search_library(libraries["sparse"], record).session
    context = dict(session.context)
    context.pop(field)
    with pytest.raises(NeuronBridgeRecordError):
        replace(session, context=context)
