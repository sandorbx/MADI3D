"""Deterministic NB-05 public protocol fixtures; no network or credentials."""
import copy
import json
import os
from threading import Event
from unittest import mock

import pytest
import requests

from madi3d_app.integrations.neuronbridge.assets import AssetMetadataClient
from madi3d_app.integrations.neuronbridge.public_api import (
    BASE_URL, PublicAPIError, PublicAPICancelled, PublicTransport, available_methods,
    collection, lookup, retrieve_matches,
)
from madi3d_app.integrations.neuronbridge.public_cache import PublicSnapshotCache
from madi3d_app.integrations.neuronbridge.records import SearchResults
from madi3d_app.integrations.neuronbridge.import_service import plan_result_rows
from madi3d_app.integrations.neuronbridge.evidence import RESULT_FLIP_X

VERSION = "v3_10_0"


def image(kind="em", id="123456789012345678901234567890", library="library one", space="JRC"):
    return {"type": "EMImage" if kind == "em" else "LMImage", "id": id,
            "libraryName": library, "alignmentSpace": space,
            "publishedName": "hemibrain:v1.2.1:636798093" if kind == "em" else "VT037023",
            "channel": None if kind == "em" else 3, "files": {"store": "brain",
                "CDSResults": id + ".json", **({"PPPMResults": id + ".json", "AlignedBodyOBJ": "body.obj",
                "AlignedBodySWC": "body.swc"} if kind == "em" else {"VisuallyLosslessStack": "stack.h5j"})}}


class Response:
    def __init__(self, value=None, status=200, headers=None, raw=None, chunks=None):
        self.body = json.dumps(value).encode() if raw is None else raw
        self.status_code, self.headers = status, headers or {}
        self.chunks = iter(chunks if chunks is not None else [self.body])
        self.raw = self
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def read1(self, size, decode_content=False):
        assert size == 64 * 1024 and not decode_content
        return next(self.chunks, b"")


def transport(responses, **kwargs):
    session = mock.Mock()
    session.get.side_effect = responses
    return PublicTransport(session=session, **kwargs)


def make_lookup(kind="em", *, images=None, identifier="636798093", extra=None):
    images = images if images is not None else [image(kind), image(kind, "second", "library two", "VNC"), image(kind, "third")]
    config = {"stores": {"brain": {"prefixes": {key: f"{BASE_URL}/{VERSION}/metadata/{key}/" for key in
        ("CDSResults", "PPPMResults", "AlignedBodyOBJ", "AlignedBodySWC", "VisuallyLosslessStack")}}}}
    t = transport([Response(raw=(VERSION + "\n").encode()), Response(config), Response({"results": images, **(extra or {})})])
    return lookup(t, identifier, kind), t


def make_results(snapshot=None, method="CDM", *, rows=None, extra=None, limit=5000):
    snapshot = snapshot or make_lookup()[0]
    target = image("lm" if snapshot["kind"] == "em" else "em")
    rows = rows if rows is not None else [{"type": "CDSMatch" if method == "CDM" else "PPPMatch", "image": target,
            "normalizedScore": 50000.25, "matchingPixels": 239, "mirrored": False,
            "pppmScore": 118, "pppmRank": 0.25, "annotations": ["original evidence"]}]
    body = {"inputImage": snapshot["images"][0], "results": rows, **(extra or {})}
    t = transport([Response(body)])
    return retrieve_matches(t, snapshot, 0, method, limit=limit), t


def parameters(result):
    return {p.name: p.value for p in result.session.parameters}


@pytest.mark.parametrize(("kind", "identifier", "lookup_path", "asset_type", "filename"), [
    ("em", "636798093", "by_body", "AlignedBodyOBJ", "body.obj"),
    ("lm", "VT037023", "by_line", "VisuallyLosslessStack", "stack.h5j"),
])
def test_native_asset_metadata_lookup(kind, identifier, lookup_path, asset_type, filename):
    record = image(kind)
    prefix = f"{BASE_URL}/{VERSION}/metadata/{asset_type}/"
    config = {"stores": {"brain": {
        "prefixes": {asset_type: prefix},
        "customSearch": {"defaultSearchLibrary": "test-library"},
    }}}
    wire = transport([Response(config), Response({"results": [record]})])
    client = AssetMetadataClient(VERSION, transport=wire)
    try:
        images = (client.get_em_images if kind == "em" else client.get_lm_images)(identifier)
        assert images == [record]
        assert client.version == VERSION
        assert client._get_files_url(images[0]["files"], asset_type) == prefix + filename
        assert wire.session.get.call_args.args[0] == f"{BASE_URL}/{VERSION}/metadata/{lookup_path}/{identifier}.json"
    finally:
        client.close()
    wire.session.close.assert_called_once()


@pytest.mark.parametrize("method,score", [("CDM", "50000.25"), ("PPPM", "0.25")])
def test_public_scene_comments_use_the_ranking_value(method, score):
    result, _ = make_results(method=method)
    (_, _, comments), = plan_result_rows(result)
    assert comments == ("", score, "")


@pytest.mark.parametrize("method,match_type,kind,space,mirrored", [
    ("CDM", "CDSMatch", "em", None, True),
    ("CDM", "CDSMatch", "em", "JRC2018U_HR", True),
    ("CDM", "CDSMatch", "em", "JRC2018VNCU_HR", True),
    ("CDM", "CDSMatch", "em", "JRC2018U", False),
    ("CDM", "CDSMatch", "em", "JRC2018U", None),
    ("CDM", "CDSMatch", "lm", "JRC2018U", True),
    ("CDM", "PPPMatch", "em", "JRC2018U", True),
    ("PPPM", "PPPMatch", "em", "JRC2018U", True),
])
def test_public_search_has_no_automatic_flip_without_the_mapped_query(method, match_type, kind, space, mirrored):
    query_kind = "em" if method == "PPPM" else "lm"
    snapshot, _ = make_lookup(query_kind, images=[image(query_kind, space="JRC2018U")])
    result, _ = make_results(snapshot, method=method, rows=[
        {"image": image(kind, space=space), "type": match_type, "mirrored": mirrored},
    ])
    (_, metadata, comments), = plan_result_rows(result)
    assert RESULT_FLIP_X not in metadata
    assert comments[2] == ""
    assert result.occurrences[0].mirrored is mirrored


@pytest.mark.parametrize("kind,method", [("em", "CDM"), ("lm", "CDM"), ("em", "PPPM")])
def test_official_directions_explicit_images_and_separate_scores(kind, method):
    snapshot, t = make_lookup(kind)
    assert len(snapshot["images"]) == 3
    assert snapshot["data_version"] == VERSION
    assert snapshot["requested_version"] == "current"
    assert t.session.get.call_count == 3
    with pytest.raises(PublicAPIError, match="Explicitly"):
        retrieve_matches(t, snapshot, None, method)
    result, t = make_results(snapshot, method)
    assert result.session.source_kind == "precomputed"
    assert result.session.query_source.image.image_id == snapshot["images"][0]["id"]
    target = result.occurrences[0].target
    assert target.kind == ("lm" if kind == "em" else "em")
    assert target.library_release is None  # NB release is not a library release.
    if target.kind == "em":
        assert target.biological.neuron_id is None  # publishedName is not a unique body-ID field
        assert target.published_name == "hemibrain:v1.2.1:636798093"
    assert target.channel is None or target.channel.index_base == 1
    hit = result.occurrences[0]
    fields = {f.name: f for f in hit.metrics}
    assert not hit.fields
    assert fields["normalizedScore"].value == 50000.25  # not a percentage
    assert hit.matched_pixels == 239
    assert fields["pppmScore"].value == 118
    assert fields["pppmRank"].value == 0.25  # fractional rank is not an integer rank or similarity score
    assert fields["normalizedScore"].meaning == ("neuronbridge:CDSMatch:normalizedScore" if method == "CDM" else None)
    assert fields["pppmScore"].meaning == ("neuronbridge:PPPMatch:pppmScore" if method == "PPPM" else None)
    assert hit.mirrored is False
    assert result.session.context["algorithm"] is None
    assert result.session.context["retrieval"]["retrieval_complete"]
    assert result.session.context["retrieval"]["search_complete"] is None
    assert SearchResults.from_dict(json.loads(json.dumps(result.to_dict()))) == result
    assert t.session.get.call_count == 1
    assert all(a.url for a in target.assets)


def test_huge_identifier_is_opaque_and_url_encoded_not_float_or_path():
    identifier = "9" * 1000
    snapshot, t = make_lookup(identifier=identifier)
    assert snapshot["identifier"] == identifier
    assert identifier in t.session.get.call_args.args[0]
    snapshot, t = make_lookup("lm", identifier="line/with?# characters")
    assert "line%2Fwith%3F%23%20characters.json" in t.session.get.call_args.args[0]
    with pytest.raises(PublicAPIError, match="not shortened"):
        lookup(t, "9" * 4097, "em")


def test_missing_methods_images_fields_and_response_evidence():
    snapshot, _ = make_lookup("lm")
    assert available_methods(snapshot["images"][0], snapshot["config"]["payload"]) == ("CDM",)
    with pytest.raises(PublicAPIError, match="available"):
        retrieve_matches(transport([]), snapshot, 0, "PPPM")
    result, _ = make_results(rows=[{"image": {}, "normalizedScore": "bad", "matchingPixels": -1}, None])
    assert result.occurrences[0].target.kind == "unknown"
    assert result.occurrences[0].metrics[0].value is None
    assert not result.session.context["retrieval"]["normalization_complete"]
    assert result.session.external_evidence["result_response"]["selected_results"]["1"] is None


@pytest.mark.parametrize("extra", [{"next": "page2"}, {"nextPageToken": "abc"}, {"total": 9000},
                                  {"pagination": {"hasMore": True}}, {"truncated": True}])
def test_unadvertised_pagination_never_becomes_complete_or_followed(extra):
    snapshot, _ = make_lookup(extra=extra)
    assert not snapshot["completeness"]["retrieval_complete"]
    result, t = make_results(extra=extra)
    retrieval = result.session.context["retrieval"]
    assert not retrieval["retrieval_complete"]
    assert retrieval["controls"] == extra
    assert retrieval["pages_retrieved"] == 1
    assert t.session.get.call_count == 1


def test_local_truncation_preserves_all_response_evidence_and_filters():
    result, _ = make_results(rows=[{"image": image("lm"), "type": "CDSMatch"}] * 3, limit=2)
    assert len(result.occurrences) == 2
    assert not result.session.context["retrieval"]["retrieval_complete"]
    assert result.session.context["retrieval"]["published_rows"] == 3
    assert result.session.context["retrieval"]["filters"]["local_limit"] == 2
    assert parameters(result)["result_limit"] == 2
    assert result.session.diagnostics and not result.diagnostics
    assert len(result.session.external_evidence["result_response"]["selected_results"]) == 2
    restored = SearchResults.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored == result
    assert restored.occurrences[0].target == restored.occurrences[1].target
    assert restored.occurrences[0].occurrence_id != restored.occurrences[1].occurrence_id


@pytest.mark.parametrize("body", [b'{', b'{"results":[],"results":[]}', b'{"score":NaN}', b'\xff'])
def test_malformed_wire_data_rejected(body):
    with pytest.raises(PublicAPIError, match="Malformed"):
        transport([Response(raw=body)]).get(BASE_URL + "/config.json")


@pytest.mark.parametrize("body", [[], {}, {"results": {}}, {"results": None}])
def test_malformed_envelopes_rejected(body):
    with pytest.raises(PublicAPIError, match="results array"):
        collection(body, 10)


def test_mismatched_query_and_unversioned_result_reference_rejected():
    snapshot, _ = make_lookup()
    with pytest.raises(PublicAPIError, match="disagrees"):
        retrieve_matches(transport([Response({"inputImage": image(id="other"), "results": []})]), snapshot, 0, "CDM")
    changed = copy.deepcopy(snapshot)
    changed["config"]["payload"]["stores"]["brain"]["prefixes"]["CDSResults"] = BASE_URL + "/current/"
    with pytest.raises(PublicAPIError, match="release"):
        retrieve_matches(transport([]), changed, 0, "CDM")


def test_rate_limits_timeout_retries_and_no_credentials():
    responses = [Response(status=429, headers={"Retry-After": "0"}), Response(status=503), Response({"results": []})]
    t = transport(responses)
    with mock.patch.object(t, "_wait", wraps=lambda _: t._check()) as wait:
        assert t.get(BASE_URL + "/test.json")["payload"] == {"results": []}
        assert wait.call_count == 2
    assert all(r.closed for r in responses)
    assert t.session.trust_env is False
    assert t.session.get.call_args.kwargs["timeout"] == (5, 10)
    assert t.session.get.call_args.kwargs["allow_redirects"] is False
    with pytest.raises(PublicAPIError, match="longer retry delay"):
        transport([Response(status=429, headers={"Retry-After": "120"})]).get(BASE_URL + "/test.json")
    t = transport([requests.ConnectionError("offline")] * 3)
    with mock.patch.object(t, "_wait"), pytest.raises(PublicAPIError, match="bounded retries"):
        t.get(BASE_URL + "/test.json")
    assert t.session.get.call_count == 3


def test_cancel_before_request_during_stream_and_retry_wait():
    cancel = Event()
    cancel.set()
    t = transport([], cancel=cancel)
    with pytest.raises(PublicAPICancelled):
        t.get(BASE_URL + "/test.json")
    t.session.get.assert_not_called()
    cancel.clear()
    def chunks():
        yield b'{'
        cancel.set()
        yield b'}'
    response = Response(chunks=chunks())
    with pytest.raises(PublicAPICancelled):
        transport([response], cancel=cancel).get(BASE_URL + "/test.json")
    assert response.closed
    cancel.clear()
    t = transport([Response(status=429)], cancel=cancel)
    with mock.patch.object(cancel, "wait", side_effect=lambda _: cancel.set()), pytest.raises(PublicAPICancelled):
        t.get(BASE_URL + "/test.json")


def test_size_and_time_limits_do_not_publish_plausible_partial_data():
    with pytest.raises(PublicAPIError, match="size limit"):
        transport([Response(raw=b"123456")], max_bytes=5).get(BASE_URL + "/test.json")
    with pytest.raises(PublicAPIError, match="time limit"):
        transport([], deadline_seconds=0).get(BASE_URL + "/test.json")


def test_slow_available_bytes_still_check_deadline_and_compression_is_bounded():
    now = [0.0]
    def chunks():
        for _ in range(100):
            now[0] += 1
            yield b" "
    with mock.patch("madi3d_app.integrations.neuronbridge.public_api.time.monotonic", side_effect=lambda: now[0]):
        t = transport([Response(chunks=chunks())], deadline_seconds=3)
        with pytest.raises(PublicAPIError, match="time limit"):
            t.get(BASE_URL + "/test.json")
    assert now[0] == 3
    with pytest.raises(PublicAPIError, match="compressed"):
        transport([Response(headers={"Content-Encoding": "gzip"})]).get(BASE_URL + "/test.json")


def test_http_failures_exhaustion_and_past_retry_date():
    t = transport([Response(status=429)] * 3)
    with mock.patch.object(t, "_wait"), pytest.raises(PublicAPIError, match="retry limit"):
        t.get(BASE_URL + "/test.json")
    assert t.session.get.call_count == 3
    with pytest.raises(PublicAPIError, match="HTTP 404"):
        transport([Response(status=404)]).get(BASE_URL + "/test.json")
    t = transport([Response(status=429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}), Response({})])
    with mock.patch.object(t, "_wait") as wait:
        t.get(BASE_URL + "/test.json")
        wait.assert_called_once_with(0)


def test_float_identifiers_are_rejected_without_rounding():
    for field in ("id", "libraryName", "alignmentSpace"):
        malformed = image()
        malformed[field] = 12345678901234567890.0
        with pytest.raises(PublicAPIError, match="not coerced"):
            make_lookup(images=[malformed])


def test_append_only_offline_cache_refresh_and_atomic_cleanup(tmp_path):
    cache = PublicSnapshotCache(tmp_path)
    first, _ = make_results()
    second, _ = make_results(rows=[{"image": image("lm"), "type": "CDSMatch", "normalizedScore": 1}])
    cache.save(first.to_dict(), "results")
    cache.save(second.to_dict(), "results")
    assert len(cache.entries()) == 2
    with mock.patch.object(requests.Session, "get", side_effect=AssertionError("offline")):
        assert cache.read(first.session.session_id) == first
    assert first.session.session_id != second.session.session_id
    with pytest.raises(PublicAPIError, match="overwritten"):
        cache.save(first.to_dict(), "results")
    snapshot, _ = make_lookup()
    original_replace = os.replace
    def replace_payload_then_fail(source, target):
        if str(target).endswith(".info.json"):
            raise OSError("disk full")
        original_replace(source, target)
    with mock.patch("madi3d_app.integrations.neuronbridge.public_cache.os.replace", side_effect=replace_payload_then_fail):
        with pytest.raises(OSError):
            cache.save(snapshot, "lookup")
    assert not (tmp_path / (snapshot["snapshot_id"] + ".json")).exists()
    assert not list(tmp_path.glob("*.part"))
    path = tmp_path / (first.session.session_id + ".json")
    path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(PublicAPIError, match="checksum"):
        cache.read(first.session.session_id)


def test_cache_cancellation_before_publication_removes_temporary_files(tmp_path):
    cache = PublicSnapshotCache(tmp_path)
    snapshot, _ = make_lookup()
    cancel = Event()
    with mock.patch("madi3d_app.integrations.neuronbridge.public_cache.os.fsync", side_effect=lambda _: cancel.set()):
        with pytest.raises(PublicAPICancelled):
            cache.save(snapshot, "lookup", cancel)
    assert not list(tmp_path.iterdir())
