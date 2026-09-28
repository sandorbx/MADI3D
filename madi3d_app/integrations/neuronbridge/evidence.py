"""NB-01A evidence ownership in existing project and object metadata contracts."""
from __future__ import annotations

import copy
import json
import hashlib
from dataclasses import replace
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit
from uuid import uuid4

from .records import SearchResults
from .remote import RemoteSource, remote_from_metadata, same_source_identity

SESSION_KEY = "neuronbridge_sessions"
OBJECT_KEY = "neuronbridge"
RETRIEVAL_KEY = "neuronbridge_retrievals"
RESOLUTION_KEY = "neuronbridge_resolutions"
SOURCE_KEY = "neuronbridge_source_observations"
REMOTE_KEYS = (RETRIEVAL_KEY, RESOLUTION_KEY, SOURCE_KEY)


def _record_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


_TRANSIENT_EVIDENCE_KEYS = frozenset({
    "cache_filename", "cache_path", "cached_path", "local_path",
    "elapsed_seconds", "completed_at", "retrieved_at",
})


def _durable_observation(value):
    """Drop execution/transport details while retaining bounded observations."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if key in _TRANSIENT_EVIDENCE_KEYS:
                continue
            if key in {"url", "asset_url"} and isinstance(child, str):
                parsed = urlsplit(child)
                if parsed.query or parsed.fragment:
                    continue
            result[key] = _durable_observation(child)
        return result
    if isinstance(value, list):
        return [_durable_observation(child) for child in value]
    return copy.deepcopy(value)


def _material_session_context(context):
    """Keep result-affecting/search-identity facts, not execution narration."""
    return _durable_observation({key: value for key, value in context.items() if key not in {
        "tie_order", "result_limit_scope",
    }})


def _durable_asset_resolution(value):
    if isinstance(value, dict):
        return {key: _durable_asset_resolution(child) for key, child in value.items()
                if key not in {"url", "asset_url", "cache_filename", "cache_path",
                               "local_path", "resolved_at", "retrieved_at", "api_metadata"}}
    if isinstance(value, list):
        return [_durable_asset_resolution(child) for child in value]
    return copy.deepcopy(value)


def _is_project_session_payload(payload):
    """Cheap marker-by-shape for the canonical payload produced below."""
    if payload.get("csv_bytes_base64") is not None:
        return False
    context = payload.get("session", {}).get("context", {})
    if any(key in context for key in (*_TRANSIENT_EVIDENCE_KEYS, "tie_order", "result_limit_scope")):
        return False
    external = payload.get("session", {}).get("external_evidence", {})
    if "catalog" in external or "manifest" in external:
        return False
    return not any(isinstance(value, dict) and "payload" in value for value in external.values())


def _selected_evidence_key(payloads):
    encoded = json.dumps(payloads, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def merge_remote_records(metadata, incoming):
    result = dict(metadata)
    for field in REMOTE_KEYS:
        registry = dict(result.get(field, {}))
        for key, value in incoming.get(field, {}).items():
            if key in registry and registry[key] != value:
                raise ValueError("Conflicting retained NeuronBridge retrieval evidence.")
            registry[key] = copy.deepcopy(value)
        if registry:
            result[field] = registry
    return result


def remote_for_reference(evidence, project_metadata, *, sources=None):
    embedded = tuple(SearchResults.from_dict(value) for value in evidence.get("selected_results", ()))
    occurrences = [(result, occurrence) for result in embedded for occurrence in result.occurrences]
    if occurrences:
        versions = {result.session.neuronbridge_data_version
                    for result, _occurrence in occurrences}
        observed = [occurrence.target for _result, occurrence in occurrences]
        if len(versions) != 1 or any(
                first != second and not same_source_identity(first, second)
                for index, first in enumerate(observed)
                for second in observed[index + 1:]):
            raise ValueError("A NeuronBridge object has conflicting selected source identities.")

        def evidence_size(value):
            if isinstance(value, dict):
                return sum(evidence_size(child) for child in value.values())
            if isinstance(value, (list, tuple)):
                return sum(evidence_size(child) for child in value)
            return int(value is not None and value != "")

        source = max(observed, key=lambda value: evidence_size(value.to_dict()))
        data_version = next(iter(versions))
    else:
        ref = evidence["source_ref"]
        if sources is None:
            sources = source_reference_index(project_metadata)
        source, data_version = sources[(ref["session_id"], ref["occurrence_id"])]
    if evidence.get("source_observation_ref"):
        from .records import SourceIdentity
        observation = project_metadata[SOURCE_KEY][evidence["source_observation_ref"]]
        source = SourceIdentity.from_dict(observation["source"])
        data_version = observation["data_version"]
    resolution = mutable_metadata_copy(project_metadata.get(RESOLUTION_KEY, {}).get(evidence.get("resolution_ref"), {}))
    if evidence.get("resolution_ref") and not resolution:
        raise ValueError("Missing NeuronBridge asset selection evidence.")
    if evidence.get("retrieval_ref"):
        retrieval = project_metadata[RETRIEVAL_KEY][evidence["retrieval_ref"]]
        if retrieval["asset_identity"] != _asset_identity(source, resolution):
            raise ValueError("Retrieval identity disagrees with selected asset.")
        resolution.update(copy.deepcopy(retrieval["facts"]))
    # Source observation and chosen asset are separate owners. Keep the source
    # assets intact; resolution records the exact choice without replacing its
    # reusable locator with the URL-free persistence projection.
    return RemoteSource(source, data_version,
                        resolution=resolution,
                        problems=tuple(evidence.get("source_problems", ())))


def _asset_identity(source, resolution):
    # Image/channel identity stays at the hit. Only the versioned file is shared.
    asset = resolution.get("selected_asset")
    if asset:
        asset = {key: value for key, value in asset.items()
                 if key != "url" and value is not None}
    return {"library": source.library, "release": source.library_release,
            "alignment": source.image.alignment_space,
            "data_version": resolution.get("data_version"),
            "asset": asset,
            "asset_type": resolution["asset_type"]}


def source_reference_index(project_metadata, *, index=None):
    if index is not None:
        return index.sources
    sources = {}
    for payload in project_metadata.get(SESSION_KEY, {}).values():
        results = SearchResults.from_dict(payload)
        sources.update({(o.session_id, o.occurrence_id): (o.target, results.session.neuronbridge_data_version)
                        for o in results.occurrences})
    return sources


def retain_remote(metadata, project_metadata, *, sources=None):
    """Normalize loader projections into hit references and shared observations.

    API claims and MADI3D interpretations remain separate fields. Full evidence
    equality is required for sharing; channel-dependent observations never merge.
    """
    result = mutable_metadata_copy(metadata)
    project = dict(project_metadata)

    def retain(evidence):
        activity = evidence.pop("last_retrieval", None)
        if activity and (activity.get("status") not in {
                "queued", "downloading", "downloaded", "decoding", "loaded", "cancelled"} or activity.get("candidates")):
            # Legacy failures were not typed. Keep potentially meaningful
            # failures and unresolved choices instead of guessing their origin.
            evidence["source_decision"] = activity
        value = evidence.pop("remote_source", None)
        if value is not None:
            remote = RemoteSource.from_dict(value)
            if not evidence.get("selected_results") and not evidence.get("source_ref"):
                # Historical standalone descriptors have no originating hit.
                evidence["remote_source"] = remote.to_dict()
                return
            original = remote_for_reference(evidence, project, sources=sources)
            if remote.source != original.source or remote.data_version != original.data_version:
                observation = {"source": remote.source.to_dict(), "data_version": remote.data_version}
                key = _record_key(observation)
                registry = dict(project.get(SOURCE_KEY, {}))
                registry[key] = observation
                project[SOURCE_KEY] = registry
                evidence["source_observation_ref"] = key
            else:
                evidence.pop("source_observation_ref", None)
            evidence.pop("source_ref", None)
            if remote.problems:
                evidence["source_problems"] = list(remote.problems)
            else:
                evidence.pop("source_problems", None)
            resolution = copy.deepcopy(dict(remote.resolution))
            resolution.pop("cache_filename", None)
            if resolution:
                facts = {key: resolution.pop(key) for key in ("retrieved_sha256", "provider_verification")
                         if key in resolution}
                resolution.pop("retrieved_at", None)
                if facts:
                    identity = _asset_identity(remote.source, resolution)
                    key = _record_key({"asset_identity": identity, "facts": facts})
                    registry = dict(project.get(RETRIEVAL_KEY, {}))
                    registry.setdefault(key, {"asset_identity": identity, "facts": facts})
                    project[RETRIEVAL_KEY] = registry
                    evidence["retrieval_ref"] = key
                else:
                    evidence.pop("retrieval_ref", None)
                resolution = _durable_asset_resolution(resolution)
                key = _record_key(resolution)
                registry = dict(project.get(RESOLUTION_KEY, {}))
                registry[key] = resolution
                project[RESOLUTION_KEY] = registry
                evidence["resolution_ref"] = key
            else:
                evidence.pop("resolution_ref", None)
                evidence.pop("retrieval_ref", None)
        for parent in evidence.get("derived_from", ()):
            retain(parent)

    if result.get(OBJECT_KEY):
        retain(result[OBJECT_KEY])
    return result, project


def project_remote_evidence(objects, project_metadata, *, index=None):
    sources = source_reference_index(project_metadata, index=index)
    rows = []
    for row in objects:
        old_remote = remote_from_metadata(row.object_metadata, project_metadata, sources=sources)
        metadata, project_metadata = retain_remote(row.object_metadata, project_metadata, sources=sources)
        paths = {}
        if old_remote is not None:
            cache_name = old_remote.resolution.get("cache_filename")
            for field in ("file_path", "alternate_file_path"):
                path = getattr(row, field)
                if path and (path == old_remote.local_path or (cache_name and cache_name in {
                        PureWindowsPath(path).name, PurePosixPath(path).name})):
                    paths[field] = ""
        rows.append(replace(row, object_metadata=metadata, **paths) if paths or metadata != row.object_metadata else row)
    return tuple(rows), project_metadata


def expanded_remote_metadata(metadata, project_metadata, *, sources=None):
    """Disposable UI/loader projection; canonical project rows retain references."""
    result = mutable_metadata_copy(metadata)
    remote = remote_from_metadata(result, project_metadata, sources=sources)
    if remote is not None:
        result[OBJECT_KEY]["remote_source"] = remote.to_dict()
    return result


def referenced_remote_records(metadata, project_metadata):
    result = {}
    pending = [(metadata or {}).get(OBJECT_KEY, {})]
    while pending:
        evidence = pending.pop()
        for ref, field in (("retrieval_ref", RETRIEVAL_KEY), ("resolution_ref", RESOLUTION_KEY),
                           ("source_observation_ref", SOURCE_KEY)):
            key = evidence.get(ref)
            if key:
                result.setdefault(field, {})[key] = mutable_metadata_copy(project_metadata[field][key])
        pending.extend(evidence.get("derived_from", ()))
    return result


def migrate_remote_backings(model, objects, project_metadata, *, index):
    """Move proven legacy cache locators out of durable volume-file fields."""
    from .assets import bind_remote_backing
    sources = source_reference_index(project_metadata, index=index)
    replacements = {}
    for row in objects:
        backing = model.backing_sources.get(row.backing_source_id)
        if backing is None:
            continue
        remote = remote_from_metadata(row.object_metadata, project_metadata, sources=sources)
        if remote is None:
            continue
        if backing.remote_locator:
            replacement = copy.deepcopy(backing)
            bind_remote_backing(replacement, backing.remote_locator, "")
            replacements[row.backing_source_id] = replacement
            continue
        if not remote.resolution:
            continue
        cache_name = remote.resolution.get("cache_filename")
        for path in (backing.primary_path, backing.runtime_path):
            if path and (path == remote.local_path or (cache_name and cache_name in {
                    PureWindowsPath(path).name, PurePosixPath(path).name})):
                replacement = copy.deepcopy(backing)
                bind_remote_backing(replacement, remote.resolution["url"], path)
                replacements[row.backing_source_id] = replacement
                break
    if not replacements:
        return model
    candidate = copy.copy(model)
    candidate.backing_sources = {**model.backing_sources, **replacements}
    return candidate


def merge_sessions(metadata, sessions):
    result = copy.deepcopy(dict(metadata or {}))
    if not sessions:
        return result
    registry = dict(result.get(SESSION_KEY, {}))
    result[SESSION_KEY] = registry
    for key, payload in sessions.items():
        parsed = project_search_results(SearchResults.from_dict(payload))
        if key != parsed.session.session_id:
            raise ValueError("NeuronBridge session key disagrees with its identity.")
        if key in registry:
            parsed = merge_search_results(SearchResults.from_dict(registry[key]), parsed)
        registry[key] = json.loads(json.dumps(parsed.to_dict(), allow_nan=False))
    return result


def project_search_results(results, hit_ids=None, *, index=None):
    """Project search evidence without raw source payloads or runtime details.

    With ``hit_ids`` this is an object-owned selected-occurrence subset. Without
    them it is an explicitly retained complete search artifact containing all
    retained candidates, but never the originating whole CSV/API/manifest.
    """
    complete = hit_ids is None
    hit_ids = ({o.occurrence_id for o in results.occurrences}
               if complete else set(hit_ids))
    hits = (tuple(sorted((index.matches[(results.session.session_id, oid)].occurrence for oid in hit_ids),
                         key=lambda hit: hit.row_index)) if index is not None else
            tuple(o for o in results.occurrences if o.occurrence_id in hit_ids))
    if len(hits) != len(hit_ids):
        raise ValueError("Missing originating NeuronBridge hit evidence.")
    external = {}
    for hit in hits:
        if hit.evidence_index is not None:
            original = results.session.external_evidence[hit.evidence_ref]
            response = external.setdefault(hit.evidence_ref, {
                "scope": "selected_results",
                "selected_results": {},
            })
            response["selected_results"][str(hit.evidence_index)] = _durable_observation(
                results.session.evidence_for(hit))
        elif hit.evidence_ref:
            pending = [hit.evidence_ref]
            while pending:
                key = pending.pop()
                if key in external:
                    continue
                record = _durable_observation(results.session.external_evidence[key])
                external[key] = record
                pending.extend(record.get("references", {}).values())
                if record.get("response_ref"):
                    pending.append(record["response_ref"])
    # Old column observations are materialized only for the selected CSV rows.
    if results.session.source_kind == "imported_csv":
        hits = tuple(replace(o, fields=o.fields or results.session.historical_fields_for(o)) for o in hits)
    elif "schema1_fields" in results.session.external_evidence:
        table = results.session.external_evidence["schema1_fields"]
        external["schema1_fields"] = {"columns": mutable_metadata_copy(table["columns"]),
            "occurrences": {oid: mutable_metadata_copy(row) for oid, row in table["occurrences"].items() if oid in hit_ids}}
    session = replace(results.session, context=_material_session_context(results.session.context),
                      external_evidence=external,
                      diagnostics=tuple(d for d in results.session.diagnostics if d.row_index is None))
    reported = results.subset["reported_retained_count"] if results.subset else len(results.occurrences)
    subset = None if complete else {
        "scope": "selected_hits", "complete_session": False,
        "included_hit_ids": sorted(hit_ids), "included_hit_count": len(hits),
        "reported_retained_count": reported,
    }
    return SearchResults(session, hits, tuple(d for d in results.diagnostics
                         if d.row_index is None or d.row_index in {o.row_index for o in hits}),
                         subset=subset)


def scoped_search_results(results, hit_ids, *, index=None):
    return project_search_results(results, hit_ids, index=index)


def merge_search_results(current, incoming):
    """Merge compatible subsets, comparing immutable context and shared hits."""
    if current.subset is None and incoming.subset is None:
        if current != incoming:
            raise ValueError(f"Conflicting NeuronBridge evidence for session {current.session.session_id}.")
        return current
    left = scoped_search_results(current, [o.occurrence_id for o in current.occurrences]) if current.subset else current
    right = scoped_search_results(incoming, [o.occurrence_id for o in incoming.occurrences]) if incoming.subset else incoming
    # Compare the same portable representation when one side has full evidence.
    if current.subset is None:
        left = scoped_search_results(current, [o.occurrence_id for o in incoming.occurrences])
    if incoming.subset is None:
        right = scoped_search_results(incoming, [o.occurrence_id for o in current.occurrences])
    if (_record_key(replace(left.session, external_evidence={}).to_dict()) != _record_key(replace(right.session, external_evidence={}).to_dict())
            or left.subset["reported_retained_count"] != right.subset["reported_retained_count"]):
        raise ValueError("Conflicting NeuronBridge session context.")
    hits = {o.occurrence_id: o for o in left.occurrences}
    for hit in right.occurrences:
        if hit.occurrence_id in hits and _record_key(hits[hit.occurrence_id].to_dict()) != _record_key(hit.to_dict()):
            raise ValueError("Conflicting NeuronBridge hit identity/content.")
        hits[hit.occurrence_id] = hit
    external = mutable_metadata_copy(left.session.external_evidence)
    for key, value in right.session.external_evidence.items():
        if key in external and external[key] != value:
            if key == "schema1_fields":
                a, b = external[key], value
                if a["columns"] != b["columns"]:
                    raise ValueError("Conflicting NeuronBridge historical columns.")
                for oid, row in b["occurrences"].items():
                    if oid in a["occurrences"] and _record_key(a["occurrences"][oid]) != _record_key(row):
                        raise ValueError("Conflicting NeuronBridge historical row.")
                    a["occurrences"][oid] = mutable_metadata_copy(row)
                continue
            if value.get("scope") != "selected_results" or external[key].get("scope") != "selected_results":
                raise ValueError("Conflicting NeuronBridge hit observations.")
            a, b = external[key], value
            if {k: v for k, v in a.items() if k != "selected_results"} != {k: v for k, v in b.items() if k != "selected_results"}:
                raise ValueError("Conflicting NeuronBridge response linkage.")
            for index, row in b["selected_results"].items():
                if index in a["selected_results"] and a["selected_results"][index] != row:
                    raise ValueError("Conflicting NeuronBridge result observation.")
                a["selected_results"][index] = mutable_metadata_copy(row)
        else:
            external[key] = mutable_metadata_copy(value)
    if current.subset is None:
        return current
    if incoming.subset is None:
        return incoming
    subset = {**left.subset, "included_hit_ids": sorted(hits), "included_hit_count": len(hits)}
    diagnostics = tuple(dict.fromkeys((*left.diagnostics, *right.diagnostics)))
    return SearchResults(replace(left.session, external_evidence=external),
                         tuple(sorted(hits.values(), key=lambda o: o.row_index)), diagnostics, subset=subset)


def reference(occurrence):
    return {"session_id": occurrence.session_id, "occurrence_id": occurrence.occurrence_id}


def _selected_payloads(metadata):
    evidence = (metadata or {}).get(OBJECT_KEY, {})
    payloads, pending = [], [evidence]
    while pending:
        origin = pending.pop()
        payloads.extend(origin.get("selected_results", ()))
        pending.extend(origin.get("derived_from", ()))
    return tuple(payloads)


def selected_results(metadata):
    """Return object-owned bounded results, including retained parent origins."""
    return tuple(SearchResults.from_dict(payload) for payload in _selected_payloads(metadata))


def _merge_selected_payloads(payloads, incoming):
    grouped = {}
    for payload in (*payloads, incoming.to_dict()):
        parsed = SearchResults.from_dict(payload)
        current = grouped.get(parsed.session.session_id)
        grouped[parsed.session.session_id] = (
            merge_search_results(current, parsed) if current is not None else parsed)
    return [grouped[key].to_dict() for key in sorted(grouped)]


def object_metadata_for_match(results, occurrence):
    from .assets import ResolutionError, validate_resolution_request

    remote = RemoteSource(
        occurrence.target, results.session.neuronbridge_data_version,
        problems=tuple(d.message for d in occurrence.diagnostics if d.code in {
            "ambiguous_identity_field", "ambiguous_target_identity", "missing_target_identity",
        }),
    )
    diagnostics = [d.message for d in occurrence.diagnostics]
    try:
        validate_resolution_request(remote)
    except ResolutionError as exc:
        if str(exc) not in diagnostics:
            diagnostics.append(str(exc))
    return {OBJECT_KEY: {
        "version": 2, "origin_id": str(uuid4()),
        "selected_results": [scoped_search_results(results, [occurrence.occurrence_id]).to_dict()],
        "remote_source": remote.to_dict(),
        "diagnostics": diagnostics,
    }}


def match_references(metadata):
    return [{"session_id": payload["session"]["session_id"],
             "occurrence_id": occurrence["occurrence_id"]}
            for payload in _selected_payloads(metadata)
            for occurrence in payload.get("occurrences", ())]


def referenced_sessions(metadata, project_metadata, *, exclude=()):
    registry = (project_metadata or {}).get(SESSION_KEY, {})
    result = {}
    session_ids = list((metadata or {}).get(OBJECT_KEY, {}).get("session_ids", ()))
    for key in session_ids:
        if key not in registry:
            raise ValueError(f"Missing NeuronBridge search session {key}.")
        if key not in exclude and key not in result:
            result[key] = copy.deepcopy(registry[key])
    return result


def _record_unavailable_sessions(metadata, session_ids, occurrence_refs=()):
    """Retain one bounded read-time warning for unavailable optional history."""
    session_ids = sorted({str(value) for value in session_ids if str(value)})
    occurrence_refs = sorted({
        (str(ref.get("session_id") or ""), str(ref.get("occurrence_id") or ""))
        for ref in occurrence_refs
        if isinstance(ref, dict)
        and str(ref.get("session_id") or "")
        and str(ref.get("occurrence_id") or "")
    })
    if not session_ids and not occurrence_refs:
        return metadata
    result = dict(metadata)
    existing = result.get("history_recovery_diagnostics", ())
    if not isinstance(existing, (list, tuple)):
        existing = ()
    details = []
    if session_ids:
        shown = ", ".join(session_ids[:5])
        if len(session_ids) > 5:
            shown += f" (+{len(session_ids) - 5} more)"
        details.append(f"session IDs {shown}")
    if occurrence_refs:
        shown = ", ".join(f"{session}/{occurrence}" for session, occurrence in occurrence_refs[:5])
        if len(occurrence_refs) > 5:
            shown += f" (+{len(occurrence_refs) - 5} more)"
        details.append(f"match occurrences {shown}")
    message = (
        "NeuronBridge history is incomplete; unavailable retained references: "
        f"{'; '.join(details)}. Local objects and bounded object evidence were retained."
    )
    result["history_recovery_diagnostics"] = list(dict.fromkeys(
        (*(str(value) for value in existing if str(value)), message)
    ))
    return result


def _read_only(*_args, **_kwargs):
    raise TypeError("Retained NeuronBridge evidence is read-only; replace a record through validation.")


class _EvidenceDict(dict):
    """JSON-compatible frozen payload; safe to share across document revisions."""
    __setitem__ = __delitem__ = __ior__ = _read_only
    clear = pop = popitem = setdefault = update = _read_only

    def __deepcopy__(self, memo):
        return self


class _EvidenceList(list):
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _read_only
    append = clear = extend = insert = pop = remove = reverse = sort = _read_only

    def __deepcopy__(self, memo):
        return self


def _freeze_evidence(value):
    if isinstance(value, (_EvidenceDict, _EvidenceList)):
        return value
    if isinstance(value, dict):
        return _EvidenceDict((key, _freeze_evidence(child)) for key, child in value.items())
    if isinstance(value, list):
        return _EvidenceList(_freeze_evidence(child) for child in value)
    if isinstance(value, tuple):
        return tuple(_freeze_evidence(child) for child in value)
    return value


def mutable_metadata_copy(value):
    """Return ordinary detached JSON containers at the serialization boundary."""
    if isinstance(value, dict):
        return {key: mutable_metadata_copy(child) for key, child in value.items()}
    if isinstance(value, list):
        return [mutable_metadata_copy(child) for child in value]
    if isinstance(value, tuple):
        return tuple(mutable_metadata_copy(child) for child in value)
    return copy.deepcopy(value)


class EvidenceValidation:
    """Document-local validation receipt, never serialized or shared globally.

    Frozen query/session payloads avoid copying retained history on the GUI
    thread. Replacements still pass full validation; serialized loads receive
    no receipt. Other metadata and all cross-references remain checked normally.
    """

    def __init__(self, queries, sessions, results, index, selected_key=None):
        self._queries = _freeze_evidence(queries)
        self._sessions = _freeze_evidence(sessions)
        self._results = results
        self._remote = {}
        self._selected_key = selected_key
        self.index = index

    def freeze_metadata(self, metadata):
        from .query import QUERY_KEY
        result = dict(metadata or {})
        for key in REMOTE_KEYS:
            if key in result:
                result[key] = {name: _freeze_evidence(value) for name, value in result[key].items()}
                self._remote[key] = dict(result[key])
        for key, values in ((QUERY_KEY, self._queries), (SESSION_KEY, self._sessions)):
            if key in result:
                # Keep registries replaceable; their retained records are frozen.
                result[key] = dict(values)
        return result


def _same_evidence(left, right):
    """Strict JSON comparison: bool/int/float equality cannot confer validation."""
    if type(left) is not type(right):
        return False
    if left is right:
        return True
    if isinstance(left, dict):
        return len(left) == len(right) and all(
            _same_evidence(lk, rk) and _same_evidence(lv, rv)
            for (lk, lv), (rk, rv) in zip(left.items(), right.items())
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same_evidence(lv, rv) for lv, rv in zip(left, right))
    return left == right


def validate_evidence(
    objects,
    project_metadata,
    *,
    previous=None,
    origins=(),
    prevalidated_sessions=None,
):
    from .query import QUERY_KEY, validate_queries, validate_query_associations, validate_query_freshness
    from .workspace import validate_workspace

    metadata = project_metadata or {}
    old_remote = previous._remote if isinstance(previous, EvidenceValidation) else {}
    for key, resolution in metadata.get(RESOLUTION_KEY, {}).items():
        if key in old_remote.get(RESOLUTION_KEY, {}) and old_remote[RESOLUTION_KEY][key] is resolution:
            continue
        if key != _record_key(resolution):
            raise ValueError("Changed NeuronBridge resolution observation.")
    for key, observation in metadata.get(SOURCE_KEY, {}).items():
        if key in old_remote.get(SOURCE_KEY, {}) and old_remote[SOURCE_KEY][key] is observation:
            continue
        if key != _record_key(observation):
            raise ValueError("Changed NeuronBridge source observation.")
    for key, retrieval in metadata.get(RETRIEVAL_KEY, {}).items():
        if key in old_remote.get(RETRIEVAL_KEY, {}) and old_remote[RETRIEVAL_KEY][key] is retrieval:
            continue
        if key != _record_key(retrieval):
            raise ValueError("Changed NeuronBridge retrieval observation.")
    old_queries = previous._queries if isinstance(previous, EvidenceValidation) else {}
    old_sessions = previous._sessions if isinstance(previous, EvidenceValidation) else {}
    queries = {}
    for key, record in metadata.get(QUERY_KEY, {}).items():
        if key in old_queries and _same_evidence(old_queries[key], record):
            queries[key] = old_queries[key]
        elif (key in old_queries and isinstance(record, dict) and old_queries[key]["version"] in (1, 3)
              and old_queries[key].keys() == record.keys()
              and all(_same_evidence(old_queries[key][field], record[field])
                      for field in record if field != "out_of_date")):
            # Auditing a source edit changes freshness, not retained image evidence.
            validate_query_freshness(record)
            queries[key] = _freeze_evidence(copy.deepcopy(record))
        else:
            snapshot = copy.deepcopy(record)
            validate_queries({QUERY_KEY: {key: snapshot}})
            queries[key] = _freeze_evidence(snapshot)
    validate_query_associations(metadata)
    validate_workspace(metadata)
    registry = metadata.get(SESSION_KEY, {})
    sessions, validated_results = {}, {}
    for key, payload in registry.items():
        if key in old_sessions and _same_evidence(old_sessions[key], payload):
            sessions[key] = old_sessions[key]
            results = previous._results[key]
        elif prevalidated_sessions is not None and key in prevalidated_sessions:
            snapshot = copy.deepcopy(payload)
            results = prevalidated_sessions[key]
            sessions[key] = _freeze_evidence(snapshot)
        else:
            snapshot = copy.deepcopy(payload)
            results = SearchResults.from_dict(snapshot)
            if snapshot.get("schema_version") == 1:
                snapshot = json.loads(json.dumps(results.to_dict(), allow_nan=False))
            sessions[key] = _freeze_evidence(snapshot)
        validated_results[key] = results
        if key != results.session.session_id:
            raise ValueError("NeuronBridge session key disagrees with its identity.")
    from .summary import EvidenceIndex
    same_sessions = (isinstance(previous, EvidenceValidation)
                     and sessions.keys() == old_sessions.keys()
                     and all(sessions[key] is old_sessions[key] for key in sessions))
    selected_metadatas = [record.object_metadata for record in objects] + list(origins)
    selected_key = _selected_evidence_key([
        mutable_metadata_copy(payload)
        for object_metadata in selected_metadatas
        for payload in _selected_payloads(object_metadata)
    ])
    if (same_sessions and isinstance(previous, EvidenceValidation)
            and selected_key == previous._selected_key):
        index = previous.index
    else:
        indexed_results = dict(validated_results)
        if origins and isinstance(previous, EvidenceValidation):
            for session_id, result in previous.index.results.items():
                indexed_results.setdefault(session_id, result)
        for object_metadata in selected_metadatas:
            for result in selected_results(object_metadata):
                current = indexed_results.get(result.session.session_id)
                indexed_results[result.session.session_id] = (
                    merge_search_results(current, result) if current is not None else result)
        index = EvidenceIndex(indexed_results.values())
    sources = index.sources
    from itertools import chain
    for metadata, is_remote in chain(((r.object_metadata, r.object_type.value == "remote_item") for r in objects),
                                     ((value, False) for value in origins)):
        evidence = metadata.get(OBJECT_KEY)
        if evidence is None:
            if is_remote and remote_from_metadata(metadata, project_metadata, sources=sources) is None:
                raise ValueError("A remote object requires a typed source descriptor.")
            continue
        if evidence.get("version") != 2 or not evidence.get("origin_id"):
            raise ValueError("Invalid NeuronBridge object evidence identity.")
        unavailable_session_ids = evidence.get("unavailable_session_ids", ())
        if ("unavailable_session_ids" in evidence
                and (not isinstance(unavailable_session_ids, list)
                     or any(not isinstance(key, str) or not key
                            for key in unavailable_session_ids))):
            raise ValueError("Invalid unavailable NeuronBridge session references.")
        unavailable_occurrences = evidence.get("unavailable_occurrences", ())
        if ("unavailable_occurrences" in evidence
                and (not isinstance(unavailable_occurrences, list)
                     or any(not isinstance(ref, dict)
                            or set(ref) != {"session_id", "occurrence_id"}
                            or any(not isinstance(ref.get(key), str) or not ref[key]
                                   for key in ("session_id", "occurrence_id"))
                            for ref in unavailable_occurrences))):
            raise ValueError("Invalid unavailable NeuronBridge occurrence references.")
        if set(unavailable_session_ids).intersection(registry):
            raise ValueError(
                "Unavailable NeuronBridge session evidence cannot also be complete."
            )
        remote = remote_from_metadata(metadata, project_metadata, sources=sources)
        if (is_remote and remote is None
                and not unavailable_occurrences and not unavailable_session_ids):
            raise ValueError("A remote object requires a typed source descriptor.")
        if any(key not in registry for key in evidence.get("session_ids", ())):
            raise ValueError("Missing NeuronBridge session evidence.")
        direct_refs = {(payload["session"]["session_id"], occurrence["occurrence_id"])
                       for payload in evidence.get("selected_results", ())
                       for occurrence in payload.get("occurrences", ())}
        for ref in match_references(metadata):
            match = index.matches.get((ref["session_id"], ref["occurrence_id"]))
            if match is None:
                raise ValueError(f"Missing NeuronBridge match occurrence {ref['occurrence_id']}.")
            occurrence = match.occurrence
            comparable_sources = [remote.source] if remote is not None else []
            if remote is not None and remote.resolution.get("selected_asset"):
                selected_asset = remote.selected_asset
                comparable_sources.append(
                    replace(
                        remote.source,
                        assets=tuple(
                            asset
                            for asset in remote.source.assets
                            if asset != selected_asset
                        ),
                    )
                )
            if (remote is not None and (ref["session_id"], ref["occurrence_id"]) in direct_refs
                    and not any(
                        source == occurrence.target
                        or same_source_identity(source, occurrence.target)
                        for source in comparable_sources
                    )):
                raise ValueError("NeuronBridge remote source disagrees with original match evidence.")
        for parent in evidence.get("derived_from", ()):
            remote_from_metadata({OBJECT_KEY: parent}, project_metadata, sources=sources)
    return EvidenceValidation(queries, sessions, validated_results, index, selected_key)


def copy_origin(metadata, *, derived=False, operation=None):
    """Copy a source result, or record ancestry without assigning it a new score."""
    result = copy.deepcopy(dict(metadata or {}))
    source = result.get(OBJECT_KEY)
    if not source:
        return result
    if not derived:
        return result
    parents = copy.deepcopy(source.get("derived_from", []))
    # An already-derived origin contributes its original evidence references,
    # not another empty intermediate origin on every edit.
    if (not source.get("derived_from") or source.get("selected_results")
            or source.get("remote_source") or source.get("source_ref")):
        parents.append({
            "origin_id": source["origin_id"],
            "selected_results": copy.deepcopy(source.get("selected_results", [])),
            "operation": operation,
            "remote_source": copy.deepcopy(source.get("remote_source")),
            **{key: copy.deepcopy(source[key]) for key in (
                "source_ref", "source_observation_ref", "resolution_ref", "retrieval_ref", "source_problems", "source_decision",
                "source_discrepancies", "channel_binding"
            ) if key in source},
        })
    parents = list({json.dumps(parent, sort_keys=True): parent for parent in parents}.values())
    result[OBJECT_KEY] = {
        "version": 2, "origin_id": str(uuid4()), "selected_results": [],
        "derived_from": parents,
    }
    return result


def _upgrade_object_origin(
    origin,
    sessions,
    *,
    tolerate_unavailable=False,
    canonicalize_selected=False,
):
    """Convert session-dependent v1 origins to direct bounded evidence."""
    result = mutable_metadata_copy(origin)
    parents = [_upgrade_object_origin(
        parent,
        sessions,
        tolerate_unavailable=tolerate_unavailable,
        canonicalize_selected=canonicalize_selected,
    )
               for parent in result.get("derived_from", ())]
    if parents:
        result["derived_from"] = parents
    if canonicalize_selected and "selected_results" in result:
        result["selected_results"] = [
            SearchResults.from_dict(payload).to_dict()
            for payload in result.get("selected_results", ())
        ]
    unavailable_sessions = set(result.get("unavailable_session_ids", ()))
    if tolerate_unavailable:
        # A retained absence marker records an earlier failed recovery.  It
        # cannot override matching evidence that this recovery has accepted.
        unavailable_sessions.difference_update(sessions)
    retained_sessions = list(result.get("session_ids", ()))
    if tolerate_unavailable and retained_sessions:
        available = [key for key in retained_sessions if key in sessions]
        unavailable_sessions.update(key for key in retained_sessions if key not in sessions)
        if available:
            result["session_ids"] = available
        else:
            result.pop("session_ids", None)
    unavailable_occurrences = list(result.get("unavailable_occurrences", ()))
    if result.get("version") == 1:
        refs = list(result.pop("matches", ()))
        source_ref = result.pop("source_ref", None)
        if source_ref and source_ref not in refs:
            refs.append(source_ref)
        payloads = list(result.get("selected_results", ()))
        by_session = {}
        for ref in refs:
            by_session.setdefault(ref["session_id"], set()).add(ref["occurrence_id"])
        for session_id, occurrence_ids in by_session.items():
            payload = sessions.get(session_id)
            if payload is None:
                unavailable_occurrences.extend(
                    {"session_id": session_id, "occurrence_id": occurrence_id}
                    for occurrence_id in sorted(occurrence_ids))
                continue
            parsed = SearchResults.from_dict(payload)
            available_ids = {item.occurrence_id for item in parsed.occurrences}
            missing_ids = occurrence_ids.difference(available_ids)
            if missing_ids and not tolerate_unavailable:
                raise ValueError("Missing originating NeuronBridge hit evidence.")
            unavailable_occurrences.extend(
                {"session_id": session_id, "occurrence_id": occurrence_id}
                for occurrence_id in sorted(missing_ids)
            )
            included_ids = occurrence_ids.intersection(available_ids)
            if included_ids:
                payloads = _merge_selected_payloads(
                    payloads, scoped_search_results(parsed, included_ids),
                )
        result["selected_results"] = payloads
        result["version"] = 2
    elif tolerate_unavailable and result.get("source_ref"):
        ref = result["source_ref"]
        available_refs = {
            (payload["session"]["session_id"], occurrence["occurrence_id"])
            for payload in result.get("selected_results", ())
            for occurrence in payload.get("occurrences", ())
        }
        session = sessions.get(ref.get("session_id"))
        if session is not None:
            parsed = SearchResults.from_dict(session)
            available_refs.update(
                (parsed.session.session_id, occurrence.occurrence_id)
                for occurrence in parsed.occurrences
            )
        key = (ref.get("session_id"), ref.get("occurrence_id"))
        if key not in available_refs:
            unavailable_occurrences.append({
                "session_id": str(ref.get("session_id") or ""),
                "occurrence_id": str(ref.get("occurrence_id") or ""),
            })
            result.pop("source_ref", None)
    if unavailable_sessions:
        result["unavailable_session_ids"] = sorted(unavailable_sessions)
    else:
        result.pop("unavailable_session_ids", None)
    if unavailable_occurrences:
        unique = {
            (str(ref.get("session_id") or ""), str(ref.get("occurrence_id") or ""))
            for ref in unavailable_occurrences
        }
        result["unavailable_occurrences"] = [
            {"session_id": session_id, "occurrence_id": occurrence_id}
            for session_id, occurrence_id in sorted(unique)
        ]
    return result


def _retained_session_ids(objects):
    retained = set()
    for record in objects:
        evidence = record.object_metadata.get(OBJECT_KEY, {})
        retained.update(evidence.get("session_ids", ()))
    return retained


def extract_embedded_sessions(
    objects,
    metadata,
    *,
    tolerate_unavailable=False,
    validated_sessions=None,
    canonicalize_selected=False,
):
    """Flat CSV embeds each referenced session once in an object metadata cell."""
    from dataclasses import replace
    from .query import QUERY_KEY, ASSOCIATION_KEY, merge_query_metadata

    if any(isinstance(record, dict) and "retrieved_at" in record.get("facts", {})
           for record in metadata.get(RETRIEVAL_KEY, {}).values()):
        metadata = dict(metadata)
        metadata[RETRIEVAL_KEY] = {
            key: {
                "asset_identity": mutable_metadata_copy(record["asset_identity"]),
                "facts": {name: mutable_metadata_copy(value)
                          for name, value in record["facts"].items()
                          if name != "retrieved_at"},
            }
            for key, record in metadata[RETRIEVAL_KEY].items()
        }
    unavailable_sessions = set()
    unavailable_occurrence_refs = []
    if tolerate_unavailable and metadata.get(SESSION_KEY):
        retained = {}
        for key, payload in metadata[SESSION_KEY].items():
            try:
                parsed = project_search_results(SearchResults.from_dict(payload))
                if key != parsed.session.session_id:
                    raise ValueError("session key disagrees with its identity")
                retained[key] = (
                    payload if _is_project_session_payload(payload)
                    else json.loads(json.dumps(parsed.to_dict(), allow_nan=False))
                )
                if validated_sessions is not None:
                    validated_sessions[str(key)] = parsed
            except (KeyError, TypeError, ValueError):
                unavailable_sessions.add(str(key))
        metadata = dict(metadata)
        if retained:
            metadata[SESSION_KEY] = retained
        else:
            metadata.pop(SESSION_KEY, None)
    project_session_ids = set(metadata.get(SESSION_KEY, {}))
    rows = []
    for record in objects:
        values = dict(record.object_metadata)
        embedded = values.pop(SESSION_KEY, None)
        remote_records = {key: values.pop(key) for key in REMOTE_KEYS if key in values}
        if remote_records:
            metadata = merge_remote_records(metadata, remote_records)
        queries = {field: values.pop(field) for field in (QUERY_KEY, ASSOCIATION_KEY) if field in values}
        if embedded:
            if tolerate_unavailable:
                for key, payload in embedded.items():
                    if key in unavailable_sessions:
                        continue
                    try:
                        metadata = merge_sessions(metadata, {key: payload})
                    except (KeyError, TypeError, ValueError):
                        registry = dict(metadata.get(SESSION_KEY, {}))
                        registry.pop(key, None)
                        metadata = dict(metadata)
                        if registry:
                            metadata[SESSION_KEY] = registry
                        else:
                            metadata.pop(SESSION_KEY, None)
                        unavailable_sessions.add(str(key))
            else:
                metadata = merge_sessions(metadata, embedded)
        if queries:
            metadata = merge_query_metadata(metadata, queries)
        if embedded or queries or remote_records:
            record = replace(record, object_metadata=values)
        rows.append(record)
    upgraded = []
    for record in rows:
        values = mutable_metadata_copy(record.object_metadata)
        if OBJECT_KEY in values:
            values[OBJECT_KEY] = _upgrade_object_origin(
                values[OBJECT_KEY], metadata.get(SESSION_KEY, {}),
                tolerate_unavailable=tolerate_unavailable,
                canonicalize_selected=canonicalize_selected,
            )
            if tolerate_unavailable:
                unavailable_sessions.update(
                    values[OBJECT_KEY].get("unavailable_session_ids", ())
                )
                unavailable_occurrence_refs.extend(
                    values[OBJECT_KEY].get("unavailable_occurrences", ())
                )
            record = replace(record, object_metadata=values)
        upgraded.append(record)
    if tolerate_unavailable and (unavailable_sessions or unavailable_occurrence_refs):
        registry = {
            key: value for key, value in metadata.get(SESSION_KEY, {}).items()
            if key not in unavailable_sessions
        }
        metadata = dict(metadata)
        if registry:
            metadata[SESSION_KEY] = registry
        else:
            metadata.pop(SESSION_KEY, None)
    sessions = metadata.get(SESSION_KEY, {})
    retained = _retained_session_ids(upgraded)
    missing = retained.difference(sessions)
    if missing:
        if not tolerate_unavailable:
            raise ValueError(f"Missing explicitly retained NeuronBridge session {sorted(missing)[0]}.")
        unavailable_sessions.update(missing)
        upgraded = tuple(
            replace(record, object_metadata={
                **mutable_metadata_copy(record.object_metadata),
                OBJECT_KEY: _upgrade_object_origin(
                    record.object_metadata[OBJECT_KEY], sessions,
                    tolerate_unavailable=True,
                    canonicalize_selected=canonicalize_selected,
                ),
            }) if OBJECT_KEY in record.object_metadata else record
            for record in upgraded
        )
    if sessions:
        metadata = dict(metadata)
        keep = project_session_ids | retained
        canonical = {
            key: (payload if _is_project_session_payload(payload)
                  else json.loads(json.dumps(
                      project_search_results(SearchResults.from_dict(payload)).to_dict(),
                      allow_nan=False,
                  )))
            for key, payload in sessions.items() if key in keep
        }
        if canonical:
            metadata[SESSION_KEY] = canonical
        else:
            metadata.pop(SESSION_KEY, None)
    if tolerate_unavailable and (unavailable_sessions or unavailable_occurrence_refs):
        from .query import ASSOCIATION_KEY
        associations = {
            key: value for key, value in metadata.get(ASSOCIATION_KEY, {}).items()
            if key not in unavailable_sessions
        }
        if associations != metadata.get(ASSOCIATION_KEY, {}):
            metadata = dict(metadata)
            if associations:
                metadata[ASSOCIATION_KEY] = associations
            else:
                metadata.pop(ASSOCIATION_KEY, None)
        metadata = _record_unavailable_sessions(
            metadata, unavailable_sessions, unavailable_occurrence_refs,
        )
    return tuple(upgraded), metadata
