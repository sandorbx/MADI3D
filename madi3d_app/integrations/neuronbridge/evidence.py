"""NB-01A evidence ownership in existing project and object metadata contracts."""
from __future__ import annotations

import copy
import json
from uuid import uuid4

from .records import SearchResults
from .remote import RemoteSource, remote_from_metadata, same_source_identity

SESSION_KEY = "neuronbridge_sessions"
OBJECT_KEY = "neuronbridge"


def merge_sessions(metadata, sessions):
    result = copy.deepcopy(dict(metadata or {}))
    if not sessions:
        return result
    registry = dict(result.get(SESSION_KEY, {}))
    result[SESSION_KEY] = registry
    for key, payload in sessions.items():
        parsed = SearchResults.from_dict(payload)
        if key != parsed.session.session_id:
            raise ValueError("NeuronBridge session key disagrees with its identity.")
        if key in registry and SearchResults.from_dict(registry[key]) != parsed:
            raise ValueError(f"Conflicting NeuronBridge evidence for session {key}.")
        registry[key] = json.loads(json.dumps(parsed.to_dict(), allow_nan=False))
    return result


def reference(occurrence):
    return {"session_id": occurrence.session_id, "occurrence_id": occurrence.occurrence_id}


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
        "version": 1, "origin_id": str(uuid4()),
        "matches": [reference(occurrence)], "remote_source": remote.to_dict(),
        "diagnostics": diagnostics,
    }}


def match_references(metadata):
    evidence = (metadata or {}).get(OBJECT_KEY, {})
    refs = list(evidence.get("matches", ()))
    for parent in evidence.get("derived_from", ()):
        refs.extend(parent.get("matches", ()))
    return refs


def referenced_sessions(metadata, project_metadata, *, exclude=()):
    registry = (project_metadata or {}).get(SESSION_KEY, {})
    result = {}
    session_ids = list((metadata or {}).get(OBJECT_KEY, {}).get("session_ids", ()))
    session_ids.extend(ref["session_id"] for ref in match_references(metadata))
    for key in session_ids:
        if key not in registry:
            raise ValueError(f"Missing NeuronBridge search session {key}.")
        if key not in exclude and key not in result:
            result[key] = copy.deepcopy(registry[key])
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

    def __init__(self, queries, sessions, results, index):
        self._queries = _freeze_evidence(queries)
        self._sessions = _freeze_evidence(sessions)
        self._results = results
        self.index = index

    def freeze_metadata(self, metadata):
        from .query import QUERY_KEY
        result = dict(metadata or {})
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


def validate_evidence(objects, project_metadata, *, previous=None):
    from .query import QUERY_KEY, validate_queries, validate_query_associations, validate_query_freshness
    from .workspace import validate_workspace

    metadata = project_metadata or {}
    old_queries = previous._queries if isinstance(previous, EvidenceValidation) else {}
    old_sessions = previous._sessions if isinstance(previous, EvidenceValidation) else {}
    queries = {}
    for key, record in metadata.get(QUERY_KEY, {}).items():
        if key in old_queries and _same_evidence(old_queries[key], record):
            queries[key] = old_queries[key]
        elif (key in old_queries and isinstance(record, dict) and old_queries[key]["version"] == 1
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
    occurrences = {}
    sessions, validated_results = {}, {}
    for key, payload in registry.items():
        if key in old_sessions and _same_evidence(old_sessions[key], payload):
            sessions[key] = old_sessions[key]
            results = previous._results[key]
        else:
            snapshot = copy.deepcopy(payload)
            results = SearchResults.from_dict(snapshot)
            sessions[key] = _freeze_evidence(snapshot)
        validated_results[key] = results
        if key != results.session.session_id:
            raise ValueError("NeuronBridge session key disagrees with its identity.")
        occurrences[key] = {o.occurrence_id: o for o in results.occurrences}
    for record in objects:
        metadata = record.object_metadata
        evidence = metadata.get(OBJECT_KEY)
        if record.object_type.value == "remote_item" and remote_from_metadata(metadata) is None:
            raise ValueError("A remote object requires a typed source descriptor.")
        if evidence is None:
            continue
        if evidence.get("version") != 1 or not evidence.get("origin_id"):
            raise ValueError("Invalid NeuronBridge object evidence identity.")
        remote = remote_from_metadata(metadata)
        if any(key not in registry for key in evidence.get("session_ids", ())):
            raise ValueError("Missing NeuronBridge session evidence.")
        for ref in match_references(metadata):
            occurrence = occurrences.get(ref["session_id"], {}).get(ref["occurrence_id"])
            if occurrence is None:
                raise ValueError(f"Missing NeuronBridge match occurrence {ref['occurrence_id']}.")
            if (remote is not None and ref in evidence.get("matches", ())
                    and remote.source != occurrence.target
                    and not same_source_identity(remote.source, occurrence.target)):
                raise ValueError("NeuronBridge remote source disagrees with original match evidence.")
    from .summary import EvidenceIndex

    same_sessions = (isinstance(previous, EvidenceValidation)
                     and sessions.keys() == old_sessions.keys()
                     and all(sessions[key] is old_sessions[key] for key in sessions))
    index = previous.index if same_sessions else EvidenceIndex(validated_results.values())
    return EvidenceValidation(queries, sessions, validated_results, index)


def copy_origin(metadata, *, derived=False, operation=None):
    """Copy a source result, or record ancestry without assigning it a new score."""
    result = copy.deepcopy(dict(metadata or {}))
    source = result.get(OBJECT_KEY)
    if not source:
        return result
    if not derived:
        return result
    parents = copy.deepcopy(source.get("derived_from", []))
    parents.append({
        "origin_id": source["origin_id"],
        "matches": copy.deepcopy(source.get("matches", [])),
        "operation": operation,
        "remote_source": copy.deepcopy(source.get("remote_source")),
    })
    result[OBJECT_KEY] = {
        "version": 1, "origin_id": str(uuid4()), "matches": [],
        "derived_from": parents,
    }
    return result


def extract_embedded_sessions(objects, metadata):
    """Flat CSV embeds each referenced session once in an object metadata cell."""
    from dataclasses import replace
    from .query import QUERY_KEY, ASSOCIATION_KEY, merge_query_metadata

    rows = []
    for record in objects:
        values = dict(record.object_metadata)
        embedded = values.pop(SESSION_KEY, None)
        queries = {field: values.pop(field) for field in (QUERY_KEY, ASSOCIATION_KEY) if field in values}
        if embedded:
            metadata = merge_sessions(metadata, embedded)
        if queries:
            metadata = merge_query_metadata(metadata, queries)
        if embedded or queries:
            record = replace(record, object_metadata=values)
        rows.append(record)
    return tuple(rows), metadata
