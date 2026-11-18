"""Deterministic, offline schema-1 import at the shared record boundary.

Historical identities and typed values are authoritative. This module does not
call a parser, scorer, library, or remote API to reinterpret an old result.
"""
from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace

from .records import (
    Diagnostic, HitMetric, MatchOccurrence, NeuronBridgeRecordError,
    SearchResults, SearchSession, SourceParameter, _json_value,
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _retain(external, value):
    key = "schema1:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
    external.setdefault(key, value)
    return key


def migrate_schema1(payload):
    """Return current records without modifying the caller's historical payload."""
    _json_value(payload)
    data = copy.deepcopy(payload)
    allowed = {"session", "occurrences", "diagnostics", "csv_headers", "csv_bytes_base64", "csv_parse_complete"}
    if set(data) - allowed:
        raise NeuronBridgeRecordError("Unexpected schema-1 result fields.")
    session_data = data["session"]
    if set(session_data) - {"session_id", "source_kind", "neuronbridge_data_version", "query_reference",
                            "query_source", "parameters", "csv_provenance"}:
        raise NeuronBridgeRecordError("Unexpected schema-1 session fields.")
    old_session = SearchSession.from_dict(session_data)
    old_hits = tuple(MatchOccurrence.from_dict(hit) for hit in data.get("occurrences", ()))
    diagnostics = tuple(Diagnostic(**d) for d in data.get("diagnostics", ()))
    # Validate old identities/order before converting any measurements.
    if any(set(hit) - {"occurrence_id", "session_id", "target", "row_index", "fields", "diagnostics",
                       "line_start", "line_end"} for hit in data.get("occurrences", ())):
        raise NeuronBridgeRecordError("Unexpected schema-1 occurrence fields.")
    encoded = data.get("csv_bytes_base64")
    try:
        raw = base64.b64decode(encoded, validate=True) if encoded is not None else None
    except (ValueError, TypeError, binascii.Error) as exc:
        raise NeuronBridgeRecordError("Invalid schema-1 CSV base64 evidence.") from exc
    migration = {"source_schema": 1, "revision": "schema1-to-schema2-v1"}
    if old_session.source_kind == "imported_csv":
        provenance = replace(old_session.csv_provenance, parser_revision="unrecorded-schema-1",
                             interpretation_revision="retained-schema-1")
        old_session = replace(old_session, csv_provenance=provenance)
    if old_session.source_kind == "imported_csv" and raw is not None:
        session = replace(old_session,
                          context={"backend": "csv-import", "migration": migration})
        return SearchResults(session, old_hits, diagnostics, tuple(data.get("csv_headers", ())), raw,
                             data.get("csv_parse_complete"))
    if raw is not None or data.get("csv_headers") or data.get("csv_parse_complete") is not None:
        raise NeuronBridgeRecordError("Native schema-1 results cannot contain CSV source bytes.")

    if old_session.source_kind == "imported_csv":
        diagnostics += (Diagnostic("missing_csv_source_bytes",
            "This older result did not retain the original CSV bytes. Its saved fields and interpretations remain available."),)
    parameters = list(old_session.parameters)
    def take(name):
        found = [p for p in parameters if p.name == name]
        if len(found) > 1:
            raise NeuronBridgeRecordError(f"Ambiguous schema-1 {name} context.")
        if found:
            parameters.remove(found[0])
            return found[0].value
        return None

    external, context, shared_diagnostics = {}, {"migration": migration}, ()
    local = take("local_search") if old_session.source_kind == "custom_query" else None
    public = take("public_api_evidence") if old_session.source_kind == "precomputed" else None
    if local is not None:
        context.update(local)
        context["implementation_revision"] = "unrecorded-schema-1"
        supplied = context.pop("parameters")
        # Retain supplementary session parameters separately from execution parameters.
        if parameters:
            external["schema1_session_parameters"] = [asdict(p) for p in parameters]
        parameters = [SourceParameter(name, value) for name, value in supplied.items()]
        for old_key, new_key in (("catalog_evidence", "catalog"), ("manifest_source", "manifest")):
            if old_key in context:
                external[new_key] = context.pop(old_key)
        warnings = context.pop("query_warnings", [])
        shared_diagnostics = tuple(d for d in diagnostics if d.code == "query_mapping_warning")
        diagnostics = tuple(d for d in diagnostics if d.code != "query_mapping_warning")
        for text in warnings:
            if not any(d.message == text for d in shared_diagnostics):
                shared_diagnostics += (Diagnostic("query_mapping_warning", text),)
    elif public is not None:
        external.update(public)
        retrieval = take("retrieval")
        algorithm = take("algorithm_version")
        context.update(backend="neuronbridge-public-s3", algorithm=algorithm,
                       implementation_revision=public.get("adapter_version"),
                       reference_revision=public.get("reference_client_commit"))
        if retrieval is not None:
            context["retrieval"] = retrieval
        context["completed_at"] = public.get("result_response", {}).get("retrieved_at")

    columns, column_ids, observations, hits = [], {}, {}, []
    for original in old_hits:
        values, references = {}, {}
        remaining = list(original.fields)
        if local is not None:
            mapping = {
                "Local rank": ("rank", "rank"),
                "Local matching pixels": ("matched_pixels", "matched_pixels"),
                "Local query overlap fraction": ("overlap", "score"),
                "Mirrored comparison": ("mirrored", "mirror"),
                "Library member": ("member_id", "unknown"),
            }
            for header, (name, role) in mapping.items():
                candidates = [f for f in original.fields if f.header == header and f.role == role]
                if len(candidates) == 1:
                    values[name] = candidates[0].value
            for header, scope in (("Identity metadata evidence", "identity"),
                                  ("Search image source evidence", "search_image")):
                candidates = [f for f in original.fields if f.header == header]
                if len(candidates) != 1 or not isinstance(candidates[0].value, str):
                    continue
                cell = candidates[0]
                try:
                    observed = json.loads(cell.value)
                    _json_value(observed)
                except (ValueError, TypeError):
                    continue
                if not isinstance(observed, dict):
                    continue
                references[scope] = _retain(external, observed)
                # Standard synthetic JSON columns become structured observations once.
                if cell.raw_text == cell.value == _canonical(observed):
                    remaining.remove(cell)
            shifts = [f for f in original.fields if f.header == "Query shift before mirroring (XY pixels)"]
            if len(shifts) == 1 and isinstance(shifts[0].value, str):
                try:
                    shift = json.loads(shifts[0].value)
                except ValueError:
                    shift = None
                if isinstance(shift, list) and len(shift) == 2 and all(type(v) is int for v in shift):
                    values["shift"] = shift
            hashes = {}
            for header, name in (("Search image SHA-256", "search_image"), ("Decoded RGB SHA-256", "decoded_rgb")):
                candidates = [f for f in original.fields if f.header == header]
                if len(candidates) == 1 and isinstance(candidates[0].value, str):
                    digest = candidates[0].value
                    if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                        hashes[name] = digest
            values["input_hashes"] = hashes
            if references:
                values["evidence_ref"] = _retain(external, {"references": references})
        else:
            scores = [f for f in original.fields if f.role == "score"]
            names = Counter(f.header for f in scores)
            values["metrics"] = tuple(HitMetric(
                f.header if f.header and names[f.header] == 1 else f"{f.header or 'Score'} [column {f.column_index + 1}]",
                f.value) for f in scores)
            for role, name in (("rank", "rank"), ("matched_pixels", "matched_pixels"), ("mirror", "mirrored")):
                candidates = [f for f in original.fields if f.role == role]
                if len(candidates) == 1:
                    values[name] = candidates[0].value
            if public is not None and "result_response" in public:
                values.update(evidence_ref="result_response", evidence_index=original.row_index)

        # Preserve unknown/duplicate columns and the old interpretation independently
        # of any API response. Shared descriptors, one raw string, sparse typed values.
        cells, typed, ids = [], {}, []
        for cell in remaining:
            descriptor = (cell.column_index, cell.header, cell.role)
            if descriptor not in column_ids:
                column_ids[descriptor] = len(columns)
                columns.append(dict(column_index=cell.column_index, header=cell.header, role=cell.role))
            ids.append(column_ids[descriptor])
            cells.append(cell.raw_text)
            if cell.value != cell.raw_text or type(cell.value) is not type(cell.raw_text):
                typed[str(len(cells) - 1)] = cell.value
        if cells:
            observations[original.occurrence_id] = dict(columns=ids, cells=cells, values=typed)
        hits.append(replace(original, fields=(), **values))
    if observations:
        external["schema1_fields"] = dict(columns=columns, occurrences=observations)
    session = replace(old_session, parameters=tuple(parameters), context=context,
                      external_evidence=external, diagnostics=shared_diagnostics)
    return SearchResults(normalize_historical_session(session), tuple(hits), diagnostics)


def normalize_historical_session(session):
    """Convert retired population locators to bounded, non-executable facts.

    Called only by serialized ingestion, after validating the old record. No
    library files, managers, repair routes or supplementary images are read.
    """
    context = session.context
    if context.get("backend") != "madi3d-local" or context.get("schema_version") not in (1, 2):
        return session
    context = copy.deepcopy(context)
    labels = {
        "snapshot_id": "snapshot identifier", "source_snapshot_id": "source snapshot identifier",
        "inventory_sha256": "inventory SHA-256", "source_inventory_sha256": "source inventory SHA-256",
        "manifest_sha256": "manifest SHA-256", "membership_sha256": "population membership SHA-256",
    }
    facts = [{"kind": label, "value": context.pop(name)}
             for name, label in labels.items() if context.get(name) is not None]
    if len(_canonical(facts).encode("utf-8")) > 32 * 1024:
        raise NeuronBridgeRecordError("Historical population provenance exceeds its compact evidence bound.")
    context["historical_provenance"] = {"source_evidence_version": context["schema_version"],
                                        "population": facts}
    for name in labels:
        context.pop(name, None)
    context["schema_version"] = 3
    return replace(session, context=context)
