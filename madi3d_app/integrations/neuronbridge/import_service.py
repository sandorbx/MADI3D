"""Lossless import planning and explicit, conservative historical reattachment."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from .csv_parser import read_csv_header, read_neuronbridge_csv
from .evidence import (
    OBJECT_KEY, SESSION_KEY, merge_sessions, object_metadata_for_match, reference,
)
from .remote import remote_from_metadata, same_source_identity
from .records import SearchResults


def is_results_csv(path):
    """Classify from the header without reading or retaining project rows."""
    headers = {" ".join(h.split()).casefold() for h in read_csv_header(path)}
    if {"id", "parentid", "type"} <= headers:
        return False
    if headers & {"neuron id", "line name", "line name / neuron id", "neuronbridge image id"}:
        return True
    return "image id" in headers and bool(headers & {"library", "target type"})


def detect_results_csv(path):
    """Run the lossless parser only for headers identifying NeuronBridge results."""
    return read_neuronbridge_csv(path) if is_results_csv(path) else None


def import_diagnostics(results):
    messages = []
    for diagnostic in results.all_diagnostics:
        prefix = f"Row {diagnostic.row_index + 1}: " if diagnostic.row_index is not None else ""
        messages.append(prefix + diagnostic.message)
    return tuple(messages)


def match_name(occurrence):
    source = occurrence.target
    name = source.published_name or source.image.image_id or "Unresolved result"
    channel = f" / channel {source.channel.selector}" if source.channel else ""
    return f"{name}{channel} / result {occurrence.row_index + 1}"


def plan_result_rows(results, *, occurrence_ids=None, cancel_check=None):
    """Plan only explicitly chosen rows without changing the complete session."""
    selected = None if occurrence_ids is None else set(occurrence_ids)
    if selected is not None and selected.difference(o.occurrence_id for o in results.occurrences):
        raise ValueError("Selected occurrences do not belong to this saved session.")
    rows = []
    for occurrence in results.occurrences:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("NeuronBridge import cancelled.")
        if selected is not None and occurrence.occurrence_id not in selected:
            continue
        rows.append((match_name(occurrence), object_metadata_for_match(results, occurrence)))
    return tuple(rows)


@dataclass(frozen=True)
class ReattachmentProposal:
    row_id: int
    occurrence_ids: tuple[str, ...]
    expected_metadata: dict


@dataclass(frozen=True)
class ReattachmentPlan:
    results: SearchResults
    proposals: tuple[ReattachmentProposal, ...]
    conflicts: tuple[str, ...]


def plan_reattachment(document, results, *, identities=None):
    """Dry-run only. Caller-supplied identities must come from verified evidence."""
    identities = identities or {}
    proposals, conflicts = [], []
    for row in document.objects:
        if row.object_metadata.get(OBJECT_KEY, {}).get("derived_from"):
            conflicts.append(f"Object {row.row_id}: derived geometry keeps its parent references; unchanged.")
            continue
        remote = remote_from_metadata(row.object_metadata)
        identity = remote.source if remote else identities.get(row.row_id)
        if identity is None:
            conflicts.append(f"Object {row.row_id}: no reliable source identity; unchanged.")
            continue
        matches = [o for o in results.occurrences if same_source_identity(identity, o.target)]
        if len(matches) > 1 and results.session.csv_provenance:
            original_rows = set()
            for ref in row.object_metadata.get(OBJECT_KEY, {}).get("matches", ()):
                payload = document.project_metadata.get(SESSION_KEY, {}).get(ref["session_id"])
                if not payload:
                    continue
                previous = SearchResults.from_dict(payload)
                provenance = previous.session.csv_provenance
                if provenance and provenance.checksum_sha256 == results.session.csv_provenance.checksum_sha256:
                    original_rows.update(o.row_index for o in previous.occurrences if o.occurrence_id == ref["occurrence_id"])
            if original_rows:
                matches = [o for o in matches if o.row_index in original_rows]
        if len(matches) != 1:
            conflicts.append(f"Object {row.row_id}: no unique target identity; unchanged.")
            continue
        proposals.append(ReattachmentProposal(
            row.row_id, tuple(o.occurrence_id for o in matches),
            copy.deepcopy(dict(row.object_metadata)),
        ))
    return ReattachmentPlan(results, tuple(proposals), tuple(conflicts))


def apply_reattachment(document, plan):
    """Apply exactly a reviewed dry run, rejecting stale metadata atomically."""
    by_id = {row.row_id: row for row in document.objects}
    occurrences = {o.occurrence_id: o for o in plan.results.occurrences}
    changes = {}
    for proposal in plan.proposals:
        row = by_id.get(proposal.row_id)
        if row is None or dict(row.object_metadata) != proposal.expected_metadata:
            raise ValueError("Reattachment proposal is stale; run the dry run again.")
        matched = [occurrences[key] for key in proposal.occurrence_ids]
        metadata = copy.deepcopy(dict(row.object_metadata))
        if OBJECT_KEY not in metadata:
            metadata.update(object_metadata_for_match(plan.results, matched[0]))
        evidence = metadata[OBJECT_KEY]
        refs = evidence.setdefault("matches", [])
        for occurrence in matched:
            ref = reference(occurrence)
            if ref not in refs:
                refs.append(ref)
        changes[row.row_id] = replace(row, object_metadata=metadata)
    if not changes:
        return document
    metadata = document.project_metadata
    if changes:
        metadata = merge_sessions(metadata, {
            plan.results.session.session_id: plan.results.to_dict(),
        })
    return replace(document, objects=tuple(changes.get(r.row_id, r) for r in document.objects), project_metadata=metadata)
