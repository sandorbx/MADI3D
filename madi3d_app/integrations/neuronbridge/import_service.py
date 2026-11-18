"""Lossless import planning and explicit, conservative historical reattachment."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import sqlite3

from madi3d_app.integrations.flywire.catalog import Catalog
from madi3d_app.integrations.flywire.catalog_review import unique_fafb_cell_types
from madi3d_app.integrations.flywire.records import ReleaseContext, validate_root_id

from .archive_catalog import FLYWIRE_FAFB_DISTRIBUTION, HEMIBRAIN_ARCHIVE, VERIFIED_LIBRARIES, candidate_identity
from .comments import CommentChange, CommentPlan
from .csv_parser import read_csv_header, read_neuronbridge_csv
from .evidence import (
    OBJECT_KEY, RESULT_FLIP_X, RESULT_FLIP_X_PENDING_FAFB, RESULT_FLIP_X_PENDING_SIDES,
    _merge_selected_payloads, object_metadata_for_match, scoped_search_results, selected_results,
)
from .remote import remote_from_metadata, same_source_identity
from .orientation import query_template_x_side
from .records import SearchResults
from .search_profiles import BRAIN
from .summary import ranking_value


def _brain_result_mesh(target):
    library = VERIFIED_LIBRARIES.get(target.library)
    image = target.image
    spaces = {"jrc2018u", BRAIN.alignment_space.casefold(), *(s.casefold() for s in BRAIN.aliases)}
    return (target.kind == "em" and library is not None
            and (target.library_release, BRAIN.profile_id) == (library[0], library[2])
            and (not image.anatomical_area or image.anatomical_area.strip().casefold() == "brain")
            and (not image.alignment_space or image.alignment_space.strip().casefold() in spaces))


def _is_fafb_v783(target):
    distribution = FLYWIRE_FAFB_DISTRIBUTION
    return (target.library, target.library_release) == (
        distribution.geometry_source_library, distribution.geometry_release)


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


def match_name(occurrence):
    source = occurrence.target
    name = (source.published_name or source.image.image_id
            or source.biological.neuron_id or source.biological.line_name or "Unresolved result")
    channel = f" / channel {source.channel.selector}" if source.channel else ""
    return f"{name}{channel} / result {occurrence.row_index + 1}"


def neuron_type_comment(session, occurrence):
    """Resolve Comment A from recorded type or retained Hemibrain member evidence."""
    biological = occurrence.target.biological
    if biological.neuron_type:
        return biological.neuron_type

    archive = HEMIBRAIN_ARCHIVE
    context = session.context
    if (context.get("backend") != "madi3d-local"
            or context.get("library_id") != archive.library_id
            or context.get("archive_id") != archive.archive_id
            or not occurrence.member_id):
        return ""
    try:
        member = candidate_identity(archive.archive_id, occurrence.member_id)
    except ValueError:
        return ""
    if member.body_id != biological.neuron_id:
        return ""

    # The archive label is scene text only. It does not establish a neuron type.
    label = member.anatomical_name.strip()
    if (not any(character.isalpha() for character in label)
            or label.casefold() in {"unknown", "unclassified", "untyped", "unnamed", "none", "na", "n/a"}):
        return ""
    return label


def _fafb_type_comments(session, occurrences, app_data, cancel_check):
    distribution = FLYWIRE_FAFB_DISTRIBUTION
    context = session.context
    library = (distribution.geometry_source_library, distribution.geometry_release)
    if (context.get("backend") != "madi3d-local"
            or context.get("library_id", distribution.library_id) != distribution.library_id
            or (context.get("library"), context.get("library_release")) != library):
        return {}

    roots = {}
    for occurrence in occurrences:
        target = occurrence.target
        if (target.kind != "em" or (target.library, target.library_release) != library
                or target.biological.neuron_type):
            continue
        try:
            roots[occurrence.occurrence_id] = validate_root_id(target.biological.neuron_id)
        except ValueError:
            continue
    if not roots:
        return {}

    catalog = Catalog.installed(app_data, context=ReleaseContext("fafb", "783"))
    try:
        # Local searches retain at most 5,000 matches. Lookup deduplicates
        # roots, bounds its payload and queries in chunks on one connection.
        snapshot = catalog.lookup(tuple(roots.values()), cancelled=cancel_check)
    except InterruptedError:
        raise
    except (OSError, sqlite3.Error, ValueError):
        # An unavailable catalog does not prevent lazy result publication.
        return {}
    types = unique_fafb_cell_types(snapshot)
    return {occurrence_id: types.get(root_id, "") for occurrence_id, root_id in roots.items()}


def _neuron_type_comments(session, occurrences, app_data, cancel_check):
    fafb_types = (_fafb_type_comments(session, occurrences, app_data, cancel_check)
                  if app_data else {})
    comments = {}
    for occurrence in occurrences:
        if cancel_check():
            raise InterruptedError("NeuronBridge annotation lookup cancelled.")
        comments[occurrence.occurrence_id] = (
            neuron_type_comment(session, occurrence) or fafb_types.get(occurrence.occurrence_id, ""))
    return comments


def plan_missing_neuron_types(targets, *, app_data, cancel_check):
    """Plan fills from detached (row key, name, retained matches) snapshots."""
    sessions = {}
    for _, _, matches in targets:
        for match in matches:
            session, occurrences = sessions.setdefault(match.session.session_id, (match.session, {}))
            occurrences[match.occurrence.occurrence_id] = match.occurrence
    annotations = {
        session_id: _neuron_type_comments(session, occurrences.values(), app_data, cancel_check)
        for session_id, (session, occurrences) in sessions.items()
    }
    changes, skipped = [], []
    for key, name, matches in targets:
        if cancel_check():
            raise InterruptedError("NeuronBridge annotation lookup cancelled.")
        values = {annotations[session_id][occurrence_id]
                  for session_id, occurrence_id in (match.key for match in matches)}
        if len(values) == 1 and "" not in values:
            changes.append(CommentChange(key, name, 1, "", values.pop()))
        else:
            skipped.append(name)
    return CommentPlan(tuple(changes), tuple(skipped))


def plan_result_rows(results, *, occurrence_ids=None, cancel_check=None, app_data=None, query_record=None):
    """Plan only explicitly chosen rows without changing the complete session."""
    cancelled = cancel_check or (lambda: False)
    selected = None if occurrence_ids is None else set(occurrence_ids)
    if selected is not None and selected.difference(o.occurrence_id for o in results.occurrences):
        raise ValueError("Selected occurrences do not belong to this saved session.")
    if cancelled():
        raise InterruptedError("NeuronBridge import cancelled.")
    occurrences = tuple(o for o in results.occurrences if selected is None or o.occurrence_id in selected)
    types = _neuron_type_comments(results.session, occurrences, app_data, cancelled)
    context = results.session.context
    brain_cdm = (context.get("backend") == "madi3d-local"
                 and context.get("search_profile") == BRAIN.profile_id)
    hemisphere_flip = None
    if (brain_cdm and query_record and query_record["query_id"] == results.session.query_reference
            and any(not _is_fafb_v783(o.target) and _brain_result_mesh(o.target) for o in occurrences)):
        side = query_template_x_side(query_record, context.get("query_png_sha256"))
        if side is not None:
            hemisphere_flip = next((state for state, value in RESULT_FLIP_X_PENDING_SIDES.items()
                                    if side == value), "managed")
    rows = []
    for occurrence in occurrences:
        if cancelled():
            raise InterruptedError("NeuronBridge import cancelled.")
        score = ranking_value(results.session, occurrence)
        comments = (
            types[occurrence.occurrence_id],
            str(score) if score is not None else "",
            "",
        )
        metadata = object_metadata_for_match(results, occurrence)
        if brain_cdm and _brain_result_mesh(occurrence.target):
            if _is_fafb_v783(occurrence.target):
                if results.session.source_kind == "custom_query":
                    # Object Info's Comparison only field uses the positive comparison.
                    # Unflipped rows still track manual Flip X changes in Comment C.
                    metadata[RESULT_FLIP_X] = (
                        RESULT_FLIP_X_PENDING_FAFB if occurrence.mirrored is False else "managed"
                    )
            elif hemisphere_flip:
                metadata[RESULT_FLIP_X] = hemisphere_flip
        rows.append((match_name(occurrence), metadata, comments))
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


def plan_reattachment(document, results, *, identities=None, cancel_check=None):
    """Dry-run only. Caller-supplied identities must come from verified evidence."""
    if results.csv_parse_complete is False:
        # An unread suffix might contain another match for an apparently unique
        # identity. Partial CSVs remain inspectable, but cannot prove attachment.
        raise ValueError("CSV parsing did not complete. Correct the CSV before attaching metadata.")
    identities = identities or {}
    proposals, conflicts = [], []
    for row in document.objects:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Metadata attachment cancelled.")
        if row.object_metadata.get(OBJECT_KEY, {}).get("derived_from"):
            conflicts.append(f"Object {row.row_id}: derived geometry keeps its parent references; unchanged.")
            continue
        remote = remote_from_metadata(row.object_metadata, document.project_metadata)
        identity = remote.source if remote else identities.get(row.row_id)
        if identity is None:
            conflicts.append(f"Object {row.row_id}: no reliable source identity; unchanged.")
            continue
        matches = []
        for occurrence in results.occurrences:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Metadata attachment cancelled.")
            if same_source_identity(identity, occurrence.target):
                matches.append(occurrence)
        if len(matches) > 1 and results.session.csv_provenance:
            original_rows = set()
            for previous in selected_results(row.object_metadata):
                provenance = previous.session.csv_provenance
                if provenance and provenance.checksum_sha256 == results.session.csv_provenance.checksum_sha256:
                    original_rows.update(o.row_index for o in previous.occurrences)
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
        for occurrence in matched:
            evidence["selected_results"] = _merge_selected_payloads(
                evidence.get("selected_results", ()),
                scoped_search_results(plan.results, [occurrence.occurrence_id]),
            )
        changes[row.row_id] = replace(row, object_metadata=metadata)
    if not changes:
        return document
    return document.with_project_metadata(
        document.project_metadata,
        objects=(changes.get(r.row_id, r) for r in document.objects),
    )
