"""Saved-evidence views. Build once at document validation, never on selection.

The index shares accepted results. Lookups touch only an object's references;
imported fields retain their original column identity and order.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType

from .records import MatchOccurrence, SearchSession


def resolution_metadata_warnings(resolution):
    """User-facing warnings from retained resolution evidence, with no lookup."""
    gender = resolution.get("metadata_discrepancies", {}).get("gender")
    if not gender:
        return ()
    returned = repr(gender["returned"]) if gender["returned"] is not None else "not supplied"
    return (f"Sex metadata discrepancy: source result {gender['requested']!r}; "
            f"public metadata {returned}. Both values are retained. This does not block EM retrieval.",)


@dataclass(frozen=True)
class EvidenceField:
    key: str
    label: str
    text: str | None


@dataclass(frozen=True)
class ResultValue:
    """Disposable display/sort projection, never a persistence record."""

    key: str
    header: str
    value: object
    role: str


def result_values(occurrence):
    if occurrence.fields:
        return tuple(ResultValue(f"csv:{f.column_index}",
            f"{f.header or 'Unlabelled'} [{f.column_index + 1}]", f.value, f.role)
            for f in occurrence.fields)
    values = [ResultValue("metric:" + m.name, m.name, m.value,
                         "unknown" if m.name == "pppmRank" else "score") for m in occurrence.metrics]
    values.extend(ResultValue(key, label, value, role) for key, label, value, role in (
        ("rank", "Rank", occurrence.rank, "rank"),
        ("matched_pixels", "Matching pixels", occurrence.matched_pixels, "matched_pixels"),
        ("overlap", "Query overlap fraction", occurrence.overlap, "score"),
        ("mirrored", "Mirrored comparison", occurrence.mirrored, "mirror"),
        ("shift", "Query shift before mirroring (XY pixels)", occurrence.shift, "unknown"),
        ("member_id", "Library member", occurrence.member_id, "unknown"),
        ("candidate_ordinal", "Original candidate position", occurrence.candidate_ordinal, "unknown"),
    ) if value is not None)
    return tuple(values)


@dataclass(frozen=True)
class MatchSummary:
    session: SearchSession
    occurrence: MatchOccurrence
    fields: tuple[EvidenceField, ...]
    compact_fields: tuple[EvidenceField, ...]

    @property
    def key(self):
        return (self.session.session_id, self.occurrence.occurrence_id)

    @property
    def selector_label(self):
        query = self.session.query_reference
        return f"{self.key[0]} / {self.key[1]}" + (f" — {query}" if query else "")

    def compact_text(self, *, derived=False):
        prefix = "NeuronBridge source-match evidence" if derived else "NeuronBridge"
        parts = [f"{f.label}: {f.text[:180]}" for f in self.compact_fields if f.text not in (None, "")]
        return prefix + (" • " + " • ".join(parts) if parts else "")


class EvidenceIndex:
    """Read-only derived index of validated records; never persisted separately."""

    def __init__(self, results=()):
        matches, sessions = {}, {}
        self._results = {}
        self._sources = {}
        self._occurrences = {}
        for result in results:
            session = result.session
            self._results[session.session_id] = result
            sessions[session.session_id] = session
            self._occurrences[session.session_id] = tuple(result.occurrences)
            parameters = {p.name: p.value for p in session.parameters}
            public = session.external_evidence
            response = public.get("result_response", {})
            context = ()
            if session.source_kind == "precomputed":
                context = (
                    EvidenceField("requested_method", "Requested result method", parameters.get("requested_method")),
                    EvidenceField("algorithm_version", "Algorithm version", session.context.get("algorithm") or "Unknown"),
                    EvidenceField("retrieval", "Retrieval completeness / filters", json.dumps(session.context.get("retrieval", {}), ensure_ascii=False, sort_keys=True)),
                    EvidenceField("response_url", "Public result URL", response.get("url")),
                    EvidenceField("response_sha256", "Public response SHA-256", response.get("sha256")),
                    EvidenceField("retrieved_at", "Retrieved at", response.get("retrieved_at")),
                )
            local = session.context if session.context.get("backend") == "madi3d-local" else None
            if session.source_kind == "custom_query" and isinstance(local, dict):
                context = (
                    EvidenceField("algorithm_version", "Local positive-CDS algorithm", local.get("algorithm")),
                    EvidenceField("reference_revision", "Numerical reference revision", local.get("reference_revision")),
                    EvidenceField("library_snapshot", "Exact library snapshot", local.get("snapshot_id")),
                    EvidenceField("inventory_sha256", "Installed content inventory SHA-256", local.get("inventory_sha256")),
                    EvidenceField("local_completion", "Local search completion", local.get("completion_status")),
                    EvidenceField("local_counts", "Total / examined / failed / matched / retained", json.dumps(local.get("counts", {}), sort_keys=True)),
                    EvidenceField("query_mapping_warnings", "Query mapping warnings", "; ".join(d.message for d in session.diagnostics)),
                    EvidenceField("local_parameters", "Local search parameters", json.dumps({p.name: p.value for p in session.parameters}, sort_keys=True)),
                )
            if result.subset is not None:
                context += (EvidenceField("subset", "Available historical hits", f"Selected subset: {len(result.occurrences)} of {result.subset['reported_retained_count']} originally retained hits. Omitted session content is unavailable; its digests are linkage references only."),)
            for occurrence in result.occurrences:
                self._sources[(session.session_id, occurrence.occurrence_id)] = (
                    occurrence.target, session.neuronbridge_data_version)
                source = occurrence.target
                query = session.query_source
                common = (
                    EvidenceField("source", "Source", session.source_kind),
                    EvidenceField("source_name", "Source name", source.published_name),
                    EvidenceField("library", "Dataset / library", source.library),
                    EvidenceField("library_release", "Library release", source.library_release),
                    EvidenceField("query", "Query", session.query_reference),
                    EvidenceField("session", "Search ID", session.session_id),
                    EvidenceField("match", "Match ID", occurrence.occurrence_id),
                    EvidenceField("query_name", "Query source", query.published_name if query else None),
                    EvidenceField("query_image", "Query image ID", query.image.image_id if query else None),
                    EvidenceField("query_library", "Query library", query.library if query else None),
                    EvidenceField("data_version", "NeuronBridge data version", session.neuronbridge_data_version),
                    EvidenceField("image_id", "Source image ID", source.image.image_id),
                    EvidenceField("neuron_id", "Source neuron ID", source.biological.neuron_id),
                )
                original_fields = occurrence.fields or session.historical_fields_for(occurrence)
                imported = tuple(EvidenceField(
                    f"field:{f.column_index}:{f.header}:{f.role}",
                    f"{f.header if f.header is not None else 'Unlabelled'} [column {f.column_index + 1}; {f.role}]",
                    f.raw_text,
                ) for f in original_fields)
                native = tuple(EvidenceField(v.key, v.header, str(v.value) if v.value is not None else None)
                               for v in result_values(occurrence)) if not occurrence.fields else ()
                compact = common[:5] + common[7:10] + tuple(
                    field for field, original in zip(imported, original_fields)
                    if original.role in {"rank", "number", "score", "matched_pixels"}
                )
                compact += tuple(f for f, v in zip(native, result_values(occurrence))
                                 if v.role in {"rank", "score", "matched_pixels"})
                matches[(session.session_id, occurrence.occurrence_id)] = MatchSummary(
                    session, occurrence, common + context + imported + native + (
                        (EvidenceField("original_response", "Original match response", json.dumps(session.evidence_for(occurrence), ensure_ascii=False, sort_keys=True)),)
                        if session.source_kind == "precomputed" and occurrence.evidence_ref else ()), compact,
                )
        self._matches = matches
        self._sessions = sessions

    @property
    def results(self):
        return MappingProxyType(self._results)

    @property
    def sources(self):
        return MappingProxyType(self._sources)

    @property
    def matches(self):
        return MappingProxyType(self._matches)

    @property
    def sessions(self):
        return MappingProxyType(self._sessions)

    def occurrences(self, session_id):
        return self._occurrences.get(session_id, ())

    def for_object(self, metadata):
        from .evidence import match_references

        keys = dict.fromkeys((ref["session_id"], ref["occurrence_id"])
                             for ref in match_references(metadata))
        return tuple(self._matches[key] for key in keys if key in self._matches)


def is_derived(metadata):
    return bool((metadata or {}).get("neuronbridge", {}).get("derived_from"))


def shared_summary(index, metadatas):
    """Shared context across every selected object, never an arbitrary score."""
    groups = [index.for_object(metadata) for metadata in metadatas]
    represented = sum(bool(group) for group in groups)
    if not represented and not any(m.get("neuronbridge") for m in metadatas):
        return ""
    matches = {match.key: match for group in groups for match in group}
    session_ids = {key[0] for key in matches}
    session_ids.update(key for metadata in metadatas
                       for key in metadata.get("neuronbridge", {}).get("session_ids", ()))
    unavailable_sessions = {key for metadata in metadatas
                            for key in metadata.get("neuronbridge", {}).get(
                                "unavailable_session_ids", ())}
    session_ids.update(unavailable_sessions)
    text = f"NeuronBridge: {represented}/{len(groups)} objects with matches • {len(matches)} matches"
    text += f" • {len(session_ids)} searches"
    if unavailable_sessions:
        text += f" • {len(unavailable_sessions)} retained searches unavailable"
    if all(groups):
        for key in ("source", "library", "library_release", "query", "query_name", "query_image", "query_library"):
            fields = [next(f for f in match.fields if f.key == key) for match in matches.values()]
            values = {f.text for f in fields}
            if len(values) == 1 and fields and fields[0].text not in (None, ""):
                text += f" • {fields[0].label}: {fields[0].text[:180]}"
    if any(is_derived(m) for m in metadatas):
        text += " • Derived objects retain source-match evidence"
    return text


def attachment_preview(document, plan):
    """Full, ordered evidence preview for an explicit NB-01B reattachment plan."""
    from .evidence import match_references

    index = EvidenceIndex([plan.results])
    rows = {row.row_id: row for row in document.objects}
    lines = ["Search / query context:", json.dumps(plan.results.session.to_dict(), ensure_ascii=False, indent=2)]
    for proposal in plan.proposals:
        row = rows[proposal.row_id]
        lines.extend(("", f"Object {row.row_id}: {row.name}", "Existing match references (retained):"))
        lines.extend(f"  {ref['session_id']} / {ref['occurrence_id']}" for ref in match_references(row.object_metadata))
        for occurrence_id in proposal.occurrence_ids:
            match = index.matches[(plan.results.session.session_id, occurrence_id)]
            lines.extend((f"Attach: {match.selector_label}", "Source identity:",
                          json.dumps(match.occurrence.target.to_dict(), ensure_ascii=False, indent=2)))
            lines.extend(f"  {field.label}: {field.text if field.text is not None else 'Not supplied'}"
                         for field in match.fields)
    lines.extend(("", "Unchanged objects / conflicts:", *plan.conflicts))
    return "\n".join(lines)
