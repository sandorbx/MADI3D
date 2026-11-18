"""Saved-evidence views. Build once at document validation, never on selection.

The index shares accepted results. Lookups touch only an object's references;
imported fields retain their original column identity and order.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from urllib.parse import quote, urlencode

from .records import MatchOccurrence, SearchSession


def session_display_name(session, query_records=None):
    """Resolve a readable search label without changing retained evidence."""
    if session.csv_provenance is not None:
        return session.csv_provenance.filename
    query = (query_records or {}).get(session.query_reference, {})
    source_name = session.query_source.published_name if session.query_source else None
    return query.get("label") or source_name or session.query_reference or session.source_kind


def match_display_label(match, query_records=None):
    query = session_display_name(match.session, query_records)
    source = match.occurrence.target
    target = (source.published_name or source.biological.neuron_instance
              or source.biological.line_name or source.biological.neuron_type
              or source.kind)
    return (f"{query} (ID: {match.session.session_id}) / "
            f"{target} (ID: {match.occurrence.occurrence_id})")


def resolution_metadata_warnings(resolution):
    """User-facing warnings from retained resolution evidence, with no lookup."""
    gender = resolution.get("metadata_discrepancies", {}).get("gender")
    if not gender:
        return ()
    returned = repr(gender["returned"]) if gender["returned"] is not None else "not supplied"
    return (f"Sex metadata discrepancy: source result {gender['requested']!r}; "
            f"public metadata {returned}. Both values are retained. This does not block EM retrieval.",)


def source_properties(source):
    """Scientific identity supplied by a result, without inferred biology."""
    biological, image = source.biological, source.image
    values = (
        ("Provider", "NeuronBridge"), ("Kind", source.kind),
        ("Library", source.library), ("Release", source.library_release),
        ("Published name", source.published_name), ("Neuron ID", biological.neuron_id),
        ("Driver line", biological.line_name), ("Cell type", biological.neuron_type),
        ("Neuron instance", biological.neuron_instance), ("Image ID", image.image_id),
        ("Anatomical area", image.anatomical_area), ("Sex (source)", image.sex),
        ("Alignment space", image.alignment_space),
        ("Magnification", image.magnification),
        ("Mounting protocol", image.mounting_protocol),
        ("Slide code", image.slide_code),
        ("Channel (source selector)", source.channel.selector if source.channel else None),
    )
    return [(label, value) for label, value in values if value not in (None, "")]


_FAFB_ID = re.compile(r"flywire_fafb:v([1-9][0-9]*):([1-9][0-9]*)")
_CONNECTOME_ID = re.compile(r"(flywire_fafb|flywire_banc|manc|male-cns):v([0-9]+(?:\.[0-9]+)*):([1-9][0-9]*)")
_CONNECTOME_DATASET = {"flywire_fafb": "fafb", "flywire_banc": "banc", "manc": "manc", "male-cns": "mcns"}
_CONNECTOME_LIBRARY = {
    "flywire fafb v783 realign": "flywire_fafb:v783",
    "flywire fafb v783": "flywire_fafb:v783",
    "flyem manc v1.2.1": "manc:v1.2.1",
    "flyem male cns brain v0.9": "male-cns:v0.9",
    "flyem male cns vnc v0.9": "male-cns:v0.9",
    "flywire banc v626": "flywire_banc:v626",
    "flywire banc v888": "flywire_banc:v888",
}
_CONNECTOME_LIBRARY_NAMESPACE = {
    "flywire brain": "flywire_fafb",
    "flywire fafb": "flywire_fafb",
    "flyem manc": "manc",
    "flyem male cns": "male-cns",
    "flywire banc": "flywire_banc",
}


def _neuron_link_id(source):
    identifier = source.biological.neuron_id
    # Public EM results can supply the qualified identifier as publishedName.
    published = _CONNECTOME_ID.fullmatch(source.published_name or "")
    if published and (not identifier or identifier == published[3]):
        return source.published_name
    if not identifier or published:
        return identifier
    namespace, _, root = identifier.rpartition(":")
    if not re.fullmatch(r"[1-9][0-9]*", root):
        return identifier
    library = " ".join((source.library or "").casefold().replace("_", " ").split())
    release = (source.library_release or "").removeprefix("v")
    prefix = _CONNECTOME_LIBRARY.get(library)
    if prefix:
        library_namespace, library_version = prefix.split(":")
        if not release or release == library_version[1:]:
            release = library_version[1:]
        else:
            return identifier
    else:
        library_namespace = _CONNECTOME_LIBRARY_NAMESPACE.get(library)
    if library_namespace and release and namespace in ("", library_namespace):
        return f"{library_namespace}:v{release}:{root}"
    return identifier


def flywire_result_identity(source):
    """Resolve the published connectome, release and exact neuron ID."""
    from madi3d_app.integrations.flywire.records import validate_root_id
    from madi3d_app.integrations.flywire.reference_links import vfb_site

    match = _CONNECTOME_ID.fullmatch(_neuron_link_id(source) or "")
    if match is not None:
        namespace, release, root_id = match.groups()
        dataset = _CONNECTOME_DATASET[namespace]
        try:
            validate_root_id(root_id)
        except ValueError:
            return None
        if vfb_site(dataset, release):
            return dataset, release, root_id
    return None


def source_links(source):
    """Project external links from recorded neuron identifiers, without lookup."""
    from madi3d_app.integrations.flywire.records import FlyWireRemoteDescriptor, validate_root_id

    if isinstance(source, FlyWireRemoteDescriptor):
        if source.context.dataset == "mcns":
            return (("Open MaleCNS source homepage (release not selected)",
                     "https://male-cns.janelia.org/"),)
        if source.context.dataset != "fafb":
            return ()
        identifier = f"flywire_fafb:v{source.context.segmentation_version}:{source.root_id}"
    else:
        identifier = _neuron_link_id(source)
        if identifier and _CONNECTOME_ID.fullmatch(identifier) and not _FAFB_ID.fullmatch(identifier):
            # Other connectomes use the exact VFB database cross-reference.
            return ()
    if not identifier:
        return ()
    links = []
    flywire = _FAFB_ID.fullmatch(identifier)
    if flywire:
        release, root_id = flywire.groups()
        try:
            validate_root_id(root_id)
        except ValueError:
            pass
        else:
            query = urlencode({"dataset": "fafb", "data_version": release, "root_id": root_id})
            links.append(("Open in FlyWire", "https://codex.flywire.ai/app/cell_details?" + query))
    links.append(("Open in Virtual Fly Brain",
                  "https://virtualflybrain.org/xref/neuronbridge/" + quote(identifier, safe=":")))
    return tuple(links)


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
        ("rank", "Positive rank" if (occurrence.gradient or occurrence.shape) else "Rank",
         occurrence.rank, "rank"),
        ("matched_pixels", "Matching pixels", occurrence.matched_pixels, "matched_pixels"),
        ("overlap", "Query overlap fraction", occurrence.overlap, "score"),
        ("mirrored", "Positive comparison mirrored" if (occurrence.gradient or occurrence.shape)
         else "Mirrored comparison", occurrence.mirrored, "mirror"),
        ("shift", "Query shift before mirroring (XY pixels)", occurrence.shift, "shift"),
        ("member_id", "Library member", occurrence.member_id, "unknown"),
        ("candidate_ordinal", "Original candidate position", occurrence.candidate_ordinal, "unknown"),
    ) if value is not None)
    if occurrence.gradient is not None:
        gradient = occurrence.gradient
        values.extend((
            ResultValue("gradient_score", "Gradient score", gradient.normalized_score, "gradient_score"),
            ResultValue("gradient_rank", "Gradient rank", gradient.gradient_rank, "gradient_rank"),
            ResultValue("gradient_area_gap", "Gradient area gap", gradient.gradient_area_gap, "penalty"),
            ResultValue("high_expression_area", "High-expression area", gradient.high_expression_area, "penalty"),
            ResultValue("combined_penalty", "Combined penalty", gradient.combined_penalty, "penalty"),
            ResultValue("gradient_mirrored", "Gradient comparison mirrored", gradient.gradient_mirrored, "gradient_mirror"),
        ))
    if occurrence.shape is not None:
        shape = occurrence.shape
        values.extend((
            ResultValue("shape_score", "Shape score", shape.combined_score, "gradient_score"),
            ResultValue("shape_rank", "Shape rank", shape.combined_rank, "gradient_rank"),
            ResultValue("positive_score", "Positive score percent", shape.positive_score, "score"),
            ResultValue("mask_to_sample", "Mask-to-sample measure", shape.mask_to_sample, "penalty"),
            ResultValue("sample_to_mask", "Sample-to-mask measure", shape.sample_to_mask, "penalty"),
            ResultValue("shape_normal_value", "Ordinary shape measure", shape.normal_value, "penalty"),
            ResultValue("shape_flip_value", "Mirrored shape measure", shape.flip_value, "penalty"),
            ResultValue("shape_value", "Selected shape value", shape.shape_gap, "penalty"),
            ResultValue("shape_mirrored", "Shape comparison mirrored", shape.shape_mirrored, "gradient_mirror"),
        ))
    return tuple(values)


def comparison_mirror(occurrence):
    """Project the winning comparison without borrowing another stage's flag."""
    if occurrence.shape is not None:
        return occurrence.shape.shape_mirrored
    if occurrence.gradient is not None:
        return occurrence.gradient.gradient_mirrored
    if any(d.code == "unknown_match_method" for d in occurrence.diagnostics):
        return None
    if occurrence.fields:
        values = {field.value for field in occurrence.fields_for("mirror")
                  if type(field.value) is bool}
        return next(iter(values)) if len(values) == 1 else None
    return occurrence.mirrored


def ranking_value(session, occurrence):
    """Return the method's ranking value at its recorded precision."""
    if occurrence.shape is not None:
        return occurrence.shape.combined_score
    if occurrence.gradient is not None:
        return occurrence.gradient.normalized_score
    if session.context.get("backend") == "madi3d-local":
        return occurrence.matched_pixels
    method = next((p.value for p in session.parameters if p.name == "requested_method"), None)
    if session.source_kind == "precomputed":
        # NeuronBridge orders PPPM by its internal rank, which can be fractional.
        meaning = {
            "PPPM": "neuronbridge:PPPMatch:pppmRank",
            "CDM": "neuronbridge:CDSMatch:normalizedScore",
        }.get(method)
        if meaning is not None:
            return next((m.value for m in occurrence.metrics if m.meaning == meaning), None)
    if method == "PPPM":
        roles = ("rank",)
    elif method == "CDM":
        roles = ("score", "matched_pixels")
    else:
        roles = ("rank", "score", "matched_pixels")
    # A blank cell in one column does not establish which metric ranked the file.
    candidates = [fields for role in roles if (fields := occurrence.fields_for(role))]
    if len(candidates) != 1:
        return None
    supplied = [field.value for field in candidates[0] if field.value is not None]
    return supplied[0] if len(supplied) == 1 else None


@dataclass(frozen=True)
class MatchSummary:
    session: SearchSession
    occurrence: MatchOccurrence
    _fields: tuple[EvidenceField, ...]
    compact_fields: tuple[EvidenceField, ...]

    @property
    def fields(self):
        """Complete ordered display fields, formatted only for explicit consumers."""
        if self.session.source_kind == "precomputed" and self.occurrence.evidence_ref:
            return self._fields + (EvidenceField(
                "original_response", "Original match response",
                json.dumps(self.session.evidence_for(self.occurrence), ensure_ascii=False, sort_keys=True),
            ),)
        return self._fields

    @property
    def key(self):
        return (self.session.session_id, self.occurrence.occurrence_id)

    @property
    def selector_label(self):
        return match_display_label(self)

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
                execution = session.gradient_execution
                shape_execution = session.shape_execution
                algorithm = local.get("algorithm")
                if execution is not None:
                    algorithm = f"{algorithm} + {execution.algorithm}"
                if shape_execution is not None:
                    algorithm = f"{algorithm} + {shape_execution.algorithm}"
                context = (
                    EvidenceField("algorithm_version", "Local search algorithms" if (execution or shape_execution) else "Local positive-CDS algorithm", algorithm),
                    EvidenceField("reference_revision", "Numerical reference revision", local.get("reference_revision")),
                    EvidenceField("library_distribution", "Installed distribution",
                                  local.get("library_id") or local.get("library")),
                    EvidenceField("archive_sha256", "Installed package digest", local.get("archive_sha256")),
                    EvidenceField("local_completion", "Local search completion", local.get("completion_status")),
                    EvidenceField("local_counts", "Total / examined / failed / matched / retained", json.dumps(local.get("counts", {}), sort_keys=True)),
                    EvidenceField("query_mapping_warnings", "Query mapping warnings", "; ".join(d.message for d in session.diagnostics)),
                    EvidenceField("local_parameters", "Local search parameters", json.dumps({p.name: p.value for p in session.parameters}, sort_keys=True)),
                )
                if local.get("historical_provenance"):
                    context += (EvidenceField("historical_population", "Historical population provenance",
                        json.dumps(local["historical_provenance"], ensure_ascii=False, sort_keys=True)),)
                if execution is not None:
                    context += (
                        EvidenceField("gradient_reference", "Gradient numerical reference", execution.reference_revision),
                        EvidenceField("gradient_parameters", "Gradient scoring parameters", json.dumps(execution.parameters, sort_keys=True)),
                        EvidenceField("gradient_inputs", "Historical LM supplementary input identifier", execution.bundle_reference.get("bundle_id") or "No retained candidates; no supplementary images consumed"),
                        EvidenceField("gradient_cohort", "Gradient normalization cohort", f"{execution.cohort_count} retained candidates; maximum positive count {execution.max_positive_pixels}; maximum penalty {execution.max_combined_penalty}; {execution.normalization_precision}"),
                    )
                if shape_execution is not None:
                    if shape_execution.shape_stage_executed:
                        cohort = (f"Shape stage scored {shape_execution.shortlist_count} of "
                                  f"{shape_execution.positive_count} positive hits (budget "
                                  f"{shape_execution.scoring_budget}); {shape_execution.emitted_count} emitted; "
                                  f"maximum positive score {shape_execution.max_positive_score}; "
                                  f"maximum shape value {shape_execution.max_shape_gap}")
                    else:
                        cohort = (f"Shape stage not run; {shape_execution.positive_count} positive hits "
                                  "(two or fewer keep the positive ranking unchanged)")
                    context += (
                        EvidenceField("shape_reference", "Shape numerical reference", shape_execution.reference_revision),
                        EvidenceField("shape_parameters", "Shape scoring parameters", json.dumps(shape_execution.parameters, sort_keys=True)),
                        EvidenceField("shape_cohort", "Shape scoring cohort", cohort),
                    )
                    if shape_execution.gradient_producer_reference_revision is not None:
                        context += (
                            EvidenceField("shape_gradient_producer_reference", "Candidate gradient producer reference",
                                          shape_execution.gradient_producer_reference_revision),
                            EvidenceField("shape_gradient_producer_implementation", "Candidate gradient producer implementation",
                                          shape_execution.gradient_producer_implementation_revision),
                            EvidenceField("shape_gradient_format", "Candidate gradient format", shape_execution.gradient_format),
                            EvidenceField("shape_scored_cohort_digest", "Consumed shape cohort SHA-256",
                                          shape_execution.scored_cohort_digest),
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
                                 if v.role in {"rank", "score", "matched_pixels", "gradient_score",
                                               "gradient_rank", "penalty", "gradient_mirror", "mirror", "shift"})
                matches[(session.session_id, occurrence.occurrence_id)] = MatchSummary(
                    session, occurrence, common + context + imported + native, compact,
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
            fields = [next(f for f in match.compact_fields if f.key == key) for match in matches.values()]
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
