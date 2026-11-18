"""Local positive-CDS cohort + Janelia EM_MIP shape reranking.

The positive CDS stage is pure NumPy and exhaustive over every declared
candidate, using the pinned NeuronBridge-services implementation. That cohort
is reranked with the separately pinned EM_MIP_search shape algorithm:

    positive CDS → Janelia shortlist (06A semantics) → candidate gradients
    through the 06B supply → bidirectional shape scoring → exact Janelia
    normalization and final assembly.

``ShapeRequest`` declares the local-mount role and the shape-search mirror
preference. The shape stage consumes only the 06B supply (no user files) and
borrows the canvas it already reconstructed for measurement, so a shortlisted
candidate is reconstructed once; the positive stage's public behavior is
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import heapq
import hashlib
import io
import json
import sys
from uuid import uuid4

import numpy as np
from PIL import Image

from .cache import GradientCache
from .local_catalog import canonical
from .library_store import DenseArchiveReader, PIXEL_DTYPE, SHARD_BYTES, open_library
from .search_profiles import compatible_query, query_profile
from .local_scorer import ALGORITHM, PIXEL_BLOCK, REFERENCE_REVISION, PreparedQuery, SearchParameters, checkpoint
from .query import QUERY_KEY, query_png, query_image_evidence, validate_queries
from .records import (Diagnostic, MatchOccurrence, SearchResults, SearchSession, ShapeExecution,
                      ShapeObservation, SourceIdentity, SourceParameter, gradient_membership_digest)

JOB_KEY = "neuronbridge_local_search_jobs"
SHAPE_ALGORITHM = "janelia-em-mip-shape-rerank/1"
SHAPE_IMPLEMENTATION_REVISION = "madi3d-janelia-shape/2"


class _RetainedMetadata:
    """Count live row/JSON storage once, including shared strings/containers.

    Updating one heap member visits only that member. Parser scratch is unknown;
    mapped pixels, the query and other borrowed buffers do not enter this owner.
    """

    def __init__(self, budget):
        self.budget = budget
        self.bytes = 0
        self.objects = {}
        self.entry_bytes = 0

    def update(self, value, delta):
        key = id(value)
        if key in self.objects:
            stored, count = self.objects[key]
            if count + delta:
                self.objects[key] = (stored, count + delta)
                return
            self.entry_bytes -= sys.getsizeof(key) + sys.getsizeof(self.objects[key])
            del self.objects[key]
        else:
            self.objects[key] = (value, delta)
            self.entry_bytes += sys.getsizeof(key) + sys.getsizeof(self.objects[key])
        self.bytes += delta * sys.getsizeof(value)
        if isinstance(value, dict):
            for name, item in value.items():
                self.update(name, delta)
                self.update(item, delta)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.update(item, delta)

    def check(self, source_bytes=0):
        tracking_bytes = sys.getsizeof(self.objects) + self.entry_bytes
        if self.bytes + source_bytes + tracking_bytes > self.budget:
            raise ValueError("Retained hit metadata exceeds the search memory budget. Reduce the result limit or increase the budget.")

    def check_observation(self, observation, source_bytes):
        # The parsed response and its raw input coexist inside the library owner.
        self.update(observation, 1)
        try:
            self.check(source_bytes)
        finally:
            self.update(observation, -1)


@dataclass(frozen=True)
class ShapeRequest:
    """LM role declaration and shape-search preference, detached from the GUI.

    The two-stage shape search needs no user gradient files: every candidate
    gradient comes from the 06B supply after shortlist admission. The caller
    must declare the LM role; filenames cannot establish it.
    """
    association: dict
    mirror: bool = True

    def __post_init__(self):
        from .query import _explicit_association
        if type(self.mirror) is not bool:
            raise ValueError("Shape mirror preference must be boolean.")
        object.__setattr__(self, "association", _explicit_association(copy.deepcopy(self.association)))


def shape_working_set(profile, *, retained_bytes=0, with_shape_boundary=False):
    """Identifiable phase peaks for the configured search hold in shape mode.

    The shape stage holds the query carriers (uint16 mask gradient, dilated
    RGB mask, binarized uint16 values, original RGB, plus flipped copies when
    mirroring) and, per candidate, the dense CDM reconstruction, the supplied
    uint16 gradient, the dilated candidate, and int64 product/gap temporaries.
    The mapped shard and SQLite's configured 2-MiB page cache coexist with
    every phase. PNG codecs and native-library scratch remain unknown, never an
    invented allowance.
    """
    pixels = int(np.prod(profile.search_canvas_yx))
    shared = SHARD_BYTES + 2 * 1024 * 1024 + retained_bytes
    masks = pixels * (2 + 3 + 2 + 3 + 2 + 3 + 2)  # query carriers plus flipped copies
    phases = {
        "positive_scan": shared + pixels * 3,
        "mask_construction": shared + pixels * (3 + 2 + 2 + 3 + 2) + masks,
        "candidate_scoring": shared + masks + pixels * (3 + 2 + 3 + 2 + 8 + 8 + 4 + 1),
    }
    if with_shape_boundary:
        # PNG uint8 pixels coexist with the newly converted boolean keep mask.
        phases["boundary_decode"] = shared + pixels * (3 + 1 + 1)
        # Retained boundary plus its inverse during query-carrier clipping.
        phases["mask_construction"] += pixels * 2
        phases["candidate_scoring"] += pixels
    return {"known_peak_bytes": max(phases.values()), "phase_bytes": phases,
            "unknown_scopes": ("PNG codec scratch", "SciPy native scratch", "Python result objects")}


def _member_filename(member_key):
    """Java-equivalent hit filename: the member key's base name."""
    return member_key.replace("\\", "/").rsplit("/", 1)[-1]


def _positive_key(result, row):
    """The reference shortlist ordering key: (rounded percent, filename)."""
    from .gradient_scorer import positive_label
    label = positive_label(result.overlap_fraction, _member_filename(row["member_key"]))
    return float(label.split("_")[0]), label.split("_", 1)[1]


def _collect_identity(reader, row, metadata, cancel):
    """Refresh one retained row's identity evidence under the metadata owner."""
    identity_evidence = row["metadata_evidence"]
    if isinstance(identity_evidence, str):
        identity_evidence = json.loads(identity_evidence)
    metadata.update(identity_evidence, 1)
    # Only this response's raw bytes overlap the already retained rows.
    # Embedded historical observations need no source-buffer allowance.
    metadata.check(identity_evidence["size"] if "response_ref" in identity_evidence else 0)
    observation = reader.identity_observation(identity_evidence, cancel=cancel,
                                              check_storage=metadata.check_observation)
    checkpoint(cancel)
    metadata.update(row, -1)
    row["metadata_evidence"] = observation
    metadata.update(row, 1)
    metadata.update(identity_evidence, -1)
    metadata.check()


def _retain_decoded_hash(row, candidate, metadata):
    """Bind retained dense pixels while keeping live row accounting consistent."""
    if "pixel_sha256" not in row:
        metadata.update(row, -1)
        row["pixel_sha256"] = hashlib.sha256(memoryview(candidate)).hexdigest()
        metadata.update(row, 1)
        metadata.check()


def search_library(library, query_record, parameters=None, cancel=None, progress=None, *,
                   session_id=None, shape_request=None, gradient_cache=None):
    """Exhaustive positive-CDS search of one installed distribution library.

    ``library`` is the installed distribution to search: an
    ``archive_install.InstalledArchive`` or an installed library directory —
    anything ``library_store.open_library`` accepts. The search opens the
    package, streams every declared candidate exactly once, and releases the
    package handles on success, error and cancellation.

    With ``shape_request`` the search continues into Janelia EM_MIP
    shape search over the positive shortlist. ``gradient_cache`` overrides the
    derived-gradient cache used by the 06B supply; production passes nothing
    and the disposable ``cache.py`` root owns the default location.

    No metadata/public API request is permitted and no source-image decoding
    happens beyond what the installed reader performs for its representation.
    """
    with open_library(library, cancel=cancel) as reader:
        return _search_reader(reader, query_record, parameters, cancel, progress,
                              session_id=session_id, shape_request=shape_request,
                              gradient_cache=gradient_cache)


def _search_reader(reader, query_record, parameters, cancel, progress, *,
                   session_id=None, shape_request=None, gradient_cache=None):
    """All declared candidates of one open installed library, or an error."""
    identity = reader.identity
    library_id, archive = identity.get("library_id"), identity.get("archive_sha256")
    if (not isinstance(library_id, str) or not library_id.strip() or
            not isinstance(archive, str) or len(archive) != 64 or
            any(c not in "0123456789abcdef" for c in archive)):
        raise ValueError("Local search requires a managed library with a canonical library ID and archive SHA-256.")
    parameters = parameters or SearchParameters()
    if shape_request is not None:
        if not isinstance(shape_request, ShapeRequest):
            raise ValueError("Shape search requires a detached ShapeRequest.")
        query_record = copy.deepcopy(query_record)
    validate_queries({QUERY_KEY: {query_record["query_id"]: query_record}})
    if not compatible_query(query_record):
        raise ValueError("The retained query is incompatible with the supported CDM profiles.")
    if not compatible_query(query_record, reader.profile_id):
        raise ValueError("Query and library search profiles disagree; Brain and VNC cannot be combined.")
    with Image.open(io.BytesIO(query_png(query_record))) as image:
        rgb = np.asarray(image).copy()
    profile = query_profile(query_record)
    prepared = PreparedQuery(rgb, parameters, cancel, profile_id=profile.profile_id)
    if shape_request is None:
        del rgb
    report = progress or (lambda phase, done, total: None)
    # Identifiable live storage, independent of library candidate count.
    # Sparse block carriers: position/sector/ratio, x/y, inverse x/y, valid,
    # intp slots, query indices and overlap mask. Dense uses fewer carriers.
    # Sector/gap expression scratch is block-bounded but its allocation peak,
    # native TIFF scratch and Python decoder objects remain unquantified.
    query_bytes = sum(array.nbytes for array in
                      (prepared.positions, prepared.sectors, prepared.ratios, prepared.query_index))
    canvas_bytes = int(np.prod(prepared.shape))
    block_bytes = PIXEL_BLOCK * (4 + 1 + 8 + 2 * 4 + 2 * 4 + 1 + np.dtype(np.intp).itemsize + 4 + 1)
    positive_bytes = query_bytes + block_bytes
    positive_bytes += (reader.decode_workers * canvas_bytes if isinstance(reader, DenseArchiveReader)
                       else SHARD_BYTES + 2 * 1024 * 1024)  # SQLite page cache
    if shape_request is not None:
        positive_bytes += canvas_bytes
    # Retained reconstruction follows release of the prepared-query arrays.
    positive_bytes = max(positive_bytes, canvas_bytes +
                         (0 if isinstance(reader, DenseArchiveReader) else SHARD_BYTES + 2 * 1024 * 1024))
    metadata = _RetainedMetadata(parameters.memory_mb * 1024 * 1024 - positive_bytes)
    total = reader.candidate_count
    maxnumber = parameters.result_limit
    # The shape stage's shortlist is the reference stage-1 cap: the top 400
    # positive hits (or maxnumber+200 when maxnumber exceeds 400) in
    # descending positive-label order, independent of the result limit.
    heap_cap = (400 if maxnumber <= 400 else maxnumber + 200) if shape_request is not None \
        else parameters.result_limit
    heap, examined, matched, fl_positive = [], 0, 0, 0
    report("Searching", 0, total)
    for row, pixels in reader.candidates():
        checkpoint(cancel)
        if pixels.dtype == PIXEL_DTYPE:
            # Sparse package: the published span feeds the scorer directly; a
            # positive search never reconstructs a dense canvas.
            result = prepared.score_sparse(pixels["position"], pixels["rgb"], cancel)
        else:
            # Dense archive: the strictly-decoded candidate goes straight to the
            # dense scorer and is released before the next decode.
            result = prepared.score(pixels, cancel)
        del pixels
        examined += 1
        # batch_search.js uses strict greater-than for the minimum ratio.
        if result.overlap_fraction > parameters.minimum_overlap:
            matched += 1
            if shape_request is None:
                key = (result.matching_pixels, -row["ordinal"])
            else:
                # The shortlist heap ordering is the oracle's: (rounded_pct,
                # filename) ascending, so the weakest label entry drops first.
                key = _positive_key(result, row)
                # L733-738: the "_FL" suffix is counted over every positive
                # hit, including hits the stage-1 cap will not retain.
                if "_FL" in _member_filename(row["member_key"]):
                    fl_positive += 1
            if len(heap) < heap_cap or key > heap[0][:2]:
                row = dict(row)
                if len(heap) == heap_cap:
                    metadata.update(heapq.heappop(heap)[2], -1)
                metadata.update(row, 1)
                metadata.check()
                heapq.heappush(heap, (*key, row, result))
        report("Searching", examined, total)
    checkpoint(cancel)
    if examined != total:
        raise ValueError("Search did not examine every declared candidate; no ranking was published.")
    # The positive pixel buffers are no longer needed. Keep the reader open
    # while consuming only the retained candidates.
    foreground_count = prepared.foreground_count
    del prepared
    # Only the bounded final heap receives full identity and provenance rows.
    for _, _, row, _ in heap:
        checkpoint(cancel)
        metadata.update(row, -1)
        row.update(reader.result_row(row))
        metadata.update(row, 1)
        metadata.check()
    if shape_request is None:
        ordered = sorted(heap, key=lambda item: (-item[0], -item[1]))
        for _, _, row, _ in ordered:
            checkpoint(cancel)
            if "pixel_sha256" not in row:
                candidate = reader.dense_canvas(row)
                _retain_decoded_hash(row, candidate, metadata)
                del candidate
            _collect_identity(reader, row, metadata, cancel)
    else:
        ordered = sorted(heap, key=lambda item: (item[0], item[1]), reverse=True)
    retained_bytes = metadata.bytes
    shape_summary = None
    publication = ()
    if shape_request is None:
        # Positive-only publication: every retained row already carries its
        # refreshed identity observation. Accounting must not prolong
        # temporary JSON/row lifetimes into result construction.
        publication = tuple((position + 1, entry, entry[2], None)
                            for position, entry in enumerate(ordered))
        del metadata
    else:
        from .gradient_scorer import (MASK_THRESHOLD, NEGATIVE_RADIUS, REFERENCE_REVISION as
                                      shape_reference, ShapeScorer, assemble_ranking,
                                      normalize_gaps, positive_label, positive_stack_slice)
        from .gradient_supply import supply_candidate_gradient
        from .gradient_preparation import GRADIENT_FORMAT, IMPLEMENTATION, PRODUCER_REVISION
        cohort_entries = []
        gradient_hashes = {}
        parameters_dict = {"mirror": shape_request.mirror, "mask_threshold": MASK_THRESHOLD,
                           "negative_radius": NEGATIVE_RADIUS}
        if matched > 2:
            boundary = reader.shape_boundary()
            if shape_working_set(profile, retained_bytes=retained_bytes,
                                 with_shape_boundary=boundary is not None)["known_peak_bytes"] > \
                    parameters.memory_mb * 1024 * 1024:
                raise ValueError("Shape working arrays exceed the search memory budget. Increase the budget or reduce the result limit.")
            budget = positive_stack_slice(len(ordered), maxnumber, fl_positive)
            shortlist = ordered[:budget]
            scorer = ShapeScorer(rgb, mirror=shape_request.mirror, mask_threshold=MASK_THRESHOLD,
                                 negative_radius=NEGATIVE_RADIUS, shape_boundary=boundary, cancel=cancel)
            del rgb
            cache = gradient_cache if gradient_cache is not None else GradientCache()
            # The shape stage runs supply (06B) then scoring (06C) per
            # shortlisted candidate; every candidate is scored exactly once and
            # reconstructed exactly once, because the supply is handed the same
            # canvas the scorer measures.
            report("Scoring shapes", 0, len(shortlist))
            labels, areas, flips, measurements, sources = [], [], [], [], []
            for index, (_, _, row, result) in enumerate(shortlist, 1):
                checkpoint(cancel)
                candidate = reader.dense_canvas(row)
                if "pixel_sha256" not in row:
                    _retain_decoded_hash(row, candidate, metadata)
                supplied = supply_candidate_gradient(reader, row, candidate_canvas=candidate,
                                                     cache=cache, cancel=cancel)
                measurement = scorer.measure(candidate, supplied.gradient, cancel)
                measurements.append(measurement)
                labels.append(positive_label(result.overlap_fraction,
                                             _member_filename(row["member_key"])))
                areas.append(measurement.selected_value)
                flips.append(measurement.mirrored)
                sources.append(supplied.source)
                gradient_hashes[row["ordinal"]] = supplied.pixel_sha256
                cohort_entries.append({
                    "candidate_ordinal": row["ordinal"], "member_key": row["member_key"],
                    "decoded_rgb_sha256": row["pixel_sha256"], "positive_label": labels[-1],
                    "gradient_source": supplied.source, "gradient_pixel_sha256": supplied.pixel_sha256,
                    "shape_gap": measurement.selected_value, "shape_mirrored": measurement.mirrored})
                del candidate, supplied
                report("Scoring shapes", index, len(shortlist))
            max_score = max(float(label.split("_")[0]) for label in labels)
            max_area = max(areas)
            gaps, gap_values = normalize_gaps([float(label.split("_")[0]) for label in labels],
                                              areas, max_score, max_area)
            emission = assemble_ranking(labels, gaps, gap_values, flips, maxnumber=maxnumber)
            # Identity observations are collected for published rows only.
            for selection in emission:
                array_index = selection.score_source_index
                checkpoint(cancel)
                _collect_identity(reader, shortlist[array_index][2], metadata, cancel)
            del metadata
            publication = tuple(
                (rank, shortlist[array_index], shortlist[array_index][2],
                 ShapeObservation(
                     positive_rank=array_index + 1,
                     positive_score=float(labels[array_index].split("_")[0]),
                     mask_to_sample=measurements[array_index].mask_to_sample,
                     sample_to_mask=measurements[array_index].sample_to_mask,
                     normal_value=measurements[array_index].normal_value,
                     flip_value=measurements[array_index].flip_value,
                     shape_mirrored=selection.mirrored,
                     shape_gap=areas[array_index],
                     combined_score=float(selection.gap_text),
                     combined_gap_text=selection.gap_text,
                     combined_rank=rank,
                     fl_renamed=selection.fl_renamed,
                     gradient_source=sources[array_index]))
                for rank, selection in enumerate(emission, 1)
                for array_index in (selection.score_source_index,)
            )
            del scorer, measurements
            shape_summary = {"executed": True, "parameters": parameters_dict,
                             "stage1_cap": len(ordered), "budget": budget,
                             "shortlist": len(shortlist), "fl_positive": fl_positive,
                             "max_score": max_score, "max_area": max_area}
        else:
            # The reference gate runs the shape stage only above two positive
            # hits (run() L1833); with two or fewer, the positive ranking is
            # published unchanged and the skipped stage is recorded honestly.
            del rgb
            for _, _, row, _ in ordered:
                checkpoint(cancel)
                if "pixel_sha256" not in row:
                    candidate = reader.dense_canvas(row)
                    _retain_decoded_hash(row, candidate, metadata)
                    del candidate
                _collect_identity(reader, row, metadata, cancel)
            del metadata
            publication = tuple((position + 1, entry, entry[2], None) for position, entry in
                                enumerate(sorted(ordered, key=lambda item:
                                                 (-item[3].matching_pixels, item[2]["ordinal"]))))
            shape_summary = {"executed": False, "parameters": parameters_dict,
                             "stage1_cap": 0, "budget": 0, "shortlist": 0,
                             "fl_positive": fl_positive, "max_score": None, "max_area": None}
    # 8. Build the result rows in publication order.
    session_id = session_id or str(uuid4())
    external = {}
    occurrences = []
    for rank, entry, row, shape in publication:
        result = entry[3]
        checkpoint(cancel)
        identity_evidence = dict(row["metadata_evidence"])
        image_evidence = row["image_evidence"]
        if isinstance(image_evidence, str):
            image_evidence = json.loads(image_evidence)
        if "observed_response" in identity_evidence:
            response = identity_evidence.pop("observed_response")
            response_ref = hashlib.sha256(canonical(response).encode("utf-8")).hexdigest()
            external.setdefault(response_ref, response)
            identity_evidence["response_ref"] = response_ref
        references = {}
        for scope, observation in (("identity", identity_evidence), ("search_image", image_evidence)):
            key = hashlib.sha256(canonical(observation).encode("utf-8")).hexdigest()
            external.setdefault(key, observation)
            references[scope] = key
        observation = {"references": references}
        evidence_ref = hashlib.sha256(canonical(observation).encode("utf-8")).hexdigest()
        external.setdefault(evidence_ref, observation)
        unresolved = (row["diagnostics"] if isinstance(row["diagnostics"], (list, tuple))
                      else json.loads(row["diagnostics"]))
        occurrence_diagnostics = tuple(Diagnostic("local_identity_unresolved", text, row_index=rank - 1)
                                       for text in unresolved)
        input_hashes = {"decoded_rgb": row["pixel_sha256"]}
        if shape is not None:
            input_hashes["shape_gradient"] = gradient_hashes[row["ordinal"]]
        if image_evidence.get("sha256"):
            input_hashes["search_image"] = image_evidence["sha256"]
        source = row["source"]
        if isinstance(source, str):
            source = json.loads(source)
        occurrences.append(MatchOccurrence(f"{session_id}:{row['ordinal']}", session_id,
            SourceIdentity.from_dict(source), rank - 1, diagnostics=occurrence_diagnostics,
            rank=rank, matched_pixels=result.matching_pixels, overlap=result.overlap_fraction,
            mirrored=result.mirrored, shift=result.shift, member_id=row["member_key"],
            candidate_ordinal=row["ordinal"], input_hashes=input_hashes, evidence_ref=evidence_ref,
            shape=shape))
    shape_execution = None
    if shape_summary is not None:
        # Consumed scoring evidence includes un-emitted shortlist members;
        # published membership remains independently verifiable from the rows.
        # A skipped stage binds an empty list for both digests.
        shape_execution = ShapeExecution(
            algorithm=SHAPE_ALGORITHM, reference_revision=shape_reference,
            implementation_revision=SHAPE_IMPLEMENTATION_REVISION,
            gradient_producer_reference_revision=PRODUCER_REVISION,
            gradient_producer_implementation_revision=IMPLEMENTATION,
            gradient_format=GRADIENT_FORMAT,
            scored_cohort_digest=hashlib.sha256(json.dumps(
                cohort_entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                allow_nan=False).encode("utf-8")).hexdigest(),
            parameters=shape_summary["parameters"], profile_id=profile.profile_id,
            shape_stage_executed=shape_summary["executed"], positive_count=matched,
            stage1_cap=shape_summary["stage1_cap"], scoring_budget=shape_summary["budget"],
            shortlist_count=shape_summary["shortlist"], fl_positive=shape_summary["fl_positive"],
            max_positive_score=shape_summary["max_score"], max_shape_gap=shape_summary["max_area"],
            emitted_count=len(occurrences) if shape_summary["executed"] else 0,
            membership_digest=gradient_membership_digest(
                tuple(occurrences) if shape_summary["executed"] else ()))
    query_evidence = query_image_evidence(query_record)
    warnings = list(query_evidence["warnings"]) + list(query_record.get("out_of_date", []))
    evidence = {
        "schema_version": 3,
        "backend": "madi3d-local", "algorithm": ALGORITHM, "reference_revision": REFERENCE_REVISION,
        "implementation_revision": "madi3d-positive-cds-v1",
        "query_id": query_record["query_id"], "query_png_sha256": query_evidence["png_file_sha256"],
        "query_pixel_sha256": query_evidence["rgb_pixel_sha256"],
        "query_alignment_quality": "unverified" if warnings else "see_retained_mapping_evidence",
        "query_foreground_pixels": foreground_count,
        **identity,
        "search_profile": profile.profile_id, "anatomical_area": profile.anatomical_area,
        "alignment_space": profile.alignment_space, "search_canvas_yx": list(profile.search_canvas_yx),
        "score_exclusions": profile.score_exclusions, "searched_collections": [reader.searched_collection],
        "library_completeness": "complete",
        "counts": {"total": total, "examined": examined, "failed": 0, "matched": matched,
                   "retained": len(occurrences)},
        "completion_status": "completed", "results_truncated": matched > len(occurrences),
        "citations": list(reader.citations),
    }
    # Retained identity observations live in external evidence; the searched
    # distribution is identified by the library ID and installed archive digest
    # projected into the context from the installation record.
    diagnostics = tuple(Diagnostic("query_mapping_warning", text) for text in warnings)
    if shape_request is not None:
        diagnostics += tuple(Diagnostic("shape_association_warning", text)
                             for text in shape_request.association["warnings"])
    session = SearchSession(session_id, "custom_query", identity["data_version"],
        query_reference=query_record["query_id"],
        parameters=tuple(SourceParameter(k, v) for k, v in parameters.to_dict().items()),
        context=evidence, external_evidence=external, diagnostics=diagnostics,
        shape_execution=shape_execution)
    checkpoint(cancel)
    return SearchResults(session, tuple(occurrences))
