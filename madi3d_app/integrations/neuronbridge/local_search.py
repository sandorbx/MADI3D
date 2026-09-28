"""Offline exhaustive positive-CDS search and the existing result handoff."""
import heapq
import hashlib
import io
import json
from uuid import uuid4

import numpy as np
from PIL import Image

from .local_catalog import Library, canonical, compatible_query, query_profile
from .local_library import INDEX_FORMAT, read_sparse
from .local_scorer import ALGORITHM, REFERENCE_REVISION, PreparedQuery, SearchParameters, checkpoint
from .query import QUERY_KEY, query_png, query_image_evidence, validate_queries
from .records import Diagnostic, MatchOccurrence, SearchResults, SearchSession, SourceIdentity, SourceParameter

JOB_KEY = "neuronbridge_local_search_jobs"


def search_library(manager, snapshot_id, query_record, parameters=None, cancel=None, progress=None, *, session_id=None):
    """All declared candidates or an error. No network or source-image decoding."""
    parameters = parameters or SearchParameters()
    validate_queries({QUERY_KEY: {query_record["query_id"]: query_record}})
    if not compatible_query(query_record):
        raise ValueError("The retained query is incompatible with the supported CDM profiles.")
    with Image.open(io.BytesIO(query_png(query_record))) as image:
        rgb = np.asarray(image).copy()
    profile = query_profile(query_record)
    prepared = PreparedQuery(rgb, parameters, cancel, profile_id=profile.profile_id)
    del rgb
    report = progress or (lambda phase, done, total: None)
    heap, examined, matched, retained_bytes = [], 0, 0, 0
    # Bound additional working data, independently of total collection size.
    # 48 MiB covers query, a 16 MiB mapped shard, and block temporaries. Runtime
    # and Qt's pre-existing memory are outside this feature's incremental budget.
    heap_budget = (parameters.memory_mb - 48) * 1024 * 1024
    with manager.indexed_candidates(snapshot_id, cancel) as (state, candidates):
        library = Library(**state["library"])
        if not compatible_query(query_record, library):
            raise ValueError("Query and library search profiles disagree; Brain and VNC cannot be combined.")
        total = library.count
        report("Searching", 0, total)
        for row, pixels in candidates:
            checkpoint(cancel)
            result = prepared.score_pixels(lambda positions: read_sparse(pixels, positions), cancel)
            examined += 1
            # batch_search.js uses strict greater-than for the minimum ratio.
            if result.overlap_fraction > parameters.minimum_overlap:
                matched += 1
                key = (result.matching_pixels, -row["ordinal"])
                if len(heap) < parameters.result_limit or key > heap[0][:2]:
                    row = dict(row)
                    identity = manager.identity_observation(snapshot_id, json.loads(row["metadata_evidence"]), cancel)
                    row["metadata_evidence"] = canonical(identity)
                    size = len(canonical(row).encode("utf-8")) * 4 + 1024
                    if len(heap) == parameters.result_limit:
                        retained_bytes -= heapq.heappop(heap)[4]
                    if retained_bytes + size > heap_budget:
                        raise ValueError("Retained hit metadata exceeds the search memory budget. Reduce the result limit or increase the budget.")
                    heapq.heappush(heap, (*key, row, result, size))
                    retained_bytes += size
            report("Searching", examined, total)
        checkpoint(cancel)
        if examined != total:
            raise ValueError("Search did not examine every declared candidate; no ranking was published.")
    session_id = session_id or str(uuid4())
    ordered = sorted(heap, key=lambda item: (-item[0], -item[1]))
    query_evidence = query_image_evidence(query_record)
    warnings = list(query_evidence["warnings"]) + list(query_record.get("out_of_date", []))
    evidence = {
        "schema_version": 1,
        "backend": "madi3d-local", "algorithm": ALGORITHM, "reference_revision": REFERENCE_REVISION,
        "implementation_revision": "madi3d-positive-cds-v1",
        "index_format": INDEX_FORMAT,
        "query_id": query_record["query_id"], "query_png_sha256": query_evidence["png_file_sha256"],
        "query_pixel_sha256": query_evidence["rgb_pixel_sha256"],
        "query_alignment_quality": "unverified" if warnings else "see_retained_mapping_evidence",
        "query_foreground_pixels": prepared.foreground_count,
        "snapshot_id": snapshot_id, "inventory_sha256": state["inventory_sha256"],
        "manifest_sha256": state["manifest"]["sha256"],
        "library": library.name, "library_release": library.release, "data_version": library.data_version,
        "search_profile": profile.profile_id, "anatomical_area": profile.anatomical_area,
        "alignment_space": profile.alignment_space, "search_canvas_yx": list(profile.search_canvas_yx),
        "score_exclusions": profile.score_exclusions, "searched_collections": [library.name],
        "library_completeness": "complete",
        "counts": {"total": total, "examined": examined, "failed": 0, "matched": matched, "retained": len(ordered)},
        "completion_status": "completed", "results_truncated": matched > len(ordered),
        "citations": list(library.citations),
    }
    # Runtime search views may use the installed catalog/manifest. Project
    # persistence projects only selected observations and snapshot digests.
    external = {"catalog": library.catalog, "manifest": state["manifest"]}
    occurrences = []
    for rank, (_, _, row, result, _) in enumerate(ordered):
        checkpoint(cancel)
        identity_evidence = json.loads(row["metadata_evidence"])
        image_evidence = json.loads(row["image_evidence"])
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
        diagnostics = tuple(Diagnostic("local_identity_unresolved", text, row_index=rank)
                            for text in json.loads(row["diagnostics"]))
        occurrences.append(MatchOccurrence(f"{session_id}:{row['ordinal']}", session_id,
            SourceIdentity.from_dict(json.loads(row["source"])), rank, diagnostics=diagnostics,
            rank=rank + 1, matched_pixels=result.matching_pixels, overlap=result.overlap_fraction,
            mirrored=result.mirrored, shift=result.shift, member_id=row["member_key"],
            candidate_ordinal=row["ordinal"],
            input_hashes={"search_image": image_evidence["sha256"], "decoded_rgb": row["pixel_sha256"]},
            evidence_ref=evidence_ref))
    diagnostics = tuple(Diagnostic("query_mapping_warning", text) for text in warnings)
    session = SearchSession(session_id, "custom_query", library.data_version,
        query_reference=query_record["query_id"],
        parameters=tuple(SourceParameter(k, v) for k, v in parameters.to_dict().items()),
        context=evidence, external_evidence=external, diagnostics=diagnostics)
    checkpoint(cancel)
    return SearchResults(session, tuple(occurrences))
