"""Offline exhaustive positive-CDS search and the existing result handoff."""
import heapq
import io
import json
import time
from uuid import uuid4

import numpy as np
from PIL import Image

from .local_catalog import Library, canonical, compatible_query, query_profile, now
from .local_library import INDEX_FORMAT, read_sparse
from .local_scorer import ALGORITHM, REFERENCE_REVISION, PreparedQuery, SearchParameters, checkpoint
from .query import QUERY_KEY, query_png, query_image_evidence, validate_queries
from .records import Diagnostic, MatchOccurrence, ResultField, SearchResults, SearchSession, SourceIdentity, SourceParameter

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
    began = time.perf_counter()
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
        "index_format": INDEX_FORMAT, "parameters": parameters.to_dict(),
        "query_id": query_record["query_id"], "query_png_sha256": query_evidence["png_file_sha256"],
        "query_pixel_sha256": query_evidence["rgb_pixel_sha256"],
        "query_mapping": query_evidence["mapping"], "query_warnings": warnings,
        "query_alignment_quality": "unverified" if warnings else "see_retained_mapping_evidence",
        "query_foreground_pixels": prepared.foreground_count,
        "snapshot_id": snapshot_id, "inventory_sha256": state["inventory_sha256"],
        "manifest_sha256": state["manifest"]["sha256"], "manifest_source": state["manifest"],
        "library": library.name, "library_release": library.release, "data_version": library.data_version,
        "search_profile": profile.profile_id, "anatomical_area": profile.anatomical_area,
        "alignment_space": profile.alignment_space, "search_canvas_yx": list(profile.search_canvas_yx),
        "score_exclusions": profile.score_exclusions, "searched_collections": [library.name],
        "library_completeness": "complete", "catalog_evidence": library.catalog,
        "counts": {"total": total, "examined": examined, "failed": 0, "matched": matched, "retained": len(ordered)},
        "completion_status": "completed", "results_truncated": matched > len(ordered),
        "result_limit_scope": "retained hits only; all library candidates examined",
        "tie_order": "matching pixels descending, manifest ordinal ascending; unmirrored and first shift win score ties",
        "completed_at": now(), "elapsed_seconds": time.perf_counter() - began,
        "citations": list(library.citations),
    }
    session = SearchSession(session_id, "custom_query", library.data_version,
        query_reference=query_record["query_id"], parameters=(SourceParameter("local_search", evidence),))
    occurrences = []
    for rank, (_, _, row, result, _) in enumerate(ordered):
        checkpoint(cancel)
        identity_evidence = json.loads(row["metadata_evidence"])
        image_evidence = json.loads(row["image_evidence"])
        supplied = [
            ("Local rank", rank + 1, "rank"),
            ("Local matching pixels", result.matching_pixels, "matched_pixels"),
            ("Local query overlap fraction", result.overlap_fraction, "score"),
            ("Mirrored comparison", result.mirrored, "mirror"),
            ("Query shift before mirroring (XY pixels)", canonical(result.shift), "unknown"),
            ("Library member", row["member_key"], "unknown"),
            ("Search image SHA-256", image_evidence["sha256"], "unknown"),
            ("Decoded RGB SHA-256", row["pixel_sha256"], "unknown"),
            ("Search image source evidence", canonical(image_evidence), "unknown"),
            ("Identity metadata evidence", canonical(identity_evidence), "unknown"),
        ]
        fields = tuple(ResultField(i, name, str(value), value, role) for i, (name, value, role) in enumerate(supplied))
        diagnostics = tuple(Diagnostic("local_identity_unresolved", text, row_index=rank)
                            for text in json.loads(row["diagnostics"]))
        occurrences.append(MatchOccurrence(f"{session_id}:{row['ordinal']}", session_id,
            SourceIdentity.from_dict(json.loads(row["source"])), rank, fields, diagnostics))
    diagnostics = tuple(Diagnostic("query_mapping_warning", text) for text in warnings)
    checkpoint(cancel)
    return SearchResults(session, tuple(occurrences), diagnostics)
