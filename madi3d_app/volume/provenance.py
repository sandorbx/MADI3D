"""Canonical, data-only provenance descriptors for scientific volumes."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import uuid
from typing import Any, Iterable, Mapping

import numpy as np


VOLUME_COMPANION_SCHEMA = "madi3d.volume-provenance.v3"


def volume_is_label(snapshot):
    metadata = snapshot.get("madi_metadata") or {}
    return bool(snapshot.get("is_label") or snapshot.get("segmentation_mask")
                or snapshot.get("channel_role") == "label/mask" or metadata.get("is_label")
                or metadata.get("derived_operation") == "segmentation_mask"
                or metadata.get("semantic_type") in {"label", "labels", "mask", "segmentation"})


def validate_label_conversion(array, dtype, background=None):
    """Bounded exact-value checks protect categorical meaning at conversion boundaries."""
    dtype = np.dtype(dtype)
    if background is not None:
        if not np.isfinite(background) or background != np.rint(background) or float(np.asarray(background).astype(dtype)) != background:
            raise ValueError("Mask background must be an exactly representable integer label.")
    for frame in np.asanyarray(array).reshape((-1, *array.shape[-2:])):
        if not np.all(np.isfinite(frame)) or np.any(frame != np.rint(frame)):
            raise ValueError("Mask/label voxels must contain finite integer identities.")
        if frame.dtype != dtype and np.any(frame.astype(dtype).astype(frame.dtype) != frame):
            raise ValueError("The requested dtype conversion changes label identities. Use a lossless output dtype.")


def volume_artifact_format(path):
    """Container identity includes compound suffixes, independently of staging names."""
    from pathlib import Path
    name = str(path).lower()
    if name.endswith((".nii", ".nii.gz")):
        return "nifti"
    return {".nrrd": "nrrd", ".tif": "tiff", ".tiff": "tiff", ".h5j": "h5j"}.get(Path(name).suffix, "")


def serialized_volume_facts(probe):
    """Describe the inspected container, never historical source observations."""
    from .geometry import general_grid_affine_from_components
    affine = None
    if all(value is not None for value in (probe.spacing, probe.origin, probe.direction)):
        affine = general_grid_affine_from_components(probe.origin, probe.spacing, probe.direction).tolist()
    time_units = probe.raw_fields.get("time_units", probe.time_units) if probe.container_format == "nifti" else probe.time_units
    return json.loads(json.dumps({
        "format": {"imagej-tiff": "tiff", "ome-tiff": "tiff"}.get(probe.container_format, probe.container_format),
        "dimensions_xyz": probe.dimensions,
        "axes": [{"label": axis.label, "semantic": axis.semantic, "size": axis.size}
                 for axis in probe.axis_semantics],
        "channel_count": probe.channel_count, "channel_selectors": probe.channel_selectors,
        "channel_names": probe.channel_names,
        "dtype": probe.scalar_dtype, "bit_depth": probe.scalar_bit_depth,
        "channel_dtypes": probe.channel_scalar_dtypes,
        "time": {"count": probe.time_count, "interval": probe.time_interval, "units": time_units},
        "grid": {"affine": affine, "space_units": probe.space_units},
    }, allow_nan=False))


def serialized_affines_agree(actual, expected, output_format):
    """NIfTI-1 sform coefficients use float32; other writers retain float64/text precision."""
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if actual.shape != (4, 4) or expected.shape != (4, 4):
        return False
    tolerance = np.full((4, 4), 1e-10)
    if output_format == "nifti":
        tolerance += 4 * np.abs(np.spacing(expected.astype(np.float32)).astype(float))
    return bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
                and np.all(np.abs(actual - expected) <= tolerance))


def validate_serialized_volume(probe, snapshots, grid, output_format, output_dtype):
    """Check the post-write structure and the representable output grid before publication."""
    from .geometry import canonical_space_units, general_grid_affine_from_components
    snapshots = list(snapshots)
    facts = serialized_volume_facts(probe)
    if probe.errors or probe.requires_axis_resolution or probe.requires_series_selection:
        raise ValueError("Serialized volume is not unambiguous: " + "; ".join((*probe.errors, *probe.ambiguities)))
    expected_dims = tuple(grid.get("dims") or grid.get("dims_xyz") or ())
    expected_time = int(snapshots[0].get("export_frame_count", 1))
    if (facts["format"] != output_format or tuple(probe.dimensions) != expected_dims
            or probe.channel_count != len(snapshots) or probe.time_count != expected_time):
        raise ValueError("Serialized volume dimensions, container, or C/T structure disagree with the export plan.")
    dtype = "uint8" if output_format == "h5j" else np.dtype(output_dtype).name
    if probe.scalar_dtype is None or np.dtype(probe.scalar_dtype).name != dtype:
        raise ValueError("Serialized volume dtype disagrees with the export plan.")
    units = grid.get("space_units", snapshots[0].get("space_units"))
    if ((canonical_space_units(probe.space_units) if probe.space_units else None)
            != (canonical_space_units(units) if units else None)):
        raise ValueError("Serialized volume units disagree with the export plan.")
    if facts["grid"]["affine"] is None or not serialized_affines_agree(facts["grid"]["affine"], grid["affine"], output_format):
        raise ValueError("Serialized volume grid disagrees with the export plan.")
    embedded = probe.madi3d_geometry_provenance
    channels = embedded.get("channels", [embedded])
    if len(channels) != len(snapshots):
        raise ValueError("Embedded volume channel structure disagrees with the serialized file.")
    for record in channels:
        working = record.get("output_working_grid", {})
        if working:
            affine = working.get("affine")
            if affine is None:
                affine = general_grid_affine_from_components(working["origin"], working["spacing"], working["direction"])
            if (tuple(working.get("dimensions", ())) != probe.dimensions
                    or not serialized_affines_agree(facts["grid"]["affine"], affine, output_format)):
                raise ValueError("Embedded MADI3D geometry disagrees with the serialized file.")
    # Temporal units may be format-normalized (e.g. seconds -> sec in NIfTI).
    if expected_time > 1:
        aliases = {"s": "sec", "second": "sec", "seconds": "sec", "ms": "msec", "millisecond": "msec", "us": "usec"}
        def time_unit(value):
            unit = str(value or "frame").casefold()
            return aliases.get(unit, unit)
        expected_units = time_unit(snapshots[0].get("time_units", "frame"))
        if output_format == "nifti":
            from .source_formats import nifti_time_unit
            expected_units = nifti_time_unit(snapshots[0].get("time_units", "frame"))
        if (time_unit(facts["time"]["units"]) != expected_units
                or not np.isclose(probe.time_interval, snapshots[0].get("time_spacing", 1.0), rtol=4*np.finfo(np.float32).eps, atol=1e-12)):
            raise ValueError("Serialized time calibration disagrees with the export plan.")
    return facts


def volume_companion_projection(model, snapshots, project_metadata=None, *, source_file=None, operations=(), parents_by_channel=None, evidence_index=None, origins=()):
    """One detached backward closure and one shared scoped NB evidence projection."""
    from madi3d_app.project.scientific_records import (
        volume_reference, create_local_reference, export_provenance_projection,
        HISTORY_KEY, REFERENCE_KEY,
    )
    from collections import ChainMap
    candidate = copy.copy(model)
    candidate.unavailable_history_record_ids = set(model.unavailable_history_record_ids)
    for name in ("scientific_records", "operation_records", "geometry_records", "geometry_events", "backing_sources"):
        setattr(candidate, name, ChainMap({}, getattr(model, name)))
    metadata = dict(project_metadata or {})
    refs, channel_metadata = [], []
    extra_refs = []
    operations = list(operations)
    for index, snap in enumerate(snapshots):
        origin = copy.deepcopy(snap.get("object_metadata") or snap.get("madi_metadata") or {})
        parents = [volume_reference(candidate, snap["channel_id"])["record_id"]] if snap.get("channel_id") in candidate.channels else []
        if origin.get(REFERENCE_KEY):
            parents.append(origin[REFERENCE_KEY]["record_id"])
            extra_refs.append(origin[REFERENCE_KEY])
        if parents_by_channel is not None:
            parents = list(dict.fromkeys([*parents, *parents_by_channel[index]]))
        operation = operations[index] if index < len(operations) else None
        if operation is None and snap.get("channel_id") not in model.channels:
            operation = (snap.get("scientific_provenance") or {}).get("producing_operation")
        from .model import _canonical_operation_record, _operation_payloads_equal
        for support in (snap.get("scientific_provenance") or {}).get("supporting_operations", ()):
            canonical_support = _canonical_operation_record(support)
            operation_id = canonical_support["operation_id"]
            prior = candidate.operation_records.get(operation_id)
            if prior is not None and not _operation_payloads_equal(
                prior, canonical_support
            ):
                raise ValueError("Conflicting volume producer history.")
            candidate.operation_records[operation_id] = canonical_support
        if operation:
            operation = _canonical_operation_record(operation)
            authored_inputs = list(operation.get("input_record_ids", ()))
            if parents and authored_inputs and parents != authored_inputs:
                raise ValueError("Volume producer inputs disagree with the captured scientific records.")
            direct_inputs = parents or authored_inputs
            if direct_inputs:
                operation["input_record_ids"] = direct_inputs
            else:
                operation.pop("input_record_ids", None)
            if operation.get("operation_id") in candidate.operation_records:
                if index < len(operations):
                    if not _operation_payloads_equal(
                        candidate.operation_records[operation["operation_id"]], operation
                    ):
                        raise ValueError("Conflicting volume producer identity.")
                operation = copy.deepcopy(candidate.operation_records[operation["operation_id"]])
            # create_local_reference publishes a new serialization result using the existing graph owner.
            source_payload = {key: copy.deepcopy(origin[key]) for key in
                              ("neuronbridge", "scene_annotations", "object_name", "warnings", "decisions", "annotations", "history_diagnostics", "source_warnings") if key in origin}
            refs.append(create_local_reference(candidate, "object_origin", source_payload, parents=parents,
                                               operation=operation.get("operation_type") or operation.get("operation") or "volume_export",
                                               operation_record=operation))
        elif parents:
            refs.append({"object_id": str(snap.get("channel_id") or uuid.uuid4()), "record_id": parents[0]})
        else:
            refs.append(create_local_reference(candidate, "volume_source", {
                "history_state": "unavailable", "reason": "Scientific input and producer records were not supplied for this artifact."}))
        # Rendering and previous serialization bookkeeping never become scientific annotations.
        channel_metadata.append({key: copy.deepcopy(origin[key]) for key in
                                 ("neuronbridge", "scene_annotations", "object_name", "warnings", "decisions", "annotations", "history_diagnostics", "source_warnings") if key in origin})
    # The shared helper finds origins in the closure; include direct channel origins as sources too.
    projection = export_provenance_projection({}, candidate, metadata, references=[*refs, *extra_refs],
        source_file=source_file, evidence_index=evidence_index, origins=[*origins, *channel_metadata])
    return {"scientific_history": projection.pop(HISTORY_KEY), "evidence": projection,
            "references": refs, "channel_metadata": channel_metadata}


def capture_volume_inputs(
    model,
    snapshots,
    project_metadata,
    *,
    source_file=None,
    evidence_index=None,
    operation_capture=None,
    supporting_operation_ids=(),
):
    """Freeze consumed descriptors in the existing owner; no export protocol or pixels.

    All selected inputs share one detached scientific subset. Portable evidence
    projection and serialization happen only when an output is actually written.
    Supporting operations may contribute their exact historical input revisions
    to that detached subset without making those revisions direct output parents.
    """
    from collections import ChainMap
    from madi3d_app.project.scientific_records import volume_reference, history_model, REFERENCE_KEY
    from madi3d_app.integrations.neuronbridge.evidence import validate_evidence
    candidate = copy.copy(model)
    candidate.scientific_records = ChainMap({}, model.scientific_records)
    candidate.unavailable_history_record_ids = set(model.unavailable_history_record_ids)
    snapshots = list(snapshots)
    operation_references = None
    if operation_capture is not None:
        direct_records = operation_capture.get("direct_records")
        operation_references = list(operation_capture.get("references") or ())
        if not isinstance(direct_records, Mapping):
            raise ValueError(
                "Portable volume capture requires direct operation input records."
            )
        if len(operation_references) != len(snapshots):
            raise ValueError(
                "Portable volume capture does not match its direct operation inputs."
            )
        from madi3d_app.project.scientific_records import insert_records
        insert_records(candidate, direct_records.values())
    refs, origins, by_channel = [], [], {}
    for index, snapshot in enumerate(snapshots):
        local_refs = []
        if operation_references is not None:
            reference = copy.deepcopy(operation_references[index])
            record_id = str(reference.get("record_id") or "")
            record = candidate.scientific_records.get(record_id)
            expected_channel_id = str(snapshot.get("channel_id") or "")
            if (
                record is None
                or record.get("record_kind") != "volume_revision"
                or str(record.get("channel_id") or "") != expected_channel_id
            ):
                raise ValueError(
                    "Portable volume capture is not aligned with its direct consumed volume revision."
                )
            local_refs.append(reference)
        elif snapshot.get("channel_id") in candidate.channels:
            local_refs.append(volume_reference(candidate, snapshot["channel_id"]))
        origin = snapshot.get("object_metadata") or snapshot.get("madi_metadata") or {}
        if origin.get(REFERENCE_KEY) and not origin[REFERENCE_KEY].get("unbound"):
            local_refs.append(copy.deepcopy(origin[REFERENCE_KEY]))
        refs.extend(local_refs)
        selected_origin = {key: copy.deepcopy(origin[key]) for key in
            ("neuronbridge", "scene_annotations", "object_name", "warnings", "decisions", "annotations", "history_diagnostics", "source_warnings") if key in origin}
        origins.append(selected_origin)
        if snapshot.get("channel_id") not in by_channel or selected_origin:
            by_channel[snapshot.get("channel_id")] = (local_refs, selected_origin)
    if evidence_index is None:
        evidence = validate_evidence((), project_metadata, origins=origins)
        project_metadata = evidence.freeze_metadata(project_metadata)
        evidence_index = evidence.index
    supporting_operation_ids = list(dict.fromkeys(
        str(operation_id)
        for operation_id in supporting_operation_ids
        if operation_id not in (None, "") and str(operation_id).strip()
    ))
    support_record_ids = list(dict.fromkeys(
        str(record_id)
        for operation_id in model.operation_dependency_closure(
            supporting_operation_ids
        )
        for record_id in model.operation_records[operation_id].get(
            "input_record_ids", ()
        )
        if str(record_id).strip()
    ))
    history_references = [
        *refs,
        *({"record_id": record_id} for record_id in support_record_ids),
    ]
    return {"model": history_model(candidate, history_references), "references": refs, "origins": origins, "by_channel": by_channel,
            "project_metadata": dict(project_metadata), "evidence_index": evidence_index, "source_file": source_file}


def capture_volume_operation_inputs(model, consumed_snapshots):
    """Capture only direct consumed volume revisions for an in-project result.

    Unlike :func:`capture_volume_inputs`, this does not construct portable
    history.  The returned records are a bounded runtime envelope which may be
    installed atomically with one completed operation.
    """
    from collections import ChainMap
    from madi3d_app.project.scientific_records import (
        validate_records,
        volume_revision_record,
    )
    from madi3d_app.volume.model import _scientific_record_payloads_equal

    direct_records = {}
    references = []
    for snapshot in consumed_snapshots or ():
        snapshot = dict(snapshot or {})
        descriptor = dict(snapshot.get("descriptor") or {})
        dependency = dict(snapshot.get("dependency_revision") or {})
        source_id = snapshot.get("source_id") or descriptor.get("source_id")
        channel_id = snapshot.get("channel_id") or descriptor.get("channel_id")
        backing_source_id = (
            snapshot.get("backing_source_id")
            or descriptor.get("backing_source_id")
        )
        if not all((source_id, channel_id, backing_source_id)):
            raise ValueError(
                "A modeled-volume operation capture requires source, channel, and backing identities."
            )
        selector = snapshot.get("selector")
        if selector is None:
            selector = snapshot.get("channel_selector", snapshot.get("channel"))
        if selector is None:
            selector = descriptor.get("selector", descriptor.get("channel_selector"))
        consumed_time_index = snapshot.get(
            "registration_time_index", snapshot.get("time_index")
        )
        if consumed_time_index is not None:
            if isinstance(consumed_time_index, bool):
                raise ValueError("Consumed volume time index must be an integer.")
            try:
                consumed_time_index = int(consumed_time_index)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Consumed volume time index must be an integer.") from exc
            if consumed_time_index < 0:
                raise ValueError("Consumed volume time index must be non-negative.")
            if isinstance(selector, dict):
                selector = {**copy.deepcopy(selector), "time_index": consumed_time_index}
            else:
                selector = {
                    "source_selector": copy.deepcopy(selector),
                    "time_index": consumed_time_index,
                }
        local_revision = (
            snapshot.get("geometry_revision")
            or descriptor.get("geometry_revision")
            or dependency.get("geometry_revision")
        )
        acquisition_revision = (
            snapshot.get("acquisition_geometry_revision")
            or descriptor.get("acquisition_geometry_revision")
        )
        producer = snapshot.get("producing_operation_id")
        if producer is None:
            producer = descriptor.get("producing_operation_id")
        if producer is None:
            source_operations = snapshot.get("source_operation_ids") or descriptor.get("source_operation_ids") or ()
            producer = source_operations[0] if len(source_operations) == 1 else None
        record = volume_revision_record(model, {
            "source_id": str(source_id),
            "channel_id": str(channel_id),
            "backing_source_id": str(backing_source_id),
            "selector": selector,
            "geometry_revision_id": local_revision,
            "acquisition_geometry_revision_id": acquisition_revision,
            "shared_pose": snapshot.get("shared_pose", snapshot.get("actor_matrix")),
            "producing_operation_id": producer,
            "source_checksum": snapshot.get("source_checksum", descriptor.get("source_checksum")),
        })
        existing = model.scientific_records.get(record["record_id"])
        if existing is not None and not _scientific_record_payloads_equal(existing, record):
            raise ValueError(
                f"Scientific record {record['record_id']!r} has conflicting payloads."
            )
        direct_records[record["record_id"]] = record
        references.append({"object_id": str(channel_id), "record_id": record["record_id"]})
    candidate = copy.copy(model)
    candidate.scientific_records = ChainMap(direct_records, model.scientific_records)
    validate_records(candidate, record_ids=direct_records, operation_ids=())
    return {"direct_records": direct_records, "references": references}


def _captured_input_candidate(model, captures):
    """Validate detached input closures without changing the live owner."""
    from collections import ChainMap
    candidate = copy.copy(model)
    for name in (
        "scientific_records",
        "operation_records",
        "geometry_records",
        "geometry_events",
        "backing_sources",
    ):
        setattr(candidate, name, ChainMap({}, getattr(model, name)))
    candidate.unavailable_history_operation_ids = set(
        model.unavailable_history_operation_ids
    )
    candidate.unavailable_history_record_ids = set(model.unavailable_history_record_ids)
    seen = set()
    for capture in captures:
        if capture is None or id(capture) in seen:
            continue
        seen.add(id(capture))
        direct_records = capture.get("direct_records")
        if direct_records is not None:
            from madi3d_app.project.scientific_records import insert_records
            insert_records(candidate, direct_records.values())
        else:
            from madi3d_app.project.scientific_records import merge_history
            merge_history(
                candidate,
                capture["model"],
                source_file=capture.get("source_file"),
            )
    return candidate


def _commit_captured_input_candidate(model, candidate):
    for name in (
        "scientific_records",
        "operation_records",
        "geometry_records",
        "geometry_events",
        "backing_sources",
    ):
        current = getattr(model, name)
        current.update(getattr(candidate, name).maps[0])
    model.unavailable_history_operation_ids = set(
        candidate.unavailable_history_operation_ids
    )
    model.unavailable_history_record_ids = set(candidate.unavailable_history_record_ids)


def merge_captured_volume_inputs(model, captures):
    """Publish only previously detached, validated consumed-volume closures."""
    candidate = _captured_input_candidate(model, captures)
    _commit_captured_input_candidate(model, candidate)


def publish_captured_volume_operation(model, captures, operation_record):
    """Atomically publish captured volume records and their completed operation."""
    captures = [capture for capture in captures if capture is not None]
    candidate = _captured_input_candidate(model, captures)
    operation = copy.deepcopy(dict(operation_record))
    direct_inputs = list(operation.get("input_record_ids", ()))
    direct_inputs.extend(
        reference["record_id"]
        for capture in captures
        for reference in capture.get("references", ())
    )
    direct_inputs = list(dict.fromkeys(direct_inputs))
    if direct_inputs:
        operation["input_record_ids"] = direct_inputs
    else:
        operation.pop("input_record_ids", None)
    operation_id = candidate.publish_operation(operation)
    _commit_captured_input_candidate(model, candidate)
    return operation_id


def volume_result_projection(
    inputs, snapshots, operations, *, supporting_operations=()
):
    """Project output history from the frozen dispatch inputs at the writer boundary."""
    from .model import (
        VolumeSourceModel,
        _canonical_operation_record,
        _operation_payloads_equal,
    )
    from madi3d_app.project.scientific_records import merge_history
    model, parents, origins = VolumeSourceModel(), [], []
    seen = set()
    metadata, index, source_file = {}, None, None
    for capture in inputs:
        if id(capture) in seen:
            continue
        seen.add(id(capture))
        merge_history(model, capture["model"], source_file=capture["source_file"])
        parents.extend(ref["record_id"] for ref in capture["references"])
        origins.extend(capture["origins"])
        if metadata and metadata != capture["project_metadata"]:
            raise ValueError("Result inputs must share their captured project evidence.")
        metadata, index, source_file = capture["project_metadata"], capture["evidence_index"], capture["source_file"]
    if supporting_operations:
        candidate = copy.copy(model)
        candidate.operation_records = dict(model.operation_records)
        support_ids = []
        for raw_record in supporting_operations:
            record = _canonical_operation_record(raw_record)
            operation_id = record.get("operation_id")
            if not operation_id:
                raise ValueError(
                    "Supporting scientific operations require a stable operation_id."
                )
            prior = candidate.operation_records.get(operation_id)
            if prior is not None and not _operation_payloads_equal(prior, record):
                raise ValueError(
                    f"Scientific operation {operation_id!r} has conflicting payloads."
                )
            candidate.operation_records[operation_id] = record
            support_ids.append(operation_id)
        candidate.operation_dependency_closure(support_ids)
        model = candidate
    return volume_companion_projection(model, snapshots, metadata, operations=operations,
        parents_by_channel=[list(dict.fromkeys(parents))]*len(snapshots), evidence_index=index,
        source_file=source_file, origins=origins)



def volume_companion_content(probe, snapshots, grid, options, projection):
    """Final capsule content describes actual serialized facts plus scoped lineage."""
    from madi3d_app.project.scientific_records import export_coordinates
    first = snapshots[0]
    coordinates = export_coordinates(first.get("actor_matrix"), baked=bool(options.get("bake_transform", True)),
        units=list(probe.space_units) if probe.space_units else "unknown",
        coordinate_space=first.get("coordinate_space_id") or "unknown")
    frame_count = probe.time_count
    frames = list(range(int(first.get("time_count", frame_count)))) if frame_count > 1 else [int(first.get("time_index", 0))]
    channels = [{"scientific_reference": projection["references"][index],
                 "name": snap.get("channel_name") or snap.get("display_name"),
                 "role": "label/mask" if volume_is_label(snap) else snap.get("channel_role", "other"),
                 "conversion": snap.get("serialized_conversion") or {"source_dtype": str(snap.get("dtype", probe.scalar_dtype))},
                 "metadata": projection.get("channel_metadata", [{}]*len(snapshots))[index]}
                for index, snap in enumerate(snapshots)]
    return {"volume": {"coordinates": coordinates, "selected_frame_indices": frames},
            "channels": channels, "scientific_history": projection["scientific_history"],
            "evidence": projection.get("evidence", {})}


def read_volume_companion(path, probe, *, cancel_check=None):
    from madi3d_app.project.scientific_records import read_artifact_companion, validate_export_coordinates, validate_reference
    from .model import VolumeSourceModel
    from madi3d_app.integrations.neuronbridge.evidence import validate_evidence
    local = probe.madi3d_geometry_provenance.get("channels", [probe.madi3d_geometry_provenance])
    generation = local[0].get("serialization_generation") if local and isinstance(local[0], dict) else None
    capsule = read_artifact_companion(path, schema=VOLUME_COMPANION_SCHEMA,
        output_format=volume_artifact_format(path), generation=generation, cancel_check=cancel_check)
    if not capsule:
        return {}
    if capsule.get("scope") != "volume_lineage" or not isinstance(capsule.get("volume"), dict):
        raise ValueError("Invalid volume lineage scope.")
    channels = capsule.get("channels")
    if not isinstance(channels, list) or len(channels) != probe.channel_count:
        raise ValueError("Volume companion channel count disagrees with the serialized file.")
    coordinates = validate_export_coordinates(capsule["volume"]["coordinates"])
    if coordinates["units"] != (list(probe.space_units) if probe.space_units else "unknown"):
        raise ValueError("Companion units disagree with the serialized file.")
    frames = capsule["volume"].get("selected_frame_indices")
    if not isinstance(frames, list) or len(frames) != probe.time_count or any(type(i) is not int or i < 0 for i in frames):
        raise ValueError("Invalid exported frame selection.")
    incoming = VolumeSourceModel.from_dict(capsule["scientific_history"])
    origins = [record["payload"] for record in incoming.scientific_records.values()
               if record.get("record_kind") == "source" and record.get("domain") == "object_origin"]
    for index, channel in enumerate(channels):
        reference = validate_reference(incoming, channel["scientific_reference"])
        if not isinstance(channel.get("role"), str) or not channel["role"].strip() or not isinstance(channel.get("metadata", {}), dict):
            raise ValueError("Invalid volume channel role or metadata.")
        if (index >= len(local) or not isinstance(local[index], dict)
                or local[index].get("serialization_generation") != capsule["exported_artifact"].get("generation")):
            raise ValueError("Companion generation disagrees with the file.")
    validate_evidence((), capsule.get("evidence", {}), origins=[*(c.get("metadata", {}) for c in channels), *origins])
    capsule["scientific_history"] = incoming
    return capsule



_H5J_POSITION_ATTRIBUTE_KEYS = frozenset(
    {
        "stagex",
        "stagey",
        "stagez",
        "xposition",
        "yposition",
        "zposition",
        "positionx",
        "positiony",
        "positionz",
        "xpositionum",
        "ypositionum",
        "zpositionum",
    }
)
_H5J_UNIT_ATTRIBUTE_KEYS = frozenset(
    {
        "PositionUnit",
        "StageUnit",
        "SpatialUnit",
        "Unit",
        "Units",
        "position_unit",
        "stage_unit",
        "spatial_unit",
        "unit",
        "units",
    }
)
_H5J_FRAME_ATTRIBUTE_KEYS = frozenset(
    {"CoordinateFrame", "coordinate_frame"}
)


def _normalized_h5j_position_key(value: Any) -> str:
    return re.sub(r"[\s_\-]+", "", str(value or "")).lower()


def _h5j_scalar_attribute(value: Any) -> str | int | float | None:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(()).item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        return value.strip() or None
    return None


def h5j_position_attribute_subset(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Return only strict scalar H5J stage position/unit/frame evidence."""

    if not isinstance(attributes, Mapping):
        return {}
    retained: dict[str, Any] = {}
    for raw_key in sorted(attributes, key=lambda item: str(item)):
        key = str(raw_key)
        normalized_key = _normalized_h5j_position_key(key)
        is_position = normalized_key in _H5J_POSITION_ATTRIBUTE_KEYS
        is_unit = key in _H5J_UNIT_ATTRIBUTE_KEYS
        is_frame = key in _H5J_FRAME_ATTRIBUTE_KEYS
        if not (is_position or is_unit or is_frame):
            continue
        value = _h5j_scalar_attribute(attributes[raw_key])
        if value is None:
            continue
        if (is_unit or is_frame) and not isinstance(value, str):
            continue
        retained[key] = value
    return retained


def canonical_h5j_position_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and detach the persisted strict H5J stage-evidence schema."""

    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("H5J position metadata must be an object.")
    unknown_sections = set(value).difference({"root_attrs", "channel_attrs"})
    if unknown_sections:
        raise ValueError(
            "H5J position metadata contains unsupported sections: "
            + ", ".join(sorted(str(key) for key in unknown_sections))
        )

    root_value = value.get("root_attrs") or {}
    if not isinstance(root_value, Mapping):
        raise ValueError("H5J root position attributes must be an object.")
    root_attrs = h5j_position_attribute_subset(root_value)
    if set(root_attrs) != {str(key) for key in root_value}:
        raise ValueError(
            "H5J root position attributes contain unsupported keys or values."
        )

    raw_channels = value.get("channel_attrs") or {}
    if not isinstance(raw_channels, Mapping):
        raise ValueError("H5J channel position attributes must be an object.")
    channel_attrs: dict[str, dict[str, Any]] = {}
    for raw_channel in sorted(raw_channels, key=lambda item: str(item)):
        if not isinstance(raw_channel, str) or not raw_channel.strip():
            raise ValueError(
                "H5J channel position attributes require non-empty channel names."
            )
        attributes = raw_channels[raw_channel]
        if not isinstance(attributes, Mapping):
            raise ValueError(
                "H5J channel position attributes must contain attribute objects."
            )
        retained = h5j_position_attribute_subset(attributes)
        if set(retained) != {str(key) for key in attributes}:
            raise ValueError(
                f"H5J channel {raw_channel!r} position attributes contain "
                "unsupported keys or values."
            )
        if retained:
            channel_attrs[raw_channel] = retained

    result: dict[str, Any] = {}
    if root_attrs:
        result["root_attrs"] = root_attrs
    if channel_attrs:
        result["channel_attrs"] = channel_attrs
    return result


def stable_scientific_operation_record(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a detached operation payload with a stable explicit identity."""

    try:
        record = json.loads(json.dumps(dict(value or {}), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Scientific operation must be JSON-serializable: {exc}"
        ) from exc
    operation_type = str(
        record.get("operation_type") or record.get("operation") or ""
    ).strip()
    if not operation_type:
        raise ValueError("Scientific operation type is required.")
    record["operation_type"] = operation_type
    record.pop("operation", None)
    operation_id = str(record.get("operation_id") or "").strip()
    if not operation_id:
        identity = copy.deepcopy(record)
        identity.pop("operation_id", None)
        encoded = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        operation_id = "sha256:" + hashlib.sha256(encoded).hexdigest()
    record["operation_id"] = operation_id
    return record


def generated_voxel_provenance_projection(
    snapshot: Mapping[str, Any],
    operation_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make a generated-data operation the producer of one detached output."""

    scientific = copy.deepcopy(
        dict(snapshot.get("scientific_provenance") or {})
    )
    previous_id = str(
        scientific.get("producing_operation_id") or ""
    ).strip()
    previous_producer = scientific.get("producing_operation")
    if previous_id and not isinstance(previous_producer, Mapping):
        raise ValueError(
            "Generated voxel provenance has a producer ID without its record."
        )

    record = copy.deepcopy(dict(operation_record or {}))
    record = stable_scientific_operation_record(record)

    from .model import _operation_payloads_equal

    supporting_by_id: dict[str, dict[str, Any]] = {}
    supporting_order: list[str] = []
    for raw_record in (
        *([previous_producer] if previous_producer is not None else []),
        *list(scientific.get("supporting_operations") or ()),
    ):
        support = stable_scientific_operation_record(raw_record)
        operation_id = support["operation_id"]
        prior = supporting_by_id.get(operation_id)
        if prior is not None and not _operation_payloads_equal(prior, support):
            raise ValueError(
                f"Scientific operation {operation_id!r} has conflicting payloads."
            )
        supporting_by_id[operation_id] = support
        if operation_id not in supporting_order:
            supporting_order.append(operation_id)
    if record["operation_id"] in supporting_by_id:
        raise ValueError(
            "Generated voxel producer identity conflicts with an input operation."
        )

    projection = {
        **scientific,
        "producing_operation_id": record["operation_id"],
        "producing_operation": copy.deepcopy(record),
        "supporting_operations": [
            copy.deepcopy(supporting_by_id[operation_id])
            for operation_id in supporting_order
        ],
    }
    projection.pop("operation_ids", None)
    projection.pop("history_operation_ids", None)
    return record, projection


def scalar_descriptor(dtype) -> tuple[str | None, int | None]:
    """Return normalized scalar dtype and storage bit depth, or explicit unknowns."""
    if dtype in (None, ""):
        return None, None
    try:
        normalized = np.dtype(dtype)
    except Exception:
        return None, None
    return normalized.name, int(normalized.itemsize * 8)


def explicit_text(value) -> str | None:
    """Normalize source-provided text while preserving missing values as unknown."""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    value = str(value).strip()
    return value or None


def first_explicit_text(metadata: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = explicit_text(metadata.get(key))
        if value is not None:
            return value
    return None


def axis_contract(axes: Iterable[Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return source-order labels and semantic names without inventing labels."""
    axes = tuple(axes or ())
    return (
        tuple(str(getattr(axis, "label", "") or "") for axis in axes),
        tuple(str(getattr(axis, "semantic", "unknown") or "unknown") for axis in axes),
    )


def decoded_payload_descriptor(
    array,
    *,
    decoded_array=None,
    dimensions=(),
    time_point_count=1,
    axis_order=(),
    axis_semantics=(),
):
    """Describe source and decoded scalar contracts without interpreting geometry."""
    source_dtype, source_bit_depth = scalar_descriptor(getattr(array, "dtype", None))
    decoded = array if decoded_array is None else decoded_array
    decoded_dtype, decoded_bit_depth = scalar_descriptor(
        getattr(decoded, "dtype", None)
    )
    return {
        "source_scalar_dtype": source_dtype,
        "source_scalar_bit_depth": source_bit_depth,
        "decoded_scalar_dtype": decoded_dtype,
        "decoded_scalar_bit_depth": decoded_bit_depth,
        "decoded_dimensions": tuple(int(value) for value in dimensions),
        "decoded_time_point_count": int(time_point_count),
        "source_axis_order": tuple(str(value) for value in axis_order),
        "source_axis_semantics": tuple(str(value) for value in axis_semantics),
    }


__all__ = [
    "axis_contract",
    "canonical_h5j_position_metadata",
    "capture_volume_operation_inputs",
    "decoded_payload_descriptor",
    "explicit_text",
    "first_explicit_text",
    "generated_voxel_provenance_projection",
    "h5j_position_attribute_subset",
    "merge_captured_volume_inputs",
    "scalar_descriptor",
    "stable_scientific_operation_record",
]
