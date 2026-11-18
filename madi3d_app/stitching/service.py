"""Stitching input planning and the consumed GUI-independent service API."""
from __future__ import annotations

import copy
import math

import numpy as np

from madi3d_app.stitching.models import StitchingCapture
from madi3d_app.volume.geometry import finite_tuple3, invertible_affine4
from madi3d_app.volume.snapshot import AcquisitionRuntimeSnapshot, ChannelRuntimeSnapshot

from . import common
from .common import (
    _registration_admission_bytes,
    validate_selection_source_ownership,
)

from .registration import (
    DEFAULT_STITCHING_MAX_ANGLE_DEG,
    DEFAULT_STITCHING_MAX_SCALE_PERCENT,
    DEFAULT_STITCHING_MAX_SHEAR,
    DEFAULT_STITCHING_MINIMUM_NCC,
    StitchRegistrationOperation,
    registration_settings_from_saved_panel,
)

from .fusion import (
    StitchFusionOperation,
    _safe_stem,
    _validate_fusion_options,
    _validated_fusion_mode,
    fusion_options_from_settings,
)


def stitching_capture_from_snapshot(
    acquisition: AcquisitionRuntimeSnapshot,
    channel: ChannelRuntimeSnapshot,
    *,
    tile_id: str,
    display_name: str,
) -> StitchingCapture:
    """Project authoritative metadata into detached stitching transport, without pixels."""
    geometry = channel.working_geometry
    if geometry is None:
        raise RuntimeError(f"Volume {display_name!r} has no valid numerical working geometry.")
    dims = finite_tuple3(geometry.dimensions, "Dimensions", positive=True, integer=True)
    finite_tuple3(geometry.spacing, "Spacing", positive=True)
    local_affine = np.array(geometry.local_index_to_working_affine, dtype=float, copy=True)
    pose = np.array(acquisition.shared_pose, dtype=float, copy=True)
    world_affine = invertible_affine4(pose @ local_affine, f"Tile {display_name!r} working transform")
    selector = copy.deepcopy(channel.selector)
    if isinstance(selector, np.integer):
        selector = int(selector)
    mapping = selector if isinstance(selector, dict) else {}

    def selector_value(*keys):
        for key in keys:
            if mapping.get(key) not in (None, ""):
                return mapping[key]
        return None

    def optional_index(value):
        if value in (None, "") or isinstance(value, bool):
            return None
        try:
            number = float(value)
            converted = int(number)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number) or number != converted or converted < 0:
            return None
        return converted

    series_index = optional_index(channel.series_index)
    if series_index is None:
        series_index = optional_index(selector_value("series_index", "series"))
    series_identity = channel.series_identity or selector_value("series_identity")
    channel_index = optional_index(selector_value("channel_index", "channel"))
    if channel_index is None and not isinstance(selector, dict):
        channel_index = optional_index(selector)
    time_index = optional_index(selector_value("time_index", "time", "frame"))
    if time_index is None:
        time_index = optional_index(channel.time_index)
    capture_selector = copy.deepcopy(selector)
    if time_index is not None:
        if isinstance(capture_selector, dict):
            capture_selector["time_index"] = time_index
        else:
            capture_selector = {"source_selector": capture_selector, "time_index": time_index}

    # Channel selection and H5J evidence precedence belong to the snapshot service.
    root_attrs = copy.deepcopy(channel.h5j_root_attrs)
    selected_attrs = copy.deepcopy(channel.h5j_selected_channel_attrs)
    if str(channel.backing_format).lower() == "h5j" and (root_attrs or selected_attrs):
        source_metadata = {
            "kind": "h5j",
            "retention_state": "retained-position-metadata",
            "root_attrs": root_attrs,
            "selected_channel_attrs": selected_attrs,
        }
    else:
        source_metadata = {
            "kind": channel.backing_format or "unknown",
            "retention_state": "non-geometry-metadata-unavailable",
            "diagnostics": ["Non-geometry source metadata was not retained in the stitching snapshot."],
        }
    return {
        "tile_id": tile_id,
        "display_name": str(display_name),
        "dims": tuple(dims),
        "dtype": channel.scalar_dtype,
        "coordinate_space_id": geometry.coordinate_space_id,
        "local_affine": local_affine,
        "actor_matrix": pose,
        "world_affine": world_affine,
        "spacing": tuple(float(v) for v in np.linalg.norm(world_affine[:3, :3], axis=0)),
        "source_id": acquisition.acquisition_id,
        "acquisition_id": acquisition.acquisition_id,
        "channel_id": channel.channel_id,
        "backing_source_id": channel.backing_source_id,
        "channel_role": channel.role,
        "channel_order": channel.order,
        "source_checksum": channel.source_checksum,
        "selector": capture_selector,
        "channel_selector": selector,
        "geometry_revision": channel.geometry_revision,
        "acquisition_geometry_revision": acquisition.geometry_revision,
        "producing_operation_id": channel.producing_operation_id,
        "dependency_revision": copy.deepcopy(channel.dependency_revision),
        "source_path": channel.runtime_path or channel.primary_path,
        "series_index": series_index,
        "series_identity": str(series_identity) if series_identity else None,
        "channel_index": channel_index,
        "channel_identity": str(selector) if selector is not None and not isinstance(selector, dict) else None,
        "time_index": time_index,
        "position_index": optional_index(selector_value("position_index", "position", "scene")),
        "backing_format": channel.backing_format,
        "source_format": channel.backing_format,
        "source_axis_sizes": list(channel.source_axis_sizes),
        "source_axis_order": list(channel.source_axis_order),
        "source_axis_semantics": list(channel.source_axis_semantics),
        "backing_physical_grid_observation": copy.deepcopy(channel.backing_physical_grid_observation),
        "source_physical_grid_observation": copy.deepcopy(channel.source_physical_grid_observation),
        "source_observation": {
            "kind": "detached-snapshot",
            "channel_local_observation": copy.deepcopy(channel.source_physical_grid_observation),
            "backing_observation": copy.deepcopy(channel.backing_physical_grid_observation),
        },
        "source_metadata": source_metadata,
        "madi_metadata": {},
        "space_units": list(geometry.physical_units) if geometry.physical_units is not None else None,
        "source_operation_ids": [channel.producing_operation_id] if channel.producing_operation_id else [],
        "stage_position_observations": copy.deepcopy(list(acquisition.stage_position_observations)),
        "position_evidence_geometry_revision": channel.geometry_revision,
        "working_geometry": geometry.to_dict(),
        "channel_local_working_grid": copy.deepcopy(channel.working_grid),
        "channel_local_geometry_revision": channel.geometry_revision,
        "snapshot_diagnostics": acquisition.diagnostic_payload(),
    }


def matrix_maps_equal(first, second, tolerance=1e-6):
    if set(first) != set(second):
        return False
    return all(
        np.allclose(
            np.asarray(first[key]),
            np.asarray(second[key]),
            atol=tolerance,
            rtol=tolerance,
        )
        for key in first
    )


def registered_target_matrix(registration_result, source_key, display_name, channel_label):
    """Copy and validate the exact saved target pose for one source key."""
    result = registration_result or {}
    stored = (result.get("target_matrices_by_source") or {}).get(source_key)
    if stored is None:
        raise RuntimeError(
            f"Saved registration has no exact target pose for {display_name}, "
            f"channel {channel_label}."
        )
    return invertible_affine4(
        stored, f"Saved registered target pose for {source_key}"
    ).copy()


def assign_source_target(targets, source_key, target):
    target = invertible_affine4(
        target, f"Registered target pose for {source_key}"
    ).copy()
    existing = targets.get(source_key)
    if existing is not None and not np.allclose(
        np.asarray(existing), target, atol=1e-8, rtol=1e-8
    ):
        raise RuntimeError(
            f"Registration produced conflicting target poses for {source_key}. "
            "One SourceID cannot be assigned to multiple stitching tiles."
        )
    targets[source_key] = target


def layout_delta(layout, tile_id):
    return np.asarray(
        (layout or {}).get("placement_deltas", {}).get(tile_id, np.eye(4)),
        dtype=float,
    )


def complete_fusion_snapshot(
    snapshot: StitchingCapture,
    *,
    descriptor,
    display_name,
    tile_id,
    source_key,
    actor_matrix,
    registration_result,
    pose_source,
):
    """Complete capture geometry without copying its detached pixel buffer."""
    snapshot = dict(snapshot)
    actor_matrix = np.asarray(actor_matrix, dtype=float).copy()
    snapshot["actor_matrix"] = actor_matrix
    snapshot["world_affine"] = actor_matrix @ snapshot["local_affine"]
    snapshot["spacing"] = tuple(
        float(value) for value in np.linalg.norm(snapshot["world_affine"][:3, :3], axis=0)
    )
    result = registration_result or {}
    snapshot["source_descriptor"] = copy.deepcopy(descriptor or {})
    if pose_source == "registered" and result:
        base = (result.get("base_matrices_by_source") or {}).get(source_key)
        if base is None:
            raise RuntimeError(
                f"Registered stitching result has no initial pose for {display_name!r}."
            )
        initial_pose = layout_delta(result.get("initial_layout", {}), tile_id) @ np.asarray(
            base, dtype=float
        )
        mosaic_correction = np.asarray(
            (result.get("mosaic_corrections") or {}).get(tile_id, np.eye(4)),
            dtype=float,
        )
        source_correction = np.asarray(
            (result.get("corrections") or {}).get(tile_id, np.eye(4)),
            dtype=float,
        )
    else:
        initial_pose = np.asarray(actor_matrix, dtype=float)
        mosaic_correction = np.eye(4, dtype=float)
        source_correction = np.eye(4, dtype=float)
    snapshot["initial_pose"] = invertible_affine4(
        initial_pose, f"Initial stitching pose for {display_name!r}"
    ).copy()
    snapshot["solved_correction_mosaic"] = invertible_affine4(
        mosaic_correction, f"Mosaic correction for {display_name!r}"
    ).copy()
    snapshot["solved_correction_source"] = invertible_affine4(
        source_correction, f"Source correction for {display_name!r}"
    ).copy()
    return snapshot


def capture_stitching_operation_inputs(model, snapshots: list[StitchingCapture]):
    """Freeze direct input revisions from the exact consumed stitching snapshots."""
    from madi3d_app.volume.provenance import capture_volume_operation_inputs

    consumed = []
    for tile in snapshots:
        projection = {key: copy.deepcopy(tile[key]) for key in (
            "source_id", "channel_id", "backing_source_id", "selector",
            "geometry_revision", "acquisition_geometry_revision",
            "producing_operation_id", "source_checksum",
        )}
        # The volume-revision schema names the consumed acquisition pose here.
        projection["shared_pose"] = np.asarray(tile["actor_matrix"], dtype=float).copy()
        consumed.append(projection)
    return capture_volume_operation_inputs(model, consumed)


def complete_fusion_channel_sets(
    channel_sets, *, operation_model, project_metadata, source_file,
    evidence_index, registration_result, pose_source,
):
    """Prepare fusion geometry and freeze the exact consumed scientific inputs."""
    from madi3d_app.volume.provenance import capture_volume_inputs

    consumed = [tile for channel in channel_sets for tile in channel["tiles"]]
    common._prepare_stitching_geometry(consumed)
    operation_capture = capture_stitching_operation_inputs(operation_model, consumed)
    registration_id = str((registration_result or {}).get("registration_id") or "").strip()
    supporting_ids = (
        ("stitching-registration:" + registration_id,)
        if registration_id and str(pose_source or "current") == "registered"
        else ()
    )
    capture = capture_volume_inputs(
        operation_model, consumed, project_metadata,
        source_file=source_file, evidence_index=evidence_index,
        operation_capture=operation_capture, supporting_operation_ids=supporting_ids,
    )
    for channel in channel_sets:
        channel["scientific_inputs"] = capture
        channel["scientific_operation_inputs"] = operation_capture
    return channel_sets


def estimate_source_loading(descriptors):
    """Plan requested channel sizes, without estimating loader allocation peaks.

    Captured channel dimensions/dtypes describe logical pixels, not shared backing
    or decoder scratch. The loader owns admission for its actual allocations.
    Missing size evidence remains unknown until that boundary resolves it.
    """
    sizes = []
    for descriptor in descriptors:
        grid = descriptor.get("working_geometry") or descriptor.get("channel_local_working_grid") or {}
        scalar_dtype = descriptor.get("scalar_dtype")
        if not scalar_dtype:
            sizes.append(None)
            continue
        try:
            dims = finite_tuple3(grid.get("dimensions"), "Source load dimensions", positive=True, integer=True)
            dtype = np.dtype(scalar_dtype)
        except (TypeError, ValueError):
            sizes.append(None)
            continue
        sizes.append(math.prod(dims) * dtype.itemsize)
    decoded = None if None in sizes else sum(sizes)
    return dict(requested_channel_bytes=decoded,
                behavior="Requested channel sizes for planning; loader admits actual allocations.")
