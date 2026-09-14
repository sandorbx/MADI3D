"""Prove CDM placement from typed reformat/crop provenance, without pixel I/O."""
from __future__ import annotations

import hashlib
import os
import numpy as np

from .cdm_records import CDMMapping, canonical_json
from .search_profiles import BRAIN, SEARCH_PROFILES, search_profile
from .query import digest
from madi3d_app.volume.geometry import VolumeWorkingGrid


def exploratory_mapping(geometry, placement="automatic", *, reason=None, world_bounds=None, profile_id=BRAIN.profile_id):
    """Place a complete input on the CDM canvas without claiming registration.

    Generic placement preserves the XY aspect ratio of the working bounds and
    maps their full depth. The isotropic preset assumes the complete template
    extent; it is intentionally recorded as an assumption, never hash evidence.
    """
    from itertools import product
    profile = search_profile(profile_id)
    dimensions = np.asarray(geometry.working_grid.dimensions)
    world = np.asarray(geometry.pose) @ np.asarray(geometry.working_grid.local_index_to_working_affine)
    assumptions = ["Template alignment is unverified; search matches may be unreliable."]
    if reason:
        assumptions.append("Recorded alignment could not be established: " + str(reason))
    isotropic = profile == BRAIN and world_bounds is None and np.array_equal(dimensions, (1652, 773, 456))
    if placement == "isotropic" and not isotropic:
        raise ValueError("The 0.38 µm isotropic preset requires dimensions 1652 × 773 × 456. Use Automatic for this source.")
    if placement == "working":
        matrix = np.diag([*(1 / np.asarray(profile.spacing_xyz_um)), 1.]) @ world
        assumptions.append("Current working coordinates are treated as micrometres in the 20× template space." if profile == BRAIN else
                           f"Current working coordinates are treated as micrometres in {profile.template_id}.")
    else:
        corners = np.array(list(product(*[(0, int(n)-1) for n in dimensions])))
        points = corners @ world[:3, :3].T + world[:3, 3]
        low, high = points.min(axis=0), points.max(axis=0)
        if world_bounds is not None:
            bounds = np.asarray(world_bounds, dtype=float)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
                raise ValueError("Explicit object bounds must be finite and ordered.")
            low, high = bounds
        span = high - low
        target = np.asarray(profile.template_shape_zyx[::-1]) - 1
        scale = np.divide(target, span, out=np.ones(3), where=span > 0)
        if not isotropic:
            available = scale[:2][span[:2] > 0]
            scale[:2] = available.min() if available.size else 1.
            assumptions.append("Full working bounds are centered on the 20× canvas; XY aspect is preserved and depth spans the color range." if profile == BRAIN else
                               f"Full working bounds are centered on {profile.template_id}; XY aspect is preserved and depth spans the color range.")
        else:
            assumptions.append("Dimensions match JRC2018_UNISEX_38um_iso_16bit; its full extent is assumed equivalent to the 20× template.")
        # Keep edge centres inside the sampled canvas despite tile-offset roundoff.
        # This guard is part of the recorded affine (less than 1e-11 canvas pixels).
        scale *= 1 + 16 * np.finfo(float).eps
        placement_affine = np.diag([*scale, 1.])
        placement_affine[:3, 3] = (target - span * scale) / 2 - low * scale
        # A single input plane must land on a sampleable integer canvas plane.
        flat = span == 0
        placement_affine[:3, 3][flat] = np.floor(target[flat] / 2 + .5) - low[flat] * scale[flat]
        matrix = placement_affine @ world
    payload = {"geometry": geometry.to_dict(), "placement": placement, "mapping": matrix.tolist(), "assumptions": assumptions,
               "world_bounds": None if world_bounds is None else np.asarray(world_bounds).tolist()}
    return CDMMapping(matrix, "assumed:" + digest(payload), assumptions=tuple(assumptions), placement=placement,
                      template_id=profile.template_id, template_sha256=profile.template_sha256)


class TemplateVerificationRequired(ValueError):
    def __init__(self, channel, path):
        super().__init__("Verifying the recorded template file against the pinned search template…")
        self.channel_id = channel.channel_id
        self.backing_source_id = channel.backing_source_id
        self.geometry_revision = channel.geometry_revision
        self.path = path


def verify_template_file(path, cancel):
    """Cancellable file verification; callers run this outside the GUI thread."""
    from .cdm import checkpoint
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checkpoint(cancel)
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    checkpoint(cancel)
    fingerprint = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if fingerprint(before) != fingerprint(after) or fingerprint(after) != fingerprint(os.stat(path)):
        raise ValueError("The template file changed during verification; retry with a stable source.")
    if digest.hexdigest() not in {p.template_sha256 for p in SEARCH_PROFILES.values()}:
        raise ValueError("The reference file is not the pinned NeuronBridge search template. Supply verified external mapping evidence for converted or external templates.")
    return digest.hexdigest(), fingerprint(after)


def automatic_mapping(model, snapshots, channel_id, *, profile_id=None):
    """Follow explicit producers to the checksum-identified search template.

    A coordinate-space label or a matching shape is never template evidence.
    Current parent/reference revisions must still match the producing operation.
    Reformat has already consumed its registration transform; do not apply it twice.
    """
    visited, evidence = set(), []
    profile = None if profile_id is None else search_profile(profile_id)

    def visit(key):
        nonlocal profile
        if key in visited:
            raise ValueError("Cyclic template mapping provenance.")
        visited.add(key)
        record = model.channels.get(key)
        if record is None:
            raise ValueError("Template mapping references a missing channel.")
        acquisition = snapshots.metadata_snapshot(record.source_id)
        channel = next(c for c in acquisition.channels if c.channel_id == key)
        if channel.working_grid is None or channel.working_geometry is None:
            raise ValueError("Template mapping requires authoritative working geometry.")
        grid = VolumeWorkingGrid.from_dict(channel.working_grid)
        affine = np.asarray(grid.local_index_to_working_affine)
        pose = np.asarray(channel.working_geometry.pose)
        entry = {"channel_id": key, "acquisition_id": record.source_id,
                 "geometry_revision": channel.geometry_revision, "working_grid": grid.to_dict(),
                 "pose": pose.tolist(), "transform_chain": acquisition.transform_chain}
        evidence.append(entry)
        def grid_matches(candidate):
            expected = np.eye(4)
            expected[:3, :3] = np.asarray(candidate.direction) @ np.diag(candidate.spacing_xyz_um)
            expected[:3, 3] = candidate.origin_xyz_um
            return (grid.dimensions == candidate.template_shape_zyx[::-1]
                    and grid.physical_units == ("micron",) * 3
                    and np.allclose(affine, expected, rtol=0, atol=1e-9))
        checksum = str(channel.source_checksum or "").removeprefix("sha256:")
        identified = next((p for p in SEARCH_PROFILES.values() if p.template_sha256 == checksum), None)
        if identified is not None:
            if profile is not None and identified != profile:
                raise ValueError("Recorded template and selected search profile disagree.")
            profile = identified
            if not grid_matches(profile):
                raise ValueError("The pinned template's working grid has changed.")
            entry["template_id"] = profile.template_id
            entry["source_checksum"] = profile.template_sha256
            return np.eye(4), channel, grid, pose
        template_grid = any(grid_matches(p) for p in
                            ((profile,) if profile is not None else SEARCH_PROFILES.values()))

        lineage = acquisition.generated_lineage
        if lineage:
            relation = lineage["parent_index_relation"]
            parent_key = relation["parent_channel_id"]
            parent_mapping, parent, parent_grid, parent_pose = visit(parent_key)
            if (relation["parent_acquisition_id"] != parent.acquisition_id or
                    lineage.get("parent_geometry_revision") != parent.geometry_revision):
                raise ValueError("Generated input's parent geometry revision is missing or changed.")
            step = np.eye(4)
            step[:3, 3] = [relation["parent_extent"][i] for i in (0, 2, 4)]
            entry["generated_lineage"] = lineage
        else:
            operation = model.operation_records.get(record.producing_operation_id, {})
            operation_type = operation.get("operation_type", operation.get("operation"))
            if operation_type in (None, "scalar_import") and not channel.source_checksum and template_grid:
                path = channel.runtime_path or channel.primary_path
                if path:
                    # Shape only selects a file to verify. Acceptance still
                    # requires the pinned full-file hash, never the filename.
                    raise TemplateVerificationRequired(channel, path)
            if (operation_type != "registration_reformat"
                    or operation.get("execution_status") != "completed"
                    or operation.get("output_acquisition_id") != record.source_id
                    or operation.get("output_channel_id", key) != key
                    or not operation.get("result_transform_id")):
                raise ValueError("No proven path from this volume to the pinned NeuronBridge template. Import external mapping evidence if available.")
            revision = operation.get("output_geometry_revision")
            if revision and revision != channel.geometry_revision:
                raise ValueError("Reformat output geometry changed since generation.")
            parent_mapping, parent, parent_grid, parent_pose = visit(operation.get("reference_channel_id"))
            if (operation.get("reference_acquisition_id") != parent.acquisition_id or
                    operation.get("reference_geometry_revision") != parent.geometry_revision):
                raise ValueError("Reformat reference geometry revision is missing or changed.")
            step = np.linalg.inv(np.asarray(parent_grid.local_index_to_working_affine)) @ affine
            entry["producing_operation"] = operation
            supporting = {}
            pending = list(operation.get("input_operation_ids", ()))
            while pending:
                operation_id = pending.pop()
                if operation_id in supporting:
                    continue
                value = model.operation_records.get(operation_id)
                if value is None:
                    raise ValueError("Reformat transform provenance is incomplete.")
                supporting[operation_id] = value
                pending.extend(value.get("input_operation_ids", ()))
            entry["supporting_operations"] = supporting
        if (grid.physical_units != parent_grid.physical_units or
                grid.coordinate_mode != parent_grid.coordinate_mode or
                not np.allclose(pose, parent_pose, rtol=0, atol=1e-9) or
                not np.allclose(affine, np.asarray(parent_grid.local_index_to_working_affine) @ step,
                                rtol=0, atol=1e-9)):
            raise ValueError("The derived working grid/pose no longer matches its recorded template placement.")
        return parent_mapping @ step, channel, grid, pose

    matrix, _channel, _grid, _pose = visit(channel_id)
    payload = {"version": 1, "template_id": profile.template_id, "template_sha256": profile.template_sha256,
               "source_channel_id": channel_id, "chain": evidence, "mapping": matrix.tolist()}
    checksum = digest(payload)
    return CDMMapping(matrix, "sha256:" + checksum, "madi3d-geometry:" + channel_id,
                      checksum, template_id=profile.template_id, template_sha256=profile.template_sha256,
                      evidence_json=canonical_json(payload))
