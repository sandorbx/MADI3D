# -*- coding: utf-8 -*-
"""Screen-space volume segmentation for MADI3D.

Python owns
interaction/orchestration and small local merge operations; VTK performs stencil
rasterization, cropping, thresholding, connectivity, and image processing.

Implemented scope: implementation phases 1-4 from the MADI3D segmentation plan.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import sys
import traceback
from time import perf_counter
from dataclasses import replace

import numpy as np
import vtk
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from vtkmodules.util import numpy_support

from madi3d_app.scene.memory import MemoryDemand, MemoryEstimate
from madi3d_app.segmentation.computation import (
    SegmentationComputation, segmentation_memory_estimate, stroke_extent,
)
from madi3d_app.segmentation.preview import label_preview_image, mark_preview_values, participating_label_ids
from madi3d_app.segmentation.workers import _SegmentationJob, SegmentationCancelled
from madi3d_app.segmentation.interaction import (
    SegmentationBrushOverlay,
    VolumeSegmentationInteractorStyle,
)
from madi3d_app.segmentation.settings import (
    EDIT_MODE_LABELS,
    _clamp,
    _sanitize_settings,
    save_settings,
)
from madi3d_app.segmentation.masks import (
    _extent_tuple,
    extent_intersection,
    extent_union,
    extent_expand,
    extent_shape_zyx,
    extent_slices_zyx,
    _image_array_view,
    _new_binary_image_like,
    _binary_numpy_image,
    _copy_image_geometry,
    _mask_patch,
)
from madi3d_app.segmentation.state import (
    HistoryEntry,
    SegmentationFrameState,
    SegmentationFrameWorkspace,
    SegmentationPreflightError,
    SegmentationTargetSnapshot,
    SegmentationState,
    LabelSetState,
    LabelRelation,
    LabelRelationKind,
    history_storage_objects,
    history_storage_bytes,
    trim_history,
)
from madi3d_app.segmentation.relations import (
    allowed_domains, relation_dependencies, dependencies_current, find_label,
    review_relations, relation_review_bytes, validate_candidate, repair_proposal, repair_memory_estimate,
)
from madi3d_app.segmentation.section import section_stroke_geometry, display_ray_to_ijk, brush_radius_ijk
from madi3d_app.segmentation.timing import span, timed
from madi3d_app.segmentation.workspace import SegmentationWorkspaceState, SegmentationChannelWorkspace
from madi3d_app.segmentation.output import (
    CategoricalOutputSupport, PreparedSegmentationOutput, SegmentationOutputPublication,
    segmentation_output_memory, output_context,
)
from madi3d_app.integrations.neuronbridge.evidence import OBJECT_KEY, copy_origin
from madi3d_app.scene.tree_roles import (
    ROLE_BACKING_SOURCE_ID,
    ROLE_CHANNEL_ORDER,
    ROLE_CHANNEL_ROLE,
    ROLE_COLOR,
    ROLE_GROUP_KIND,
    ROLE_LOADED,
    ROLE_LOADING,
    ROLE_ITEM_TYPE,
    ROLE_VOLUME_ID,
    ROLE_VOLUME_META,
    ROLE_SOURCE_PATH,
    ROLE_VOLUME_TIME_SERIES,
    ROLE_VOLUME_TIME_PLAYING,
    ROLE_UNSAVED,
    ROLE_SOURCE_CHANNEL,
    ROLE_VOLUME_CHANNEL_ID,
    ROLE_VOLUME_SOURCE_ID,
    ROLE_OBJECT_META,
)
from madi3d_app.volume.model import (
    GROUP_KIND_MULTICHANNEL,
)
from madi3d_app.volume.snapshot import AcquisitionSnapshotError
from madi3d_app.volume.geometry import canonical_space_units, spatial_unit_scale
from madi3d_app.volume.rendering import (
    resolve_volume_scalar_range,
    serialize_volume_rendering_metadata,
)

_LOG = logging.getLogger(__name__)

# Ring resolution for volume-projection brush solids; shared by the solid
# builder and the per-stroke connectivity template so the two never drift.
_BRUSH_RING_SEGMENTS = 28

def _numpy_matrix4(vtk_matrix):
    out = np.eye(4, dtype=float)
    if vtk_matrix is None:
        return out
    for r in range(4):
        for c in range(4):
            out[r, c] = float(vtk_matrix.GetElement(r, c))
    return out


def _matrix_signature(matrix):
    arr = np.asarray(matrix, dtype=float).reshape(4, 4)
    return tuple(np.round(arr.ravel(), 10))


class VolumeSegmentationController(QtCore.QObject):
    """Selection-following editor with retained channel/frame workspaces."""

    def __init__(
        self, panel, volume_container_class, *, render_window, selection,
        snapshots, relationships, volume_pose, clipping, transforms,
        volume_map, get_next_volume_id, commit_volume_id, update_item_icon,
        refresh_scale_bar, project_modified, stop_volume_playback,
        claim_mouse_tool, remove_volume_container, open_selected_signal,
        scene_tree=None, parent=None,
    ):
        super().__init__(parent)
        self.dialog_parent = parent
        self.panel = panel
        self.VolumeContainer = volume_container_class
        self.render_window = render_window
        self.selection = selection
        self.snapshots = snapshots
        self.relationships = relationships
        self.scene_tree = scene_tree
        self.volume_pose = volume_pose
        self.clipping = clipping
        self.transforms = transforms
        # This is the existing runtime registry, not a second scientific owner.
        self.volume_map = volume_map
        self._get_next_volume_id = get_next_volume_id
        self._commit_volume_id = commit_volume_id
        self._update_item_icon = update_item_icon
        self._refresh_scale_bar = refresh_scale_bar
        self._project_modified = project_modified
        self._stop_volume_playback = stop_volume_playback
        self._claim_mouse_tool = claim_mouse_tool
        self._remove_volume_container = remove_volume_container
        self._open_selected_signal = open_selected_signal
        self.renderer = self.render_window.renderer
        self.overlay = SegmentationBrushOverlay(self.render_window.overlay_renderer)
        self._brush_outline_key = None
        self._brush_outline_projection = None
        self._brush_render_timer = QtCore.QTimer(self)
        self._brush_render_timer.setSingleShot(True)
        self._brush_render_timer.timeout.connect(self._render_requested_brush)
        self.interactor = self._get_interactor()
        self.style = VolumeSegmentationInteractorStyle(self)
        self.style.SetDefaultRenderer(self.renderer)
        self.settings = _sanitize_settings(panel.settings_payload())
        self.mode = "select"
        self.signal_threshold = 0.05
        self.upper_threshold = 1.0
        self.active = False
        self.brush_interaction_active = False
        self.target_vc = None
        self.target_item = None
        self.state = None
        self.states_by_channel = {}
        self._saved_workspace = SegmentationWorkspaceState()
        self.section_actor = vtk.vtkImageActor()
        self.section_actor.InterpolateOff()
        self.section_actor.SetVisibility(False)
        self.renderer.AddActor(self.section_actor)
        self._section_target = None
        self.section_labels_actor = vtk.vtkImageActor()
        self.section_labels_actor.InterpolateOff()
        self.section_labels_actor.SetVisibility(False)
        self.renderer.AddActor(self.section_labels_actor)
        self._section_label_map = vtk.vtkImageMapToColors()
        self._section_label_map.SetOutputFormatToRGBA()
        self._neuropil_proposal = None
        self._neuropil_live = False
        self._neuropil_refresh_pending = False
        self._neuropil_region_frame = None
        self._neuropil_refresh_timer = QtCore.QTimer(self)
        self._neuropil_refresh_timer.setSingleShot(True)
        self._neuropil_refresh_timer.timeout.connect(self._refresh_neuropil_if_requested)
        self._neuropil_boxes = []
        for color in ((1.0, 0.9, 0.1), (0.0, 0.8, 0.9)):
            outline = vtk.vtkOutlineSource()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(outline.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.SetVisibility(False)
            self.renderer.AddActor(actor)
            self._neuropil_boxes.append((outline, actor))
        self.preview_port = None
        self.preview_volume = None
        self.preview_property = None
        self.preview_producer = None
        self._preview_output_image = None
        self._preview_matrix_signature = None
        self._preview_warning_shown = False
        self._previous_style = None
        self._previous_cursor_mode = None
        self._target_volume_observer = None
        self._target_image_observer = None
        self._observed_image = None
        self._last_frame_index = None
        self._last_geometry_signature = None
        self._sync_guard = False
        self._session_target_snapshot = None
        self._session_residency_lease = None
        self._activation_inputs = None
        self._activation_item = None
        self._authoritative_refresh_pending = False
        self._saved_clipping_process = None
        self._saved_transform_interaction = None
        self._tree_connections_installed = False
        self._settings_save_timer = QtCore.QTimer(self)
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.setInterval(250)
        self._settings_save_timer.timeout.connect(self._flush_settings)

        self._brush_feedback_phase = False
        self._brush_feedback_timer = QtCore.QTimer(self)
        self._brush_feedback_timer.setInterval(160)
        self._brush_feedback_timer.timeout.connect(self._pulse_brush_feedback)
        self._brush_feedback_reset_timer = QtCore.QTimer(self)
        self._brush_feedback_reset_timer.setSingleShot(True)
        self._brush_feedback_reset_timer.setInterval(300)
        self._brush_feedback_reset_timer.timeout.connect(self._reset_brush_feedback)

        # Threshold edits can repeatedly revisit the same latest Diffuse ROI.
        # Boundary Guard's gradient is independent of the threshold itself, so
        # retain one reasonably sized gradient result and reuse it across edits.
        # The image MTime is part of the key, so changing a time frame/source
        # invalidates the cache automatically. Large ROIs are deliberately not
        # retained to avoid turning a convenience cache into a memory problem.
        self._boundary_gradient_cache_key = None
        self._boundary_gradient_cache = None

        self._pool = QtCore.QThreadPool.globalInstance()
        self._job_serial = 0
        self._worker_refs = {}
        self._publishing_operation = None
        self._publishing_resources = None
        self._publication_edit = None
        self._publishing_result_bytes = 0
        self._publishing_inputs = {}
        self._retained_storage = []
        # Definition fingerprint of the label widgets last published to the panel.
        # None forces the next _sync_label_controls to rebuild every widget.
        self._synced_label_signature = None
        self._history_sequence = 0
        self._admitting = False
        self._operation_job = None
        self._operation = None
        self._operation_generation = 0
        self._pending_exact_threshold = None
        self._threshold_drain_scheduled = False
        self._install_connections()
        self.overlay.set_radii(self.settings["seed_radius"], self.settings["growth_radius"])
        self._apply_preview_property()
        self._install_tree_guards()

    # ------------------------------------------------------------------
    # UI wiring
    # ------------------------------------------------------------------
    def _install_connections(self):
        p = self.panel
        p.enabledToggled.connect(self.set_enabled)
        p.modeChanged.connect(self.set_mode)
        p.brushInteractionChanged.connect(self.set_brush_interaction_active)
        p.cancelOperationRequested.connect(self.cancel_operation)
        p.clearRequested.connect(self.clear)
        p.seedRadiusChanged.connect(self.set_seed_radius)
        p.growthRadiusChanged.connect(self.set_growth_radius)
        p.thresholdChanged.connect(self.set_threshold)
        p.thresholdPreviewChanged.connect(self.request_threshold_preview)
        p.liveThresholdChanged.connect(self.set_live_threshold_preview)
        p.maskOpacityChanged.connect(self.set_mask_opacity)
        p.maskColorChanged.connect(self.set_mask_color)
        p.undoRequested.connect(self.undo)
        p.redoRequested.connect(self.redo)
        p.historyBudgetChanged.connect(self.set_history_budget)
        p.extractRequested.connect(self.extract_selection)
        p.deleteSelectedRequested.connect(self.delete_selected)
        p.createMaskRequested.connect(self.create_mask_volume)
        p.createCategoricalRequested.connect(self.create_categorical_volume)
        p.continueToCdmRequested.connect(self.continue_to_cdm)
        p.smartSettingsChanged.connect(self.set_smart_settings)
        p.extractOriginalRequested.connect(self.extract_original_signal)
        p.removeSpecksRequested.connect(self.remove_specks)
        p.labelCreated.connect(self.create_label)
        p.labelSetCreated.connect(self.create_label_set)
        p.labelSetDuplicated.connect(self.duplicate_label_set)
        p.labelSetSelected.connect(self.select_label_set)
        p.labelSetRenamed.connect(self.rename_label_set)
        p.labelRemoved.connect(self.remove_label)
        p.relationAdded.connect(self.add_relation)
        p.relationRemoved.connect(self.remove_relation)
        p.relationRepairRequested.connect(self.preview_relation_repair)
        p.labelLogicPreviewRequested.connect(self.preview_label_logic)
        p.labelLogicSettingsChanged.connect(self.cancel_neuropil)
        p.labelSelected.connect(self.select_label)
        p.labelPropertyChanged.connect(self._label_property_changed)
        p.labelParticipationChanged.connect(self._label_participation_changed)
        p.sectionChanged.connect(self._sync_section)
        p.section_radius.valueChanged.connect(self._section_brush_geometry_changed)
        p.section_growth.valueChanged.connect(self._section_brush_geometry_changed)
        p.advancedToggled.connect(self._advanced_toggled)
        p.faceSectionRequested.connect(self.face_section)
        p.neuropilPreviewRequested.connect(self.preview_neuropil)
        p.neuropilApplyRequested.connect(self.apply_neuropil)
        p.neuropilCancelRequested.connect(self.cancel_neuropil)
        p.neuropilSettingsChanged.connect(self._neuropil_settings_changed)
        p.neuropilEnabledChanged.connect(self._neuropil_enabled_changed)

    def face_section(self):
        plane = self.panel.editing_plane
        context = self._session_target_snapshot
        if context is None or plane == "Volume projection":
            return
        snap = self._geometry_snapshot(context)
        axes = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}[plane]
        center = np.array([(context.extent[i]+context.extent[i+1])/2 for i in (0, 2, 4)])
        center[axes[2]] = self.panel.section_index.value()
        matrix = snap["vtk_ijk_to_world_affine"]
        world = (matrix @ np.r_[center, 1])[:3]
        normal = np.cross(matrix[:3, axes[0]], matrix[:3, axes[1]])
        normal /= np.linalg.norm(normal)
        camera = self.renderer.GetActiveCamera()
        distance = max(camera.GetDistance(), 1)
        camera.SetFocalPoint(*world)
        camera.SetPosition(*(world + normal*distance))
        camera.SetViewUp(*matrix[:3, axes[1]])
        self.renderer.ResetCameraClippingRange()
        self.render_once()

    def _frame_workspace(self):
        return self.state.frame_workspace(self.current_frame_index())

    def create_label_set(self):
        if self.state is None or self._target_locked():
            return
        workspace = self._frame_workspace()
        frame = workspace.create_set(f"Label set {len(workspace.sets) + 1}")
        frame.labels.create_label("Label 1", color=tuple(self.settings["mask_color"]))
        self.select_label_set(frame.labels.set_id)
        self._project_modified("segmentation authored work")

    def select_label_set(self, set_id):
        if self.state is None or not set_id:
            return
        self._invalidate_operation()
        self._neuropil_live = False
        self._frame_workspace().select_set(set_id)
        self._sync_current_frame_preview(render=True)
        self._update_history_ui()

    def duplicate_label_set(self):
        if not self.active or self._target_locked():
            return
        from madi3d_app.scene.memory import MemoryDemand, MemoryEstimate
        self.cancel_neuropil()
        frame, workspace = self._frame_state(), self._frame_workspace()
        try:
            context = self._preflight_authoritative_target("Duplicate label set")
        except SegmentationPreflightError as exc:
            self.set_status(str(exc))
            return
        marks = (*frame.labels.include_marks.values(), *frame.labels.negative_marks.values(),
                 frame.labels.outside_marks, frame.labels.protected_marks)
        mark_bytes = sum(mark.values.nbytes for mark in marks if mark is not None)
        assignment_bytes = (0 if frame.assignments is None else
                            frame._assignment_array(frame.assignments).nbytes)
        # Detached resource arrays overlap the new VTK assignment copy.
        peak = int(mark_bytes + 2 * assignment_bytes)
        estimate = MemoryEstimate(MemoryDemand(peak, peak), resident_bytes=None,
            basis="Alternative set: copied authored mark arrays and detached uint32 assignments overlapping the new VTK copy. Source/resident work is borrowed.",
            unknown_scopes=("main", "host"))
        revision, generation = frame.revision, self._operation_generation
        operand_ids = {rule.target_label_id for rule in frame.labels.relations}
        operands = tuple((key, owner, owner.revision, dict(owner.label_revisions))
            for key, owner in workspace.sets.items()
            if any(label.id in operand_ids for label in owner.labels.labels))
        def work(report):
            report(10, "Copying authored label work")
            detached = SegmentationFrameWorkspace(sets=dict(workspace.sets))
            duplicate = detached.duplicate_set_as_alternative(frame.labels.set_id)
            report.reserve_memory(relation_review_bytes(detached,
                {label.id for label in duplicate.labels.labels}), "Alternative relationship review",
                materialized_bytes=mark_bytes + assignment_bytes)
            review_relations(detached, context.source_image, {label.id for label in duplicate.labels.labels})
            report.check_cancelled()
            return duplicate
        def publish(duplicate):
            if frame.revision != revision or generation != self._operation_generation:
                self.set_status("Alternative discarded because source work changed. Duplicate again.")
                return
            if any(workspace.sets.get(key) is not owner or owner.revision != version
                   or owner.label_revisions != labels for key, owner, version, labels in operands):
                self.set_status("Alternative discarded because relationship operands changed. Duplicate again.")
                return
            workspace.sets[duplicate.labels.set_id] = duplicate
            self.select_label_set(duplicate.labels.set_id)
            self._project_modified("segmentation authored work")
            self._sync_label_controls()
            self.set_status("Independent alternative created. Rules to other sets retain their named operands; edits here leave the original unchanged.")
        try:
            resources = self._acquire_job_resources("Duplicate label set", context, estimate=estimate)
            self._run_background("Duplicate label set", work, publish, operation_snapshot=context,
                                 resources=resources, memory_extent=frame.assignment_extent)
        except (ValueError, InterruptedError, SegmentationPreflightError) as exc:
            self.set_status(str(exc))

    def rename_label_set(self, name):
        if self.state is None or self._target_locked():
            return
        frame = self._frame_state()
        frame.labels.name = str(name).strip() or frame.labels.name
        # Force so the name field is restored to the canonical set name even
        # when the user's text normalizes to the unchanged definition.
        self._sync_label_controls(force=True)
        self._project_modified("segmentation authored work")

    def remove_label(self):
        if self.state is None or self._target_locked():
            return
        frame = self._frame_state()
        label = frame.labels.active_label
        if label is None or label.locked:
            self.set_status("Unlock the label before removing it.")
            return
        self._invalidate_operation()
        if frame.assignment_extent is not None:
            values = frame.categorical_patch(frame.assignment_extent)
            values[values == label.value] = 0
            try:
                frame.write_assignment_patch(frame.assignment_extent, values, self.target_vc.image)
            except PermissionError as exc:
                self.set_status(str(exc))
                return
        frame.labels.labels.remove(label)
        frame.labels.include_marks.pop(label.id, None)
        frame.labels.negative_marks.pop(label.id, None)
        frame.labels.nonparticipant_ids = [item for item in frame.labels.nonparticipant_ids if item != label.id]
        frame.labels.active_label_id = frame.labels.labels[0].id if frame.labels.labels else None
        frame.revision += 1
        frame.undo_stack.clear()
        frame.redo_stack.clear()
        # Every affected rule now lacks this source/operand; review resolves
        # that state before allocating any categorical or mark crops.
        review_relations(self._frame_workspace(), self.target_vc.image, {label.id})
        self._project_modified("segmentation authored work")
        self._sync_current_frame_preview(render=True)
        self._trim_history()
        self._update_history_ui()
        self.set_status("Label removed; this set's Undo history cleared. Referencing rules remain for correction.")

    def add_relation(self):
        if self.state is None or self._target_locked():
            return
        frame = self._frame_state()
        source = frame.labels.active_label_id
        target = self.panel.relation_operand.currentData()
        if not source or not target:
            self.set_status("Create another label on this source frame to use as the operand.")
            return
        rule = LabelRelation(LabelRelationKind(self.panel.relation_kind.currentData()), source, target)
        if any((item.kind, item.source_label_id, item.target_label_id) ==
               (rule.kind, source, target) for item in frame.labels.relations):
            return
        self._relations_edited([*frame.labels.relations, rule])

    def remove_relation(self):
        if self.state is None or self._target_locked():
            return
        frame = self._frame_state()
        index = self.panel.relation_list.currentRow()
        if 0 <= index < len(frame.labels.relations):
            self._relations_edited([rule for i, rule in enumerate(frame.labels.relations) if i != index])

    def _relations_edited(self, relations):
        frame = self._frame_state()
        workspace = self._frame_workspace()
        changed = {rule.source_label_id for rule in relations}
        # Candidate headers borrow arrays; review only replaces relation records.
        candidate = SegmentationFrameWorkspace(sets={
            key: self._capture_frame_state(owner) if owner is frame or any(
                {rule.source_label_id, rule.target_label_id} & changed for rule in owner.labels.relations)
                else owner for key, owner in workspace.sets.items()},
            active_set_id=workspace.active_set_id)
        candidate.sets[frame.labels.set_id].labels.relations = relations
        demand = relation_review_bytes(candidate, changed)
        revision = frame.revision
        resources = None
        try:
            snapshot = self._preflight_authoritative_target("Relationship review")
            resources = self._acquire_job_resources("Relationship review", snapshot,
                estimate=MemoryEstimate(MemoryDemand(demand, demand), None,
                    "Affected relationship extents: categorical crop, domain, Include/conflict and comparison arrays."))
            if frame.revision != revision or self._frame_workspace() is not workspace:
                raise InterruptedError("Label work changed during relationship review admission.")
            review_relations(candidate, snapshot.source_image, changed)
            for key, owner in candidate.sets.items():
                workspace.sets[key].labels.relations = owner.labels.relations
        except (ValueError, InterruptedError, SegmentationPreflightError, MemoryError) as exc:
            self.set_status(f"Relationship edit could not finish: {exc}")
            return
        finally:
            if resources is not None:
                resources.finish()
        frame.revision += 1
        self._invalidate_operation()
        self._sync_current_frame_preview(render=True)
        self._project_modified("segmentation authored work")

    def _review_history_change(self, frame, entry):
        review_relations(self._frame_workspace(), self.target_vc.image, set(entry.evidence_new))

    def create_label(self):
        if self.state is None or self._target_locked():
            return
        frame = self._frame_state()
        label = frame.labels.create_label(f"Label {len(frame.labels.labels) + 1}",
                                         color=tuple(self.settings["mask_color"]))
        self.select_label(label.id)
        self._project_modified("segmentation authored work")

    def select_label(self, label_id):
        frame = self._frame_state() if self.state is not None else None
        if frame is None or not label_id:
            return
        if frame.labels.active_label_id == label_id:
            return
        refresh_live_preview = self._neuropil_live
        self._invalidate_operation()
        frame.labels.select_label(label_id)
        self._sync_label_controls()
        self._sync_current_frame_preview(render=True)
        if refresh_live_preview:
            self._neuropil_refresh_timer.start(250)

    def _label_property_changed(self, label_id, property_name, value):
        if property_name not in {"name", "color", "visible", "locked"}:
            return
        self.edit_label(label_id, **{str(property_name): value})

    def edit_label(self, label_id=None, **changes):
        frame = self._frame_state() if self.state is not None else None
        if frame is None or self._target_locked():
            return
        label = frame.labels.active_label if label_id is None else frame.labels.label_by_id(label_id)
        if label is None:
            return
        if "name" in changes:
            changes["name"] = str(changes["name"]).strip() or label.name
        if "color" in changes:
            changes["color"] = tuple(float(component) for component in changes["color"])
        if "visible" in changes:
            changes["visible"] = bool(changes["visible"])
        if "locked" in changes:
            changes["locked"] = bool(changes["locked"])
        changes = {key: value for key, value in changes.items()
                   if getattr(label, key) != value}
        if not changes:
            return
        frame.labels.update_label(label.id, **changes)
        if "locked" in changes:
            frame.revision += 1
            self._invalidate_operation()
            if self._neuropil_live:
                self._neuropil_refresh_timer.start(250)
        presentation_only = changes.keys() <= {"name", "color"}
        self._sync_label_controls(update_preview_property="color" in changes or not presentation_only)
        if changes.keys() == {"name"}:
            self.render_once()
        else:
            self._refresh_label_preview(recompose=not presentation_only)
        self._project_modified("segmentation authored work")

    def _refresh_label_preview(self, *, recompose):
        if self.preview_port is None or self.preview_producer is None:
            self.render_once()
            return
        if not recompose and self.panel.editing_plane == "Volume projection":
            self.render_once()
            return
        frame = self._frame_state()
        computation = self._capture_computation(frame)
        preview = None if recompose else self._detached_image(self._preview_output_image)
        pending = self._neuropil_proposal
        proposal = None if pending is None else pending[0]
        estimate = computation.display_memory_estimate(preview=preview, proposal=proposal)
        def work(report):
            computation.report = report
            return computation.prepare_display(preview=preview, proposal=proposal)
        def publish(prepared):
            self._publication_edit = prepared
            self._sync_current_frame_preview(render=True)
        self._run_background("Update label preview", work, publish,
                             memory_extent=frame.assignment_extent, estimate=estimate,
                             display_only=True)

    def _label_participation_changed(self, label_id, enabled):
        frame = self._frame_state() if self.state is not None else None
        if frame is None or self._target_locked():
            return
        label = frame.labels.label_by_id(label_id)
        if label is None or label.locked:
            return
        excluded = set(frame.labels.nonparticipant_ids)
        if bool(enabled):
            excluded.discard(label.id)
        else:
            excluded.add(label.id)
        updated = [item.id for item in frame.labels.labels if item.id in excluded]
        if updated == frame.labels.nonparticipant_ids:
            return
        frame.labels.nonparticipant_ids = updated
        frame.revision += 1
        self._invalidate_operation()
        self._project_modified("segmentation authored work")
        self._sync_current_frame_preview(render=True)
        if self._neuropil_live:
            self._neuropil_refresh_timer.start(250)

    def _sync_label_controls(self, *, update_preview_property=True, force=False):
        frame = self._frame_state() if self.state is not None else None
        if frame is None:
            return
        if not frame.labels.labels:
            frame.labels.create_label("Label 1", color=tuple(self.settings["mask_color"]))
        workspace = self._frame_workspace()
        signature = self._label_controls_signature(workspace)
        if force or signature != self._synced_label_signature:
            self._synced_label_signature = signature
            self.panel.set_labels(frame.labels.labels, frame.labels.active_label_id,
                                  frame.labels.nonparticipant_ids)
            self.panel.set_label_sets(workspace)
            self.panel.set_relations(workspace)
            self.panel.set_logic_operands(workspace)
        label = frame.labels.active_label
        self.panel.settings["mask_color"] = list(label.color)
        self.panel._update_color_button()
        if update_preview_property:
            self._apply_preview_property()

    def _label_controls_signature(self, workspace):
        """Cheap definition-only fingerprint of the rebuilt label widgets.

        Covers exactly the inputs of set_labels, set_label_sets, set_relations
        and set_logic_operands, including per-rule review state and message so
        an accepted edit that changes a relationship dependency still refreshes
        the review controls while an unchanged definition rebuilds nothing.
        """
        return (
            workspace.active_set_id,
            tuple(
                (
                    set_id,
                    frame.labels.name,
                    frame.labels.active_label_id,
                    tuple(frame.labels.nonparticipant_ids),
                    tuple((label.id, label.name, tuple(label.color),
                           label.visible, label.locked)
                          for label in frame.labels.labels),
                    tuple((relation.kind.value, relation.source_label_id,
                           relation.target_label_id, relation.review_state,
                           relation.review_message)
                          for relation in frame.labels.relations),
                )
                for set_id, frame in workspace.sets.items()
            ),
        )

    def _sync_section(self, *_, render=True):
        actor = self.section_actor
        if not self.active or self.target_vc is None:
            actor.SetVisibility(False)
            self.section_labels_actor.SetVisibility(False)
            self.panel.previous_section.setEnabled(False)
            self.panel.next_section.setEnabled(False)
            return
        self._sync_neuropil_region()
        plane = self.panel.editing_plane
        if plane == "Volume projection":
            self.panel.section_position.setText("Volume projection; choose XY, XZ or YZ for marks and protection.")
            actor.SetVisibility(False)
            self.section_labels_actor.SetVisibility(False)
            self.panel.section_index.setEnabled(False)
            self.panel.section_depth.setEnabled(False)
            self.panel.previous_section.setEnabled(False)
            self.panel.next_section.setEnabled(False)
            self.panel.sync_advanced_sliders()
            self.panel._update_edit_context(self.mode if self.brush_interaction_active else None)
            if self.brush_interaction_active:
                self.update_brush_cursor(*self.overlay.center)
            if render:
                self.render_once()
            return
        self.panel.section_index.setEnabled(True)
        self.panel.section_depth.setEnabled(True)
        self.panel.previous_section.setEnabled(True)
        self.panel.next_section.setEnabled(True)
        image = self.target_vc.image
        extent = list(image.GetExtent())
        axis = {"XY": 2, "XZ": 1, "YZ": 0}[plane]
        lo, hi = extent[axis*2:axis*2+2]
        target = (id(self.target_vc), plane)
        blocker = QtCore.QSignalBlocker(self.panel.section_index)
        depth_blocker = QtCore.QSignalBlocker(self.panel.section_depth)
        self.panel.section_index.setRange(lo, hi)
        self.panel.section_depth.setMaximum(hi - lo + 1)
        if target != self._section_target:
            self.panel.section_index.setValue((lo+hi)//2)
            self._section_target = target
        index = self.panel.section_index.value()
        del blocker
        del depth_blocker
        depth = self.panel.section_depth.value()
        first = index - (depth - 1) // 2
        last = first + depth - 1
        self.panel.section_position.setText(f"{plane}: {'XYZ'[axis]} = {index}; painted slab {max(lo, first)}–{min(hi, last)}.")
        extent[axis*2:axis*2+2] = [index, index]
        actor.SetInputData(image)
        actor.SetDisplayExtent(*extent)
        actor.SetUserMatrix(self.target_vc.volume.GetMatrix())
        low, high = self._scalar_range(self.target_vc)
        actor.GetProperty().SetColorWindow(max(high-low, 1e-9))
        actor.GetProperty().SetColorLevel((high+low)/2)
        actor.GetProperty().SetOpacity(0.7)
        actor.SetVisibility(True)
        preview = self._preview_output_image
        overlap = extent_intersection(extent, preview.GetExtent()) if preview is not None else None
        self.section_labels_actor.SetVisibility(overlap is not None)
        if overlap is not None:
            if (self._publication_edit is not None
                    and self._publication_edit.section is not None):
                section = self._publication_edit.section
            else:
                self._section_label_map.SetInputData(preview)
                # Bound RGBA allocation and mapping to the visible section.
                self._section_label_map.UpdateExtent(overlap)
                section = self._section_label_map.GetOutput()
            self.section_labels_actor.SetInputData(section)
            self.section_labels_actor.SetDisplayExtent(*overlap)
            self.section_labels_actor.SetUserMatrix(self.target_vc.volume.GetMatrix())
        context = self._session_target_snapshot
        units = context.physical_units if context is not None else None
        self.panel.section_radius.setSuffix(" µm" if units else " vox")
        self.panel.section_growth.setSuffix(" µm" if units else " vox")
        self.panel.sync_advanced_sliders()
        self.panel._update_edit_context(self.mode if self.brush_interaction_active else None)
        if self.brush_interaction_active:
            self.update_brush_cursor(*self.overlay.center)
        if render:
            self.render_once()

    def _section_brush_geometry_changed(self, *_):
        if (self.active and self.brush_interaction_active
                and self.panel.editing_plane != "Volume projection"):
            self.update_brush_cursor(*self.overlay.center)
            self.render_once()

    def _neuropil_spacing(self, context):
        if context.physical_units is None:
            return (1.0, 1.0, 1.0)
        units = canonical_space_units(context.physical_units)
        affine = np.asarray(context.local_index_affine).reshape(4, 4)
        return tuple(np.linalg.norm(affine[:3, :3], axis=0) * spatial_unit_scale(units[0], "micron"))

    def _neuropil_extents(self, frame, context):
        editable = frame.labels.editable_extent or context.extent
        if extent_intersection(editable, context.extent) != editable:
            raise ValueError("The editable box lies outside the source. Adjust its voxel limits.")
        if any(editable[i] > editable[i+1] for i in (0, 2, 4)):
            raise ValueError("Each editable-box first index must be at or before its last index.")
        spacing = self._neuropil_spacing(context)
        halo = tuple(max(1, math.ceil(4 * frame.labels.boundary_scale / step)) for step in spacing)
        expanded = tuple(editable[i] + (-halo[i//2] if i % 2 == 0 else halo[i//2]) for i in range(6))
        return editable, extent_intersection(expanded, context.extent), spacing

    def _sync_neuropil_region(self):
        context = self._session_target_snapshot
        if not self.active or context is None or not self.panel.advanced_toggle.isChecked():
            for _, actor in self._neuropil_boxes:
                actor.SetVisibility(False)
            return
        frame = self._frame_state()
        if self._neuropil_region_frame is not frame:
            self._neuropil_region_frame = frame
            self.panel.set_neuropil_region(context.extent, frame.labels.editable_extent or context.extent,
                                          frame.labels.boundary_scale, context.physical_units)
        try:
            editable, filtering, _ = self._neuropil_extents(frame, context)
        except ValueError:
            for _, actor in self._neuropil_boxes:
                actor.SetVisibility(False)
            return
        matrix = self._geometry_snapshot(context)["vtk_ijk_to_world_affine"]
        vtk_matrix = vtk.vtkMatrix4x4()
        vtk_matrix.DeepCopy(matrix.ravel())
        for (outline, actor), extent in zip(self._neuropil_boxes, (editable, filtering)):
            outline.SetBounds(*(extent[i] + (-0.5 if i % 2 == 0 else 0.5) for i in range(6)))
            actor.SetUserMatrix(vtk_matrix)
            actor.SetVisibility(self.panel.neuropil_box.isChecked())

    def _neuropil_settings_changed(self):
        if not self.active or self._target_locked() or not self.panel.advanced_toggle.isChecked():
            return
        frame = self._frame_state()
        extent = tuple(spin.value() for spin in self.panel.neuropil_crop)
        scale = self.panel.neuropil_scale.value()
        previous_extent = frame.labels.editable_extent or self.state.source_extent
        changed = extent != previous_extent or scale != frame.labels.boundary_scale
        if changed:
            had_proposal = self._neuropil_proposal is not None
            participants_changed = (extent != previous_extent and
                self._participating_label_ids(frame, previous_extent) != self._participating_label_ids(frame, extent))
            frame.labels.editable_extent = extent
            frame.labels.boundary_scale = scale
            frame.revision += 1
            self._project_modified("segmentation authored work")
            self._invalidate_operation()
            if self._neuropil_live:
                self._neuropil_refresh_timer.start(250)
            # Box/smoothing edits do not alter the authored overlay unless the
            # box changes fixed participants or a displayed proposal is retired.
            if had_proposal or participants_changed:
                self._sync_current_frame_preview(render=True)
                return
        self._sync_neuropil_region()
        self.render_once()

    def _neuropil_enabled_changed(self, enabled):
        if not self.active or self._target_locked() or not self.panel.advanced_toggle.isChecked():
            return
        if enabled:
            if self.panel.mark_mode.currentData() == "assignments":
                self.panel.mark_mode.setCurrentIndex(self.panel.mark_mode.findData("include"))
            self.set_mode("select")
        else:
            self.cancel_neuropil()
        self._sync_neuropil_region()
        self._sync_current_frame_preview(render=True)

    def _discard_neuropil_proposal(self):
        if getattr(self, "_neuropil_proposal", None) is None:
            return
        self._neuropil_proposal = None
        self.panel.set_neuropil_proposal(False, "No proposal. Yellow box: editable; cyan box: filtering context.")
        self._release_retained_storage()

    def _refresh_neuropil_if_requested(self):
        if self.active and self._neuropil_live:
            if self._target_locked():
                self._neuropil_refresh_pending = True
                return
            self.preview_neuropil()

    @staticmethod
    def _participating_label_ids(frame, editable):
        return participating_label_ids(frame, editable)

    def preview_neuropil(self):
        if not self.active or self._target_locked() or not self.panel.advanced_toggle.isChecked():
            return
        from madi3d_app.segmentation.neuropil import propose_neuropil, neuropil_memory_estimate
        self._discard_neuropil_proposal()
        self._sync_current_frame_preview(render=True)
        self._neuropil_live = True
        frame = self._frame_state()
        try:
            context = self._preflight_authoritative_target("Neuropil preview")
            editable, filtering, spacing = self._neuropil_extents(frame, context)
            # Evidence outside the box is reported, never silently consumed.
            outside_count = 0
            for marks in (*frame.labels.include_marks.values(),
                          *frame.labels.negative_marks.values(), frame.labels.outside_marks):
                if marks is None:
                    continue
                overlap = extent_intersection(marks.extent, editable)
                inside = 0 if overlap is None else np.count_nonzero(marks.values[extent_slices_zyx(overlap, marks.extent)])
                outside_count += int(np.count_nonzero(marks.values)) - int(inside)
            labels = tuple(frame.labels.labels)
            participants = self._participating_label_ids(frame, editable)
            if not participants:
                self.set_status("Check a participant and paint its Include marks inside the editable box before Preview.")
                return
            source = self._detached_image(context.source_image)
            workspace = self._frame_workspace()
            dependencies = relation_dependencies(workspace, frame, participants)
            revision = frame.revision
            generation = self._operation_generation
            scale = frame.labels.boundary_scale
            # All crops are allocated after admission, in the worker. Frame writes remain serialized.
            def work(report):
                report(1, "Capturing native crop and markers")
                values = _image_array_view(source)[extent_slices_zyx(filtering, context.extent)]
                baseline = frame.categorical_patch(editable)
                includes = {label_id: frame.mark_patch("include", editable, label_id) for label_id in participants}
                negatives = {label_id: frame.mark_patch("negative", editable, label_id)
                             for label_id in participants}
                outside = frame.mark_patch("outside", editable)
                domains = allowed_domains(workspace, frame, editable, source, participants)
                return propose_neuropil(values, context_extent=filtering, editable_extent=editable,
                    baseline=baseline, labels=labels, include_marks=includes, outside_marks=outside,
                    negative_marks=negatives, allowed_domains=domains,
                    protected_marks=frame.mark_patch("protected", editable),
                    participants=participants, spacing_xyz=spacing,
                    smoothing=scale, report=report)
            def publish(proposal):
                if (frame.revision != revision or generation != self._operation_generation or
                        not dependencies_current(workspace, dependencies)):
                    self.set_status("Neuropil proposal discarded because authored work changed.")
                    return
                self._neuropil_proposal = (proposal, context, frame, revision, generation, dependencies, "neuropil")
                details = []
                for label in labels:
                    if label.id in participants:
                        details.append(f"{label.name}: +{proposal.gains[label.id]:,} / −{proposal.losses[label.id]:,}")
                edges = [label.name for label in labels if label.id in proposal.edge_labels]
                summary = "; ".join(details) + f". {proposal.elapsed_seconds:.2f} s."
                names = {label.value: label.name for label in labels}
                transfers = [f"{names[left]} → {names[right]}: {count:,}"
                             for (left, right), count in proposal.transfers.items()]
                if transfers:
                    summary += " Transfers: " + "; ".join(transfers) + "."
                if edges:
                    summary += " Crop-edge contact: " + ", ".join(edges) + "."
                if outside_count:
                    summary += f" {outside_count:,} marks outside the editable box have no effect; expand the box to use them."
                if proposal.invalid_voxels:
                    summary += f" {proposal.invalid_voxels:,} invalid source voxels remain unchanged."
                self.panel.set_neuropil_proposal(True, summary)
                self._sync_current_frame_preview(render=True)
                self.set_status("Proposal ready. Correct marks to refresh, Apply to accept, or Cancel preview.")
            resources = self._acquire_job_resources("Neuropil preview", context, memory_extent=editable,
                estimate=neuropil_memory_estimate(filtering, editable, frame, len(participants),
                    relation_domain_count=len({rule.source_label_id for rule in frame.labels.relations
                                               if rule.source_label_id in participants})))
            self._run_background("Neuropil preview", work, publish, operation_snapshot=context,
                                 resources=resources, memory_extent=editable)
        except (ValueError, InterruptedError, SegmentationPreflightError) as exc:
            self.set_status(str(exc))

    def preview_relation_repair(self):
        if not self.active or self._target_locked() or not self.panel.advanced_toggle.isChecked():
            return
        self.cancel_neuropil()
        frame = self._frame_state()
        workspace = self._frame_workspace()
        try:
            context = self._preflight_authoritative_target("Relationship repair")
            dependencies = relation_dependencies(workspace, frame)
            revision, generation = frame.revision, self._operation_generation
            geometry = self._detached_image(context.source_image)
            def work(report):
                report(10, "Checking accepted operands and previewing repair")
                return repair_proposal(workspace, frame, geometry, report)
            def publish(proposal):
                if (frame.revision != revision or generation != self._operation_generation or
                        not dependencies_current(workspace, dependencies)):
                    self.set_status("Repair discarded because affected label work changed.")
                    return
                self._neuropil_proposal = (proposal, context, frame, revision, generation, dependencies, "repair")
                losses = [f"{label.name}: −{proposal.losses[label.id]:,}" for label in frame.labels.labels]
                self.panel.set_neuropil_proposal(True, "Repair preview: " + "; ".join(losses) + ". Parents remain unchanged. Apply or Cancel preview.")
                self._sync_current_frame_preview(render=True)
                self.set_status("Repair ready. Inspect the preview, then Apply to accept with one Undo.")
            resources = self._acquire_job_resources("Relationship repair", context,
                memory_extent=frame.assignment_extent, estimate=repair_memory_estimate(frame))
            self._run_background("Relationship repair", work, publish, operation_snapshot=context,
                                 resources=resources, memory_extent=frame.assignment_extent)
        except (ValueError, PermissionError, SegmentationPreflightError) as exc:
            self.set_status(str(exc))

    def preview_label_logic(self):
        if not self.active or self._target_locked() or not self.panel.advanced_toggle.isChecked():
            return
        from madi3d_app.segmentation.label_logic import logic_extent, logic_memory_estimate, logic_proposal
        self.cancel_neuropil()
        frame, workspace = self._frame_state(), self._frame_workspace()
        operation = self.panel.logic_operation.currentData()
        operands = (self.panel.logic_first.currentData(), *(item.data(Qt.ItemDataRole.UserRole)
                    for item in self.panel.logic_others.selectedItems()))
        destination = frame.labels.active_label_id
        reassign = self.panel.logic_reassign.isChecked()
        try:
            context = self._preflight_authoritative_target("Label logic")
            if len(operands) < 2 or not operands[0]:
                raise ValueError("Select a first operand and at least one further accepted label.")
            extent = logic_extent(workspace, frame, operands, context.source_image)
            dependencies = []
            for label_id in operands:
                operand, _ = find_label(workspace, label_id)
                dependencies.append((operand, label_id, operand.label_revisions.get(label_id, 0)))
            dependencies = tuple(dependencies)
            revision, generation = frame.revision, self._operation_generation
            geometry = self._detached_image(context.source_image)
            names = {label.id: f"{other.labels.name} / {label.name}"
                     for other in workspace.sets.values() for label in other.labels.labels}
            def work(report):
                report(10, "Combining accepted label masks")
                proposal = logic_proposal(workspace, frame, geometry, operation, operands, destination,
                                          allow_reassignment=reassign, report=report)
                affected = {label.id for label in frame.labels.labels
                            if proposal.gains[label.id] or proposal.losses[label.id]}
                return proposal, relation_dependencies(workspace, frame, affected)
            def publish(result):
                proposal, rule_dependencies = result
                accepted_dependencies = dependencies + rule_dependencies
                if (frame.revision != revision or generation != self._operation_generation or
                        not dependencies_current(workspace, accepted_dependencies)):
                    self.set_status("Label logic discarded because affected labels changed. Preview again.")
                    return
                self._neuropil_proposal = (proposal, context, frame, revision, generation, accepted_dependencies, "logic")
                changes = [f"{label.name}: +{proposal.gains[label.id]:,} / −{proposal.losses[label.id]:,}"
                           for label in frame.labels.labels]
                by_value = {label.value: label.name for label in frame.labels.labels}
                transfers = [f"{by_value[left]} → {by_value[right]}: {count:,}"
                             for (left, right), count in proposal.transfers.items()]
                summary = f"{operation.title()}: " + "; ".join(names[label_id] for label_id in operands)
                summary += f" → {names[destination]}. " + "; ".join(changes) + "."
                if transfers:
                    summary += " Reassignment: " + "; ".join(transfers) + "."
                self.panel.set_neuropil_proposal(True, summary)
                self._sync_current_frame_preview(render=True)
                self.set_status("Label logic ready. Inspect the preview, then Apply or Cancel preview.")
            resources = self._acquire_job_resources("Label logic", context, memory_extent=extent,
                                                   estimate=logic_memory_estimate(frame, extent))
            self._run_background("Label logic", work, publish, operation_snapshot=context,
                                 resources=resources, memory_extent=extent)
        except (ValueError, PermissionError, InterruptedError, SegmentationPreflightError) as exc:
            self.set_status(str(exc))

    def apply_neuropil(self):
        pending = self._neuropil_proposal
        if pending is None or self._target_locked() or not self.panel.advanced_toggle.isChecked():
            return
        proposal, context, frame, revision, generation, dependencies, kind = pending
        workspace = self._frame_workspace()
        try:
            self._accept_background_result(context, "Neuropil Apply")
            if (frame is not self._frame_state() or revision != frame.revision or
                    generation != self._operation_generation or self._neuropil_proposal is not pending or
                    not dependencies_current(workspace, dependencies)):
                raise ValueError("The proposal changed. Preview again before Apply.")
            computation = self._capture_computation(frame)
            def work(report):
                computation.report = report
                report(20, "Preparing joint Undo")
                validate_candidate(computation.workspace, computation.frame_state,
                                   proposal.extent, proposal.values,
                                   computation.source, require_complete=kind == "repair",
                                   require_marks=kind == "logic")
                entry = HistoryEntry.make_categorical(proposal.extent, proposal.baseline, proposal.values, mode="neuropil")
                prepared, _ = computation.prepare_assignment(
                    entry, proposal.values, categorical=True, baseline=proposal.baseline,
                    borrowed_arrays=(proposal.values,))
                return entry, prepared
            def publish(result):
                entry, prepared = result
                if (frame.revision != revision or self._operation_generation != generation or
                        self._neuropil_proposal is not pending or
                        not dependencies_current(workspace, dependencies)):
                    self._discard_neuropil_proposal()
                    self._sync_current_frame_preview(render=True)
                    self.set_status("The proposal changed. Preview again before Apply.")
                    return
                retain = self._history_decision(frame, entry)
                self._adopt_prepared_edit(frame, prepared)
                self._neuropil_live = False
                self._push_history(frame, entry, retain=retain)
                self._neuropil_live = kind == "neuropil"
                self._project_modified("segmentation authored work")
                self._sync_current_frame_preview(render=True)
                self.set_status("Applied the displayed joint proposal. Undo restores all affected labels.")
            self._run_background("Apply Neuropil", work, publish, operation_snapshot=context,
                                 memory_extent=proposal.extent)
        except (ValueError, PermissionError, SegmentationPreflightError) as exc:
            self._discard_neuropil_proposal()
            self._sync_current_frame_preview(render=True)
            self.set_status(str(exc))

    def cancel_neuropil(self):
        self._neuropil_live = False
        self._neuropil_refresh_pending = False
        self._neuropil_refresh_timer.stop()
        self._discard_neuropil_proposal()
        self.cancel_operation()
        self._sync_current_frame_preview(render=True)

    def _advanced_toggled(self, enabled):
        # Activation is runtime interaction state; authored work stays in its
        # existing frame workspace. Retire captured strokes and queued results
        # before changing the effective brush geometry.
        self.style.cancel_interaction()
        self._neuropil_live = False
        self._invalidate_operation()
        self._reset_brush_feedback(render=False)
        self.overlay.set_radii(self.settings["seed_radius"], self.settings["growth_radius"])
        self._sync_current_frame_preview(render=True)
        if self.active and self.brush_interaction_active:
            self.update_brush_cursor(*self.overlay.center)
        if not enabled:
            self.set_status("Advanced editing off. The volume brush edits the current label; advanced work is retained.")

    def capture_workspace(self, *, cooperative=False, reserve_memory=None):
        """Detach authored state, optionally returning per-frame GUI steps."""
        channels = dict(self._saved_workspace.channels)
        for state in self.states_by_channel.values():
            geometry = state.session_geometry_current or state.session_geometry_start
            if not geometry or not geometry.get("channel_id"):
                continue
            channels[geometry["channel_id"]] = SegmentationChannelWorkspace(
                state.source_extent, state.source_dimensions, geometry, state.frame_states)
        steps = SegmentationWorkspaceState(channels).capture_steps(
            reserve_memory=reserve_memory)
        if cooperative:
            return steps
        while True:
            try:
                next(steps)
            except StopIteration as completed:
                return completed.value

    def restore_workspace(self, workspace):
        self.reset()
        self._saved_workspace = workspace

    @QtCore.Slot()
    def _schedule_target_validation(self):
        self._target_validation_timer.start(0)

    def _install_tree_guards(self):
        if self._tree_connections_installed:
            return True
        tree = self.snapshots.tree_adapter.tree
        if tree is None:
            return False
        connections = []
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        context = "tree model notifications"
        try:
            model = tree.model()
            bindings = [
                ("target validation timer", timer.timeout, self.validate_target),
                ("model.rowsRemoved", model.rowsRemoved, self._schedule_target_validation),
                ("model.modelReset", model.modelReset, self._schedule_target_validation),
            ]
            # Some hosts expose only model notifications. Selection/load following
            # uses the item signals when the host provides them.
            for name, callback in (
                ("itemChanged", self.notify_volume_load_state_changed),
                ("currentItemChanged", self.sync_target_to_latest_selection),
                ("itemSelectionChanged", self.sync_target_to_latest_selection),
            ):
                signal = getattr(tree, name, None)
                if signal is not None:
                    bindings.append((f"tree.{name}", signal,
                                     lambda *_, callback=callback: QtCore.QTimer.singleShot(0, callback)))
            self._target_validation_timer = timer
            for context, signal, slot in bindings:
                signal.connect(slot)
                connections.append((signal, slot))
        except Exception:
            _LOG.exception("Failed to install segmentation tree guard: %s", context)
            for signal, slot in reversed(connections):
                signal.disconnect(slot)
            timer.stop()
            timer.deleteLater()
            self._target_validation_timer = None
            return False
        self._tree_connections_installed = True
        return True

    def set_status(self, text):
        self.panel.set_status(text)

    def _stop_for_authoritative_error(self, error):
        message = self._preflight_diagnostic(error)
        self.set_status(message)
        if bool(getattr(self, "active", False)):
            self._deactivate_session(keep_target=True)
        self.panel.set_enabled_checked(False)
        QtWidgets.QMessageBox.warning(
            self.dialog_parent,
            "Volume Segmentation Geometry Mismatch",
            message,
        )

    def _get_interactor(self):
        try:
            getter = getattr(self.render_window, "_get_interactor", None)
            if callable(getter):
                obj = getter()
                if obj is not None:
                    return obj
        except Exception:
            pass
        return self.render_window.vtk_widget.GetRenderWindow().GetInteractor()

    @timed("render_wall")
    def render_once(self):
        self._brush_render_timer.stop()
        try:
            self.render_window.render()
        except Exception:
            try:
                self.interactor.GetRenderWindow().Render()
            except Exception:
                pass

    def request_brush_render(self):
        if not self._brush_render_timer.isActive():
            self._brush_render_timer.start(0)

    def _render_requested_brush(self):
        if (self.brush_interaction_active and not self.style.right_camera_dragging
                and not self.style.middle_camera_dragging):
            self.render_once()

    def _pulse_brush_feedback(self):
        color_index = 1 if self._brush_feedback_phase else 0
        self._brush_feedback_phase = not self._brush_feedback_phase
        self.overlay.set_cursor_colors(self.overlay.BUSY_COLORS[color_index])
        if self.brush_interaction_active:
            self.render_once()

    def _start_brush_feedback(self):
        """Start visual feedback for a brush-driven segmentation operation."""
        if not self.brush_interaction_active:
            return False
        self._brush_feedback_reset_timer.stop()
        self._brush_feedback_timer.stop()
        self._brush_feedback_phase = False
        self._pulse_brush_feedback()
        self._brush_feedback_timer.start()
        return True

    def _finish_brush_feedback(self, success):
        self._brush_feedback_timer.stop()
        self._brush_feedback_reset_timer.stop()
        if success:
            self.overlay.set_cursor_colors(self.overlay.DONE_COLOR)
            if self.brush_interaction_active:
                self.render_once()
            self._brush_feedback_reset_timer.start()
            return
        self._reset_brush_feedback()

    def _reset_brush_feedback(self, render=True):
        self._brush_feedback_timer.stop()
        self._brush_feedback_reset_timer.stop()
        self._brush_feedback_phase = False
        self.overlay.reset_cursor_colors()
        if render and self.brush_interaction_active:
            self.render_once()

    # ------------------------------------------------------------------
    # target/session lifecycle
    # ------------------------------------------------------------------
    def _volume_for_tree_item(self, item):
        if item is None:
            return None
        try:
            if not self.snapshots.tree_adapter.is_volume_item(item) or not bool(item.data(0, ROLE_LOADED)):
                return None
            key = self.scene_tree.residency_key(item)
            if (key is not None and self.scene_tree.residency.operation_only(key)
                    and item is not self._activation_item
                    and not (self.active and item is self.target_item)):
                return None
            return self.selection.get_volume_container(item)
        except Exception:
            return None

    def _latest_selected_volume(self):
        tree = self.snapshots.tree_adapter.tree

        # The current row is the authoritative "latest selected" row when it is
        # a loaded volume.  If Qt leaves currentItem on a non-volume during a
        # programmatic/multi-selection update, fall back to the selected volume
        # rows without disturbing the previously remembered target unnecessarily.
        current = tree.currentItem()
        vc = self._volume_for_tree_item(current)
        if vc is not None:
            return vc, current

        try:
            selected = list(tree.selectedItems())
        except Exception:
            selected = []
        for item in reversed(selected):
            vc = self._volume_for_tree_item(item)
            if vc is not None:
                return vc, item
        return None, None

    def sync_target_to_latest_selection(self, *, load_if_active=True):
        selected = self.snapshots.tree_adapter.tree.currentItem()
        identity = self.snapshots.resolve_registered_channel(selected)
        if self._activation_item is not None and selected is not self._activation_item:
            self._cancel_activation()
        if (load_if_active and self.active and identity is not None
                and self._volume_for_tree_item(selected) is None):
            self._request_activation(selected)
            return True
        vc, item = self._latest_selected_volume()
        if vc is None or item is None:
            return False
        if vc is self.target_vc:
            self.target_item = item
            return True

        if self.active:
            try:
                if not self._switch_session_target(vc, item):
                    return False
            except SegmentationPreflightError as exc:
                self._stop_for_authoritative_error(exc)
                return False
            self.set_status(
                f"Segmentation target switched to {item.text(0)}. "
                "Masks and Undo/Redo history for other volumes are retained."
            )
        else:
            self._set_target(vc, item, announce=False)
            self.set_status(f"Target follows selection: {item.text(0)}")
        return True

    def notify_volume_load_state_changed(self):
        """Re-evaluate the selected target after a volume loads or unloads.

        Selection and loading are independent state transitions in MADI3D.  In
        particular, a lazy/unloaded volume can already be the current selected
        row when its VolumeContainer is created.  No selectionChanged signal is
        emitted at that point, so explicitly reconsider the current selection
        first and only fall back to target validation if no selected loaded
        volume is available.
        """
        try:
            if self.sync_target_to_latest_selection(load_if_active=False):
                return True
        except Exception as exc:
            print(f"[VolumeSegmentation] Could not follow volume load state: {exc}")

        # No selected loaded volume could be adopted.  This still handles the
        # inverse transition where the current target was unloaded or removed.
        try:
            self.validate_target()
        except Exception as exc:
            print(f"[VolumeSegmentation] Could not validate target after load-state change: {exc}")
        return False
    # automatic, so this simply synchronizes to the latest selected volume.

    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if not enabled:
            self._cancel_activation()
        if enabled == self.active:
            self.panel.set_enabled_checked(enabled)
            return
        if enabled:
            item = self.snapshots.tree_adapter.tree.currentItem()
            identity = self.snapshots.resolve_registered_channel(item)
            if identity is not None:
                self._request_activation(identity.item)
                return
            self.sync_target_to_latest_selection()
            if self.target_vc is None:
                self.panel.set_enabled_checked(False)
                self.set_status("Select a volume before enabling Volume Segmentation.")
                return
            if not self._activate_session() and not self.active:
                self.panel.set_enabled_checked(False)
        else:
            self._deactivate_session(keep_target=True)

    def _cancel_activation(self):
        inputs, self._activation_inputs = self._activation_inputs, None
        self._activation_item = None
        if inputs is not None:
            inputs.cancel()

    def _request_activation(self, item):
        """Explicit editing demand; queue loads never enter this route."""
        if self._activation_item is item:
            return
        self._cancel_activation()
        self._activation_item = item
        self.panel.set_enabled_checked(True)

        def ready(inputs):
            if self._activation_item is not item:
                inputs.release()
                return
            self._activation_inputs = None
            self._activation_item = None
            try:
                vc = self.selection.get_volume_container(item)
                if self.active:
                    self._switch_session_target(vc, item)
                else:
                    self._set_target(vc, item, announce=False)
                    self._activate_session()
            except SegmentationPreflightError as exc:
                self._stop_for_authoritative_error(exc)
            finally:
                # The continuous session lease is installed before this handoff.
                inputs.release()

        def terminal(inputs, error, cancelled):
            if self._activation_item is not item:
                return
            self._activation_inputs = None
            self._activation_item = None
            self.panel.set_enabled_checked(self.active)
            self.set_status("Segmentation loading cancelled." if cancelled else
                            f"Could not load segmentation signal: {error}")

        try:
            inputs = self.scene_tree.acquire_operation_inputs(
                self, [item], on_ready=ready, on_terminal=terminal,
                on_loading=lambda *_: self.set_status("Loading segmentation signal..."))
        except Exception:
            self._activation_item = None
            self.panel.set_enabled_checked(self.active)
            raise
        if self._activation_item is item:
            self._activation_inputs = inputs

    def _default_fluorescence_threshold(self, vc, lo, hi):
        """Choose a cheap non-zero signal floor without scanning the full volume."""
        md = getattr(vc, "metadata", {}) or {}
        try:
            display_floor = float(md.get("lower_threshold", lo))
        except Exception:
            display_floor = float(lo)
        span = max(0.0, float(hi) - float(lo))
        # Five percent of the observed scalar range is conservative for typical
        # fluorescence data: clearly above an all-zero/background floor, but low
        # enough to leave dim processes available for refinement.
        fluorescence_floor = float(lo) + 0.05 * span
        if hi > 0.0:
            fluorescence_floor = max(0.0, fluorescence_floor)
        candidate = max(float(lo), display_floor, fluorescence_floor)
        candidate = min(float(hi), candidate)
        if candidate == 0.0 and hi > 0.0:
            candidate = min(float(hi), max(float(hi) * 0.05, np.finfo(float).eps))
        return float(candidate)

    def _set_target(self, vc, item, announce=True):
        self._invalidate_operation()
        self._boundary_gradient_cache_key = None
        self._boundary_gradient_cache = None
        self._session_target_snapshot = None
        if vc is None or getattr(vc, "image", None) is None:
            raise RuntimeError("Invalid segmentation target")
        source_extent = _extent_tuple(vc.image.GetExtent())
        dims = tuple(int(v) for v in vc.image.GetDimensions())
        if source_extent is None or any(v <= 0 for v in dims):
            raise RuntimeError("Invalid target image geometry")
        identity = self.snapshots.resolve_registered_channel(item or vc)
        if identity is None:
            raise SegmentationPreflightError("The target has no registered source channel.")
        key = identity.channel_id
        state = self.states_by_channel.get(key)
        if state is None:
            state = SegmentationState(vc, source_extent, dims)
            self.states_by_channel[key] = state
        # The accepted source/grid evidence remains unchanged until preflight.
        # A replacement container is only a transient binding, never a new mask owner.
        state.target_vc = vc
        self.target_vc = vc
        self.target_item = item
        self.state = state
        name = item.text(0) if item is not None else os.path.basename(str(getattr(vc, "name", "Volume")))
        self.panel.set_target_name(name)
        lo, hi = self._scalar_range(vc)
        if state.ui_threshold is None or not math.isfinite(float(state.ui_threshold)):
            state.ui_threshold = self._default_fluorescence_threshold(vc, lo, hi)
        state.ui_threshold = max(lo, min(hi, float(state.ui_threshold)))
        self.signal_threshold = float(state.ui_threshold)
        self.upper_threshold = hi
        self.panel.configure_threshold(lo, hi, self.signal_threshold)
        self._update_history_ui()
        if announce:
            self.set_status(f"Target follows selection: {name}")

    def _activate_session(self):
        if self.target_vc is None or not self._target_is_valid():
            return False
        if getattr(self, "_session_residency_lease", None) is not None and not self.active:
            target = self.target_vc
            self._deactivate_session(keep_target=True)
            if self.active or self.target_vc is not target:
                return self.active
        lease = self._acquire_session_lease(self.target_item)
        self._session_residency_lease = lease
        try:
            try:
                self._preflight_authoritative_target(
                    "Enable segmentation", reset_session=True
                )
            except Exception as exc:
                if self._session_residency_lease is not lease:
                    self._release_session_lease(lease)
                    return False
                self._session_target_snapshot = None
                self._session_residency_lease = None
                self._release_session_lease(lease)
                if isinstance(exc, SegmentationPreflightError):
                    self._stop_for_authoritative_error(exc)
                    return False
                raise
            if not self._session_lease_is_current(lease):
                return False

            try:
                self._invalidate_operation()
                self.active = True
                self.brush_interaction_active = False
                self.render_window._volume_segmentation_active = False
                self.panel.set_enabled_checked(True)
                self.panel.set_controls_active(not self._target_locked())
                self._update_history_ui()
                self._stop_volume_playback()
                if not self._session_lease_is_current(lease):
                    return False
                self._install_target_observers()
                if not self._session_lease_is_current(lease):
                    return False
                self._acquire_preview_port()
                if not self._session_lease_is_current(lease):
                    return False
                self._sync_current_frame_preview(render=False)
                use_brush = any(
                    button.isChecked() for button in self.panel.mode_buttons.values()
                )
                self.set_brush_interaction_active(use_brush, render=False)
                self.render_once()
            except Exception:
                if self._session_residency_lease is not lease:
                    return False
                self._deactivate_session(keep_target=True)
                raise
            if not self._session_lease_is_current(lease):
                return False
            if self.brush_interaction_active:
                self._set_brush_status()
            else:
                self.set_status(
                    "Mask editing enabled in Navigate mode. Choose Paint, Erase or Grow to edit."
                )
            return True
        finally:
            if self._session_residency_lease is lease and not self._session_lease_is_current(lease):
                self._deactivate_session(keep_target=True)

    def _acquire_session_lease(self, item):
        tree = self.scene_tree
        key = tree.residency_key(item) or tree.register_residency_item(item)
        return tree.residency, tree.residency.acquire_lease(key, "segmentation session")

    @staticmethod
    def _release_session_lease(lease):
        if lease is not None:
            registry, token = lease
            registry.release_lease(token)

    def _session_lease_is_current(self, lease):
        if self._session_residency_lease is not lease:
            return False
        registry, token = lease
        try:
            return (
                registry.epoch == token.key.epoch
                and registry.snapshot(token.key).owner is token.owner
            )
        except KeyError:
            return False

    def _switch_session_target(self, vc, item):
        # Protect the replacement before callbacks can release its storage, and
        # retain the old binding until its observers and preview are detached.
        lease = self._acquire_session_lease(item)
        previous = self._session_residency_lease
        old_inputs_retired = False
        lease_installed = False
        try:
            self.style.cancel_interaction()
            if self._session_residency_lease is not previous:
                return False
            self._session_residency_lease = lease
            lease_installed = True
            if not self._session_lease_is_current(lease):
                return False
            self._release_preview_port()
            if not self._session_lease_is_current(lease):
                old_inputs_retired = True
                return False
            self._remove_target_observers()
            old_inputs_retired = True
            if not self._session_lease_is_current(lease):
                return False
            self._session_target_snapshot = None
            self._set_target(vc, item, announce=False)
            if not self._session_lease_is_current(lease):
                return False
            self._preflight_authoritative_target(
                "Segmentation target switch", reset_session=True
            )
            if not self._session_lease_is_current(lease):
                return False
            self._install_target_observers()
            if not self._session_lease_is_current(lease):
                return False
            self._acquire_preview_port()
            if not self._session_lease_is_current(lease):
                return False
            self._sync_current_frame_preview(render=True)
        except Exception:
            if not old_inputs_retired:
                if self._session_residency_lease is lease:
                    self._session_residency_lease = previous
                if lease_installed:
                    self._release_session_lease(lease)
                raise
            if self._session_residency_lease is not lease:
                return False
            self._deactivate_session(keep_target=True)
            raise
        finally:
            if not lease_installed:
                self._release_session_lease(lease)
            elif self._session_residency_lease is lease and not self._session_lease_is_current(lease):
                self._deactivate_session(keep_target=True)
                old_inputs_retired = True
            if old_inputs_retired:
                self._release_session_lease(previous)
        self._release_retained_storage()
        return self._session_lease_is_current(lease)

    def has_active_work(self):
        return self._target_locked()

    def volume_unload_block_reason(self, vc):
        # Cancelled jobs can still borrow snapshots until their terminal callback.
        borrowed = any(
            worker.resources is not None
            and any(token.owner is vc for token in worker.resources.inputs)
            for worker in self._worker_refs.values()
        )
        if borrowed or (self.active and self.target_vc is vc):
            return "Finish segmentation and its background cleanup before unloading."
        return ""

    def forget_unloaded_volume(self, vc):
        """Drop only disposable references after the unload guard accepted release."""
        reason = self.volume_unload_block_reason(vc)
        if reason:
            raise RuntimeError(reason)
        retained_workspace = False
        for state in self.states_by_channel.values():
            if state.target_vc is vc:
                retained_workspace = retained_workspace or bool(state.frame_states)
                state.target_vc = None
        if self.target_vc is vc:
            self._remove_target_observers()
            self._session_target_snapshot = None
            self._boundary_gradient_cache_key = None
            self._boundary_gradient_cache = None
            self._release_preview_port()
            self._discard_neuropil_proposal()
            self.target_vc = None
            self.target_item = None
            self.state = None
            self.panel.set_target_name("None")
        if retained_workspace:
            self.set_status("Signal released; the segmentation workspace, including any masks and history, remains available.")

    def reset(self, *, preserve_masks=False):
        """Retire project-bound operations; cancelled jobs retain their borrowed inputs."""
        self._deactivate_session(keep_target=False, force_cancel_all=True)
        if not preserve_masks:
            self.states_by_channel.clear()
            self._saved_workspace = SegmentationWorkspaceState()
            self._boundary_gradient_cache_key = None
            self._boundary_gradient_cache = None
        self._release_retained_storage()

    def _deactivate_session(self, keep_target=True, *, force_cancel_all=False):
        self._cancel_activation()
        lease = getattr(self, "_session_residency_lease", None)
        self._invalidate_operation(force_all=force_cancel_all)
        self.set_brush_interaction_active(False, render=False)
        if self._session_residency_lease is not lease:
            return

        self._session_residency_lease = None
        self.active = False
        self._neuropil_live = False
        self._session_target_snapshot = None
        self.brush_interaction_active = False
        self.render_window._volume_segmentation_active = False
        self.section_actor.SetVisibility(False)
        self.section_labels_actor.SetVisibility(False)
        for _, actor in self._neuropil_boxes:
            actor.SetVisibility(False)
        self.panel.set_enabled_checked(False)
        self.panel.set_controls_active(False)
        self.panel.set_history_available(False, False)
        if not keep_target:
            self.target_vc = None
            self.target_item = None
            self.state = None
            self.panel.set_target_name("None")
        # Clear logical ownership before foreign VTK/renderer cleanup can reenter.
        # Preview slots detach first; a replacement activation then removes the
        # old observers itself and installs its own independent resources.
        try:
            self._release_preview_port()
            if self._session_residency_lease is None:
                self._remove_target_observers()
        except Exception:
            # If teardown did not finish, retain its exact protection for retry.
            if self._session_residency_lease is None:
                self._session_residency_lease = lease
            raise
        self._release_session_lease(lease)
        if self._session_residency_lease is None:
            self._release_retained_storage()
            self.render_once()

    def _purge_dead_states(self):
        live = set(self.relationships.model.channels)
        purged = False
        for key in list(self.states_by_channel):
            if key not in live:
                self.states_by_channel.pop(key)
                purged = True
        if purged:
            self._release_retained_storage()

    def validate_target(self):
        self._purge_dead_states()
        if self.target_vc is None:
            # If a loaded volume is already selected, keep the panel target label
            # synchronized even while segmentation is off.
            self.sync_target_to_latest_selection()
            return True
        if self._target_is_valid():
            return True

        stale = self.target_vc
        # The tree often selects another row as part of delete/unload. Prefer an
        # immediate in-session switch to that newest loaded volume over tearing the
        # segmentation interaction mode down.
        vc, item = self._latest_selected_volume()
        if vc is stale:
            vc, item = None, None

        if vc is not None and item is not None:
            if self.active:
                try:
                    if not self._switch_session_target(vc, item):
                        return False
                except SegmentationPreflightError as exc:
                    self._stop_for_authoritative_error(exc)
                    return False
                self.set_status(f"Previous target was removed; segmentation switched to {item.text(0)}.")
            else:
                self._set_target(vc, item, announce=False)
            return True

        self.set_status("The segmentation target was unloaded or removed.")
        if self.active:
            self._deactivate_session(keep_target=False)
            self.panel.set_enabled_checked(False)
        else:
            self.target_vc = None
            self.target_item = None
            self.state = None
            self.panel.set_target_name("None")
        return False

    def _target_is_valid(self):
        vc = self.target_vc
        if vc is None or getattr(vc, "image", None) is None or getattr(vc, "volume", None) is None:
            return False
        if vc not in self.volume_map.values():
            return False
        item = self._item_for_vc(vc)
        if item is None or not bool(item.data(0, ROLE_LOADED)):
            return False
        self.target_item = item
        return True

    def _item_for_vc(self, vc):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.snapshots.tree_adapter.tree)
        while iterator.value():
            item = iterator.value()
            try:
                if self.selection.get_volume_container(item) is vc:
                    return item
            except Exception:
                pass
            iterator += 1
        return None

    def _scalar_range(self, vc):
        lo, hi = resolve_volume_scalar_range(vc)
        if not math.isfinite(lo): lo = 0.0
        if not math.isfinite(hi) or hi <= lo: hi = lo + 1.0
        return lo, hi

    def current_frame_index(self):
        if self.target_vc is None:
            return 0
        return int(getattr(self.target_vc, "time_index", 0) or 0)

    def _frame_state(self, index=None):
        if self.state is None:
            return None
        frame = self.state.frame(self.current_frame_index() if index is None else index)
        if not frame.labels.labels:
            frame.labels.create_label("Label 1", color=tuple(self.settings["mask_color"]))
        return frame

    # ------------------------------------------------------------------
    # interaction/widget coordination
    # ------------------------------------------------------------------
    def _set_brush_status(self):
        self.set_status({
            "select": (
                "Paint active. Left-drag adds signal; right-drag rotates; middle-drag pans; "
                "Ctrl+wheel changes radius."
            ),
            "unselect": (
                "Erase active. Left-drag removes selected voxels; right-drag rotates; "
                "middle-drag pans; Ctrl+wheel changes radius."
            ),
            "diffuse": (
                "Grow active. Left-drag grows connected signal; right-drag rotates; "
                "middle-drag pans; Ctrl+wheel changes radius."
            ),
        }.get(self.mode, "Segmentation brush active."))

    def set_brush_interaction_active(self, enabled, render=True):
        """Give or release mouse ownership without ending segmentation itself."""
        enabled = bool(enabled)
        if not self.active:
            self.brush_interaction_active = False
            self.render_window._volume_segmentation_active = False
            return
        if enabled == self.brush_interaction_active:
            return

        if enabled:
            self._claim_mouse_tool()
            self._previous_cursor_mode = getattr(
                self.render_window, "current_cursor_mode", None
            )
            try:
                current_style = self.interactor.GetInteractorStyle()
                if current_style is not self.style:
                    self._previous_style = current_style
            except Exception:
                self._previous_style = None

            self.brush_interaction_active = True
            self.render_window._volume_segmentation_active = True
            self.interactor.SetInteractorStyle(self.style)
            try:
                if getattr(self.render_window, "pointer_actor", None) is not None:
                    self.render_window.pointer_actor.SetVisibility(False)
                self.render_window._applied_cursor_key = None
                self.render_window.vtk_widget.setCursor(QtGui.QCursor(Qt.CursorShape.BlankCursor))
            except Exception:
                pass
            self._suspend_conflicting_widgets()
            self.overlay.set_visible(True)
            try:
                x, y = self.interactor.GetEventPosition()
                self.overlay.set_center(x, y)
            except Exception:
                pass
            self._set_brush_status()
        else:
            if self.style is not None:
                self.style.cancel_interaction()
            self._reset_brush_feedback(render=False)
            self.overlay.set_visible(False)
            self._restore_conflicting_widgets()
            self.brush_interaction_active = False
            self.render_window._volume_segmentation_active = False
            try:
                if self._previous_style is not None:
                    self.interactor.SetInteractorStyle(self._previous_style)
                else:
                    self.interactor.SetInteractorStyle(
                        getattr(
                            self.render_window,
                            "default_style",
                            vtk.vtkInteractorStyleTrackballCamera(),
                        )
                    )
            except Exception:
                pass
            try:
                if self._previous_cursor_mode is not None:
                    self.render_window.set_cursor_mode(self._previous_cursor_mode)
                else:
                    self.render_window.vtk_widget.unsetCursor()
            except Exception:
                pass
            self.set_status(
                "Segmentation remains enabled; normal mouse/navigation restored. "
                "Choose Paint, Erase or Grow to resume editing."
            )

        self.panel._set_only_mode_checked(self.mode if enabled else None)
        if render:
            self.render_once()

    def _suspend_conflicting_widgets(self):
        self._saved_clipping_process = self.clipping.suspend_widget_interaction()
        self._saved_transform_interaction = self.transforms.suspend_widget_interaction()

    def _restore_conflicting_widgets(self):
        self.clipping.restore_widget_interaction(self._saved_clipping_process)
        self.transforms.restore_widget_interaction(self._saved_transform_interaction)
        self._saved_clipping_process = None
        self._saved_transform_interaction = None

    def camera_signature(self):
        cam = self.renderer.GetActiveCamera()
        vals = list(cam.GetPosition()) + list(cam.GetFocalPoint()) + list(cam.GetViewUp())
        vals += [float(cam.GetViewAngle()), float(cam.GetParallelScale()), float(cam.GetParallelProjection())]
        return tuple(round(float(v), 9) for v in vals)

    def update_brush_cursor(self, x, y):
        self.overlay.set_center(x, y)
        plane = self.panel.editing_plane
        context = self._session_target_snapshot
        if plane == "Volume projection" or context is None:
            self._brush_outline_key = None
            return
        snap = self._geometry_snapshot(context)
        center = display_ray_to_ijk(x, y, self._display_to_world,
                                   snap["world_to_vtk_ijk_affine"], plane,
                                   self.panel.section_index.value())
        if center is None:
            return
        axes = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}[plane]
        camera = self.renderer.GetActiveCamera()
        radii = (self.panel.section_radius.value(),
                 max(self.panel.section_radius.value(), self.panel.section_growth.value()))
        key = (tuple(np.rint(center)), plane, radii, camera, camera.GetMTime(),
               self.renderer.GetOrigin(), self.renderer.GetSize(), self.renderer.GetViewport(),
               self.renderer.GetTiledAspectRatio(), context.channel_id, context.frame_index,
               context.geometry_revision, context.physical_units,
               tuple(snap["vtk_ijk_to_world_affine"].ravel()),
               tuple(snap["local_index_affine"].ravel()))
        if key != self._brush_outline_key:
            # Keep homogeneous coordinates until after translation: perspective
            # division depends on the precise pointer location, even within a voxel.
            projection = _numpy_matrix4(camera.GetCompositeProjectionTransformMatrix(
                self.renderer.GetTiledAspectRatio(), 0, 1))
            transform = projection @ snap["vtk_ijk_to_world_affine"]
            angles = np.linspace(0, 2*np.pi, 48, endpoint=False)
            circle = np.array([np.cos(angles), np.sin(angles)])
            offsets = []
            for radius in radii:
                radius_map = brush_radius_ijk(radius, plane, snap["local_index_affine"], context.physical_units)
                points = np.zeros((4, len(angles)))
                points[list(axes)] = radius_map @ circle
                offsets.append((transform @ points).T)
            self._brush_outline_projection = transform, offsets
            self._brush_outline_key = key
        transform, offsets = self._brush_outline_projection
        projected_center = transform @ np.r_[center, 1]
        origin = np.asarray(self.renderer.GetOrigin())
        size = np.asarray(self.renderer.GetSize())
        for poly, offset in zip((self.overlay.inner_poly, self.overlay.outer_poly), offsets):
            homogeneous = offset + projected_center
            vertices = origin + (homogeneous[:, :2] / homogeneous[:, 3:4] + 1) * size / 2
            self.overlay.set_outline(poly, vertices)

    def step_section(self, delta):
        if self.panel.editing_plane == "Volume projection" or not self.panel.section_index.isEnabled():
            return False
        index = self.panel.section_index
        next_value = min(index.maximum(), max(index.minimum(), index.value() + int(delta)))
        if next_value != index.value() and self.style.left_dragging:
            self.style.cancel_stroke()
        self.panel.section_index.stepBy(int(delta))
        return True

    def pick_section_label(self, x, y):
        if not self.active or self.state is None:
            return False
        plane = self.panel.editing_plane
        if plane == "Volume projection":
            self.set_status("Choose XY, XZ or YZ to pick a label on a source section.")
            return False
        context = self._session_target_snapshot
        if context is None:
            return False
        snap = self._geometry_snapshot(context)
        center = display_ray_to_ijk(
            x, y, self._display_to_world, snap["world_to_vtk_ijk_affine"], plane,
            self.panel.section_index.value(),
        )
        if center is None or not np.all(np.isfinite(center)):
            return False
        ijk = tuple(int(np.rint(value)) for value in center)
        extent = context.extent
        if any(ijk[axis] < extent[2 * axis] or ijk[axis] > extent[2 * axis + 1]
               for axis in range(3)):
            return False
        point_extent = tuple(value for index in ijk for value in (index, index))
        value = int(self._frame_state().categorical_patch(point_extent).item())
        label = next((item for item in self._frame_state().labels.labels
                      if item.value == value), None)
        if label is None:
            self.set_status("No accepted label at this source position.")
            return False
        self.select_label(label.id)
        return True

    def set_mode(self, mode):
        mode = str(mode or "select").lower()
        if mode not in {"select", "unselect", "diffuse"}:
            return
        if mode != self.mode:
            self.settings["brush_radii"][self.mode] = {
                key: self.settings[key] for key in ("seed_radius", "growth_radius")
            }
            self.settings.update(self.settings["brush_radii"][mode])
            self.panel.set_radii(self.settings["seed_radius"], self.settings["growth_radius"])
            self.overlay.set_radii(self.settings["seed_radius"], self.settings["growth_radius"])
            self._persist_settings()
        self.mode = mode
        self.panel.set_mode(mode)
        self.overlay.set_growth_visible(mode == "diffuse")
        if self.active:
            self.set_brush_interaction_active(True)
        else:
            self.set_status({
                "select": "Paint: threshold-passing signal under the seed brush is added to the mask.",
                "unselect": "Erase: selected voxels under the seed brush are removed regardless of intensity.",
                "diffuse": "Grow: connected threshold-passing signal grows from the seed inside the growth brush.",
            }[mode])

    def set_smart_settings(self, payload):
        payload = dict(payload or {})
        for key in (
            "smart_local_threshold",
            "smart_faint_recovery",
            "smart_boundary_guard",
            "smart_visible_seeds",
        ):
            if key in payload:
                self.settings[key] = bool(payload[key])
        self._persist_settings()

    def set_seed_radius(self, value):
        self.settings["seed_radius"] = int(_clamp(value, 2, 260))
        if self.settings["growth_radius"] < self.settings["seed_radius"]:
            self.settings["growth_radius"] = self.settings["seed_radius"]
        self.settings["brush_radii"][self.mode] = {
            key: self.settings[key] for key in ("seed_radius", "growth_radius")
        }
        self.panel.set_radii(self.settings["seed_radius"], self.settings["growth_radius"])
        self.overlay.set_radii(self.settings["seed_radius"], self.settings["growth_radius"])
        self._persist_settings()
        self.render_once()

    def set_growth_radius(self, value):
        self.settings["growth_radius"] = max(self.settings["seed_radius"], int(_clamp(value, 5, 1600)))
        self.settings["brush_radii"][self.mode] = {
            key: self.settings[key] for key in ("seed_radius", "growth_radius")
        }
        self.overlay.set_radii(self.settings["seed_radius"], self.settings["growth_radius"])
        self._persist_settings()
        self.render_once()

    def step_radius(self, delta):
        if self.panel.editing_plane != "Volume projection":
            control = self.panel.section_radius
            control.setValue(max(control.minimum(), control.value() + float(delta)))
            return
        delta = int(delta)
        if self.mode == "diffuse":
            new_value = max(self.settings["seed_radius"], min(1600, self.settings["growth_radius"] + delta*2))
            self.panel.growth_spin.setValue(new_value)
        else:
            new_value = max(2, min(260, self.settings["seed_radius"] + delta))
            self.panel.seed_spin.setValue(new_value)

    def set_live_threshold_preview(self, enabled):
        self.settings["live_threshold_preview"] = bool(enabled)
        self._persist_settings()

    def request_threshold_preview(self, value):
        if bool(self.settings.get("live_threshold_preview", False)) and self.active:
            self.set_threshold(value)

    def set_threshold(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        if self._target_locked() or self._threshold_drain_scheduled:
            worker = self._worker_refs.get(self._operation_job)
            if worker is None or not worker.cancel_event.is_set():
                self._pending_exact_threshold = (
                    self._operation_generation, self.state, self.current_frame_index(), value
                )
            return
        self._apply_threshold(value)

    def _apply_threshold(self, value):
        lo, hi = self._scalar_range(self.target_vc) if self.target_vc is not None else (0.0, 1.0)
        new_value = max(lo, min(hi, float(value)))
        previous_value = float(self.signal_threshold)
        changed = abs(new_value - previous_value) > max(1e-12, abs(hi-lo)*1e-12)
        self.signal_threshold = new_value
        self.upper_threshold = hi
        if self.state is not None:
            self.state.ui_threshold = new_value
        self.panel.configure_threshold(lo, hi, new_value)
        if not changed or not self.active:
            return
        frame = self._frame_state()
        if frame is None or not frame.undo_stack:
            self.set_status(f"Signal threshold {new_value:.6g}. No applied stroke to recalculate.")
            return
        entry = frame.undo_stack[-1]
        entry_label = EDIT_MODE_LABELS.get(entry.mode, entry.mode.title())
        if entry.mode not in {"select", "diffuse"}:
            self.set_status(
                f"Signal threshold {new_value:.6g}. The latest stroke ({entry_label}) "
                "is not threshold-sensitive."
            )
            return
        state = self.state
        frame_index = self.current_frame_index()
        def rollback():
            if self.state is state and self.current_frame_index() == frame_index:
                self.signal_threshold = previous_value
                state.ui_threshold = previous_value
                self.panel.configure_threshold(lo, hi, previous_value)
        try:
            snapshot = self._preflight_authoritative_target("Threshold update start validation")
            computation = self._capture_computation(frame, entry.label_id)
            def work(report):
                computation.report = report
                report(10, "Threshold: recalculating latest edit")
                old = entry.unpack("old")
                patch = computation._recompute_history_entry(entry, old, threshold_override=new_value)
                updated = copy.copy(entry)
                report(85, "Threshold: preparing history")
                updated.rebase(old, patch, threshold=new_value)
                report.result_materialized_bytes = computation.result_materialized_bytes(
                    updated, patch, borrowed_entry=entry)
                # The captured current state is only needed as toggle input;
                # toggling it in place yields the pre-edit baseline without a
                # second full-ROI copy that would preserve a discarded crop.
                baseline = computation.frame_state.categorical_patch(entry.extent)
                entry.toggle_categorical(baseline, copy=False)
                prepared, _ = computation.prepare_assignment(
                    updated, patch, baseline=baseline, borrowed_entry=entry)
                return updated, prepared
            def publish(result):
                updated, prepared = result
                retain = self._history_decision(frame, updated)
                self._adopt_prepared_edit(frame, prepared)
                self._project_modified("segmentation authored work")
                if retain:
                    frame.undo_stack[-1] = updated
                else:
                    frame.undo_stack.clear()
                frame.redo_stack.clear()
                self._trim_history()
                self._adopt_computation_cache(computation)
                self._sync_current_frame_preview(render=True)
                self.set_status(
                    f"Signal threshold {new_value:.6g}: latest {entry_label} stroke recalculated exactly."
                )
            self._start_brush_feedback()
            self._run_background("Threshold", work, publish, operation_snapshot=snapshot,
                                 preserve_threshold=True, rollback=rollback,
                                 memory_extent=entry.extent, connectivity=entry.mode == "diffuse")
        except SegmentationPreflightError as exc:
            rollback()
            self._stop_for_authoritative_error(exc)
        except Exception as exc:
            rollback()
            self._finish_brush_feedback(False)
            self.set_status(f"Threshold update failed: {exc}")

    def set_mask_opacity(self, value):
        self.settings["mask_opacity"] = _clamp(value, 0.0, 1.0)
        self._apply_preview_property()
        self._persist_settings()
        self.render_once()

    def set_mask_color(self, color):
        self.settings["mask_color"] = [_clamp(v, 0.0, 1.0) for v in list(color)[:3]]
        self.panel.settings["mask_color"] = list(self.settings["mask_color"])
        self._apply_preview_property()
        self._persist_settings()
        if self.state is not None:
            self.edit_label(color=tuple(self.settings["mask_color"]))
        self.render_once()

    def _persist_settings(self):
        # Sliders can emit dozens of valueChanged signals per drag. Persist once
        # after the burst instead of performing synchronous disk I/O per pixel.
        self._settings_save_timer.start()

    def _flush_settings(self):
        self.settings = save_settings(self.settings)

    # ------------------------------------------------------------------
    # authoritative target geometry
    # ------------------------------------------------------------------
    @staticmethod
    def _preflight_diagnostic(error):
        if isinstance(error, AcquisitionSnapshotError) and error.blocking_errors:
            return str(error.blocking_errors[0].message)
        message = str(error).strip()
        return message or error.__class__.__name__

    def _capture_authoritative_target(self, stage):
        """Resolve and validate the exact loaded channel used by segmentation."""
        try:
            if not self._target_is_valid():
                raise RuntimeError("Segmentation target is no longer loaded.")
            service = self.snapshots
            identity = service.resolve_registered_channel(
                self.target_item or self.target_vc
            )
            if identity is None:
                raise RuntimeError(
                    "The segmentation target has no authoritative acquisition/channel identity."
                )
            modeled = service.loaded_scientific_channel(
                identity.item,
                operation="segmentation",
                include_data_reference=False,
                temporal_contract="selected-frame",
            )
            geometry = modeled.channel.working_geometry
            if geometry is None:
                raise RuntimeError(
                    "The segmentation target has no numerical working geometry."
                )

            target = self.target_vc
            image = getattr(target, "image", None)
            if image is None:
                raise RuntimeError("The segmentation target image is unavailable.")
            scalars = image.GetPointData().GetScalars()
            if scalars is None:
                raise RuntimeError("The segmentation target image has no scalar data.")
            extent = _extent_tuple(image.GetExtent())
            dimensions = tuple(int(value) for value in image.GetDimensions())
            if extent is None or any(value <= 0 for value in dimensions):
                raise RuntimeError("The segmentation target has an invalid VTK extent.")
            extent_dimensions = tuple(
                extent[index + 1] - extent[index] + 1
                for index in (0, 2, 4)
            )
            if extent_dimensions != dimensions:
                raise RuntimeError(
                    "The segmentation target VTK extent does not map exactly to its dimensions."
                )
            if dimensions != tuple(geometry.dimensions):
                raise RuntimeError(
                    "The segmentation target dimensions disagree with its authoritative channel grid."
                )
            frame_index = int(self.current_frame_index())
            if frame_index < 0 or (
                geometry.time_point_count is not None
                and frame_index >= int(geometry.time_point_count)
            ):
                raise RuntimeError(
                    "The segmentation target frame lies outside its authoritative time geometry."
                )
            self.target_item = identity.item
            return SegmentationTargetSnapshot(
                target_vc=target,
                target_item=identity.item,
                source_image=image,
                source_scalars=image.GetPointData().GetScalars(),
                source_scalars_mtime=int(image.GetPointData().GetScalars().GetMTime()),
                acquisition_id=str(identity.acquisition_id),
                channel_id=str(identity.channel_id),
                backing_source_id=str(modeled.channel.backing_source_id or ""),
                selector=copy.deepcopy(
                    self.relationships.model.channels[
                        str(identity.channel_id)
                    ].selector
                ),
                producing_operation_id=(
                    str(modeled.channel.producing_operation_id)
                    if modeled.channel.producing_operation_id else None
                ),
                backing_format=str(modeled.channel.backing_format or ""),
                series_identity=str(modeled.channel.series_identity or ""),
                source_checksum=modeled.channel.source_checksum,
                frame_index=frame_index,
                extent=extent,
                dimensions=dimensions,
                spacing=tuple(float(value) for value in geometry.spacing),
                physical_units=(
                    tuple(str(value) for value in geometry.physical_units)
                    if geometry.physical_units is not None
                    else None
                ),
                origin=tuple(float(value) for value in geometry.origin),
                direction=tuple(
                    tuple(float(value) for value in row)
                    for row in geometry.direction
                ),
                local_index_affine=tuple(
                    tuple(float(value) for value in row)
                    for row in geometry.local_index_to_working_affine
                ),
                pose=tuple(
                    tuple(float(value) for value in row)
                    for row in geometry.pose
                ),
                world_index_affine=tuple(
                    tuple(float(value) for value in row)
                    for row in geometry.index_to_world_affine
                ),
                coordinate_space_id=str(geometry.coordinate_space_id),
                geometry_revision=str(modeled.channel.geometry_revision or ""),
                acquisition_geometry_revision=str(
                    modeled.acquisition.geometry_revision or ""
                ),
                grid_state=str(modeled.channel.grid_state or ""),
                operation_ids=tuple(
                    operation_id
                    for operation_id in (
                        modeled.channel.producing_operation_id,
                    )
                    if operation_id
                ),
            )
        except SegmentationPreflightError:
            raise
        except Exception as exc:
            diagnostic = self._preflight_diagnostic(exc)
            raise SegmentationPreflightError(
                f"{stage}: {diagnostic}"
            ) from exc

    @staticmethod
    def _capture_volume_input_record(model, snapshot):
        selector = copy.deepcopy(snapshot.selector)
        if isinstance(selector, dict):
            selector["time_index"] = int(snapshot.frame_index)
        else:
            selector = {
                "source_selector": selector,
                "time_index": int(snapshot.frame_index),
            }
        from madi3d_app.volume.provenance import capture_volume_operation_inputs
        capture = capture_volume_operation_inputs(model, [{
            "source_id": snapshot.acquisition_id,
            "channel_id": snapshot.channel_id,
            "backing_source_id": snapshot.backing_source_id,
            "selector": selector,
            "geometry_revision": snapshot.geometry_revision,
            "acquisition_geometry_revision": snapshot.acquisition_geometry_revision,
            "shared_pose": snapshot.pose,
            "producing_operation_id": snapshot.producing_operation_id,
            "source_checksum": snapshot.source_checksum,
        }])
        return copy.deepcopy(next(iter(capture["direct_records"].values())))

    @staticmethod
    def _require_compatible_target_snapshot(
        previous,
        current,
        *,
        stage,
        require_same_frame,
    ):
        def incompatible(message):
            raise SegmentationPreflightError(f"{stage}: {message}")

        if previous.target_vc is not current.target_vc:
            incompatible("the target volume changed while the operation was running.")
        if (
            previous.acquisition_id != current.acquisition_id
            or previous.channel_id != current.channel_id
            or previous.target_item is not current.target_item
        ):
            incompatible(
                "the authoritative acquisition/channel identity changed while the operation was running."
            )
        backing_changed = (
            previous.backing_source_id != current.backing_source_id
        )
        checksum_changed = previous.source_checksum != current.source_checksum
        series_changed = previous.series_identity != current.series_identity
        persisted_generated_backing = (
            not backing_changed
            and not checksum_changed
            and previous.backing_format == "generated"
            and current.backing_format not in {"", "generated"}
        )
        if (
            backing_changed
            or checksum_changed
            or (series_changed and not persisted_generated_backing)
        ):
            incompatible(
                "the authoritative source-image identity changed while the operation was running."
            )
        frame_changed = previous.frame_index != current.frame_index
        if require_same_frame and frame_changed:
            incompatible("the target time frame changed while the operation was running.")
        image_changed = previous.source_image is not current.source_image
        if image_changed and (require_same_frame or not frame_changed):
            incompatible("the target source image changed while the operation was running.")
        if require_same_frame:
            if (previous.source_scalars is not current.source_scalars
                    or previous.source_scalars_mtime != current.source_scalars_mtime):
                incompatible("the source voxel data changed while the operation was running.")
        if previous.extent != current.extent:
            incompatible("the target VTK extent/index mapping changed.")
        if previous.dimensions != current.dimensions:
            incompatible("the target voxel dimensions changed.")

    @staticmethod
    def _require_compatible_source_grid(geometry, extent, dimensions, current):
        """Compare accepted scientific evidence, allowing new runtime identities/pose."""
        payload = current.provenance_payload()
        generated_saved = (geometry.get("backing_format") == "generated"
                           and current.backing_format not in {"", "generated"})
        identity_fields = ("acquisition_id", "channel_id", "backing_source_id",
                           "selector", "producing_operation_id")
        # Backing format may be refined by its canonical owner (e.g. TIFF to
        # OME-TIFF); absent checksum/series evidence may likewise be enriched.
        # Compare established realization evidence without requiring new hashes.
        optional_fields = ("source_checksum",) if generated_saved else (
            "source_checksum", "series_identity")
        changed = (any(key in geometry and geometry[key] != payload[key]
                       for key in identity_fields)
                   or any(geometry.get(key) not in (None, "")
                          and payload[key] not in (None, "")
                          and geometry[key] != payload[key]
                          for key in optional_fields))
        affine = geometry.get("local_index_to_working_affine")
        grid_changed = (tuple(extent) != current.extent
                        or tuple(dimensions) != current.dimensions
                        or affine is None
                        or not np.allclose(affine, current.local_index_affine,
                                           rtol=0, atol=1e-10))
        if changed or grid_changed:
            kind = "source" if changed else "source grid"
            raise SegmentationPreflightError(
                f"Retained label work uses a different {kind}. The work is retained; "
                "restore the original source and grid to resume editing.")

    def _adopt_session_snapshot(self, current, *, stage, reset=False):
        state = self.state
        saved = self._saved_workspace.channels.get(current.channel_id)
        geometry = state.session_geometry_current or state.session_geometry_start
        if geometry is not None and (reset or self._session_target_snapshot is None):
            self._require_compatible_source_grid(
                geometry, state.source_extent, state.source_dimensions, current)
        elif geometry is None and saved is not None and not state.workspace_bound:
            self._require_compatible_source_grid(
                saved.geometry, saved.source_extent, saved.source_dimensions, current)
            state.frame_states = {int(key): frame for key, frame in saved.frames.items()}
            state.workspace_bound = True
        previous = None if reset else getattr(
            self, "_session_target_snapshot", None
        )
        if previous is not None:
            self._require_compatible_target_snapshot(
                previous,
                current,
                stage=stage,
                require_same_frame=False,
            )
        previous_payload = (
            previous.provenance_payload() if previous is not None else None
        )
        current_payload = current.provenance_payload()
        source_data_changed = bool(previous is not None and (
            previous.source_scalars is not current.source_scalars
            or previous.source_scalars_mtime != current.source_scalars_mtime
        ))
        if previous is not None and (previous_payload != current_payload or source_data_changed):
            self._invalidate_operation()
        self._session_target_snapshot = current
        state = getattr(self, "state", None)
        if state is not None:
            if reset or state.session_geometry_start is None:
                state.session_geometry_start = copy.deepcopy(current_payload)
                state.session_geometry_refresh_count = 0
            elif previous_payload != current_payload:
                state.session_geometry_refresh_count += 1
            state.session_geometry_current = copy.deepcopy(current_payload)

        if previous is not None and previous_payload != current_payload:
            local_changed = (
                previous.spacing != current.spacing
                or previous.origin != current.origin
                or previous.direction != current.direction
                or previous.local_index_affine != current.local_index_affine
            )
            if local_changed and hasattr(self, "_sync_all_mask_geometry"):
                self._sync_all_mask_geometry()
            if bool(getattr(self, "active", False)):
                # No enclosing render exists for an external geometry refresh;
                # the preview sync itself must issue the single render.
                self._sync_current_frame_preview(render=True)
        return current

    def _preflight_authoritative_target(self, stage, *, reset_session=False):
        current = self._capture_authoritative_target(stage)
        state = getattr(self, "state", None)
        if state is not None:
            if state.target_vc is not current.target_vc:
                raise SegmentationPreflightError(
                    f"{stage}: the segmentation mask state belongs to a "
                    "different target volume."
                )
            if (
                tuple(state.source_extent) != current.extent
                or tuple(state.source_dimensions) != current.dimensions
            ):
                raise SegmentationPreflightError(
                    f"{stage}: the target voxel dimensions or extent/index "
                    "mapping changed during the segmentation session."
                )
        return self._adopt_session_snapshot(
            current, stage=stage, reset=reset_session
        )

    def _accept_background_result(self, operation_snapshot, label):
        current = self._capture_authoritative_target(
            f"{label} result validation"
        )
        self._require_compatible_target_snapshot(
            operation_snapshot,
            current,
            stage=f"{label} result validation",
            require_same_frame=True,
        )
        if any(getattr(operation_snapshot, key) != getattr(current, key) for key in (
            "world_index_affine", "geometry_revision", "acquisition_geometry_revision",
            "coordinate_space_id", "physical_units",
        )):
            raise SegmentationPreflightError(
                f"{label} result validation: the target geometry changed while the operation was running."
            )
        return self._adopt_session_snapshot(
            current, stage=f"{label} result validation"
        )

    def _geometry_snapshot(self, context=None):
        context = context or self._preflight_authoritative_target(
            "Segmentation geometry validation"
        )
        extent = context.extent
        extent_min = np.array([extent[0], extent[2], extent[4]], dtype=float)
        world_zero = np.asarray(
            context.world_index_affine, dtype=float
        ).reshape(4, 4)
        # snapshot maps zero-based retained array index to world. Convert to
        # absolute vtk IJK by array_index = vtk_ijk - extent_min.
        vtk_ijk_to_world = world_zero.copy()
        vtk_ijk_to_world[:3, 3] = world_zero[:3, 3] - world_zero[:3, :3] @ extent_min
        try:
            world_to_vtk_ijk = np.linalg.inv(vtk_ijk_to_world)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Target volume transform is singular.") from exc
        return {
            "extent": extent,
            "dims": context.dimensions,
            "source_spacing": context.spacing,
            "array_origin": context.origin,
            "source_direction": np.asarray(context.direction, dtype=float),
            "local_index_affine": np.asarray(
                context.local_index_affine, dtype=float
            ),
            "actor_matrix": np.asarray(context.pose, dtype=float),
            "world_index_affine": world_zero,
            "coordinate_space_id": context.coordinate_space_id,
            "geometry_revision": context.geometry_revision,
            "vtk_ijk_to_world_affine": vtk_ijk_to_world,
            "world_to_vtk_ijk_affine": world_to_vtk_ijk,
            "_target_context": context,
        }

    def _validate_source_geometry(self):
        return self._preflight_authoritative_target(
            "Segmentation edit validation"
        )

    def _target_display_depth_range(self):
        vc = self.target_vc
        image = vc.image
        b = image.GetBounds()
        actor = _numpy_matrix4(vc.volume.GetMatrix())
        zs = []
        for x in (b[0], b[1]):
            for y in (b[2], b[3]):
                for z in (b[4], b[5]):
                    world = actor @ np.array([x, y, z, 1.0], dtype=float)
                    self.renderer.SetWorldPoint(float(world[0]), float(world[1]), float(world[2]), 1.0)
                    self.renderer.WorldToDisplay()
                    display = self.renderer.GetDisplayPoint()
                    if len(display) >= 3 and math.isfinite(float(display[2])):
                        zs.append(float(display[2]))
        if not zs:
            return (0.0, 1.0)
        z0, z1 = min(zs), max(zs)
        span = max(1e-6, z1-z0)
        margin = max(0.002, span*0.05)
        return (max(0.0, z0-margin), min(1.0, z1+margin))

    def _display_to_world(self, x, y, z):
        self.renderer.SetDisplayPoint(float(x), float(y), float(z))
        self.renderer.DisplayToWorld()
        p = self.renderer.GetWorldPoint()
        if p is None or len(p) < 4 or abs(float(p[3])) < 1e-12:
            raise RuntimeError("Could not unproject segmentation brush point.")
        w = float(p[3])
        return np.array([float(p[0])/w, float(p[1])/w, float(p[2])/w], dtype=float)

    # ------------------------------------------------------------------
    # stroke geometry/stencils
    # ------------------------------------------------------------------
    @staticmethod
    def resample_stroke(points, brush_radius):
        pts = [(float(x), float(y)) for x, y in (points or [])]
        if not pts:
            return []
        max_step = max(1.0, 0.4 * max(1.0, float(brush_radius)))  # 0.2 x diameter
        result = [pts[0]]
        for end in pts[1:]:
            start = result[-1]
            dx, dy = end[0]-start[0], end[1]-start[1]
            distance = math.hypot(dx, dy)
            if distance < 1e-9:
                continue
            steps = max(1, int(math.ceil(distance/max_step)))
            for i in range(1, steps+1):
                t = i/steps
                p = (start[0] + dx*t, start[1] + dy*t)
                if math.hypot(p[0]-result[-1][0], p[1]-result[-1][1]) > 0.25:
                    result.append(p)
        return result

    def _brush_solid_ijk(self, cx, cy, radius, depth_range, world_to_ijk,
                         segments=_BRUSH_RING_SEGMENTS, cell_template=None):
        znear, zfar = depth_range
        n = max(20, int(segments))
        world = [None] * (2 * n)
        for i in range(n):
            a = 2.0*math.pi*i/n
            x = float(cx) + float(radius)*math.cos(a)
            y = float(cy) + float(radius)*math.sin(a)
            world[i] = self._display_to_world(x, y, znear)
            world[n+i] = self._display_to_world(x, y, zfar)

        # Projective unprojection and the world->ijk mapping stay per-point at
        # VTK/numpy's reference operations; only the VTK-side assembly below is
        # built in bulk, which never re-associates the coordinate arithmetic.
        ring = np.empty((2 * n, 3), dtype=float)
        for k, p in enumerate(world):
            q = world_to_ijk @ np.array([p[0], p[1], p[2], 1.0], dtype=float)
            if abs(float(q[3])) > 1e-12:
                q = q / q[3]
            ring[k, 0] = float(q[0])
            ring[k, 1] = float(q[1])
            ring[k, 2] = float(q[2])
        points = vtk.vtkPoints()
        # Store with vtkPoints' own default component type so the bulk load
        # rounds identically to the per-point InsertNextPoint reference path.
        ring_dtype = np.float32 if points.GetDataType() == vtk.VTK_FLOAT else np.float64
        points.SetData(numpy_support.numpy_to_vtk(
            np.ascontiguousarray(ring, dtype=ring_dtype), deep=True))

        polys = vtk.vtkCellArray()
        if cell_template is not None:
            polys.DeepCopy(cell_template)
        else:
            polys.DeepCopy(self._brush_cell_template(n))

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(polys)
        poly.Modified()
        return poly

    @staticmethod
    def _brush_cell_template(n):
        """Connectivity shared by every ring solid of one segment count."""
        polys = vtk.vtkCellArray()
        for i in range(n):
            j = (i+1) % n
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, i)
            quad.GetPointIds().SetId(1, j)
            quad.GetPointIds().SetId(2, n+j)
            quad.GetPointIds().SetId(3, n+i)
            polys.InsertNextCell(quad)

        near_cap = vtk.vtkPolygon()
        near_cap.GetPointIds().SetNumberOfIds(n)
        far_cap = vtk.vtkPolygon()
        far_cap.GetPointIds().SetNumberOfIds(n)
        for i in range(n):
            near_cap.GetPointIds().SetId(i, n-1-i)
            far_cap.GetPointIds().SetId(i, n+i)
        polys.InsertNextCell(near_cap)
        polys.InsertNextCell(far_cap)
        return polys


    def _capture_stroke_geometry(self, points, radius, snap, depth_range):
        """Capture renderer-dependent brush solids without rasterizing them."""
        inv = snap["world_to_vtk_ijk_affine"]
        solids = []
        if points:
            template = self._brush_cell_template(max(20, int(_BRUSH_RING_SEGMENTS)))
            solids = [
                self._brush_solid_ijk(cx, cy, radius, depth_range, inv,
                                      cell_template=template)
                for cx, cy in points
            ]
        return solids


    # ------------------------------------------------------------------
    # clipping and compiled VTK image operations
    # ------------------------------------------------------------------
    def _clipping_plane_coeffs(self, vtk_ijk_to_world):
        """Freeze the target's Global + Individual planes in the stroke voxel frame."""
        world_planes = self.render_window.effective_volume_clipping_planes(self.target_vc)
        ijk_planes = np.asarray(world_planes) @ np.asarray(vtk_ijk_to_world, dtype=float)
        return tuple(tuple(float(v) for v in plane) for plane in ijk_planes)


    # ------------------------------------------------------------------
    # persistent cropped mask + stroke-local history
    # ------------------------------------------------------------------
    def _visible_seed_projection_snapshot(self, snap):
        """Capture the center-camera projection used to reproduce visible seeding.

        The stored matrices are tiny compared with a brush domain and let threshold
        editing/Redo re-evaluate which signal was front-most from the original view.
        Built-in stereo still represents one user viewpoint here: the canonical
        center camera is used rather than treating left/right eyes as separate views.
        """
        try:
            cam = self.renderer.GetActiveCamera()
            rw = self.renderer.GetRenderWindow()
            width, height = rw.GetSize()
            width = max(1, int(width))
            height = max(1, int(height))
            aspect = float(self.renderer.GetTiledAspectRatio())
            matrix = cam.GetCompositeProjectionTransformMatrix(aspect, -1.0, 1.0)
            world_to_clip = _numpy_matrix4(matrix)
            ijk_to_clip = world_to_clip @ np.asarray(
                snap["vtk_ijk_to_world_affine"], dtype=float
            ).reshape(4, 4)

            viewport = tuple(float(v) for v in self.renderer.GetViewport())
            viewport_px = (
                viewport[0] * width,
                viewport[1] * height,
                viewport[2] * width,
                viewport[3] * height,
            )

            direction = np.asarray(cam.GetDirectionOfProjection(), dtype=float)
            norm = float(np.linalg.norm(direction))
            if not math.isfinite(norm) or norm < 1e-12:
                return None
            direction /= norm
            position = np.asarray(cam.GetPosition(), dtype=float)
            world_depth_row = np.array([
                direction[0], direction[1], direction[2],
                -float(np.dot(direction, position)),
            ], dtype=float)
            ijk_depth = world_depth_row @ np.asarray(
                snap["vtk_ijk_to_world_affine"], dtype=float
            ).reshape(4, 4)

            if not np.all(np.isfinite(ijk_to_clip)) or not np.all(np.isfinite(ijk_depth)):
                return None
            return {
                "ijk_to_clip": tuple(float(v) for v in ijk_to_clip.ravel()),
                "depth_coeff": tuple(float(v) for v in np.asarray(ijk_depth).ravel()),
                "viewport_px": tuple(float(v) for v in viewport_px),
            }
        except Exception as exc:
            print(f"[VolumeSegmentation] Could not capture Visible Seeds view: {exc}")
            return None


    def _history_frames(self):
        # Restored workspaces can share the live frame owners. They also retain
        # work for channels which currently have no editable scene binding.
        frames = {id(frame): frame for state in self.states_by_channel.values()
                  for frame in state.iter_set_frames()}
        for channel in self._saved_workspace.channels.values():
            for workspace in channel.frames.values():
                frames.update((id(frame), frame) for frame in workspace.sets.values())
        return list(frames.values())

    @property
    def history_bytes(self):
        return history_storage_bytes(entry for frame in self._history_frames()
                                     for entry in (*frame.undo_stack, *frame.redo_stack))

    @property
    def history_byte_limit(self):
        return int(self.settings.get("history_mib", 256)) * 1024**2

    def set_history_budget(self, mib):
        self.settings["history_mib"] = max(0, min(65536, int(mib)))
        blocker = QtCore.QSignalBlocker(self.panel.history_budget_spin)
        self.panel.history_budget_spin.setValue(self.settings["history_mib"])
        del blocker
        self._persist_settings()
        # A pending undo/recompute still owns its input and timeline position.
        # Apply reductions after its terminal callback, including cancellation.
        if not self._target_locked():
            self._trim_history()
            self._update_history_ui()

    def _trim_history(self):
        if self._publishing_operation is not None and self._publication_edit is not None:
            return  # Keep rollback owners until assignment/history/display acceptance.
        frames = self._history_frames()
        trim_history(frames, self.history_byte_limit)
        self._release_retained_storage()

    def _storage_inventory(self, frames=None, *, auxiliary=True):
        """Explicit retained backing, without reading pixels or copying payloads.

        Records keep backing alive until a terminal/lifecycle sweep proves it is
        no longer reachable. Keys identify shared numerical storage and bytes;
        source pixels and non-history wrapper headers are excluded. Command
        headers and payloads follow the history owner's storage inventory.
        """
        storage = {}
        def array(value, owner=None):
            if value is None:
                return
            root = np.asarray(value)
            while isinstance(root.base, np.ndarray):
                root = root.base
            if root.nbytes:
                key = ("array", root.__array_interface__["data"][0], root.nbytes)
                storage[key] = (root.nbytes, value if owner is None else owner)
        def image(value):
            if value is not None and value.GetPointData().GetScalars() is not None:
                array(numpy_support.vtk_to_numpy(value.GetPointData().GetScalars()), value)
        for frame in self._history_frames() if frames is None else frames:
            image(frame.assignments)
            for entry in (*frame.undo_stack, *frame.redo_stack):
                for _, value in history_storage_objects(entry):
                    storage[("history", id(value))] = (sys.getsizeof(value), value)
            for marks in (*frame.labels.include_marks.values(), *frame.labels.negative_marks.values(),
                          frame.labels.outside_marks, frame.labels.protected_marks):
                if marks is not None:
                    array(marks.values)
        if auxiliary:
            image(getattr(self, "_preview_output_image", None))
            image(self.section_labels_actor.GetInput())
            cache = self._boundary_gradient_cache
            if cache is not None:
                array(cache[1])
            pending = self._neuropil_proposal
            if pending is not None:
                array(pending[0].baseline)
                array(pending[0].values)
        return storage

    def _release_retained_storage(self):
        records = self._retained_storage
        if not records:
            return
        live = self._storage_inventory()
        for worker in self._worker_refs.values():
            live.update(worker.storage_inputs)
        live.update(self._publishing_inputs)
        retained = []
        for resources, storage in records:
            storage = {key: value for key, value in storage.items() if key in live}
            active = resources.release_materialized(remaining_bytes=sum(value[0] for value in storage.values()))
            if active and storage:
                retained.append((resources, storage))
        self._retained_storage = retained

    def _history_decision(self, frame, entry):
        """Resolve an oversized command before accepting any authored change."""
        if entry.has_categorical and not entry.categorical_delta.materialized:
            command_bytes = (history_storage_bytes((entry,))
                             + entry.categorical_delta.analysis.retained_delta_bytes)
        else:
            command_bytes = history_storage_bytes((entry,))
        command_budget = entry.history_byte_limit
        if command_budget is None:
            command_budget = self.history_byte_limit
        if command_bytes <= command_budget:
            return True
        generation, state, revision = self._operation_generation, self.state, frame.revision
        operation = self._publishing_operation
        accepted = self.panel.confirm_without_undo(command_bytes, command_budget)
        if (not accepted or generation != self._operation_generation or state is not self.state
                or revision != frame.revision or frame is not self._frame_state()
                or (operation is not None and not self._operation_is_current(operation))):
            raise SegmentationCancelled("Edit cancelled; label work and history are unchanged.")
        if operation is not None:
            self._accept_background_result(operation["snapshot"], operation["label"])
        return False

    def _reserve_history_allocation(self, future_bytes, basis, *, materialized_bytes=0,
                                    exact=False):
        """Keep new history/evidence beside scratch on the operation's original token."""
        resources = self._publishing_resources
        operation = self._publishing_operation
        if resources is None or operation is None:
            return
        current = resources.registry.memory.admit(resources.request).estimate
        future = int(future_bytes)
        if not exact:
            publication = segmentation_memory_estimate(
                operation["snapshot"].source_image, operation["frame"],
                extent=operation["memory_extent"], publication_only=True,
                mark_preview=self.panel.advanced_toggle.isChecked(),
                history_byte_limit=self.history_byte_limit)
            future = max(future, publication.demand.main_bytes)
        resources.resize(replace(current, revision=current.revision + 1,
                                 demand=MemoryDemand(future, future), basis=basis),
                         materialized_bytes=max(materialized_bytes, self._publishing_result_bytes))
        if not self._operation_is_current(operation):
            raise SegmentationCancelled("The segmentation target changed during memory admission.")
        self._accept_background_result(operation["snapshot"], operation["label"])

    def _mark_preview_bytes(self, frame, extent):
        if not self.panel.advanced_toggle.isChecked():
            return 0
        marks = (*frame.labels.include_marks.values(), *frame.labels.negative_marks.values(),
                 frame.labels.outside_marks, frame.labels.protected_marks)
        if not any(mark is not None for mark in marks):
            return 0
        extent = extent_union(extent, frame.assignment_extent)
        for mark in marks:
            if mark is not None:
                extent = extent_union(extent, mark.extent)
        return int(np.prod(extent_shape_zyx(extent))) * 18

    def _mark_history_plan(self, frame, entry):
        worker, publication, result, retained = frame.mark_replay_memory(entry)
        review = self._mark_review_bytes(frame, entry.extent, entry.evidence_new, entry.mark_kind)
        publication = max(publication, retained + self._mark_preview_bytes(frame, entry.extent),
                          retained + review)
        peak = max(worker, result + publication)
        return MemoryEstimate(MemoryDemand(peak, peak), resident_bytes=None,
            basis="Mark history: unpacked evidence, normalization, replacement mark unions, relationship review and preview; "
                  "resident packed history and current marks excluded; native/helper scratch incomplete.",
            unknown_scopes=("main", "host")), publication

    def _mark_review_bytes(self, frame, extent, changed, kind):
        planned = ({(frame.labels.set_id, label_id): extent for label_id in changed}
                   if kind == "include" else None)
        return relation_review_bytes(self._frame_workspace(), changed, mark_extents=planned)

    def _reserve_mark_edit(self, frame, extent, future, basis, materialized, kind):
        count = basis["affected_mark_patches"]
        retained = (basis["command_overhead_bytes"] + 2 * count * basis["packed_bytes_per_patch"]
                    + count * basis["roi_voxels"] + basis["replacement_mark_voxels"])
        self._reserve_history_allocation(
            max(future, retained + self._mark_preview_bytes(frame, extent),
                retained + self._mark_review_bytes(frame, extent, basis["affected_mark_ids"], kind)),
            f"Segmentation evidence: {count} affected mark patches; crop, packing, replacement, relationship review and preview stages",
            materialized_bytes=materialized, exact=True)

    def _push_history(self, frame_state, entry, clear_redo=True, *, retain=True):
        if entry.mark_kind:
            self._review_history_change(frame_state, entry)
        self._discard_neuropil_proposal()
        self._neuropil_refresh_pending = self._neuropil_live
        if not retain:
            # Crossing an unrecorded edit disconnects both sides of this timeline.
            frame_state.undo_stack.clear()
            frame_state.redo_stack.clear()
        else:
            self._history_sequence += 1
            entry.accepted_sequence = self._history_sequence
            frame_state.undo_stack.append(entry)
        if clear_redo and retain:
            frame_state.redo_stack.clear()
        self._trim_history()
        self._update_history_ui()

    def _commit_history_entry(self, frame_state, entry, clear_redo=True, *,
                              prepared, redo_entry=None):
        retain = self._history_decision(frame_state, entry)
        if prepared is not None:
            self._adopt_prepared_edit(frame_state, prepared)
            self._project_modified("segmentation authored work")
        if redo_entry is not None:
            frame_state.redo_stack.pop()
        self._push_history(frame_state, entry, clear_redo=clear_redo, retain=retain)

    def undo(self):
        if not self.active or self._target_locked():
            return
        frame = self._frame_state()
        if frame is None or not frame.undo_stack:
            return
        entry = frame.undo_stack[-1]
        estimate, publication = self._mark_history_plan(frame, entry) if entry.mark_kind else (None, None)
        computation = self._capture_computation(frame, entry.label_id)
        def work(report):
            computation.report = report
            report(20, "Undo: restoring label work")
            if entry.mark_kind:
                report.reserve_memory(estimate.demand.main_bytes, estimate.basis)
                report.publication_memory_bytes = publication
                patches = entry.unpack_marks("old")
                report.result_materialized_bytes = sum(value.nbytes for value in patches.values())
                return patches
            if entry.has_categorical:
                baseline = computation.frame_state.categorical_patch(entry.extent)
                patch = entry.toggle_categorical(baseline)
                prepared, _ = computation.prepare_assignment(
                    entry, patch, categorical=True, baseline=baseline,
                    borrowed_entry=entry)
            else:
                patch = entry.unpack("old")
                prepared, _ = computation.prepare_assignment(
                    copy.copy(entry), patch, borrowed_entry=entry)
            return prepared
        def publish(patch):
            if entry.mark_kind:
                frame.write_mark_patches(entry.mark_kind, entry.extent, patch)
            else:
                self._adopt_prepared_edit(frame, patch)
            frame.undo_stack.pop()
            frame.redo_stack.append(entry)
            if entry.mark_kind:
                self._review_history_change(frame, entry)
            self._discard_neuropil_proposal()
            self._neuropil_refresh_pending = self._neuropil_live
            self._project_modified("segmentation authored work")
            self._sync_current_frame_preview(render=True)
            self.set_status("Undo: restored previous label work.")
        self._run_background("Undo", work, publish, memory_extent=entry.extent, estimate=estimate)

    def redo(self):
        if not self.active or self._target_locked():
            return
        frame = self._frame_state()
        if frame is None or not frame.redo_stack:
            return
        entry = frame.redo_stack[-1]
        estimate, publication = self._mark_history_plan(frame, entry) if entry.mark_kind else (None, None)
        computation = self._capture_computation(frame, entry.label_id)
        def work(report):
            computation.report = report
            report(10, "Redo: restoring edit")
            if entry.mark_kind:
                report.reserve_memory(estimate.demand.main_bytes, estimate.basis)
                report.publication_memory_bytes = publication
                patches = entry.unpack_marks("new")
                report.result_materialized_bytes = sum(value.nbytes for value in patches.values())
                return entry, patches
            if entry.has_categorical and not entry.new_bits:
                baseline = computation.frame_state.categorical_patch(entry.extent)
                patch = entry.toggle_categorical(baseline)
                prepared, _ = computation.prepare_assignment(
                    entry, patch, categorical=True, baseline=baseline,
                    borrowed_entry=entry)
                return entry, prepared
            old = _mask_patch(computation.frame_state, entry.extent)
            patch = computation._recompute_history_entry(entry, old)
            updated = copy.copy(entry)
            updated.rebase(old, patch)
            prepared, _ = computation.prepare_assignment(updated, patch, borrowed_entry=entry)
            return updated, prepared
        def publish(result):
            updated, patch = result
            if entry.mark_kind:
                retain = self._history_decision(frame, updated)
                frame.write_mark_patches(entry.mark_kind, entry.extent, patch)
                frame.redo_stack.pop()
                self._push_history(frame, updated, clear_redo=False, retain=retain)
            elif entry.has_categorical and not entry.new_bits:
                retain = self._history_decision(frame, updated)
                self._adopt_prepared_edit(frame, patch)
                frame.redo_stack.pop()
                self._push_history(frame, updated, clear_redo=False, retain=retain)
            else:
                self._commit_history_entry(frame, updated, clear_redo=False, prepared=patch, redo_entry=entry)
            self._project_modified("segmentation authored work")
            self._adopt_computation_cache(computation)
            self._sync_current_frame_preview(render=True)
            self.set_status("Redo: reapplied label work.")
        self._run_background("Redo", work, publish, memory_extent=entry.extent,
                             connectivity=entry.mode == "diffuse", estimate=estimate)

    def clear(self):
        if not self.active or self._target_locked():
            return
        frame = self._frame_state()
        if frame is None or frame.assignments is None:
            return
        extent = frame.assignment_extent
        computation = self._capture_computation(frame)
        def work(report):
            computation.report = report
            report(20, "Clear Mask: preparing history")
            source = computation.frame_state
            label = source.labels.active_label
            if label is None:
                return None
            # One worker-owned categorical baseline serves the zero candidate,
            # reversible delta, validation, and rollback for this operation.
            baseline = source.categorical_patch(extent)
            membership = baseline == label.value
            if not np.any(membership):
                return None
            if label.locked:
                raise PermissionError(f"Label {label.name!r} is locked")
            values = baseline.copy()
            values[membership] = 0
            if source.labels.protected_marks is not None:
                protected = source.mark_patch("protected", extent) != 0
                protected &= membership
                values[protected] = baseline[protected]
            entry = HistoryEntry.make_categorical(extent, baseline, values, mode="clear")
            entry.label_id = label.id
            prepared, _ = computation.prepare_assignment(
                entry, values, categorical=True, baseline=baseline)
            return entry, prepared
        def publish(result):
            if result is not None:
                entry, prepared = result
                self._commit_history_entry(frame, entry, prepared=prepared)
                self._sync_current_frame_preview(render=True)
            self.set_status("Segmentation cleared.")
        self._run_background("Clear Mask", work, publish, memory_extent=extent)

    def _update_history_ui(self):
        fs = self._frame_state() if self.state is not None else None
        self.panel.set_history_available(bool(fs and fs.undo_stack), bool(fs and fs.redo_stack))

    # ------------------------------------------------------------------
    # Select / Unselect / Diffuse
    # ------------------------------------------------------------------

    @timed("gui_dispatch")
    def process_stroke(self, raw_points, frame_index):
        if not self.active or not raw_points or self._target_locked():
            return
        try:
            context = self._validate_source_geometry()
            if int(frame_index) != self.current_frame_index():
                return
            with span("capture_geometry"):
                section = self.panel.editing_plane
                points = (self.resample_stroke(raw_points, self.settings["seed_radius"])
                          if section == "Volume projection" else raw_points)
                if not points:
                    return
                mode = self.mode
                snap = self._geometry_snapshot(context)
                mark_mode = self.panel.editing_mark
                if section == "Volume projection":
                    if mark_mode != "assignments":
                        self.set_status("Choose XY, XZ or YZ to place depth-controlled marks.")
                        return
                    depth = self._target_display_depth_range()
                    seed = self._capture_stroke_geometry(points, self.settings["seed_radius"], snap, depth)
                    growth = (self._capture_stroke_geometry(points, self.settings["growth_radius"], snap, depth)
                              if mode == "diffuse" else ())
                else:
                    args = (points, self._display_to_world, snap, section, self.panel.section_index.value())
                    options = dict(depth=self.panel.section_depth.value(), physical_units=context.physical_units)
                    seed = section_stroke_geometry(*args, self.panel.section_radius.value(), **options)
                    growth = (section_stroke_geometry(*args, max(self.panel.section_radius.value(),
                              self.panel.section_growth.value()), **options) if mode == "diffuse" else ())
                planes = self._clipping_plane_coeffs(snap["vtk_ijk_to_world_affine"])
                settings = dict(self.settings)
                projection = None
                if mode != "unselect" and settings.get("smart_visible_seeds", False):
                    projection = self._visible_seed_projection_snapshot(snap)
            frame = self._frame_state(frame_index)
            if frame.labels.active_label.locked and not any(kind in mark_mode for kind in ("outside", "protected")):
                self.set_status(f"{frame.labels.active_label.name} is locked. Unlock it to edit.")
                return
            label_id = frame.labels.active_label_id
            computation = self._capture_computation(frame)
            threshold = float(self.signal_threshold)

            @timed("worker")
            def work(report):
                computation.report = report
                if mark_mode != "assignments":
                    result = computation.mark_stroke(seed, snap, planes)
                    if result is not None:
                        report.result_materialized_bytes = result[1].nbytes
                    return result
                result = computation.stroke(mode, seed, growth, snap, planes, threshold, settings, projection)
                if result is None:
                    return None
                entry, patch, changed, detail = result
                if mode == "unselect" and not changed:
                    return (entry, changed, detail), None, 0
                prepared, transferred = computation.prepare_assignment(entry, patch)
                return (entry, changed, detail), prepared, transferred

            @timed("gui_publish")
            def publish(result):
                if result is None:
                    self.set_status("The stroke did not intersect the target volume.")
                    return
                if mark_mode != "assignments":
                    extent, domain = result
                    if mark_mode in {"protected", "erase_protected"}:
                        kind = "protected"
                    elif mark_mode in {"outside", "erase_outside"}:
                        kind = "outside"
                    elif mark_mode in {"negative", "erase_negative"}:
                        kind = "negative"
                    else:
                        kind = "include"
                    marks = frame.mark_patch(kind, extent, label_id)
                    if mark_mode.startswith("erase_") or mode == "unselect":
                        marks[domain] = 0
                    else:
                        marks[domain] = 1
                    entry, updates = frame.prepare_mark_patch(kind, extent, marks, label_id,
                        reserve_memory=lambda future, basis: self._reserve_mark_edit(
                            frame, extent, future, basis, domain.nbytes + marks.nbytes, kind))
                    retain = self._history_decision(frame, entry)
                    frame.write_mark_patches(kind, extent, updates)
                    self._push_history(frame, entry, retain=retain)
                    self._project_modified("segmentation authored work")
                    self._sync_current_frame_preview(render=True)
                    self.set_status("Protection updated; yellow regions keep their current values. Undo restores this protection edit." if kind == "protected"
                                    else "Marks updated. Accepted assignments are unchanged. Orange marks conflict with fixed assignments or other evidence.")
                    return
                result, prepared, transferred = result
                entry, changed, detail = result
                if mode == "unselect" and not changed:
                    self.set_status("Erase: no selected voxels changed.")
                    return
                with span("publish_assignment"):
                    self._commit_history_entry(frame, entry, prepared=prepared)
                self._adopt_computation_cache(computation)
                if entry.smart_local_threshold:
                    lo, hi = computation.scalar_range
                    self.signal_threshold = entry.threshold
                    self.state.ui_threshold = entry.threshold
                    self.panel.configure_threshold(lo, hi, entry.threshold)
                with span("final_preview"):
                    self._sync_current_frame_preview(render=True)
                self.set_status(detail + (f" Reassigned {transferred:,} voxels from other labels; Undo restores them."
                                         if transferred else ""))
            self._start_brush_feedback()
            # Publication and Undo retain the original brush extent even
            # when the worker rasterizes and grows in a smaller clipping ROI.
            geometry = (*seed, *growth) if mark_mode == "assignments" and mode == "diffuse" else seed
            memory_extent = stroke_extent(geometry, snap["extent"])
            self._run_background(EDIT_MODE_LABELS[mode], work, publish, operation_snapshot=context,
                                 memory_extent=memory_extent, connectivity=mode == "diffuse")
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
        except Exception as exc:
            self._finish_brush_feedback(False)
            self.set_status(f"Segmentation operation failed: {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # preview volume
    # ------------------------------------------------------------------
    def _acquire_preview_port(self):
        if self.preview_port is not None:
            return
        port = self.render_window._allocate_port()
        if port is None:
            if not self._preview_warning_shown:
                self._preview_warning_shown = True
                QtWidgets.QMessageBox.warning(
                    self.dialog_parent, "Segmentation preview unavailable",
                    "All shared volume-renderer ports are in use. Segmentation remains functional, but the temporary 3-D mask preview is disabled."
                )
            return
        self.preview_port = port
        self.preview_producer = vtk.vtkTrivialProducer()
        self.preview_volume = vtk.vtkVolume()
        self.preview_property = vtk.vtkVolumeProperty()
        self.preview_volume.SetProperty(self.preview_property)
        self.preview_volume.SetVisibility(False)
        self._apply_preview_property()
        empty_extent = self.state.source_extent
        tiny = (empty_extent[0], empty_extent[0], empty_extent[2], empty_extent[2], empty_extent[4], empty_extent[4])
        empty = _new_binary_image_like(self.target_vc.image, tiny, fill=0)
        self.preview_producer.SetOutput(empty)
        self._preview_output_image = empty
        self._preview_matrix_signature = None
        self.render_window.volume_scene.mapper.SetInputConnection(port, self.preview_producer.GetOutputPort())
        self.render_window.volume_scene.actor.SetVolume(self.preview_volume, port)
        self._sync_preview_transform()

    def _release_preview_port(self):
        port = self.preview_port
        volume = self.preview_volume
        self.preview_port = None
        self.preview_volume = None
        self.preview_property = None
        self.preview_producer = None
        self._preview_output_image = None
        self._preview_matrix_signature = None
        self.section_actor.SetInputData(None)
        self.section_labels_actor.SetInputData(None)
        self._section_label_map.SetInputData(None)
        self._section_label_map.GetOutput().Initialize()
        self._section_target = None
        if port is None:
            return
        try:
            volume.SetVisibility(False)
        except Exception:
            pass
        try:
            self.render_window.volume_scene.remove_volume(port)
        except Exception:
            try:
                self.render_window.volume_scene.mapper.RemoveInputConnection(port, 0)
                self.render_window.volume_scene.actor.RemoveVolume(port)
            except Exception:
                pass
        try:
            self.render_window._release_port(port)
        except Exception:
            pass

    def _apply_preview_property(self):
        prop = self.preview_property
        if prop is None:
            return
        opacity = float(self.settings["mask_opacity"])
        ctf = vtk.vtkDiscretizableColorTransferFunction()
        ctf.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        otf = vtk.vtkPiecewiseFunction()
        otf.AddPoint(0.0, 0.0)
        frame = self._frame_state() if self.state is not None else None
        if frame is not None:
            for label in sorted(frame.labels.labels, key=lambda label: label.value):
                ctf.AddRGBPoint(float(label.value), *label.color)
                otf.AddPoint(float(label.value), opacity if label.visible else 0.0)
            mark_values = self._mark_preview_values(frame)
            label_count = len(frame.labels.labels)
            for index, label in enumerate(frame.labels.labels):
                include_value = mark_values[index]
                negative_value = mark_values[label_count + index]
                color = tuple(0.35 + 0.65 * component for component in label.color)
                ctf.AddRGBPoint(float(include_value), *color)
                otf.AddPoint(float(include_value), 1.0 if label.visible else 0.0)
                ctf.AddRGBPoint(float(negative_value), 1.0, 0.1, 0.8)
                otf.AddPoint(float(negative_value), 1.0 if label.visible else 0.0)
            conflict_value = mark_values[2 * label_count]
            ctf.AddRGBPoint(float(conflict_value), 1.0, 0.25, 0.0)
            otf.AddPoint(float(conflict_value), 1.0)
            outside_value = mark_values[2 * label_count + 1]
            ctf.AddRGBPoint(float(outside_value), 0.0, 0.9, 1.0)
            otf.AddPoint(float(outside_value), 1.0)
            protected_value = mark_values[2 * label_count + 2]
            ctf.AddRGBPoint(float(protected_value), 1.0, 0.9, 0.1)
            otf.AddPoint(float(protected_value), 0.7)
        ctf.SetScalarOpacityFunction(otf)
        ctf.EnableOpacityMappingOn()
        ctf.Build()
        self._section_label_map.SetLookupTable(ctf)
        prop.SetColor(ctf)
        prop.SetScalarOpacity(otf)
        prop.SetInterpolationTypeToNearest()
        prop.ShadeOff()
        prop.Modified()

    @staticmethod
    def _mark_preview_values(frame):
        return mark_preview_values(frame)

    def _label_preview_image(self, frame):
        pending = self._neuropil_proposal
        return label_preview_image(frame, self.target_vc.image, self._frame_workspace(),
            advanced=self.panel.advanced_toggle.isChecked(),
            proposal=None if pending is None else pending[0])

    def _set_preview_prop_matrix(self, matrix):
        if self.preview_volume is None:
            return
        mat = vtk.vtkMatrix4x4()
        mat.DeepCopy(matrix)
        self.preview_volume.SetUserTransform(None)
        try:
            self.preview_volume.SetUserMatrix(None)
        except Exception:
            pass
        self.preview_volume.SetOrigin(0.0, 0.0, 0.0)
        self.preview_volume.SetPosition(0.0, 0.0, 0.0)
        self.preview_volume.SetOrientation(0.0, 0.0, 0.0)
        self.preview_volume.SetScale(1.0, 1.0, 1.0)
        self.preview_volume.SetUserMatrix(mat)
        self.preview_volume.Modified()

    def _sync_preview_transform(self):
        if self.preview_port is None or self.preview_volume is None or self.target_vc is None:
            return
        matrix = self.target_vc.volume.GetMatrix()
        signature = _matrix_signature(_numpy_matrix4(matrix))
        if signature == self._preview_matrix_signature:
            return
        self._set_preview_prop_matrix(matrix)
        helper = getattr(self.render_window, "apply_multivolume_port_matrix", None)
        if callable(helper):
            helper(self.preview_port, self.preview_volume, self.preview_volume.GetMatrix())
        else:
            # Old source fallback: only invalidate the owning MultiVolume when the
            # transform actually changed, never on ordinary mask edits.
            self.preview_volume.Modified()
            self.render_window.volume_scene.actor.Modified()
        self._preview_matrix_signature = signature

    def _sync_mask_geometry(self, mask):
        if mask is None or self.target_vc is None:
            return
        _copy_image_geometry(self.target_vc.image, mask)

    def _sync_all_mask_geometry(self):
        if self.state is None:
            return
        for fs in self.state.iter_set_frames():
            if fs.assignments is not None:
                self._sync_mask_geometry(fs.assignments)

    def _sync_current_frame_preview(self, render=False):
        if self.preview_port is None or self.preview_producer is None:
            if render:
                with span("preview_render_wall"):
                    self.render_once()
            return
        fs = self._frame_state()
        prepared = self._publication_edit
        image = (prepared.preview if prepared is not None and prepared.preview is not None else
                 fs.assignments if prepared is not None and fs is not None else
                 self._label_preview_image(fs) if fs is not None else None)
        if image is None:
            e = self.state.source_extent
            tiny = (e[0], e[0], e[2], e[2], e[4], e[4])
            image = _new_binary_image_like(self.target_vc.image, tiny, fill=0)
            self.preview_volume.SetVisibility(False)
        else:
            self._sync_mask_geometry(image)
            # A zero-valued mask is harmless with the transfer function and avoids
            # rescanning a potentially large cropped mask on every preview sync.
            self.preview_volume.SetVisibility(True)
        replaced = self._preview_output_image is not image
        if replaced:
            self.preview_producer.SetOutput(image)
            self.preview_producer.Modified()
            self._preview_output_image = image
        # If the same vtkImageData was edited in-place, its scalar/image MTime is
        # sufficient for vtkOpenGLGPUVolumeRayCastMapper to refresh the texture.
        # Avoid mapper.Modified()/vtkMultiVolume.Modified() here because those can
        # trigger broader mapper/shader invalidation unrelated to the mask data.
        self._sync_preview_transform()
        self._update_history_ui()
        self._sync_label_controls()
        # Section synchronization must not add a second render to this path;
        # the single render request below is the only one for a preview sync.
        self._sync_section(render=False)
        if render:
            with span("preview_render_wall"):
                self.render_once()
        if replaced:
            self._release_retained_storage()

    # ------------------------------------------------------------------
    # target transform / time / geometry observers
    # ------------------------------------------------------------------
    def _image_geometry_signature(self):
        image = getattr(self.target_vc, "image", None)
        if image is None:
            return None
        direction = []
        try:
            mat = image.GetDirectionMatrix()
            direction = [float(mat.GetElement(r, c)) for r in range(3) for c in range(3)]
        except Exception:
            direction = list(np.eye(3, dtype=float).ravel())
        values = list(image.GetExtent()) + list(image.GetOrigin()) + list(image.GetSpacing()) + direction
        return tuple(round(float(v), 10) for v in values)

    def _install_target_observers(self):
        lease = self._session_residency_lease
        self._remove_target_observers()
        if lease is not self._session_residency_lease:
            return
        if self.target_vc is None:
            return
        try:
            volume = self.target_vc.volume
            tag = volume.AddObserver(
                vtk.vtkCommand.ModifiedEvent, self._on_target_modified
            )
            self._target_volume_observer = (volume, tag)
        except Exception:
            self._target_volume_observer = None
        self._install_image_observer()
        self._last_frame_index = self.current_frame_index()
        self._last_geometry_signature = self._image_geometry_signature()

    def _install_image_observer(self):
        lease = self._session_residency_lease
        image = getattr(self.target_vc, "image", None)
        if image is self._observed_image:
            return
        if self._observed_image is not None and self._target_image_observer is not None:
            try:
                self._observed_image.RemoveObserver(self._target_image_observer)
            except Exception:
                pass
        if lease is not self._session_residency_lease:
            return
        self._observed_image = image
        self._target_image_observer = None
        if image is not None:
            try:
                self._target_image_observer = image.AddObserver(
                    vtk.vtkCommand.ModifiedEvent, self._on_target_modified
                )
            except Exception:
                pass

    def _remove_target_observers(self):
        volume_observer = self._target_volume_observer
        image = self._observed_image
        image_observer = self._target_image_observer
        self._target_volume_observer = None
        self._target_image_observer = None
        self._observed_image = None
        self._last_frame_index = None
        self._last_geometry_signature = None
        self._authoritative_refresh_pending = False
        if volume_observer is not None:
            try:
                volume, tag = volume_observer
                volume.RemoveObserver(tag)
            except Exception:
                pass
        if image is not None and image_observer is not None:
            try:
                image.RemoveObserver(image_observer)
            except Exception:
                pass

    def _on_target_modified(self, caller=None, _event=None):
        if not self.active or self._sync_guard:
            return
        # Preserve invalidation even if the user switches away and back before
        # deferred validation. Image/actor MTime alone is not a source revision.
        if self.current_frame_index() != self._last_frame_index:
            self._invalidate_operation()
        # Pose and voxel-size transactions update several runtime/model fields
        # synchronously. Validate once on the next event-loop turn, after the
        # transaction has either committed or rolled back.
        if not self._authoritative_refresh_pending:
            self._authoritative_refresh_pending = True
            lease = self._session_residency_lease
            QtCore.QTimer.singleShot(
                0, lambda: self._validate_observed_target_change(lease)
            )
        # Keep a synchronized pose visually responsive; the queued authoritative
        # validation still stops the session if this was a runtime-only mutation.
        self._sync_preview_transform()

    def _validate_observed_target_change(self, lease=None):
        if lease is not None and lease is not self._session_residency_lease:
            return
        self._authoritative_refresh_pending = False
        if not self.active or self._sync_guard:
            return
        self._sync_guard = True
        try:
            if not self._target_is_valid():
                QtCore.QTimer.singleShot(0, self.validate_target)
                return
            previous = self._session_target_snapshot
            current = self._preflight_authoritative_target(
                "Segmentation session geometry refresh"
            )
            self._install_image_observer()
            frame_index = current.frame_index
            geometry_signature = self._image_geometry_signature()
            frame_changed = frame_index != self._last_frame_index
            geometry_changed = geometry_signature != self._last_geometry_signature
            image_changed = bool(
                previous is not None
                and previous.source_image is not current.source_image
            )
            self._last_geometry_signature = geometry_signature
            if frame_changed or geometry_changed or image_changed:
                self._last_frame_index = frame_index
                # Each time frame owns its own already-thresholded stroke history;
                # switching frames changes only which retained mask is previewed.
                # The visible preview changed, so this sync carries the render.
                self._sync_current_frame_preview(render=True)
            else:
                self._sync_preview_transform()
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
        finally:
            self._sync_guard = False

    def _geometry_invalidated(self):
        self._stop_for_authoritative_error(
            SegmentationPreflightError(
                "Segmentation session geometry refresh: the target voxel "
                "dimensions or extent/index mapping changed."
            )
        )

    # ------------------------------------------------------------------
    # Phase 4 outputs
    # ------------------------------------------------------------------
    def create_mask_volume(self):
        self._create_output("Create Mask Volume", "mask", "segmentation_mask", mask_volume=True)

    def create_categorical_volume(self):
        self._create_output("Create Categorical Label Set", "labels", "segmentation_labels",
                            mask_volume=True, categorical=True)

    def continue_to_cdm(self):
        self._create_output(
            "Prepare NeuronBridge selected signal",
            "selection",
            "segmentation_extract",
            on_published=self._open_signal_in_cdm,
        )

    def _open_signal_in_cdm(self, channel_id):
        try:
            self._open_selected_signal(channel_id)
        except Exception as exc:
            # Publication has succeeded; a navigation failure must not mark the
            # generated scientific output as failed or roll it back.
            self.set_status(
                f"Selected signal created, but NeuronBridge could not open it: {exc}"
            )

    def extract_selection(self):
        self._create_output("Extract Selected Voxels", "selection", "segmentation_extract")

    def delete_selected(self):
        self._create_output("Delete Selected", "unselected", "segmentation_delete_selected")

    def _derived_name(self, suffix):
        base = self.target_item.text(0) if self.target_item is not None else "volume"
        frame_suffix = f"_frame{self.current_frame_index()+1}" if getattr(self.target_vc, "has_time_series", lambda: False)() else ""
        # Generated IDs, not labels, own scientific identity. Repeated display
        # labels are preferable to scanning an arbitrarily large sibling set.
        return f"{base}_{suffix}{frame_suffix}"

    def _capture_output_publication_context(self, snapshot):
        model = self.relationships.model
        backing = model.backing_sources[snapshot.backing_source_id]
        state = self.state
        return {
            "model": model,
            "input_record": self._capture_volume_input_record(model, snapshot),
            "parent_item": snapshot.target_item,
            "origin_metadata": copy.deepcopy(
                snapshot.target_item.data(0, ROLE_OBJECT_META) or {}
            ),
            "check_state": snapshot.target_item.checkState(0),
            "target_metadata": copy.deepcopy(getattr(snapshot.target_vc, "metadata", {}) or {}),
            "session_geometry": {
                "start": copy.deepcopy(getattr(state, "session_geometry_start", None)),
                "current": copy.deepcopy(getattr(state, "session_geometry_current", None)),
                "refresh_count": int(getattr(state, "session_geometry_refresh_count", 0) or 0),
            },
            "parent_acquisition": copy.deepcopy(model.acquisitions[snapshot.acquisition_id]),
            "parent_channel": copy.deepcopy(model.channels[snapshot.channel_id]),
            "parent_backing": copy.deepcopy(backing),
            "parent_geometry": copy.deepcopy(
                model.geometry_records[snapshot.geometry_revision]
            ),
            "parent_acquisition_geometry": copy.deepcopy(
                model.geometry_records[snapshot.acquisition_geometry_revision]
            ),
            "mask_color": tuple(float(value) for value in self.settings["mask_color"]),
        }

    def _add_derived_volume(
        self,
        prepared,
        display_name,
        *,
        mask_volume=False,
        operation="segmentation",
        operation_parameters=None,
        operation_snapshot,
        publication_context,
    ):
        acceptance_started = perf_counter()
        renderer_seconds = 0.0
        relationship_controller = self.relationships
        model = relationship_controller.model
        tree = relationship_controller.tree_adapter.tree
        publication = SegmentationOutputPublication.build(
            model, prepared, display_name, operation_snapshot, publication_context,
            mask_volume=mask_volume, operation=operation,
            operation_parameters=operation_parameters,
        )
        validated_parent = operation_snapshot
        parent_item = publication_context.get("parent_item")
        acquisition_id = publication.acquisition.source_id
        working_grid = publication.acquisition.working_grid()
        categorical_support = publication.categorical_support
        normalized_image = prepared.image

        target_meta = copy.deepcopy(publication_context.get("target_metadata") or {})

        producer = vtk.vtkTrivialProducer()
        producer.SetOutput(normalized_image)
        scalar_range = prepared.scalar_range
        vc = self.VolumeContainer(
            normalized_image, display_name, None, scalar_range=scalar_range
        )
        vc.reader = producer
        if categorical_support is not None:
            vc.categorical_labels = copy.deepcopy(categorical_support["labels"])
            vc.metadata.update({"data_min": float(scalar_range[0]), "data_max": float(scalar_range[1]),
                                "global_opacity": 1.0, "representation": "volume"})
        elif mask_volume:
            mask_color = publication_context["mask_color"]
            vc.metadata.update(
                {
                    "data_min": 0.0,
                    "data_max": 1.0,
                    "lower_threshold": 0.5,
                    "upper_threshold": 1.0,
                    "saturation_point": 1.0,
                    "peak": 1.0,
                    "color": mask_color,
                    "color_lut": "solid",
                    "global_opacity": 1.0,
                    "representation": "volume",
                }
            )
        else:
            copied = serialize_volume_rendering_metadata(target_meta)
            copied.pop("clipping", None)  # Source attachments do not belong to a new result.
            copied["representation"] = "volume"
            vc.metadata.update(copied)
        vc.metadata["time_index"] = 0
        vc._madi_unsaved = True
        configure = getattr(vc, "configure_working_grid", None)
        if not callable(configure):
            raise RuntimeError(
                "Generated volume container cannot apply authoritative grid geometry."
            )
        configure(working_grid)
        try:
            vc.update_transfer_functions()
        except Exception:
            pass
        if mask_volume and categorical_support is None:
            vc.transfer["opacity"].RemoveAllPoints()
            vc.transfer["opacity"].AddPoint(0.0, 0.0)
            vc.transfer["opacity"].AddPoint(0.5, 0.0)
            vc.transfer["opacity"].AddPoint(1.0, 0.85)
            vc.transfer["color"].RemoveAllPoints()
            vc.transfer["color"].AddRGBPoint(0.0, 0.0, 0.0, 0.0)
            vc.transfer["color"].AddRGBPoint(
                1.0,
                *publication_context["mask_color"],
            )
            vc.prop.SetInterpolationTypeToNearest()
            vc.prop.ShadeOff()

        self.volume_pose.apply_exact(
            vc, validated_parent.pose, scale_mode="center"
        )
        if not np.allclose(
            _numpy_matrix4(vc.volume.GetMatrix()),
            np.asarray(validated_parent.pose, dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("Generated volume did not retain the parent scene pose.")
        try:
            vc.capture_original_pose()
        except Exception:
            pass

        item = QtWidgets.QTreeWidgetItem([display_name])
        inherited = copy_origin(
            publication_context.get("origin_metadata"),
            derived=True, operation=operation,
        ).get(OBJECT_KEY)
        if inherited:
            item.setData(0, ROLE_OBJECT_META, {OBJECT_KEY: inherited})
        item.setFlags(
            item.flags()
            | Qt.ItemFlag.ItemIsEditable
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        item.setData(0, ROLE_ITEM_TYPE, "volume")
        item.setData(0, ROLE_LOADED, True)
        item.setData(0, ROLE_LOADING, False)
        item.setData(0, ROLE_VOLUME_TIME_SERIES, False)
        item.setData(0, ROLE_VOLUME_TIME_PLAYING, False)
        item.setData(0, ROLE_SOURCE_PATH, "")
        item.setData(0, ROLE_UNSAVED, True)
        item.setData(0, ROLE_VOLUME_META, dict(vc.metadata))
        dirty_font = item.font(0)
        dirty_font.setItalic(True)
        item.setFont(0, dirty_font)
        color = tuple(
            float(v) for v in vc.metadata.get("color", (1.0, 1.0, 1.0))
        )
        item.setData(
            0,
            ROLE_COLOR,
            f"{color[0]:.3f},{color[1]:.3f},{color[2]:.3f}",
        )
        item.setCheckState(
            0,
            publication_context["check_state"],
        )

        append_scene_state = relationship_controller.capture_append_scene_state()
        old_volume_id_counter = int(self._get_next_volume_id())
        vol_id = old_volume_id_counter
        item.setData(0, ROLE_VOLUME_ID, vol_id)
        published_to_renderer = False
        append_result = None
        try:
            target_item = parent_item
            if target_item is not None:
                try:
                    if target_item.treeWidget() is not tree:
                        target_item = None
                except RuntimeError:
                    target_item = None
            if target_item is None:
                anchor = None
                insertion_parent = None
            else:
                containing_parent = target_item.parent()
                if (
                    containing_parent is not None
                    and containing_parent.data(0, ROLE_GROUP_KIND) == GROUP_KIND_MULTICHANNEL
                ):
                    anchor = containing_parent
                    insertion_parent = containing_parent.parent()
                else:
                    anchor = target_item
                    insertion_parent = containing_parent
            # Suppress item callbacks, but let the Qt model notify the live view.
            tree_blocker = QtCore.QSignalBlocker(tree)
            try:
                if insertion_parent is not None:
                    index = insertion_parent.indexOfChild(anchor)
                    insertion_parent.insertChild(
                        index + 1 if index >= 0 else insertion_parent.childCount(),
                        item,
                    )
                else:
                    index = tree.indexOfTopLevelItem(anchor) if anchor is not None else -1
                    tree.insertTopLevelItem(
                        index + 1
                        if index >= 0
                        else tree.topLevelItemCount(),
                        item,
                    )
            finally:
                del tree_blocker

            self._commit_volume_id(old_volume_id_counter + 1)
            self.volume_map[vol_id] = vc
            tree_blocker = QtCore.QSignalBlocker(tree)
            try:
                append_result = publication.append_to(relationship_controller, item)
                vc.source_id = acquisition_id
                vc.metadata.update(dict(item.data(0, ROLE_VOLUME_META) or {}))
                self._update_item_icon(
                    item, *color, icon_size=16, shape="disc"
                )
                if self.scene_tree is not None:
                    self.scene_tree.register_residency_item(item, owner=vc)
            finally:
                del tree_blocker

            bounds = self.render_window._volume_world_bounds(vc)
            vc.bbox_center_world = (
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            )
            try:
                should_show = bool(self.selection._item_effectively_checked(item))
            except Exception:
                should_show = item.checkState(0) == Qt.CheckState.Checked
            renderer_started = perf_counter()
            try:
                if should_show:
                    if not self.render_window.set_volume_visibility(
                        vc, True, render=False
                    ):
                        raise RuntimeError(
                            "No volume-renderer port is available for the generated output."
                        )
                    published_to_renderer = True
            finally:
                renderer_seconds += perf_counter() - renderer_started
            self._refresh_scale_bar()
            renderer_started = perf_counter()
            try:
                self.render_window.render()
            finally:
                # Measures synchronous VTK calls, not asynchronous GPU completion.
                renderer_seconds += perf_counter() - renderer_started
            self._project_modified("generated volume publication")
        except Exception:
            if self.scene_tree is not None:
                self.scene_tree.unregister_residency_item(item)
            if published_to_renderer or getattr(vc, "volume_index", None) is not None:
                self._remove_volume_container(vc)
            self.volume_map.pop(vol_id, None)
            self._commit_volume_id(old_volume_id_counter)
            if append_result is not None:
                append_result.delta.rollback(model)
            tree_blocker = QtCore.QSignalBlocker(tree)
            try:
                item_parent = item.parent()
                if item_parent is not None:
                    item_index = item_parent.indexOfChild(item)
                    if item_index >= 0:
                        item_parent.takeChild(item_index)
                else:
                    item_index = tree.indexOfTopLevelItem(item)
                    if item_index >= 0:
                        tree.takeTopLevelItem(item_index)
            finally:
                del tree_blocker
            append_scene_state.restore()
            try:
                self.render_window.render()
            except Exception:
                pass
            raise
        finally:
            _LOG.debug(
                "Segmentation output publication: gui_acceptance_seconds=%.6f "
                "renderer_upload_seconds=%.6f",
                perf_counter() - acceptance_started - renderer_seconds, renderer_seconds,
            )
        return vc, item


    def _start_job(self, function, resources):
        job_id = self._next_job()
        worker = _SegmentationJob(job_id, function, resources)
        operation = self._operation
        worker.storage_before = set(self._storage_inventory())
        worker.storage_inputs = self._storage_inventory(
            tuple(self._frame_workspace().sets.values())
            if operation and operation["frame"] is not None else (),
            auxiliary=False)
        worker.storage_inputs.update(self._storage_inventory((), auxiliary=True))
        worker.independent_output = bool(operation and operation.get("independent_output"))
        self._worker_refs[job_id] = worker
        worker.signals.progress.connect(self._job_progress)
        worker.signals.finished.connect(self._job_finished)
        worker.signals.failed.connect(self._job_failed)
        worker.signals.cancelled.connect(self._job_cancelled)
        worker.signals.memory_requested.connect(self._job_memory_requested)
        try:
            self._pool.start(worker)
        except Exception:
            self._worker_refs.pop(job_id, None)
            worker.function = None
            raise
        return job_id

    @QtCore.Slot(int, object)
    def _job_memory_requested(self, job_id, request):
        """Resolve worker allocation demand on the GUI thread and original token."""
        try:
            worker = self._worker_refs.get(int(job_id))
            if (worker is None or worker.cancel_event.is_set()
                    or int(job_id) != self._operation_job
                    or not self._operation_is_current(self._operation)):
                raise InterruptedError("The segmentation allocation request is no longer current.")
            resources = worker.resources
            if resources is None or not resources.publication_allowed:
                raise InterruptedError("The segmentation memory request is no longer current.")
            estimate = resources.registry.memory.admit(resources.request).estimate
            current_revision = estimate.revision
            # The worker is paused before allocating. Reconcile touched buffers
            # against fresh telemetry, then reserve only remaining allocations.
            # The shared owner keeps uncovered backing beside future demand.
            if request["publication"] and not request.get("exact_publication", False):
                estimate = segmentation_memory_estimate(
                    self._operation["snapshot"].source_image, self._operation["frame"],
                    extent=self._operation["memory_extent"], publication_only=True,
                    mark_preview=self.panel.advanced_toggle.isChecked(),
                    history_byte_limit=self.history_byte_limit,
                )
            else:
                future = request["future_bytes"]
                estimate = replace(estimate, demand=MemoryDemand(future, future),
                    basis=request["basis"] + "; materialized backing reported separately; native/helper demand incomplete.")
            resources.resize(replace(estimate, revision=current_revision + 1),
                             materialized_bytes=request["materialized_bytes"])
            worker.memory_stage_resolved = True
        except Exception as exc:
            # Do not transfer traceback frames (and their GUI/buffer owners) to
            # the worker or retain them in the request packet.
            request["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            request["done"].set()

    @QtCore.Slot(int, object)
    @timed("gui_result")
    def _job_finished(self, job_id, result):
        worker = self._worker_refs.pop(int(job_id), None)
        if int(job_id) != self._operation_job:
            self._finish_worker_resources(worker)
            self._finish_background_ui()
            return
        operation = self._operation
        self._operation_job = None
        self._operation = None
        self._publishing_operation = operation
        self._publishing_resources = worker.resources if worker is not None else None
        self._publishing_result_bytes = worker.result_materialized_bytes if worker is not None else 0
        self._publishing_inputs = worker.storage_inputs if worker is not None else {}
        success = False
        try:
            independent_output = bool(
                operation and operation.get("independent_output")
            )
            if worker is None or worker.cancel_event.is_set() or (
                worker.resources is not None and not worker.resources.publication_allowed
            ) or (
                not self._operation_is_current(operation)
            ):
                self._rollback_operation(operation)
                self.panel.finish_activity(
                    "Cancelled" if worker is not None and worker.cancel_event.is_set() else "Interrupted")
                return
            if not independent_output:
                # Selection signals are queued. Check actual selection before accepting
                # a result which mutates the active mask or adopts a destructive edit.
                selected, _item = self._latest_selected_volume()
                if selected is not None and selected is not self.target_vc:
                    self._rollback_operation(operation)
                    self.sync_target_to_latest_selection()
                    self.set_status("Segmentation result discarded because the target changed.")
                    return
                self._accept_background_result(operation["snapshot"], operation["label"])
            operation["publish"](result)
            success = True
            operation.pop("assignment_rollback", None)
            if (self._publication_edit is not None
                    and self._publication_edit.section is not None):
                self._section_label_map.SetInputData(None)
                self._section_label_map.GetOutput().Initialize()
            self.panel.finish_activity("Completed")
        except SegmentationCancelled as exc:
            self._rollback_operation(operation)
            self.panel.finish_activity("Cancelled")
            self.set_status(str(exc) or "Edit cancelled; label work and history are unchanged.")
        except SegmentationPreflightError as exc:
            self.panel.finish_activity("Interrupted", str(exc))
            self._rollback_operation(operation)
            self.set_status(f"{operation['label']} result discarded: {exc}")
        except Exception as exc:
            self.panel.finish_activity("Failed", str(exc))
            self._rollback_operation(operation)
            self.set_status(f"{operation['label']} failed: {exc}")
            traceback.print_exc()
        finally:
            if operation is not None:
                operation.clear()
            result = None
            self._publication_edit = None
            self._publishing_operation = None
            self._publishing_resources = None
            self._publishing_inputs = {}
            self._publishing_result_bytes = 0
            self._finish_worker_resources(worker)
            self._finish_brush_feedback(success)
            self._finish_background_ui()

    @QtCore.Slot(int, str, str)
    def _job_failed(self, job_id, message, details):
        worker = self._worker_refs.pop(int(job_id), None)
        if int(job_id) == self._operation_job:
            operation = self._operation
            self._operation = None
            self._operation_job = None
            self._rollback_operation(operation)
            if worker is not None and not worker.cancel_event.is_set() and self._operation_is_current(operation):
                self.set_status(f"{operation['label']} failed: {message}")
                self.panel.finish_activity("Failed", details)
                print(f"[VolumeSegmentation] {details}")
            self._finish_brush_feedback(False)
        self._finish_worker_resources(worker)
        self._finish_background_ui()

    def _finish_worker_resources(self, worker):
        if worker is not None and worker.resources is not None:
            resources, worker.resources = worker.resources, None
            worker.storage_inputs = {}
            if worker.independent_output:
                # Detached generated output has an independent publication owner.
                resources.finish()
            else:
                live = self._storage_inventory()
                owned = {key: value for key, value in live.items() if key not in worker.storage_before}
                resources.finish(disposed=not owned)
                if owned:
                    self._retained_storage.append((resources, owned))
            self._release_retained_storage()

    def _run_background(self, label, function, publish, *, operation_snapshot=None,
                        preserve_threshold=False, rollback=None,
                        independent_output=False, publication_context=None,
                        resources=None, memory_extent=None, connectivity=False, estimate=None,
                        display_only=False):
        """Dispatch once; completion returns through a queued Qt signal."""
        if self._target_locked():
            if resources is not None:
                resources.finish()
            self.set_status("A segmentation calculation is still finishing. You can navigate or cancel it.")
            return None
        try:
            snapshot = operation_snapshot or self._preflight_authoritative_target(f"{label} start validation")
        except SegmentationPreflightError as exc:
            if resources is not None:
                resources.finish()
            if rollback is not None:
                rollback()
            self._stop_for_authoritative_error(exc)
            return None
        try:
            if resources is None:
                resources = self._acquire_job_resources(
                    label, snapshot, memory_extent=memory_extent, connectivity=connectivity, estimate=estimate)
            if not resources.publication_allowed:
                raise InterruptedError("The segmentation request is no longer current.")
        except Exception as exc:
            if resources is not None:
                resources.finish()
            if rollback is not None:
                rollback()
            self._finish_brush_feedback(False)
            self.set_status(f"{label} could not start: {exc}")
            return None
        operation = {
            "label": label, "snapshot": snapshot, "publish": publish,
            "memory_extent": memory_extent,
            "generation": self._operation_generation,
            "state": None if independent_output else self.state,
            "frame": None if independent_output else self._frame_state(snapshot.frame_index),
            "frame_revision": None if independent_output else self._frame_state(snapshot.frame_index).revision,
            "rollback": rollback,
            "independent_output": bool(independent_output),
            "display_only": bool(display_only),
            "model_context": (
                publication_context.get("model")
                if independent_output and publication_context is not None
                else self.relationships.model
            ),
        }
        self.panel.set_background_busy(True, label, preserve_threshold=preserve_threshold)
        self._operation = operation
        try:
            self._operation_job = self._start_job(function, resources)
        except Exception as exc:
            self._operation = None
            self._operation_job = None
            self._rollback_operation(operation)
            function = publish = None
            operation.clear()
            resources.finish()
            self._finish_brush_feedback(False)
            self._finish_background_ui()
            self.set_status(f"{label} could not start: {exc}")
            return None
        return self._operation_job

    def _acquire_job_resources(self, label, snapshot, *, memory_extent=None,
                               connectivity=False, estimate=None):
        estimate = estimate or segmentation_memory_estimate(
            snapshot.source_image, self._frame_state(snapshot.frame_index),
            extent=memory_extent, connectivity=connectivity,
            mark_preview=self.panel.advanced_toggle.isChecked(),
            history_byte_limit=self.history_byte_limit)
        generation, state = self._operation_generation, self.state
        self._admitting = True
        resources = None
        try:
            resources = self.scene_tree.acquire_consumer_resources(
                f"segmentation {label}", (snapshot.target_item,), estimate)
            # The decision dialog can dispatch selection/reset/frame signals.
            # Nothing reads the captured buffers until this exact check succeeds.
            if generation != self._operation_generation or state is not self.state:
                raise InterruptedError("The segmentation target changed during memory admission.")
            self._accept_background_result(snapshot, label)
            return resources
        except BaseException:
            if resources is not None:
                resources.finish(disposed=True)
            raise
        finally:
            self._admitting = False

    def _target_locked(self):
        # Cancelled workers also count until their completion signal releases inputs.
        # This bounds concurrency and keeps borrowed mask scalars immutable while read.
        return bool(getattr(self, "_admitting", False) or getattr(self, "_worker_refs", {})
                    or getattr(self, "_publishing_operation", None))
    def _schedule_pending_exact_threshold(self):
        if self._target_locked() or self._threshold_drain_scheduled:
            return
        if self._pending_exact_threshold is None:
            return
        self._threshold_drain_scheduled = True
        def drain():
            self._threshold_drain_scheduled = False
            pending = self._pending_exact_threshold
            self._pending_exact_threshold = None
            if pending is None:
                return
            generation, state, frame, value = pending
            if (generation == self._operation_generation and state is self.state
                    and frame == self.current_frame_index() and self.active):
                self.set_threshold(value)
        QtCore.QTimer.singleShot(0, drain)
    def _next_job(self):
        self._job_serial += 1
        return int(self._job_serial)
    @QtCore.Slot(int, int, str)
    def _job_progress(self, job_id, value, text):
        worker = self._worker_refs.get(int(job_id))
        if int(job_id) == self._operation_job and worker is not None and not worker.cancel_event.is_set():
            self.panel.set_progress(value, text)
    @staticmethod
    def _filter_specks(mask, extent, minimum, *, report):
        mask = np.ascontiguousarray(
            np.asarray(mask, dtype=np.uint8)
        )
        image = _binary_numpy_image(mask, extent, index_geometry=True)
        filt = vtk.vtkImageConnectivityFilter()
        filt.SetInputData(image)
        filt.SetScalarRange(1.0, 1.0)
        filt.SetExtractionModeToAllRegions()
        filt.SetSizeRange(max(1, int(minimum)), max(1, int(mask.size)))
        filt.SetLabelModeToConstantValue()
        filt.SetLabelConstantValue(1)
        try:
            filt.SetLabelScalarTypeToUnsignedChar()
        except Exception:
            pass
        report.update(filt)
        return (
            np.asarray(
                _image_array_view(filt.GetOutput()), dtype=np.uint8
            ) > 0
        ).astype(np.uint8)

    def remove_specks(self, minimum):
        """Apply one small-component cleanup to the currently accepted mask.

        Threshold editing selects the minimum size only; this explicit
        operation filters once and records one Undo step.
        """
        minimum = max(1, int(minimum))
        if not self.active or self._target_locked():
            return
        frame = self._frame_state()
        if frame is None or frame.assignments is None:
            self.set_status("Remove small components: the current frame has no selected voxels.")
            return
        try:
            snapshot = self._preflight_authoritative_target("Remove small components start validation")
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
            return
        extent = tuple(int(v) for v in frame.assignment_extent)
        computation = self._capture_computation(frame)
        def work(report):
            computation.report = report
            report(10, "Remove small components: filtering selection")
            baseline = _mask_patch(computation.frame_state, extent)
            filtered = self._filter_specks(baseline, extent, minimum, report=report)
            if np.array_equal(baseline, filtered):
                report(90, "Remove small components: nothing removed")
                return None
            report(80, "Remove small components: preparing history")
            entry = HistoryEntry.make(extent, baseline, filtered, mode="specks")
            entry.label_id = computation.frame_state.labels.active_label_id
            prepared, _ = computation.prepare_assignment(entry, filtered)
            return {"minimum": minimum, "entry": entry, "prepared": prepared,
                    "before": int(np.count_nonzero(baseline)),
                    "after": int(np.count_nonzero(filtered))}
        self._run_background("Remove small components", work, self._publish_speck_removal,
                             operation_snapshot=snapshot, memory_extent=extent,
                             connectivity=True)

    def _publish_speck_removal(self, result):
        if result is None:
            self.set_status(
                "Remove small components: every connected component met the minimum size.")
            return
        frame = self._frame_state()
        entry = result["entry"]
        entry.label_id = frame.labels.active_label_id
        self._commit_history_entry(frame, entry, prepared=result["prepared"])
        self._sync_current_frame_preview(render=True)
        removed = max(0, int(result["before"]) - int(result["after"]))
        self.set_status(
            f"Remove small components: minimum {int(result['minimum'])} voxels; "
            f"{removed:,} voxels removed from the whole selection."
        )

    def _output_mask(self, *, categorical=False):
        if not self.active or self._target_locked():
            return None
        frame = self._frame_state()
        if frame is None or (frame.assignments is None and not categorical):
            self.set_status("The current frame has no selected voxels.")
            return None
        return frame

    def _create_output(self, label, suffix, operation, *, mask_volume=False, margin=0,
                       on_published=None, categorical=False):
        frame = self._output_mask(categorical=categorical)
        if frame is None:
            return
        try:
            snapshot = self._preflight_authoritative_target(f"{label} start validation")
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
            return
        resources = None
        try:
            resources = self._acquire_job_resources(
                label, snapshot, memory_extent=frame.assignment_extent,
                estimate=segmentation_output_memory(snapshot.source_image, frame, self._frame_workspace(),
                                                    operation=operation, margin=margin,
                                                    categorical=categorical, mask_volume=mask_volume))
            source = self._deep_detached_image(snapshot.source_image) if not mask_volume else None
            categorical_data = CategoricalOutputSupport.capture(frame, snapshot.extent) if categorical else None
            mask_image = (frame._new_assignments(categorical_data.values, snapshot.extent, snapshot.source_image)
                          if categorical else frame.active_label_image())
            material_context = output_context(frame, self._frame_workspace(), categorical=categorical)
            active_label = frame.labels.active_label
            active_label_meaning = {"id": active_label.id, "value": active_label.value,
                                    "name": active_label.name, "set_id": frame.labels.set_id}
            extent = tuple(snapshot.extent if categorical else frame.assignment_extent)
            name = self._derived_name(suffix)
            publication_context = self._capture_output_publication_context(snapshot)
            # Keep exact scientific evidence, without retaining live VTK buffers
            # or the container through a supposedly detached output snapshot.
            snapshot = replace(snapshot, target_vc=None, target_item=None,
                               source_image=None, source_scalars=None)
            # All subsequent numerical reads use these independent deep copies.
            resources.release_inputs()
        except Exception as exc:
            source = mask_image = None
            if resources is not None:
                resources.finish()
            self.set_status(f"{label} could not start: {exc}")
            return
        parameters = {} if categorical else {"active_label": active_label_meaning}
        if operation == "segmentation_extract_original_signal":
            parameters["margin_voxels"] = int(margin)
        def work(report):
            report(5, f"{label}: reading mask")
            parameters["segmentation_context"] = material_context.to_dict()
            if categorical:
                parameters["categorical_support"] = categorical_data.to_dict()
                report(95, f"{label}: ready")
                return PreparedSegmentationOutput.prepare(mask_image, report)
            mask = np.array(_image_array_view(mask_image), copy=True, dtype=np.uint8)
            if not np.any(mask):
                return None
            # A threshold recipe cannot reconstruct interactive segmentation.
            import base64
            import zlib
            parameters["manual_support"] = {
                "encoding": "zlib_base64_uint8_c_order", "axis_order": "ZYX",
                "shape": list(mask.shape), "extent": list(extent),
                "checksum_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "data": base64.b64encode(zlib.compress(mask.tobytes())).decode("ascii"),
                "contribution": "committed_interactive_mask",
                "reconstruction": "use_saved_support_not_thresholds",
            }
            if mask_volume:
                output = vtk.vtkImageData()
                output.DeepCopy(mask_image)
            else:
                if operation == "segmentation_delete_selected":
                    # Dispatch already owns an exclusive detached source image.
                    output = source
                else:
                    output = vtk.vtkImageData()
                    report(15, f"{label}: allocating output grid")
                    output.CopyStructure(source)
                    source_scalars = source.GetPointData().GetScalars()
                    output.AllocateScalars(source_scalars.GetDataType(),
                                           source_scalars.GetNumberOfComponents())
                    output_scalars = output.GetPointData().GetScalars()
                    output_scalars.SetName(source_scalars.GetName())
                    output_scalars.CopyComponentNames(source_scalars)
                    output_scalars.CopyInformation(source_scalars.GetInformation())
                arr = _image_array_view(output)
                report(55, f"{label}: applying mask")
                if operation == "segmentation_delete_selected":
                    arr[extent_slices_zyx(extent, output.GetExtent())][mask != 0] = 0
                else:
                    source_arr = _image_array_view(source)
                    out_extent = extent_expand(extent, margin, limit=source.GetExtent())
                    roi = np.zeros(extent_shape_zyx(out_extent), dtype=np.uint8)
                    roi[extent_slices_zyx(extent, out_extent)] = mask
                    if margin:
                        roi = self._dilate_roi(roi, out_extent, source, margin, report=report)
                    report(75, f"{label}: copying selected signal")
                    slices = extent_slices_zyx(out_extent, output.GetExtent())
                    arr.fill(0)
                    np.copyto(arr[slices], source_arr[slices], where=(roi != 0))
                output.GetPointData().GetScalars().Modified()
                output.Modified()
            report(95, f"{label}: preparing output")
            return PreparedSegmentationOutput.prepare(output, report)
        def publish(output):
            if output is None:
                self.set_status("The current frame has no selected voxels.")
                return
            _container, item = self._add_derived_volume(output, name, mask_volume=mask_volume,
                operation=operation, operation_parameters=parameters, operation_snapshot=snapshot,
                publication_context=publication_context)
            self.set_status(f"{label}: created {name}")
            if on_published is not None:
                on_published(str(item.data(0, ROLE_VOLUME_CHANNEL_ID)))
        self._run_background(label, work, publish, operation_snapshot=snapshot,
                             independent_output=True,
                             publication_context=publication_context, resources=resources)

    @staticmethod
    def _dilate_roi(mask, extent, source, margin, *, report):
        if int(margin) <= 0:
            return np.ascontiguousarray(mask, dtype=np.uint8)
        image = _binary_numpy_image(
            mask, extent, source_geometry=source
        )
        filt = vtk.vtkImageDilateErode3D()
        filt.SetInputData(image)
        diameter = 2 * int(margin) + 1
        filt.SetKernelSize(diameter, diameter, diameter)
        filt.SetDilateValue(1)
        filt.SetErodeValue(0)
        report.update(filt)
        return np.ascontiguousarray(
            _image_array_view(filt.GetOutput()), dtype=np.uint8
        )
    def extract_original_signal(self):
        margin = max(0, int(self.panel.extract_margin.value()))
        suffix = "original_signal" if margin == 0 else f"original_signal_m{margin}"
        self._create_output("Extract Original Signal", suffix,
                            "segmentation_extract_original_signal", margin=margin)


    def _operation_is_current(self, operation):
        if operation and operation.get("independent_output"):
            return operation.get("model_context") is self.relationships.model
        return bool(operation and self.active
                    and operation["generation"] == self._operation_generation
                    and operation["state"] is self.state
                    and operation["snapshot"].frame_index == self.current_frame_index()
                    and operation["frame"] is self._frame_state()
                    and operation["frame"].revision == operation["frame_revision"])

    @staticmethod
    def _rollback_operation(operation):
        if operation and operation.get("assignment_rollback") is not None:
            operation.pop("assignment_rollback")()
        if operation and operation.get("rollback") is not None:
            rollback = operation.pop("rollback")
            rollback()

    def _invalidate_operation(self, *, force_all=False):
        self._discard_neuropil_proposal()
        self._neuropil_refresh_pending = False
        timer = getattr(self, "_neuropil_refresh_timer", None)
        if timer is not None:
            timer.stop()
        self._operation_generation = getattr(self, "_operation_generation", 0) + 1
        self._pending_exact_threshold = None
        operation = getattr(self, "_operation", None)
        cancel_workers = force_all or not (operation and operation.get("independent_output"))
        if cancel_workers:
            for worker in getattr(self, "_worker_refs", {}).values():
                worker.cancel_event.set()
                if worker.resources is not None:
                    worker.resources.cancel()
        self._rollback_operation(getattr(self, "_operation", None))
        # The Activity Center record moves to "Cancelling" when the user
        # cancels, which disables its own cancel control; re-cancelling an
        # already invalid operation is a harmless no-op.

    def cancel_operation(self):
        self._neuropil_live = False
        if not self._target_locked():
            return
        retain_display = bool(self._operation and self._operation.get("display_only"))
        self._invalidate_operation(force_all=True)
        if not retain_display:
            self._sync_current_frame_preview(render=True)
        self.set_brush_interaction_active(False)
        self.panel.set_progress(0, "Cancelling")
        self.panel.mark_activity_cancelling()
        self.set_status("Cancelling operation. The last completed mask is retained; navigation is active.")

    @QtCore.Slot(int)
    def _job_cancelled(self, job_id):
        worker = self._worker_refs.pop(int(job_id), None)
        if int(job_id) == self._operation_job:
            self._rollback_operation(self._operation)
            self._operation = None
            self._operation_job = None
            self._finish_brush_feedback(False)
            self.panel.finish_activity("Cancelled")
        self._finish_worker_resources(worker)
        self._finish_background_ui()

    def _finish_background_ui(self):
        if self._target_locked():
            return
        self._trim_history()
        self.panel.set_background_busy(False)
        self._update_history_ui()
        self._schedule_pending_exact_threshold()
        if self._neuropil_refresh_pending and self._neuropil_live:
            self._neuropil_refresh_pending = False
            self._neuropil_refresh_timer.start(0)

    @staticmethod
    def _detached_image(image):
        if image is None:
            return None
        detached = vtk.vtkImageData()
        detached.ShallowCopy(image)
        return detached

    @staticmethod
    def _deep_detached_image(image):
        if image is None:
            return None
        detached = vtk.vtkImageData()
        detached.DeepCopy(image)
        return detached

    @classmethod
    def _capture_frame_state(cls, frame):
        # Scalar and mark backing is read-only under the serialized operation
        # lock, retained through cancellation until the worker terminal signal.
        labels = copy.copy(frame.labels)
        labels.labels = list(frame.labels.labels)
        labels.include_marks = dict(frame.labels.include_marks)
        labels.negative_marks = dict(frame.labels.negative_marks)
        labels.relations = list(frame.labels.relations)
        labels.nonparticipant_ids = list(frame.labels.nonparticipant_ids)
        return SegmentationFrameState(
            assignments=cls._detached_image(frame.assignments),
            assignment_extent=frame.assignment_extent, labels=labels,
            revision=frame.revision, label_revisions=dict(frame.label_revisions))

    def _capture_computation(self, frame, label_id=None):
        source = self._detached_image(self.target_vc.image)
        mask = self._capture_frame_state(frame)
        if label_id is not None:
            mask.labels.active_label_id = label_id
        computation = SegmentationComputation(
            source, self._scalar_range(self.target_vc), mask, memory_reserved=True)
        computation.history_byte_limit = self.history_byte_limit
        workspace = self._frame_workspace()
        computation.workspace = SegmentationFrameWorkspace(
            sets={key: mask if owner is frame else self._capture_frame_state(owner)
                  for key, owner in workspace.sets.items()},
            active_set_id=workspace.active_set_id)
        computation.preview_advanced = self.panel.advanced_toggle.isChecked()
        if self.panel.editing_plane != "Volume projection":
            live_lookup = self._section_label_map.GetLookupTable()
            lookup = vtk.vtkDiscretizableColorTransferFunction()
            lookup.DeepCopy(live_lookup)
            opacity = vtk.vtkPiecewiseFunction()
            opacity.DeepCopy(live_lookup.GetScalarOpacityFunction())
            lookup.SetScalarOpacityFunction(opacity)
            lookup.SetEnableOpacityMapping(live_lookup.GetEnableOpacityMapping())
            lookup.Build()
            computation.section_lookup = lookup
        computation.signal_threshold = float(self.signal_threshold)
        source_key = (id(self.target_vc.image), self.target_vc.image.GetMTime())
        cached = self._boundary_gradient_cache
        if self._boundary_gradient_cache_key == source_key and cached is not None:
            computation._boundary_gradient_cache_key, computation._boundary_gradient_cache = cached
            computation._borrowed_boundary_gradient_cache = computation._boundary_gradient_cache
        return computation

    def _adopt_prepared_edit(self, frame, prepared):
        # Preallocate small mutable containers and retain old backing until the
        # complete GUI publication succeeds. Contained edits retain only their
        # already-prepared old ROI; expanding edits roll back by pointer.
        workspace = self._frame_workspace()
        for key, revision, label_revisions in prepared.review_revisions:
            owner = workspace.sets.get(key)
            if (owner is None or owner.revision != revision
                    or owner.label_revisions != dict(label_revisions)):
                raise SegmentationPreflightError("Relationship operands changed before the reviewed result could be adopted.")
        relations = [(workspace.sets[key], list(values)) for key, values in prepared.relation_updates]
        old_relations = [(owner, owner.labels.relations) for owner, _ in relations]
        old = frame.capture_assignment_state()
        old_undo, old_redo = frame.undo_stack, frame.redo_stack
        next_undo, next_redo = list(frame.undo_stack), list(frame.redo_stack)
        old_preview = self._preview_output_image
        old_section = self.section_labels_actor.GetInput()
        old_section_visibility = self.section_labels_actor.GetVisibility()
        old_section_extent = self.section_labels_actor.GetDisplayExtent()
        old_visibility = None if self.preview_volume is None else self.preview_volume.GetVisibility()
        def restore():
            frame.restore_assignment_state(old, prepared.assignment)
            frame.undo_stack, frame.redo_stack = old_undo, old_redo
            for owner, values in old_relations:
                owner.labels.relations = values
            self._preview_output_image = old_preview
            if self.preview_producer is not None:
                self.preview_producer.SetOutput(old_preview)
            if self.preview_volume is not None:
                self.preview_volume.SetVisibility(old_visibility)
            self.section_labels_actor.SetInputData(old_section)
            self.section_labels_actor.SetVisibility(old_section_visibility)
            self.section_labels_actor.SetDisplayExtent(*old_section_extent)
        operation = self._publishing_operation
        if operation is not None:
            operation["assignment_rollback"] = restore
        if prepared.assignment is not None:
            frame.adopt_assignment_patch(prepared.assignment)
        frame.undo_stack, frame.redo_stack = next_undo, next_redo
        for owner, values in relations:
            owner.labels.relations = values
        if self._publishing_operation is not None:
            self._publication_edit = prepared

    def _adopt_computation_cache(self, computation):
        self._boundary_gradient_cache_key = (id(self.target_vc.image), self.target_vc.image.GetMTime())
        self._boundary_gradient_cache = (
            computation._boundary_gradient_cache_key, computation._boundary_gradient_cache
        )

__all__ = [
    "VolumeSegmentationController",
]
