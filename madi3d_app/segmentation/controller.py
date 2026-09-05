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
import math
import os
import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import vtk
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from vtkmodules.util import numpy_support

from madi3d_storage import atomic_write_json, config_file, read_json_object
from madi3d_version import MADI3D_VERSION
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
)
from madi3d_app.volume.model import (
    CHANNEL_ROLE_LABEL_MASK,
    CHANNEL_ROLE_OTHER,
    GROUP_KIND_MULTICHANNEL,
    VolumeSourceError,
    VolumeSourceModel,
    new_volume_identity,
)
from madi3d_app.volume.snapshot import AcquisitionSnapshotError
from madi3d_app.volume.rendering import (
    resolve_volume_scalar_range,
    serialize_volume_rendering_metadata,
)

vtk_to_numpy = numpy_support.vtk_to_numpy
numpy_to_vtk = numpy_support.numpy_to_vtk

DEFAULT_SETTINGS = {
    "seed_radius": 15,
    "growth_radius": 180,
    "mask_color": [0.56, 0.93, 0.68],
    "mask_opacity": 0.35,
    "history_limit": 40,
    "smart_defaults_initialized": True,
    # Conservative Diffuse guards are useful defaults; the more situational
    # Local threshold and Visible seeds helpers remain opt-in.
    "smart_local_threshold": False,
    "smart_faint_recovery": True,
    "smart_boundary_guard": True,
    "smart_visible_seeds": False,
    "live_threshold_preview": False,
}


def _settings_path() -> Path:
    return config_file("volume_segmentation_settings.json")


def _clamp(value, lo, hi):
    try:
        value = float(value)
    except Exception:
        return float(lo)
    return max(float(lo), min(float(hi), value))


def _sanitize_settings(raw):
    data = dict(DEFAULT_SETTINGS)
    raw = dict(raw or {})
    migrate_smart_defaults = bool(raw) and not bool(raw.get("smart_defaults_initialized", False))
    data["seed_radius"] = int(round(_clamp(raw.get("seed_radius", data["seed_radius"]), 2, 260)))
    data["growth_radius"] = int(round(_clamp(raw.get("growth_radius", data["growth_radius"]), 5, 1600)))
    data["growth_radius"] = max(data["seed_radius"], data["growth_radius"])
    try:
        color = [float(v) for v in raw.get("mask_color", data["mask_color"])[:3]]
        if len(color) != 3:
            raise ValueError
        data["mask_color"] = [_clamp(v, 0.0, 1.0) for v in color]
    except Exception:
        data["mask_color"] = list(DEFAULT_SETTINGS["mask_color"])
    data["mask_opacity"] = _clamp(raw.get("mask_opacity", data["mask_opacity"]), 0.0, 1.0)
    data["history_limit"] = int(round(_clamp(raw.get("history_limit", data["history_limit"]), 5, 100)))
    for key in (
        "smart_local_threshold",
        "smart_faint_recovery",
        "smart_boundary_guard",
        "smart_visible_seeds",
        "live_threshold_preview",
    ):
        data[key] = bool(raw.get(key, data[key]))
    if migrate_smart_defaults:
        data["smart_faint_recovery"] = True
        data["smart_boundary_guard"] = True
    data["smart_defaults_initialized"] = True
    return data


def load_settings(path=None):
    path = Path(path) if path else _settings_path()
    raw = read_json_object(path)
    return _sanitize_settings(raw)


def save_settings(settings, path=None):
    path = Path(path) if path else _settings_path()
    clean = _sanitize_settings(settings)
    try:
        atomic_write_json(path, clean)
    except Exception as exc:
        print(f"[VolumeSegmentation] Could not save settings: {exc}")
    return clean


def _numpy_matrix4(vtk_matrix):
    out = np.eye(4, dtype=float)
    if vtk_matrix is None:
        return out
    for r in range(4):
        for c in range(4):
            out[r, c] = float(vtk_matrix.GetElement(r, c))
    return out


def _vtk_matrix4(array):
    arr = np.asarray(array, dtype=float).reshape(4, 4)
    mat = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            mat.SetElement(r, c, float(arr[r, c]))
    return mat


def _copy_direction(source, target):
    if source is None or target is None:
        return
    if not hasattr(source, "GetDirectionMatrix") or not hasattr(target, "SetDirectionMatrix"):
        return
    try:
        mat = vtk.vtkMatrix3x3()
        mat.DeepCopy(source.GetDirectionMatrix())
        target.SetDirectionMatrix(mat)
    except Exception:
        pass


def _extent_tuple(extent):
    if extent is None:
        return None
    vals = tuple(int(v) for v in extent)
    if len(vals) != 6:
        return None
    if vals[1] < vals[0] or vals[3] < vals[2] or vals[5] < vals[4]:
        return None
    return vals


def _normalized_generated_image(image):
    """Return a zero-based deep copy while preserving the input physical support."""
    if image is None:
        raise ValueError("No generated image was supplied.")
    extent = _extent_tuple(image.GetExtent())
    if extent is None:
        raise ValueError("Generated images require a non-empty VTK extent.")
    dimensions = tuple(
        extent[index + 1] - extent[index] + 1 for index in (0, 2, 4)
    )
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    direction_matrix = image.GetDirectionMatrix()
    direction = np.asarray(
        [
            [direction_matrix.GetElement(row, column) for column in range(3)]
            for row in range(3)
        ],
        dtype=float,
    )
    index_offset = np.asarray(
        tuple(extent[index] for index in (0, 2, 4)), dtype=float
    )
    shifted_origin = origin + direction @ (spacing * index_offset)

    normalized = vtk.vtkImageData()
    normalized.DeepCopy(image)
    normalized.SetExtent(
        0,
        dimensions[0] - 1,
        0,
        dimensions[1] - 1,
        0,
        dimensions[2] - 1,
    )
    normalized.SetOrigin(*(float(value) for value in shifted_origin))
    normalized.SetSpacing(*(float(value) for value in spacing))
    _copy_direction(image, normalized)
    normalized.Modified()
    return normalized, extent


def _vtk_scalar_checksum(image) -> str | None:
    scalars = image.GetPointData().GetScalars() if image is not None else None
    if scalars is None:
        return None
    values = vtk_to_numpy(scalars)
    if not values.flags.c_contiguous:
        values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(memoryview(values).cast("B"))
    return "sha256:" + digest.hexdigest()


def extent_intersection(a, b):
    a = _extent_tuple(a)
    b = _extent_tuple(b)
    if a is None or b is None:
        return None
    out = (
        max(a[0], b[0]), min(a[1], b[1]),
        max(a[2], b[2]), min(a[3], b[3]),
        max(a[4], b[4]), min(a[5], b[5]),
    )
    return _extent_tuple(out)


def extent_union(a, b):
    a = _extent_tuple(a)
    b = _extent_tuple(b)
    if a is None:
        return b
    if b is None:
        return a
    return (
        min(a[0], b[0]), max(a[1], b[1]),
        min(a[2], b[2]), max(a[3], b[3]),
        min(a[4], b[4]), max(a[5], b[5]),
    )


def extent_expand(extent, amount, limit=None):
    e = _extent_tuple(extent)
    if e is None:
        return None
    n = max(0, int(amount))
    out = (e[0]-n, e[1]+n, e[2]-n, e[3]+n, e[4]-n, e[5]+n)
    return extent_intersection(out, limit) if limit is not None else out


def extent_shape_zyx(extent):
    e = _extent_tuple(extent)
    if e is None:
        return (0, 0, 0)
    return (e[5]-e[4]+1, e[3]-e[2]+1, e[1]-e[0]+1)


def extent_voxel_count(extent):
    shape = extent_shape_zyx(extent)
    return int(shape[0]) * int(shape[1]) * int(shape[2])


def extent_slices_zyx(sub_extent, parent_extent):
    sub = _extent_tuple(sub_extent)
    parent = _extent_tuple(parent_extent)
    if sub is None or parent is None:
        raise ValueError("Invalid extent")
    if extent_intersection(sub, parent) != sub:
        raise ValueError(f"Sub-extent {sub} lies outside parent extent {parent}")
    return (
        slice(sub[4]-parent[4], sub[5]-parent[4]+1),
        slice(sub[2]-parent[2], sub[3]-parent[2]+1),
        slice(sub[0]-parent[0], sub[1]-parent[0]+1),
    )


def extent_relative_to_parent(sub_extent, parent_extent):
    """Map an absolute VTK extent to zero-based retained parent-array indices."""
    sub = _extent_tuple(sub_extent)
    parent = _extent_tuple(parent_extent)
    if sub is None or parent is None or extent_intersection(sub, parent) != sub:
        raise ValueError(
            f"Generated extent {sub_extent!r} lies outside target extent {parent_extent!r}."
        )
    return (
        sub[0] - parent[0], sub[1] - parent[0],
        sub[2] - parent[2], sub[3] - parent[2],
        sub[4] - parent[4], sub[5] - parent[4],
    )


def _difference_extent(old_arr, new_arr, absolute_extent):
    diff = np.asarray(old_arr) != np.asarray(new_arr)
    if not np.any(diff):
        return None
    zz, yy, xx = np.nonzero(diff)
    e = _extent_tuple(absolute_extent)
    return (
        e[0] + int(xx.min()), e[0] + int(xx.max()),
        e[2] + int(yy.min()), e[2] + int(yy.max()),
        e[4] + int(zz.min()), e[4] + int(zz.max()),
    )


def _image_array_view(image):
    if image is None:
        raise RuntimeError("Missing vtkImageData")
    scalars = image.GetPointData().GetScalars()
    if scalars is None:
        raise RuntimeError("vtkImageData has no scalars")
    if int(scalars.GetNumberOfComponents()) != 1:
        raise RuntimeError("Segmentation requires scalar image data")
    return vtk_to_numpy(scalars).reshape(extent_shape_zyx(image.GetExtent()), order="C")


def _new_binary_image_like(source, extent, fill=0):
    e = _extent_tuple(extent)
    if e is None:
        raise ValueError("Invalid binary image extent")
    image = vtk.vtkImageData()
    image.SetExtent(*e)
    image.SetOrigin(*source.GetOrigin())
    image.SetSpacing(*source.GetSpacing())
    _copy_direction(source, image)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    arr = _image_array_view(image)
    arr.fill(int(fill))
    image.GetPointData().GetScalars().Modified()
    image.Modified()
    return image


def _binary_numpy_image(array_zyx, extent, source_geometry=None, index_geometry=False):
    arr = np.ascontiguousarray(np.asarray(array_zyx, dtype=np.uint8))
    expected = extent_shape_zyx(extent)
    if tuple(arr.shape) != tuple(expected):
        raise ValueError(f"Binary patch shape {arr.shape} does not match extent {extent} => {expected}")
    image = vtk.vtkImageData()
    image.SetExtent(*extent)
    if index_geometry:
        image.SetOrigin(0.0, 0.0, 0.0)
        image.SetSpacing(1.0, 1.0, 1.0)
    elif source_geometry is not None:
        image.SetOrigin(*source_geometry.GetOrigin())
        image.SetSpacing(*source_geometry.GetSpacing())
        _copy_direction(source_geometry, image)
    else:
        image.SetOrigin(0.0, 0.0, 0.0)
        image.SetSpacing(1.0, 1.0, 1.0)
    vtk_arr = numpy_to_vtk(arr.ravel(order="C"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    image.GetPointData().SetScalars(vtk_arr)
    image.Modified()
    return image


def _stencil_from_binary(array_zyx, extent, inside_value=1):
    image = _binary_numpy_image(array_zyx, extent, index_geometry=True)
    to_stencil = vtk.vtkImageToImageStencil()
    to_stencil.SetInputData(image)
    to_stencil.ThresholdBetween(int(inside_value), int(inside_value))
    to_stencil.Update()
    stencil = vtk.vtkImageStencilData()
    stencil.DeepCopy(to_stencil.GetOutput())
    return stencil


def _stencil_to_bool(stencil, expected_extent=None):
    if stencil is None:
        if expected_extent is None:
            return None
        return np.zeros(extent_shape_zyx(expected_extent), dtype=bool)
    conv = vtk.vtkImageStencilToImage()
    conv.SetInputData(stencil)
    conv.SetInsideValue(1)
    conv.SetOutsideValue(0)
    conv.SetOutputScalarTypeToUnsignedChar()
    conv.Update()
    image = conv.GetOutput()
    arr = np.array(_image_array_view(image), copy=True, dtype=bool)
    out_extent = _extent_tuple(image.GetExtent())
    if expected_extent is None or out_extent == _extent_tuple(expected_extent):
        return arr
    result = np.zeros(extent_shape_zyx(expected_extent), dtype=bool)
    inter = extent_intersection(out_extent, expected_extent)
    if inter is not None:
        result[extent_slices_zyx(inter, expected_extent)] = arr[extent_slices_zyx(inter, out_extent)]
    return result


def _copy_image_geometry(source, target):
    target.SetOrigin(*source.GetOrigin())
    target.SetSpacing(*source.GetSpacing())
    _copy_direction(source, target)
    target.Modified()


def _matrix_signature(matrix):
    arr = np.asarray(matrix, dtype=float).reshape(4, 4)
    return tuple(np.round(arr.ravel(), 10))


@dataclass
class HistoryEntry:
    """One undoable segmentation command plus its local recomputation payload.

    ``old_bits`` and ``new_bits`` are the visible mask immediately before/after
    the command.  Threshold-sensitive strokes also retain their local screen-
    space-derived seed/growth domains.  This lets the threshold slider recompute
    only the latest applied stroke, and lets Redo safely rebase a stroke after an
    older stroke was threshold-adjusted while later strokes were undone.
    """
    extent: Tuple[int, int, int, int, int, int]
    shape_zyx: Tuple[int, int, int]
    old_bits: bytes
    new_bits: bytes
    voxel_count: int
    mode: str
    threshold: Optional[float] = None
    seed_bits: bytes = b""
    growth_bits: bytes = b""
    smart_local_threshold: bool = False
    smart_faint_recovery: bool = False
    smart_boundary_guard: bool = False
    smart_visible_seeds: bool = False
    visible_ijk_to_clip: Tuple[float, ...] = ()
    visible_depth_coeff: Tuple[float, ...] = ()
    visible_viewport_px: Tuple[float, ...] = ()

    @staticmethod
    def _pack(array):
        arr = np.ascontiguousarray(np.asarray(array, dtype=np.uint8))
        return np.packbits(arr.ravel(order="C"), bitorder="little").tobytes()

    @classmethod
    def make(
        cls,
        extent,
        old_patch,
        new_patch,
        *,
        mode,
        threshold=None,
        seed_domain=None,
        growth_domain=None,
        smart_local_threshold=False,
        smart_faint_recovery=False,
        smart_boundary_guard=False,
        smart_visible_seeds=False,
        visible_ijk_to_clip=None,
        visible_depth_coeff=None,
        visible_viewport_px=None,
    ):
        e = _extent_tuple(extent)
        old = np.ascontiguousarray(np.asarray(old_patch, dtype=np.uint8))
        new = np.ascontiguousarray(np.asarray(new_patch, dtype=np.uint8))
        expected = extent_shape_zyx(e)
        if old.shape != new.shape or tuple(old.shape) != tuple(expected):
            raise ValueError("Undo patch geometry mismatch")

        def checked_domain(value, name):
            if value is None:
                return b""
            arr = np.ascontiguousarray(np.asarray(value, dtype=np.uint8))
            if tuple(arr.shape) != tuple(expected):
                raise ValueError(f"{name} geometry mismatch")
            return cls._pack(arr)

        return cls(
            extent=e,
            shape_zyx=tuple(int(v) for v in old.shape),
            old_bits=cls._pack(old),
            new_bits=cls._pack(new),
            voxel_count=int(old.size),
            mode=str(mode).lower(),
            threshold=None if threshold is None else float(threshold),
            seed_bits=checked_domain(seed_domain, "Seed domain"),
            growth_bits=checked_domain(growth_domain, "Growth domain"),
            smart_local_threshold=bool(smart_local_threshold),
            smart_faint_recovery=bool(smart_faint_recovery),
            smart_boundary_guard=bool(smart_boundary_guard),
            smart_visible_seeds=bool(smart_visible_seeds),
            visible_ijk_to_clip=tuple(float(v) for v in (visible_ijk_to_clip or ())),
            visible_depth_coeff=tuple(float(v) for v in (visible_depth_coeff or ())),
            visible_viewport_px=tuple(float(v) for v in (visible_viewport_px or ())),
        )

    def _unpack_payload(self, payload):
        if not payload:
            return None
        raw = np.frombuffer(payload, dtype=np.uint8)
        bits = np.unpackbits(raw, bitorder="little", count=self.voxel_count)
        return bits.reshape(self.shape_zyx, order="C").astype(np.uint8, copy=False)

    def unpack(self, which):
        payload = self.old_bits if which == "old" else self.new_bits
        return self._unpack_payload(payload)

    def domain(self, which):
        if which == "seed":
            return self._unpack_payload(self.seed_bits)
        if which == "growth":
            return self._unpack_payload(self.growth_bits)
        raise ValueError(which)

    def rebase(self, old_patch, new_patch, threshold=None):
        old = np.ascontiguousarray(np.asarray(old_patch, dtype=np.uint8))
        new = np.ascontiguousarray(np.asarray(new_patch, dtype=np.uint8))
        if tuple(old.shape) != self.shape_zyx or tuple(new.shape) != self.shape_zyx:
            raise ValueError("History rebase geometry mismatch")
        self.old_bits = self._pack(old)
        self.new_bits = self._pack(new)
        if threshold is not None:
            self.threshold = float(threshold)


@dataclass
class SegmentationFrameState:
    # ``mask`` is the canonical visible segmentation. History entries retain the
    # local threshold-independent brush domains needed to revisit older strokes.
    mask: Optional[vtk.vtkImageData] = None
    mask_extent: Optional[Tuple[int, int, int, int, int, int]] = None
    undo_stack: List[HistoryEntry] = field(default_factory=list)
    redo_stack: List[HistoryEntry] = field(default_factory=list)


class SegmentationPreflightError(RuntimeError):
    """One precise authoritative-geometry failure at a segmentation boundary."""


@dataclass(frozen=True)
class SegmentationTargetSnapshot:
    """Exact target identity and geometry accepted for one segmentation step."""

    target_vc: object = field(repr=False, compare=False)
    target_item: object = field(repr=False, compare=False)
    source_image: object = field(repr=False, compare=False)
    acquisition_id: str
    channel_id: str
    backing_source_id: str
    backing_format: str
    series_identity: str
    source_checksum: Optional[str]
    frame_index: int
    extent: Tuple[int, int, int, int, int, int]
    dimensions: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    physical_units: Optional[Tuple[str, str, str]]
    origin: Tuple[float, float, float]
    direction: Tuple[Tuple[float, float, float], ...]
    local_index_affine: Tuple[Tuple[float, float, float, float], ...]
    pose: Tuple[Tuple[float, float, float, float], ...]
    world_index_affine: Tuple[Tuple[float, float, float, float], ...]
    coordinate_space_id: str
    geometry_revision: str
    acquisition_geometry_revision: str
    grid_state: str
    operation_ids: Tuple[str, ...]

    @property
    def extent_min(self) -> Tuple[int, int, int]:
        return (self.extent[0], self.extent[2], self.extent[4])

    def provenance_payload(self) -> Dict[str, Any]:
        """Stable JSON-ready operation evidence; runtime object IDs stay transient."""
        return {
            "acquisition_id": self.acquisition_id,
            "channel_id": self.channel_id,
            "backing_source_id": self.backing_source_id,
            "backing_format": self.backing_format,
            "series_identity": self.series_identity,
            "source_checksum": self.source_checksum,
            "source_frame_index": self.frame_index,
            "channel_geometry_revision": self.geometry_revision,
            "acquisition_geometry_revision": self.acquisition_geometry_revision,
            "grid_state": self.grid_state,
            "coordinate_space_id": self.coordinate_space_id,
            "physical_units": (
                list(self.physical_units)
                if self.physical_units is not None
                else None
            ),
            "runtime_extent": list(self.extent),
            "runtime_extent_min": list(self.extent_min),
            "dimensions": list(self.dimensions),
            "spacing": list(self.spacing),
            "origin": list(self.origin),
            "direction": [list(row) for row in self.direction],
            "local_index_to_working_affine": [
                list(row) for row in self.local_index_affine
            ],
            "pose": [list(row) for row in self.pose],
            "index_to_world_affine": [
                list(row) for row in self.world_index_affine
            ],
        }

    def operation_reference_payload(self) -> Dict[str, Any]:
        """Compact stable references for persisted generation records."""
        return {
            "acquisition_id": self.acquisition_id,
            "channel_id": self.channel_id,
            "backing_source_id": self.backing_source_id,
            "source_frame_index": self.frame_index,
            "channel_geometry_revision": self.geometry_revision,
            "acquisition_geometry_revision": self.acquisition_geometry_revision,
            "input_operation_ids": list(self.operation_ids),
        }


@dataclass
class SegmentationState:
    target_vc: object
    source_extent: Tuple[int, int, int, int, int, int]
    source_dimensions: Tuple[int, int, int]
    frame_states: Dict[int, SegmentationFrameState] = field(default_factory=dict)
    ui_threshold: Optional[float] = None
    session_geometry_start: Optional[Dict[str, Any]] = None
    session_geometry_current: Optional[Dict[str, Any]] = None
    session_geometry_refresh_count: int = 0

    def frame(self, index):
        index = int(index)
        if index not in self.frame_states:
            self.frame_states[index] = SegmentationFrameState()
        return self.frame_states[index]


class _JobSignals(QtCore.QObject):
    progress = QtCore.Signal(int, int, str)
    finished = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str, str)


class _SegmentationJob(QtCore.QRunnable):
    def __init__(self, job_id, function):
        super().__init__()
        self.job_id = int(job_id)
        self.function = function
        self.signals = _JobSignals()
        self.done_event = threading.Event()
        self.result = None
        self.error = None
        self.error_details = ""

    @QtCore.Slot()
    def run(self):
        try:
            def report(value, text=""):
                self.signals.progress.emit(
                    self.job_id,
                    max(0, min(100, int(value))),
                    str(text or ""),
                )
            self.result = self.function(report)
        except Exception as exc:
            self.error = str(exc)
            self.error_details = traceback.format_exc()
            self.signals.failed.emit(self.job_id, self.error, self.error_details)
        finally:
            self.done_event.set()
        if self.error is None:
            self.signals.finished.emit(self.job_id, self.result)


class VolumeSegmentationPanel(QtWidgets.QWidget):
    """Compact user-facing controls.  Controller owns all algorithmic state."""

    THRESHOLD_SLIDER_STEPS = 4096
    THRESHOLD_SLIDER_POWER = 2.0

    enabledToggled = QtCore.Signal(bool)
    modeChanged = QtCore.Signal(str)
    brushInteractionChanged = QtCore.Signal(bool)
    newSelectionRequested = QtCore.Signal()
    clearRequested = QtCore.Signal()
    seedRadiusChanged = QtCore.Signal(int)
    growthRadiusChanged = QtCore.Signal(int)
    thresholdChanged = QtCore.Signal(float)
    thresholdPreviewChanged = QtCore.Signal(float)
    liveThresholdChanged = QtCore.Signal(bool)
    maskOpacityChanged = QtCore.Signal(float)
    maskColorChanged = QtCore.Signal(object)
    undoRequested = QtCore.Signal()
    redoRequested = QtCore.Signal()
    extractRequested = QtCore.Signal()
    deleteSelectedRequested = QtCore.Signal()
    createMaskRequested = QtCore.Signal()
    smartSettingsChanged = QtCore.Signal(dict)
    extractOriginalRequested = QtCore.Signal()
    speckThresholdChanged = QtCore.Signal(int)

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        panel_policy = self.sizePolicy()
        panel_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
        self.setSizePolicy(panel_policy)
        self.setMinimumWidth(0)
        self.settings = _sanitize_settings(settings or load_settings())
        self._threshold_min = 0.0
        self._threshold_max = 1.0
        self._syncing = False
        self._speck_sync = False
        self._background_busy = False
        self._speck_busy = False
        self._last_brush_mode = "select"
        self._mode_buttons_locked = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self.quick_help = QtWidgets.QLabel(
            "Select a loaded volume, enable Segmentation, then choose a brush. "
            "Select adds threshold-passing signal, Unselect removes selected voxels, and Diffuse grows connected signal inside the Growth radius. "
            "With a brush active: left-drag paints, right-drag rotates, middle-drag pans, and Ctrl+wheel changes the active brush radius. "
            "Click the active brush again to return to normal navigation without ending the segmentation session.",
            self,
        )
        self.quick_help.setWordWrap(True)
        help_policy = self.quick_help.sizePolicy()
        help_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
        self.quick_help.setSizePolicy(help_policy)
        self.quick_help.setMinimumWidth(0)
        root.addWidget(self.quick_help)

        self.toggle = QtWidgets.QPushButton("Volume Segmentation Off", self)
        self.toggle.setCheckable(True)
        self.toggle.setToolTip("Edit a voxel selection directly over the 3-D fluorescence volume.")
        self.toggle.toggled.connect(self._on_toggle)
        root.addWidget(self.toggle)

        self.target_box = QtWidgets.QGroupBox("Target volume", self)
        target_layout = QtWidgets.QVBoxLayout(self.target_box)
        target_layout.setContentsMargins(8, 4, 8, 6)
        target_layout.setSpacing(2)
        self.target_label = QtWidgets.QLabel("None", self.target_box)
        target_font = self.target_label.font()
        target_font.setBold(True)
        target_point_size = target_font.pointSizeF()
        if target_point_size > 0.0:
            target_font.setPointSizeF(target_point_size + 1.0)
        self.target_label.setFont(target_font)
        self.target_label.setWordWrap(True)
        self.target_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.target_label.setToolTip(
            "Automatically follows the most recently selected loaded volume. "
            "Each volume keeps its own segmentation mask and history while you switch between them."
        )
        target_layout.addWidget(self.target_label)
        root.addWidget(self.target_box)

        self.brush_box = QtWidgets.QGroupBox("Brushes", self)
        brush_layout = QtWidgets.QVBoxLayout(self.brush_box)
        brush_layout.setContentsMargins(8, 6, 8, 8)
        brush_layout.setSpacing(5)
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.setExclusive(False)
        self.mode_buttons = {}
        for text, key, tip in (
            ("Select", "select", "Add threshold-passing signal under the inner brush. Click again to return to normal mouse/navigation while segmentation stays enabled."),
            ("Unselect", "unselect", "Remove selected voxels under the inner brush, independent of threshold. Click again to return to normal mouse/navigation while segmentation stays enabled."),
            ("Diffuse", "diffuse", "Grow the current selection through connected signal inside the outer brush. Click again to return to normal mouse/navigation while segmentation stays enabled."),
        ):
            button = QtWidgets.QPushButton(text, self)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.setStyleSheet(
                "QPushButton:checked { background-color: #2e7d32; color: white; "
                "font-weight: 600; border: 1px solid #4caf50; }"
            )
            self.mode_group.addButton(button)
            self.mode_buttons[key] = button
            button.clicked.connect(
                lambda checked=False, k=key: self._brush_mode_clicked(k, checked)
            )
            mode_layout.addWidget(button, 1)
        brush_layout.addLayout(mode_layout)

        self.seed_slider, self.seed_spin = self._integer_control(2, 260, self.settings["seed_radius"])
        self.seed_slider.setToolTip("Area used to mark signal as the starting selection.")
        self.seed_spin.setToolTip(self.seed_slider.toolTip())
        self.seed_slider.valueChanged.connect(self._seed_slider_changed)
        self.seed_spin.valueChanged.connect(self._seed_spin_changed)
        brush_layout.addWidget(
            self._labeled_row("Seed radius", self.seed_slider, self.seed_spin, "px")
        )

        self.growth_slider, self.growth_spin = self._integer_control(5, 1600, self.settings["growth_radius"])
        self.growth_slider.setToolTip("Maximum screen-space region in which Diffuse may grow connected signal.")
        self.growth_spin.setToolTip(self.growth_slider.toolTip())
        self.growth_slider.valueChanged.connect(self._growth_slider_changed)
        self.growth_spin.valueChanged.connect(self._growth_spin_changed)
        brush_layout.addWidget(
            self._labeled_row("Growth radius", self.growth_slider, self.growth_spin, "px")
        )
        root.addWidget(self.brush_box)

        selection_row = QtWidgets.QHBoxLayout()
        self.new_selection = QtWidgets.QPushButton("New Selection", self)
        self.new_selection.setToolTip("Clear the current mask as one undoable edit and switch to Select.")
        self.new_selection.clicked.connect(self.newSelectionRequested)
        selection_row.addWidget(self.new_selection)
        self.clear = QtWidgets.QPushButton("Clear", self)
        self.clear.setToolTip("Clear the current frame's segmentation mask. This can be undone.")
        self.clear.clicked.connect(self.clearRequested)
        selection_row.addWidget(self.clear)
        root.addLayout(selection_row)

        threshold_frame = QtWidgets.QFrame(self)
        threshold_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        threshold_layout = QtWidgets.QVBoxLayout(threshold_frame)
        threshold_layout.setContentsMargins(6, 6, 6, 6)
        threshold_layout.setSpacing(4)

        self.threshold_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        self.threshold_slider.setRange(0, self.THRESHOLD_SLIDER_STEPS)
        self.threshold_slider.setSingleStep(1)
        self.threshold_slider.setPageStep(max(1, self.THRESHOLD_SLIDER_STEPS // 64))
        # Recomputing a smart 3-D stroke can involve connected growth, visible-
        # seed depth filtering, a boundary-gradient pass, history patching, and a
        # GPU mask refresh. QSlider's default tracking emits valueChanged for
        # every handle movement, which can queue many expensive exact updates and
        # make dragging effectively unusable. Keep the handle/numeric readout
        # live via sliderMoved, but commit the exact segmentation only on release.
        self.threshold_slider.setTracking(False)
        self.threshold_spin = QtWidgets.QDoubleSpinBox(self)
        self.threshold_spin.setDecimals(6)
        self.threshold_spin.setKeyboardTracking(False)
        self.threshold_spin.setMinimumWidth(100)
        threshold_tip = (
            "Signal below this intensity is treated as background for Select and Diffuse. "
            "The slider gives extra travel to low intensities so faint-signal thresholds "
            "are easier to tune precisely. "
            "Changing it recalculates only the latest applied Select/Diffuse stroke. "
            "By default, dragging updates the displayed number immediately and recalculates the 3-D "
            "segmentation only once when you release the handle. Enable Live threshold preview below "
            "for throttled cached feedback while dragging; release still performs one exact update. "
            "The value box applies an exact update when its edit is committed. "
            "Undo to revisit an earlier stroke, adjust the threshold, then Redo later strokes."
        )
        self.threshold_slider.setToolTip(threshold_tip)
        self.threshold_spin.setToolTip(threshold_tip)
        # sliderMoved is lightweight UI feedback only. With tracking disabled,
        # valueChanged is emitted when the user releases the handle (or performs
        # a discrete keyboard/groove change), so the controller receives one exact
        # threshold update instead of a backlog of expensive 3-D recomputations.
        self.threshold_slider.sliderMoved.connect(self._threshold_slider_preview)
        self.threshold_slider.valueChanged.connect(self._threshold_slider_changed)
        # Always request one exact commit on release, even if the user drags away
        # and returns to the original value (a case where valueChanged may not fire).
        self.threshold_slider.sliderReleased.connect(self._threshold_slider_released)
        self.threshold_spin.valueChanged.connect(self._threshold_spin_changed)
        # Give the disabled/no-target panel a meaningful non-zero fluorescence
        # floor as well; selecting a volume immediately replaces this with a
        # target-range-aware value.
        self.configure_threshold(0.0, 1.0, 0.05)
        threshold_layout.addWidget(
            self._labeled_row("Signal threshold", self.threshold_slider, self.threshold_spin, "")
        )

        self.live_threshold_preview = QtWidgets.QCheckBox("Live threshold", self)
        self.live_threshold_preview.setChecked(self.settings["live_threshold_preview"])
        self.live_threshold_preview.setToolTip(
            "Recalculate the latest Select or Diffuse stroke exactly while the threshold slider moves."
        )
        self.live_threshold_preview.toggled.connect(self._live_threshold_toggled)
        threshold_layout.addWidget(self.live_threshold_preview)
        root.addWidget(threshold_frame)

        smart_box = QtWidgets.QGroupBox("Smart helpers", self)
        self.smart_box = smart_box
        smart_box.setToolTip(
            "Optional assistants layered onto the normal brushes. They can be enabled independently and "
            "combined: Local threshold adapts signal detection, Visible seeds limits where a new stroke "
            "starts in depth, Recover faint signal relaxes connected Diffuse growth, and Boundary guard "
            "resists crossing strong edges. Unselect deliberately ignores Smart helpers so removal remains "
            "literal and predictable. Recover faint signal and Boundary guard are enabled by default."
        )
        smart_layout = QtWidgets.QVBoxLayout(smart_box)
        smart_layout.setContentsMargins(8, 6, 8, 6)
        smart_layout.setSpacing(3)

        self.smart_local_threshold = QtWidgets.QCheckBox("Local threshold", self)
        self.smart_local_threshold.setChecked(self.settings["smart_local_threshold"])
        self.smart_local_threshold.setToolTip(
            "Select and Diffuse. Estimate a signal/background cutoff from the 3-D region covered by "
            "this stroke instead of assuming that one global threshold is appropriate everywhere. "
            "This is useful when fluorescence intensity changes with depth or across the specimen. "
            "The estimated value is shown in Signal threshold after the stroke, so you can refine it "
            "manually; changing that control still recalculates only the latest threshold-sensitive stroke."
        )

        self.smart_faint_recovery = QtWidgets.QCheckBox("Recover faint signal", self)
        self.smart_faint_recovery.setChecked(self.settings["smart_faint_recovery"])
        self.smart_faint_recovery.setToolTip(
            "Diffuse only. Keep the displayed Signal threshold (or Local threshold result) strict for "
            "finding reliable seed signal, but let connectivity continue through somewhat dimmer voxels "
            "inside the Growth radius. This can recover weak neurites that remain connected to a bright "
            "seed. It does not make dim voxels into independent seeds, which limits uncontrolled flooding."
        )
        smart_layout.addWidget(self.smart_faint_recovery)

        self.smart_boundary_guard = QtWidgets.QCheckBox("Boundary guard", self)
        self.smart_boundary_guard.setChecked(self.settings["smart_boundary_guard"])
        self.smart_boundary_guard.setToolTip(
            "Diffuse only. Measure the local 3-D intensity gradient and prevent growth through the "
            "strongest boundaries inside the painted Growth region. This can reduce leakage from one "
            "touching structure into another. The user-painted seed is always preserved, so a strong "
            "edge at the starting point cannot erase the seed. Boundary Guard can be combined with "
            "Recover faint signal: one relaxes intensity along a connected path while the other resists "
            "crossing sharp boundaries."
        )
        smart_layout.addWidget(self.smart_boundary_guard)

        self.smart_visible_seeds = QtWidgets.QCheckBox("Visible seeds only", self)
        self.smart_visible_seeds.setChecked(self.settings["smart_visible_seeds"])
        self.smart_visible_seeds.setToolTip(
            "Select and Diffuse. When several threshold-passing structures lie behind one another under "
            "the brush, use only the front-most signal layer as new seed material from the current camera "
            "view. This reduces accidental seeding of bright structures hidden behind the one you are "
            "painting. For Diffuse, only the starting seed is visibility-filtered: after a visible seed is "
            "chosen, connected growth may still continue behind other structures inside the Growth radius. "
            "Existing selected voxels can also remain valid Diffuse starting points. In Select mode this "
            "deliberately favors the front-facing signal layer; turn it off when you intentionally want to "
            "select every threshold-passing structure through the painted screen area. Built-in stereo is "
            "treated as one viewing direction; the left and right eyes are not considered separate seed views."
        )
        smart_layout.addWidget(self.smart_visible_seeds)
        smart_layout.addWidget(self.smart_local_threshold)

        for checkbox in (
            self.smart_faint_recovery,
            self.smart_boundary_guard,
            self.smart_visible_seeds,
            self.smart_local_threshold,
        ):
            checkbox.toggled.connect(self._smart_settings_changed)
        root.addWidget(smart_box)

        self.opacity_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(round(100.0 * self.settings["mask_opacity"])))
        self.opacity_spin = QtWidgets.QDoubleSpinBox(self)
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(self.settings["mask_opacity"])
        self.opacity_slider.valueChanged.connect(self._opacity_slider_changed)
        self.opacity_spin.valueChanged.connect(self._opacity_spin_changed)
        root.addWidget(self._labeled_row("Mask opacity", self.opacity_slider, self.opacity_spin, ""))

        color_row = QtWidgets.QHBoxLayout()
        color_row.addWidget(QtWidgets.QLabel("Mask color"), 0)
        self.color_button = QtWidgets.QPushButton("Choose…", self)
        self.color_button.clicked.connect(self._choose_color)
        color_row.addWidget(self.color_button, 1)
        root.addLayout(color_row)
        self._update_color_button()

        history_row = QtWidgets.QHBoxLayout()
        self.undo = QtWidgets.QPushButton("Undo", self)
        self.redo = QtWidgets.QPushButton("Redo", self)
        self.undo.clicked.connect(self.undoRequested)
        self.redo.clicked.connect(self.redoRequested)
        history_row.addWidget(self.undo)
        history_row.addWidget(self.redo)
        root.addLayout(history_row)

        output_box = QtWidgets.QGroupBox("Output", self)
        output_layout = QtWidgets.QGridLayout(output_box)
        self.extract = QtWidgets.QPushButton("Extract Selected Voxels", self)
        self.extract.setToolTip("Create a normal MADI3D intensity volume containing the selected voxels of the current frame.")
        self.extract.clicked.connect(self.extractRequested)
        output_layout.addWidget(self.extract, 0, 0)
        self.delete_selected = QtWidgets.QPushButton("Delete Selected", self)
        self.delete_selected.setToolTip("Create a normal MADI3D intensity volume with the currently selected voxels removed.")
        self.delete_selected.clicked.connect(self.deleteSelectedRequested)
        output_layout.addWidget(self.delete_selected, 0, 1)
        self.extract_original = QtWidgets.QPushButton("Extract Original Signal", self)
        self.extract_original.setToolTip(
            "Copy original source intensities under the segmentation support plus the 3-D mask margin."
        )
        self.extract_original.clicked.connect(self.extractOriginalRequested)
        output_layout.addWidget(self.extract_original, 1, 0)
        self.extract_margin = QtWidgets.QSpinBox(self)
        self.extract_margin.setRange(0, 20)
        self.extract_margin.setValue(1)
        self.extract_margin.setPrefix("Margin ")
        self.extract_margin.setSuffix(" vox")
        self.extract_margin.setToolTip("3-D voxel margin used by Extract Original Signal.")
        output_layout.addWidget(self.extract_margin, 1, 1)
        self.create_mask = QtWidgets.QPushButton("Create Mask Volume", self)
        self.create_mask.setToolTip("Create a normal MADI3D binary volume from the current frame's live segmentation mask.")
        self.create_mask.clicked.connect(self.createMaskRequested)
        output_layout.addWidget(self.create_mask, 2, 0, 1, 2)
        root.addWidget(output_box)

        cleanup = QtWidgets.QGroupBox("Remove specks", self)
        cleanup.setToolTip(
            "Remove connected components smaller than this voxel count from the whole current selection. "
            "Changes apply automatically. Repeated adjustments use the same pre-cleanup selection, so lowering the value can restore components removed by a higher setting until another segmentation edit is made."
        )
        cleanup_layout = QtWidgets.QHBoxLayout(cleanup)
        cleanup_layout.setContentsMargins(8, 6, 8, 6)
        self.speck_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, cleanup)
        self.speck_slider.setRange(1, 5000)
        self.speck_slider.setValue(20)
        self.speck_slider.setToolTip(cleanup.toolTip())
        self.speck_spin = QtWidgets.QSpinBox(cleanup)
        self.speck_spin.setRange(1, 1_000_000)
        self.speck_spin.setValue(20)
        self.speck_spin.setSuffix(" vox")
        self.speck_spin.setKeyboardTracking(False)
        self.speck_spin.setToolTip(cleanup.toolTip())
        cleanup_layout.addWidget(self.speck_slider, 1)
        cleanup_layout.addWidget(self.speck_spin, 0)
        root.addWidget(cleanup)

        # Keep one progress-row height permanently so background work never
        # moves the panel contents under the pointer. Hide only the progress bar
        # while idle so collapsed-panel accessibility does not report an active
        # operation; the fixed-height slot remains in the layout.
        self.progress_slot = QtWidgets.QWidget(self)
        progress_slot_layout = QtWidgets.QVBoxLayout(self.progress_slot)
        progress_slot_layout.setContentsMargins(0, 0, 0, 0)
        progress_slot_layout.setSpacing(0)
        self.work_progress = QtWidgets.QProgressBar(self.progress_slot)
        self.work_progress.setRange(0, 100)
        self.work_progress.setValue(0)
        self.work_progress.setFormat("")
        self.work_progress.setTextVisible(False)
        self.work_progress.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.work_progress.setMinimumWidth(0)
        progress_slot_layout.addWidget(self.work_progress)
        self.progress_slot.setFixedHeight(
            max(1, self.work_progress.sizeHint().height())
        )
        self.progress_slot.setMinimumWidth(0)
        self.work_progress.hide()
        root.addWidget(self.progress_slot)

        self.status = QtWidgets.QLabel("Select a loaded volume, then enable segmentation.", self)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.status)

        # Text-heavy controls may shrink below their natural text width. Keep
        # numeric fields and sliders out of this list so their values remain
        # readable while the containing panel follows a narrow dock width.
        compressible_text_controls = (
            self.quick_help,
            self.toggle,
            self.target_label,
            *self.mode_buttons.values(),
            self.new_selection,
            self.clear,
            self.live_threshold_preview,
            self.smart_local_threshold,
            self.smart_faint_recovery,
            self.smart_boundary_guard,
            self.smart_visible_seeds,
            self.color_button,
            self.undo,
            self.redo,
            self.extract,
            self.delete_selected,
            self.extract_original,
            self.create_mask,
            self.status,
        )
        for widget in compressible_text_controls:
            widget.setMinimumWidth(0)
            policy = widget.sizePolicy()
            policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Ignored)
            widget.setSizePolicy(policy)

        root.addStretch(1)

        self.speck_slider.valueChanged.connect(self._speck_slider_changed)
        self.speck_spin.valueChanged.connect(self._speck_spin_changed)
        self.set_controls_active(False)
        self.set_history_available(False, False)

    def _speck_slider_changed(self, value):
        if self._speck_sync:
            return
        self._speck_sync = True
        try:
            blocker = QtCore.QSignalBlocker(self.speck_spin)
            self.speck_spin.setValue(int(value))
            del blocker
        finally:
            self._speck_sync = False
        self.speckThresholdChanged.emit(int(value))

    def _speck_spin_changed(self, value):
        if self._speck_sync:
            return
        self._speck_sync = True
        try:
            blocker = QtCore.QSignalBlocker(self.speck_slider)
            self.speck_slider.setValue(min(5000, int(value)))
            del blocker
        finally:
            self._speck_sync = False
        self.speckThresholdChanged.emit(int(value))

    def set_background_busy(self, busy, label="Working", preserve_threshold=False):
        busy = bool(busy)
        self._background_busy = busy
        if busy:
            focus_widget = QtWidgets.QApplication.focusWidget()
            if (
                focus_widget is not None
                and (focus_widget is self or self.isAncestorOf(focus_widget))
            ):
                focus_widget.clearFocus()
        self.toggle.setEnabled(not busy and not self._speck_busy)
        if busy:
            self.set_controls_active(False, preserve_threshold=preserve_threshold)
            self.work_progress.setTextVisible(True)
            self.work_progress.setValue(0)
            self.work_progress.setFormat(f"{label} — %p%")
            self.work_progress.show()
        else:
            self.work_progress.setValue(0)
            self.work_progress.setFormat("")
            self.work_progress.setTextVisible(False)
            self.work_progress.hide()
            self.toggle.setEnabled(not self._speck_busy)
            if not self._speck_busy:
                self.set_controls_active(self.toggle.isChecked())
        self.set_mode_buttons_locked(self._background_busy or self._speck_busy)

    def set_progress(self, value, label=""):
        self.work_progress.setValue(max(0, min(100, int(value))))
        if label:
            self.work_progress.setFormat(f"{label} — %p%")

    def set_speck_busy(self, busy):
        self._speck_busy = bool(busy)
        self.toggle.setEnabled(not self._background_busy and not self._speck_busy)
        if self._speck_busy:
            self.set_controls_active(False, preserve_specks=True)
        elif not self._background_busy:
            self.set_controls_active(self.toggle.isChecked())
        self.set_mode_buttons_locked(self._background_busy or self._speck_busy)

    def _integer_control(self, lo, hi, value):
        slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        slider.setRange(int(lo), int(hi))
        slider.setValue(int(value))
        spin = QtWidgets.QSpinBox(self)
        spin.setRange(int(lo), int(hi))
        spin.setValue(int(value))
        spin.setSuffix(" px")
        spin.setMinimumWidth(88)
        return slider, spin

    def _labeled_row(self, label, slider, value_widget, suffix):
        holder = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(QtWidgets.QLabel(label, holder))
        layout.addWidget(slider)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(
            value_widget,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        if suffix and not isinstance(value_widget, QtWidgets.QSpinBox):
            row.addWidget(QtWidgets.QLabel(suffix, holder), 0)
        layout.addLayout(row)
        return holder

    def _on_toggle(self, checked):
        checked = bool(checked)
        self._on_toggle_visual(checked)
        self._sync_mode_checks(checked)
        self.enabledToggled.emit(checked)

    def set_enabled_checked(self, enabled):
        enabled = bool(enabled)
        blocker = QtCore.QSignalBlocker(self.toggle)
        self.toggle.setChecked(enabled)
        del blocker
        self._on_toggle_visual(enabled)
        self._sync_mode_checks(enabled)

    def _on_toggle_visual(self, checked):
        self.toggle.setText("Volume Segmentation On" if checked else "Volume Segmentation Off")
        if checked:
            self.toggle.setStyleSheet(
                "QPushButton { background-color: #2e7d32; color: white; font-weight: 600; }"
                "QPushButton:hover { background-color: #388e3c; }"
            )
        else:
            self.toggle.setStyleSheet("")

    def set_controls_active(self, active, *, preserve_threshold=False, preserve_specks=False):
        active = bool(active)
        preserved = set()
        if preserve_threshold:
            preserved.update((
                self.threshold_slider,
                self.threshold_spin,
                self.live_threshold_preview,
            ))
        if preserve_specks:
            preserved.update((self.speck_slider, self.speck_spin))
        for widget in (
            self.new_selection, self.clear, self.seed_slider, self.seed_spin,
            self.growth_slider, self.growth_spin,
            self.threshold_slider, self.threshold_spin,
            self.smart_local_threshold, self.smart_faint_recovery, self.smart_boundary_guard,
            self.smart_visible_seeds, self.live_threshold_preview, self.opacity_slider, self.opacity_spin, self.color_button,
            self.undo, self.redo, self.extract, self.delete_selected, self.create_mask,
            self.extract_original, self.extract_margin,
            self.speck_slider, self.speck_spin,
        ):
            if not active and widget in preserved:
                widget.setEnabled(True)
                continue
            widget.setEnabled(active)
        self._update_mode_button_enabled()

    def set_target_name(self, name):
        self.target_label.setText(str(name or "None"))
        self._update_mode_button_enabled()

    def set_status(self, text):
        self.status.setText(str(text or ""))

    def set_mode(self, mode):
        mode = str(mode or "").lower()
        if mode not in self.mode_buttons:
            return
        self._last_brush_mode = mode
        if self.toggle.isChecked():
            self._set_only_mode_checked(mode)

    def set_mode_buttons_locked(self, locked):
        self._mode_buttons_locked = bool(locked)
        self._update_mode_button_enabled()

    def _has_target(self):
        text = str(self.target_label.text()).strip()
        return bool(text and text.lower() != "none")

    def _update_mode_button_enabled(self):
        enabled = self._has_target() and not self._mode_buttons_locked
        for button in self.mode_buttons.values():
            button.setEnabled(enabled)

    def _set_only_mode_checked(self, mode):
        blockers = [
            QtCore.QSignalBlocker(button)
            for button in self.mode_buttons.values()
        ]
        try:
            for key, button in self.mode_buttons.items():
                button.setChecked(key == mode)
        finally:
            del blockers

    def _sync_mode_checks(self, enabled):
        if enabled:
            self._set_only_mode_checked(self._last_brush_mode)
        else:
            self._set_only_mode_checked(None)

    def _brush_mode_clicked(self, mode, checked):
        mode = str(mode).lower()
        if checked:
            self._last_brush_mode = mode
            self._set_only_mode_checked(mode)
            self.modeChanged.emit(mode)
            if not self.toggle.isChecked():
                self.toggle.setChecked(True)
            else:
                self.brushInteractionChanged.emit(True)
        elif not any(
            button.isChecked() for button in self.mode_buttons.values()
        ):
            self.brushInteractionChanged.emit(False)

    def configure_threshold(self, minimum, maximum, value):
        lo = float(minimum)
        hi = float(maximum)
        if not math.isfinite(lo): lo = 0.0
        if not math.isfinite(hi): hi = lo + 1.0
        if hi <= lo: hi = lo + 1.0
        value = max(lo, min(hi, float(value)))
        self._threshold_min = lo
        self._threshold_max = hi
        blockers = [QtCore.QSignalBlocker(self.threshold_slider), QtCore.QSignalBlocker(self.threshold_spin)]
        try:
            decimals = 6 if max(abs(lo), abs(hi)) < 1e6 else 3
            self.threshold_spin.setDecimals(decimals)
            self.threshold_spin.setRange(lo, hi)
            step = max((hi-lo)/float(self.THRESHOLD_SLIDER_STEPS), 1e-9)
            self.threshold_spin.setSingleStep(step)
            self.threshold_spin.setValue(value)
            self.threshold_slider.setValue(self._threshold_to_slider(value))
        finally:
            del blockers

    def _threshold_to_slider(self, value):
        lo, hi = self._threshold_min, self._threshold_max
        fraction = (float(value) - lo) / max(1e-12, hi - lo)
        fraction = max(0.0, min(1.0, fraction))
        position = fraction ** (1.0 / float(self.THRESHOLD_SLIDER_POWER))
        return int(round(float(self.THRESHOLD_SLIDER_STEPS) * position))

    def _slider_to_threshold(self, ivalue):
        fraction = float(ivalue) / float(self.THRESHOLD_SLIDER_STEPS)
        fraction = max(0.0, min(1.0, fraction))
        scaled = fraction ** float(self.THRESHOLD_SLIDER_POWER)
        return (
            self._threshold_min
            + (self._threshold_max - self._threshold_min) * scaled
        )

    def _threshold_slider_preview(self, value):
        """Mirror a dragged threshold into the value box without recomputing 3-D data."""
        if self._syncing:
            return
        self._syncing = True
        try:
            physical = self._slider_to_threshold(value)
            blocker = QtCore.QSignalBlocker(self.threshold_spin)
            self.threshold_spin.setValue(physical)
            del blocker
        finally:
            self._syncing = False
        if self.live_threshold_preview.isChecked():
            self.thresholdPreviewChanged.emit(float(physical))

    def _threshold_slider_changed(self, value):
        if self._syncing: return
        self._syncing = True
        try:
            physical = self._slider_to_threshold(value)
            blocker = QtCore.QSignalBlocker(self.threshold_spin)
            self.threshold_spin.setValue(physical)
            del blocker
        finally:
            self._syncing = False
        self.thresholdChanged.emit(float(physical))

    def _threshold_slider_released(self):
        if self._syncing:
            return
        physical = self._slider_to_threshold(self.threshold_slider.value())
        blocker = QtCore.QSignalBlocker(self.threshold_spin)
        try:
            self.threshold_spin.setValue(physical)
        finally:
            del blocker
        self.thresholdChanged.emit(float(physical))

    def _threshold_spin_changed(self, value):
        if self._syncing: return
        self._syncing = True
        try:
            blocker = QtCore.QSignalBlocker(self.threshold_slider)
            self.threshold_slider.setValue(self._threshold_to_slider(value))
            del blocker
        finally:
            self._syncing = False
        self.thresholdChanged.emit(float(value))

    def _seed_slider_changed(self, value):
        if self._syncing: return
        self._syncing = True
        try:
            blocker = QtCore.QSignalBlocker(self.seed_spin)
            self.seed_spin.setValue(int(value))
            del blocker
            if value > self.growth_spin.value():
                self.growth_spin.setValue(int(value))
        finally:
            self._syncing = False
        self.seedRadiusChanged.emit(int(value))

    def _seed_spin_changed(self, value):
        if self._syncing: return
        blocker = QtCore.QSignalBlocker(self.seed_slider)
        self.seed_slider.setValue(int(value))
        del blocker
        if value > self.growth_spin.value():
            self.growth_spin.setValue(int(value))
        self.seedRadiusChanged.emit(int(value))

    def _growth_slider_changed(self, value):
        value = max(int(value), self.seed_spin.value())
        blocker = QtCore.QSignalBlocker(self.growth_spin)
        self.growth_spin.setValue(value)
        del blocker
        if self.growth_slider.value() != value:
            blocker = QtCore.QSignalBlocker(self.growth_slider)
            self.growth_slider.setValue(value)
            blocker.unblock()
        self.growthRadiusChanged.emit(value)

    def _growth_spin_changed(self, value):
        value = max(int(value), self.seed_spin.value())
        blocker = QtCore.QSignalBlocker(self.growth_slider)
        self.growth_slider.setValue(value)
        del blocker
        if self.growth_spin.value() != value:
            blocker = QtCore.QSignalBlocker(self.growth_spin)
            self.growth_spin.setValue(value)
            blocker.unblock()
        self.growthRadiusChanged.emit(value)

    def _live_threshold_toggled(self, checked):
        checked = bool(checked)
        self.settings["live_threshold_preview"] = checked
        self.liveThresholdChanged.emit(checked)

    def _smart_settings_changed(self, *_):
        payload = {
            "smart_local_threshold": self.smart_local_threshold.isChecked(),
            "smart_faint_recovery": self.smart_faint_recovery.isChecked(),
            "smart_boundary_guard": self.smart_boundary_guard.isChecked(),
            "smart_visible_seeds": self.smart_visible_seeds.isChecked(),
        }
        self.settings.update(payload)
        self.smartSettingsChanged.emit(dict(payload))

    def _opacity_slider_changed(self, value):
        opacity = float(value) / 100.0
        blocker = QtCore.QSignalBlocker(self.opacity_spin)
        self.opacity_spin.setValue(opacity)
        del blocker
        self.maskOpacityChanged.emit(opacity)

    def _opacity_spin_changed(self, value):
        blocker = QtCore.QSignalBlocker(self.opacity_slider)
        self.opacity_slider.setValue(int(round(float(value)*100.0)))
        del blocker
        self.maskOpacityChanged.emit(float(value))

    def _update_color_button(self):
        r, g, b = (int(round(255.0*_clamp(v, 0, 1))) for v in self.settings["mask_color"])
        self.color_button.setStyleSheet(f"QPushButton {{ background-color: rgb({r},{g},{b}); }}")
        self.color_button.setText(f"{r}, {g}, {b}")

    def _choose_color(self):
        r, g, b = self.settings["mask_color"]
        current = QtGui.QColor.fromRgbF(r, g, b)
        chosen = QtWidgets.QColorDialog.getColor(current, self, "Segmentation mask color")
        if not chosen.isValid():
            return
        self.settings["mask_color"] = [chosen.redF(), chosen.greenF(), chosen.blueF()]
        self._update_color_button()
        self.maskColorChanged.emit(list(self.settings["mask_color"]))

    def set_history_available(self, can_undo, can_redo):
        active = self.toggle.isChecked()
        self.undo.setEnabled(bool(active and can_undo))
        self.redo.setEnabled(bool(active and can_redo))

    def settings_payload(self):
        return _sanitize_settings({
            "seed_radius": self.seed_spin.value(),
            "growth_radius": self.growth_spin.value(),
            "mask_color": self.settings["mask_color"],
            "mask_opacity": self.opacity_spin.value(),
            "history_limit": self.settings.get("history_limit", 40),
            "smart_defaults_initialized": True,
            "smart_local_threshold": self.smart_local_threshold.isChecked(),
            "smart_faint_recovery": self.smart_faint_recovery.isChecked(),
            "smart_boundary_guard": self.smart_boundary_guard.isChecked(),
            "smart_visible_seeds": self.smart_visible_seeds.isChecked(),
            "live_threshold_preview": self.live_threshold_preview.isChecked(),
        })


class SegmentationBrushOverlay:
    """Layer-1 screen-space rings and freehand stroke."""

    IDLE_INNER_COLOR = (1.0, 1.0, 0.0)
    IDLE_OUTER_COLOR = (1.0, 0.9, 0.2)
    BUSY_COLORS = ((0.10, 0.42, 1.0), (0.25, 0.72, 1.0))
    DONE_COLOR = (0.20, 0.95, 0.35)

    def __init__(self, overlay_renderer):
        self.renderer = overlay_renderer
        self.inner_actor, self.inner_poly = self._make_line_actor(self.IDLE_INNER_COLOR, 2.0, 1.0)
        self.outer_actor, self.outer_poly = self._make_line_actor(self.IDLE_OUTER_COLOR, 1.5, 0.55)
        self.stroke_actor, self.stroke_poly = self._make_line_actor((1.0, 1.0, 0.2), 1.5, 0.55)
        for actor in (self.inner_actor, self.outer_actor, self.stroke_actor):
            self.renderer.AddViewProp(actor)
            actor.SetVisibility(False)
        self.seed_radius = 15
        self.growth_radius = 35
        self.center = (0.0, 0.0)
        self._visible = False
        self._growth_visible = False

    def _make_line_actor(self, color, width, opacity):
        poly = vtk.vtkPolyData()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(poly)
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()
        mapper.SetTransformCoordinate(coordinate)
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(float(width))
        actor.GetProperty().SetOpacity(float(opacity))
        return actor, poly

    def set_cursor_colors(self, inner_color, outer_color=None):
        inner = tuple(float(v) for v in inner_color)
        outer = inner if outer_color is None else tuple(float(v) for v in outer_color)
        self.inner_actor.GetProperty().SetColor(*inner)
        self.outer_actor.GetProperty().SetColor(*outer)
        self.inner_actor.Modified()
        self.outer_actor.Modified()

    def reset_cursor_colors(self):
        self.set_cursor_colors(
            self.IDLE_INNER_COLOR, self.IDLE_OUTER_COLOR
        )

    def set_visible(self, visible):
        visible = bool(visible)
        self._visible = visible
        self.inner_actor.SetVisibility(visible)
        self.outer_actor.SetVisibility(visible and self._growth_visible)
        if not visible:
            self.stroke_actor.SetVisibility(False)

    def set_growth_visible(self, visible):
        """Show the outer growth ring only for Diffuse mode."""
        self._growth_visible = bool(visible)
        self.outer_actor.SetVisibility(self._visible and self._growth_visible)

    def set_radii(self, seed, growth):
        self.seed_radius = int(seed)
        self.growth_radius = max(int(seed), int(growth))
        self.set_center(*self.center)

    def set_center(self, x, y):
        self.center = (float(x), float(y))
        self._set_circle(self.inner_poly, self.center, self.seed_radius)
        self._set_circle(self.outer_poly, self.center, self.growth_radius)

    def set_stroke(self, points):
        pts = [(float(x), float(y)) for x, y in (points or [])]
        if len(pts) < 2:
            self.stroke_actor.SetVisibility(False)
            return
        vtk_points = vtk.vtkPoints()
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(pts))
        for i, (x, y) in enumerate(pts):
            vtk_points.InsertNextPoint(x, y, 0.0)
            line.GetPointIds().SetId(i, i)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)
        self.stroke_poly.SetPoints(vtk_points)
        self.stroke_poly.SetLines(cells)
        self.stroke_poly.Modified()
        self.stroke_actor.SetVisibility(True)

    def clear_stroke(self):
        self.stroke_actor.SetVisibility(False)
        self.stroke_poly.Initialize()
        self.stroke_poly.Modified()

    @staticmethod
    def _set_circle(poly, center, radius, segments=64):
        cx, cy = center
        n = max(16, int(segments))
        points = vtk.vtkPoints()
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(n + 1)
        for i in range(n):
            a = 2.0 * math.pi * i / n
            points.InsertNextPoint(cx + radius*math.cos(a), cy + radius*math.sin(a), 0.0)
            line.GetPointIds().SetId(i, i)
        line.GetPointIds().SetId(n, 0)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)
        poly.SetPoints(points)
        poly.SetLines(cells)
        poly.Modified()


class VolumeSegmentationInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Left drag records a screen stroke; right drag retains trackball rotation."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.left_dragging = False
        self.right_camera_dragging = False
        self.middle_camera_dragging = False
        self.stroke = []
        self.stroke_frame = None
        self.camera_signature = None

        self.AddObserver("LeftButtonPressEvent", self.on_left_button_press)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_button_release)
        self.AddObserver("RightButtonPressEvent", self.on_right_button_press)
        self.AddObserver("RightButtonReleaseEvent", self.on_right_button_release)
        self.AddObserver("MiddleButtonPressEvent", self.on_middle_button_press)
        self.AddObserver("MiddleButtonReleaseEvent", self.on_middle_button_release)
        self.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self.AddObserver("MouseWheelForwardEvent", self.on_mouse_wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent", self.on_mouse_wheel_backward)

    def _event_position(self):
        pos = self.GetInteractor().GetEventPosition()
        return (int(pos[0]), int(pos[1]))

    def _camera_sig(self):
        return self.controller.camera_signature()

    def on_left_button_press(self, _obj=None, _event=None):
        if not self.controller.brush_interaction_active:
            return vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonDown(self)
        self.left_dragging = True
        self.right_camera_dragging = False
        self.stroke = [self._event_position()]
        self.stroke_frame = self.controller.current_frame_index()
        self.camera_signature = self._camera_sig()
        self.controller.overlay.set_stroke(self.stroke)

    def on_left_button_release(self, _obj=None, _event=None):
        if not self.controller.brush_interaction_active or not self.left_dragging:
            return vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonUp(self)
        self.left_dragging = False
        stroke = list(self.stroke)
        self.stroke = []
        self.controller.overlay.clear_stroke()
        if self.stroke_frame != self.controller.current_frame_index():
            self.controller.set_status("Stroke cancelled because the time frame changed.")
            return
        if self.camera_signature != self._camera_sig():
            self.controller.set_status("Stroke cancelled because the camera changed during the drag.")
            return
        self.controller.process_stroke(stroke, self.stroke_frame)

    def on_right_button_press(self, _obj=None, _event=None):
        if self.controller.brush_interaction_active:
            if self.middle_camera_dragging:
                vtk.vtkInteractorStyleTrackballCamera.OnMiddleButtonUp(self)
            self.cancel_stroke()
            self.right_camera_dragging = True
            # Match the established MADI mesh-brush convention: right drag acts
            # like TrackballCamera's normal left-button rotation.
            return vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonDown(self)
        return vtk.vtkInteractorStyleTrackballCamera.OnRightButtonDown(self)

    def on_right_button_release(self, _obj=None, _event=None):
        if self.controller.brush_interaction_active and self.right_camera_dragging:
            self.right_camera_dragging = False
            return vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonUp(self)
        return vtk.vtkInteractorStyleTrackballCamera.OnRightButtonUp(self)

    def on_middle_button_press(self, _obj=None, _event=None):
        if self.controller.brush_interaction_active:
            if self.right_camera_dragging:
                vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonUp(self)
            self.cancel_stroke()
            self.middle_camera_dragging = True
        return vtk.vtkInteractorStyleTrackballCamera.OnMiddleButtonDown(self)

    def on_middle_button_release(self, _obj=None, _event=None):
        if self.controller.brush_interaction_active:
            self.middle_camera_dragging = False
        return vtk.vtkInteractorStyleTrackballCamera.OnMiddleButtonUp(self)

    def on_mouse_move(self, _obj=None, _event=None):
        if not self.controller.brush_interaction_active:
            return vtk.vtkInteractorStyleTrackballCamera.OnMouseMove(self)
        x, y = self._event_position()
        self.controller.overlay.set_center(x, y)
        if self.right_camera_dragging or self.middle_camera_dragging:
            self.controller.overlay.clear_stroke()
            # TrackballCamera performs its own render while rotating/panning.
            # Avoid issuing a second full render from the segmentation layer.
            vtk.vtkInteractorStyleTrackballCamera.OnMouseMove(self)
            return
        if self.left_dragging:
            if not self.stroke or (x, y) != self.stroke[-1]:
                self.stroke.append((x, y))
                self.controller.overlay.set_stroke(self.stroke)
        self.controller.render_once()

    def on_mouse_wheel_forward(self, _obj=None, _event=None):
        if self.controller.brush_interaction_active and self.GetInteractor().GetControlKey():
            self.controller.step_radius(+1)
            return
        return vtk.vtkInteractorStyleTrackballCamera.OnMouseWheelForward(self)

    def on_mouse_wheel_backward(self, _obj=None, _event=None):
        if self.controller.brush_interaction_active and self.GetInteractor().GetControlKey():
            self.controller.step_radius(-1)
            return
        return vtk.vtkInteractorStyleTrackballCamera.OnMouseWheelBackward(self)

    def cancel_stroke(self):
        self.left_dragging = False
        self.right_camera_dragging = False
        self.middle_camera_dragging = False
        self.stroke = []
        self.stroke_frame = None
        self.camera_signature = None
        self.controller.overlay.clear_stroke()

    def cancel_interaction(self):
        # Balance any synthetic/native TrackballCamera button-down state before
        # this style is detached, otherwise the next activation can inherit a
        # stale Rotate/Pan interaction state.
        try:
            if self.right_camera_dragging:
                vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonUp(self)
            if self.middle_camera_dragging:
                vtk.vtkInteractorStyleTrackballCamera.OnMiddleButtonUp(self)
        except Exception:
            pass
        self.cancel_stroke()


class VolumeSegmentationController(QtCore.QObject):
    """Automatic latest-selected target controller with per-volume/per-frame masks."""

    def __init__(self, main_window, panel, volume_container_class, parent=None):
        super().__init__(parent or main_window)
        self.main = main_window
        self.panel = panel
        self.VolumeContainer = volume_container_class
        self.render_window = main_window.render_window
        self.renderer = self.render_window.renderer
        self.overlay = SegmentationBrushOverlay(self.render_window.overlay_renderer)
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
        self.states_by_vc = {}
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
        self._authoritative_refresh_pending = False
        self._saved_clipping_process = None
        self._saved_transform_process = None
        self._saved_transform_toggle_enabled = None
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
        self._boundary_gradient_cache_limit_bytes = 256 * 1024 * 1024

        self._pool = QtCore.QThreadPool.globalInstance()
        self._job_serial = 0
        self._worker_refs = {}
        self._blocking_job = None
        self._deferred_target_sync = False
        self._threshold_recompute_active = False
        self._pending_exact_threshold = None
        self._threshold_drain_scheduled = False
        self._speck_generation = 0
        self._speck_target_vc = None
        self._speck_extent = None
        self._speck_frame = None
        self._speck_base = None
        self._speck_history_anchor = None
        self._speck_history_depth = 0
        self._speck_history_entry = None
        self._speck_job = None
        self._speck_pending = None
        self._speck_context = None
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
        p.newSelectionRequested.connect(self.new_selection)
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
        p.extractRequested.connect(self.extract_selection)
        p.deleteSelectedRequested.connect(self.delete_selected)
        p.createMaskRequested.connect(self.create_mask_volume)
        p.smartSettingsChanged.connect(self.set_smart_settings)
        p.extractOriginalRequested.connect(self.extract_original_signal)
        p.speckThresholdChanged.connect(self.remove_specks)

    def _install_tree_guards(self):
        if self._tree_connections_installed:
            return
        try:
            model = self.main.tree.model()
            model.rowsRemoved.connect(lambda *_: QtCore.QTimer.singleShot(0, self.validate_target))
            model.modelReset.connect(lambda *_: QtCore.QTimer.singleShot(0, self.validate_target))
            # A tree item's load state can change without the selection changing.
            # This matters for the common lazy-load sequence: select an unloaded
            # volume, then load it.  Once ROLE_LOADED becomes true the same selected
            # row must immediately become the segmentation target.
            self.main.tree.itemChanged.connect(
                lambda *_: QtCore.QTimer.singleShot(0, self.notify_volume_load_state_changed)
            )
            # ``currentItem`` is Qt's best representation of the row most recently
            # clicked/focused.  Only a loaded volume changes the segmentation
            # target; selecting a mesh/group afterward does not erase knowledge of
            # the latest selected volume.
            self.main.tree.currentItemChanged.connect(
                lambda *_: QtCore.QTimer.singleShot(0, self.sync_target_to_latest_selection)
            )
            self.main.tree.itemSelectionChanged.connect(
                lambda: QtCore.QTimer.singleShot(0, self.sync_target_to_latest_selection)
            )
            self._tree_connections_installed = True
        except Exception:
            pass

    def set_status(self, text):
        self.panel.set_status(text)

    def _stop_for_authoritative_error(self, error):
        message = self._preflight_diagnostic(error)
        self.set_status(message)
        if bool(getattr(self, "active", False)):
            self._deactivate_session(keep_target=True)
        self.panel.set_enabled_checked(False)
        QtWidgets.QMessageBox.warning(
            self.main,
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

    def render_once(self):
        try:
            self.render_window.render()
        except Exception:
            try:
                self.interactor.GetRenderWindow().Render()
            except Exception:
                pass

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
            if not self.main._is_volume_item(item) or not bool(item.data(0, ROLE_LOADED)):
                return None
            volume_id = item.data(0, ROLE_VOLUME_ID)
            return getattr(self.main, "volume_map", {}).get(volume_id)
        except Exception:
            return None

    def _latest_selected_volume(self):
        tree = getattr(self.main, "tree", None)
        if tree is None:
            return None, None

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

    def sync_target_to_latest_selection(self):
        if self._target_locked():
            self._deferred_target_sync = True
            return self.target_vc is not None
        vc, item = self._latest_selected_volume()
        if vc is None or item is None:
            return False
        if vc is self.target_vc:
            self.target_item = item
            return True

        if self.active:
            self.style.cancel_interaction()
            self._remove_target_observers()
            self._set_target(vc, item, announce=False)
            try:
                self._preflight_authoritative_target(
                    "Segmentation target switch", reset_session=True
                )
            except SegmentationPreflightError as exc:
                self._stop_for_authoritative_error(exc)
                return False
            self._install_target_observers()
            self._sync_current_frame_preview(render=True)
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
            if self.sync_target_to_latest_selection():
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
        if enabled == self.active:
            self.panel._on_toggle_visual(enabled)
            return
        if enabled:
            self.sync_target_to_latest_selection()
            if self.target_vc is None:
                self.panel.set_enabled_checked(False)
                QtWidgets.QMessageBox.information(
                    self.main, "Volume Segmentation",
                    "Select a loaded volume before enabling Volume Segmentation."
                )
                return
            if not self._activate_session():
                self.panel.set_enabled_checked(False)
        else:
            self._deactivate_session(keep_target=True)

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
        self._reset_speck_adjustment()
        self._session_target_snapshot = None
        if vc is None or getattr(vc, "image", None) is None:
            raise RuntimeError("Invalid segmentation target")
        source_extent = _extent_tuple(vc.image.GetExtent())
        dims = tuple(int(v) for v in vc.image.GetDimensions())
        if source_extent is None or any(v <= 0 for v in dims):
            raise RuntimeError("Invalid target image geometry")
        key = id(vc)
        state = self.states_by_vc.get(key)
        if state is None or state.target_vc is not vc:
            state = SegmentationState(vc, source_extent, dims)
            self.states_by_vc[key] = state
        elif state.source_extent != source_extent or state.source_dimensions != dims:
            # Existing masks cannot safely survive index-grid replacement.
            state = SegmentationState(vc, source_extent, dims)
            self.states_by_vc[key] = state
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
        try:
            self._preflight_authoritative_target(
                "Enable segmentation", reset_session=True
            )
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
            return False

        self.active = True
        self.brush_interaction_active = False
        self.render_window._volume_segmentation_active = False
        self.panel._on_toggle_visual(True)
        self.panel.set_controls_active(True)
        self._update_history_ui()

        timer = getattr(self.main, "volume_time_play_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
            try:
                self.main._sync_volume_time_playback_indicators()
            except Exception:
                pass

        self._install_target_observers()
        self._acquire_preview_port()
        self._sync_current_frame_preview(render=False)
        use_brush = any(
            button.isChecked() for button in self.panel.mode_buttons.values()
        )
        self.set_brush_interaction_active(use_brush, render=False)
        self.render_once()
        if self.brush_interaction_active:
            self._set_brush_status()
        else:
            self.set_status(
                "Segmentation enabled; normal mouse/navigation is active. "
                "Choose a brush to edit the mask."
            )
        return True

    def _deactivate_session(self, keep_target=True):
        self._reset_speck_adjustment()
        self.set_brush_interaction_active(False, render=False)
        self._remove_target_observers()
        self._release_preview_port()

        self.active = False
        self._session_target_snapshot = None
        self.brush_interaction_active = False
        self.render_window._volume_segmentation_active = False
        self.panel._on_toggle_visual(False)
        self.panel.set_controls_active(False)
        self.panel.set_history_available(False, False)
        if not keep_target:
            self.target_vc = None
            self.target_item = None
            self.state = None
            self.panel.set_target_name("None")
        self.render_once()

    def _purge_dead_states(self):
        live = set(id(vc) for vc in getattr(self.main, "volume_map", {}).values())
        for key in list(self.states_by_vc):
            if key not in live:
                self.states_by_vc.pop(key, None)

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
        stale_key = id(stale) if stale is not None else None
        # The tree often selects another row as part of delete/unload. Prefer an
        # immediate in-session switch to that newest loaded volume over tearing the
        # segmentation interaction mode down.
        vc, item = self._latest_selected_volume()
        if vc is stale:
            vc, item = None, None

        if stale_key is not None:
            self.states_by_vc.pop(stale_key, None)

        if vc is not None and item is not None:
            if self.active:
                self.style.cancel_interaction()
                self._remove_target_observers()
                self._set_target(vc, item, announce=False)
                try:
                    self._preflight_authoritative_target(
                        "Segmentation target switch", reset_session=True
                    )
                except SegmentationPreflightError as exc:
                    self._stop_for_authoritative_error(exc)
                    return False
                self._install_target_observers()
                self._sync_current_frame_preview(render=True)
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
        if vc not in getattr(self.main, "volume_map", {}).values():
            return False
        item = self._item_for_vc(vc)
        if item is None or not bool(item.data(0, ROLE_LOADED)):
            return False
        self.target_item = item
        return True

    def _item_for_vc(self, vc):
        tree = getattr(self.main, "tree", None)
        if tree is None:
            return None
        iterator = QtWidgets.QTreeWidgetItemIterator(tree)
        while iterator.value():
            item = iterator.value()
            try:
                if self.main.get_volume_container(item) is vc:
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
        return self.state.frame(self.current_frame_index() if index is None else index)

    # ------------------------------------------------------------------
    # interaction/widget coordination
    # ------------------------------------------------------------------
    def _set_brush_status(self):
        self.set_status({
            "select": "Select brush active. Left-drag edits; right-drag rotates; middle-drag pans; Ctrl+wheel changes radius.",
            "unselect": "Unselect brush active. Left-drag removes selected voxels; right-drag rotates; middle-drag pans; Ctrl+wheel changes radius.",
            "diffuse": "Diffuse brush active. Left-drag grows connected signal; right-drag rotates; middle-drag pans; Ctrl+wheel changes radius.",
        }.get(self.mode, "Segmentation brush active."))

    def _disable_competing_mouse_tools(self):
        brush_button = getattr(self.main, "brushModeBtn", None)
        if brush_button is not None and brush_button.isChecked():
            blocker = QtCore.QSignalBlocker(brush_button)
            brush_button.setChecked(False)
            del blocker
            try:
                self.main.toggle_brush_mode(False)
            except Exception:
                pass

        ff = getattr(self.render_window, "freefly", None)
        if ff is not None and getattr(ff, "enabled", False):
            try:
                ff.disable()
            except Exception:
                pass

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
            self._previous_cursor_mode = getattr(
                self.render_window, "current_cursor_mode", None
            )
            try:
                current_style = self.interactor.GetInteractorStyle()
                if current_style is not self.style:
                    self._previous_style = current_style
            except Exception:
                self._previous_style = None

            self._disable_competing_mouse_tools()
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
                "Choose Select, Unselect or Diffuse to resume editing."
            )

        if render:
            self.render_once()

    def _suspend_conflicting_widgets(self):
        clipping = getattr(self.render_window, "clippingWidget", None)
        if clipping is not None and hasattr(clipping, "GetProcessEvents"):
            try:
                self._saved_clipping_process = int(clipping.GetProcessEvents())
                clipping.SetProcessEvents(False)
            except Exception:
                pass
        controller = getattr(self.main, "_transform_widget_controller", None)
        widget = getattr(controller, "widget", None) if controller is not None else None
        if widget is not None and hasattr(widget, "GetProcessEvents"):
            try:
                self._saved_transform_process = int(widget.GetProcessEvents())
                widget.SetProcessEvents(False)
            except Exception:
                pass

        # Prevent the transform toggle from creating/re-enabling a vtkBoxWidget2
        # while segmentation owns the mouse. Its checked/requested state is not
        # changed, so the normal transform state machine can resume afterward.
        transform_panel = getattr(self.main, "transform_panel", None)
        toggle = getattr(transform_panel, "btnInteractiveTransform", None)
        if toggle is not None:
            try:
                self._saved_transform_toggle_enabled = bool(toggle.isEnabled())
                toggle.setEnabled(False)
            except Exception:
                self._saved_transform_toggle_enabled = None

    def _restore_conflicting_widgets(self):
        clipping = getattr(self.render_window, "clippingWidget", None)
        if clipping is not None:
            refresh = getattr(self.render_window, "_set_clipping_widget_active", None)
            try:
                if callable(refresh):
                    refresh(bool(getattr(self.render_window, "_clipping_widget_requested", False)))
                elif hasattr(clipping, "SetProcessEvents"):
                    clipping.SetProcessEvents(bool(self._saved_clipping_process))
            except Exception:
                pass
        controller = getattr(self.main, "_transform_widget_controller", None)
        if controller is not None:
            try:
                sync = getattr(controller, "_sync_enabled", None)
                if callable(sync):
                    sync()
                elif getattr(controller, "widget", None) is not None:
                    controller.widget.SetProcessEvents(bool(self._saved_transform_process))
            except Exception:
                pass

        transform_panel = getattr(self.main, "transform_panel", None)
        toggle = getattr(transform_panel, "btnInteractiveTransform", None)
        if toggle is not None and self._saved_transform_toggle_enabled is not None:
            try:
                toggle.setEnabled(bool(self._saved_transform_toggle_enabled))
            except Exception:
                pass
        self._saved_clipping_process = None
        self._saved_transform_process = None
        self._saved_transform_toggle_enabled = None

    def camera_signature(self):
        cam = self.renderer.GetActiveCamera()
        vals = list(cam.GetPosition()) + list(cam.GetFocalPoint()) + list(cam.GetViewUp())
        vals += [float(cam.GetViewAngle()), float(cam.GetParallelScale()), float(cam.GetParallelProjection())]
        return tuple(round(float(v), 9) for v in vals)

    def set_mode(self, mode):
        mode = str(mode or "select").lower()
        if mode not in {"select", "unselect", "diffuse"}:
            return
        self.mode = mode
        self.panel.set_mode(mode)
        self.overlay.set_growth_visible(mode == "diffuse")
        if self.active:
            self.set_brush_interaction_active(True)
        else:
            self.set_status({
                "select": "Select: threshold-passing signal under the inner ring is added to the mask.",
                "unselect": "Unselect: selected voxels under the inner ring are removed regardless of intensity.",
                "diffuse": "Diffuse: connected threshold-passing signal grows from the seed inside the outer ring.",
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
        self.overlay.set_radii(self.settings["seed_radius"], self.settings["growth_radius"])
        self._persist_settings()
        self.render_once()

    def set_growth_radius(self, value):
        self.settings["growth_radius"] = max(self.settings["seed_radius"], int(_clamp(value, 5, 1600)))
        self.overlay.set_radii(self.settings["seed_radius"], self.settings["growth_radius"])
        self._persist_settings()
        self.render_once()

    def step_radius(self, delta):
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
        except Exception:
            return
        if self._blocking_job is not None:
            self._pending_exact_threshold = value
            return
        self._threshold_recompute_active = True
        try:
            self._apply_threshold(value)
        finally:
            self._threshold_recompute_active = False
        self._schedule_pending_exact_threshold()

    def _apply_threshold(self, value):
        lo, hi = self._scalar_range(self.target_vc) if self.target_vc is not None else (0.0, 1.0)
        new_value = max(lo, min(hi, float(value)))
        changed = abs(new_value - float(self.signal_threshold)) > max(1e-12, abs(hi-lo)*1e-12)
        self.signal_threshold = new_value
        self.upper_threshold = hi
        if self.state is not None:
            self.state.ui_threshold = float(new_value)
        if not changed or not self.active:
            return

        fs = self._frame_state()
        if fs is None or not fs.undo_stack:
            self.set_status(f"Signal threshold {self.signal_threshold:.6g}. No applied stroke to recalculate.")
            return

        entry = fs.undo_stack[-1]
        if entry.mode not in {"select", "diffuse"}:
            self.set_status(
                f"Signal threshold {self.signal_threshold:.6g}. "
                f"The latest stroke ({entry.mode.title()}) is not threshold-sensitive."
            )
            return

        feedback_started = self._start_brush_feedback()
        feedback_success = False
        try:
            old_patch = entry.unpack("old")
            new_patch = self._recompute_history_entry(entry, old_patch, threshold_override=new_value)
            entry.rebase(old_patch, new_patch, threshold=new_value)
            self._write_mask_patch(fs, entry.extent, new_patch)
            self._sync_current_frame_preview(render=True)
            self.set_status(
                f"Signal threshold {self.signal_threshold:.6g}: latest {entry.mode.title()} stroke recalculated exactly."
            )
            feedback_success = True
        except Exception as exc:
            print(f"[VolumeSegmentation] Threshold stroke update failed: {exc}")
            traceback.print_exc()
            self.set_status(f"Threshold update failed: {exc}")
        finally:
            if feedback_started:
                self._finish_brush_feedback(feedback_success)












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
            service = getattr(self.main, "volume_snapshot_service", None)
            if service is None:
                raise RuntimeError(
                    "The authoritative volume snapshot service is unavailable."
                )
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
                acquisition_id=str(identity.acquisition_id),
                channel_id=str(identity.channel_id),
                backing_source_id=str(modeled.channel.backing_source_id or ""),
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
                operation_ids=tuple(modeled.acquisition.operation_ids),
            )
        except SegmentationPreflightError:
            raise
        except Exception as exc:
            diagnostic = self._preflight_diagnostic(exc)
            raise SegmentationPreflightError(
                f"{stage}: {diagnostic}"
            ) from exc

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
        if previous.extent != current.extent:
            incompatible("the target VTK extent/index mapping changed.")
        if previous.dimensions != current.dimensions:
            incompatible("the target voxel dimensions changed.")

    def _adopt_session_snapshot(self, current, *, stage, reset=False):
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
                self._sync_current_frame_preview(render=False)
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

    def _brush_solid_ijk(self, cx, cy, radius, depth_range, world_to_ijk, segments=28):
        znear, zfar = depth_range
        world_near = []
        world_far = []
        n = max(20, int(segments))
        for i in range(n):
            a = 2.0*math.pi*i/n
            x = float(cx) + float(radius)*math.cos(a)
            y = float(cy) + float(radius)*math.sin(a)
            world_near.append(self._display_to_world(x, y, znear))
            world_far.append(self._display_to_world(x, y, zfar))

        points = vtk.vtkPoints()
        for ring in (world_near, world_far):
            for p in ring:
                q = world_to_ijk @ np.array([p[0], p[1], p[2], 1.0], dtype=float)
                if abs(float(q[3])) > 1e-12:
                    q = q / q[3]
                points.InsertNextPoint(float(q[0]), float(q[1]), float(q[2]))

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

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(polys)
        poly.Modified()
        return poly

    def _poly_stencil(self, poly, source_extent):
        bounds = poly.GetBounds()
        if bounds is None or any(not math.isfinite(float(v)) for v in bounds):
            return None, None
        e = (
            math.floor(bounds[0])-1, math.ceil(bounds[1])+1,
            math.floor(bounds[2])-1, math.ceil(bounds[3])+1,
            math.floor(bounds[4])-1, math.ceil(bounds[5])+1,
        )
        e = extent_intersection(e, source_extent)
        if e is None:
            return None, None
        raster = vtk.vtkPolyDataToImageStencil()
        raster.SetInputData(poly)
        raster.SetOutputOrigin(0.0, 0.0, 0.0)
        raster.SetOutputSpacing(1.0, 1.0, 1.0)
        raster.SetOutputWholeExtent(*e)
        raster.SetTolerance(0.0)
        raster.Update()
        stencil = vtk.vtkImageStencilData()
        stencil.DeepCopy(raster.GetOutput())
        return stencil, e

    def _capture_stroke_geometry(self, points, radius, snap, depth_range):
        """Capture renderer-dependent brush solids without rasterizing them."""
        inv = snap["world_to_vtk_ijk_affine"]
        return [
            self._brush_solid_ijk(cx, cy, radius, depth_range, inv)
            for cx, cy in points
        ]

    def _rasterize_stroke_geometry(self, geometry, source_extent):
        """Rasterize captured brush solids; safe to run in a segmentation worker."""
        combined = None
        combined_extent = None
        for poly in geometry or ():
            stencil, extent = self._poly_stencil(poly, source_extent)
            if stencil is None:
                continue
            if combined is None:
                combined = vtk.vtkImageStencilData()
                combined.DeepCopy(stencil)
            else:
                combined.Add(stencil)
            combined_extent = extent_union(combined_extent, extent)
        return combined, combined_extent

    def _stroke_stencil(self, points, radius, snap, depth_range):
        geometry = self._capture_stroke_geometry(points, radius, snap, depth_range)
        return self._rasterize_stroke_geometry(geometry, snap["extent"])

    # ------------------------------------------------------------------
    # clipping and compiled VTK image operations
    # ------------------------------------------------------------------
    def _clipping_plane_coeffs(self, vtk_ijk_to_world):
        """Capture clipping planes as immutable IJK-space coefficients."""
        plane_coeffs = []
        planes = getattr(self.render_window, "planes", {}) or {}
        for plane in planes.values():
            try:
                n = np.asarray(plane.GetNormal(), dtype=float)
                o = np.asarray(plane.GetOrigin(), dtype=float)
                p_world = np.array(
                    [n[0], n[1], n[2], -float(np.dot(n, o))], dtype=float
                )
                p_ijk = np.asarray(vtk_ijk_to_world, dtype=float).T @ p_world
                plane_coeffs.append(tuple(float(v) for v in p_ijk))
            except Exception:
                continue
        return tuple(plane_coeffs)

    def _clipping_inside_mask(self, extent, vtk_ijk_to_world, plane_coeffs=None):
        shape = extent_shape_zyx(extent)
        if any(v <= 0 for v in shape):
            return np.zeros(shape, dtype=bool)
        if plane_coeffs is None:
            plane_coeffs = self._clipping_plane_coeffs(vtk_ijk_to_world)
        if not plane_coeffs:
            return np.ones(shape, dtype=bool)

        x = np.arange(extent[0], extent[1]+1, dtype=np.float64)[None, None, :]
        y = np.arange(extent[2], extent[3]+1, dtype=np.float64)[None, :, None]
        total = extent_voxel_count(extent)
        if total <= 8_000_000:
            z = np.arange(extent[4], extent[5]+1, dtype=np.float64)[:, None, None]
            inside = np.ones(shape, dtype=bool)
            for a, b, c, d in plane_coeffs:
                inside &= (a*x + b*y + c*z + d) >= -1e-8
            return inside

        # Large local ROI: retain vectorized XY work but chunk Z to limit temporaries.
        inside = np.ones(shape, dtype=bool)
        plane_xy = max(1, shape[1]*shape[2])
        chunk_z = max(1, min(shape[0], int(2_000_000/plane_xy)))
        for z0 in range(0, shape[0], chunk_z):
            z1 = min(shape[0], z0+chunk_z)
            z = np.arange(extent[4]+z0, extent[4]+z1, dtype=np.float64)[:, None, None]
            block = inside[z0:z1]
            for a, b, c, d in plane_coeffs:
                block &= (a*x + b*y + c*z + d) >= -1e-8
        return inside

    def _restrict_stencil_to_clipping(self, stencil, operation_extent, snap, inside_mask=None):
        if stencil is None:
            if inside_mask is None:
                inside_mask = np.zeros(extent_shape_zyx(operation_extent), dtype=bool)
            return None, inside_mask
        inside = inside_mask
        if inside is None:
            inside = self._clipping_inside_mask(operation_extent, snap["vtk_ijk_to_world_affine"])
        if np.all(inside):
            return stencil, inside
        outside_stencil = _stencil_from_binary((~inside).astype(np.uint8), operation_extent, inside_value=1)
        result = vtk.vtkImageStencilData()
        result.DeepCopy(stencil)
        result.Subtract(outside_stencil)
        return result, inside

    def _threshold_source_extent(self, extent, threshold_value=None, upper_threshold=None):
        """Threshold only ``extent`` directly from the persistent source image.

        ``vtkImageThreshold`` honors UPDATE_EXTENT, so there is no need to create
        an intermediate ``vtkImageClip`` copy for every brush operation.  The
        output is UCHAR 0/1 so downstream connectivity also runs on the smallest
        practical scalar type.
        """
        e = _extent_tuple(extent)
        if e is None:
            raise ValueError("Invalid threshold extent")
        threshold = vtk.vtkImageThreshold()
        threshold.SetInputData(self.target_vc.image)
        lower = self.signal_threshold if threshold_value is None else float(threshold_value)
        upper = self.upper_threshold if upper_threshold is None else float(upper_threshold)
        threshold.ThresholdBetween(float(lower), float(upper))
        threshold.ReplaceInOn()
        threshold.ReplaceOutOn()
        threshold.SetInValue(1)
        threshold.SetOutValue(0)
        threshold.SetOutputScalarTypeToUnsignedChar()
        threshold.UpdateInformation()
        try:
            info = threshold.GetOutputInformation(0)
            info.Set(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_EXTENT(), e, 6)
        except Exception:
            # Older VTK builds still work correctly; they may simply request the
            pass
        threshold.Update()
        return threshold


    @staticmethod
    def _apply_stencil_to_binary(input_connection, stencil):
        filt = vtk.vtkImageStencil()
        filt.SetInputConnection(input_connection)
        filt.SetStencilData(stencil)
        filt.ReverseStencilOff()
        filt.SetBackgroundValue(0)
        filt.Update()
        return filt

    def _seed_points_from_candidates(self, candidates, extent):
        """Return cheap seed points using one seed per contiguous X run.

        Every non-empty 3-D connected component contains at least one X run, so
        this guarantees coverage without generating a full int32 component-label
        image and scanning it with ``np.unique``.  The VTK connectivity filter
        itself de-duplicates overlapping flood fronts through its visited mask.
        """
        candidates = np.asarray(candidates, dtype=bool)
        if candidates.size == 0 or not np.any(candidates):
            return vtk.vtkPoints(), 0

        starts = np.empty_like(candidates, dtype=bool)
        starts[..., 0] = candidates[..., 0]
        if candidates.shape[2] > 1:
            np.logical_and(
                candidates[..., 1:],
                np.logical_not(candidates[..., :-1]),
                out=starts[..., 1:],
            )
        zz, yy, xx = np.nonzero(starts)
        count = int(xx.size)
        if count <= 0:
            return vtk.vtkPoints(), 0

        # vtkImageThresholdConnectivity converts seed points back to indices with
        # (point-origin)/spacing and currently ignores vtkImageData direction.
        # Feed exactly that coordinate convention, vectorized, so non-identity
        # image directions do not corrupt seed indexing.
        origin = np.asarray(self.target_vc.image.GetOrigin(), dtype=np.float64)
        spacing = np.asarray(self.target_vc.image.GetSpacing(), dtype=np.float64)
        ijk = np.column_stack((
            xx.astype(np.float64, copy=False) + float(extent[0]),
            yy.astype(np.float64, copy=False) + float(extent[2]),
            zz.astype(np.float64, copy=False) + float(extent[4]),
        ))
        points_xyz = origin[None, :] + ijk * spacing[None, :]
        vtk_points_array = numpy_to_vtk(
            np.ascontiguousarray(points_xyz, dtype=np.float64),
            deep=True,
        )
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk_points_array)
        return vtk_points, count

    # ------------------------------------------------------------------
    # persistent cropped mask + stroke-local history
    # ------------------------------------------------------------------
    def _ensure_mask_extent(self, frame_state, required_extent):
        required = _extent_tuple(required_extent)
        if required is None:
            return None
        current = frame_state.mask_extent
        union = extent_union(current, required)
        if union is None:
            return None

        if frame_state.mask is None:
            frame_state.mask = _new_binary_image_like(self.target_vc.image, union, fill=0)
            frame_state.mask_extent = union
        elif union != current:
            expanded = _new_binary_image_like(self.target_vc.image, union, fill=0)
            if current is not None:
                old_arr = _image_array_view(frame_state.mask)
                _image_array_view(expanded)[extent_slices_zyx(current, union)] = old_arr
            expanded.GetPointData().GetScalars().Modified()
            expanded.Modified()
            frame_state.mask = expanded
            frame_state.mask_extent = union
        else:
            self._sync_mask_geometry(frame_state.mask)
        return frame_state.mask
    # brush domains inside each HistoryEntry rather than in one global domain.

    def _mask_patch(self, frame_state, extent):
        result = np.zeros(extent_shape_zyx(extent), dtype=np.uint8)
        if frame_state is None or frame_state.mask is None:
            return result
        inter = extent_intersection(frame_state.mask_extent, extent)
        if inter is None:
            return result
        src = _image_array_view(frame_state.mask)
        result[extent_slices_zyx(inter, extent)] = src[
            extent_slices_zyx(inter, frame_state.mask_extent)
        ]
        return result

    def _write_mask_patch(self, frame_state, extent, patch):
        patch = np.ascontiguousarray(np.asarray(patch, dtype=np.uint8))
        if tuple(patch.shape) != extent_shape_zyx(extent):
            raise RuntimeError("Segmentation patch extent mismatch")
        self._ensure_mask_extent(frame_state, extent)
        target = _image_array_view(frame_state.mask)
        target[extent_slices_zyx(extent, frame_state.mask_extent)] = patch
        frame_state.mask.GetPointData().GetScalars().Modified()
        frame_state.mask.Modified()

    def _source_patch_view(self, extent):
        source_extent = _extent_tuple(self.target_vc.image.GetExtent())
        e = _extent_tuple(extent)
        if source_extent is None or e is None or extent_intersection(e, source_extent) != e:
            raise ValueError("Smart-brush extent lies outside the target image")
        return _image_array_view(self.target_vc.image)[extent_slices_zyx(e, source_extent)]

    @staticmethod
    def _sample_masked_values(array, mask, max_samples=500_000):
        """Return a bounded deterministic sample without first copying all masked voxels."""
        values = np.asarray(array)
        domain = np.asarray(mask, dtype=bool)
        if values.shape != domain.shape or values.size == 0:
            return np.empty(0, dtype=values.dtype)

        selected_count = int(np.count_nonzero(domain))
        if selected_count <= 0:
            return np.empty(0, dtype=values.dtype)
        max_samples = max(64, int(max_samples))
        if selected_count <= max_samples:
            return values[domain]

        # Spatially sample the complete ROI. This bounds both the temporary index
        # array and the gathered values even when a very large brush contains
        # millions of eligible voxels.
        flat_values = values.reshape(-1)
        flat_domain = domain.reshape(-1)
        sample_count = min(max_samples, flat_values.size)
        indices = np.linspace(0, flat_values.size - 1, sample_count, dtype=np.intp)
        sampled_domain = flat_domain[indices]
        return flat_values[indices[sampled_domain]]

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

    def _visible_seed_domain(
        self,
        extent,
        seed_domain,
        intensity_allowed,
        *,
        visible_ijk_to_clip=None,
        visible_depth_coeff=None,
        visible_viewport_px=None,
    ):
        """Keep the front-most threshold-passing seed layer for each screen bin.

        This is intentionally a seed filter, not a visibility-constrained region
        grow. The first pass builds a small depth buffer in two-pixel screen bins;
        the second keeps candidates close to the nearest signal in each bin. A
        several-voxel depth tolerance provides robust seed thickness while still
        rejecting clearly separate structures farther behind the painted target.
        """
        seed = np.asarray(seed_domain, dtype=bool)
        allowed = np.asarray(intensity_allowed, dtype=bool)
        if seed.shape != allowed.shape or seed.size == 0:
            return seed
        candidates = seed & allowed
        candidate_count = int(np.count_nonzero(candidates))
        if candidate_count <= 0:
            return np.zeros_like(seed, dtype=bool)

        try:
            ijk_to_clip = np.asarray(visible_ijk_to_clip, dtype=float).reshape(4, 4)
            depth_coeff = np.asarray(visible_depth_coeff, dtype=float).reshape(4)
            vx0, vy0, vx1, vy1 = (float(v) for v in visible_viewport_px)
        except Exception:
            # Fail open if an old in-memory history entry lacks projection data.
            return seed

        viewport_w = max(1.0, vx1 - vx0)
        viewport_h = max(1.0, vy1 - vy0)
        if not (
            np.all(np.isfinite(ijk_to_clip))
            and np.all(np.isfinite(depth_coeff))
            and all(math.isfinite(v) for v in (vx0, vy0, vx1, vy1))
        ):
            return seed

        # Two-pixel bins are more stable than exact raster pixels for discrete
        # voxel centers and halve the depth-buffer dimensions in each direction.
        bin_size = 2.0
        bins_x = max(1, int(math.ceil(viewport_w / bin_size)))
        bins_y = max(1, int(math.ceil(viewport_h / bin_size)))
        front_depth = np.full(bins_x * bins_y, np.inf, dtype=np.float32)

        e = _extent_tuple(extent)
        if e is None:
            return seed
        shape = candidates.shape
        nz, ny, nx = (int(v) for v in shape)
        flat_candidates = candidates.reshape(-1)
        block_voxels = 500_000

        def projected_block(start, stop):
            local = np.flatnonzero(flat_candidates[start:stop])
            if local.size == 0:
                return None
            flat_index = local.astype(np.int64, copy=False) + int(start)
            xx = flat_index % nx
            tmp = flat_index // nx
            yy = tmp % ny
            zz = tmp // ny

            coords = np.empty((flat_index.size, 4), dtype=np.float64)
            coords[:, 0] = xx + float(e[0])
            coords[:, 1] = yy + float(e[2])
            coords[:, 2] = zz + float(e[4])
            coords[:, 3] = 1.0

            clip = coords @ ijk_to_clip.T
            w = clip[:, 3]
            valid = np.isfinite(w) & (np.abs(w) > 1e-12)
            if not np.any(valid):
                return None
            ndc_x = np.empty_like(w)
            ndc_y = np.empty_like(w)
            ndc_x.fill(np.nan)
            ndc_y.fill(np.nan)
            ndc_x[valid] = clip[valid, 0] / w[valid]
            ndc_y[valid] = clip[valid, 1] / w[valid]
            px = vx0 + 0.5 * (ndc_x + 1.0) * viewport_w
            py = vy0 + 0.5 * (ndc_y + 1.0) * viewport_h
            bx_float = np.floor((px - vx0) / bin_size)
            by_float = np.floor((py - vy0) / bin_size)
            valid &= (
                np.isfinite(px) & np.isfinite(py)
                & np.isfinite(bx_float) & np.isfinite(by_float)
                & (bx_float >= 0) & (bx_float < bins_x)
                & (by_float >= 0) & (by_float < bins_y)
            )
            indices = np.flatnonzero(valid)
            if indices.size == 0:
                return None
            depth = coords[indices] @ depth_coeff
            depth_valid = np.isfinite(depth) & (depth >= -1e-9)
            if not np.any(depth_valid):
                return None
            indices = indices[depth_valid]
            depth = depth[depth_valid]
            bx = bx_float[indices].astype(np.int64, copy=False)
            by = by_float[indices].astype(np.int64, copy=False)
            return (
                flat_index[indices].astype(np.intp, copy=False),
                (by * bins_x + bx).astype(np.intp, copy=False),
                depth.astype(np.float32, copy=False),
            )

        # First pass: nearest threshold-passing seed signal in each screen bin.
        for start in range(0, flat_candidates.size, block_voxels):
            stop = min(flat_candidates.size, start + block_voxels)
            projected = projected_block(start, stop)
            if projected is None:
                continue
            _flat_index, pixels, depth = projected
            np.minimum.at(front_depth, pixels, depth)

        if not np.any(np.isfinite(front_depth)):
            return seed

        # Approximate four source-voxel layers along the camera direction. This
        # gives the seed enough thickness for connectivity without admitting a
        # clearly separated structure farther down the same viewing ray.
        one_voxel_depth = float(np.max(np.abs(depth_coeff[:3])))
        if not math.isfinite(one_voxel_depth) or one_voxel_depth <= 1e-12:
            one_voxel_depth = 1e-6
        depth_tolerance = max(1e-6, 4.0 * one_voxel_depth)

        visible_flat = np.zeros(flat_candidates.size, dtype=bool)
        for start in range(0, flat_candidates.size, block_voxels):
            stop = min(flat_candidates.size, start + block_voxels)
            projected = projected_block(start, stop)
            if projected is None:
                continue
            flat_index, pixels, depth = projected
            keep = depth <= (front_depth[pixels] + depth_tolerance)
            if np.any(keep):
                visible_flat[flat_index[keep]] = True

        return visible_flat.reshape(shape)

    def _estimate_local_threshold(self, extent, domain):
        """Return a deterministic Otsu-like threshold from the painted 3-D domain.

        The stroke frustum naturally contains foreground plus local background
        along the viewing rays, which makes it a useful local intensity sample.
        Sampling is bounded before masked values are materialized so very large
        screen-space brushes cannot create an equally large temporary array.
        """
        mask = np.asarray(domain, dtype=bool)
        if mask.size == 0 or np.count_nonzero(mask) < 64:
            return None
        source = np.asarray(self._source_patch_view(extent))
        values = self._sample_masked_values(source, mask)
        if values.size < 64:
            return None
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 64:
            return None

        lo = float(np.percentile(values, 1.0))
        hi = float(np.percentile(values, 99.5))
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            return None

        hist, edges = np.histogram(values, bins=256, range=(lo, hi))
        hist = hist.astype(np.float64, copy=False)
        total = float(hist.sum())
        if total < 64.0 or np.count_nonzero(hist) < 2:
            return None
        centers = 0.5 * (edges[:-1] + edges[1:])
        weight_left = np.cumsum(hist)
        sum_left = np.cumsum(hist * centers)
        total_sum = float(sum_left[-1])
        weight_right = total - weight_left

        valid = (weight_left > 0.0) & (weight_right > 0.0)
        if not np.any(valid):
            return None
        mean_left = np.zeros_like(centers)
        mean_right = np.zeros_like(centers)
        mean_left[valid] = sum_left[valid] / weight_left[valid]
        mean_right[valid] = (total_sum - sum_left[valid]) / weight_right[valid]
        between = np.full_like(centers, -1.0)
        between[valid] = (
            weight_left[valid] * weight_right[valid]
            * np.square(mean_left[valid] - mean_right[valid])
        )
        index = int(np.argmax(between))
        value = float(centers[index])
        scalar_lo, scalar_hi = self._scalar_range(self.target_vc)
        return max(float(scalar_lo), min(float(scalar_hi), value))

    def _faint_growth_threshold(self, seed_threshold):
        """Relax a confident seed threshold toward the target's background floor."""
        lo, hi = self._scalar_range(self.target_vc)
        seed_threshold = max(float(lo), min(float(hi), float(seed_threshold)))
        # Keep roughly 72% of the distance above the scalar floor. This is
        # intentionally conservative: faint connected branches become available
        # without turning Diffuse into an unrestricted low-threshold flood fill.
        return max(float(lo), min(seed_threshold, float(lo) + 0.72 * (seed_threshold - float(lo))))

    def _boundary_guard_domain(self, extent, growth_domain, intensity_allowed, seed_domain):
        """Remove the strongest local signal gradients from the Diffuse domain."""
        growth_domain = np.asarray(growth_domain, dtype=bool)
        intensity_allowed = np.asarray(intensity_allowed, dtype=bool)
        seed_domain = np.asarray(seed_domain, dtype=bool)
        eligible = growth_domain & intensity_allowed
        if np.count_nonzero(eligible) < 128:
            return growth_domain, None

        # Stream only the requested ROI directly from the persistent source image.
        # Avoid vtkImageClip here: its extra cropped source allocation is expensive
        # for the large screen-space growth radii supported by MADI3D. Reuse the
        # gradient while repeatedly adjusting the threshold of the same latest
        # stroke; only the intensity eligibility/cutoff changes between updates.
        image = self.target_vc.image
        try:
            image_mtime = int(image.GetMTime())
        except Exception:
            image_mtime = -1
        cache_key = (id(image), image_mtime, tuple(int(v) for v in extent))
        magnitude = None
        if cache_key == self._boundary_gradient_cache_key:
            cached = self._boundary_gradient_cache
            if cached is not None and cached.shape == growth_domain.shape:
                magnitude = cached

        if magnitude is None:
            gradient = vtk.vtkImageGradientMagnitude()
            gradient.SetInputData(image)
            gradient.SetDimensionality(3)
            gradient.HandleBoundariesOn()
            gradient.UpdateInformation()
            try:
                gradient.GetOutputInformation(0).Set(
                    vtk.vtkStreamingDemandDrivenPipeline.UPDATE_EXTENT(), extent, 6
                )
            except Exception:
                pass
            gradient.Update()
            magnitude = np.array(
                _image_array_view(gradient.GetOutput()), dtype=np.float32, copy=True
            )
            if magnitude.shape != growth_domain.shape:
                return growth_domain, None
            if magnitude.nbytes <= int(self._boundary_gradient_cache_limit_bytes):
                self._boundary_gradient_cache_key = cache_key
                self._boundary_gradient_cache = magnitude
            else:
                self._boundary_gradient_cache_key = None
                self._boundary_gradient_cache = None

        values = self._sample_masked_values(magnitude, eligible)
        values = values[np.isfinite(values)]
        if values.size < 128:
            return growth_domain, None
        cutoff = float(np.percentile(values, 95.0))
        if not math.isfinite(cutoff) or cutoff <= 0.0:
            return growth_domain, None

        guarded = growth_domain & (magnitude <= cutoff)
        # Never let the boundary test erase the user's confident seed itself.
        guarded |= growth_domain & seed_domain
        return guarded, cutoff

    def _threshold_allowed_patch(self, extent, threshold_value):
        """Fast Boolean threshold for consumers that do not need a VTK pipeline."""
        source = np.asarray(self._source_patch_view(extent))
        lower = float(threshold_value)
        upper = float(self.upper_threshold)
        allowed = np.greater_equal(source, lower)
        if math.isfinite(upper):
            np.logical_and(allowed, np.less_equal(source, upper), out=allowed)
        return allowed

    def _compute_diffuse_patch(
        self,
        extent,
        old_patch,
        seed_domain,
        growth_domain,
        threshold_value,
        growth_stencil=None,
        *,
        smart_faint_recovery=False,
        smart_boundary_guard=False,
        smart_visible_seeds=False,
        visible_ijk_to_clip=None,
        visible_depth_coeff=None,
        visible_viewport_px=None,
    ):
        old = np.asarray(old_patch, dtype=bool)
        seed_domain = np.asarray(seed_domain, dtype=bool)
        growth_domain = np.asarray(growth_domain, dtype=bool)

        seed_threshold = float(threshold_value)
        growth_threshold = (
            self._faint_growth_threshold(seed_threshold)
            if smart_faint_recovery else seed_threshold
        )

        # Seeds remain strict even when faint recovery is enabled. Connectivity
        # may then traverse the lower growth threshold inside the outer domain.
        seed_filter = self._threshold_source_extent(extent, seed_threshold, self.upper_threshold)
        seed_allowed = np.asarray(_image_array_view(seed_filter.GetOutput()), dtype=bool)

        effective_seed_domain = seed_domain
        if smart_visible_seeds:
            effective_seed_domain = self._visible_seed_domain(
                extent,
                seed_domain,
                seed_allowed,
                visible_ijk_to_clip=visible_ijk_to_clip,
                visible_depth_coeff=visible_depth_coeff,
                visible_viewport_px=visible_viewport_px,
            )

        if abs(growth_threshold - seed_threshold) <= max(1e-12, abs(self.upper_threshold) * 1e-12):
            growth_filter = seed_filter
            growth_allowed = seed_allowed
        else:
            growth_filter = self._threshold_source_extent(extent, growth_threshold, self.upper_threshold)
            growth_allowed = np.asarray(_image_array_view(growth_filter.GetOutput()), dtype=bool)

        if smart_boundary_guard:
            growth_domain, _cutoff = self._boundary_guard_domain(
                extent, growth_domain, growth_allowed, effective_seed_domain
            )
            # The original screen-space stencil no longer represents the guarded
            # Boolean domain, so rebuild the local stencil once.
            growth_stencil = None

        # Existing selected voxels remain valid Diffuse starting points even when
        # Visible Seeds is enabled. Visibility filtering applies only to new seed
        # material from this stroke; it must not invalidate deliberate prior work.
        candidates = ((old & seed_allowed) | (effective_seed_domain & seed_allowed)) & growth_domain
        seed_points, seed_count = self._seed_points_from_candidates(candidates, extent)
        if seed_count <= 0:
            return old.astype(np.uint8), 0

        if growth_stencil is None:
            growth_stencil = _stencil_from_binary(
                growth_domain.astype(np.uint8), extent, inside_value=1
            )

        grow = vtk.vtkImageThresholdConnectivity()
        grow.SetInputConnection(growth_filter.GetOutputPort())
        grow.ThresholdBetween(1.0, 1.0)
        grow.SetSeedPoints(seed_points)
        grow.SetStencilData(growth_stencil)
        grow.ReplaceInOn()
        grow.ReplaceOutOn()
        grow.SetInValue(1)
        grow.SetOutValue(0)
        # vtkImageThresholdConnectivity otherwise requests the full upstream
        # WHOLE_EXTENT even when vtkImageThreshold was previously updated only
        # for this ROI. Constrain both its flood-fill slice range and output
        # UPDATE_EXTENT so the source threshold remains strictly local.
        grow.SetSliceRangeX(int(extent[0]), int(extent[1]))
        grow.SetSliceRangeY(int(extent[2]), int(extent[3]))
        grow.SetSliceRangeZ(int(extent[4]), int(extent[5]))
        grow.UpdateInformation()
        try:
            grow.GetOutputInformation(0).Set(
                vtk.vtkStreamingDemandDrivenPipeline.UPDATE_EXTENT(), extent, 6
            )
        except Exception:
            pass
        grow.Update()

        result = np.asarray(_image_array_view(grow.GetOutput()), dtype=bool)
        result &= growth_domain
        return (old | result).astype(np.uint8), seed_count

    def _recompute_history_entry(self, entry, old_patch, threshold_override=None):
        old = np.ascontiguousarray(np.asarray(old_patch, dtype=np.uint8))
        mode = str(entry.mode).lower()
        if mode == "clear":
            return np.zeros_like(old, dtype=np.uint8)

        seed = entry.domain("seed")
        if seed is None:
            seed = np.zeros_like(old, dtype=np.uint8)
        seed = np.asarray(seed, dtype=bool)
        if mode == "unselect":
            return (np.asarray(old, dtype=bool) & ~seed).astype(np.uint8)

        threshold_value = entry.threshold if threshold_override is None else float(threshold_override)
        if threshold_value is None:
            threshold_value = float(self.signal_threshold)
        if mode == "select":
            allowed = self._threshold_allowed_patch(entry.extent, threshold_value)
            effective_seed = seed
            if bool(entry.smart_visible_seeds):
                effective_seed = self._visible_seed_domain(
                    entry.extent,
                    seed,
                    allowed,
                    visible_ijk_to_clip=entry.visible_ijk_to_clip,
                    visible_depth_coeff=entry.visible_depth_coeff,
                    visible_viewport_px=entry.visible_viewport_px,
                )
            return (np.asarray(old, dtype=bool) | (effective_seed & allowed)).astype(np.uint8)
        if mode == "diffuse":
            growth = entry.domain("growth")
            if growth is None:
                growth = seed
            new_patch, _components = self._compute_diffuse_new_patch(
                entry.extent,
                old,
                seed,
                growth,
                threshold_value,
                smart_faint_recovery=bool(entry.smart_faint_recovery),
                smart_boundary_guard=bool(entry.smart_boundary_guard),
                smart_visible_seeds=bool(entry.smart_visible_seeds),
                visible_ijk_to_clip=entry.visible_ijk_to_clip,
                visible_depth_coeff=entry.visible_depth_coeff,
                visible_viewport_px=entry.visible_viewport_px,
            )
            return new_patch
        if mode == "specks":
            return np.array(entry.unpack("new"), copy=True, dtype=np.uint8)
        raise ValueError(f"Unsupported segmentation history mode: {mode}")

    def _push_history(self, frame_state, entry, clear_redo=True):
        if str(entry.mode).lower() != "specks":
            self._reset_speck_adjustment()
        frame_state.undo_stack.append(entry)
        limit = int(self.settings.get("history_limit", 40))
        if len(frame_state.undo_stack) > limit:
            del frame_state.undo_stack[:len(frame_state.undo_stack)-limit]
        if clear_redo:
            frame_state.redo_stack.clear()
        self._update_history_ui()

    def _clear_frame_mask_storage(self, frame_state):
        frame_state.mask = None
        frame_state.mask_extent = None

    def _commit_history_entry(self, frame_state, entry, clear_redo=True):
        if str(entry.mode).lower() == "clear":
            self._clear_frame_mask_storage(frame_state)
        else:
            self._write_mask_patch(frame_state, entry.extent, entry.unpack("new"))
        self._push_history(frame_state, entry, clear_redo=clear_redo)

    def _apply_history_entry(self, frame_state, entry, which):
        if str(entry.mode).lower() == "clear" and which == "new":
            self._clear_frame_mask_storage(frame_state)
            return
        self._write_mask_patch(frame_state, entry.extent, entry.unpack(which))

    def undo(self):
        self._reset_speck_adjustment()
        if not self.active:
            return
        fs = self._frame_state()
        if fs is None or not fs.undo_stack:
            return
        entry = fs.undo_stack.pop()
        self._apply_history_entry(fs, entry, "old")
        fs.redo_stack.append(entry)
        self._update_history_ui()
        self._sync_current_frame_preview(render=True)
        self.set_status(f"Undo {entry.mode.title()} stroke. Threshold unchanged at {self.signal_threshold:.6g}.")

    def redo(self):
        self._reset_speck_adjustment()
        if not self.active:
            return
        fs = self._frame_state()
        if fs is None or not fs.redo_stack:
            return
        entry = fs.redo_stack.pop()
        # Rebase Redo on the current mask. This is essential after the user undoes
        # to an older stroke and changes only that stroke's threshold.
        old_patch = self._mask_patch(fs, entry.extent)
        new_patch = self._recompute_history_entry(entry, old_patch)
        entry.rebase(old_patch, new_patch)
        if str(entry.mode).lower() == "clear":
            self._clear_frame_mask_storage(fs)
        else:
            self._write_mask_patch(fs, entry.extent, new_patch)
        fs.undo_stack.append(entry)
        self._update_history_ui()
        self._sync_current_frame_preview(render=True)
        self.set_status(f"Redo {entry.mode.title()} stroke. Threshold unchanged at {self.signal_threshold:.6g}.")

    def clear(self):
        if not self.active:
            return
        fs = self._frame_state()
        if fs is None or fs.mask is None:
            return
        old = np.array(_image_array_view(fs.mask), copy=True, dtype=np.uint8)
        if not np.any(old):
            return
        new = np.zeros_like(old, dtype=np.uint8)
        entry = HistoryEntry.make(fs.mask_extent, old, new, mode="clear")
        self._commit_history_entry(fs, entry)
        self._sync_current_frame_preview(render=True)
        self.set_status("Segmentation cleared.")

    def new_selection(self):
        self.clear()
        self.set_mode("select")

    def _update_history_ui(self):
        fs = self._frame_state() if self.state is not None else None
        self.panel.set_history_available(bool(fs and fs.undo_stack), bool(fs and fs.redo_stack))

    # ------------------------------------------------------------------
    # Select / Unselect / Diffuse
    # ------------------------------------------------------------------
    def _process_diffuse_stroke(self, points, frame_index, snap, depth_range):
        """Capture view state on the GUI thread, then defer 3-D Diffuse work."""
        smart_local = bool(self.settings.get("smart_local_threshold", False))
        smart_faint = bool(self.settings.get("smart_faint_recovery", False))
        smart_boundary = bool(self.settings.get("smart_boundary_guard", False))
        smart_visible = bool(self.settings.get("smart_visible_seeds", False))
        visible_projection = None
        visible_fallback = False

        if smart_visible:
            visible_projection = self._visible_seed_projection_snapshot(snap)
            if visible_projection is None:
                smart_visible = False
                visible_fallback = True
                self.set_status(
                    "Visible Seeds could not read the current camera projection; "
                    "this stroke will use normal seeding."
                )

        # Display->world unprojection belongs to the renderer thread. Capture
        # only the small immutable brush solids here; expensive 3-D stencil
        # rasterization and domain materialization happen in the worker below.
        seed_geometry = self._capture_stroke_geometry(
            points, self.settings["seed_radius"], snap, depth_range
        )
        growth_geometry = self._capture_stroke_geometry(
            points, self.settings["growth_radius"], snap, depth_range
        )
        clipping_plane_coeffs = self._clipping_plane_coeffs(
            snap["vtk_ijk_to_world_affine"]
        )

        fs = self._frame_state(frame_index)
        result = self._compute_diffuse_stroke(
            seed_geometry,
            growth_geometry,
            snap,
            fs,
            frame_index,
            float(self.signal_threshold),
            clipping_plane_coeffs=clipping_plane_coeffs,
            smart_local_threshold=smart_local,
            smart_faint_recovery=smart_faint,
            smart_boundary_guard=smart_boundary,
            smart_visible_seeds=smart_visible,
            visible_ijk_to_clip=(visible_projection or {}).get("ijk_to_clip"),
            visible_depth_coeff=(visible_projection or {}).get("depth_coeff"),
            visible_viewport_px=(visible_projection or {}).get("viewport_px"),
        )
        if result is None:
            self.set_status("The stroke did not intersect the target volume.")
            return

        operation_extent = result["operation_extent"]
        old_patch = result["old_patch"]
        new_patch = result["new_patch"]
        seed_domain = result["seed_domain"]
        growth_domain = result["growth_domain"]
        threshold_value = float(result["threshold_value"])
        used_local_threshold = bool(result["used_local_threshold"])
        component_count = int(result["component_count"])

        if used_local_threshold:
            # The UI must show the threshold actually used by this stroke, but
            # updating it here must not trigger a second threshold recomputation.
            lo, hi = self._scalar_range(self.target_vc)
            self.signal_threshold = threshold_value
            self.upper_threshold = float(hi)
            if self.state is not None:
                self.state.ui_threshold = threshold_value
            self.panel.configure_threshold(lo, hi, threshold_value)

        entry = HistoryEntry.make(
            operation_extent,
            old_patch,
            new_patch,
            mode="diffuse",
            threshold=threshold_value,
            seed_domain=seed_domain,
            growth_domain=growth_domain,
            smart_local_threshold=used_local_threshold,
            smart_faint_recovery=smart_faint,
            smart_boundary_guard=smart_boundary,
            smart_visible_seeds=smart_visible,
            visible_ijk_to_clip=(visible_projection or {}).get("ijk_to_clip"),
            visible_depth_coeff=(visible_projection or {}).get("depth_coeff"),
            visible_viewport_px=(visible_projection or {}).get("viewport_px"),
        )
        changed = not np.array_equal(old_patch, new_patch)
        detail = (
            f"Diffuse ({component_count} seed run"
            f"{'s' if component_count != 1 else ''})"
        )

        helper_labels = []
        if used_local_threshold:
            helper_labels.append("local threshold")
        if smart_faint:
            helper_labels.append("faint recovery")
        if smart_boundary:
            helper_labels.append("boundary guard")
        if smart_visible:
            helper_labels.append("visible seeds")
        elif visible_fallback:
            helper_labels.append("visible seeds unavailable for this stroke")
        helper_text = f" Smart: {', '.join(helper_labels)}." if helper_labels else ""

        self._commit_history_entry(fs, entry)
        self._sync_current_frame_preview(render=True)
        dims = extent_shape_zyx(operation_extent)
        if changed:
            self.set_status(
                f"{detail} updated the mask at threshold {threshold_value:.6g}."
                f"{helper_text} Local extent: {dims[2]}×{dims[1]}×{dims[0]} voxels."
            )
        else:
            self.set_status(
                f"{detail}: this stroke did not add voxels at threshold {threshold_value:.6g}."
                f"{helper_text} Adjust the threshold to recalculate this stroke only."
            )

    def process_stroke(self, raw_points, frame_index):
        if not self.active or not raw_points:
            return
        feedback_started = False
        feedback_success = False
        try:
            context = self._validate_source_geometry()
            if int(frame_index) != self.current_frame_index():
                return
            spacing_radius = self.settings["seed_radius"]
            points = self.resample_stroke(raw_points, spacing_radius)
            if not points:
                return
            feedback_started = self._start_brush_feedback()
            snap = self._geometry_snapshot(context)
            depth_range = self._target_display_depth_range()

            if self.mode == "diffuse":
                self._process_diffuse_stroke(points, frame_index, snap, depth_range)
                feedback_success = True
                return

            seed_stencil, seed_extent = self._stroke_stencil(
                points, self.settings["seed_radius"], snap, depth_range
            )
            if seed_stencil is None or seed_extent is None:
                self.set_status("The stroke did not intersect the target volume.")
                feedback_success = True
                return

            operation_extent = extent_intersection(seed_extent, snap["extent"])
            if operation_extent is None:
                feedback_success = True
                return
            inside_clip = self._clipping_inside_mask(
                operation_extent, snap["vtk_ijk_to_world_affine"]
            )
            seed_stencil, _inside_clip = self._restrict_stencil_to_clipping(
                seed_stencil, operation_extent, snap, inside_mask=inside_clip
            )

            fs = self._frame_state(frame_index)
            old_patch = self._mask_patch(fs, operation_extent)
            seed_domain = _stencil_to_bool(seed_stencil, operation_extent)
            threshold_value = float(self.signal_threshold)
            used_local_threshold = False
            smart_local = bool(self.settings.get("smart_local_threshold", False))
            smart_visible = bool(self.settings.get("smart_visible_seeds", False))
            visible_projection = None
            visible_fallback = False

            if self.mode == "select" and smart_local:
                local_threshold = self._estimate_local_threshold(
                    operation_extent, seed_domain
                )
                if local_threshold is not None:
                    threshold_value = float(local_threshold)
                    used_local_threshold = True
                    # The threshold widget must represent the threshold actually
                    # used by this stroke. Otherwise the next tiny manual slider
                    # movement would jump from the old global value and radically
                    # change a locally-thresholded result. Update without emitting
                    # thresholdChanged/recomputing the stroke a second time.
                    lo, hi = self._scalar_range(self.target_vc)
                    self.signal_threshold = threshold_value
                    self.upper_threshold = float(hi)
                    if self.state is not None:
                        self.state.ui_threshold = threshold_value
                    self.panel.configure_threshold(lo, hi, threshold_value)

            if self.mode == "select" and smart_visible:
                visible_projection = self._visible_seed_projection_snapshot(snap)
                if visible_projection is None:
                    # Keep the stroke usable if a renderer/backend cannot provide
                    # the required projection matrix; report it and fall back to
                    # normal seeding for this stroke only.
                    smart_visible = False
                    visible_fallback = True
                    self.set_status(
                        "Visible Seeds could not read the current camera projection; "
                        "this stroke will use normal seeding."
                    )

            if self.mode == "unselect":
                new_patch = (
                    np.asarray(old_patch, dtype=bool) & ~seed_domain
                ).astype(np.uint8)
                entry = HistoryEntry.make(
                    operation_extent,
                    old_patch,
                    new_patch,
                    mode="unselect",
                    seed_domain=seed_domain,
                )
                detail = "Unselect"
                changed = not np.array_equal(old_patch, new_patch)
                if not changed:
                    self.set_status("Unselect: no selected voxels changed.")
                    feedback_success = True
                    return
            else:
                allowed = self._threshold_allowed_patch(
                    operation_extent, threshold_value
                )
                effective_seed = seed_domain
                if smart_visible:
                    effective_seed = self._visible_seed_domain(
                        operation_extent,
                        seed_domain,
                        allowed,
                        visible_ijk_to_clip=visible_projection["ijk_to_clip"],
                        visible_depth_coeff=visible_projection["depth_coeff"],
                        visible_viewport_px=visible_projection["viewport_px"],
                    )
                new_patch = (
                    np.asarray(old_patch, dtype=bool) | (effective_seed & allowed)
                ).astype(np.uint8)
                entry = HistoryEntry.make(
                    operation_extent,
                    old_patch,
                    new_patch,
                    mode="select",
                    threshold=threshold_value,
                    seed_domain=seed_domain,
                    smart_local_threshold=used_local_threshold,
                    smart_visible_seeds=smart_visible,
                    visible_ijk_to_clip=(visible_projection or {}).get("ijk_to_clip"),
                    visible_depth_coeff=(visible_projection or {}).get("depth_coeff"),
                    visible_viewport_px=(visible_projection or {}).get("viewport_px"),
                )
                detail = "Select"
                changed = not np.array_equal(old_patch, new_patch)
                # Keep even a currently-empty thresholded Select stroke. Lowering
                # the threshold immediately afterward must be able to reveal it.

            helper_labels = []
            if used_local_threshold:
                helper_labels.append("local threshold")
            if self.mode == "select" and smart_visible:
                helper_labels.append("visible seeds")
            elif visible_fallback:
                helper_labels.append("visible seeds unavailable for this stroke")
            helper_text = f" Smart: {', '.join(helper_labels)}." if helper_labels else ""

            self._commit_history_entry(fs, entry)
            self._sync_current_frame_preview(render=True)
            dims = extent_shape_zyx(operation_extent)
            if changed:
                self.set_status(
                    f"{detail} updated the mask at threshold {threshold_value:.6g}."
                    f"{helper_text} Local extent: {dims[2]}×{dims[1]}×{dims[0]} voxels."
                )
            else:
                self.set_status(
                    f"{detail}: this stroke did not add voxels at threshold {threshold_value:.6g}."
                    f"{helper_text} Adjust the threshold to recalculate this stroke only."
                )
            feedback_success = True
        except SegmentationPreflightError as exc:
            print(f"[VolumeSegmentation] Preflight failed: {exc}")
            self._stop_for_authoritative_error(exc)
        except Exception as exc:
            print(f"[VolumeSegmentation] Stroke failed: {exc}")
            traceback.print_exc()
            self.set_status(f"Segmentation operation failed: {exc}")
            QtWidgets.QMessageBox.warning(self.main, "Volume Segmentation", str(exc))
        finally:
            if feedback_started:
                self._finish_brush_feedback(feedback_success)
    # _compute_diffuse_new_patch so the same code can be reused by threshold edit
    # and Redo rebasing.

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
                    self.main, "Segmentation preview unavailable",
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
        if self.preview_port is None:
            self.preview_volume = None
            self.preview_property = None
            self.preview_producer = None
            self._preview_output_image = None
            self._preview_matrix_signature = None
            return
        port = self.preview_port
        try:
            self.preview_volume.SetVisibility(False)
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
        self.preview_port = None
        self.preview_volume = None
        self.preview_property = None
        self.preview_producer = None
        self._preview_output_image = None
        self._preview_matrix_signature = None

    def _apply_preview_property(self):
        prop = self.preview_property
        if prop is None:
            return
        color = self.settings["mask_color"]
        opacity = float(self.settings["mask_opacity"])
        ctf = vtk.vtkColorTransferFunction()
        ctf.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        ctf.AddRGBPoint(1.0, *color)
        otf = vtk.vtkPiecewiseFunction()
        otf.AddPoint(0.0, 0.0)
        otf.AddPoint(0.5, 0.0)
        otf.AddPoint(1.0, opacity)
        prop.SetColor(ctf)
        prop.SetScalarOpacity(otf)
        prop.SetInterpolationTypeToNearest()
        prop.ShadeOff()
        prop.Modified()

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
        for fs in self.state.frame_states.values():
            if fs.mask is not None:
                self._sync_mask_geometry(fs.mask)

    def _sync_current_frame_preview(self, render=False):
        if self.preview_port is None or self.preview_producer is None:
            if render:
                self.render_once()
            return
        fs = self._frame_state()
        image = fs.mask if fs is not None else None
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
        if self._preview_output_image is not image:
            self.preview_producer.SetOutput(image)
            self.preview_producer.Modified()
            self._preview_output_image = image
        # If the same vtkImageData was edited in-place, its scalar/image MTime is
        # sufficient for vtkOpenGLGPUVolumeRayCastMapper to refresh the texture.
        # Avoid mapper.Modified()/vtkMultiVolume.Modified() here because those can
        # trigger broader mapper/shader invalidation unrelated to the mask data.
        self._sync_preview_transform()
        self._update_history_ui()
        if render:
            self.render_once()

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
        self._remove_target_observers()
        if self.target_vc is None:
            return
        try:
            self._target_volume_observer = self.target_vc.volume.AddObserver(
                vtk.vtkCommand.ModifiedEvent, self._on_target_modified
            )
        except Exception:
            self._target_volume_observer = None
        self._install_image_observer()
        self._last_frame_index = self.current_frame_index()
        self._last_geometry_signature = self._image_geometry_signature()

    def _install_image_observer(self):
        image = getattr(self.target_vc, "image", None)
        if image is self._observed_image:
            return
        if self._observed_image is not None and self._target_image_observer is not None:
            try:
                self._observed_image.RemoveObserver(self._target_image_observer)
            except Exception:
                pass
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
        if self.target_vc is not None and self._target_volume_observer is not None:
            try:
                self.target_vc.volume.RemoveObserver(self._target_volume_observer)
            except Exception:
                pass
        if self._observed_image is not None and self._target_image_observer is not None:
            try:
                self._observed_image.RemoveObserver(self._target_image_observer)
            except Exception:
                pass
        self._target_volume_observer = None
        self._target_image_observer = None
        self._observed_image = None
        self._last_frame_index = None
        self._last_geometry_signature = None

    def _on_target_modified(self, caller=None, _event=None):
        if not self.active or self._sync_guard:
            return
        # Pose and voxel-size transactions update several runtime/model fields
        # synchronously. Validate once on the next event-loop turn, after the
        # transaction has either committed or rolled back.
        if not self._authoritative_refresh_pending:
            self._authoritative_refresh_pending = True
            QtCore.QTimer.singleShot(0, self._validate_observed_target_change)
        # Keep a synchronized pose visually responsive; the queued authoritative
        # validation still stops the session if this was a runtime-only mutation.
        self._sync_preview_transform()

    def _validate_observed_target_change(self):
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
                if frame_changed:
                    self._reset_speck_adjustment()
                # Each time frame owns its own already-thresholded stroke history;
                # switching frames changes only which retained mask is previewed.
                self._sync_current_frame_preview(render=False)
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
        if not self.active:
            return
        fs = self._frame_state()
        if fs is None or fs.mask is None or not np.any(_image_array_view(fs.mask)):
            QtWidgets.QMessageBox.information(self.main, "Volume Segmentation", "The current frame has no selected voxels.")
            return
        try:
            operation_snapshot = self._preflight_authoritative_target(
                "Create Mask Volume start validation"
            )
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
            return
        output = vtk.vtkImageData()
        output.DeepCopy(fs.mask)
        name = self._derived_name("mask")
        try:
            self._add_derived_volume(
                output,
                name,
                mask_volume=True,
                operation="segmentation_mask",
                operation_snapshot=operation_snapshot,
            )
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
            return
        self.set_status(f"Created mask volume: {name}")

    def extract_selection(self):
        fs = self._output_mask()
        if fs is None:
            return
        try:
            operation_snapshot = self._preflight_authoritative_target(
                "Extract Selected Voxels start validation"
            )
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
            return
        extent = tuple(int(v) for v in fs.mask_extent)
        mask = np.array(
            _image_array_view(fs.mask), copy=True, dtype=np.uint8
        )
        target = self.target_vc
        source = target.image

        def work(report):
            report(10, "Extract: cropping")
            crop = vtk.vtkImageClip()
            crop.SetInputData(source)
            crop.ClipDataOn()
            crop.SetOutputWholeExtent(*extent)
            crop.Update()
            report(65, "Extract: applying mask")
            output = vtk.vtkImageData()
            output.DeepCopy(crop.GetOutput())
            arr = _image_array_view(output)
            arr[mask == 0] = 0
            output.GetPointData().GetScalars().Modified()
            output.Modified()
            report(95, "Extract: finalizing")
            return output

        output = self._run_output_background(
            "Extract Selected Voxels",
            work,
            operation_snapshot=operation_snapshot,
        )
        if output is None:
            return
        name = self._derived_name("selection")
        try:
            self._add_derived_volume(
                output,
                name,
                mask_volume=False,
                operation="segmentation_extract",
                operation_snapshot=operation_snapshot,
            )
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
            return
        self.set_status(f"Extracted selected intensity volume: {name}")

    def delete_selected(self):
        fs = self._output_mask()
        if fs is None:
            return
        try:
            operation_snapshot = self._preflight_authoritative_target(
                "Delete Selected start validation"
            )
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
            return
        target = self.target_vc
        source = target.image
        extent = tuple(int(v) for v in fs.mask_extent)
        mask = np.array(
            _image_array_view(fs.mask), copy=True, dtype=np.uint8
        )

        def work(report):
            report(10, "Delete Selected: copying")
            output = vtk.vtkImageData()
            output.DeepCopy(source)
            report(65, "Delete Selected: applying")
            arr = _image_array_view(output)
            roi = arr[extent_slices_zyx(extent, output.GetExtent())]
            roi[mask != 0] = 0
            output.GetPointData().GetScalars().Modified()
            output.Modified()
            report(95, "Delete Selected: finalizing")
            return output

        output = self._run_output_background(
            "Delete Selected",
            work,
            operation_snapshot=operation_snapshot,
        )
        if output is None:
            return
        name = self._derived_name("unselected")
        try:
            self._add_derived_volume(
                output,
                name,
                mask_volume=False,
                operation="segmentation_delete_selected",
                operation_snapshot=operation_snapshot,
            )
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
            return
        self.set_status(f"Created volume with selected voxels removed: {name}")

    def _derived_name(self, suffix):
        base = self.target_item.text(0) if self.target_item is not None else "volume"
        frame_suffix = f"_frame{self.current_frame_index()+1}" if getattr(self.target_vc, "has_time_series", lambda: False)() else ""
        # Generated IDs, not labels, own scientific identity. Repeated display
        # labels are preferable to scanning an arbitrarily large sibling set.
        return f"{base}_{suffix}{frame_suffix}"

    def _add_derived_volume(
        self,
        image,
        display_name,
        mask_volume=False,
        operation="segmentation",
        operation_parameters=None,
        operation_snapshot=None,
    ):
        if image is None:
            raise RuntimeError("No output image was produced.")
        if self.target_item is None or self.target_vc is None:
            raise RuntimeError("Generated output has no active parent volume.")
        validated_parent = self._preflight_authoritative_target(
            "Generated output publication validation"
        )
        if operation_snapshot is not None:
            self._require_compatible_target_snapshot(
                operation_snapshot,
                validated_parent,
                stage="Generated output publication validation",
                require_same_frame=True,
            )

        relationship_controller = getattr(
            self.main, "volume_relationship_controller", None
        )
        model = getattr(self.main, "volume_source_model", None)
        if relationship_controller is None or not isinstance(model, VolumeSourceModel):
            raise RuntimeError(
                "Generated output cannot be published without the authoritative volume model."
            )
        parent_acquisition_id = validated_parent.acquisition_id
        parent_channel_id = validated_parent.channel_id
        if (
            str(self.target_item.data(0, ROLE_VOLUME_SOURCE_ID) or "").strip()
            != parent_acquisition_id
            or str(
                self.target_item.data(0, ROLE_VOLUME_CHANNEL_ID) or ""
            ).strip()
            != parent_channel_id
        ):
            raise SegmentationPreflightError(
                "Generated output publication validation: the target tree row "
                "identity disagrees with the authoritative snapshot."
            )
        parent = model.acquisitions.get(parent_acquisition_id)
        parent_channel = model.channels.get(parent_channel_id)
        if (
            parent is None
            or parent_channel is None
            or parent_channel.source_id != parent_acquisition_id
        ):
            raise VolumeSourceError(
                "The segmentation target has no complete authoritative parent identity."
            )

        parent_local = parent_channel.local_geometry
        if (
            parent_local is None
            or parent_local.working_grid is None
            or parent_local.geometry_revision != validated_parent.geometry_revision
        ):
            raise SegmentationPreflightError(
                "Generated output publication validation: the parent channel "
                "geometry revision changed before publication."
            )
        working_parent = parent_local.working_grid
        if (
            tuple(working_parent.dimensions) != validated_parent.dimensions
            or tuple(working_parent.spacing) != validated_parent.spacing
            or tuple(working_parent.origin) != validated_parent.origin
            or tuple(tuple(row) for row in working_parent.direction)
            != validated_parent.direction
            or str(working_parent.source_coordinate_space_id)
            != validated_parent.coordinate_space_id
            or not np.allclose(
                np.asarray(working_parent.local_index_to_working_affine),
                np.asarray(validated_parent.local_index_affine),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise SegmentationPreflightError(
                "Generated output publication validation: the parent index "
                "geometry changed before publication."
            )

        normalized_image, runtime_parent_extent = _normalized_generated_image(image)
        parent_extent = extent_relative_to_parent(
            runtime_parent_extent, validated_parent.extent
        )
        target_matrix = _numpy_matrix4(self.target_vc.volume.GetMatrix())
        if not np.allclose(
            target_matrix,
            np.asarray(validated_parent.pose, dtype=float),
            rtol=0.0,
            atol=1e-12,
        ) or not np.allclose(
            np.asarray(parent.shared_pose, dtype=float),
            np.asarray(validated_parent.pose, dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise SegmentationPreflightError(
                "Generated output publication validation: the target pose "
                "disagrees with the validated acquisition pose."
            )
        output_scalars = normalized_image.GetPointData().GetScalars()
        output_array = vtk_to_numpy(output_scalars) if output_scalars is not None else None
        scalar_dtype = (
            np.dtype(output_array.dtype).name if output_array is not None else None
        )
        scalar_bit_depth = (
            int(output_array.dtype.itemsize * 8) if output_array is not None else None
        )
        parent_backing = model.backing_sources.get(parent_channel.backing_source_id)
        source_checksum = validated_parent.source_checksum
        if source_checksum is None and parent_backing is not None:
            source_checksum = parent_backing.source_checksum
        generated_checksum = _vtk_scalar_checksum(normalized_image)
        generation_parameters = {
            "signal_threshold": float(self.signal_threshold),
            "upper_threshold": float(self.upper_threshold),
            "seed_radius_voxels": int(self.settings["seed_radius"]),
            "growth_radius_voxels": int(self.settings["growth_radius"]),
            "mask_extent": list(parent_extent),
            "runtime_mask_extent": list(runtime_parent_extent),
        }
        generation_parameters.update(copy.deepcopy(dict(operation_parameters or {})))
        generation_parameters["validated_parent_reference"] = (
            validated_parent.operation_reference_payload()
        )
        state = getattr(self, "state", None)

        def session_reference(value):
            value = dict(value or {})
            if not value:
                return None
            return {
                key: copy.deepcopy(value.get(key))
                for key in (
                    "acquisition_id",
                    "channel_id",
                    "backing_source_id",
                    "source_frame_index",
                    "channel_geometry_revision",
                    "acquisition_geometry_revision",
                )
            }

        generation_parameters["segmentation_session_geometry"] = {
            "start": session_reference(
                getattr(state, "session_geometry_start", None)
            ),
            "current": session_reference(
                getattr(state, "session_geometry_current", None)
            ),
            "refresh_count": int(
                getattr(state, "session_geometry_refresh_count", 0) or 0
            ),
        }
        acquisition_id = new_volume_identity()
        channel_id = new_volume_identity()
        generated_data_id = new_volume_identity()
        candidate = VolumeSourceModel()
        model._seed_append_parent_chain(candidate, parent_acquisition_id)
        candidate.add_generated_acquisition(
            parent_acquisition_id=parent_acquisition_id,
            parent_channel_id=parent_channel_id,
            parent_extent=parent_extent,
            source_frame_index=int(validated_parent.frame_index),
            display_name=display_name,
            operation=str(operation or "segmentation"),
            parameters=generation_parameters,
            acquisition_id=acquisition_id,
            channel_id=channel_id,
            generated_data_id=generated_data_id,
            source_checksum=source_checksum,
            generated_checksum=generated_checksum,
            scalar_dtype=scalar_dtype,
            scalar_bit_depth=scalar_bit_depth,
            channel_role=(
                CHANNEL_ROLE_LABEL_MASK if mask_volume else CHANNEL_ROLE_OTHER
            ),
            software_version=MADI3D_VERSION,
            import_version=MADI3D_VERSION,
        )
        child = candidate.acquisitions[acquisition_id]
        child_channel = candidate.channels[channel_id]
        working_grid = child.working_grid()
        if (
            child.generated_lineage.parent_geometry_revision
            != validated_parent.geometry_revision
            or not np.allclose(
                np.asarray(child.shared_pose, dtype=float),
                np.asarray(validated_parent.pose, dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise RuntimeError(
                "Generated output did not inherit the validated parent geometry and pose."
            )

        target_meta = getattr(self.target_vc, "metadata", {}) or {}

        producer = vtk.vtkTrivialProducer()
        producer.SetOutput(normalized_image)
        scalar_range = normalized_image.GetScalarRange()
        vc = self.VolumeContainer(
            normalized_image, display_name, None, scalar_range=scalar_range
        )
        vc.reader = producer
        if mask_volume:
            vc.metadata.update(
                {
                    "data_min": 0.0,
                    "data_max": 1.0,
                    "lower_threshold": 0.5,
                    "upper_threshold": 1.0,
                    "saturation_point": 1.0,
                    "peak": 1.0,
                    "color": tuple(float(v) for v in self.settings["mask_color"]),
                    "color_lut": "solid",
                    "global_opacity": 1.0,
                    "representation": "volume",
                }
            )
        else:
            copied = serialize_volume_rendering_metadata(target_meta)
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
        if mask_volume:
            vc.transfer["opacity"].RemoveAllPoints()
            vc.transfer["opacity"].AddPoint(0.0, 0.0)
            vc.transfer["opacity"].AddPoint(0.5, 0.0)
            vc.transfer["opacity"].AddPoint(1.0, 0.85)
            vc.transfer["color"].RemoveAllPoints()
            vc.transfer["color"].AddRGBPoint(0.0, 0.0, 0.0, 0.0)
            vc.transfer["color"].AddRGBPoint(1.0, *self.settings["mask_color"])
            vc.prop.SetInterpolationTypeToNearest()
            vc.prop.ShadeOff()

        apply_exact = getattr(self.main, "apply_volume_actor_matrix_exact", None)
        if callable(apply_exact):
            apply_exact(vc, validated_parent.pose, scale_mode="center")
        else:
            vc.volume.SetUserMatrix(_vtk_matrix4(validated_parent.pose))
            vc.volume.Modified()
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
        item.setCheckState(0, self.target_item.checkState(0))

        append_scene_state = relationship_controller.capture_append_scene_state()
        old_volume_id_counter = int(self.main.volume_id_counter)
        vol_id = old_volume_id_counter
        item.setData(0, ROLE_VOLUME_ID, vol_id)
        published_to_renderer = False
        append_result = None
        try:
            parent_item = self.target_item.parent()
            if (
                parent_item is not None
                and parent_item.data(0, ROLE_GROUP_KIND) == GROUP_KIND_MULTICHANNEL
            ):
                anchor = parent_item
                insertion_parent = parent_item.parent()
            else:
                anchor = self.target_item
                insertion_parent = self.target_item.parent()
            # Suppress item callbacks, but let the Qt model notify the live view.
            tree_blocker = QtCore.QSignalBlocker(self.main.tree)
            try:
                if insertion_parent is not None:
                    index = insertion_parent.indexOfChild(anchor)
                    insertion_parent.insertChild(
                        index + 1 if index >= 0 else insertion_parent.childCount(),
                        item,
                    )
                else:
                    index = self.main.tree.indexOfTopLevelItem(anchor)
                    self.main.tree.insertTopLevelItem(
                        index + 1
                        if index >= 0
                        else self.main.tree.topLevelItemCount(),
                        item,
                    )
            finally:
                del tree_blocker

            self.main.volume_id_counter = old_volume_id_counter + 1
            self.main.volume_map[vol_id] = vc
            tree_blocker = QtCore.QSignalBlocker(self.main.tree)
            try:
                append_result = relationship_controller.append_typed_acquisition(
                    child,
                    (child_channel,),
                    (candidate.backing_sources[child_channel.backing_source_id],),
                    channel_items={child_channel.channel_id: item},
                )
                vc.source_id = acquisition_id
                vc.metadata.update(dict(item.data(0, ROLE_VOLUME_META) or {}))
                self.main.update_item_color_icon(
                    item, *color, icon_size=16, shape="disc"
                )
            finally:
                del tree_blocker

            bounds = self.render_window._volume_world_bounds(vc)
            vc.bbox_center_world = (
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            )
            try:
                should_show = bool(self.main._item_effectively_checked(item))
            except Exception:
                should_show = item.checkState(0) == Qt.CheckState.Checked
            if should_show:
                if not self.render_window.set_volume_visibility(
                    vc, True, render=False
                ):
                    raise RuntimeError(
                        "No volume-renderer port is available for the generated output."
                    )
                published_to_renderer = True
            schedule_scale_bar = getattr(
                self.main, "_schedule_scale_bar_update", None
            )
            if callable(schedule_scale_bar):
                schedule_scale_bar()
            self.render_window.render()
            self.main.mark_project_modified("generated volume publication")
        except Exception:
            if published_to_renderer or getattr(vc, "volume_index", None) is not None:
                remove = getattr(
                    self.main, "_remove_volume_container_without_refresh", None
                )
                if callable(remove):
                    remove(vc)
                else:
                    try:
                        self.render_window.set_volume_visibility(
                            vc, False, render=False
                        )
                    except Exception:
                        pass
            self.main.volume_map.pop(vol_id, None)
            self.main.volume_id_counter = old_volume_id_counter
            if append_result is not None:
                append_result.delta.rollback(model)
            tree_blocker = QtCore.QSignalBlocker(self.main.tree)
            try:
                item_parent = item.parent()
                if item_parent is not None:
                    item_index = item_parent.indexOfChild(item)
                    if item_index >= 0:
                        item_parent.takeChild(item_index)
                else:
                    item_index = self.main.tree.indexOfTopLevelItem(item)
                    if item_index >= 0:
                        self.main.tree.takeTopLevelItem(item_index)
            finally:
                del tree_blocker
            append_scene_state.restore()
            try:
                self.render_window.render()
            except Exception:
                pass
            raise
        return vc, item


    def _start_job(self, function):
        job_id = self._next_job()
        worker = _SegmentationJob(job_id, function)
        self._worker_refs[job_id] = worker
        worker.signals.progress.connect(self._job_progress)
        worker.signals.finished.connect(self._job_finished)
        worker.signals.failed.connect(self._job_failed)
        self._pool.start(worker)
        return job_id

    @QtCore.Slot(int, object)
    def _job_finished(self, job_id, result):
        job_id = int(job_id)
        self._worker_refs.pop(job_id, None)
        if job_id == self._blocking_job:
            return
        if job_id == self._speck_job:
            self._speck_job = None
            self._handle_speck_result(result)

    @QtCore.Slot(int, str, str)
    def _job_failed(self, job_id, message, details):
        job_id = int(job_id)
        self._worker_refs.pop(job_id, None)
        print(f"[VolumeSegmentation] Worker failed: {message}")
        if details:
            print(details)
        if job_id == self._blocking_job:
            return
        if job_id == self._speck_job:
            self._speck_job = None
            self._speck_pending = None
            self.panel.set_speck_busy(False)
            self._reset_speck_adjustment()
            self._update_history_ui()
            self.set_status(f"Remove Specks failed: {message}")
            self._drain_target_sync()

    @staticmethod
    def _pump_gui_events(max_ms=15):
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, int(max_ms))

    def _run_background(self, label, function, *, operation_snapshot=None):
        if self._speck_job is not None:
            raise RuntimeError("Remove specks is still updating the selection.")
        if self._blocking_job is not None:
            raise RuntimeError("Another segmentation operation is still running.")
        operation_snapshot = operation_snapshot or self._preflight_authoritative_target(
            f"{label} start validation"
        )
        preserve_threshold = bool(self._threshold_recompute_active and label == "Diffuse")
        self.panel.set_background_busy(True, label, preserve_threshold=preserve_threshold)
        job_id = int(self._start_job(function))
        self._blocking_job = job_id
        worker = self._worker_refs.get(job_id)
        if worker is None:
            self._blocking_job = None
            self.panel.set_background_busy(False)
            raise RuntimeError("Could not start segmentation background worker.")
        try:
            while not worker.done_event.wait(0.010):
                self._pump_gui_events(15)
            self._pump_gui_events(15)
            if worker.error is not None:
                raise RuntimeError(worker.error)
            self._accept_background_result(operation_snapshot, label)
            self.panel.set_progress(100, f"{label}: complete")
            self._pump_gui_events(5)
            return worker.result
        finally:
            self._blocking_job = None
            self.panel.set_background_busy(False)
            self._update_history_ui()
            self._drain_target_sync()
            self._schedule_pending_exact_threshold()

    def _compute_diffuse_stroke(
        self,
        seed_geometry,
        growth_geometry,
        snap,
        frame_state,
        frame_index,
        threshold_value,
        *,
        clipping_plane_coeffs=(),
        smart_local_threshold=False,
        smart_faint_recovery=False,
        smart_boundary_guard=False,
        smart_visible_seeds=False,
        visible_ijk_to_clip=None,
        visible_depth_coeff=None,
        visible_viewport_px=None,
    ):
        """Rasterize and solve an initial Diffuse stroke inside the background job."""
        source_extent = tuple(int(v) for v in snap["extent"])

        def work(report):
            report(5, "Diffuse: rasterizing brush")
            seed_stencil, seed_extent = self._rasterize_stroke_geometry(
                seed_geometry, source_extent
            )
            growth_stencil, growth_extent = self._rasterize_stroke_geometry(
                growth_geometry, source_extent
            )
            if seed_stencil is None or seed_extent is None:
                return None

            operation_extent = extent_intersection(
                extent_union(seed_extent, growth_extent), source_extent
            )
            if operation_extent is None:
                return None

            report(25, "Diffuse: clipping brush")
            inside_clip = self._clipping_inside_mask(
                operation_extent,
                snap["vtk_ijk_to_world_affine"],
                plane_coeffs=clipping_plane_coeffs,
            )
            seed_stencil, _inside_clip = self._restrict_stencil_to_clipping(
                seed_stencil, operation_extent, snap, inside_mask=inside_clip
            )
            growth_stencil, _inside_clip = self._restrict_stencil_to_clipping(
                growth_stencil, operation_extent, snap, inside_mask=inside_clip
            )

            report(40, "Diffuse: preparing domains")
            old_patch = self._mask_patch(frame_state, operation_extent)
            seed_domain = _stencil_to_bool(seed_stencil, operation_extent)
            growth_domain = _stencil_to_bool(growth_stencil, operation_extent)

            effective_threshold = float(threshold_value)
            used_local_threshold = False
            if smart_local_threshold:
                local_threshold = self._estimate_local_threshold(
                    operation_extent, seed_domain
                )
                if local_threshold is not None:
                    effective_threshold = float(local_threshold)
                    used_local_threshold = True

            report(55, "Diffuse: threshold/connectivity")
            new_patch, component_count = self._compute_diffuse_patch(
                operation_extent,
                old_patch,
                seed_domain,
                growth_domain,
                effective_threshold,
                growth_stencil=growth_stencil,
                smart_faint_recovery=smart_faint_recovery,
                smart_boundary_guard=smart_boundary_guard,
                smart_visible_seeds=smart_visible_seeds,
                visible_ijk_to_clip=visible_ijk_to_clip,
                visible_depth_coeff=visible_depth_coeff,
                visible_viewport_px=visible_viewport_px,
            )
            report(95, "Diffuse: finalizing")
            return {
                "operation_extent": operation_extent,
                "old_patch": old_patch,
                "new_patch": new_patch,
                "seed_domain": seed_domain,
                "growth_domain": growth_domain,
                "threshold_value": effective_threshold,
                "used_local_threshold": used_local_threshold,
                "component_count": component_count,
            }

        return self._run_background(
            "Diffuse",
            work,
            operation_snapshot=snap.get("_target_context"),
        )

    def _compute_diffuse_new_patch(self, *args, **kwargs):
        def work(report):
            report(10, "Diffuse: threshold/connectivity")
            result = self._compute_diffuse_patch(*args, **kwargs)
            report(95, "Diffuse: finalizing")
            return result

        return self._run_background("Diffuse", work)

    def _target_locked(self):
        return bool(self._blocking_job is not None or self._speck_job is not None)
    def _drain_target_sync(self):
        if self._target_locked() or not self._deferred_target_sync:
            return
        self._deferred_target_sync = False
        self.sync_target_to_latest_selection()
    def _schedule_pending_exact_threshold(self):
        if self._blocking_job is not None or self._threshold_drain_scheduled:
            return
        pending = self._pending_exact_threshold
        if pending is None:
            return
        self._pending_exact_threshold = None
        self._threshold_drain_scheduled = True
        def drain():
            self._threshold_drain_scheduled = False
            self.set_threshold(pending)
        QtCore.QTimer.singleShot(0, drain)
    def _next_job(self):
        self._job_serial += 1
        return int(self._job_serial)
    @QtCore.Slot(int, int, str)
    def _job_progress(self, job_id, value, text):
        if int(job_id) == self._blocking_job:
            self.panel.set_progress(value, text)
    @staticmethod
    def _filter_specks(mask, extent, minimum):
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
        filt.Update()
        return (
            np.asarray(
                _image_array_view(filt.GetOutput()), dtype=np.uint8
            ) > 0
        ).astype(np.uint8)

    def _reset_speck_adjustment(self):
        self._speck_generation += 1
        self._speck_target_vc = None
        self._speck_extent = None
        self._speck_frame = None
        self._speck_base = None
        self._speck_history_anchor = None
        self._speck_history_depth = 0
        self._speck_history_entry = None
        self._speck_pending = None
        self._speck_context = None

    def _speck_baseline_matches(self, frame_state):
        if (
            self._speck_base is None
            or self._speck_target_vc is not self.target_vc
            or int(self._speck_frame if self._speck_frame is not None else -1)
               != int(self.current_frame_index())
            or tuple(self._speck_extent or ()) != tuple(frame_state.mask_extent or ())
        ):
            return False
        if self._speck_history_entry is not None:
            return bool(
                frame_state.undo_stack
                and frame_state.undo_stack[-1] is self._speck_history_entry
            )
        if len(frame_state.undo_stack) != int(self._speck_history_depth):
            return False
        if self._speck_history_anchor is None:
            return not frame_state.undo_stack
        return bool(
            frame_state.undo_stack
            and frame_state.undo_stack[-1] is self._speck_history_anchor
        )

    def _start_speck_adjustment(self, frame_state):
        self._speck_generation += 1
        self._speck_target_vc = self.target_vc
        self._speck_frame = int(self.current_frame_index())
        self._speck_extent = tuple(int(v) for v in frame_state.mask_extent)
        self._speck_base = np.array(
            _image_array_view(frame_state.mask), copy=True, dtype=np.uint8
        )
        self._speck_history_depth = len(frame_state.undo_stack)
        self._speck_history_anchor = (
            frame_state.undo_stack[-1] if frame_state.undo_stack else None
        )
        self._speck_history_entry = None
        self._speck_pending = None

    @staticmethod
    def _replace_speck_history_entry(
        frame_state, extent, base, filtered, previous_entry, history_limit
    ):
        base = np.ascontiguousarray(np.asarray(base, dtype=np.uint8))
        filtered = np.ascontiguousarray(np.asarray(filtered, dtype=np.uint8))
        previous_is_current = bool(
            previous_entry is not None
            and frame_state.undo_stack
            and frame_state.undo_stack[-1] is previous_entry
        )
        changed = not np.array_equal(base, filtered)
        if not changed:
            if previous_is_current:
                frame_state.undo_stack.pop()
            return None

        entry = HistoryEntry.make(extent, base, filtered, mode="specks")
        if previous_is_current:
            frame_state.undo_stack[-1] = entry
        else:
            frame_state.undo_stack.append(entry)
            limit = max(1, int(history_limit))
            if len(frame_state.undo_stack) > limit:
                del frame_state.undo_stack[:len(frame_state.undo_stack)-limit]
            frame_state.redo_stack.clear()
        return entry

    def remove_specks(self, minimum):
        minimum = max(1, int(minimum))
        if not self.active:
            return
        if self._blocking_job is not None:
            self.set_status("Finish the current segmentation operation before changing Remove specks.")
            return
        try:
            self._preflight_authoritative_target(
                "Remove Specks start validation"
            )
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
            return
        fs = self._frame_state()
        if fs is None or fs.mask is None or not np.any(
            _image_array_view(fs.mask)
        ):
            self.set_status("Remove specks: the current frame has no selected voxels.")
            return

        if not self._speck_baseline_matches(fs):
            self._start_speck_adjustment(fs)
        if self._speck_job is not None:
            self._speck_pending = minimum
            return

        self._launch_speck_job(minimum)

    def _launch_speck_job(self, minimum):
        if self._speck_base is None:
            return
        try:
            self._speck_context = self._preflight_authoritative_target(
                "Remove Specks start validation"
            )
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
            return
        minimum = max(1, int(minimum))
        generation = int(self._speck_generation)
        base = self._speck_base
        extent = self._speck_extent
        def work(_report):
            filtered = self._filter_specks(base, extent, minimum)
            return {
                "generation": generation,
                "minimum": minimum,
                "mask": filtered,
                "before": int(np.count_nonzero(base)),
                "after": int(np.count_nonzero(filtered)),
            }
        self.panel.set_speck_busy(True)
        self._speck_job = self._start_job(work)

    def _handle_speck_result(self, result):
        if int(result["generation"]) != int(self._speck_generation):
            self.panel.set_speck_busy(False)
            self._update_history_ui()
            self._drain_target_sync()
            return
        try:
            if self._speck_context is None:
                raise SegmentationPreflightError(
                    "Remove Specks result validation: the operation target snapshot is missing."
                )
            self._accept_background_result(
                self._speck_context, "Remove Specks"
            )
        except SegmentationPreflightError as exc:
            self.panel.set_speck_busy(False)
            self._reset_speck_adjustment()
            self._update_history_ui()
            self._drain_target_sync()
            self._stop_for_authoritative_error(exc)
            return

        pending = self._speck_pending
        self._speck_pending = None
        if pending is not None and int(pending) != int(result["minimum"]):
            self._launch_speck_job(int(pending))
            return

        fs = self._frame_state(self._speck_frame)
        if (
            fs is None
            or self._speck_target_vc is not self.target_vc
            or int(self.current_frame_index()) != int(self._speck_frame)
            or tuple(fs.mask_extent or ()) != tuple(self._speck_extent)
            or not self._speck_baseline_matches(fs)
        ):
            self.panel.set_speck_busy(False)
            self._reset_speck_adjustment()
            self._update_history_ui()
            self._drain_target_sync()
            return

        filtered = np.array(result["mask"], copy=True, dtype=np.uint8)
        self._write_mask_patch(fs, self._speck_extent, filtered)
        self._speck_history_entry = self._replace_speck_history_entry(
            fs,
            self._speck_extent,
            self._speck_base,
            filtered,
            self._speck_history_entry,
            self.settings.get("history_limit", 40),
        )
        self._sync_current_frame_preview(render=True)
        self.panel.set_speck_busy(False)
        self._update_history_ui()
        removed = int(result["before"]) - int(result["after"])
        self.set_status(
            f"Remove specks: minimum {int(result['minimum'])} voxels; "
            f"{max(0, removed):,} voxels removed from the whole selection."
        )
        self._drain_target_sync()

    def _output_mask(self):
        if not self.active:
            return None
        fs = self._frame_state()
        if fs is None or fs.mask is None or not np.any(
            _image_array_view(fs.mask)
        ):
            QtWidgets.QMessageBox.information(
                self.main, "Volume Segmentation",
                "The current frame has no selected voxels."
            )
            return None
        return fs
    def _run_output_background(
        self, label, function, *, operation_snapshot=None
    ):
        try:
            return self._run_background(
                label,
                function,
                operation_snapshot=operation_snapshot,
            )
        except SegmentationPreflightError as exc:
            print(f"[VolumeSegmentation] {label} discarded: {exc}")
            self._stop_for_authoritative_error(exc)
            return None
        except Exception as exc:
            print(f"[VolumeSegmentation] {label} failed: {exc}")
            self.set_status(f"{label} failed: {exc}")
            QtWidgets.QMessageBox.warning(
                self.main, "Volume Segmentation", f"{label} failed:\n{exc}"
            )
            return None
    @staticmethod
    def _dilate_roi(mask, extent, source, margin):
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
        filt.Update()
        return np.ascontiguousarray(
            _image_array_view(filt.GetOutput()), dtype=np.uint8
        )
    def extract_original_signal(self):
        fs = self._output_mask()
        if fs is None:
            return
        try:
            operation_snapshot = self._preflight_authoritative_target(
                "Extract Original Signal start validation"
            )
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
            return
        margin = max(0, int(self.panel.extract_margin.value()))
        target = self.target_vc
        source = target.image
        source_extent = _extent_tuple(source.GetExtent())
        mask_extent = tuple(int(v) for v in fs.mask_extent)
        mask = np.array(
            _image_array_view(fs.mask), copy=True, dtype=np.uint8
        )
        out_extent = extent_expand(
            mask_extent, margin, limit=source_extent
        )

        def work(report):
            report(10, "Original signal: ROI")
            roi = np.zeros(
                extent_shape_zyx(out_extent), dtype=np.uint8
            )
            roi[extent_slices_zyx(mask_extent, out_extent)] = mask
            roi = self._dilate_roi(roi, out_extent, source, margin)
            report(45, "Original signal: cropping")
            crop = vtk.vtkImageClip()
            crop.SetInputData(source)
            crop.ClipDataOn()
            crop.SetOutputWholeExtent(*out_extent)
            crop.Update()
            output = vtk.vtkImageData()
            output.DeepCopy(crop.GetOutput())
            arr = _image_array_view(output)
            if arr.shape != roi.shape:
                raise RuntimeError("Original-signal ROI geometry mismatch.")
            arr[roi == 0] = 0
            output.GetPointData().GetScalars().Modified()
            output.Modified()
            report(95, "Original signal: finalizing")
            return output

        output = self._run_output_background(
            "Extract Original Signal",
            work,
            operation_snapshot=operation_snapshot,
        )
        if output is None:
            return
        suffix = "original_signal" if margin == 0 else f"original_signal_m{margin}"
        name = self._derived_name(suffix)
        try:
            self._add_derived_volume(
                output, name, mask_volume=False,
                operation="segmentation_extract_original_signal",
                operation_parameters={"margin_voxels": int(margin)},
                operation_snapshot=operation_snapshot,
            )
        except SegmentationPreflightError as exc:
            self._stop_for_authoritative_error(exc)
            return
        self.set_status(
            f"Extracted original signal with {margin}-voxel 3-D margin: {name}"
        )

__all__ = [
    "VolumeSegmentationPanel",
    "VolumeSegmentationController",
    "VolumeSegmentationInteractorStyle",
    "SegmentationBrushOverlay",
    "SegmentationState",
    "SegmentationFrameState",
]
