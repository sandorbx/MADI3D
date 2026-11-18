"""Qt thread adapters for GUI-independent stitching operations."""
from __future__ import annotations

import threading

from PySide6 import QtCore

from madi3d_app.scene.memory import MemoryAdmissionPending, MemoryDemand, MemoryEstimate
from madi3d_app.stitching.models import StitchingCapture
from madi3d_app.stitching.service import (
    StitchFusionOperation,
    StitchRegistrationOperation,
    _registration_admission_bytes,
)


class StitchRegistrationWorker(QtCore.QThread):
    progress = QtCore.Signal(int, str)
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    memory_requested = QtCore.Signal(object)

    def __init__(self, tiles: list[StitchingCapture], settings, mode, parent=None, *, reuse_result=None,
                 scientific_capture=None, resources=None, retained_input_bytes=None):
        super().__init__(parent)
        self._tiles = tiles
        self._settings = dict(settings)
        self._mode = str(mode)
        self._reuse_result = reuse_result
        self.scientific_capture = scientific_capture
        self.was_cancelled = False
        self._resources = resources
        self._retained_input_bytes = (_registration_admission_bytes(tiles)
                                      if retained_input_bytes is None else int(retained_input_bytes))
        self._admitted_working_bytes = None
        self.memory_requested.connect(self._admit_memory, QtCore.Qt.QueuedConnection)

    def _reserve_memory(self, working_bytes, policy_bytes, retained_bytes=0):
        stage = (working_bytes, retained_bytes)
        if self._resources is None or stage == self._admitted_working_bytes:
            return
        request = {"bytes": int(working_bytes), "policy_bytes": int(policy_bytes),
                   "working_total": int(working_bytes + retained_bytes),
                   "retained_bytes": self._retained_input_bytes + retained_bytes,
                   "done": threading.Event(), "error": None}
        self.memory_requested.emit(request)
        while not request["done"].wait(0.05):
            if self.isInterruptionRequested():
                raise InterruptedError("Stitching registration cancelled during admission.")
        if self.isInterruptionRequested():
            raise InterruptedError("Stitching registration cancelled during admission.")
        if request["error"] is not None:
            if request.get("cancelled"):
                self.was_cancelled = True
                raise InterruptedError(request["error"])
            raise MemoryError(request["error"])
        self._admitted_working_bytes = stage

    @QtCore.Slot(object)
    def _admit_memory(self, request):
        """The worker pauses; this QThread object's slot runs on the GUI thread."""
        deferred = False
        try:
            resources = self._resources
            if self.isInterruptionRequested() or not resources.publication_allowed:
                raise InterruptedError("The stitching memory request is no longer current.")
            current = resources.registry.memory.admit(resources.request).estimate
            future = MemoryDemand(request["bytes"], request["bytes"])
            # Captured pixels are now touched and held by this worker (and the
            # queue's other channels). A fresh sample may replace their hold;
            # future pair/graph allocations keep coverage throughout.
            policy_reasons = ()
            if request["working_total"] > request["policy_bytes"]:
                policy_reasons = (
                    f"Known stitching working buffers need {request['working_total'] / 1024**2:.1f} MiB, "
                    f"exceeding the configured {request['policy_bytes'] / 1024**2:.1f} MiB registration budget.",)
            resources.resize(MemoryEstimate(
                future, resident_bytes=0,
                basis="Stitching pair/pose-graph working storage; captured backing materialized; native scratch incomplete.",
                revision=current.revision + 1, unknown_scopes=("main",),
                policy_reasons=policy_reasons,
            ), materialized_bytes=request["retained_bytes"])
            if self.isInterruptionRequested() or not resources.publication_allowed:
                raise InterruptedError("The stitching memory request expired during admission.")
        except MemoryAdmissionPending:
            deferred = True
            delay = max(1, round(self._resources.registry.memory.policy.sample_cadence * 1000))
            QtCore.QTimer.singleShot(delay, self, lambda: self._admit_memory(request))
        except Exception as exc:
            request["error"] = f"{type(exc).__name__}: {exc}"
            request["cancelled"] = isinstance(exc, InterruptedError)
        finally:
            if not deferred:
                request["done"].set()

    def run(self):
        def failed(message):
            self.failed.emit(message)

        def cancelled():
            self.was_cancelled = self.was_cancelled or self.isInterruptionRequested()
            return self.was_cancelled
        StitchRegistrationOperation(
            self._tiles,
            self._settings,
            self._mode,
            progress_callback=self.progress.emit,
            cancelled=cancelled,
            completed_callback=self.completed.emit,
            failed_callback=failed,
            reuse_result=self._reuse_result,
            reserve_memory=self._reserve_memory if self._resources is not None else None,
        ).run()


class StitchingInputValidationWorker(QtCore.QThread):
    """Stream input hashes off the UI thread before consuming stored constraints."""

    def __init__(self, tiles, fingerprints, parent=None):
        super().__init__(parent)
        self._tiles = tiles
        self._fingerprints = dict(fingerprints)
        self.error = ""
        self.was_cancelled = False

    def run(self):
        from madi3d_app.stitching.discovery import verify_registration_pixels
        try:
            verify_registration_pixels(self._tiles, self._fingerprints, self.isInterruptionRequested)
        except InterruptedError:
            self.was_cancelled = True
        except Exception as exc:
            self.error = str(exc)


class StitchFusionWorker(QtCore.QThread):
    progress = QtCore.Signal(int, str)
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        channel_sets,
        output_dir,
        base_name,
        options,
        project_payload,
        writer_callback,
        bundle_writer_callback=None,
        ffmpeg_executable=None,
        parent=None,
        *,
        supporting_operations=(),
    ):
        super().__init__(parent)
        self._operation_arguments = {
            "channel_sets": channel_sets,
            "output_dir": output_dir,
            "base_name": base_name,
            "options": dict(options),
            "project_payload": dict(project_payload),
            "writer_callback": writer_callback,
            "bundle_writer_callback": bundle_writer_callback,
            "ffmpeg_executable": ffmpeg_executable,
            "supporting_operations": tuple(supporting_operations),
        }

    def run(self):
        def failed(message):
            self.failed.emit(message)
        def succeeded(result):
            self.completed.emit(result)

        StitchFusionOperation(
            **self._operation_arguments,
            progress_callback=self.progress.emit,
            cancelled=self.isInterruptionRequested,
            completed_callback=succeeded,
            failed_callback=failed,
        ).run()


__all__ = ["StitchFusionWorker", "StitchRegistrationWorker"]
