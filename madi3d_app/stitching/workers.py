"""Qt thread adapters for GUI-independent stitching operations."""
from __future__ import annotations

from PySide6 import QtCore

from madi3d_app.stitching.service import (
    StitchFusionOperation,
    StitchRegistrationOperation,
)


class StitchRegistrationWorker(QtCore.QThread):
    progress = QtCore.Signal(int, str)
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, tiles, settings, mode, parent=None, *, reuse_result=None):
        super().__init__(parent)
        self._tiles = tiles
        self._settings = dict(settings)
        self._mode = str(mode)
        self._reuse_result = reuse_result
        self.was_cancelled = False

    def run(self):
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
            failed_callback=self.failed.emit,
            reuse_result=self._reuse_result,
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
        }

    def run(self):
        StitchFusionOperation(
            **self._operation_arguments,
            progress_callback=self.progress.emit,
            cancelled=self.isInterruptionRequested,
            completed_callback=self.completed.emit,
            failed_callback=self.failed.emit,
        ).run()


__all__ = ["StitchFusionWorker", "StitchRegistrationWorker"]
