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

    def __init__(self, tiles, settings, mode, parent=None):
        super().__init__(parent)
        self._tiles = tiles
        self._settings = dict(settings)
        self._mode = str(mode)

    def run(self):
        StitchRegistrationOperation(
            self._tiles,
            self._settings,
            self._mode,
            progress_callback=self.progress.emit,
            cancelled=self.isInterruptionRequested,
            completed_callback=self.completed.emit,
            failed_callback=self.failed.emit,
        ).run()


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
