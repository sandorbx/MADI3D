"""Qt worker for explicit NeuronBridge asset retrieval."""
from __future__ import annotations
from threading import Event

from PySide6 import QtCore

from .assets import fetch_source, make_nb_client


class NBDownloadThread(QtCore.QThread):
    file_ready = QtCore.Signal(object, object)
    source_resolved = QtCore.Signal(object, object)
    progress = QtCore.Signal(int)
    error = QtCore.Signal(object, object)

    def __init__(self, jobs, parent=None, *, client_factory=make_nb_client, fetch=fetch_source):
        super().__init__(parent)
        self.jobs = tuple(jobs)
        self.client_factory = client_factory
        self.fetch = fetch
        self.cancel_event = Event()

    def requestInterruption(self):
        self.cancel_event.set()
        super().requestInterruption()

    def run(self):
        clients = {}

        def client(version):
            if version not in clients:
                clients[version] = (make_nb_client(version, cancel=self.cancel_event)
                                    if self.client_factory is make_nb_client else self.client_factory(version))
            return clients[version]

        try:
            self._retrieve(client)
        finally:
            for value in clients.values():
                close = getattr(value, "close", None)
                if close is not None:
                    close()

    def _retrieve(self, client):
        for index, (token, remote) in enumerate(self.jobs):
            if self.isInterruptionRequested():
                break
            try:
                resolved = self.fetch(
                    remote, client, cancel_check=self.isInterruptionRequested,
                    on_resolved=lambda value: self.source_resolved.emit(token, value),
                )
                if self.isInterruptionRequested():
                    break
                self.file_ready.emit(token, resolved)
            except InterruptedError:
                break
            except Exception as exc:
                self.error.emit(token, exc)
            self.progress.emit(int(100 * (index + 1) / len(self.jobs)))
