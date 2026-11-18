"""Bounded URL rules and shared helpers for the public Janelia S3 objects.

This module owns the supported public bucket URL grammar, the wakeable
cancellation event and the small shared helpers (hashing, timestamps, strict
JSON, canonical encoding). The normal local-search route's available libraries
are the pinned distributions in ``archive_catalog``, installed by
``archive_install`` through its own Range-capable transport and read by
``library_store``; the dynamic public catalog route and its snapshot download
transport have been removed. The public Online Lookup subsystem
(``public_api``) is separate.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock
from urllib.parse import unquote, urlsplit

from .local_scorer import checkpoint

DATA_BUCKET = "janelia-neuronbridge-data-prod"
IMAGE_BUCKET = "janelia-flylight-color-depth"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def parse_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key in public library evidence.")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    canonical(value)  # Reject non-finite numerical observations.
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path, cancel=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(1024 * 1024):
            checkpoint(cancel)
            h.update(data)
    return h.hexdigest()


def bulk_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port or
            parsed.query or parsed.fragment or "\\" in url):
        raise ValueError("Only direct public NeuronBridge S3 objects are supported.")
    allowed = {DATA_BUCKET, IMAGE_BUCKET}
    if parsed.hostname == "s3.amazonaws.com":
        bucket, _, key = parsed.path.lstrip("/").partition("/")
    else:
        bucket = next((b for b in allowed if parsed.hostname in
                       {f"{b}.s3.amazonaws.com", f"{b}.s3.us-east-1.amazonaws.com"}), "")
        key = parsed.path.lstrip("/")
    if bucket not in allowed or not key or any(p in (".", "..") for p in unquote(key).split("/")):
        raise ValueError("Object URL is outside the supported public data buckets.")
    return url


class CancellationEvent(Event):
    """Event that can also wake resources blocked outside Python checkpoints."""

    def __init__(self):
        super().__init__()
        self._callback_lock = Lock()
        self._callbacks = set()

    def set(self):
        with self._callback_lock:
            super().set()
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            callback()

    def add_callback(self, callback):
        with self._callback_lock:
            if not self.is_set():
                self._callbacks.add(callback)
                return
        callback()

    def remove_callback(self, callback):
        with self._callback_lock:
            self._callbacks.discard(callback)
