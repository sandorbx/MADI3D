"""NB-05 public precomputed metadata API. No Qt, credentials or asset I/O.

Protocol: neuronbridge-python 15a268c68a33983cbff6243fd99aa2efbed371ab.
The public protocol serves whole JSON objects, not a paginated search endpoint.
Unrecognized pagination is retained and marked incomplete, never guessed at.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import Event
from urllib.parse import quote, urlsplit
from uuid import uuid4

import requests
from urllib3.exceptions import HTTPError as Urllib3HTTPError

from .records import (AssetIdentity, BiologicalIdentity, ChannelSelection, Diagnostic,
                      ImageIdentity, MatchOccurrence, ResultField, SearchResults,
                      SearchSession, SourceIdentity, SourceParameter)

CLIENT_COMMIT = "15a268c68a33983cbff6243fd99aa2efbed371ab"
BASE_URL = "https://janelia-neuronbridge-data-prod.s3.us-east-1.amazonaws.com"
METHODS = {"CDM": "CDSResults", "PPPM": "PPPMResults"}
MAX_BYTES = 16 * 1024 * 1024
MAX_ROWS = 5000


class PublicAPIError(ValueError):
    pass


class PublicAPICancelled(PublicAPIError):
    pass


def checkpoint(cancel):
    if cancel is not None and cancel.is_set():
        raise PublicAPICancelled("Lookup cancelled. No incomplete snapshot was published.")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def public_url(url):
    """Only public S3 metadata hosts used by the verified data service."""
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port
            or parsed.query or parsed.fragment or parsed.hostname not in {
                "janelia-neuronbridge-data-prod.s3.us-east-1.amazonaws.com",
                "s3.amazonaws.com"}):
        raise PublicAPIError("The result reference is not a supported public metadata URL.")
    if parsed.hostname == "s3.amazonaws.com" and not parsed.path.startswith("/janelia-neuronbridge-data-prod/"):
        raise PublicAPIError("The result reference is outside the public metadata bucket.")
    return url


class PublicTransport:
    """One request at a time; bounded bytes, attempts, elapsed time and waits.

    Cancellation during a blocked socket is bounded by the read/connect timeout.
    No redirect following or ambient authentication (including .netrc).
    """
    def __init__(self, cancel=None, *, session=None, max_bytes=MAX_BYTES,
                 deadline_seconds=90, retries=2):
        self.cancel = cancel or Event()
        self.session = session or requests.Session()
        self.session.trust_env = False
        self.max_bytes = max_bytes
        self.deadline = time.monotonic() + deadline_seconds
        self.retries = retries
        self.bytes_received = 0

    def close(self):
        self.session.close()

    def _check(self):
        checkpoint(self.cancel)
        if time.monotonic() >= self.deadline:
            raise PublicAPIError("Public metadata retrieval exceeded its time limit. Retry explicitly.")

    def _wait(self, seconds):
        if seconds > 15 or time.monotonic() + seconds >= self.deadline:
            raise PublicAPIError("Server requested a longer retry delay. Try again later.")
        self.cancel.wait(seconds)
        self._check()

    def get(self, url, *, text=False):
        public_url(url)
        for attempt in range(self.retries + 1):
            self._check()
            delay = min(2 ** attempt, 4)
            try:
                remaining = max(.01, self.deadline - time.monotonic())
                with self.session.get(url, stream=True, timeout=(min(5, remaining), min(10, remaining)),
                                      allow_redirects=False, headers={"Accept-Encoding": "identity"}) as response:
                    status = response.status_code
                    if status in (429, 500, 502, 503, 504):
                        retry = response.headers.get("Retry-After")
                        if retry:
                            try:
                                delay = max(0, float(retry))
                            except ValueError:
                                try:
                                    delay = max(0, (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds())
                                except (ValueError, TypeError, OverflowError):
                                    pass
                        if not math.isfinite(delay):
                            raise PublicAPIError("Invalid server retry delay.")
                        if attempt == self.retries:
                            raise PublicAPIError(f"Public metadata service returned HTTP {status}; retry limit reached.")
                    elif status != 200:
                        raise PublicAPIError(f"Public metadata service returned HTTP {status}. Check the identifier and data version.")
                    else:
                        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                            raise PublicAPIError("Unexpected compressed metadata response; bounded identity transfer required.")
                        data = bytearray()
                        while True:
                            self._check()
                            # read1 returns available bytes without waiting for a
                            # full 64 KiB block; slow transfers still reach the
                            # deadline and cancellation checkpoints promptly.
                            chunk = response.raw.read1(64 * 1024, decode_content=False)
                            self._check()
                            if not chunk:
                                break
                            self.bytes_received += len(chunk)
                            if len(data) + len(chunk) > self.max_bytes or self.bytes_received > 2 * self.max_bytes:
                                raise PublicAPIError("Metadata exceeds the retrieval size limit; no complete snapshot was produced.")
                            data.extend(chunk)
                        self._check()
                        try:
                            decoded = data.decode("utf-8")
                            if text:
                                value = decoded.strip()
                            else:
                                def unique(pairs):
                                    result = {}
                                    for key, item in pairs:
                                        if key in result:
                                            raise ValueError("duplicate JSON key")
                                        result[key] = item
                                    return result
                                value = json.loads(decoded, object_pairs_hook=unique,
                                                   parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
                                canonical(value)
                        except (ValueError, UnicodeError, RecursionError) as exc:
                            raise PublicAPIError("Malformed public metadata response.") from exc
                        return {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "etag": response.headers.get("ETag"), "payload": value}
            except (requests.RequestException, Urllib3HTTPError) as exc:
                if attempt == self.retries:
                    raise PublicAPIError("Public metadata request failed after bounded retries. Cached snapshots remain available.") from exc
            self._wait(delay)
        raise AssertionError("unreachable")


def _text(value):
    # IDs remain opaque strings; never round a JSON float into an identifier.
    if isinstance(value, str):
        return value if value.strip() else None
    if type(value) is int:
        return str(value)
    if value is None:
        return None
    raise PublicAPIError("An image identifier or selector has an invalid type; it was not coerced.")


def file_url(image, key, config):
    files = image.get("files") or {}
    if not isinstance(files, dict):
        raise PublicAPIError("Image files must be an object.")
    path = files.get(key)
    if not isinstance(path, str) or not path:
        return None
    if urlsplit(path).scheme:
        return path
    prefix = config.get("stores", {}).get(files.get("store"), {}).get("prefixes", {}).get(key)
    return prefix + path if isinstance(prefix, str) else None


def image_channel(image):
    """Interpret the public LM channel convention, retaining unknown selectors.

    colormipsearch LMNeuronMetadata.channel is explicitly one-based:
    https://github.com/JaneliaSciComp/colormipsearch/blob/8b85c458f3b9231ec7bff5f1b4262cbf2abf8c3d/colormipsearch-api/src/main/java/org/janelia/colormipsearch/dto/LMNeuronMetadata.java
    Historical CSV selectors do not pass through this protocol adapter.
    """
    selector = _text(image.get("channel"))
    if selector is None:
        return None
    known = (image.get("type") == "LMImage" and selector.isascii()
             and selector.isdigit() and int(selector) > 0)
    return ChannelSelection(selector, index_base=1 if known else None)


def source_identity(image, config):
    if not isinstance(image, dict):
        raise PublicAPIError("A result image must be an object.")
    kind = {"EMImage": "em", "LMImage": "lm"}.get(_text(image.get("type")), "unknown")
    name = _text(image.get("publishedName"))
    assets = []
    for key in (("AlignedBodyOBJ", "AlignedBodySWC") if kind == "em" else
                ("VisuallyLosslessStack",) if kind == "lm" else ()):
        url = file_url(image, key, config)
        if url:
            assets.append(AssetIdentity(asset_id=image["files"][key], asset_type=key, url=url))
    return SourceIdentity(kind=kind, library=_text(image.get("libraryName")),
        library_release=_text(image.get("libraryRelease")), published_name=name,
        biological=BiologicalIdentity(neuron_id=(_text(image.get("bodyId")) or _text(image.get("neuronId"))) if kind == "em" else None,
            line_name=name if kind == "lm" else None, neuron_type=_text(image.get("neuronType")),
            neuron_instance=_text(image.get("neuronInstance"))),
        image=ImageIdentity(image_id=_text(image.get("id")), slide_code=_text(image.get("slideCode")),
            alignment_space=_text(image.get("alignmentSpace")), sex=_text(image.get("gender")),
            magnification=_text(image.get("objective")), anatomical_area=_text(image.get("anatomicalArea")),
            mounting_protocol=_text(image.get("mountingProtocol"))), assets=tuple(assets),
        channel=image_channel(image))


def collection(payload, limit):
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise PublicAPIError("Public metadata must contain a results array.")
    rows = payload["results"]
    reasons = []
    # No next-page contract exists in the pinned protocol. Retain unfamiliar
    # controls and refuse to describe them as a fully retrieved collection.
    controls = {k: v for k, v in payload.items() if k not in {"results", "inputImage"}}
    if controls:
        reasons.append("Unrecognized collection controls; pagination/completeness is unknown.")
    if len(rows) > limit:
        reasons.append(f"Local row limit {limit} retained {limit} of {len(rows)} published rows.")
    return rows[:limit], {"published_rows": len(rows), "retained_rows": min(len(rows), limit),
        "pages_retrieved": 1, "pagination_supported": False,
        "retrieval_complete": not reasons, "search_complete": None,
        "server_limit": None, "controls": controls, "reasons": reasons}


def available_methods(image, config):
    return tuple(method for method, key in METHODS.items()
                 if (method != "PPPM" or image.get("type") == "EMImage") and file_url(image, key, config))


def lookup(transport, identifier, kind, version="current", *, limit=MAX_ROWS):
    if kind not in {"em", "lm"} or not isinstance(identifier, str) or not identifier.strip():
        raise PublicAPIError("Enter a body identifier or driver-line name and select its kind.")
    if len(identifier.encode("utf-8")) > 4096:
        raise PublicAPIError("Identifier exceeds the 4096-byte request limit; it was not shortened.")
    if type(limit) is not int or not 1 <= limit <= MAX_ROWS:
        raise PublicAPIError("Invalid lookup row limit.")
    pointer = transport.get(BASE_URL + "/current.txt", text=True) if version == "current" else None
    resolved = pointer["payload"] if pointer else version
    if not isinstance(resolved, str) or not re.fullmatch(r"v[0-9]+(?:[_.][0-9]+){1,3}", resolved):
        raise PublicAPIError("The data version must be an explicit release, not a floating alias.")
    config = transport.get(f"{BASE_URL}/{resolved}/config.json")
    if not isinstance(config["payload"], dict) or not isinstance(config["payload"].get("stores"), dict):
        raise PublicAPIError("Malformed data configuration.")
    for store in config["payload"]["stores"].values():
        if (not isinstance(store, dict) or not isinstance(store.get("prefixes"), dict)
                or any(not isinstance(v, str) for v in store["prefixes"].values())):
            raise PublicAPIError("Malformed public data store prefixes.")
    route = "by_body" if kind == "em" else "by_line"
    # Published EM prefixes are retained; the final component is only a lookup selector.
    token = identifier.rsplit(":", 1)[-1] if kind == "em" else identifier
    response = transport.get(f"{BASE_URL}/{resolved}/metadata/{route}/{quote(token, safe='')}.json")
    images, completeness = collection(response["payload"], limit)
    for image in images:
        source_identity(image, config["payload"])
    checkpoint(transport.cancel)
    return {"snapshot_id": str(uuid4()), "identifier": identifier, "kind": kind,
            "requested_version": version, "data_version": resolved, "pointer": pointer,
            "config": config, "lookup": response, "images": images, "completeness": completeness}


def retrieve_matches(transport, lookup_snapshot, image_index, method, *, limit=MAX_ROWS):
    images = lookup_snapshot["images"]
    if type(image_index) is not int or not 0 <= image_index < len(images):
        raise PublicAPIError("Explicitly select one returned image.")
    if type(limit) is not int or not 1 <= limit <= MAX_ROWS:
        raise PublicAPIError("Invalid result row limit.")
    image = images[image_index]
    config = lookup_snapshot["config"]["payload"]
    if method not in available_methods(image, config):
        raise PublicAPIError("Select an available result method for this image.")
    version = lookup_snapshot["data_version"]
    url = public_url(file_url(image, METHODS[method], config))
    if f"/{version}/" not in urlsplit(url).path:
        raise PublicAPIError("Result reference does not identify the selected data release.")
    response = transport.get(url)
    rows, completeness = collection(response["payload"], limit)
    input_image = response["payload"].get("inputImage")
    if not isinstance(input_image, dict):
        completeness["retrieval_complete"] = False
        completeness["reasons"].append("Response input image is missing; query association cannot be verified.")
    elif any(input_image.get(k) != image.get(k) for k in ("id", "type", "libraryName", "alignmentSpace")):
        raise PublicAPIError("Result input image disagrees with the selected image; no snapshot imported.")
    session_id = str(uuid4())
    diagnostics = []
    occurrences = []
    specs = (("type", "unknown"), ("normalizedScore", "score"), ("matchingPixels", "matched_pixels"),
             ("pppmScore", "score"), ("pppmRank", "unknown"), ("mirrored", "mirror"))
    for position, row in enumerate(rows):
        checkpoint(transport.cancel)
        problems, values = [], []
        if not isinstance(row, dict):
            row = {"malformedResponse": row}
            problems.append(Diagnostic("malformed_match", "Malformed match retained as unresolved evidence.", row_index=position))
        target = row.get("image")
        if not isinstance(target, dict):
            target = {}
            problems.append(Diagnostic("missing_target_identity", "Target image is missing or malformed.", row_index=position))
        source = source_identity(target, config)
        for key, role in specs:
            raw = row.get(key)
            raw_text = None if raw is None else str(raw) if isinstance(raw, str) else canonical(raw)
            value = raw if role != "unknown" else raw_text
            try:
                field = ResultField(len(values), key, raw_text, value, role)
            except ValueError:
                field = ResultField(len(values), key, raw_text, None, role)
                problems.append(Diagnostic("invalid_api_field", f"Invalid {key}; original value retained.", row_index=position))
            values.append(field)
        expected = "CDSMatch" if method == "CDM" else "PPPMatch"
        if row.get("type") != expected:
            problems.append(Diagnostic("unknown_match_method", "Match method missing or inconsistent with requested method; retained without inference.", row_index=position))
        occurrences.append(MatchOccurrence(str(position), session_id, source, position, tuple(values), tuple(problems)))
    completeness["filters"] = {"selected_image_index": image_index,
        "image_id": image.get("id"), "library": image.get("libraryName"),
        "alignment_space": image.get("alignmentSpace"), "requested_method": method,
        "target_filter": None, "local_limit": limit}
    completeness["normalization_complete"] = not any(o.diagnostics for o in occurrences)
    diagnostics.extend(Diagnostic("incomplete_retrieval", reason) for reason in completeness["reasons"])
    evidence = {"protocol": "neuronbridge-public-s3", "reference_client_commit": CLIENT_COMMIT,
        "reference_client_package_version": "3.3.0", "adapter_version": "nb05-public-v1", "algorithm_version": None,
        "lookup_snapshot": lookup_snapshot, "result_response": response}
    session = SearchSession(session_id, "precomputed", version,
        query_reference=lookup_snapshot["identifier"], query_source=source_identity(image, config),
        parameters=(SourceParameter("requested_method", method), SourceParameter("algorithm_version", None),
                    SourceParameter("retrieval", completeness), SourceParameter("public_api_evidence", evidence)))
    checkpoint(transport.cancel)
    return SearchResults(session, tuple(occurrences), tuple(diagnostics))
