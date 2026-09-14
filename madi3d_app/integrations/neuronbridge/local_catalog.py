"""Public profile-bound library discovery and narrowly scoped, bounded data transfer."""
from dataclasses import asdict, dataclass
import base64
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import tempfile
from urllib.parse import quote, unquote, urlsplit

import requests
from urllib3.exceptions import HTTPError as Urllib3HTTPError

from .cdm_records import CDMArtifact, ImportedCDMImage, PROFILE, TEMPLATE_ID
from .search_profiles import BRAIN, VNC, SEARCH_PROFILES, search_profile
from .local_scorer import checkpoint
from .public_api import BASE_URL

DATA_BUCKET = "janelia-neuronbridge-data-prod"
IMAGE_BUCKET = "janelia-flylight-color-depth"
REPLICA = "0"
CATALOG_REFERENCE = "https://github.com/JaneliaSciComp/neuronbridge-services/blob/3847c602fcc5e3ae2bae47b7308ee11263ae6cef/search/src/main/nodejs/cds_input.js"
CITATIONS = (
    "NeuronBridge: Clements et al. (2024), https://doi.org/10.1186/s12859-024-05732-7",
    "Public search images: Janelia FlyLight / FlyEM, https://neuronbridge.janelia.org/",
)
# Each allowlisted collection is bound to its verified search contract. New
# collections require an explicit profile review and a searchable-image probe.
VERIFIED_LIBRARIES = {"FlyEM_Hemibrain_v1.2.1": ("v1.2.1", "hemibrain:v1.2.1", BRAIN.profile_id),
                      "FlyWire_FAFB_v783_realign": ("v783", "flywire_fafb:v783", BRAIN.profile_id),
                      "FlyEM_MANC_v1.2.1": ("v1.2.1", "manc:v1.2.1", VNC.profile_id)}


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


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


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


class ObjectMissing(FileNotFoundError):
    pass


class BulkTransport:
    """Serial GETs; each object has a byte limit, 90s deadline and two retries.

    Never uploads, follows redirects, reads credentials, or interprets ETags as
    checksums. Resume is at committed object boundaries, not unsafe byte ranges.
    """

    def __init__(self, cancel=None, *, session=None):
        from threading import Event
        self.cancel = cancel or Event()
        self.session = session or requests.Session()
        self.session.trust_env = False

    def close(self):
        self.session.close()

    def download(self, url, path, *, max_bytes):
        bulk_url(url)
        path = Path(path)
        partial = path.with_name(path.name + ".part")
        deadline = time.monotonic() + 90
        path.parent.mkdir(parents=True, exist_ok=True)

        def check():
            checkpoint(self.cancel)
            if time.monotonic() > deadline:
                raise TimeoutError("Object download exceeded 90 seconds. Resume to retry this object.")

        try:
            for attempt in range(3):
                check()
                delay = 2 ** attempt
                try:
                    with self.session.get(url, stream=True, timeout=(5, 5), allow_redirects=False,
                                          headers={"Accept-Encoding": "identity"}) as response:
                        status = response.status_code
                        if status == 404:
                            raise ObjectMissing(f"Public search object is missing: {url}")
                        if status in (429, 500, 502, 503, 504):
                            retry = response.headers.get("Retry-After")
                            if retry:
                                try:
                                    delay = float(retry)
                                except ValueError:
                                    delay = (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds()
                            if not math.isfinite(delay) or not 0 <= delay <= 15 or attempt == 2:
                                raise OSError(f"HTTP {status}; bounded retry limit reached. Resume later.")
                        elif status != 200:
                            raise OSError(f"Public object returned HTTP {status}: {url}")
                        else:
                            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                                raise ValueError("Compressed HTTP transfer is unsupported for search objects.")
                            length = response.headers.get("Content-Length")
                            if length is not None and (not length.isdigit() or int(length) > max_bytes):
                                raise ValueError("Object exceeds its bounded download size.")
                            h, size = hashlib.sha256(), 0
                            with partial.open("wb") as stream:
                                while True:
                                    check()
                                    chunk = response.raw.read1(64 * 1024, decode_content=False)
                                    check()
                                    if not chunk:
                                        break
                                    size += len(chunk)
                                    if size > max_bytes:
                                        raise ValueError("Object exceeds its bounded download size.")
                                    stream.write(chunk)
                                    h.update(chunk)
                                stream.flush()
                                os.fsync(stream.fileno())
                            if length is not None and size != int(length):
                                raise OSError("Truncated object; resume will restart it.")
                            check()
                            evidence = {"url": url, "retrieved_at": now(), "size": size,
                                        "sha256": h.hexdigest(), "hash_authority": "locally_observed",
                                        "etag": response.headers.get("ETag"),
                                        "last_modified": response.headers.get("Last-Modified"),
                                        "provider_checksum_sha256": response.headers.get("x-amz-checksum-sha256"),
                                        "provider_checksum_type": response.headers.get("x-amz-checksum-type")}
                            if evidence["provider_checksum_sha256"] and evidence["provider_checksum_type"] == "FULL_OBJECT":
                                if base64.b64encode(h.digest()).decode("ascii") != evidence["provider_checksum_sha256"]:
                                    raise ValueError("Provider whole-object SHA-256 verification failed.")
                            os.replace(partial, path)
                            return evidence
                except (requests.RequestException, Urllib3HTTPError) as exc:
                    if attempt == 2:
                        raise OSError("Download failed after three attempts. Resume later.") from exc
                check()
                self.cancel.wait(delay)
            raise OSError("Download retry limit reached.")
        finally:
            partial.unlink(missing_ok=True)


@dataclass(frozen=True)
class Library:
    name: str
    release: str
    data_version: str
    count: int
    store: str
    prefix: str
    published_name_prefix: str
    catalog: dict
    region: str = "Brain"
    template_id: str = TEMPLATE_ID
    profile: str = PROFILE
    replica: str = REPLICA

    def __post_init__(self):
        profile = search_profile(self.profile)
        if (self.name not in VERIFIED_LIBRARIES or self.region != profile.anatomical_area
                or self.template_id != profile.template_id or self.replica != REPLICA
                or type(self.count) is not int or not 0 < self.count <= 2_000_000):
            raise ValueError("Unsupported or empty local-search library/profile.")
        if not re.fullmatch(r"v\d+(?:[_.]\d+){1,3}", self.data_version):
            raise ValueError("A fixed public data release is required.")
        if (self.release, self.published_name_prefix, self.profile) != VERIFIED_LIBRARIES[self.name]:
            raise ValueError("Library release disagrees with the verified collection.")
        expected = f"https://s3.amazonaws.com/{IMAGE_BUCKET}/{self.template_id}/{self.name}/searchable_neurons/"
        if self.prefix != expected:
            raise ValueError("Unsupported searchable-image prefix.")
        if self.catalog["data_version"] != self.data_version:
            raise ValueError("Library and catalog data releases disagree.")
        config = self.catalog["config"]["payload"]
        store = config["stores"][self.store]
        declarations = [v for v in store.get("customSearch", {}).get("emLibraries", []) if v.get("name") == self.name]
        if (len(declarations) != 1 or declarations[0].get("count") != self.count or
                declarations[0].get("publishedNamePrefix") != self.published_name_prefix or
                store.get("anatomicalArea") != self.region or
                config.get("anatomicalAreas", {}).get(self.region, {}).get("alignmentSpace") != self.template_id):
            raise ValueError("Library declaration disagrees with retained catalog evidence.")
        object.__setattr__(self, "catalog", json.loads(canonical(self.catalog)))

    @property
    def manifest_url(self):
        return self.prefix + f"KEYS/{self.replica}/keys_denormalized.json"

    @property
    def catalog_id(self):
        return digest(self.to_dict())

    def to_dict(self):
        return asdict(self)

    @property
    def citations(self):
        collection = ("MANC: https://www.janelia.org/project-team/flyem/manc-connectome" if self.profile == VNC.profile_id else
                      "Hemibrain: https://doi.org/10.7554/eLife.57443" if self.name.startswith("FlyEM_Hemibrain") else
                      "FlyWire FAFB: https://flywire.ai/")
        return (*CITATIONS, collection, "Public NeuronBridge data: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/")

    def estimate(self):
        # No inventory size is advertised by this catalog. Explicit conservative
        # estimates; no HEAD-per-member fan-out. Sparse RGB index worst case is
        # seven bytes per pixel, independent of search thresholds.
        height, width = search_profile(self.profile).search_canvas_yx
        pixels = height * width
        source = self.count * (pixels * 3 + 64 * 1024)
        index = self.count * pixels * 7
        metadata = self.count * 16 * 1024
        return {"download_bytes": source + metadata, "estimated": True,
                "required_disk_bytes": source + index + metadata + 128 * 1024 * 1024,
                "basis": "Conservative uncompressed RGB sources, worst-case lossless sparse index, metadata and staging; actual sizes may differ."}


def query_profile(record):
    """Generated identity is authoritative; imported bare pixels remain unverified.

    For bare imports a canvas selects only the numerical search contract. It
    never supplies mapping evidence, calibration or template identity.
    """
    if record["version"] == 2:
        image = ImportedCDMImage.from_dict(record["imported_image"])
        artifact = image.generation_artifact
        if artifact is None:
            return next((p for p in SEARCH_PROFILES.values()
                         if image.shape_yx_rgb == (*p.search_canvas_yx, 3)), None)
    else:
        artifact = CDMArtifact.from_dict(record["artifact"])
    return search_profile(artifact.profile)


def compatible_query(record, library=None):
    profile = query_profile(record)
    return profile is not None and (library is None or profile.profile_id == library.profile)


def libraries_from_catalog(catalog):
    config = catalog["config"]["payload"]
    version = catalog["data_version"]
    libraries = []
    for store_id, store in config["stores"].items():
        region = store.get("anatomicalArea")
        area = config.get("anatomicalAreas", {}).get(region, {})
        profile = next((p for p in SEARCH_PROFILES.values() if p.anatomical_area == region
                        and p.alignment_space == area.get("alignmentSpace")), None)
        if profile is None:
            continue
        search = store.get("customSearch", {})
        if (search.get("searchFolder") != "searchable_neurons" or
                store.get("prefixes", {}).get("CDM") != f"https://s3.amazonaws.com/{IMAGE_BUCKET}/"):
            continue
        for entry in search.get("emLibraries", []):
            if entry.get("name") not in VERIFIED_LIBRARIES:
                continue
            name = entry["name"]
            release, published, profile_id = VERIFIED_LIBRARIES[name]
            if entry.get("publishedNamePrefix") != published or profile_id != profile.profile_id:
                continue
            libraries.append(Library(name, release, version, entry["count"], store_id,
                f"https://s3.amazonaws.com/{IMAGE_BUCKET}/{profile.template_id}/{name}/searchable_neurons/",
                published, catalog, region=region, template_id=profile.template_id, profile=profile.profile_id))
    return libraries


def acquire_catalog(cancel=None):
    transport = BulkTransport(cancel)
    try:
        with tempfile.TemporaryDirectory(prefix="madi3d-library-catalog-") as root:
            def retrieve(url, name, maximum, *, text=False):
                path = Path(root) / name
                evidence = transport.download(url, path, max_bytes=maximum)
                raw = path.read_bytes()
                payload = raw.decode("utf-8").strip() if text else parse_json(raw)
                return dict(evidence, payload=payload, raw_base64=base64.b64encode(raw).decode("ascii"))
            pointer = retrieve(BASE_URL + "/current.txt", "pointer.txt", 4096, text=True)
            version = pointer["payload"]
            if not re.fullmatch(r"v\d+(?:[_.]\d+){1,3}", version):
                raise ValueError("Invalid production data pointer.")
            config = retrieve(f"{BASE_URL}/{version}/config.json", "config.json", 1024 * 1024)
            catalog = {"pointer": pointer, "data_version": version, "config": config,
                       "reference": CATALOG_REFERENCE, "acquired_at": now()}
            return libraries_from_catalog(catalog)
    finally:
        transport.close()


def manifest_members(path, cancel=None):
    """Incrementally parse the upstream JSON array of keys, at most 64 KiB text.

    Only strings are in this verified manifest protocol. Truncated arrays,
    oversized entries, trailing commas/data and non-string members are errors.
    """
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as stream:
        buffer, eof = "", False

        def refill():
            nonlocal buffer, eof
            checkpoint(cancel)
            more = stream.read(16384)
            eof = not more
            buffer += more
            if len(buffer) > 65536:
                raise ValueError("Oversized manifest member.")

        def whitespace():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                refill()
                buffer = buffer.lstrip()

        whitespace()
        if not buffer.startswith("["):
            raise ValueError("Search manifest must be a JSON array of object keys.")
        buffer = buffer[1:]
        first = True
        while True:
            whitespace()
            if buffer.startswith("]"):
                buffer = buffer[1:]
                break
            if not first:
                if not buffer.startswith(","):
                    raise ValueError("Truncated or malformed membership manifest.")
                buffer = buffer[1:]
                whitespace()
            if not buffer.startswith('"'):
                raise ValueError("Manifest members must be nonempty strings.")
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError("Truncated membership manifest.") from exc
                    refill()
            if not value or len(value.encode("utf-8")) > 8192:
                raise ValueError("Invalid manifest object key.")
            buffer = buffer[end:]
            checkpoint(cancel)
            yield value
            first = False
        whitespace()
        if buffer:
            raise ValueError("Unexpected trailing manifest content.")


def member_url(library, key):
    expected = f"{library.template_id}/{library.name}/searchable_neurons/"
    if (not isinstance(key, str) or not key.startswith(expected) or "\\" in key or
            any(part in (".", "..", "") for part in key.split("/")) or
            not key.lower().endswith((".tif", ".tiff"))):
        raise ValueError("Member is not a searchable TIFF in the selected collection.")
    return bulk_url(f"https://s3.amazonaws.com/{IMAGE_BUCKET}/" + quote(key, safe="/"))
