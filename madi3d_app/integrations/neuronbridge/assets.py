"""Explicit asset resolution and retrieval; no GUI or search ranking calls."""
from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit

from .cache import asset_cache_filename, cached_file_ready, neuronbridge_cache_root, stream_download
from .records import AssetIdentity


ASSET_SUFFIXES = {
    "AlignedBodyOBJ": ".obj", "AlignedBodySWC": ".swc",
    "VisuallyLosslessStack": ".h5j",
}


class ResolutionError(ValueError):
    def __init__(self, message, *, status="unresolved", candidates=(), asset_options=()):
        super().__init__(message)
        self.status = status
        self.candidates = tuple(compact_candidate(data) for data in candidates)
        # Fully resolved alternatives are transient and never persisted as diagnostics.
        self.asset_options = tuple(asset_options)


def compact_candidate(data):
    """Keep only image identities and selectors needed to explain ambiguity."""
    keys = ("id", "type", "libraryName", "libraryRelease", "publishedName",
            "slideCode", "alignmentSpace", "gender", "objective", "channel",
            "anatomicalArea", "mountingProtocol", "neuronType", "neuronInstance")
    result = {key: copy.deepcopy(data[key]) for key in keys if data.get(key) is not None}
    result["files"] = {key: data["files"][key] for key in ASSET_SUFFIXES
                       if data.get("files", {}).get(key)}
    if data.get("mismatches"):
        result["mismatches"] = copy.deepcopy(data["mismatches"])
    for key in ("asset_urls", "data_version"):
        if data.get(key):
            result[key] = copy.deepcopy(data[key])
    return result


def _resolved_asset(remote, asset, version, api):
    if urlsplit(asset.url).scheme.lower() not in {"http", "https"}:
        raise ResolutionError("A NeuronBridge download requires an HTTP(S) asset URL.")
    source = remote.source
    filename = asset_cache_filename(
        library=source.library, release=source.library_release,
        data_version=version, alignment=source.image.alignment_space,
        asset_type=asset.asset_type, url=asset.url,
        asset_id=asset.asset_id, checksum=asset.checksum_sha256,
    )
    return replace(remote, resolution={
        "asset_type": asset.asset_type, "url": asset.url, "cache_filename": filename,
        "selected_asset": asdict(asset), "data_version": version, "api_metadata": api,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "identity_normalizations": identity_normalizations(source, api) if api else {},
        "metadata_discrepancies": image_metadata_discrepancies(source, api) if api else {},
    })


def _choose_asset(options, candidates=()):
    if len(options) == 1:
        return options[0]
    raise ResolutionError(
        "Choose the geometry to retrieve." if options else "No matching geometry asset is available.",
        status="ambiguous" if options else "unresolved", candidates=candidates,
        asset_options=options,
    )


def _json_metadata(value):
    if isinstance(value, Enum):
        return value.value
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, dict):
        return {k: _json_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(v) for v in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return _json_metadata(value.dict())
    return _json_metadata(vars(value))


def _channel_equal(requested, actual):
    # NB API channels are integers; retain the opaque CSV spelling separately.
    return str(requested) == str(actual) or (
        str(requested).isascii() and str(requested).isdigit()
        and type(actual) is int and int(requested) == actual
    )


def _identity_checks(source):
    """Hard image selectors shared by geometry and thumbnail resolution."""
    return {
        "type": "EMImage" if source.kind == "em" else "LMImage",
        "libraryName": source.library, "libraryRelease": source.library_release,
        "publishedName": source.published_name,
        "id": source.image.image_id,
        "slideCode": source.image.slide_code,
        "alignmentSpace": source.image.alignment_space,
        # Sex can distinguish LM acquisitions. An EM neuron's dataset/name
        # identifies it independently of conflicting biological metadata.
        "gender": source.image.sex if source.kind == "lm" else None,
        "objective": source.image.magnification,
        "anatomicalArea": source.image.anatomical_area,
        "mountingProtocol": source.image.mounting_protocol,
        "neuronType": source.biological.neuron_type,
        "neuronInstance": source.biological.neuron_instance,
    }

def _identity_equal(source, key, expected, actual, data):
    if expected == actual:
        return True
    if key == "gender":
        sex = {"f": "female", "female": "female", "m": "male", "male": "male"}
        return (isinstance(expected, str) and isinstance(actual, str)
                and expected.casefold() in sex
                and sex[expected.casefold()] == sex.get(actual.casefold()))
    if key == "id" and type(actual) is int:
        return expected == str(actual)
    if key == "publishedName" and source.kind == "em":
        # A bare body token is only meaningful inside an explicitly matched
        # dataset. Two different published namespaces must never be equated.
        if source.library and source.library == data.get("libraryName"):
            if isinstance(expected, str) and isinstance(actual, str):
                bare, qualified = (expected, actual) if ":" not in expected else (actual, expected)
                return (bare.isascii() and bare.isdigit() and ":" in qualified
                        and qualified.rsplit(":", 1)[-1] == bare)
    return False


def image_mismatches(source, data):
    result = {key: {"requested": expected, "returned": data.get(key)}
              for key, expected in _identity_checks(source).items()
              if expected is not None and not _identity_equal(source, key, expected, data.get(key), data)}
    if source.channel is not None and not _channel_equal(source.channel.selector, data.get("channel")):
        result["channel"] = {"requested": source.channel.selector, "returned": data.get("channel")}
    return result


def image_metadata_discrepancies(source, data):
    """Retain non-blocking EM evidence conflicts without correcting either value."""
    sex = source.image.sex
    if source.kind == "em" and sex is not None and not _identity_equal(
            source, "gender", sex, data.get("gender"), data):
        return {"gender": {"requested": sex, "returned": data.get("gender")}}
    return {}


def lm_stack_channel(remote, matches=()):
    """Consume recorded channel identity or exact public evidence without rewriting it."""
    from .public_api import image_channel

    channel = remote.source.channel
    if remote.source.kind != "lm" or (channel is not None and channel.index_base is not None):
        return channel
    images = [remote.resolution.get("api_metadata", {})]
    # Earlier public results retained the API response but not its index base.
    # Read the indexed match evidence, including for already-saved projects.
    for match in matches:
        public = next((p.value for p in match.session.parameters if p.name == "public_api_evidence"), None)
        if not public:
            continue
        try:
            image = public["result_response"]["payload"]["results"][match.occurrence.row_index]["image"]
            if str(image.get("id")) == match.occurrence.target.image.image_id:
                images.append(image)
        except (KeyError, IndexError, TypeError, AttributeError):
            continue
    selections = set()
    for image in images:
        if image and not image_mismatches(remote.source, image):
            selected = image_channel(image)
            if selected is not None and selected.index_base is not None:
                selections.add(selected)
    return next(iter(selections)) if len(selections) == 1 else channel


def identity_normalizations(source, data):
    return {key: {"original": expected, "resolved": data.get(key)}
            for key, expected in dict(_identity_checks(source), gender=source.image.sex).items()
            if expected is not None and expected != data.get(key)
            and _identity_equal(source, key, expected, data.get(key), data)}


def _matches(source, data):
    return not image_mismatches(source, data)


def _check_cancel(cancel_check):
    if cancel_check and cancel_check():
        raise InterruptedError("NeuronBridge download cancelled.")


def validate_resolution_request(remote):
    """Report locally known resolution gaps without constructing an API client."""
    source = remote.source
    if remote.problems:
        raise ResolutionError("\n".join(remote.problems), status="ambiguous")
    if source.kind not in {"em", "lm"}:
        raise ResolutionError("The CSV does not identify an EM or LM target.")
    if not any(a.url and a.asset_type in ASSET_SUFFIXES for a in source.assets):
        if not source.library or not source.image.alignment_space:
            raise ResolutionError("Library and alignment space are required to resolve this result.")
        name = source.biological.neuron_id if source.kind == "em" else source.biological.line_name
        if not name:
            raise ResolutionError("This client requires the original neuron or line name for lookup.")


def resolve_source(remote, nb, *, cancel_check=None):
    """Require one image and one requested geometry asset; report every ambiguity."""
    _check_cancel(cancel_check)
    validate_resolution_request(remote)
    source = remote.source
    supplied = [a for a in source.assets if a.url and a.asset_type in ASSET_SUFFIXES]
    if supplied:
        options = []
        for asset in supplied:
            if (source.kind == "lm") != (asset.asset_type == "VisuallyLosslessStack"):
                raise ResolutionError("The supplied asset type conflicts with the target type.")
            options.append(_resolved_asset(remote, asset, remote.data_version, {}))
        return _choose_asset(options)
    else:
        name = source.biological.neuron_id if source.kind == "em" else source.biological.line_name
        # This is a transport selector only. Never replace the opaque published
        # identity by the body lookup token or convert it to an integer.
        lookup = name.rsplit(":", 1)[-1] if source.kind == "em" else name
        images = nb.get_em_images(lookup) if source.kind == "em" else nb.get_lm_images(lookup)
        _check_cancel(cancel_check)
        pairs = [(image, _json_metadata(image)) for image in images]
        matches = [(image, data) for image, data in pairs if _matches(source, data)]
        if len(matches) != 1:
            candidates = [dict(data, mismatches=image_mismatches(source, data),
                               data_version=getattr(nb, "version", remote.data_version),
                               asset_urls={key: nb._get_files_url(data.get("files", {}), key)
                                           for key in ASSET_SUFFIXES if data.get("files", {}).get(key)})
                          for _image, data in pairs]
            rejected = sorted({key for data in candidates for key in data["mismatches"]})
            detail = (" Rejected fields: " + ", ".join(rejected) + ". Review the result's retrieval details."
                      if rejected else " The lookup returned no candidates." if not pairs else " Review the candidate images.")
            raise ResolutionError(
                f"Expected one matching {source.kind.upper()} image; found {len(matches)}. "
                "The original result is retained; no image was selected." + detail,
                status="ambiguous" if len(matches) > 1 else "unresolved",
                candidates=candidates,
            )
        image, api = matches[0]
        supported = ("VisuallyLosslessStack",) if source.kind == "lm" else ("AlignedBodyOBJ", "AlignedBodySWC")
        files = api.get("files", {})
        available = [key for key in supported if files.get(key) and (
            not source.assets or any(
                a.asset_type == key or (a.asset_type is None and a.asset_id in (None, files[key]))
                for a in source.assets
            )
        )]
        version = getattr(nb, "version", remote.data_version)
        if remote.data_version and version != remote.data_version:
            raise ResolutionError("The client data version differs from the imported search version.")
        available_ids = {api["files"][key] for key in available}
        for asset in source.assets:
            if asset.asset_type is None and (
                (asset.asset_id and asset.asset_id not in available_ids)
                or (not asset.asset_id and len(available) > 1)
            ):
                raise ResolutionError("Asset evidence requires a type or matching asset ID.",
                                      status="ambiguous", candidates=[api])
        options = []
        for asset_type in available:
            asset_id = api["files"][asset_type]
            evidence = [a for a in source.assets if a.asset_type == asset_type or (
                a.asset_type is None and (a.asset_id == asset_id or (
                    not a.asset_id and len(available) == 1))
            )]
            if len(evidence) > 1:
                raise ResolutionError("Multiple identities describe the same asset; resolve the conflicting evidence.",
                                      status="ambiguous", candidates=[api])
            asset = evidence[0] if evidence else AssetIdentity()
            if asset.asset_id and asset.asset_id != asset_id:
                raise ResolutionError("The resolved asset does not match the supplied asset ID.", candidates=[api])
            url = nb._get_files_url(api["files"], asset_type)
            if not url:
                continue
            if asset.url and asset.url != url:
                raise ResolutionError("The resolved asset does not match the supplied URL.", candidates=[api])
            options.append(_resolved_asset(remote, replace(
                asset, asset_type=asset_type, asset_id=asset_id, url=url,
            ), version, api))
        return _choose_asset(options, [api])


def candidate_options(remote, candidate):
    """Resolve an explicitly selected, compatible image from retained evidence."""
    if image_mismatches(remote.source, candidate):
        raise ResolutionError("This candidate conflicts with the recorded source. Review the rejected fields before importing a corrected result.")
    version = candidate.get("data_version")
    if remote.data_version is not None and version != remote.data_version:
        raise ResolutionError("Candidate and original result identify different data releases.")
    kinds = ("VisuallyLosslessStack",) if remote.source.kind == "lm" else ("AlignedBodyOBJ", "AlignedBodySWC")
    options = []
    for kind in kinds:
        url = candidate.get("asset_urls", {}).get(kind)
        asset_id = candidate.get("files", {}).get(kind)
        if not url or not asset_id:
            continue
        evidence = [a for a in remote.source.assets if a.asset_type in (None, kind)
                    and a.asset_id in (None, asset_id)]
        if remote.source.assets and not evidence:
            continue
        if len(evidence) > 1:
            raise ResolutionError("Multiple recorded identities describe this asset. Review the conflicting evidence.")
        asset = evidence[0] if evidence else AssetIdentity()
        if asset.url and asset.url != url:
            raise ResolutionError("The candidate asset URL conflicts with the recorded source URL.")
        option = _resolved_asset(remote, replace(asset, asset_id=asset_id, asset_type=kind, url=url), version, candidate)
        options.append(option)
    if not options:
        raise ResolutionError("This candidate has no compatible downloadable geometry. Refresh its public metadata.")
    return tuple(options)


def fetch_source(remote, client_factory, *, cancel_check=None, cache_root=None, download=stream_download, on_resolved=None):
    """Called only for explicit retrieval. Missing local data never erases evidence."""
    _check_cancel(cancel_check)
    if not remote.resolution:
        validate_resolution_request(remote)
        direct = any(a.url and a.asset_type in ASSET_SUFFIXES for a in remote.source.assets)
        nb = None if direct else client_factory(remote.data_version)
        _check_cancel(cancel_check)
        remote = resolve_source(remote, nb, cancel_check=cancel_check)
    if on_resolved is not None:
        on_resolved(remote)
    checksum = remote.expected_sha256
    existing = remote.cached_path()
    if existing and cached_file_ready(existing, expected_sha256=checksum, cancel_check=cancel_check):
        _check_cancel(cancel_check)
        return _record_retrieval(remote, existing, checksum, cancel_check)
    root = Path(cache_root) if cache_root is not None else neuronbridge_cache_root()
    destination = (root / remote.resolution["cache_filename"]).resolve()
    if existing != destination and cached_file_ready(destination, expected_sha256=checksum, cancel_check=cancel_check):
        _check_cancel(cancel_check)
        return _record_retrieval(remote, destination, checksum, cancel_check)
    kwargs = {"cancel_check": cancel_check}
    if checksum:
        kwargs["expected_sha256"] = checksum
    download(remote.resolution["url"], destination, **kwargs)
    _check_cancel(cancel_check)
    return _record_retrieval(remote, destination, checksum, cancel_check)


def _record_retrieval(remote, path, checksum, cancel_check):
    if checksum is None:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                _check_cancel(cancel_check)
                digest.update(chunk)
        checksum = digest.hexdigest()
    _check_cancel(cancel_check)
    resolution = dict(remote.resolution)
    resolution["retrieved_sha256"] = checksum
    resolution.setdefault("retrieved_at", datetime.now(timezone.utc).isoformat())
    return replace(remote, local_path=str(path), resolution=resolution)


class AssetMetadataClient:
    """MADI3D-owned public metadata boundary; all requests share cancellation.

    Resolution retains the pinned public lookup/file-store protocol. No upstream
    requests, authentication or process-wide transport modifications are used.
    """
    def __init__(self, version=None, *, cancel=None, transport=None):
        from .public_api import PublicTransport
        self.version = version or "current"
        self._new_transport = lambda: PublicTransport(cancel, retries=0)
        self.transport = transport or PublicTransport(cancel, retries=0)
        self._owns_transport = transport is None
        self._used = False
        self.config = None

    def _images(self, identifier, kind):
        from .public_api import lookup
        if self._used and self._owns_transport:
            self.transport.close()
            self.transport = self._new_transport()
        self._used = True
        snapshot = lookup(self.transport, identifier, kind, self.version)
        if not snapshot["completeness"]["retrieval_complete"]:
            raise ResolutionError("Image lookup is incomplete; no asset was selected.")
        self.version = snapshot["data_version"]
        self.config = snapshot["config"]["payload"]
        return snapshot["images"]

    def get_em_images(self, identifier):
        return self._images(identifier, "em")

    def get_lm_images(self, identifier):
        return self._images(identifier, "lm")

    def _get_files_url(self, files, asset_type):
        from .public_api import file_url
        return file_url({"files": files}, asset_type, self.config)

    def close(self):
        self.transport.close()


def make_nb_client(version=None, *, cancel=None):
    return AssetMetadataClient(version, cancel=cancel)
