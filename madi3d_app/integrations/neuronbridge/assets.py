"""Explicit asset resolution and retrieval; no GUI or search ranking calls."""
from __future__ import annotations

import copy
import hashlib
import threading
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from madi3d_app.io_utils import file_revision, filename_unit_budget

from .cache import (
    asset_cache_filename, cached_file_ready, compact_asset_cache_path,
    neuronbridge_cache_root, stream_download,
)
from .local_geometry import AMBIGUOUS, GEOMETRY_SUFFIXES, MATCHED, UNAVAILABLE
from .records import AssetIdentity


ASSET_SUFFIXES = {
    "AlignedBodyOBJ": ".obj", "AlignedBodySWC": ".swc",
    "VisuallyLosslessStack": ".h5j",
}

_RETRIEVAL_SLOT = threading.BoundedSemaphore(1)


@contextmanager
def _retrieval_slot(cancel_check):
    """Serialize acceptance/download so queued duplicates reuse complete bytes."""
    while not _RETRIEVAL_SLOT.acquire(timeout=0.05):
        _check_cancel(cancel_check)
    try:
        _check_cancel(cancel_check)
        yield
    finally:
        _RETRIEVAL_SLOT.release()


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

def _asset_matches_selection(selected, candidate):
    if selected is None:
        return True
    if selected.asset_type and candidate.asset_type != selected.asset_type:
        return False
    if selected.asset_id and candidate.asset_id != selected.asset_id:
        return False
    if selected.url and candidate.url and candidate.url != selected.url:
        return False
    if (selected.checksum_sha256 and candidate.checksum_sha256
            and selected.checksum_sha256 != candidate.checksum_sha256):
        return False
    return True


def _merge_selected_asset(selected, candidate):
    if selected is None:
        return candidate
    return replace(
        candidate,
        asset_type=selected.asset_type or candidate.asset_type,
        asset_id=selected.asset_id or candidate.asset_id,
        checksum_sha256=selected.checksum_sha256 or candidate.checksum_sha256,
    )


def _same_selected_asset(expected, actual):
    if expected is None or actual is None:
        return expected is actual
    if expected.asset_type != actual.asset_type:
        return False
    if expected.asset_id:
        if expected.asset_id != actual.asset_id:
            return False
    elif expected.url:
        if expected.url != actual.url:
            return False
    elif expected.checksum_sha256:
        if expected.checksum_sha256 != actual.checksum_sha256:
            return False
    else:
        # Representation-only evidence (for example a local-acquisition
        # descriptor) asserts no locator to compare; the type must agree.
        return bool(expected.asset_type) and expected.asset_type == actual.asset_type
    return not (
        expected.checksum_sha256 and actual.checksum_sha256
        and expected.checksum_sha256 != actual.checksum_sha256
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
        public = match.session.external_evidence if match.session.source_kind == "precomputed" else None
        if not public:
            continue
        try:
            image = match.session.evidence_for(match.occurrence)["image"]
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


def _geometry_mismatches(source, data):
    from .archive_catalog import VERIFIED_LIBRARIES

    result = image_mismatches(source, data)
    # The pinned catalog independently binds these geometry libraries to releases.
    # Public image metadata may omit the release; an explicit conflict still fails.
    catalog = VERIFIED_LIBRARIES.get(source.library)
    if (source.kind == "em" and source.provider == "NeuronBridge" and catalog
            and source.library_release == catalog[0] and data.get("libraryRelease") is None):
        result.pop("libraryRelease", None)
    return result


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
    """Resolve the exact retained asset choice; never substitute another format."""
    _check_cancel(cancel_check)
    validate_resolution_request(remote)
    source = remote.source
    selected = remote.selected_asset
    supplied = [a for a in source.assets if a.url and a.asset_type in ASSET_SUFFIXES]
    if selected is not None:
        supplied = [
            asset for asset in supplied
            if _asset_matches_selection(selected, asset)
        ]
    if supplied:
        options = []
        for asset in supplied:
            if (source.kind == "lm") != (asset.asset_type == "VisuallyLosslessStack"):
                raise ResolutionError("The supplied asset type conflicts with the target type.")
            options.append(_resolved_asset(
                remote, _merge_selected_asset(selected, asset),
                remote.data_version, {},
            ))
        return _choose_asset(options)
    else:
        name = source.biological.neuron_id if source.kind == "em" else source.biological.line_name
        # This is a transport selector only. Never replace the opaque published
        # identity by the body lookup token or convert it to an integer.
        lookup = name.rsplit(":", 1)[-1] if source.kind == "em" else name
        images = nb.get_em_images(lookup) if source.kind == "em" else nb.get_lm_images(lookup)
        _check_cancel(cancel_check)
        pairs = [(image, _json_metadata(image)) for image in images]
        matches = [(image, data) for image, data in pairs if not _geometry_mismatches(source, data)]
        if len(matches) != 1:
            candidates = [dict(data, mismatches=_geometry_mismatches(source, data),
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
            (selected is None or selected.asset_type == key)
            and (not source.assets or any(
                a.asset_type == key or (a.asset_type is None and a.asset_id in (None, files[key]))
                for a in source.assets
            ))
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
            if selected is not None:
                if selected.asset_id and selected.asset_id != asset_id:
                    raise ResolutionError(
                        "The recorded selected asset is not available in this data release.",
                        candidates=[api],
                    )
                asset = _merge_selected_asset(selected, asset)
            if asset.asset_id and asset.asset_id != asset_id:
                raise ResolutionError("The resolved asset does not match the supplied asset ID.", candidates=[api])
            url = nb._get_files_url(api["files"], asset_type)
            if not url:
                continue
            if asset.url and asset.url != url and not (
                    asset.asset_id and asset.asset_id == asset_id):
                raise ResolutionError("The resolved asset does not match the supplied URL.", candidates=[api])
            options.append(_resolved_asset(remote, replace(
                asset, asset_type=asset_type, asset_id=asset_id, url=url,
            ), version, api))
        return _choose_asset(options, [api])


def candidate_options(remote, candidate):
    """Resolve an explicitly selected, compatible image from retained evidence."""
    if _geometry_mismatches(remote.source, candidate):
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


LOCAL_GEOMETRY_TYPES = ("AlignedBodyOBJ", "AlignedBodySWC")


def _local_geometry_hit(remote, batch, cancel_check):
    """Resolve a typed EM neuron ID inside the user-configured local folder.

    The typed neuron ID remains authoritative; the filename only locates a
    candidate file for that ID. A hit records local-acquisition evidence and
    an observed checksum only — never a NeuronBridge asset URL, asset ID,
    published checksum or download claim. A miss, an ambiguity, an unavailable
    folder, an unconfigured library and a descriptor without a typed ID return
    None; a Local-mode batch reports those as unresolved instead of retrieving
    them from NeuronBridge.
    """
    if batch is None or remote.problems or remote.source.kind != "em":
        return None
    identifier = remote.source.biological.neuron_id
    # Published representation availability is not an explicit selection.
    asset_type = remote.resolution.get("asset_type")
    if not identifier:
        return None
    if asset_type is None:
        available = [kind for kind in LOCAL_GEOMETRY_TYPES
                     if batch.lookup(remote.source.library, identifier, kind).status in {MATCHED, AMBIGUOUS}]
        if len(available) == 2:
            _choose_asset([replace(remote, resolution={
                "asset_type": kind, "selected_asset": {"asset_type": kind},
            }) for kind in available])
        asset_type = available[0] if len(available) == 1 else None
    if asset_type not in LOCAL_GEOMETRY_TYPES:
        return None
    result = batch.lookup(remote.source.library, identifier, asset_type)
    if result.status != MATCHED:
        return None
    path = result.path
    _check_cancel(cancel_check)
    before = file_revision(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            _check_cancel(cancel_check)
            digest.update(chunk)
    if file_revision(path) != before:
        raise ResolutionError("The local geometry file changed during verification. Retry loading it.")
    acquisition = remote.resolution.get("local_acquisition")
    if acquisition:
        if digest.hexdigest() != acquisition["checksum_sha256"]:
            raise ResolutionError("The local geometry no longer matches its accepted checksum. "
                                  "Restore the file or import the changed source separately.")
        _check_cancel(cancel_check)
        return replace(remote, local_path=str(path), local_revision=before)
    resolution = {key: copy.deepcopy(value) for key, value in remote.resolution.items()
                  if key in {"api_metadata", "data_version", "identity_normalizations",
                             "metadata_discrepancies", "user_resolution", "prior_source_decision"}}
    resolution.update({
        "asset_type": asset_type,
        "selected_asset": {"asset_type": asset_type},
        "local_acquisition": {
            "origin": "user-supplied-local-source",
            "file_name": path.name,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "checksum_sha256": digest.hexdigest(),
            "verification": "locally-observed",
        },
    })
    _check_cancel(cancel_check)
    return replace(remote, resolution=resolution, local_path=str(path), local_revision=before)


def local_geometry_shortfall(remote, batch):
    """Explain EM geometry that this batch's local folders cannot supply.

    The message names the neuron, its library and the requested representation
    but never a machine-local folder path: an unresolved retrieval is retained
    as a project source decision, while the folder association belongs to this
    computer only and must not become portable project state.
    """
    source = remote.source
    library = source.library or "this library"
    identifier = source.biological.neuron_id
    if not identifier:
        return ("Local geometry is on, but this result has no neuron/body ID, so no local file can "
                "be selected for it. Switch the NeuronBridge geometry source to Online to retrieve it.")
    kinds = ((remote.resolution.get("asset_type"),)
             if remote.resolution.get("asset_type") in LOCAL_GEOMETRY_TYPES else LOCAL_GEOMETRY_TYPES)
    representation = (" / ".join(GEOMETRY_SUFFIXES[kind] for kind in kinds))
    if not batch.configured(source.library):
        return (f"Local geometry is on, but no local neuron folder is configured for {library}, so no "
                f"{representation} file for neuron {identifier} can be found. Choose a local folder for "
                f"{library}, or switch the NeuronBridge geometry source to Online.")
    results = [batch.lookup(source.library, identifier, kind) for kind in kinds]
    if any(result.status == UNAVAILABLE for result in results):
        return (f"Local geometry is on, but the local neuron folder configured for {library} cannot be "
                f"read, so neuron {identifier} stays unresolved. Reconnect that folder or choose another "
                "one, or switch the NeuronBridge geometry source to Online.")
    ambiguous = [result for result in results if result.status == AMBIGUOUS]
    if ambiguous:
        names = ambiguous[0].candidates
        listed = ", ".join(names[:3]) + (", …" if len(names) > 3 else "")
        return (f"Local geometry is on, but {len(names)} files in the local {library} folder match neuron "
                f"{identifier} ({listed}). Keep exactly one {representation} file for this neuron, "
                "or switch the NeuronBridge geometry source to Online.")
    return (f"Local geometry is on, but the local {library} folder has no {representation} file for "
            f"neuron {identifier}. Add it to the configured folder, choose another folder, or switch the "
            "NeuronBridge geometry source to Online to retrieve it from NeuronBridge.")


def fetch_source(remote, client_factory, *, cancel_check=None, cache_root=None,
                 download=stream_download, on_resolved=None, geometry_source=None):
    """Retrieve one exact asset under this batch's captured EM geometry source.

    ``geometry_source`` is the transient ``GeometrySourceBatch`` captured when
    the load batch started. An Online batch (the default) never enumerates a
    local folder and keeps the existing verified-cache-first
    metadata/download route. A Local batch makes no request, constructs no
    metadata client and never falls through to the hosted NeuronBridge cache:
    EM geometry comes from the configured local folders only, and geometry
    those folders cannot supply stays unresolved with a practical message
    instead of being downloaded. A descriptor whose own accepted evidence is a
    user-supplied local file keeps its verified local-bytes route in either
    mode, because those bytes are the accepted source and not a hosted cache.
    """
    _check_cancel(cancel_check)
    root = (Path(cache_root).expanduser() if cache_root is not None
            else neuronbridge_cache_root())
    local_only = (geometry_source is not None and geometry_source.local
                  and remote.source.kind == "em")

    with _retrieval_slot(cancel_check):
        checksum = remote.expected_sha256
        accepted_local = bool(remote.resolution.get("local_acquisition"))
        existing = (remote.cached_path(cache_root=root)
                    if remote.resolution and (not local_only or accepted_local) else None)
        before = file_revision(existing) if existing is not None else None
        if existing and cached_file_ready(
                existing, expected_sha256=checksum, cancel_check=cancel_check):
            _check_cancel(cancel_check)
            if accepted_local:
                if file_revision(existing) != before:
                    raise ResolutionError("The local geometry file changed during verification. Retry loading it.")
                return replace(remote, local_path=str(existing), local_revision=before)
            return _record_retrieval(remote, existing, checksum, cancel_check, source_revision=before)
        if existing and checksum:
            raise ResolutionError("The local source does not match its saved checksum. "
                                  "The file was preserved; restore it or import the changed source separately.")

        if local_only:
            # Contradictory recorded evidence is still reported, but the online
            # resolution prerequisites (alignment space, lookup name) are not
            # imposed: a configured folder plus the typed ID defines this lookup.
            if remote.problems:
                raise ResolutionError("\n".join(remote.problems), status="ambiguous")
            hit = _local_geometry_hit(remote, geometry_source, cancel_check)
            if hit is None:
                raise ResolutionError(local_geometry_shortfall(remote, geometry_source))
            if on_resolved is not None:
                on_resolved(hit)
            return hit

        if not remote.resolution.get("url"):
            retained = dict(remote.resolution)
            retained_selection = remote.selected_asset
            validate_resolution_request(remote)
            supplied = [
                asset for asset in remote.source.assets
                if asset.url and asset.asset_type in ASSET_SUFFIXES
                and _asset_matches_selection(retained_selection, asset)
            ]
            nb = None if supplied else client_factory(remote.data_version)
            _check_cancel(cancel_check)
            resolved = resolve_source(remote, nb, cancel_check=cancel_check)
            if retained_selection is not None and not _same_selected_asset(
                    retained_selection, resolved.selected_asset):
                raise ResolutionError(
                    "Resolved asset identity differs from the recorded selection; "
                    "the cached or remote source was not loaded."
                )
            if retained:
                resolution = dict(resolved.resolution)
                # Locally observed bytes for a user-supplied file must never
                # become the verification expectation for a hosted asset.
                carried = ("user_resolution",) if retained.get("local_acquisition") else (
                    "retrieved_sha256", "retrieved_at", "provider_verification", "user_resolution")
                for key in carried:
                    if key in retained:
                        resolution[key] = copy.deepcopy(retained[key])
                resolved = replace(resolved, resolution=resolution)
            remote = resolved

        if on_resolved is not None:
            on_resolved(remote)

        checksum = remote.expected_sha256
        existing = remote.cached_path(cache_root=root)
        before = file_revision(existing) if existing is not None else None
        if existing and cached_file_ready(
                existing, expected_sha256=checksum, cancel_check=cancel_check):
            _check_cancel(cancel_check)
            return _record_retrieval(remote, existing, checksum, cancel_check, source_revision=before)
        if existing and checksum:
            raise ResolutionError("The local source does not match its saved checksum. "
                                  "The file was preserved; restore it or import the changed source separately.")

        filename = remote.resolution["cache_filename"]
        destination = root / filename
        if len(filename) > filename_unit_budget(root):
            destination = compact_asset_cache_path(root, filename)
            if len(destination.name) > filename_unit_budget(root):
                raise OSError("The cache folder is too deep for the download. Choose a shorter cache folder.")
        before = file_revision(destination) if destination.is_file() else None
        if existing != destination and cached_file_ready(
                destination, expected_sha256=checksum, cancel_check=cancel_check):
            _check_cancel(cancel_check)
            return _record_retrieval(remote, destination, checksum, cancel_check, source_revision=before)
        if destination.exists():
            raise ResolutionError("An existing source at the download location could not be verified. "
                                  "It was preserved; restore it or choose another download location.")
        kwargs = {"cancel_check": cancel_check}
        if checksum:
            kwargs["expected_sha256"] = checksum
        accepted = []
        if download is stream_download:
            kwargs["provider_only"] = True
            kwargs["on_accepted"] = lambda digest, revision: accepted.append((digest, revision))
        download(remote.resolution["url"], destination, **kwargs)
        _check_cancel(cancel_check)
        if accepted:
            digest, revision = accepted[0]
            return _record_retrieval(
                remote, destination, digest, cancel_check, downloaded=True,
                source_revision=revision,
            )
        return _record_retrieval(
            remote, destination, checksum, cancel_check, downloaded=True
        )


def _record_retrieval(remote, path, checksum, cancel_check, *, downloaded=False, source_revision=None):
    before = source_revision if source_revision is not None else file_revision(path)
    if checksum is None or source_revision is None:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                _check_cancel(cancel_check)
                digest.update(chunk)
        observed = digest.hexdigest()
        if checksum is not None and checksum != observed:
            from .cache import AssetChecksumError
            raise AssetChecksumError(remote.selected_asset.url, checksum, observed)
        checksum = observed
    _check_cancel(cancel_check)
    if file_revision(path) != before:
        raise ResolutionError("The local source changed during verification. Retry loading it.")
    resolution = dict(remote.resolution)
    resolution["retrieved_sha256"] = checksum
    if downloaded:
        resolution.setdefault("retrieved_at", datetime.now(timezone.utc).isoformat())
    provider_checksum = remote.selected_asset.checksum_sha256
    if provider_checksum:
        resolution["provider_verification"] = {"checksum_sha256": provider_checksum, "status": "matched"}
    return replace(remote, local_path=str(path), local_revision=before, resolution=resolution)


def bind_remote_volume_plan(batch, remote):
    """Bind a newly inspected cache file to its durable external locator.

    This runs before publication; local loader requests keep their runtime path.
    Existing user-file and generated-output backings are never modified here.
    """
    batch = copy.deepcopy(batch)
    locator = remote.resolution["url"]
    for plan in batch.publication_plans:
        backings = (plan.backing_source,) if hasattr(plan, "backing_source") else plan.backing_sources
        for backing in backings:
            bind_remote_backing(backing, locator, remote.local_path)
    return batch


def bind_remote_backing(backing, locator, cache_path):
    """Remove disposable cache/transport locators from durable backing state."""
    backing.remote_locator = ""
    for field in ("primary_path", "alternate_path"):
        if getattr(backing, field) == cache_path:
            setattr(backing, field, "")
    metadata = backing.microscopy_source_metadata
    if metadata is not None:
        backing.microscopy_source_metadata = replace(
            metadata,
            reported_primary_source_path=("" if metadata.reported_primary_source_path in {cache_path, locator}
                                          else metadata.reported_primary_source_path),
            source_members=tuple(replace(member, path="") if member.path in {cache_path, locator} else member
                                 for member in metadata.source_members),
        )


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
