"""Persistent remote asset descriptors. These values are never filesystem paths."""
from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .records import AssetIdentity, SourceIdentity


@dataclass(frozen=True)
class RemoteSource:
    source: SourceIdentity
    data_version: str | None = None
    resolution: Mapping[str, Any] = field(default_factory=dict)
    local_path: str = ""
    problems: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.source, SourceIdentity):
            raise ValueError("NeuronBridge remote source requires a SourceIdentity.")
        if self.data_version is not None and (
            not isinstance(self.data_version, str) or not self.data_version.strip()
        ):
            raise ValueError("NeuronBridge data version must be a nonempty string.")
        resolution = copy.deepcopy(dict(self.resolution))
        retrieved = resolution.get("retrieved_sha256")
        if retrieved is not None and (not isinstance(retrieved, str) or not re.fullmatch(r"[0-9a-f]{64}", retrieved)):
            raise ValueError("Invalid retrieved asset checksum.")
        json.dumps(resolution, allow_nan=False)
        filename = resolution.get("cache_filename")
        if filename and (not isinstance(filename, str) or not re.fullmatch(
            r"nb-[0-9a-f]{64}\.(obj|swc|h5j)", filename,
        )):
            raise ValueError("NeuronBridge cache filename must be a versioned asset key.")
        if resolution and not all(resolution.get(k) for k in (
            "asset_type", "url", "cache_filename", "resolved_at",
        )):
            raise ValueError("Incomplete NeuronBridge asset resolution.")
        if "selected_asset" in resolution:
            asset = AssetIdentity(**resolution["selected_asset"])
            if asset.asset_type != resolution["asset_type"] or asset.url != resolution["url"]:
                raise ValueError("Selected asset conflicts with the resolution.")
        if not isinstance(self.local_path, str):
            raise ValueError("NeuronBridge local cache hint must be a string.")
        object.__setattr__(self, "resolution", resolution)
        if any(not isinstance(problem, str) for problem in self.problems):
            raise ValueError("Remote-source diagnostics must be text.")
        object.__setattr__(self, "problems", tuple(self.problems))

    def to_dict(self):
        return {
            "type": "neuronbridge", "version": 1,
            "source": json.loads(json.dumps(self.source.to_dict())), "data_version": self.data_version,
            "resolution": copy.deepcopy(dict(self.resolution)),
            "local_path": self.local_path,
            "problems": list(self.problems),
        }

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        if data.pop("type", None) != "neuronbridge" or data.pop("version", None) != 1:
            raise ValueError("Unsupported NeuronBridge remote descriptor.")
        data["source"] = SourceIdentity.from_dict(data["source"])
        return cls(**data)

    def cached_path(self):
        """Cheap existence/stat probe only; retrieval workers verify integrity."""
        from .cache import cached_file_ready, neuronbridge_cache_root

        candidates = [Path(self.local_path)] if self.local_path else []
        filename = self.resolution.get("cache_filename")
        if filename:
            candidates.append(neuronbridge_cache_root() / filename)
        return next((path for path in candidates if cached_file_ready(path)), None)

    @property
    def selected_asset(self):
        if not self.resolution:
            return None
        if "selected_asset" in self.resolution:
            return AssetIdentity(**self.resolution["selected_asset"])
        # Earlier NB-01B records identify selection by type and URL. Bind only
        # compatible evidence; never collect IDs/checksums from sibling assets.
        asset_type, url = self.resolution["asset_type"], self.resolution["url"]
        asset_id = self.resolution.get("api_metadata", {}).get("files", {}).get(asset_type)
        matches = [a for a in self.source.assets
                   if a.asset_type in (None, asset_type) and a.url in (None, url)
                   and (not asset_id or a.asset_id in (None, asset_id))]
        if len(matches) > 1:
            raise ValueError("Conflicting asset evidence requires an explicit selection.")
        evidence = matches[0] if matches else AssetIdentity()
        return AssetIdentity(asset_type=asset_type, url=url,
                             asset_id=asset_id or evidence.asset_id,
                             checksum_sha256=evidence.checksum_sha256)

    @property
    def expected_sha256(self):
        asset = self.selected_asset
        return (asset.checksum_sha256 if asset else None) or self.resolution.get("retrieved_sha256")


def remote_from_metadata(metadata):
    value = (metadata or {}).get("neuronbridge", {}).get("remote_source")
    return RemoteSource.from_dict(value) if value is not None else None


def same_source_identity(first, second):
    """Require a reliable shared anchor and no contradictory supplied evidence."""
    if first.kind != second.kind or first.kind == "unknown":
        return False

    def flatten(value, prefix=""):
        out = {}
        for key, child in value.items():
            if isinstance(child, dict):
                out.update(flatten(child, prefix + key + "."))
            elif child is not None and child != [] and child != ():
                out[prefix + key] = child
        return out

    left, right = flatten(first.to_dict()), flatten(second.to_dict())
    if any(left[k] != right[k] for k in left.keys() & right.keys()):
        return False
    if first.kind == "lm" and (first.channel is None or second.channel is None):
        return False
    if first.image.image_id and first.image.image_id == second.image.image_id:
        return bool(first.library and first.library == second.library)
    urls1 = {a.url for a in first.assets if a.url}
    urls2 = {a.url for a in second.assets if a.url}
    # Names and historical cache filenames are insufficient identity.
    return bool(urls1 & urls2)


def source_path_only(value):
    """Normalize local paths, excluding typed and historical remote placeholders."""
    if isinstance(value, RemoteSource):
        return ""
    if isinstance(value, (tuple, list)) and value:
        if value[0] in ("nb", "nb_lm"):
            return ""
        value = value[0]
    if value is None or value == "":
        return ""
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError("A local source must be a path, not a remote descriptor.")
    text = os.fspath(value).strip()
    return os.path.abspath(text) if text else ""
