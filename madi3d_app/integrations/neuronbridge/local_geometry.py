"""User-supplied local neuron geometry locations for known NeuronBridge IDs.

Machine-local configuration, source-mode vocabulary and filename lookup only.
The authoritative neuron/body/root ID comes from the typed result record; a
filename is used solely to find a candidate file for that already-known ID.
This module never derives biological identity from filename annotations, never
searches recursively, and never copies or mutates the user's files.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# Geometry representations already meaningful to the existing NeuronBridge
# asset retrieval path. Extension comparison is case-insensitive, so a
# companion such as "10001_18U.obj.ive" keeps its ".ive" suffix and is never
# selected as OBJ geometry.
GEOMETRY_ASSET_TYPES = {".obj": "AlignedBodyOBJ", ".swc": "AlignedBodySWC"}
GEOMETRY_SUFFIXES = {asset_type: suffix for suffix, asset_type in GEOMETRY_ASSET_TYPES.items()}

# Exactly one EM mesh-source mode is captured per retrieval batch and never
# changes during it. An "online" batch uses the NeuronBridge metadata/cache/
# download route and never enumerates a local folder; a "local" batch resolves
# OBJ/SWC from the configured folders only. There is no local->online fallback.
ONLINE = "online"
LOCAL = "local"
GEOMETRY_SOURCE_MODES = (ONLINE, LOCAL)

MATCHED = "matched"
MISSING = "missing"
AMBIGUOUS = "ambiguous"
UNAVAILABLE = "unavailable"

# A filename token boundary is the start or end of the stem, or any character
# that is not ASCII alphanumeric. Unicode letters are therefore boundaries: ID
# "1000" matches "café1000_18U.obj" and never "abc1000x.obj".
_TOKEN_SEPARATORS = re.compile(r"[^0-9A-Za-z]+")
_IDENTIFIER_TOKEN = re.compile(r"[0-9A-Za-z]+")


def geometry_source_label(library):
    """Return the readable name of one geometry-source library.

    Only separators are opened for a person: ``FlyEM_Hemibrain_v1.2.1`` reads as
    "FlyEM Hemibrain v1.2.1". Case, version text and every character that carries
    identity are preserved, and the stored name stays authoritative. An unknown
    or empty name is returned trimmed rather than invented.
    """
    return " ".join(str(library or "").replace("_", " ").split())


def normalise_geometry_source_mode(value):
    """Return a usable batch mode for a stored preference value.

    Anything unrecognised becomes the default Online mode: a damaged preference
    must never make a retrieval batch silently stop working.
    """
    mode = str(value or "").strip().lower()
    return mode if mode in GEOMETRY_SOURCE_MODES else ONLINE


def geometry_asset_type(name):
    """Return the represented asset type for a filename, or None."""
    return GEOMETRY_ASSET_TYPES.get(Path(str(name)).suffix.lower())


def filename_stem(name):
    """Return a geometry filename without its represented extension."""
    text = str(name)
    suffix = Path(text).suffix
    return text[:len(text) - len(suffix)] if suffix else text


def identifier_token_matcher(identifier):
    """Matcher for one authoritative identifier occurring as a complete token.

    The identifier is escaped before it enters the pattern and always stays a
    string, so regex metacharacters, leading zeros and long FlyWire/BANC IDs
    remain exact text and can never widen the match.
    """
    return re.compile(f"(?<![0-9A-Za-z]){re.escape(str(identifier))}(?![0-9A-Za-z])")


def filename_matches_identifier(name, identifier):
    """True when the authoritative ID occurs anywhere in the stem as one token.

    The occurrence is bounded by the start or end of the stem, or by a
    character that is not ASCII alphanumeric. ID "1000" therefore matches
    "processed_1000_18U.obj" and "1000-extra.swc", and never
    "processed_10001_18U.obj" or "abc1000x.obj".
    """
    text, token = str(identifier), str(name)
    if not text:
        return False
    return identifier_token_matcher(text).search(filename_stem(token)) is not None


def _list_children(root):
    return Path(root).iterdir()


@dataclass(frozen=True)
class LocalGeometryResult:
    """Outcome of one identifier/representation lookup.

    ``candidates`` and ``message`` are diagnostics only; they never select a
    file, rewrite identity or replace the existing retrieval route.
    """

    status: str
    path: Path | None = None
    candidates: tuple[str, ...] = ()
    message: str = ""


class GeometrySourceBatch:
    """Immutable EM geometry-source intent for one "Load selected" batch.

    The controller captures the mode, the required geometry-source libraries,
    the configured local folders and the requested representations once, before
    any retrieval worker starts; workers read neither live widgets nor
    preferences for this batch. The mode is fixed for the batch lifetime, so a
    preference change mid-batch cannot mix sources.

    An Online batch never consults a configured folder. For a Local batch each
    configured directory is listed at most once for the whole batch, on first
    use and only for direct children, so a batch touching many result IDs never
    rescans a large collection per result. Only requested
    (library, identifier, representation) matches are retained; a None
    representation requests both OBJ and SWC for an unselected result.
    Subdirectories such as ``.madicache`` or ``.covercache`` are never traversed.
    Callers perform lookups on their worker thread; the Qt GUI thread must not
    enumerate a configured directory through this batch.
    """

    def __init__(self, mode, sources, requested):
        self.mode = normalise_geometry_source_mode(mode)
        self._sources = {}
        for library, root in dict(sources or {}).items():
            library = str(library or "").strip()
            root = str(root or "").strip()
            if library and root:
                self._sources[library] = Path(root).expanduser()
        wanted = {}
        for library, identifier, asset_type in requested:
            root = self._sources.get(library)
            if root is not None and isinstance(identifier, str) and identifier:
                types = GEOMETRY_SUFFIXES if asset_type is None else (asset_type,)
                for kind in types:
                    if kind in GEOMETRY_SUFFIXES:
                        wanted.setdefault(root, {}).setdefault(kind, set()).add(identifier)
        self._wanted = {}
        for root, by_type in wanted.items():
            self._wanted[root] = {
                asset_type: (
                    frozenset(token for token in tokens if _IDENTIFIER_TOKEN.fullmatch(token)),
                    tuple((token, identifier_token_matcher(token)) for token in tokens
                          if not _IDENTIFIER_TOKEN.fullmatch(token)),
                ) for asset_type, tokens in by_type.items()
            }
        self._directories = {}

    @property
    def local(self):
        """True when this batch resolves EM geometry from local folders only."""
        return self.mode == LOCAL

    def configured(self, library):
        return str(library or "").strip() in self._sources

    def _directory(self, root):
        cached = self._directories.get(root)
        if cached is not None:
            return cached
        index = {}
        error = ""
        try:
            wanted = self._wanted.get(root, {})
            for entry in _list_children(root):
                asset_type = geometry_asset_type(entry.name)
                requested = wanted.get(asset_type) if asset_type else None
                if requested is None or not entry.is_file():
                    continue
                simple, compound = requested
                stem = filename_stem(entry.name)
                found = [token for token in _TOKEN_SEPARATORS.split(stem) if token in simple]
                found.extend(token for token, matcher in compound if matcher.search(stem))
                for identifier in dict.fromkeys(found):
                    index.setdefault(identifier, {}).setdefault(asset_type, []).append(entry)
        except OSError as exc:
            index = {}
            error = f"{root}: {exc.strerror or exc}"
        result = (index, error)
        self._directories[root] = result
        return result

    def lookup(self, library, identifier, asset_type):
        """Locate one opaque identifier's file for the requested representation.

        The identifier stays an exact string; it is never converted to an
        integer, so long FlyWire/BANC IDs remain distinct. Matching honours the
        complete-token rule, and two eligible files for the same identifier and
        representation are ambiguous: they are never resolved by name ordering.
        """
        root = self._sources.get(str(library or "").strip())
        if root is None or not isinstance(identifier, str) or not identifier:
            return LocalGeometryResult(MISSING)
        if asset_type not in GEOMETRY_SUFFIXES:
            return LocalGeometryResult(MISSING)
        index, error = self._directory(root)
        if error:
            return LocalGeometryResult(UNAVAILABLE, message=error)
        candidates = tuple(index.get(identifier, {}).get(asset_type, ()))
        if len(candidates) == 1:
            return LocalGeometryResult(MATCHED, path=candidates[0])
        if candidates:
            names = tuple(sorted(entry.name for entry in candidates))
            return LocalGeometryResult(
                AMBIGUOUS, candidates=names,
                message=f"{root}: {len(names)} local files match identifier {identifier}",
            )
        return LocalGeometryResult(MISSING)
