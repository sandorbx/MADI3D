"""Shared, Qt/network-independent NeuronBridge evidence records (schema 1).

CSV fields are sequences, never header-keyed dictionaries. Identity is supplied
evidence, not a promise of uniqueness or permission to resolve/download an asset.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Mapping


SCHEMA_VERSION = 1
FIELD_ROLES = frozenset({
    "unknown", "number", "rank", "score", "matched_pixels", "mirror",
    "neuron_id", "line_name", "published_name", "target_kind", "library",
    "library_release", "image_id", "slide_code", "alignment_space", "sex",
    "magnification", "anatomical_area", "mounting_protocol", "neuron_type",
    "neuron_instance", "asset_id", "asset_type", "asset_url", "channel",
})


class NeuronBridgeRecordError(ValueError):
    """A record or serialized payload violates the shared data contract."""


def _string(value: Any, name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise NeuronBridgeRecordError(f"{name} must be a nonempty string.")


def _string_fields(record: Any) -> None:
    for item in fields(record):
        _string(getattr(record, item.name), item.name, optional=True)


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise NeuronBridgeRecordError(f"{name} must be an integer >= {minimum}.")


def _timestamp(value: str) -> None:
    _string(value, "imported_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NeuronBridgeRecordError("imported_at must be an ISO timestamp.") from exc
    if parsed.utcoffset() is None:
        raise NeuronBridgeRecordError("imported_at must include a timezone.")


def _json_value(value: Any) -> None:
    """Reject coercion, non-string mapping keys, and non-finite JSON numbers."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _json_value(child)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for child in value.values():
            _json_value(child)
        return
    raise NeuronBridgeRecordError("Values must be finite JSON data with string keys.")


def _sequence(record: Any, name: str, record_type: type) -> None:
    value = getattr(record, name)
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, record_type) for item in value
    ):
        raise NeuronBridgeRecordError(f"{name} must contain {record_type.__name__} records.")
    object.__setattr__(record, name, tuple(value))


def _mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NeuronBridgeRecordError("Record payload must be a mapping.")
    return dict(value)


@dataclass(frozen=True)
class BiologicalIdentity:
    """Neuron ID is opaque; an LM line name identifies a line, not one neuron."""

    neuron_id: str | None = None
    line_name: str | None = None
    neuron_type: str | None = None
    neuron_instance: str | None = None

    def __post_init__(self) -> None:
        _string_fields(self)


@dataclass(frozen=True)
class ImageIdentity:
    """An explicit NB image ID plus supplied selectors; no fabricated image ID."""

    image_id: str | None = None
    slide_code: str | None = None
    alignment_space: str | None = None
    sex: str | None = None
    magnification: str | None = None
    anatomical_area: str | None = None
    mounting_protocol: str | None = None

    def __post_init__(self) -> None:
        _string_fields(self)


@dataclass(frozen=True)
class AssetIdentity:
    """A supplied downloadable asset reference; never derived from a target label."""

    asset_id: str | None = None
    asset_type: str | None = None
    url: str | None = None
    checksum_sha256: str | None = None

    def __post_init__(self) -> None:
        _string_fields(self)


@dataclass(frozen=True)
class ChannelSelection:
    """Opaque source channel selector. Its index base is unknown unless supplied."""

    selector: str
    index_base: int | None = None

    def __post_init__(self) -> None:
        _string(self.selector, "selector")
        if self.index_base is not None and (
            type(self.index_base) is not int or self.index_base not in (0, 1)
        ):
            raise NeuronBridgeRecordError("Channel index_base must be 0, 1, or None.")


@dataclass(frozen=True)
class SourceIdentity:
    """Partial source identity, scoped by library and its separate data release.

    None/unknown means unsupplied or unresolved, never a wildcard for first-match
    lookup. Scores deliberately have no home in this record.
    """

    kind: str = "unknown"
    library: str | None = None
    library_release: str | None = None
    published_name: str | None = None
    biological: BiologicalIdentity = field(default_factory=BiologicalIdentity)
    image: ImageIdentity = field(default_factory=ImageIdentity)
    assets: tuple[AssetIdentity, ...] = ()
    channel: ChannelSelection | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("em", "lm", "unknown"):
            raise NeuronBridgeRecordError("Source kind must be em, lm, or unknown.")
        for name in ("library", "library_release", "published_name"):
            _string(getattr(self, name), name, optional=True)
        if not isinstance(self.biological, BiologicalIdentity):
            raise NeuronBridgeRecordError("biological must be BiologicalIdentity.")
        if not isinstance(self.image, ImageIdentity):
            raise NeuronBridgeRecordError("image must be ImageIdentity.")
        if self.channel is not None and not isinstance(self.channel, ChannelSelection):
            raise NeuronBridgeRecordError("channel must be ChannelSelection or None.")
        _sequence(self, "assets", AssetIdentity)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceIdentity:
        data = _mapping(value)
        data["biological"] = BiologicalIdentity(**data.get("biological", {}))
        data["image"] = ImageIdentity(**data.get("image", {}))
        data["assets"] = tuple(AssetIdentity(**item) for item in data.get("assets", ()))
        if data.get("channel") is not None:
            data["channel"] = ChannelSelection(**data["channel"])
        return cls(**data)


@dataclass(frozen=True)
class SourceParameter:
    """One actually supplied parameter. Order and repeated names are retained."""

    name: str
    value: Any

    def __post_init__(self) -> None:
        _string(self.name, "parameter name")
        _json_value(self.value)
        # Canonical JSON containers also make tuple-valued caller parameters
        # survive an actual JSON round trip as the same accepted record.
        object.__setattr__(self, "value", json.loads(json.dumps(self.value, allow_nan=False)))


@dataclass(frozen=True)
class CSVProvenance:
    filename: str
    checksum_sha256: str
    imported_at: str
    encoding: str

    def __post_init__(self) -> None:
        for item in fields(self):
            _string(getattr(self, item.name), item.name)
        _timestamp(self.imported_at)
        if len(self.checksum_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.checksum_sha256
        ):
            raise NeuronBridgeRecordError("CSV checksum must be lowercase SHA-256 hex.")


def _validate_local_execution(session, evidence):
    """Validate retained execution facts without touching a library or network."""
    if not isinstance(evidence, dict) or type(evidence.get("schema_version")) is not int or evidence["schema_version"] != 1:
        raise NeuronBridgeRecordError("Unsupported local search evidence.")
    if evidence.get("backend") != "madi3d-local" or evidence.get("completion_status") != "completed" or evidence.get("library_completeness") != "complete":
        raise NeuronBridgeRecordError("Only complete local rankings can become search results.")
    for name in ("algorithm", "reference_revision", "snapshot_id", "query_id", "library", "library_release", "data_version", "query_alignment_quality"):
        _string(evidence.get(name), name)
    for name in ("inventory_sha256", "manifest_sha256", "query_png_sha256", "query_pixel_sha256"):
        value = evidence.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise NeuronBridgeRecordError(f"Invalid local search {name}.")
    if evidence["query_id"] != session.query_reference or evidence["data_version"] != session.neuronbridge_data_version:
        raise NeuronBridgeRecordError("Local search context disagrees with its session.")
    counts = evidence.get("counts")
    if not isinstance(counts, dict):
        raise NeuronBridgeRecordError("Local search candidate counts are required.")
    for name in ("total", "examined", "failed", "matched", "retained"):
        _integer(counts.get(name), name)
    if not (counts["total"] > 0 and counts["total"] == counts["examined"] and counts["failed"] == 0 and
            0 <= counts["retained"] <= counts["matched"] <= counts["total"]):
        raise NeuronBridgeRecordError("Local search counts describe an incomplete or inconsistent ranking.")
    from .local_scorer import SearchParameters
    try:
        parameters = SearchParameters(**evidence["parameters"])
    except (ValueError, KeyError, TypeError) as exc:
        raise NeuronBridgeRecordError("Invalid retained local search parameters.") from exc
    if counts["retained"] != min(counts["matched"], parameters.result_limit) or type(evidence.get("results_truncated")) is not bool or evidence["results_truncated"] != (counts["matched"] > counts["retained"]):
        raise NeuronBridgeRecordError("Local result limit and truncation evidence disagree.")
    if not isinstance(evidence.get("query_warnings"), list) or any(not isinstance(w, str) for w in evidence["query_warnings"]):
        raise NeuronBridgeRecordError("Query warnings must be retained separately from execution status.")


@dataclass(frozen=True)
class SearchSession:
    session_id: str
    source_kind: str
    neuronbridge_data_version: str | None = None
    query_reference: str | None = None
    query_source: SourceIdentity | None = None
    parameters: tuple[SourceParameter, ...] = ()
    csv_provenance: CSVProvenance | None = None

    def __post_init__(self) -> None:
        _string(self.session_id, "session_id")
        if self.source_kind not in ("imported_csv", "precomputed", "custom_query"):
            raise NeuronBridgeRecordError("Unknown search source_kind.")
        for name in ("neuronbridge_data_version", "query_reference"):
            _string(getattr(self, name), name, optional=True)
        if self.query_source is not None and not isinstance(self.query_source, SourceIdentity):
            raise NeuronBridgeRecordError("query_source must be SourceIdentity or None.")
        _sequence(self, "parameters", SourceParameter)
        if self.csv_provenance is not None and not isinstance(self.csv_provenance, CSVProvenance):
            raise NeuronBridgeRecordError("csv_provenance must be CSVProvenance or None.")
        if self.source_kind == "imported_csv" and self.csv_provenance is None:
            raise NeuronBridgeRecordError("Imported CSV sessions require CSV provenance.")
        local = [p.value for p in self.parameters if p.name == "local_search"]
        if local:
            if len(local) != 1 or self.source_kind != "custom_query":
                raise NeuronBridgeRecordError("Local search evidence requires one custom-query execution record.")
            _validate_local_execution(self, local[0])

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        _json_value(data)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SearchSession:
        data = _mapping(value)
        if data.get("query_source") is not None:
            data["query_source"] = SourceIdentity.from_dict(data["query_source"])
        if data.get("csv_provenance") is not None:
            data["csv_provenance"] = CSVProvenance(**data["csv_provenance"])
        data["parameters"] = tuple(SourceParameter(**item) for item in data.get("parameters", ()))
        return cls(**data)


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    severity: str = "warning"
    row_index: int | None = None
    column_indexes: tuple[int, ...] = ()
    line_start: int | None = None
    line_end: int | None = None

    def __post_init__(self) -> None:
        _string(self.code, "diagnostic code")
        _string(self.message, "diagnostic message")
        if self.severity not in ("info", "warning", "error"):
            raise NeuronBridgeRecordError("Unknown diagnostic severity.")
        for name in ("row_index", "line_start", "line_end"):
            value = getattr(self, name)
            if value is not None:
                _integer(value, name, 0 if name == "row_index" else 1)
        for index in self.column_indexes:
            _integer(index, "column index")
        object.__setattr__(self, "column_indexes", tuple(self.column_indexes))


@dataclass(frozen=True)
class ResultField:
    """One ordered cell, including duplicate headers and absent trailing cells.

    raw_text=None means no cell was supplied; '' is an explicitly empty cell.
    value=None means missing or invalid; diagnostics distinguish the latter.
    A score role only enables numeric sorting; it does not identify an algorithm.
    """

    column_index: int
    header: str | None
    raw_text: str | None
    value: str | int | float | bool | None
    role: str = "unknown"

    def __post_init__(self) -> None:
        _integer(self.column_index, "column_index")
        if self.role not in FIELD_ROLES:
            raise NeuronBridgeRecordError(f"Unknown field role: {self.role}.")
        for name in ("header", "raw_text"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), str):
                raise NeuronBridgeRecordError(f"{name} must be a string or None.")
        if self.value is None:
            return
        if self.role in ("number", "rank", "matched_pixels"):
            _integer(self.value, self.role)
        elif self.role == "score":
            if type(self.value) not in (int, float) or (
                type(self.value) is float and not math.isfinite(self.value)
            ):
                raise NeuronBridgeRecordError("Score values must be finite numbers.")
        elif self.role == "mirror":
            if type(self.value) is not bool:
                raise NeuronBridgeRecordError("Mirror values must be booleans.")
        elif not isinstance(self.value, str):
            raise NeuronBridgeRecordError("Identity and unknown field values must be strings.")


@dataclass(frozen=True)
class MatchOccurrence:
    occurrence_id: str
    session_id: str
    target: SourceIdentity
    row_index: int
    fields: tuple[ResultField, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    line_start: int | None = None
    line_end: int | None = None

    def __post_init__(self) -> None:
        _string(self.occurrence_id, "occurrence_id")
        _string(self.session_id, "session_id")
        _integer(self.row_index, "row_index")
        if not isinstance(self.target, SourceIdentity):
            raise NeuronBridgeRecordError("target must be SourceIdentity.")
        _sequence(self, "fields", ResultField)
        _sequence(self, "diagnostics", Diagnostic)
        if tuple(item.column_index for item in self.fields) != tuple(range(len(self.fields))):
            raise NeuronBridgeRecordError("Fields must retain contiguous original column order.")
        for name in ("line_start", "line_end"):
            if getattr(self, name) is not None:
                _integer(getattr(self, name), name, 1)
        if self.line_start is not None and self.line_end is not None and self.line_end < self.line_start:
            raise NeuronBridgeRecordError("Invalid source line range.")

    def fields_for(self, role: str) -> tuple[ResultField, ...]:
        """Return all supplied columns of a role, retaining duplicates and order."""
        return tuple(item for item in self.fields if item.role == role)

    @property
    def scores(self) -> tuple[ResultField, ...]:
        return self.fields_for("score")

    @property
    def unrecognized_fields(self) -> tuple[ResultField, ...]:
        return self.fields_for("unknown")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MatchOccurrence:
        data = _mapping(value)
        data["target"] = SourceIdentity.from_dict(data["target"])
        data["fields"] = tuple(ResultField(**item) for item in data.get("fields", ()))
        data["diagnostics"] = tuple(Diagnostic(**item) for item in data.get("diagnostics", ()))
        return cls(**data)


@dataclass(frozen=True)
class SearchResults:
    """Versioned handoff envelope, independent of project persistence.

    Original CSV bytes include BOM, quoting, line endings and malformed tails.
    Base64 in to_dict() makes even undecodable input losslessly serializable.
    csv_parse_complete says whether all record boundaries could be parsed, not
    whether identities or numerical values are valid.
    """

    session: SearchSession
    occurrences: tuple[MatchOccurrence, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    csv_headers: tuple[str, ...] = ()
    csv_bytes: bytes | None = None
    csv_parse_complete: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.session, SearchSession):
            raise NeuronBridgeRecordError("session must be SearchSession.")
        _sequence(self, "occurrences", MatchOccurrence)
        _sequence(self, "diagnostics", Diagnostic)
        if any(not isinstance(header, str) for header in self.csv_headers):
            raise NeuronBridgeRecordError("CSV headers must be strings.")
        object.__setattr__(self, "csv_headers", tuple(self.csv_headers))
        if any(item.session_id != self.session.session_id for item in self.occurrences):
            raise NeuronBridgeRecordError("Every occurrence must belong to this session.")
        local = next((p.value for p in self.session.parameters if p.name == "local_search"), None)
        if local and local["counts"]["retained"] != len(self.occurrences):
            raise NeuronBridgeRecordError("Local search retained count disagrees with its hit records.")
        ids = [item.occurrence_id for item in self.occurrences]
        if len(set(ids)) != len(ids):
            raise NeuronBridgeRecordError("Occurrence IDs must be unique within a result set.")
        order = [item.row_index for item in self.occurrences]
        if order != sorted(set(order)):
            raise NeuronBridgeRecordError("Occurrences must retain unique original row order.")
        if self.csv_bytes is not None:
            if not isinstance(self.csv_bytes, bytes) or self.session.csv_provenance is None:
                raise NeuronBridgeRecordError("CSV bytes require bytes and CSV provenance.")
            if hashlib.sha256(self.csv_bytes).hexdigest() != self.session.csv_provenance.checksum_sha256:
                raise NeuronBridgeRecordError("CSV bytes do not match the source checksum.")
            if type(self.csv_parse_complete) is not bool:
                raise NeuronBridgeRecordError("CSV bytes require an explicit parse status.")
            for occurrence in self.occurrences:
                if len(occurrence.fields) < len(self.csv_headers) or any(
                    item.header != (
                        self.csv_headers[item.column_index]
                        if item.column_index < len(self.csv_headers) else None
                    ) for item in occurrence.fields
                ):
                    raise NeuronBridgeRecordError("Occurrence columns must match the ordered CSV headers.")
        elif self.csv_parse_complete is not None or self.csv_headers:
            raise NeuronBridgeRecordError("CSV parsing evidence requires original CSV bytes.")

    @property
    def all_diagnostics(self) -> tuple[Diagnostic, ...]:
        return self.diagnostics + tuple(
            diagnostic for item in self.occurrences for diagnostic in item.diagnostics
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = SCHEMA_VERSION
        data["csv_bytes_base64"] = (
            base64.b64encode(self.csv_bytes).decode("ascii")
            if self.csv_bytes is not None else None
        )
        del data["csv_bytes"]
        _json_value(data)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SearchResults:
        data = _mapping(value)
        if "csv_bytes" in data:
            raise NeuronBridgeRecordError("Serialized CSV evidence must use csv_bytes_base64.")
        version = data.pop("schema_version", None)
        if type(version) is not int or version != SCHEMA_VERSION:
            raise NeuronBridgeRecordError(f"Unsupported NeuronBridge schema version: {version}.")
        data["session"] = SearchSession.from_dict(data["session"])
        data["occurrences"] = tuple(MatchOccurrence.from_dict(item) for item in data.get("occurrences", ()))
        data["diagnostics"] = tuple(Diagnostic(**item) for item in data.get("diagnostics", ()))
        encoded = data.pop("csv_bytes_base64", None)
        try:
            if encoded is not None and not isinstance(encoded, str):
                raise ValueError("Expected a base64 string.")
            data["csv_bytes"] = base64.b64decode(encoded, validate=True) if encoded is not None else None
        except (ValueError, binascii.Error) as exc:
            raise NeuronBridgeRecordError("Invalid CSV base64 evidence.") from exc
        return cls(**data)
