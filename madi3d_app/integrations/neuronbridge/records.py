"""Shared, Qt/network-independent NeuronBridge search records (schema 2).

CSV fields are sequences, never header-keyed dictionaries. Identity is supplied
evidence, not a promise of uniqueness or permission to resolve/download an asset.
"""
from __future__ import annotations

import base64
import binascii
import csv
import io
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Mapping


SCHEMA_VERSION = 2
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
    provider: str | None = "NeuronBridge"
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
        for name in ("provider", "library", "library_release", "published_name"):
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
    parser_revision: str = "madi3d-nb-csv-v1"
    interpretation_revision: str = "madi3d-nb-identity-v1"

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
    for name in ("algorithm", "reference_revision", "implementation_revision", "snapshot_id", "query_id", "library", "library_release", "data_version", "query_alignment_quality"):
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
    parameter_names = [p.name for p in session.parameters]
    if len(parameter_names) != len(set(parameter_names)) or set(parameter_names) != {f.name for f in fields(SearchParameters)}:
        raise NeuronBridgeRecordError("Local search must retain every executed parameter exactly once.")
    try:
        parameters = SearchParameters(**{p.name: p.value for p in session.parameters})
    except (ValueError, KeyError, TypeError) as exc:
        raise NeuronBridgeRecordError("Invalid retained local search parameters.") from exc
    if counts["retained"] != min(counts["matched"], parameters.result_limit) or type(evidence.get("results_truncated")) is not bool or evidence["results_truncated"] != (counts["matched"] > counts["retained"]):
        raise NeuronBridgeRecordError("Local result limit and truncation evidence disagree.")


@dataclass(frozen=True)
class SearchSession:
    session_id: str
    source_kind: str
    neuronbridge_data_version: str | None = None
    query_reference: str | None = None
    query_source: SourceIdentity | None = None
    parameters: tuple[SourceParameter, ...] = ()
    csv_provenance: CSVProvenance | None = None
    # Shared execution facts and separately retained external observations.
    context: dict[str, Any] = field(default_factory=dict)
    external_evidence: dict[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()

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
        for name in ("context", "external_evidence"):
            if not isinstance(getattr(self, name), dict):
                raise NeuronBridgeRecordError(f"{name} must be a mapping.")
            _json_value(getattr(self, name))
            object.__setattr__(self, name, json.loads(json.dumps(getattr(self, name), allow_nan=False)))
        _sequence(self, "diagnostics", Diagnostic)
        if self.context.get("backend") == "madi3d-local":
            if self.source_kind != "custom_query":
                raise NeuronBridgeRecordError("Local execution requires a custom-query session.")
            _validate_local_execution(self, self.context)

    def evidence_for(self, occurrence):
        """Observed evidence only; never a refreshed identity or score lookup."""
        record = self.external_evidence.get(occurrence.evidence_ref, {})
        if occurrence.evidence_index is not None:
            if record.get("scope") == "selected_results":
                return record["selected_results"][str(occurrence.evidence_index)]
            return record["payload"]["results"][occurrence.evidence_index]
        if "references" in record:
            observed = {scope: self.external_evidence[key] for scope, key in record["references"].items()}
            identity = observed.get("identity", {})
            if "response_ref" in identity:
                observed["identity"] = dict(identity, observed_response=self.external_evidence[identity["response_ref"]])
            return observed
        return record

    def historical_fields_for(self, occurrence):
        """Materialize retained schema-1 column observations for evidence views only."""
        table = self.external_evidence.get("schema1_fields", {})
        row = table.get("occurrences", {}).get(occurrence.occurrence_id)
        if row is None:
            return ()
        try:
            if (len(row["columns"]) != len(row["cells"]) or
                    any(type(c) is not int or not 0 <= c < len(table["columns"]) for c in row["columns"]) or
                    any(not k.isdecimal() or str(int(k)) != k or int(k) >= len(row["cells"]) for k in row["values"])):
                raise NeuronBridgeRecordError("Invalid historical column references.")
            return tuple(ResultField(**table["columns"][column], raw_text=raw,
                                     value=row["values"].get(str(i), raw))
                         for i, (column, raw) in enumerate(zip(row["columns"], row["cells"], strict=True)))
        except (KeyError, TypeError, ValueError) as exc:
            raise NeuronBridgeRecordError("Invalid historical field evidence.") from exc

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
        data["diagnostics"] = tuple(Diagnostic(**item) for item in data.get("diagnostics", ()))
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
    """Imported CSV cell view, including duplicate headers and absent cells.

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
class HitMetric:
    """Named numerical observation; meaning is explicit, never inferred from Score."""

    name: str
    value: int | float | None
    meaning: str | None = None

    def __post_init__(self):
        _string(self.name, "metric name")
        _string(self.meaning, "metric meaning", optional=True)
        if self.value is not None and (type(self.value) not in (int, float) or
                not math.isfinite(self.value)):
            raise NeuronBridgeRecordError("Metrics must be finite numbers or unknown.")


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

    rank: int | None = None
    metrics: tuple[HitMetric, ...] = ()
    matched_pixels: int | None = None
    overlap: float | None = None
    mirrored: bool | None = None
    shift: tuple[int, int] | None = None
    member_id: str | None = None
    candidate_ordinal: int | None = None
    input_hashes: dict[str, str] = field(default_factory=dict)
    evidence_ref: str | None = None
    evidence_index: int | None = None

    def __post_init__(self) -> None:
        _sequence(self, "metrics", HitMetric)
        for name in ("rank", "matched_pixels", "candidate_ordinal", "evidence_index"):
            if getattr(self, name) is not None:
                _integer(getattr(self, name), name)
        if self.overlap is not None and (type(self.overlap) not in (int, float) or
                not math.isfinite(self.overlap) or not 0 <= self.overlap <= 1):
            raise NeuronBridgeRecordError("Overlap must be a finite fraction.")
        if self.mirrored is not None and type(self.mirrored) is not bool:
            raise NeuronBridgeRecordError("Mirrored must be bool or unknown.")
        if self.shift is not None:
            if len(self.shift) != 2 or any(type(v) is not int for v in self.shift):
                raise NeuronBridgeRecordError("Shift must contain two integer pixel offsets.")
            object.__setattr__(self, "shift", tuple(self.shift))
        for name in ("member_id", "evidence_ref"):
            _string(getattr(self, name), name, optional=True)
        if not isinstance(self.input_hashes, dict):
            raise NeuronBridgeRecordError("Input hashes must be a mapping.")
        for key, value in self.input_hashes.items():
            _string(key, "input hash name")
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise NeuronBridgeRecordError("Input hashes must be SHA-256 hex.")
        object.__setattr__(self, "input_hashes", dict(self.input_hashes))
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
        data = asdict(self)
        if not self.fields:
            del data["fields"]
        return {key: value for key, value in data.items() if value is not None and value != () and value != {}}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MatchOccurrence:
        data = _mapping(value)
        data["target"] = SourceIdentity.from_dict(data["target"])
        data["fields"] = tuple(ResultField(**item) for item in data.get("fields", ()))
        data["metrics"] = tuple(HitMetric(**item) for item in data.get("metrics", ()))
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
    # Portable evidence membership, separate from immutable search context.
    subset: dict[str, Any] | None = None

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
        for occurrence in self.occurrences:
            self.session.historical_fields_for(occurrence)
            if occurrence.fields and self.session.source_kind != "imported_csv":
                raise NeuronBridgeRecordError("Native hits cannot contain CSV fields.")
            if occurrence.evidence_ref is not None:
                if occurrence.evidence_ref not in self.session.external_evidence:
                    raise NeuronBridgeRecordError("Missing hit evidence reference.")
                try:
                    self.session.evidence_for(occurrence)
                except (KeyError, IndexError, TypeError) as exc:
                    raise NeuronBridgeRecordError("Invalid hit evidence position.") from exc
        counts = self.session.context.get("counts")
        if self.subset is not None:
            subset = self.subset
            ids = sorted(o.occurrence_id for o in self.occurrences)
            if (subset.get("scope") != "selected_hits" or subset.get("complete_session") is not False
                    or subset.get("included_hit_ids") != ids
                    or type(subset.get("included_hit_count")) is not int
                    or subset["included_hit_count"] != len(ids)):
                raise NeuronBridgeRecordError("Invalid selected-hit subset membership.")
            _integer(subset.get("reported_retained_count"), "reported retained count")
            if subset["reported_retained_count"] < len(ids) or self.csv_bytes is not None:
                raise NeuronBridgeRecordError("A hit subset cannot contain a whole result CSV or exceed reported counts.")
            object.__setattr__(self, "subset", json.loads(json.dumps(subset, allow_nan=False)))
        if counts is not None and (not isinstance(counts, dict) or
                type(counts.get("retained")) is not int or counts["retained"] != (
                    self.subset["reported_retained_count"] if self.subset else len(self.occurrences))):
            raise NeuronBridgeRecordError("Search retained count disagrees with its hit records.")
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
        return self.session.diagnostics + self.diagnostics + tuple(
            diagnostic for item in self.occurrences for diagnostic in item.diagnostics
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema_version": SCHEMA_VERSION,
            "session": self.session.to_dict(),
            "occurrences": [],
            "diagnostics": [asdict(d) for d in self.diagnostics],
        }
        if self.csv_bytes is not None:
            data.update({
                "csv_header_count": len(self.csv_headers),
                "csv_parse_complete": self.csv_parse_complete,
                "csv_bytes_base64": base64.b64encode(self.csv_bytes).decode("ascii"),
            })
        if self.subset is not None:
            data["subset"] = json.loads(json.dumps(self.subset, allow_nan=False))
        # CSV definitions and raw bytes occur once. Persist interpretations rather
        # than rerunning today's identity/number inference during offline reload.
        columns = {i: {"header": header, "role": None} for i, header in enumerate(self.csv_headers)}
        for occurrence in self.occurrences:
            hit = occurrence.to_dict()
            if self.csv_bytes is not None:
                hit.pop("fields", None)
            if self.csv_bytes is not None:
                hit["csv_interpretations"] = {str(f.column_index): f.value for f in occurrence.fields
                                              if f.value != f.raw_text or type(f.value) is not type(f.raw_text)}
                for f in occurrence.fields:
                    descriptor = {"header": f.header, "role": f.role}
                    if f.column_index in columns and columns[f.column_index]["role"] is not None and columns[f.column_index] != descriptor:
                        raise NeuronBridgeRecordError("CSV interpretation differs between rows.")
                    columns[f.column_index] = descriptor
            data["occurrences"].append(hit)
        if columns:
            data["csv_columns"] = [columns[i] for i in sorted(columns)]
        _json_value(data)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SearchResults:
        data = _mapping(value)
        if "csv_bytes" in data:
            raise NeuronBridgeRecordError("Serialized CSV evidence must use csv_bytes_base64.")
        version = data.pop("schema_version", None)
        if type(version) is int and version == 1:
            from .record_migration import migrate_schema1
            return migrate_schema1(data)
        if type(version) is not int or version != SCHEMA_VERSION:
            raise NeuronBridgeRecordError(f"Unsupported NeuronBridge schema version: {version}.")
        data["session"] = SearchSession.from_dict(data["session"])
        hits = data.pop("occurrences", ())
        columns = data.pop("csv_columns", ())
        header_count = data.pop("csv_header_count", 0)
        _integer(header_count, "CSV header count")
        if header_count > len(columns):
            raise NeuronBridgeRecordError("Missing CSV column descriptors.")
        data["csv_headers"] = tuple(c["header"] for c in columns[:header_count])
        data["diagnostics"] = tuple(Diagnostic(**item) for item in data.get("diagnostics", ()))
        encoded = data.pop("csv_bytes_base64", None)
        try:
            if encoded is not None and not isinstance(encoded, str):
                raise ValueError("Expected a base64 string.")
            data["csv_bytes"] = base64.b64decode(encoded, validate=True) if encoded is not None else None
        except (ValueError, binascii.Error) as exc:
            raise NeuronBridgeRecordError("Invalid CSV base64 evidence.") from exc
        raw_rows = []
        if data["csv_bytes"] is not None and hits:
            try:
                text = data["csv_bytes"].decode(data["session"].csv_provenance.encoding).removeprefix("\ufeff")
                reader = csv.reader(io.StringIO(text, newline=""), strict=True)
                if tuple(next(reader)) != data["csv_headers"]:
                    raise NeuronBridgeRecordError("CSV columns disagree with original header.")
                source_index = -1
                for hit in hits:
                    row_index = hit["row_index"]
                    _integer(row_index, "row_index")
                    if row_index <= source_index:
                        raise NeuronBridgeRecordError("CSV occurrences must retain original row order.")
                    while source_index < row_index:
                        row = next(reader)
                        source_index += 1
                    raw_rows.append(row)
            except (UnicodeError, LookupError, csv.Error, StopIteration) as exc:
                raise NeuronBridgeRecordError("CSV source does not contain the retained rows.") from exc
        occurrences = []
        for position, payload in enumerate(hits):
            hit = dict(payload)
            interpretations = hit.pop("csv_interpretations", None)
            if interpretations is not None:
                if position >= len(raw_rows):
                    raise NeuronBridgeRecordError("Invalid CSV interpretation row.")
                raw = raw_rows[position]
                width = max(len(raw), len(data.get("csv_headers", ())))
                if width > len(columns) or any(not k.isdecimal() or str(int(k)) != k or int(k) >= width for k in interpretations):
                    raise NeuronBridgeRecordError("Invalid CSV interpretation width.")
                hit["fields"] = [dict(column_index=i, header=columns[i]["header"], role=columns[i]["role"],
                    raw_text=raw[i] if i < len(raw) else None,
                    value=interpretations.get(str(i), raw[i] if i < len(raw) else None)) for i in range(width)]
            elif data["csv_bytes"] is not None:
                raise NeuronBridgeRecordError("Missing retained CSV interpretation.")
            occurrences.append(MatchOccurrence.from_dict(hit))
        data["occurrences"] = tuple(occurrences)
        return cls(**data)
