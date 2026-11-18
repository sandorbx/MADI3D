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
import struct
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


# Identity carriers retired by the installed-library search route: a new
# session names the distribution by library ID and installed archive digest,
# never by a snapshot ID or an inventory/manifest hash.
_RETIRED_LOCAL_EVIDENCE_FIELDS = frozenset({
    "snapshot_id", "source_snapshot_id", "inventory_sha256", "source_inventory_sha256",
    "manifest_sha256", "membership_sha256"})


def _validate_local_execution(session, evidence):
    """Validate retained execution facts without touching a library or network."""
    if not isinstance(evidence, dict) or type(evidence.get("schema_version")) is not int \
            or evidence["schema_version"] not in (1, 2, 3):
        raise NeuronBridgeRecordError("Unsupported local search evidence.")
    if evidence.get("backend") != "madi3d-local" or evidence.get("completion_status") != "completed" or evidence.get("library_completeness") != "complete":
        raise NeuronBridgeRecordError("Only complete local rankings can become search results.")
    version = evidence["schema_version"]
    historical = evidence.get("historical_provenance")
    if historical is not None:
        if (not isinstance(historical, dict) or set(historical) != {"source_evidence_version", "population"}
                or historical["source_evidence_version"] not in (1, 2)
                or not isinstance(historical["population"], list) or len(historical["population"]) > 6
                or any(not isinstance(fact, dict) or set(fact) != {"kind", "value"}
                       or not all(isinstance(v, str) and v for v in fact.values())
                       for fact in historical["population"])
                or len(json.dumps(historical).encode("utf-8")) > 32 * 1024):
            raise NeuronBridgeRecordError("Invalid bounded historical population provenance.")
    names = ["algorithm", "reference_revision", "implementation_revision", "query_id",
             "library", "library_release", "data_version", "query_alignment_quality"]
    if version == 1:
        names.insert(3, "snapshot_id")
    else:
        if not evidence.get("historical_provenance"):
            names += ["provider", "representation"]
    for name in names:
        _string(evidence.get(name), name)
    hexes = ["query_png_sha256", "query_pixel_sha256"]
    if version == 1:
        hexes = ["inventory_sha256", "manifest_sha256"] + hexes
    elif version == 2:
        for name in ("library_id", "source_snapshot_id"):
            _string(evidence.get(name), name, optional=True)
        # At least one published/installed package digest pins which collection
        # content was searched: the transferred archive digest, the published
        # inventory digest, or both.
        hexes = ["archive_sha256", "source_inventory_sha256", "membership_sha256"] + hexes
    else:
        # A validated historical_provenance block marks the normalized form of a
        # supported schema-1/2 session: the managed distribution identity that
        # postdates it cannot be recovered and is checked only when the older
        # record happened to retain it. Without that historical assertion, new
        # local results require the canonical managed distribution identity.
        if historical is None:
            _string(evidence.get("library_id"), "library_id")
            hexes.insert(0, "archive_sha256")
        else:
            _string(evidence.get("library_id"), "library_id", optional=True)
            if evidence.get("archive_sha256") is not None:
                hexes.insert(0, "archive_sha256")
        retired = sorted(name for name in _RETIRED_LOCAL_EVIDENCE_FIELDS
                         if evidence.get(name) is not None)
        if retired:
            raise NeuronBridgeRecordError(
                "Installed-library search evidence cannot retain retired fields: "
                + ", ".join(retired) + ".")
    for name in hexes:
        value = evidence.get(name)
        if value is None and version == 2 and name in {"archive_sha256", "source_inventory_sha256", "membership_sha256"}:
            continue
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise NeuronBridgeRecordError(f"Invalid local search {name}.")
    if version == 2 and not any(evidence.get(name) for name in ("archive_sha256", "source_inventory_sha256")):
        raise NeuronBridgeRecordError(
            "Local search must record the installed distribution's package digest.")
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
    if getattr(session, "shape_execution", None) is not None:
        # A completed shape stage republishes only the emitted shortlist rows,
        # so a shape-cohort session may honestly retain fewer than the limit.
        retained_ok = 0 <= counts["retained"] <= min(counts["matched"], parameters.result_limit)
    else:
        retained_ok = counts["retained"] == min(counts["matched"], parameters.result_limit)
    if not retained_ok or type(evidence.get("results_truncated")) is not bool or evidence["results_truncated"] != (counts["matched"] > counts["retained"]):
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
    gradient_execution: GradientExecution | None = None
    shape_execution: ShapeExecution | None = None

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
        if self.gradient_execution is not None and not isinstance(self.gradient_execution, GradientExecution):
            raise NeuronBridgeRecordError("gradient_execution must be GradientExecution or None.")
        if self.shape_execution is not None and not isinstance(self.shape_execution, ShapeExecution):
            raise NeuronBridgeRecordError("shape_execution must be ShapeExecution or None.")
        if (self.gradient_execution is not None and self.shape_execution is not None):
            raise NeuronBridgeRecordError("A session records at most one reranking generation.")
        if self.context.get("backend") == "madi3d-local":
            if self.source_kind != "custom_query":
                raise NeuronBridgeRecordError("Local execution requires a custom-query session.")
            _validate_local_execution(self, self.context)
            if self.gradient_execution is not None:
                reference = self.gradient_execution.bundle_reference
                if (reference["query_id"] != self.query_reference or
                        reference["query_png_sha256"] != self.context["query_png_sha256"] or
                        reference["query_pixel_sha256"] != self.context["query_pixel_sha256"]):
                    raise NeuronBridgeRecordError("Gradient bundle disagrees with local query evidence.")

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
        if self.gradient_execution is None:
            data.pop("gradient_execution")
        if self.shape_execution is None:
            data.pop("shape_execution")
        else:
            for name in ("gradient_producer_reference_revision",
                         "gradient_producer_implementation_revision", "gradient_format",
                         "scored_cohort_digest"):
                if data["shape_execution"][name] is None:
                    data["shape_execution"].pop(name)
        _json_value(data)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SearchSession:
        data = _mapping(value)
        if data.get("query_source") is not None:
            data["query_source"] = SourceIdentity.from_dict(data["query_source"])
        if data.get("csv_provenance") is not None:
            data["csv_provenance"] = CSVProvenance(**data["csv_provenance"])
        if data.get("gradient_execution") is not None:
            data["gradient_execution"] = GradientExecution(**data["gradient_execution"])
        if data.get("shape_execution") is not None:
            data["shape_execution"] = ShapeExecution(**data["shape_execution"])
        data["parameters"] = tuple(SourceParameter(**item) for item in data.get("parameters", ()))
        data["diagnostics"] = tuple(Diagnostic(**item) for item in data.get("diagnostics", ()))
        session = cls(**data)
        from .record_migration import normalize_historical_session
        return normalize_historical_session(session)


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


def _sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise NeuronBridgeRecordError(f"{name} must be lowercase SHA-256 hex.")


@dataclass(frozen=True)
class GradientObservation:
    """One candidate's raw and normalized supplementary gradient measurement."""

    gradient_area_gap: int
    high_expression_area: int
    combined_penalty: int
    gradient_mirrored: bool
    normalized_score: float
    gradient_rank: int

    def __post_init__(self) -> None:
        for name in ("gradient_area_gap", "high_expression_area", "combined_penalty"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < (1 << 63):
                raise NeuronBridgeRecordError(f"{name} must be a nonnegative signed 64-bit integer.")
        if self.combined_penalty != self.gradient_area_gap + self.high_expression_area // 3:
            raise NeuronBridgeRecordError("Combined penalty disagrees with its raw components.")
        if type(self.gradient_mirrored) is not bool:
            raise NeuronBridgeRecordError("gradient_mirrored must be bool.")
        if type(self.normalized_score) is not float or not math.isfinite(self.normalized_score):
            raise NeuronBridgeRecordError("normalized_score must be a finite binary32 float.")
        try:
            stored = struct.unpack("!f", struct.pack("!f", self.normalized_score))[0]
        except (OverflowError, struct.error) as exc:
            raise NeuronBridgeRecordError("normalized_score must fit binary32.") from exc
        if stored != self.normalized_score:
            raise NeuronBridgeRecordError("normalized_score must be the authoritative binary32 value.")
        _integer(self.gradient_rank, "gradient_rank", 1)


@dataclass(frozen=True)
class GradientExecution:
    """Compact immutable cohort context for an accepted gradient reranking."""

    algorithm: str
    reference_revision: str
    implementation_revision: str
    parameters: dict[str, Any]
    profile_id: str
    label_policy: str
    normalization_policy: str
    normalization_precision: str
    cohort_count: int
    membership_digest: str
    max_positive_pixels: int | None
    max_combined_penalty: int | None
    bundle_reference: dict[str, Any]

    def __post_init__(self) -> None:
        for name in ("algorithm", "reference_revision", "implementation_revision", "profile_id",
                     "label_policy", "normalization_policy", "normalization_precision"):
            _string(getattr(self, name), name)
        for name in ("parameters", "bundle_reference"):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise NeuronBridgeRecordError(f"{name} must be a mapping.")
            _json_value(value)
            object.__setattr__(self, name, json.loads(json.dumps(value, allow_nan=False)))
        _integer(self.cohort_count, "cohort_count")
        _sha256(self.membership_digest, "membership_digest")
        maxima = (self.max_positive_pixels, self.max_combined_penalty)
        if self.cohort_count == 0:
            if maxima != (None, None):
                raise NeuronBridgeRecordError("An empty gradient cohort has no maxima.")
        else:
            for name, value in zip(("max_positive_pixels", "max_combined_penalty"), maxima, strict=True):
                if type(value) is not int or not 0 <= value < (1 << 63):
                    raise NeuronBridgeRecordError(f"{name} must be a nonnegative signed 64-bit integer.")
        try:
            from .gradient_scorer import GradientParameters
            from .search_profiles import search_profile
            if set(self.parameters) != {item.name for item in fields(GradientParameters)}:
                raise ValueError("Gradient parameters must retain their exact executed field set.")
            parameters = GradientParameters(**self.parameters)
            profile = search_profile(self.profile_id)
        except (TypeError, ValueError) as exc:
            raise NeuronBridgeRecordError("Gradient execution parameters or profile are invalid.") from exc
        if self.normalization_policy != "madi3d-lm-retained-cohort/1":
            raise NeuronBridgeRecordError("Unsupported gradient normalization policy.")
        if self.normalization_precision != "binary64-to-binary32":
            raise NeuronBridgeRecordError("Unsupported gradient normalization precision.")
        label_policies = {
            "reference-labels": (True, True),
            "no-labels": (False, False),
            "name-label-only": (True, False),
            "color-scale-only": (False, True),
        }
        if self.label_policy not in label_policies:
            raise NeuronBridgeRecordError("Unknown gradient label policy.")
        if (parameters.clear_name_label, parameters.clear_color_scale) != label_policies[self.label_policy]:
            raise NeuronBridgeRecordError("Gradient parameters disagree with their label policy.")
        if profile.profile_id != self.profile_id:
            raise NeuronBridgeRecordError("Unknown gradient search profile.")
        required = ("bundle_id", "query_id", "query_png_sha256", "query_pixel_sha256", "profile_id",
                    "gradient_png_sha256", "gradient_pixel_sha256", "zgap_png_sha256", "zgap_pixel_sha256")
        # No derivatives are consumed when the retained positive cohort is empty.
        # Keep exact query evidence without manufacturing an unused bundle.
        if self.cohort_count == 0 and "bundle_id" not in self.bundle_reference:
            query_keys = {"query_id", "query_png_sha256", "query_pixel_sha256", "profile_id"}
            if set(self.bundle_reference) != query_keys or self.bundle_reference["profile_id"] != self.profile_id:
                raise NeuronBridgeRecordError("Empty execution requires exact query-only evidence.")
            _string(self.bundle_reference["query_id"], "query_id")
            for name in ("query_png_sha256", "query_pixel_sha256"):
                _sha256(self.bundle_reference[name], name)
            return
        for name in required:
            if name not in self.bundle_reference:
                raise NeuronBridgeRecordError(f"bundle_reference is missing {name}.")
        for name in ("bundle_id", "query_id", "profile_id"):
            _string(self.bundle_reference[name], f"bundle_reference.{name}")
        if self.bundle_reference["profile_id"] != self.profile_id:
            raise NeuronBridgeRecordError("Bundle profile disagrees with gradient execution.")
        for name in required[2:4] + required[5:]:
            _sha256(self.bundle_reference[name], f"bundle_reference.{name}")
        for name in self.bundle_reference:
            if name.endswith("_sha256"):
                _sha256(self.bundle_reference[name], f"bundle_reference.{name}")
        allowed_reference_keys = {
            "bundle_id", "query_id", "origin", "gradient_origin", "zgap_origin",
            "query_png_sha256", "query_pixel_sha256", "profile_id",
            "gradient_png_sha256", "gradient_pixel_sha256", "zgap_png_sha256", "zgap_pixel_sha256",
            "association", "producer", "preparation", "roi_png_sha256", "roi_pixel_sha256",
        }
        if not set(self.bundle_reference) <= allowed_reference_keys:
            raise NeuronBridgeRecordError("bundle_reference contains unsupported or unbounded fields.")
        encoded_reference = json.dumps(self.bundle_reference, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(encoded_reference) > 32 * 1024:
            raise NeuronBridgeRecordError("bundle_reference exceeds its compact evidence bound.")


def gradient_membership_digest(occurrences) -> str:
    ordinals = [item.candidate_ordinal for item in occurrences]
    member_ids = [item.member_id for item in occurrences]
    for item in occurrences:
        _validate_gradient_identity(item)
    if len(set(ordinals)) != len(ordinals) or len(set(member_ids)) != len(member_ids):
        raise NeuronBridgeRecordError("Gradient membership requires unique candidate ordinals and member IDs.")
    members = [{
        "candidate_ordinal": item.candidate_ordinal,
        "member_id": item.member_id,
        "target": {
            "kind": item.target.kind,
            "provider": item.target.provider,
            "library": item.target.library,
            "library_release": item.target.library_release,
            "published_name": item.target.published_name,
            "biological": asdict(item.target.biological),
            "image": asdict(item.target.image),
        },
        "input_hashes": item.input_hashes,
    } for item in occurrences]
    encoded = json.dumps(members, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_gradient_identity(occurrence) -> None:
    if occurrence.candidate_ordinal is None or occurrence.member_id is None:
        raise NeuronBridgeRecordError("Gradient membership requires local member IDs and candidate ordinals.")
    _sha256(occurrence.input_hashes.get("decoded_rgb"), "decoded_rgb input hash")


def _validate_gradient_measurement(occurrence, execution) -> None:
    _validate_gradient_identity(occurrence)
    observation = occurrence.gradient
    if observation is None:
        raise NeuronBridgeRecordError("Gradient occurrence is missing its measurement.")
    positive = occurrence.matched_pixels
    if positive is None:
        raise NeuronBridgeRecordError("Gradient normalization requires the positive pixel count.")
    if occurrence.rank is None:
        raise NeuronBridgeRecordError("Gradient evidence must preserve the positive rank.")
    penalty = observation.combined_penalty
    if execution.max_positive_pixels is None or execution.max_combined_penalty is None:
        raise NeuronBridgeRecordError("Nonempty gradient cohort requires independent maxima.")
    if positive > execution.max_positive_pixels or penalty > execution.max_combined_penalty:
        raise NeuronBridgeRecordError("Gradient measurement exceeds its cohort maximum.")
    try:
        from .gradient_scorer import normalize_score
        expected = float(normalize_score(
            positive, penalty, execution.max_positive_pixels, execution.max_combined_penalty,
        ).stored_float32)
    except (ValueError, OverflowError) as exc:
        raise NeuronBridgeRecordError("Gradient measurements exceed the normalization contract.") from exc
    if observation.normalized_score != expected:
        raise NeuronBridgeRecordError("Stored gradient score disagrees with the pinned normalization.")


_GRADIENT_SOURCES = frozenset({"supplied-archive", "generated-producer"})


@dataclass(frozen=True)
class ShapeObservation:
    """One emitted candidate's separated positive, shape and combined evidence.

    The three layers stay distinct on purpose: the positive Color-Depth
    evidence (rank/score), the raw bidirectional shape measurements with the
    shape-selected mirror, and the final Janelia combined score/rank.
    ``positive_score`` is the reference label value (matched percent rounded
    HALF_UP to two decimals); ``shape_gap`` is the reference's selected
    ``realval`` entering the area-gap normalization.
    """

    positive_rank: int
    positive_score: float
    mask_to_sample: int
    sample_to_mask: int
    normal_value: int
    flip_value: int | None
    shape_mirrored: bool
    shape_gap: int
    combined_score: float
    combined_gap_text: str
    combined_rank: int
    fl_renamed: bool
    gradient_source: str

    def __post_init__(self) -> None:
        _integer(self.positive_rank, "positive_rank", 1)
        _integer(self.combined_rank, "combined_rank", 1)
        if type(self.positive_score) is not float or not math.isfinite(self.positive_score) \
                or not 0.0 <= self.positive_score <= 100.0:
            raise NeuronBridgeRecordError("positive_score must be a finite percent value.")
        for name in ("mask_to_sample", "sample_to_mask", "normal_value", "shape_gap"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < (1 << 63):
                raise NeuronBridgeRecordError(f"{name} must be a nonnegative signed 64-bit integer.")
        if self.normal_value != (self.sample_to_mask + self.mask_to_sample) // 2:
            raise NeuronBridgeRecordError("normal_value disagrees with its raw components.")
        if type(self.shape_mirrored) is not bool:
            raise NeuronBridgeRecordError("shape_mirrored must be bool.")
        if self.shape_mirrored:
            if self.flip_value is None:
                raise NeuronBridgeRecordError("A mirrored shape selection requires the flip measurement.")
            if type(self.flip_value) is not int or not 0 <= self.flip_value < (1 << 63):
                raise NeuronBridgeRecordError("flip_value must be a nonnegative signed 64-bit integer.")
            if not self.flip_value <= self.normal_value:
                raise NeuronBridgeRecordError("The shape flip wins ties; the stored selection disagrees.")
            if self.shape_gap != self.flip_value:
                raise NeuronBridgeRecordError("shape_gap must be the shape-selected value.")
        else:
            if self.shape_gap != self.normal_value:
                raise NeuronBridgeRecordError("An unmirrored shape selection must carry its ordinary value.")
            if self.flip_value is not None:
                # The mirroring trial ran and the ordinary trial won (L3124:
                # the flip wins only ties); a retained losing value must
                # therefore be strictly worse than the ordinary measurement.
                if type(self.flip_value) is not int or not self.flip_value > self.normal_value:
                    raise NeuronBridgeRecordError("A losing flip measurement must exceed the ordinary value.")
        if type(self.combined_score) is not float or not math.isfinite(self.combined_score):
            raise NeuronBridgeRecordError("combined_score must be a finite binary64 gap value.")
        _string(self.combined_gap_text, "combined_gap_text")
        if type(self.fl_renamed) is not bool:
            raise NeuronBridgeRecordError("fl_renamed must be bool.")
        if self.gradient_source not in _GRADIENT_SOURCES:
            raise NeuronBridgeRecordError("Unknown candidate gradient source.")


@dataclass(frozen=True)
class ShapeExecution:
    """Compact immutable cohort context for a two-stage Janelia shape search."""

    algorithm: str
    reference_revision: str
    implementation_revision: str
    parameters: dict[str, Any]
    profile_id: str
    shape_stage_executed: bool
    positive_count: int
    stage1_cap: int
    scoring_budget: int
    shortlist_count: int
    fl_positive: int
    max_positive_score: float | None
    max_shape_gap: int | None
    emitted_count: int
    membership_digest: str
    gradient_producer_reference_revision: str | None = None
    gradient_producer_implementation_revision: str | None = None
    gradient_format: str | None = None
    scored_cohort_digest: str | None = None

    def __post_init__(self) -> None:
        for name in ("algorithm", "reference_revision", "implementation_revision", "profile_id"):
            _string(getattr(self, name), name)
        if type(self.shape_stage_executed) is not bool:
            raise NeuronBridgeRecordError("shape_stage_executed must be bool.")
        if not isinstance(self.parameters, dict):
            raise NeuronBridgeRecordError("parameters must be a mapping.")
        _json_value(self.parameters)
        object.__setattr__(self, "parameters",
                           json.loads(json.dumps(self.parameters, allow_nan=False)))
        if set(self.parameters) != {"mirror", "mask_threshold", "negative_radius"}:
            raise NeuronBridgeRecordError("Shape parameters must retain their exact executed field set.")
        if type(self.parameters["mirror"]) is not bool:
            raise NeuronBridgeRecordError("Shape mirror preference must be boolean.")
        if type(self.parameters["mask_threshold"]) is not int or not 0 <= self.parameters["mask_threshold"] <= 255:
            raise NeuronBridgeRecordError("Shape mask threshold must be an integer from 0 to 255.")
        if type(self.parameters["negative_radius"]) is not int or not 0 <= self.parameters["negative_radius"] <= 60:
            raise NeuronBridgeRecordError("Shape negative radius must be an integer from 0 to 60.")
        try:
            from .search_profiles import search_profile
            profile = search_profile(self.profile_id)
        except (TypeError, ValueError) as exc:
            raise NeuronBridgeRecordError("Shape execution profile is invalid.") from exc
        if profile.profile_id != self.profile_id:
            raise NeuronBridgeRecordError("Unknown shape search profile.")
        for name in ("positive_count", "stage1_cap", "scoring_budget", "shortlist_count",
                     "fl_positive", "emitted_count"):
            _integer(getattr(self, name), name)
        _sha256(self.membership_digest, "membership_digest")
        if self.implementation_revision == "madi3d-janelia-shape/2":
            for name in ("gradient_producer_reference_revision",
                         "gradient_producer_implementation_revision", "gradient_format"):
                _string(getattr(self, name), name)
            _sha256(self.scored_cohort_digest, "scored_cohort_digest")
        if not self.shape_stage_executed:
            if (self.shortlist_count, self.emitted_count, self.max_positive_score, self.max_shape_gap) \
                    != (0, 0, None, None):
                raise NeuronBridgeRecordError("An unexecuted shape stage cannot carry a scored cohort.")
            if self.scoring_budget or self.stage1_cap:
                raise NeuronBridgeRecordError("An unexecuted shape stage cannot carry budgets.")
        else:
            if self.shortlist_count == 0 or self.emitted_count > self.scoring_budget:
                raise NeuronBridgeRecordError("Executed shape cohort is inconsistent.")
            if self.scoring_budget > self.shortlist_count:
                raise NeuronBridgeRecordError("Shape scoring budget exceeds its shortlist.")
            if type(self.max_positive_score) is not float or not math.isfinite(self.max_positive_score) \
                    or not 0.0 <= self.max_positive_score <= 100.0:
                raise NeuronBridgeRecordError("max_positive_score must be a finite percent value.")
            if type(self.max_shape_gap) is not int or not 0 <= self.max_shape_gap < (1 << 63):
                raise NeuronBridgeRecordError("max_shape_gap must be a nonnegative signed 64-bit integer.")
        if self.fl_positive > self.positive_count:
            raise NeuronBridgeRecordError("FL-positive count exceeds the positive cohort.")


def _validate_shape_measurement(occurrence, execution) -> None:
    _validate_gradient_identity(occurrence)
    if execution.implementation_revision == "madi3d-janelia-shape/2":
        _sha256(occurrence.input_hashes.get("shape_gradient"), "shape_gradient")
    observation = occurrence.shape
    if observation is None:
        raise NeuronBridgeRecordError("Shape occurrence is missing its measurement.")
    if occurrence.rank is None or occurrence.rank != observation.combined_rank:
        raise NeuronBridgeRecordError("Shape rank evidence must match the combined rank.")
    if observation.positive_rank > execution.positive_count:
        raise NeuronBridgeRecordError("Shape positive rank exceeds the positive cohort.")
    if execution.max_positive_score is None or execution.max_shape_gap is None:
        raise NeuronBridgeRecordError("Executed shape cohort requires independent maxima.")
    if observation.positive_score > execution.max_positive_score \
            or observation.shape_gap > execution.max_shape_gap:
        raise NeuronBridgeRecordError("Shape measurement exceeds its cohort maximum.")
    try:
        from .gradient_scorer import _format_gap, combined_gap
        text, value = combined_gap(observation.positive_score, observation.shape_gap,
                                   execution.max_positive_score, execution.max_shape_gap)
    except (TypeError, ValueError) as exc:
        raise NeuronBridgeRecordError("Shape measurements exceed the normalization contract.") from exc
    if observation.fl_renamed and execution.implementation_revision == "madi3d-janelia-shape/1":
        # The reference FL merge hands the emitted twin the strictly larger
        # "_FL" duplicate's gap string (L3287-3306). The own recomputation
        # therefore bounds the stored score from below, and the stored text
        # must remain the reference string form of its own stored value.
        if not math.isfinite(value) or observation.combined_score <= value:
            raise NeuronBridgeRecordError("FL-renamed combined score disagrees with its own measurement.")
        if _format_gap(observation.combined_score) != observation.combined_gap_text:
            raise NeuronBridgeRecordError("FL-renamed combined gap text is not the reference string form.")
    elif text != observation.combined_gap_text or value != observation.combined_score:
        raise NeuronBridgeRecordError("Stored shape score disagrees with the pinned normalization.")


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
    gradient: GradientObservation | None = None
    shape: ShapeObservation | None = None

    def __post_init__(self) -> None:
        _sequence(self, "metrics", HitMetric)
        if self.gradient is not None and not isinstance(self.gradient, GradientObservation):
            raise NeuronBridgeRecordError("gradient must be GradientObservation or None.")
        if self.shape is not None and not isinstance(self.shape, ShapeObservation):
            raise NeuronBridgeRecordError("shape must be ShapeObservation or None.")
        if self.gradient is not None and self.shape is not None:
            raise NeuronBridgeRecordError("An occurrence carries at most one reranking generation.")
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
        if self.gradient is None:
            del data["gradient"]
        if self.shape is None:
            del data["shape"]
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
        if data.get("gradient") is not None:
            data["gradient"] = GradientObservation(**data["gradient"])
        if data.get("shape") is not None:
            data["shape"] = ShapeObservation(**data["shape"])
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
        execution = self.session.gradient_execution
        if execution is not None:
            if execution.bundle_reference["query_id"] != self.session.query_reference:
                raise NeuronBridgeRecordError("Gradient bundle query disagrees with its session.")
            if self.subset is not None:
                if (self.subset["reported_retained_count"] != execution.cohort_count or
                        execution.cohort_count < len(self.occurrences)):
                    raise NeuronBridgeRecordError("Selected gradient subset counts disagree with its original cohort.")
                selected_ranks = [item.gradient.gradient_rank for item in self.occurrences if item.gradient is not None]
                if len(set(selected_ranks)) != len(selected_ranks):
                    raise NeuronBridgeRecordError("Selected gradient ranks must be unique.")
                for occurrence in self.occurrences:
                    _validate_gradient_measurement(occurrence, execution)
                    if occurrence.gradient.gradient_rank > execution.cohort_count:
                        raise NeuronBridgeRecordError("Selected gradient rank exceeds its original cohort.")
            else:
                if execution.cohort_count != len(self.occurrences):
                    raise NeuronBridgeRecordError("Gradient cohort count disagrees with complete occurrences.")
                if any(item.gradient is None for item in self.occurrences):
                    raise NeuronBridgeRecordError("Every complete gradient cohort occurrence needs a measurement.")
                if execution.cohort_count:
                    for occurrence in self.occurrences:
                        _validate_gradient_measurement(occurrence, execution)
                    actual_max_positive = max(item.matched_pixels for item in self.occurrences)
                    actual_max_penalty = max(item.gradient.combined_penalty for item in self.occurrences)
                    if (actual_max_positive != execution.max_positive_pixels or
                            actual_max_penalty != execution.max_combined_penalty):
                        raise NeuronBridgeRecordError("Gradient cohort maxima disagree with its measurements.")
                    if gradient_membership_digest(self.occurrences) != execution.membership_digest:
                        raise NeuronBridgeRecordError("Gradient cohort digest disagrees with its occurrences.")
                    if any(item.rank is None or item.candidate_ordinal is None for item in self.occurrences):
                        raise NeuronBridgeRecordError("Gradient cohort requires positive ranks and candidate ordinals.")
                    ranked = sorted(self.occurrences, key=lambda item: (
                        -item.gradient.normalized_score, item.rank, item.candidate_ordinal,
                    ))
                    if sorted(item.gradient.gradient_rank for item in ranked) != list(range(1, execution.cohort_count + 1)) or any(
                            item.gradient.gradient_rank != position
                            for position, item in enumerate(ranked, 1)):
                        raise NeuronBridgeRecordError("Gradient ranks disagree with stored float32 score ordering.")
                else:
                    if self.occurrences:
                        raise NeuronBridgeRecordError("An empty gradient cohort cannot contain occurrences.")
                    if gradient_membership_digest(()) != execution.membership_digest:
                        raise NeuronBridgeRecordError("Empty gradient cohort digest is invalid.")
        elif any(item.gradient is not None for item in self.occurrences):
            raise NeuronBridgeRecordError("Gradient measurements require a gradient execution context.")
        shape_execution = self.session.shape_execution
        if shape_execution is not None:
            if any(item.gradient is not None for item in self.occurrences):
                raise NeuronBridgeRecordError("Shape and retired gradient measurements cannot mix.")
            if not shape_execution.shape_stage_executed:
                if any(item.shape is not None for item in self.occurrences):
                    raise NeuronBridgeRecordError("An unexecuted shape stage cannot carry measurements.")
            else:
                if self.subset is None and len(self.occurrences) != shape_execution.emitted_count:
                    raise NeuronBridgeRecordError("Shape cohort size disagrees with its emitted rows.")
                if any(item.shape is None for item in self.occurrences):
                    raise NeuronBridgeRecordError("Every shape cohort occurrence needs a measurement.")
                ranks = [item.shape.combined_rank for item in self.occurrences]
                if len(set(ranks)) != len(ranks):
                    raise NeuronBridgeRecordError("Shape combined ranks must be unique.")
                if any(rank > shape_execution.emitted_count for rank in ranks):
                    raise NeuronBridgeRecordError("Shape combined rank exceeds its published cohort.")
                for occurrence in self.occurrences:
                    _validate_shape_measurement(occurrence, shape_execution)
                if self.subset is None:
                    if sorted(ranks) != list(range(1, len(self.occurrences) + 1)) or any(
                            occurrence.shape.combined_rank != position
                            for position, occurrence in enumerate(self.occurrences, 1)):
                        raise NeuronBridgeRecordError("Shape ranks disagree with publication order.")
                    if gradient_membership_digest(self.occurrences) != shape_execution.membership_digest:
                        raise NeuronBridgeRecordError("Shape cohort digest disagrees with its occurrences.")
        elif any(item.shape is not None for item in self.occurrences):
            raise NeuronBridgeRecordError("Shape measurements require a shape execution context.")
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
