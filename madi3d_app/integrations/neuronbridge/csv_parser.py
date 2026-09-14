"""Lossless NeuronBridge CSV ingestion; no Qt, NB client, or network imports."""
from __future__ import annotations

import codecs
import csv
import hashlib
import io
import math
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

from .records import (
    AssetIdentity, BiologicalIdentity, ChannelSelection, CSVProvenance,
    Diagnostic, ImageIdentity, MatchOccurrence, ResultField, SearchResults,
    SearchSession, SourceIdentity, SourceParameter,
)


# Only labels with an explicit meaning are projected into the shared schema.
# In the upstream exporter "Neuron ID" is image.publishedName, NOT image.id.
_HEADER_ROLES = {
    "number": "number", "rank": "rank", "score": "score",
    "matched pixels": "matched_pixels", "matched pixel": "matched_pixels",
    "mirror": "mirror", "mirrored": "mirror", "mirror status": "mirror",
    "neuron id": "neuron_id", "line name": "line_name",
    "line name / neuron id": "published_name",
    "target type": "target_kind", "library": "library",
    "library release": "library_release", "image id": "image_id",
    "neuronbridge image id": "image_id", "slide code": "slide_code",
    "alignment space": "alignment_space", "sex": "sex",
    "magnification": "magnification", "anatomical area": "anatomical_area",
    "mounting protocol": "mounting_protocol", "neuron type": "neuron_type",
    "neuron instance": "neuron_instance", "asset id": "asset_id",
    "asset type": "asset_type", "asset url": "asset_url", "channel": "channel",
}
_NUMERIC = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_BOM_ENCODINGS = (
    # UTF-32LE must precede UTF-16LE, which shares its initial bytes.
    (codecs.BOM_UTF32_LE, "utf-32", ("utf-32", "utf-32-le")),
    (codecs.BOM_UTF32_BE, "utf-32", ("utf-32", "utf-32-be")),
    (codecs.BOM_UTF16_LE, "utf-16", ("utf-16", "utf-16-le")),
    (codecs.BOM_UTF16_BE, "utf-16", ("utf-16", "utf-16-be")),
    (codecs.BOM_UTF8, "utf-8-sig", ("utf-8", "utf-8-sig")),
)


def _role(header: str | None) -> str:
    normalized = " ".join((header or "").split()).casefold()
    if normalized.endswith(" score"):
        return "score"
    return _HEADER_ROLES.get(normalized, "unknown")


def _typed_field(index: int, header: str | None, raw: str | None) -> tuple[ResultField, str | None]:
    role = _role(header)
    value = None if raw is None or not raw.strip() else raw.strip()
    problem = None
    if value is not None and role in ("number", "rank", "score", "matched_pixels"):
        try:
            if not _NUMERIC.fullmatch(value):
                raise ValueError("Not a decimal number")
            number = Decimal(value)
            if not number.is_finite():
                raise ValueError("Non-finite number")
            if role == "score":
                value = float(number)
                if not math.isfinite(value) or (value == 0 and number != 0):
                    raise ValueError("Outside finite float range")
            else:
                # Bound conversion size: enormous exponents are not useful counts
                # and must not allocate arbitrarily large Python integers.
                if number < 0 or number != number.to_integral_value() or number.adjusted() > 1000:
                    raise ValueError("Not a representable nonnegative integer")
                value = int(number)
        except (InvalidOperation, ValueError, OverflowError):
            value = None
            problem = "invalid_numeric"
    elif value is not None and role == "mirror":
        lowered = value.casefold()
        if lowered in ("true", "yes", "1"):
            value = True
        elif lowered in ("false", "no", "0"):
            value = False
        else:
            value = None
            problem = "invalid_mirror"
    elif role == "unknown" and value is not None:
        value = raw
    return ResultField(index, header, raw, value, role), problem


def _target_identity(
    fields: tuple[ResultField, ...], row_index: int,
) -> tuple[SourceIdentity, tuple[Diagnostic, ...]]:
    diagnostics = []
    conflicts = set()

    def supplied(role: str) -> str | None:
        candidates = [item for item in fields if item.role == role and item.value is not None]
        values = {item.value for item in candidates}
        if len(values) > 1:
            conflicts.add(role)
            diagnostics.append(Diagnostic(
                "ambiguous_identity_field",
                f"Conflicting {role} values; none was selected.",
                row_index=row_index,
                column_indexes=tuple(item.column_index for item in candidates),
            ))
            return None
        return next(iter(values), None)

    values = {role: supplied(role) for role in (
        "neuron_id", "line_name", "published_name", "target_kind", "library",
        "library_release", "image_id", "slide_code", "alignment_space", "sex",
        "magnification", "anatomical_area", "mounting_protocol", "neuron_type",
        "neuron_instance", "asset_id", "asset_type", "asset_url", "channel",
    )}
    neuron_id, line_name = values["neuron_id"], values["line_name"]
    published = values["published_name"]
    declared_kind = (values["target_kind"] or "").casefold()
    kind = "unknown"
    ambiguous = bool(conflicts & {"neuron_id", "line_name", "published_name", "target_kind"})
    if declared_kind and declared_kind not in ("em", "lm"):
        ambiguous = True
    if neuron_id and line_name:
        ambiguous = True
    elif neuron_id:
        kind = "em"
    elif line_name:
        kind = "lm"
    if declared_kind in ("em", "lm"):
        if kind != "unknown" and kind != declared_kind:
            ambiguous = True
        else:
            kind = declared_kind
    names = {name for name in (neuron_id, line_name) if name is not None}
    specific_name = next(iter(names)) if len(names) == 1 else None
    if published and specific_name and published != specific_name:
        ambiguous = True
    if published and not specific_name and not ambiguous:
        if kind == "em":
            neuron_id = published
        elif kind == "lm":
            line_name = published
        else:
            ambiguous = True
    if ambiguous:
        kind = "unknown"
        diagnostics.append(Diagnostic(
            "ambiguous_target_identity",
            "EM/LM target identity cannot be selected unambiguously from the supplied fields.",
            row_index=row_index,
        ))
    if not any((neuron_id, line_name, published, values["image_id"])):
        diagnostics.append(Diagnostic(
            "missing_target_identity", "No target neuron, line, published name, or image ID was supplied.",
            row_index=row_index,
        ))
    if not values["library"] and (neuron_id or line_name or published):
        diagnostics.append(Diagnostic(
            "missing_library", "Target library is missing; the supplied name is not globally unique.",
            row_index=row_index,
        ))
    if kind == "lm" and not values["image_id"] and any(
        not values[name] for name in ("slide_code", "channel", "alignment_space", "magnification", "sex")
    ):
        diagnostics.append(Diagnostic(
            "incomplete_image_identity",
            "LM image ID and some image selectors are missing; retain the partial identity.",
            row_index=row_index,
        ))
    asset = AssetIdentity(
        asset_id=values["asset_id"], asset_type=values["asset_type"], url=values["asset_url"],
    )
    target = SourceIdentity(
        kind=kind, library=values["library"], library_release=values["library_release"],
        published_name=published or specific_name,
        biological=BiologicalIdentity(
            neuron_id=neuron_id, line_name=line_name,
            neuron_type=values["neuron_type"], neuron_instance=values["neuron_instance"],
        ),
        image=ImageIdentity(**{name: values[name] for name in (
            "image_id", "slide_code", "alignment_space", "sex", "magnification",
            "anatomical_area", "mounting_protocol",
        )}),
        assets=(asset,) if any((asset.asset_id, asset.asset_type, asset.url)) else (),
        channel=ChannelSelection(values["channel"]) if values["channel"] is not None else None,
    )
    return target, tuple(diagnostics)


def _encoding(data: bytes, supplied_encoding: str | None) -> str:
    if supplied_encoding is not None:
        return supplied_encoding
    for bom, default, _compatible in _BOM_ENCODINGS:
        if data.startswith(bom):
            return default
    return "utf-8-sig"


def read_csv_header(path: str | Path) -> tuple[str, ...]:
    """Read only the first CSV record, honoring the parser's Unicode BOM rules."""
    with Path(path).open("rb") as handle:
        encoding = _encoding(handle.read(4), None)
        handle.seek(0)
        # Replacement is only for classification. The lossless parser retains
        # original bytes and reports decoding errors if this is an NB table.
        with io.TextIOWrapper(handle, encoding=encoding, errors="replace", newline="") as text:
            return tuple(next(csv.reader(text), ()))


def parse_neuronbridge_csv(
    data: bytes,
    *,
    filename: str,
    session_id: str | None = None,
    imported_at: str | None = None,
    encoding: str | None = None,
    neuronbridge_data_version: str | None = None,
    query_reference: str | None = None,
    query_source: SourceIdentity | None = None,
    parameters: tuple[SourceParameter, ...] = (),
    cancel_check=None,
) -> SearchResults:
    """Parse CSV bytes and retain the entire original payload and diagnostics.

    Defaults to strict UTF-8; Unicode BOMs are authoritative. No legacy-encoding
    heuristic, identity lookup, algorithm inference, ranking, or deduplication is
    performed. A caller may explicitly supply a known encoding. Metadata keyword
    arguments must come from source evidence, not assumed query defaults.

    Row and column indexes are zero-based; physical source lines are one-based.
    Width errors retain every parsed cell and parsing continues. Invalid quoting
    stops parsing: the remaining bytes are preserved without guessing boundaries.
    """
    if not isinstance(data, bytes):
        raise TypeError("CSV input must be bytes so its checksum covers the original file.")
    chosen_encoding = _encoding(data, encoding)
    session = SearchSession(
        session_id=session_id if session_id is not None else str(uuid4()),
        source_kind="imported_csv",
        neuronbridge_data_version=neuronbridge_data_version,
        query_reference=query_reference, query_source=query_source, parameters=parameters,
        csv_provenance=CSVProvenance(
            filename, hashlib.sha256(data).hexdigest(),
            imported_at if imported_at is not None else datetime.now(timezone.utc).isoformat(),
            chosen_encoding,
        ),
    )
    diagnostics = []
    occurrences = []
    headers = ()

    def result(complete: bool) -> SearchResults:
        return SearchResults(session, tuple(occurrences), tuple(diagnostics), headers, data, complete)

    try:
        canonical_encoding = codecs.lookup(chosen_encoding).name
        for bom, _default, compatible in _BOM_ENCODINGS:
            if data.startswith(bom):
                if canonical_encoding not in compatible:
                    raise ValueError("Supplied encoding conflicts with the Unicode BOM.")
                break
        text = data.decode(chosen_encoding, errors="strict")
        # Also handle an explicitly requested UTF-8/UTF-16LE codec with BOM.
        if text.startswith("\ufeff"):
            text = text[1:]
        if "\x00" in text:
            raise ValueError("NUL characters may indicate an unspecified encoding.")
    except (UnicodeError, LookupError, ValueError) as exc:
        diagnostics.append(Diagnostic(
            "encoding_error", f"CSV could not be decoded unambiguously as {chosen_encoding}: {exc}",
            severity="error",
        ))
        return result(False)

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        headers = tuple(next(reader))
    except StopIteration:
        diagnostics.append(Diagnostic("missing_header", "CSV is empty.", severity="error"))
        return result(False)
    except csv.Error as exc:
        diagnostics.append(Diagnostic(
            "malformed_header", f"CSV header could not be parsed: {exc}", severity="error",
            line_start=1, line_end=max(1, reader.line_num),
        ))
        return result(False)
    if not headers:
        diagnostics.append(Diagnostic("missing_header", "CSV header is blank.", severity="error"))
        return result(False)
    by_header: dict[str, list[int]] = {}
    for index, header in enumerate(headers):
        by_header.setdefault(header, []).append(index)
        if not header.strip():
            diagnostics.append(Diagnostic(
                "empty_header", "Column has an empty header; its cells are retained as unknown fields.",
                column_indexes=(index,),
            ))
    for header, indexes in by_header.items():
        if len(indexes) > 1:
            diagnostics.append(Diagnostic(
                "duplicate_header", f"Repeated header {header!r}; columns remain separate.",
                severity="info", column_indexes=tuple(indexes),
            ))

    while True:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("NeuronBridge import cancelled.")
        row_index = len(occurrences)
        line_start = reader.line_num + 1
        try:
            row = next(reader)
        except StopIteration:
            return result(True)
        except csv.Error as exc:
            diagnostics.append(Diagnostic(
                "malformed_csv", f"CSV parsing stopped; remaining source bytes are retained: {exc}",
                severity="error", row_index=row_index,
                line_start=line_start, line_end=max(line_start, reader.line_num),
            ))
            return result(False)
        row_diagnostics = []
        if len(row) != len(headers):
            row_diagnostics.append(Diagnostic(
                "row_width_mismatch", f"Expected {len(headers)} cells, received {len(row)}; all cells retained.",
                severity="error", row_index=row_index, line_start=line_start, line_end=reader.line_num,
            ))
        parsed_fields = []
        for index in range(max(len(headers), len(row))):
            item, problem = _typed_field(
                index, headers[index] if index < len(headers) else None,
                row[index] if index < len(row) else None,
            )
            parsed_fields.append(item)
            if problem:
                row_diagnostics.append(Diagnostic(
                    problem, f"Invalid {item.role} text was preserved without a typed value.",
                    row_index=row_index, column_indexes=(index,),
                    line_start=line_start, line_end=reader.line_num,
                ))
        ordered_fields = tuple(parsed_fields)
        target, identity_diagnostics = _target_identity(ordered_fields, row_index)
        occurrences.append(MatchOccurrence(
            occurrence_id=f"{session.session_id}:row:{row_index}", session_id=session.session_id,
            target=target, row_index=row_index, fields=ordered_fields,
            diagnostics=tuple(row_diagnostics) + identity_diagnostics,
            line_start=line_start, line_end=reader.line_num,
        ))


def read_neuronbridge_csv(path: str | Path, **metadata) -> SearchResults:
    """Read a local CSV once; OSError propagates and no network access is made."""
    path = Path(path)
    return parse_neuronbridge_csv(path.read_bytes(), filename=path.name, **metadata)
