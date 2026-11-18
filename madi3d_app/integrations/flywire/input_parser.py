"""Lossless local Codex CSV and root-ID parsing; no scene or network imports."""
from __future__ import annotations

import codecs
import csv
import hashlib
import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from madi3d_app.io_utils import check_cancelled

from .records import (
    FlyWireInputs, InputDiagnostic, InputObservation, InputRow, ReleaseContext,
    validate_root_id,
)


_CHARACTERISTIC_HEADERS = {"nt_type", "super_class", "sub_class", "hemilineage", "connectivity_tag"}
# Qualified annotation-product headings, not an observed MCNS Codex export.
_MCNS_PRODUCT_HEADERS = {"type", "instance", "superclass", "class", "subclass", "supertype",
                         "somaside", "rootside", "somaneuromere", "statuslabel"}


@dataclass(frozen=True)
class InputLimits:
    """Bound raw input and retained review objects, including invalid rows."""
    max_bytes: int = 32 * 1024**2
    max_rows: int = 100_000
    max_cells: int = 2_000_000

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in
               (self.max_bytes, self.max_rows, self.max_cells)):
            raise ValueError("Input limits must be positive integers.")


DEFAULT_LIMITS = InputLimits()


def _check_size(size, limit, kind):
    if size > limit:
        raise ValueError(f"FlyWire input exceeds the {limit:,} {kind} review limit. "
                         "Split the input into smaller batches. No partial batch was accepted.")


class _InputReader(io.RawIOBase):
    """Apply one byte budget to full reads and header-only routing probes."""
    def __init__(self, stream, limits, cancelled=None):
        super().__init__()
        self.stream, self.limits, self.cancelled = stream, limits, cancelled
        self.size = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        check_cancelled(self.cancelled)
        capacity = min(len(buffer), self.limits.max_bytes - self.size + 1)
        count = self.stream.readinto(memoryview(buffer)[:capacity])
        _check_size(self.size + count, self.limits.max_bytes, "byte")
        self.size += count
        return count


def _read_input(path, limits, cancelled):
    with Path(path).open("rb") as stream, io.BufferedReader(_InputReader(stream, limits, cancelled)) as bounded:
        data = bounded.read()
    check_cancelled(cancelled)
    return data


def _encoding(prefix: bytes) -> str:
    for bom, encoding in ((codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
                          (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16")):
        if prefix.startswith(bom):
            return encoding
    return "utf-8-sig"


def _header_key(value: str) -> str:
    return value.strip().casefold()


def _codex_headers(headers: list[str], *, allow_id_only: bool = False,
                   context: ReleaseContext | None = None) -> bool:
    keys = {_header_key(header) for header in headers}
    if {"id", "parentid", "type"} <= keys:
        return False
    if keys & {"neuron id", "line name", "line name / neuron id", "neuronbridge image id"}:
        return False
    if "image id" in keys and keys & {"library", "target type"}:
        return False
    if "root_id" in keys and (allow_id_only or bool(keys & _CHARACTERISTIC_HEADERS)):
        return True
    # Only explicit MCNS selection can qualify a bodyId annotation table. The
    # context-free Open/drop classifier must not claim unrelated body-ID CSVs.
    return (context is not None and context.dataset == "mcns" and "bodyid" in keys
            and (allow_id_only or bool(keys & _MCNS_PRODUCT_HEADERS)))


def is_codex_csv(path: str | Path, *, limits=DEFAULT_LIMITS) -> bool:
    """Header-only dispatch probe shared by future Open/drop/panel consumers.

    A recognized file stays FlyWire-owned even when its subsequent rows are bad.
    Recognition requires root_id and a characteristic classification column;
    filenames and numeric content supply no release or query evidence.
    """
    with Path(path).open("rb") as stream, io.BufferedReader(_InputReader(stream, limits)) as bounded:
        prefix = bounded.peek(4)[:4]
        # Decoding ahead into a damaged body must not unclaim a valid header.
        with io.TextIOWrapper(bounded, encoding=_encoding(prefix), errors="replace", newline="") as text:
            try:
                headers = next(csv.reader(text, strict=True), [])
            except (UnicodeError, csv.Error):
                return False
            _check_size(len(headers), limits.max_cells, "cell")
            return _codex_headers(headers)


def _observation(data: bytes, kind: str, source_name: str | None, headers=(), *, observation_id=None) -> InputObservation:
    return InputObservation(
        observation_id or "sha256:" + hashlib.sha256(data).hexdigest(), kind, source_name,
        datetime.now(timezone.utc).isoformat(), tuple(headers),
    )


def _row(index: int, cells: tuple[str, ...], token: str, seen: set[str], diagnostics: list) -> InputRow:
    try:
        root_id = validate_root_id(token.strip())
    except ValueError as exc:
        diagnostics.append(InputDiagnostic("invalid_root_id", str(exc), index, token, "root_id"))
        return InputRow(index, cells, None)
    duplicate = root_id in seen
    seen.add(root_id)
    if duplicate:
        diagnostics.append(InputDiagnostic("duplicate_root_id", "Repeated root ID; retrieve once.", index, token, "root_id"))
    return InputRow(index, cells, root_id, duplicate)


def parse_root_ids(text: str, *, context: ReleaseContext, limits=DEFAULT_LIMITS,
                   cancelled=None) -> FlyWireInputs:
    """Parse one clipboard batch; invalid tokens do not discard valid siblings."""
    check_cancelled(cancelled)
    _check_size(len(text), limits.max_bytes, "byte")
    data = text.encode("utf-8")
    _check_size(len(data), limits.max_bytes, "byte")
    return _parse_ids(text.lstrip("\ufeff"), data, "root_id_paste", None, context, limits, cancelled)


def _parse_ids(text: str, data: bytes, kind: str, name: str | None, context: ReleaseContext,
               limits, cancelled) -> FlyWireInputs:
    rows, diagnostics, seen = [], [], set()
    for index, match in enumerate(re.finditer(r"[^,;\s]+", text), 1):
        check_cancelled(cancelled)
        _check_size(index, limits.max_rows, "row")
        _check_size(index, limits.max_cells, "cell")
        token = match.group()
        rows.append(_row(index, (token,), token, seen, diagnostics))
    if not rows:
        diagnostics.append(InputDiagnostic("empty_input", "Paste or select a file containing decimal root IDs."))
    check_cancelled(cancelled)
    return FlyWireInputs(context, _observation(data, kind, name), tuple(rows), tuple(diagnostics))


def parse_root_id_file(path: str | Path, *, context: ReleaseContext, limits=DEFAULT_LIMITS,
                       cancelled=None) -> FlyWireInputs:
    data = _read_input(path, limits, cancelled)
    try:
        text = data.decode(_encoding(data))
    except UnicodeError:
        check_cancelled(cancelled)
        return _encoding_failure(data, "root_id_file", Path(path).name, context, limits, cancelled)
    return _parse_ids(text, data, "root_id_file", Path(path).name, context, limits, cancelled)


def _encoding_failure(data: bytes, kind: str, name: str | None, context: ReleaseContext,
                       limits, cancelled, *, allow_id_only=False) -> FlyWireInputs:
    headers = []
    if kind == "codex_csv":
        try:
            headers = next(csv.reader(io.StringIO(data.decode(_encoding(data), errors="replace")), strict=True), [])
        except csv.Error:
            pass
    _check_size(len(headers), limits.max_cells, "cell")
    check_cancelled(cancelled)
    return FlyWireInputs(context, _observation(data, kind, name, headers), (), (
        InputDiagnostic("invalid_encoding", "Save the input as UTF-8 or BOM-marked UTF-16/UTF-32 text."),
    ), recognized=kind == "codex_csv" and _codex_headers(headers, allow_id_only=allow_id_only, context=context))


def parse_codex_csv(path: str | Path, *, context: ReleaseContext, limits=DEFAULT_LIMITS,
                    cancelled=None, allow_id_only: bool = False) -> FlyWireInputs:
    """Parse an explicit CSV selection using the same header contract as dispatch."""
    return parse_codex_csv_bytes(_read_input(path, limits, cancelled), context=context,
                                source_name=Path(path).name, limits=limits, cancelled=cancelled,
                                allow_id_only=allow_id_only)


def parse_codex_csv_bytes(data: bytes, *, context: ReleaseContext, source_name: str | None = None,
                          limits=DEFAULT_LIMITS, cancelled=None,
                          allow_id_only: bool = False, delimiter: str = ",",
                          input_kind: str = "codex_csv",
                          observation_id: str | None = None) -> FlyWireInputs:
    """Parse a bounded Codex CSV or copied delimited table losslessly."""
    if type(allow_id_only) is not bool:
        raise ValueError("ID-only CSV acceptance must be an explicit boolean choice.")
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValueError("A delimited Codex table requires one delimiter character.")
    check_cancelled(cancelled)
    _check_size(len(data), limits.max_bytes, "byte")
    try:
        text = data.decode(_encoding(data))
    except UnicodeError:
        check_cancelled(cancelled)
        return _encoding_failure(data, input_kind, source_name, context, limits, cancelled,
                                 allow_id_only=allow_id_only)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    try:
        headers = next(reader, [])
    except csv.Error:
        observation = _observation(data, input_kind, source_name, observation_id=observation_id)
        return FlyWireInputs(context, observation, (), (
            InputDiagnostic("invalid_csv_header", "The CSV header is malformed; export it again from Codex."),
        ), recognized=False)
    _check_size(len(headers), limits.max_cells, "cell")
    observation = _observation(data, input_kind, source_name, headers, observation_id=observation_id)
    if not _codex_headers(headers, allow_id_only=allow_id_only, context=context):
        return FlyWireInputs(context, observation, (), (
            InputDiagnostic("unsupported_csv", "Expected root_id and classification columns, or an MCNS bodyId table in selected MCNS context."),
        ), recognized=False)
    keys = [_header_key(header) for header in headers]
    rows, diagnostics, seen = [], [], set()
    duplicate_headers = len(set(keys)) != len(keys)
    if duplicate_headers:
        diagnostics.append(InputDiagnostic("duplicate_headers", "Repeated CSV columns are retained separately; export distinct names to use named metadata."))
    identity_headers = {"root_id", "bodyid"} if context.dataset == "mcns" else {"root_id"}
    root_columns = [column for column, key in enumerate(keys) if key in identity_headers]
    index = 0
    cell_count = len(headers)
    try:
        for index, raw_cells in enumerate(reader, 1):
            check_cancelled(cancelled)
            _check_size(index, limits.max_rows, "row")
            cell_count += len(raw_cells)
            _check_size(cell_count, limits.max_cells, "cell")
            if not raw_cells:
                continue
            cells = tuple(raw_cells)
            if len(cells) != len(headers):
                rows.append(InputRow(index, cells, None))
                diagnostics.append(InputDiagnostic("invalid_row_structure", "Row does not map unambiguously to the CSV columns.", index))
                continue
            if len({cells[column].strip() for column in root_columns}) != 1:
                rows.append(InputRow(index, cells, None))
                diagnostics.append(InputDiagnostic("ambiguous_root_id", "Identity columns disagree; select the intended ID before import.", index))
                continue
            rows.append(_row(index, cells, cells[root_columns[0]], seen, diagnostics))
    except csv.Error:
        diagnostics.append(InputDiagnostic("malformed_csv", "CSV quoting is malformed; earlier valid rows remain reviewable.", index + 1))
    if not rows and not diagnostics:
        diagnostics.append(InputDiagnostic("empty_input", "This Codex CSV contains no result rows."))
    check_cancelled(cancelled)
    return FlyWireInputs(context, observation, tuple(rows), tuple(diagnostics))
