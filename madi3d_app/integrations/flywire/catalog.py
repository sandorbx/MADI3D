"""Bounded local annotation catalogs for FAFB, MANC and MCNS.

Source headers, cells and product evidence are retained alongside normalized
terms so search never replaces the source observations.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit

from .records import ReleaseContext, validate_root_id


_MAX_LOOKUP_IDS = 5000
_LOOKUP_CHUNK_SIZE = 400
_SEARCH_FIELDS = frozenset({
    "cell_type", "class", "super_class", "sub_class", "label", "side",
    "nt_type", "nt_type_verified", "hemilineage",
})
_MANC_SEARCH_FIELDS = frozenset({"cell_type", "tag"})
_MANC_PRODUCT = "segment_properties"
_MANC_REVISION = "segment_properties_v1.2.1"
MANC_PROPERTIES_SHA256 = "0d6e476e5f344ca17b1ff9f63048fa33b4228d1d702ac59943d7cd8956c4d4cb"
_MANC_HEADERS = ["body_id", "type", "PreSyn", "PostSyn", "tags"]
MCNS_ANNOTATION_PRODUCT = "body_annotations"
MCNS_TRANSMITTER_PRODUCT = "body_neurotransmitters"
_MCNS_REVISIONS = {
    MCNS_ANNOTATION_PRODUCT: "body-annotations-male-cns-v1.0-minconf-0.5",
    MCNS_TRANSMITTER_PRODUCT: "body-neurotransmitters-male-cns-v1.0",
}
MCNS_CITATIONS = (
    "https://male-cns.janelia.org/download/",
    "https://creativecommons.org/licenses/by/4.0/",
    "https://www.cell.com/cell/fulltext/S0092-8674%2826%2900942-6",
)
# These are projections of source fields, not fabricated FAFB classifications.
_MCNS_TERM_FIELDS = {
    "type": "cell_type", "instance": "instance", "superclass": "super_class",
    "class": "class", "subclass": "sub_class", "supertype": "supertype",
    "somaSide": "side", "rootSide": "root_side", "somaNeuromere": "soma_neuromere",
    "itoleeHl": "itolee_hl", "trumanHl": "truman_hl", "entryNerve": "entry_nerve",
    "exitNerve": "exit_nerve", "synonyms": "label", "status": "status",
    "statusLabel": "status_label",
    "predicted_nt": "predicted_nt", "ground_truth": "ground_truth",
    "celltype_predicted_nt": "celltype_predicted_nt", "consensus_nt": "consensus_nt",
}
_PRODUCTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "consolidated_cell_types": (frozenset({"root_id", "primary_type"}),
                                 frozenset({"additional_type(s)"})),
    "classification": (frozenset({"root_id"}), frozenset({
        "class", "super_class", "sub_class", "side", "hemilineage",
        "nt_type", "nt_type_verified",
    })),
    "neurons": (frozenset({"root_id", "nt_type"}), frozenset({"nt_type_verified"})),
    "labels": (frozenset({"root_id", "label"}), frozenset()),
}


class CatalogCancelledError(InterruptedError):
    """Raised when an install or query is cancelled by its caller."""


@dataclass(frozen=True)
class CatalogProduct:
    path: str | os.PathLike[str]
    product: str
    source_date: str | None = None
    acquired_at_utc: str | None = None
    source_locator: str | None = None
    citation_urls: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CatalogLimits:
    max_compressed_bytes: int = 64 * 1024 * 1024
    max_decompressed_bytes: int = 256 * 1024 * 1024
    max_database_bytes: int = 512 * 1024 * 1024
    max_rows: int = 2_000_000
    max_line_bytes: int = 1024 * 1024
    max_field_bytes: int = 64 * 1024
    reserve_disk_bytes: int = 256 * 1024 * 1024
    max_lookup_payload_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SearchResult:
    root_ids: tuple[str, ...]
    truncated: bool


class _HashingReader:
    """File-like compressed-byte reader that hashes and caps every read."""

    def __init__(self, path: Path, limit: int):
        self._file = path.open("rb")
        self.limit = limit
        self.bytes_read = 0
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = min(64 * 1024, self.limit - self.bytes_read + 1)
        data = self._file.read(size)
        self.bytes_read += len(data)
        if self.bytes_read > self.limit:
            raise ValueError("Compressed product exceeds the configured size limit")
        self.digest.update(data)
        return data

    def seekable(self) -> bool:
        return False

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        self._file.close()


def _cancelled(cancel: Callable[[], bool]) -> None:
    if cancel():
        raise CatalogCancelledError("Catalog operation cancelled")


class Catalog:
    """Atomically managed, read-only searchable annotation catalog."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        limits: CatalogLimits | None = None,
        *,
        context: ReleaseContext = ReleaseContext("fafb", "783"),
    ):
        self.path = Path(path)
        self.limits = limits or CatalogLimits()
        if not isinstance(context, ReleaseContext):
            raise TypeError("context must be a ReleaseContext")
        self._context = context

    @property
    def context(self) -> ReleaseContext:
        return self._context

    @classmethod
    def installed(cls, app_data: str | Path, *, context: ReleaseContext) -> Catalog:
        """Locate the installed catalog for an explicit release."""
        name = f"{context.dataset}{context.segmentation_version}.sqlite"
        return cls(Path(app_data) / "flywire" / name, context=context)

    @property
    def search_fields(self) -> frozenset[str]:
        if self.context.dataset == "mcns":
            return frozenset(_MCNS_TERM_FIELDS.values())
        return _MANC_SEARCH_FIELDS if self.context.dataset == "manc" else _SEARCH_FIELDS

    def install(
        self,
        products: Sequence[CatalogProduct],
        cancelled: Callable[[], bool] = lambda: False,
        progress: Callable[[str], None] = lambda _text: None,
    ) -> dict:
        """Build and atomically publish a replacement catalog from supported local products."""
        if not products:
            raise ValueError("Select at least one annotation product")
        if self.path.suffix.lower() not in {".sqlite", ".db"}:
            raise ValueError("Catalog path must use a .sqlite or .db extension")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        prepared: list[tuple[CatalogProduct, Path]] = []
        is_manc = self.context.dataset == "manc"
        is_mcns = self.context.dataset == "mcns"
        for item in products:
            allowed_products = set(_MCNS_REVISIONS) if is_mcns else {_MANC_PRODUCT} if is_manc else set(_PRODUCTS)
            if not isinstance(item, CatalogProduct) or item.product not in allowed_products:
                raise ValueError("Unsupported FlyWire annotation product")
            source = Path(item.path)
            if is_manc:
                if len(products) != 1 or (source.suffix.lower() != ".json" and source.name.lower() != "info"):
                    raise ValueError("MANC accepts one local segment_properties JSON product")
            elif is_mcns:
                if source.suffix.lower() != ".feather":
                    raise ValueError("MCNS accepts local Feather products")
            elif source.suffix.lower() != ".gz" or not source.name[:-3].lower().endswith(".csv"):
                raise ValueError("Catalog products must be local .csv.gz files")
            if item.source_date is not None and not isinstance(item.source_date, str):
                raise ValueError("source_date must be text or None")
            self._validate_provenance(item)
            prepared.append((item, source))

        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(temp_path)
            self._configure_writer(conn)
            conn.set_progress_handler(lambda: 1 if cancelled() else 0, 1000)
            self._create_schema(conn)
            conn.commit()
            total_decompressed = 0
            total_rows = 0
            product_evidence: list[dict] = []
            for product_index, (item, source) in enumerate(prepared):
                _cancelled(cancelled)
                progress(f"Reading {item.product} ({product_index + 1}/{len(prepared)})")
                if is_mcns:
                    evidence, row_count, decompressed = self._ingest_mcns_product(
                        conn, product_index, item, source, temp_path, total_decompressed, total_rows, cancelled
                    )
                elif is_manc:
                    evidence, row_count, decompressed = self._ingest_manc_product(
                        conn, product_index, item, source, temp_path, total_decompressed, total_rows, cancelled
                    )
                else:
                    evidence, row_count, decompressed = self._ingest_product(
                        conn, product_index, item, source, temp_path, total_decompressed, total_rows, cancelled
                    )
                total_decompressed += decompressed
                total_rows += row_count
                product_evidence.append(evidence)
                self._check_database_size(temp_path)

            _cancelled(cancelled)
            progress("Building catalog indexes")
            conn.execute("CREATE INDEX terms_field_value_root ON terms(field, value_fold, root_id)")
            conn.execute("CREATE INDEX terms_root_field ON terms(root_id, field)")
            conn.commit()
            self._check_database_size(temp_path)
            root_count = conn.execute("SELECT COUNT(DISTINCT root_id) FROM source_rows").fetchone()[0]
            available_fields = sorted({field for item in product_evidence for field in item["fields"]})
            # The database is self-contained and closed before its atomic replacement.
            conn.close()
            conn = None
            _cancelled(cancelled)
            os.replace(temp_path, self.path)
            result = {
                "dataset": self.context.dataset,
                "segmentation_version": self.context.segmentation_version,
                "products": product_evidence,
                "root_count": root_count,
                "fields": available_fields,
            }
            return result
        except sqlite3.OperationalError as exc:
            if conn is not None:
                conn.close()
            if cancelled():
                raise CatalogCancelledError("Catalog installation cancelled") from exc
            raise
        except BaseException:
            if conn is not None:
                conn.close()
            raise
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _configure_writer(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA cache_size=-8192")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("PRAGMA page_size=4096")
        conn.execute(f"PRAGMA max_page_count={max(1, self.limits.max_database_bytes // 4096)}")

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript("""
            CREATE TABLE catalog_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE products (
                product_index INTEGER PRIMARY KEY,
                product TEXT NOT NULL,
                source_date TEXT,
                acquired_at_utc TEXT,
                source_locator TEXT,
                citation_urls_json TEXT,
                sha256 TEXT NOT NULL,
                compressed_bytes INTEGER NOT NULL,
                row_count INTEGER NOT NULL,
                fields_json TEXT NOT NULL,
                headers_json TEXT NOT NULL,
                property_schema_json TEXT
            );
            CREATE TABLE source_rows (
                product_index INTEGER NOT NULL,
                row_index INTEGER NOT NULL,
                root_id TEXT NOT NULL,
                cells_json TEXT NOT NULL,
                PRIMARY KEY (product_index, row_index)
            );
            CREATE TABLE terms (
                root_id TEXT NOT NULL,
                field TEXT NOT NULL,
                value_fold TEXT NOT NULL
            );
            CREATE INDEX source_rows_root ON source_rows(root_id);
        """)
        metadata = [
            ("dataset", self.context.dataset),
            ("segmentation_version", self.context.segmentation_version),
        ]
        if self.context.dataset == "manc":
            metadata.append(("annotation_revision", _MANC_REVISION))
        elif self.context.dataset == "mcns":
            metadata.append(("annotation_revision", "male-cns-v1.0"))
            conn.execute("CREATE UNIQUE INDEX mcns_product_body ON source_rows(product_index, root_id)")
        conn.executemany("INSERT INTO catalog_meta VALUES (?, ?)", metadata)

    def _ingest_product(
        self,
        conn: sqlite3.Connection,
        product_index: int,
        item: CatalogProduct,
        source: Path,
        database_path: Path,
        decompressed_so_far: int,
        rows_so_far: int,
        cancelled: Callable[[], bool],
    ) -> tuple[dict, int, int]:
        reader = _HashingReader(source, self.limits.max_compressed_bytes)
        total_uncompressed = 0
        row_count = 0
        available_fields: set[str] = set()
        headers: list[str] | None = None
        staged_terms: list[tuple[str, str, str]] = []
        try:
            with gzip.GzipFile(fileobj=reader, mode="rb") as stream:
                def lines() -> Iterable[str]:
                    nonlocal total_uncompressed
                    while True:
                        _cancelled(cancelled)
                        raw = stream.readline(self.limits.max_line_bytes + 1)
                        if not raw:
                            return
                        if len(raw) > self.limits.max_line_bytes:
                            raise ValueError("CSV line exceeds the configured size limit")
                        total_uncompressed += len(raw)
                        if decompressed_so_far + total_uncompressed > self.limits.max_decompressed_bytes:
                            raise ValueError("Decompressed catalog exceeds the configured size limit")
                        try:
                            yield raw.decode("utf-8-sig" if total_uncompressed == len(raw) else "utf-8")
                        except UnicodeDecodeError as exc:
                            raise ValueError("Catalog CSV must be valid UTF-8") from exc

                reader_csv = csv.reader(lines(), strict=True)
                try:
                    headers = next(reader_csv)
                except StopIteration:
                    raise ValueError(f"{item.product} product is empty")
                if not headers or any(not h for h in headers):
                    raise ValueError("CSV headers must be non-empty")
                required, optional = _PRODUCTS[item.product]
                header_set = set(headers)
                missing = set(required - header_set)
                if item.product == "classification" and not (optional & header_set):
                    missing.add("one supported classification field")
                if missing:
                    raise ValueError(f"{item.product} is missing required CSV fields: {', '.join(sorted(missing))}")
                root_columns = [i for i, header in enumerate(headers) if header == "root_id"]
                if len(root_columns) != 1:
                    raise ValueError("CSV product must contain exactly one root_id column")
                available_fields = {
                    field for header in headers
                    if (field := self._term_field(item.product, header)) is not None
                }
                conn.execute("BEGIN")

                def flush_terms() -> None:
                    if staged_terms:
                        conn.executemany("INSERT INTO terms VALUES (?, ?, ?)", staged_terms)
                        staged_terms.clear()

                for row_index, cells in enumerate(reader_csv, start=1):
                    _cancelled(cancelled)
                    if len(cells) != len(headers):
                        raise ValueError("CSV row width does not match its header")
                    if any(len(cell.encode("utf-8")) > self.limits.max_field_bytes for cell in cells):
                        raise ValueError("CSV field exceeds the configured size limit")
                    root_id = validate_root_id(cells[root_columns[0]])
                    row_count += 1
                    if rows_so_far + row_count > self.limits.max_rows:
                        raise ValueError("Catalog exceeds the configured row limit")
                    conn.execute(
                        "INSERT INTO source_rows VALUES (?, ?, ?, ?)",
                        (product_index, row_index, root_id,
                         json.dumps(cells, ensure_ascii=False, separators=(",", ":"))),
                    )
                    for column, header in enumerate(headers):
                        field = self._term_field(item.product, header)
                        if field is None:
                            continue
                        value = cells[column].strip()
                        if not value:
                            continue
                        values = value.split(",") if field == "cell_type" and header == "additional_type(s)" else [value]
                        for term in values:
                            term = term.strip()
                            if term:
                                staged_terms.append((root_id, field, term.casefold()))
                                if len(staged_terms) >= 10000:
                                    flush_terms()
                    if row_index % 1000 == 0:
                        flush_terms()
                        conn.commit()
                        conn.execute("BEGIN")
                        self._check_database_size(database_path)
                flush_terms()
                conn.execute(
                    "INSERT INTO products (product_index, product, source_date, acquired_at_utc, source_locator, citation_urls_json, sha256, compressed_bytes, row_count, fields_json, headers_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (product_index, item.product, item.source_date, item.acquired_at_utc, item.source_locator,
                     json.dumps(item.citation_urls) if item.citation_urls is not None else None,
                     reader.digest.hexdigest(), reader.bytes_read,
                     row_count, json.dumps(sorted(available_fields)),
                     json.dumps(headers, ensure_ascii=False, separators=(",", ":"))),
                )
                conn.commit()
        except (gzip.BadGzipFile, EOFError, zlib.error) as exc:
            raise ValueError("Compressed catalog product is malformed or truncated") from exc
        finally:
            reader.close()
        return ({"product": item.product, "source_date": item.source_date,
                 "acquired_at_utc": item.acquired_at_utc, "source_locator": item.source_locator,
                 "citation_urls": list(item.citation_urls) if item.citation_urls is not None else None,
                 "sha256": reader.digest.hexdigest(), "compressed_bytes": reader.bytes_read,
                 "rows": row_count, "fields": sorted(available_fields)}, row_count, total_uncompressed)

    def _ingest_mcns_product(
        self, conn, product_index, item, source, database_path,
        decompressed_so_far, rows_so_far, cancelled,
    ) -> tuple[dict, int, int]:
        # Feather v2 is an Arrow IPC file. Read one record batch at a time;
        # Python row materialization is limited to 1000 rows and needs no pandas.
        import pyarrow as pa
        import pyarrow.ipc as ipc

        locator = self.context.profile.annotation_source
        if item.product == MCNS_TRANSMITTER_PRODUCT:
            locator = locator.rsplit("/", 1)[0] + "/" + _MCNS_REVISIONS[item.product] + ".feather"
        id_header = "bodyId" if item.product == MCNS_ANNOTATION_PRODUCT else "body"
        digest = hashlib.sha256()
        size = row_count = decompressed = 0
        try:
            with source.open("rb") as stream:
                original = os.fstat(stream.fileno())
                while block := stream.read(64 * 1024):
                    _cancelled(cancelled)
                    size += len(block)
                    if size > self.limits.max_compressed_bytes:
                        raise ValueError("Feather product exceeds the configured size limit")
                    digest.update(block)
                stream.seek(0)
                with ipc.open_file(stream, options=ipc.IpcReadOptions(use_threads=False)) as reader:
                    headers = reader.schema.names
                    if len(set(headers)) != len(headers) or any(not name for name in headers):
                        raise ValueError("Feather product must have unique non-empty headings")
                    if id_header not in headers or reader.schema.field(id_header).type != pa.int64():
                        raise ValueError(f"MCNS {id_header} must be an int64 column")
                    fields = sorted({self._term_field(item.product, name) for name in headers} - {None})
                    schema = [{"id": field.name, "type": str(field.type)} for field in reader.schema]
                    conn.execute("BEGIN")
                    for batch_index in range(reader.num_record_batches):
                        _cancelled(cancelled)
                        batch = reader.get_batch(batch_index)
                        decompressed += batch.nbytes
                        if decompressed_so_far + decompressed > self.limits.max_decompressed_bytes:
                            raise ValueError("Decompressed catalog exceeds the configured size limit")
                        for column, field in zip(batch.columns, schema):
                            if pa.types.is_dictionary(column.type):
                                vocabulary = field.setdefault("dictionary_values", [])
                                for value in column.dictionary.to_pylist():
                                    if value not in vocabulary:
                                        vocabulary.append(value)
                        for start in range(0, batch.num_rows, 1000):
                            _cancelled(cancelled)
                            terms = []
                            for row in batch.slice(start, 1000).to_pylist():
                                _cancelled(cancelled)
                                body = row[id_header]
                                if type(body) is not int:
                                    raise ValueError("MCNS body IDs cannot be null")
                                root_id = validate_root_id(str(body))
                                cells = [row[name] for name in headers]
                                encoded = [json.dumps(cell, ensure_ascii=False, allow_nan=False) for cell in cells]
                                if any(len(cell.encode("utf-8")) > self.limits.max_field_bytes for cell in encoded):
                                    raise ValueError("Feather field exceeds the configured size limit")
                                payload = "[" + ",".join(encoded) + "]"
                                if len(payload.encode("utf-8")) > self.limits.max_line_bytes:
                                    raise ValueError("Feather row exceeds the configured size limit")
                                row_count += 1
                                if rows_so_far + row_count > self.limits.max_rows:
                                    raise ValueError("Catalog exceeds the configured row limit")
                                try:
                                    conn.execute("INSERT INTO source_rows VALUES (?, ?, ?, ?)",
                                                 (product_index, row_count, root_id, payload))
                                except sqlite3.IntegrityError as exc:
                                    raise ValueError("MCNS product contains duplicate body IDs") from exc
                                for name, value in row.items():
                                    field = self._term_field(item.product, name)
                                    if field is not None and value is not None:
                                        if not isinstance(value, str):
                                            raise ValueError(f"MCNS {name} must contain text or null")
                                        if value.strip():
                                            terms.append((root_id, field, value.strip().casefold()))
                            conn.executemany("INSERT INTO terms VALUES (?, ?, ?)", terms)
                            conn.commit()
                            self._check_database_size(database_path)
                            conn.execute("BEGIN")
                    current = os.fstat(stream.fileno())
                    if (size != original.st_size or current.st_size != original.st_size or
                            current.st_mtime_ns != original.st_mtime_ns):
                        raise ValueError("MCNS annotation product changed while being read")
                    conn.execute(
                        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (product_index, item.product, item.source_date, item.acquired_at_utc,
                         item.source_locator,
                         json.dumps(item.citation_urls if item.citation_urls is not None else (locator, *MCNS_CITATIONS)),
                         digest.hexdigest(), size, row_count, json.dumps(fields), json.dumps(headers),
                         json.dumps(schema, ensure_ascii=False, allow_nan=False)),
                    )
                    conn.commit()
        except pa.ArrowException as exc:
            raise ValueError("MCNS Feather product is malformed or unsupported") from exc
        return self._product_evidence(conn)[product_index], row_count, decompressed

    def _ingest_manc_product(
        self,
        conn: sqlite3.Connection,
        product_index: int,
        item: CatalogProduct,
        source: Path,
        database_path: Path,
        decompressed_so_far: int,
        rows_so_far: int,
        cancelled: Callable[[], bool],
    ) -> tuple[dict, int, int]:
        if item.product != _MANC_PRODUCT:
            raise ValueError("Unsupported MANC annotation product")
        expected_locator = self.context.profile.annotation_source
        if item.source_locator is not None and item.source_locator != expected_locator:
            raise ValueError("MANC source locator does not match the selected annotation revision")
        size = source.stat().st_size
        if size > self.limits.max_compressed_bytes or size > self.limits.max_decompressed_bytes:
            raise ValueError("MANC annotation product exceeds the configured size limit")
        read_limit = min(
            self.limits.max_compressed_bytes,
            self.limits.max_decompressed_bytes - decompressed_so_far,
        )
        with source.open("rb") as stream:
            raw = stream.read(read_limit + 1)
        if len(raw) > read_limit:
            raise ValueError("MANC annotation product exceeds the configured size limit")
        if len(raw) != size:
            raise ValueError("MANC annotation product changed while being read")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != MANC_PROPERTIES_SHA256:
            raise ValueError("MANC annotation product checksum does not match segment_properties_v1.2.1")
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("MANC annotation product must be valid UTF-8 JSON") from exc
        if not isinstance(document, dict) or document.get("@type") != "neuroglancer_segment_properties":
            raise ValueError("MANC annotation product has an unsupported Neuroglancer schema")
        inline = document.get("inline")
        ids = inline.get("ids") if isinstance(inline, dict) else None
        properties = inline.get("properties") if isinstance(inline, dict) else None
        if not isinstance(ids, list) or not isinstance(properties, list):
            raise ValueError("MANC annotation product is missing inline IDs or properties")
        expected_schema = [("type", "label"), ("PreSyn", "number"),
                           ("PostSyn", "number"), ("tags", "tags")]
        if len(properties) != len(expected_schema):
            raise ValueError("MANC annotation property schema is unsupported")
        schema: list[dict] = []
        values_by_property: list[list] = []
        tags_vocabulary: list[str] | None = None
        for descriptor, expected in zip(properties, expected_schema):
            if not isinstance(descriptor, dict) or (descriptor.get("id"), descriptor.get("type")) != expected:
                raise ValueError("MANC annotation property schema is unsupported")
            if expected[0] == "tags":
                vocabulary = descriptor.get("tags")
                tag_rows = descriptor.get("values")
                if not isinstance(vocabulary, list) or not isinstance(tag_rows, list) or len(tag_rows) != len(ids):
                    raise ValueError("MANC tag vocabulary or row values do not match the ID list")
                if any(not isinstance(value, str) or not value for value in vocabulary):
                    raise ValueError("MANC tag vocabulary must contain non-empty text")
                if len(set(vocabulary)) != len(vocabulary):
                    raise ValueError("MANC tag vocabulary must contain unique values")
                if any(len(value.encode("utf-8")) > self.limits.max_field_bytes for value in vocabulary):
                    raise ValueError("MANC tag vocabulary exceeds the configured field limit")
                tags_vocabulary = vocabulary
                values_by_property.append(tag_rows)
                retained = {key: value for key, value in descriptor.items() if key != "values"}
            else:
                values = descriptor.get("values")
                if not isinstance(values, list) or len(values) != len(ids):
                    raise ValueError("MANC annotation property values do not match the ID list")
                if expected[0] in {"PreSyn", "PostSyn"} and descriptor.get("data_type") != "int32":
                    raise ValueError("MANC synapse properties must use int32 values")
                values_by_property.append(values)
                retained = {key: value for key, value in descriptor.items() if key != "values"}
            schema.append(retained)
        if tags_vocabulary is None:
            raise ValueError("MANC annotation product is missing its tag vocabulary")
        folded_tags = [tag.casefold() for tag in tags_vocabulary]
        if len(ids) > self.limits.max_rows or rows_so_far + len(ids) > self.limits.max_rows:
            raise ValueError("Catalog exceeds the configured row limit")
        _cancelled(cancelled)
        available_fields = sorted(_MANC_SEARCH_FIELDS)
        conn.execute("BEGIN")
        staged_terms: list[tuple[str, str, str]] = []
        for row_index, body_id in enumerate(ids, start=1):
            _cancelled(cancelled)
            root_id = validate_root_id(body_id)
            cell_type, presyn, postsyn, tag_indexes = (values[row_index - 1] for values in values_by_property)
            if not isinstance(cell_type, str):
                raise ValueError("MANC cell type values must be text")
            if any(len(value.encode("utf-8")) > self.limits.max_field_bytes for value in (body_id, cell_type)):
                raise ValueError("MANC annotation field exceeds the configured size limit")
            if any(type(count) is not int or count < 0 or count > 2_147_483_647 for count in (presyn, postsyn)):
                raise ValueError("MANC synapse counts must be non-negative int32 values")
            if not isinstance(tag_indexes, list) or any(
                type(index) is not int or index < 0 or index >= len(tags_vocabulary)
                for index in tag_indexes
            ) or any(left >= right for left, right in zip(tag_indexes, tag_indexes[1:])):
                raise ValueError("MANC row tags must be increasing indexes into the tag vocabulary")
            cells = [body_id, cell_type, presyn, postsyn, tag_indexes]
            if len(json.dumps(cells, ensure_ascii=False).encode("utf-8")) > self.limits.max_field_bytes:
                raise ValueError("MANC annotation row exceeds the configured field limit")
            conn.execute(
                "INSERT INTO source_rows VALUES (?, ?, ?, ?)",
                (product_index, row_index, root_id,
                 json.dumps(cells, ensure_ascii=False, separators=(",", ":"))),
            )
            if cell_type:
                staged_terms.append((root_id, "cell_type", cell_type.casefold()))
            for index in tag_indexes:
                term = folded_tags[index]
                if term:
                    staged_terms.append((root_id, "tag", term))
            if len(staged_terms) >= 10000:
                conn.executemany("INSERT INTO terms VALUES (?, ?, ?)", staged_terms)
                staged_terms.clear()
            if row_index % 1000 == 0:
                if staged_terms:
                    conn.executemany("INSERT INTO terms VALUES (?, ?, ?)", staged_terms)
                    staged_terms.clear()
                conn.commit()
                conn.execute("BEGIN")
                self._check_database_size(database_path)
        if staged_terms:
            conn.executemany("INSERT INTO terms VALUES (?, ?, ?)", staged_terms)
        conn.execute(
            "INSERT INTO products (product_index, product, source_date, acquired_at_utc, source_locator, citation_urls_json, sha256, compressed_bytes, row_count, fields_json, headers_json, property_schema_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (product_index, item.product, item.source_date, item.acquired_at_utc,
             item.source_locator or expected_locator,
             json.dumps(item.citation_urls) if item.citation_urls is not None else None,
             digest, len(raw), len(ids), json.dumps(available_fields),
             json.dumps(_MANC_HEADERS), json.dumps(schema, ensure_ascii=False, separators=(",", ":"))),
        )
        conn.commit()
        evidence = {
            "product": item.product,
            "source_date": item.source_date,
            "acquired_at_utc": item.acquired_at_utc,
            "source_locator": item.source_locator or expected_locator,
            "citation_urls": list(item.citation_urls) if item.citation_urls is not None else None,
            "sha256": digest,
            "source_bytes": len(raw),
            "rows": len(ids),
            "fields": available_fields,
            "dataset": self.context.dataset,
            "segmentation_version": self.context.segmentation_version,
            "annotation_revision": _MANC_REVISION,
            "property_schema": schema,
        }
        return evidence, len(ids), len(raw)

    @staticmethod
    def _validate_provenance(item: CatalogProduct) -> None:
        if item.acquired_at_utc is not None:
            if not isinstance(item.acquired_at_utc, str):
                raise ValueError("acquired_at_utc must be UTC text or None")
            try:
                acquired = datetime.fromisoformat(item.acquired_at_utc.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("acquired_at_utc must be an ISO timestamp with a UTC offset") from exc
            if acquired.tzinfo is None or acquired.utcoffset() != timezone.utc.utcoffset(acquired):
                raise ValueError("acquired_at_utc must have a UTC offset")
        if item.source_locator is not None:
            Catalog._validate_evidence_url(item.source_locator, "source_locator")
        if item.citation_urls is not None:
            if not isinstance(item.citation_urls, (tuple, list)):
                raise ValueError("citation_urls must be a sequence of HTTPS URLs or None")
            for url in item.citation_urls:
                Catalog._validate_evidence_url(url, "citation_urls")

    @staticmethod
    def _validate_evidence_url(value: str, field: str) -> None:
        if not isinstance(value, str):
            raise ValueError(f"{field} must contain HTTPS URLs")
        parsed = urlsplit(value)
        query_keys = [key.casefold() for key, _value in parse_qsl(parsed.query, keep_blank_values=True)]
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or
                "api_token" in query_keys):
            raise ValueError(f"{field} must be a credential-free HTTPS URL")

    def _product_evidence(self, conn: sqlite3.Connection) -> list[dict]:
        cursor = conn.execute("SELECT * FROM products ORDER BY product_index")
        columns = [column[0] for column in cursor.description]
        products = []
        for values in cursor:
            row = dict(zip(columns, values))
            evidence = {
                "product": row["product"], "source_date": row["source_date"],
                "acquired_at_utc": row.get("acquired_at_utc"),
                "source_locator": row.get("source_locator"),
                "citation_urls": json.loads(row["citation_urls_json"])
                if row.get("citation_urls_json") is not None else None,
                "sha256": row["sha256"], "compressed_bytes": row["compressed_bytes"],
                "rows": row["row_count"], "fields": json.loads(row["fields_json"]),
            }
            if "property_schema_json" in row.keys() and row["property_schema_json"] is not None:
                evidence.update({
                    "dataset": self.context.dataset,
                    "segmentation_version": self.context.segmentation_version,
                    "annotation_revision": (_MCNS_REVISIONS[row["product"]]
                                            if self.context.dataset == "mcns" else _MANC_REVISION),
                    "property_schema": json.loads(row["property_schema_json"]),
                })
                evidence["source_bytes"] = evidence.pop("compressed_bytes")
            products.append(evidence)
        return products

    @staticmethod
    def _term_field(product: str, header: str) -> str | None:
        if product in _MCNS_REVISIONS:
            return _MCNS_TERM_FIELDS.get(header)
        if product == "consolidated_cell_types":
            return "cell_type" if header in {"primary_type", "additional_type(s)"} else None
        if product == "classification":
            return header if header in _SEARCH_FIELDS - {"cell_type", "label"} else None
        if product == "neurons":
            return header if header in {"nt_type", "nt_type_verified"} else None
        if product == "labels":
            return "label" if header == "label" else None
        return None

    def _check_database_size(self, path: Path) -> None:
        total = 0
        for candidate in (path, Path(str(path) + "-journal"), Path(str(path) + "-wal")):
            try:
                total += candidate.stat().st_size
            except FileNotFoundError:
                pass
        if total > self.limits.max_database_bytes:
            raise ValueError("Catalog database exceeds the configured size limit")
        if shutil.disk_usage(path.parent).free < self.limits.reserve_disk_bytes:
            raise ValueError("Installing the catalog would consume the configured disk reserve")

    def _connect_readonly(self) -> sqlite3.Connection | None:
        if not self.path.is_file():
            return None
        uri = self.path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA cache_size=-4096")
        conn.execute("PRAGMA temp_store=FILE")
        try:
            self._validate_context(conn)
        except BaseException:
            conn.close()
            raise
        return conn

    def _validate_context(self, conn: sqlite3.Connection) -> None:
        try:
            metadata = dict(conn.execute("SELECT key, value FROM catalog_meta"))
        except sqlite3.DatabaseError as exc:
            raise ValueError("Catalog is missing valid release metadata") from exc
        expected = (self.context.dataset, self.context.segmentation_version)
        actual = (metadata.get("dataset"), metadata.get("segmentation_version"))
        if actual != expected:
            raise ValueError("Catalog dataset/release does not match the selected context")
        revision = {"manc": _MANC_REVISION, "mcns": "male-cns-v1.0"}.get(self.context.dataset)
        if revision is not None and metadata.get("annotation_revision") != revision:
            raise ValueError("Catalog annotation revision does not match the selected context")

    def info(self, *, include_root_count=False) -> dict | None:
        """Read product metadata; full distinct-root counting is explicitly opt-in."""
        conn = self._connect_readonly()
        if conn is None:
            return None
        try:
            meta = dict(conn.execute("SELECT key, value FROM catalog_meta"))
            products = self._product_evidence(conn)
            root_count = (conn.execute("SELECT COUNT(DISTINCT root_id) FROM source_rows").fetchone()[0]
                          if include_root_count else None)
            fields = sorted({field for product in products for field in product["fields"]})
            return {"dataset": meta["dataset"], "segmentation_version": meta["segmentation_version"],
                    "products": products, "root_count": root_count, "fields": fields}
        finally:
            conn.close()

    def search(
        self,
        text: str = "",
        filters: Mapping[str, str | Sequence[str]] | None = None,
        limit: int = 500,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> SearchResult:
        if type(limit) is not int or limit < 1 or limit > 5000:
            raise ValueError("Search limit must be between 1 and 5000")
        if not isinstance(text, str):
            raise ValueError("Search text must be text")
        filter_values: dict[str, tuple[str, ...]] = {}
        for field, raw_values in (filters or {}).items():
            if field not in self.search_fields:
                raise ValueError(f"Unsupported catalog filter: {field}")
            values = (raw_values,) if isinstance(raw_values, str) else tuple(raw_values)
            if not values or any(not isinstance(value, str) for value in values):
                raise ValueError("Catalog filter values must be non-empty text values")
            filter_values[field] = tuple(value.casefold() for value in values)
        conn = self._connect_readonly()
        if conn is None:
            return SearchResult((), False)
        try:
            conn.set_progress_handler(lambda: 1 if cancelled() else 0, 1000)
            clauses: list[str] = []
            params: list[str] = []
            folded = text.casefold()
            if folded:
                clauses.append("(instr(root_id, ?) > 0 OR EXISTS (SELECT 1 FROM terms t WHERE t.root_id=source_rows.root_id AND instr(t.value_fold, ?) > 0))")
                params.extend((folded, folded))
            for field, values in filter_values.items():
                if field == "label":
                    alternatives = " OR ".join("instr(t.value_fold, ?) > 0" for _ in values)
                    clauses.append(
                        "EXISTS (SELECT 1 FROM terms t WHERE t.root_id=source_rows.root_id "
                        f"AND t.field=? AND ({alternatives}))"
                    )
                else:
                    placeholders = ",".join("?" for _ in values)
                    clauses.append(
                        "root_id IN (SELECT t.root_id FROM terms t "
                        f"WHERE t.field=? AND t.value_fold IN ({placeholders}))"
                    )
                params.extend((field, *values))
            sql = "SELECT DISTINCT root_id FROM source_rows"
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY length(root_id), root_id LIMIT ?"
            params.append(str(limit + 1))
            try:
                rows = [row[0] for row in conn.execute(sql, params)]
            except sqlite3.OperationalError as exc:
                if cancelled():
                    raise CatalogCancelledError("Catalog search cancelled") from exc
                raise
            _cancelled(cancelled)
            truncated = len(rows) > limit
            return SearchResult(tuple(rows[:limit]), truncated)
        finally:
            conn.close()

    def lookup(self, root_ids: Sequence[str], cancelled: Callable[[], bool] = lambda: False) -> dict:
        ids = tuple(dict.fromkeys(validate_root_id(value) for value in root_ids))
        if len(ids) > _MAX_LOOKUP_IDS:
            raise ValueError(f"Lookup is limited to {_MAX_LOOKUP_IDS} root IDs")
        conn = self._connect_readonly()
        if conn is None:
            return {"products": [], "rows": []}
        try:
            conn.set_progress_handler(lambda: 1 if cancelled() else 0, 1000)
            products = self._product_evidence(conn)
            rows: list[dict] = []
            payload_bytes = 0
            for start in range(0, len(ids), _LOOKUP_CHUNK_SIZE):
                _cancelled(cancelled)
                chunk = ids[start:start + _LOOKUP_CHUNK_SIZE]
                placeholders = ",".join("?" for _ in chunk)
                sql = f"SELECT r.product_index, r.row_index, r.root_id, p.headers_json, r.cells_json FROM source_rows r JOIN products p USING(product_index) WHERE r.root_id IN ({placeholders}) ORDER BY r.product_index, r.row_index"
                for row in conn.execute(sql, chunk):
                    result_row = {"product_index": row[0], "row_index": row[1], "root_id": row[2],
                                  "headers": json.loads(row[3]), "cells": json.loads(row[4])}
                    payload_bytes += len(json.dumps(result_row, ensure_ascii=False).encode("utf-8"))
                    if payload_bytes > self.limits.max_lookup_payload_bytes:
                        raise ValueError("Catalog lookup payload exceeds the configured size limit")
                    rows.append(result_row)
                    if len(rows) > self.limits.max_rows:
                        raise ValueError("Catalog lookup exceeds the configured row limit")
            _cancelled(cancelled)
            return {"products": products, "rows": rows}
        except sqlite3.OperationalError as exc:
            if cancelled():
                raise CatalogCancelledError("Catalog lookup cancelled") from exc
            raise
        finally:
            conn.close()
