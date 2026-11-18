"""Detached catalog review projections; original inputs and scene state stay intact."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json

from .records import FlyWireInputs, InputObservation, InputRow


def enrich_inputs(inputs, catalog, *, cancelled=lambda: False):
    # Equal numeric body/root IDs do not establish a shared annotation identity.
    if inputs.context != catalog.context:
        return inputs
    if inputs.catalog is not None or inputs.observation.input_kind not in {
            "root_id_paste", "root_id_file", "catalog_search", "browser_drop"}:
        return inputs
    snapshot = catalog.lookup(inputs.retrieval_root_ids, cancelled=cancelled)
    return replace(inputs, catalog=snapshot)


def search_inputs(catalog, *, text="", filters=None, cancelled=lambda: False):
    result = catalog.search(text=text, filters=filters, cancelled=cancelled)
    snapshot = catalog.lookup(result.root_ids, cancelled=cancelled)
    query = {"text": text, "filters": filters or {}, "products": snapshot["products"]}
    digest = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
    observation = InputObservation("sha256:" + digest, "catalog_search", "Local annotation search",
                                   datetime.now(timezone.utc).isoformat())
    inputs = FlyWireInputs(catalog.context, observation,
                          tuple(InputRow(i, (root,), root) for i, root in enumerate(result.root_ids, 1)),
                          (), catalog=snapshot)
    return inputs, query, result.truncated


def catalog_values(snapshot, root_id):
    """Display all observed values; conflicting products never overwrite each other."""
    values = {}
    for row in (snapshot or {}).get("rows", ()):
        if row["root_id"] != root_id:
            continue
        products = (snapshot or {}).get("products", ())
        product = products[row["product_index"]] if products else {}
        schema = {item["id"]: item for item in product.get("property_schema", ())}
        for header, value in zip(row["headers"], row["cells"]):
            key = header.strip().casefold()
            if product.get("product") == "segment_properties":
                if key == "type":
                    key = "cell_type"
                elif key == "tags":
                    value = "; ".join(schema[header]["tags"][index] for index in value)
                else:
                    value = str(value)
            elif product.get("dataset") == "mcns":
                key = {"type": "cell_type", "superclass": "super_class", "subclass": "sub_class",
                       "somaside": "side"}.get(key, key)
                # Display projection only: snapshot cells retain nulls, typed
                # lists and numbers. Predictions keep their source headings.
                if value is None:
                    continue
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
            if key == "primary_type":
                key = "cell_type"
            if key == "additional_type(s)":
                key = "additional_cell_types"
            if value and value not in values.setdefault(key, []):
                values[key].append(value)
    return {key: "; ".join(items) for key, items in values.items()}


def unique_fafb_cell_types(snapshot):
    """Project only unambiguous explicit types from FAFB catalog evidence."""
    products = snapshot["products"]
    values = {}
    for row in snapshot["rows"]:
        if products[row["product_index"]]["product"] != "consolidated_cell_types":
            continue
        types = values.setdefault(row["root_id"], set())
        for header, cell in zip(row["headers"], row["cells"]):
            if header == "primary_type":
                types.add(cell.strip())
            elif header == "additional_type(s)":
                types.update(value.strip() for value in cell.split(","))
        types.discard("")
    return {root_id: next(iter(types)) for root_id, types in values.items() if len(types) == 1}


def group_snapshot_rows(snapshot):
    """Build one batch-local index; rows remain exact accepted observations."""
    grouped = {}
    for row in (snapshot or {}).get("rows", ()):
        grouped.setdefault(row["root_id"], []).append(row)
    return grouped


def subset_snapshot(snapshot, root_id, *, rows_by_root=None):
    """Freeze accepted metadata, excluding other reviewed neurons' annotations."""
    if not snapshot:
        return None
    rows = (rows_by_root.get(root_id, ()) if rows_by_root is not None else
            [row for row in snapshot["rows"] if row["root_id"] == root_id])
    used = sorted({row["product_index"] for row in rows})
    remap = {old: new for new, old in enumerate(used)}
    return {"products": [snapshot["products"][index] for index in used],
            "rows": [dict(row, product_index=remap[row["product_index"]]) for row in rows]}
