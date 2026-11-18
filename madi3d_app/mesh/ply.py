"""PLY polygon-mesh codec with lossless numeric attribute preservation."""
from __future__ import annotations

import json
import io
import math
import os
import struct
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

from madi3d_app.io_utils import check_cancelled, check_resources, file_revision, staged_file


_SCHEMA_NAME = "__madi3d_ply_schema_v1"
_EXPORT_MARKER = "madi3d_export_generation"
_BRUSH_MASK = "MADI_BrushMask"
_MAX_RECORDS = 100_000_000
_MAX_LIST_VALUES = 100_000_000
_CHECK_EVERY = 4096
_RESOURCE_CHECK_BYTES = 1024 * 1024
_TYPES = {
    "char": ("i1", "b"), "int8": ("i1", "b"),
    "uchar": ("u1", "B"), "uint8": ("u1", "B"),
    "short": ("i2", "h"), "int16": ("i2", "h"),
    "ushort": ("u2", "H"), "uint16": ("u2", "H"),
    "int": ("i4", "i"), "int32": ("i4", "i"),
    "uint": ("u4", "I"), "uint32": ("u4", "I"),
    "long": ("i4", "i"), "ulong": ("u4", "I"),
    "int64": ("i8", "q"), "uint64": ("u8", "Q"),
    "float": ("f4", "f"), "float32": ("f4", "f"),
    "double": ("f8", "d"), "float64": ("f8", "d"),
}
_CANONICAL_TYPE = {
    "i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort",
    "i4": "int", "u4": "uint", "i8": "int64", "u8": "uint64",
    "f4": "float", "f8": "double",
}


def _type(name):
    try:
        return _TYPES[name.lower()]
    except (AttributeError, KeyError):
        raise ValueError(f"Unsupported PLY numeric type: {name!r}.") from None


def _array_name(index, used):
    base = f"__madi3d_ply_attribute_{index}"
    name = base
    suffix = 0
    reserved = {"CurrentColors", "MADI_BrushMask", _SCHEMA_NAME}
    while any(candidate in used or candidate in reserved
              for candidate in (name, name + "_offsets", name + "_values")):
        suffix += 1
        name = f"{base}_{suffix}"
    used.add(name)
    return name


def _schema_array_name(source_names):
    name = _SCHEMA_NAME
    suffix = 0
    while name in source_names or name in {"CurrentColors", "MADI_BrushMask"}:
        suffix += 1
        name = f"{_SCHEMA_NAME}_{suffix}"
    return name


def _numpy_to_vtk_array(values, name):
    arr = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
    arr.SetName(name)
    return arr


def _vtk_array_values(array):
    if not array.IsA("vtkDataArray"):
        raise ValueError(f"PLY cannot represent nonnumeric VTK array {array.GetName()!r}.")
    values = vtk_to_numpy(array)
    if values.dtype.kind not in "iuf":
        raise ValueError(f"PLY cannot represent VTK array {array.GetName()!r}.")
    return values


def _parse_header(stream):
    first = stream.readline()
    if first.rstrip(b"\r\n") != b"ply":
        raise ValueError("The file does not have a PLY header.")
    fmt = None
    version = None
    elements = []
    comments, obj_info = [], []
    current = None
    header_bytes = len(first)
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("The PLY header is truncated before end_header.")
        header_bytes += len(line)
        if header_bytes > 16 * 1024**2:
            raise ValueError("The PLY header exceeds the supported size limit.")
        try:
            words = line.decode("ascii").strip().split()
        except UnicodeDecodeError:
            raise ValueError("The PLY header must contain ASCII text.") from None
        if not words:
            continue
        key, rest = words[0], words[1:]
        if key == "format":
            if fmt is not None or len(rest) != 2:
                raise ValueError("The PLY header has an invalid format declaration.")
            fmt, version = rest
        elif key == "comment":
            comments.append(line.decode("ascii").rstrip("\r\n")[len("comment"):].lstrip())
        elif key == "obj_info":
            obj_info.append(line.decode("ascii").rstrip("\r\n")[len("obj_info"):].lstrip())
        elif key == "element":
            if len(rest) != 2:
                raise ValueError("The PLY header has an invalid element declaration.")
            try:
                count = int(rest[1])
            except ValueError:
                raise ValueError("The PLY element count is not an integer.") from None
            if count < 0 or count > _MAX_RECORDS:
                raise ValueError("The PLY element count exceeds the supported resource limit.")
            names = {item["name"] for item in elements}
            if rest[0] in names:
                raise ValueError(f"Duplicate PLY element declaration {rest[0]!r}.")
            current = {"name": rest[0], "count": count, "properties": []}
            elements.append(current)
        elif key == "property":
            if current is None:
                raise ValueError("A PLY property appears before its element.")
            if len(rest) == 2:
                typ, prop = rest
                _type(typ)
                p = {"name": prop, "type": typ}
            elif len(rest) == 4 and rest[0] == "list":
                _, count_type, value_type, prop = rest
                if _type(count_type)[0][0] not in "iu":
                    raise ValueError("PLY list count types must be integers.")
                _type(value_type)
                p = {"name": prop, "list": True, "count_type": count_type, "type": value_type}
            else:
                raise ValueError("The PLY header has an invalid property declaration.")
            if p["name"] in {item["name"] for item in current["properties"]}:
                raise ValueError(f"Duplicate property declaration {p['name']!r}.")
            current["properties"].append(p)
        elif key == "end_header":
            break
    if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"} or version != "1.0":
        raise ValueError(f"Unsupported PLY format/version: {fmt} {version}.")
    return fmt, elements, comments, obj_info


def _read_values(stream, fmt, typ, ascii_tokens):
    dtype, code = _type(typ)
    if fmt == "ascii":
        try:
            value = next(ascii_tokens)
            return float(value) if dtype[0] == "f" else int(value, 10)
        except StopIteration:
            raise ValueError("The PLY body is truncated.") from None
        except ValueError:
            raise ValueError("The PLY body contains an invalid numeric value.") from None
    endian = "<" if fmt == "binary_little_endian" else ">"
    raw = stream.read(struct.calcsize(code))
    if len(raw) != struct.calcsize(code):
        raise ValueError("The PLY body is truncated.")
    return struct.unpack(endian + code, raw)[0]


def _read_list_chunk(stream, fmt, typ, size, ascii_tokens, charge):
    dtype, _code = _type(typ)
    itemsize = np.dtype(dtype).itemsize
    charge(size * itemsize, scratch_bytes=size * itemsize)
    if fmt == "ascii":
        try:
            if dtype[0] == "f":
                return np.fromiter((float(next(ascii_tokens)) for _ in range(size)), dtype=dtype, count=size)
            return np.fromiter((int(next(ascii_tokens), 10) for _ in range(size)), dtype=dtype, count=size)
        except StopIteration:
            raise ValueError("The PLY body is truncated.") from None
        except RuntimeError as exc:
            if isinstance(exc.__cause__, StopIteration):
                raise ValueError("The PLY body is truncated.") from None
            raise
        except (ValueError, OverflowError):
            raise ValueError("The PLY body contains an invalid or out-of-range numeric value.") from None
    raw = stream.read(size * itemsize)
    if len(raw) != size * itemsize:
        raise ValueError("The PLY body is truncated.")
    endian = "<" if fmt == "binary_little_endian" else ">"
    return np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder(endian))


def _read_list_chunks(stream, fmt, typ, count, ascii_tokens, cancel_check, charge):
    """Yield bounded typed portions of one list property."""
    dtype, _ = _type(typ)
    itemsize = np.dtype(dtype).itemsize
    per_chunk = max(1, min(_CHECK_EVERY, _RESOURCE_CHECK_BYTES // itemsize))
    remaining = count
    while remaining:
        check_cancelled(cancel_check)
        size = min(remaining, per_chunk)
        yield _read_list_chunk(stream, fmt, typ, size, ascii_tokens, charge)
        remaining -= size


def _chunk_rows(stream, fmt, element, cancel_check, charge, *, retain_lists):
    """Decode element records in bounded chunks, preserving declared property order."""
    props = element["properties"]
    if not props:
        remaining = element["count"]
        while remaining:
            check_cancelled(cancel_check)
            rows = min(remaining, _CHECK_EVERY)
            yield {}
            remaining -= rows
        return
    if fmt != "ascii" and all(not prop.get("list") for prop in props):
        endian = "<" if fmt == "binary_little_endian" else ">"
        record = np.dtype([(prop["name"], np.dtype(_type(prop["type"])[0]).newbyteorder(endian))
                           for prop in props], align=False)
        row_bytes = max(1, record.itemsize)
        per_chunk = max(1, min(_CHECK_EVERY, _RESOURCE_CHECK_BYTES // row_bytes))
        remaining = element["count"]
        while remaining:
            check_cancelled(cancel_check)
            rows = min(remaining, per_chunk)
            charge(rows * row_bytes, force=True)
            raw = stream.read(rows * row_bytes)
            if len(raw) != rows * row_bytes:
                raise ValueError("The PLY body is truncated.")
            decoded = np.frombuffer(raw, dtype=record, count=rows)
            yield {prop["name"]: decoded[prop["name"]] for prop in props}
            remaining -= rows
        return

    token_iter = None
    if fmt == "ascii":
        def tokens():
            while True:
                line = stream.readline()
                if not line:
                    return
                yield from line.split()
        token_iter = tokens()
    row_index = 0
    while row_index < element["count"]:
        check_cancelled(cancel_check)
        rows = min(_CHECK_EVERY, element["count"] - row_index)
        scalar_values = {prop["name"]: [] for prop in props if not prop.get("list")}
        lists = {prop["name"]: [] for prop in props if prop.get("list")}
        for _ in range(rows):
            for prop in props:
                if prop.get("list"):
                    size = _read_values(stream, fmt, prop["count_type"], token_iter)
                    if size < 0 or size > _MAX_LIST_VALUES:
                        raise ValueError("A PLY list length exceeds the supported resource limit.")
                    pieces = []
                    minimum, maximum = None, None
                    if size <= min(_CHECK_EVERY, _RESOURCE_CHECK_BYTES // np.dtype(_type(prop["type"])[0]).itemsize):
                        check_cancelled(cancel_check)
                        values_iter = (_read_list_chunk(stream, fmt, prop["type"], int(size), token_iter, charge),) if size else ()
                    else:
                        values_iter = _read_list_chunks(stream, fmt, prop["type"], int(size), token_iter,
                                                        cancel_check, charge)
                    for values in values_iter:
                        if (not retain_lists and prop["name"] == "vertex_indices"
                                and element["name"] == "face" and len(values)):
                            low, high = (values[0], values[0]) if len(values) == 1 else (min(values), max(values))
                            minimum = low if minimum is None else min(minimum, low)
                            maximum = high if maximum is None else max(maximum, high)
                        if retain_lists:
                            pieces.append(values)
                    if retain_lists:
                        if len(pieces) > 1:
                            dtype = np.dtype(_type(prop["type"])[0])
                            typed_pieces = [piece if isinstance(piece, np.ndarray) and piece.dtype == dtype
                                            else np.asarray(piece, dtype=dtype) for piece in pieces]
                            values = np.concatenate(typed_pieces)
                        else:
                            values = pieces[0] if pieces else ()
                    else:
                        values = (int(size), minimum, maximum)
                    lists[prop["name"]].append(values)
                else:
                    scalar_values[prop["name"]].append(_read_values(stream, fmt, prop["type"], token_iter))
        decoded = {}
        for name, values in scalar_values.items():
            prop = next(p for p in props if p["name"] == name)
            if (fmt == "ascii" and element["name"] == "vertex" and name in {"x", "y", "z"}
                    and _type(prop["type"])[0][0] == "f"):
                # read_ply historically promotes parsed ASCII coordinates directly to float64;
                # retain their decimal parse values while other attributes use declared types.
                charge(rows * np.dtype(np.float64).itemsize, force=True,
                       scratch_bytes=rows * np.dtype(np.float64).itemsize)
                decoded[name] = np.asarray(values, dtype=np.float64)
            else:
                itemsize = np.dtype(_type(prop["type"])[0]).itemsize
                charge(rows * itemsize, force=True, scratch_bytes=rows * itemsize)
                decoded[name] = np.asarray(values, dtype=_type(prop["type"])[0])
        decoded.update(lists)
        yield decoded
        row_index += rows


def _parse_ply(path, cancel_check=None, *, memory_reserved=False, retain=True):
    """Shared streaming decoder and semantic validator for read and validate routes."""
    revision_path = None
    if isinstance(path, (bytes, bytearray, memoryview)):
        payload = bytes(path)
        before = None
        stream = io.BytesIO(payload)
        close_stream = True
    elif hasattr(path, "read"):
        stream = path
        before = None
        close_stream = False
    else:
        revision_path = os.fspath(path)
        before = file_revision(revision_path)
        stream = open(revision_path, "rb")
        close_stream = True
    with (stream if close_stream else nullcontext(stream)):
        fmt, elements, comments, obj_info = _parse_header(stream)
        records = {element["name"]: {prop["name"]: [] for prop in element["properties"]}
                   for element in elements} if retain else None
        memory_estimate = 0
        checked_estimate = 0
        memory_limit = None
        temporary_since_check = 0
        temporary_row_bytes = 0
        temporary_limit = check_resources(memory_bytes=0) if not retain and not memory_reserved else None
        vertex_count = next((element["count"] for element in elements if element["name"] == "vertex"), 0)
        vertex_element = next((element for element in elements if element["name"] == "vertex"), None)
        if vertex_count and (vertex_element is None or not {"x", "y", "z"}.issubset(
                {p["name"] for p in vertex_element["properties"]})):
            raise ValueError("PLY vertex elements must declare x, y, and z coordinates.")
        for name in ("x", "y", "z") if vertex_element is not None else ():
            prop = next(p for p in vertex_element["properties"] if p["name"] == name)
            if prop.get("list"):
                raise ValueError("PLY vertex coordinates must be scalar properties.")
        face_prop = None
        face_element = next((e for e in elements if e["name"] == "face"), None)
        if face_element is not None:
            face_prop = next((p for p in face_element["properties"] if p["name"] == "vertex_indices"), None)
            if face_prop is None or not face_prop.get("list"):
                raise ValueError("PLY face elements must declare a vertex_indices list.")
            if _type(face_prop["type"])[0][0] not in "iu":
                raise ValueError("PLY face vertex_indices must use an integer value type.")
        edge_element = next((e for e in elements if e["name"] == "edge"), None)
        if edge_element is not None:
            names = {p["name"] for p in edge_element["properties"]}
            if not {"vertex1", "vertex2"}.issubset(names):
                raise ValueError("PLY edge elements must declare vertex1 and vertex2.")
            for name in ("vertex1", "vertex2"):
                prop = next(p for p in edge_element["properties"] if p["name"] == name)
                if prop.get("list") or _type(prop["type"])[0][0] not in "iu":
                    raise ValueError("PLY edge endpoints must use integer scalar property types.")

        def charge(amount, *, force=False, scratch_bytes=None):
            nonlocal memory_estimate, memory_limit, checked_estimate
            memory_estimate += int(amount)
            if not memory_reserved and (force or (memory_limit is not None and memory_estimate > memory_limit) or
                                        memory_estimate - checked_estimate >= _RESOURCE_CHECK_BYTES):
                memory_limit = check_resources(memory_bytes=memory_estimate)
                checked_estimate = memory_estimate

        def charge_temporary(amount, *, force=False, scratch_bytes=None):
            nonlocal temporary_since_check, temporary_limit, temporary_row_bytes
            amount = int(amount)
            live_scratch = int(amount if scratch_bytes is None else scratch_bytes)
            if force:
                temporary_row_bytes = live_scratch
            temporary_since_check += amount
            scratch_required = live_scratch if force else temporary_row_bytes + live_scratch
            if not memory_reserved and (force or (temporary_limit is not None and scratch_required > temporary_limit) or
                                        temporary_since_check >= _RESOURCE_CHECK_BYTES):
                temporary_limit = check_resources(memory_bytes=scratch_required)
                temporary_since_check = 0

        for element in elements:
            chunk_charge = charge if retain else charge_temporary
            for chunk in _chunk_rows(stream, fmt, element, cancel_check, chunk_charge,
                                     retain_lists=retain):
                if element["name"] == "vertex" and vertex_element is not None:
                    if not retain:
                        coordinate_scratch = len(chunk["x"]) * (3 * 8 + 3)
                        charge_temporary(0, force=True,
                                         scratch_bytes=temporary_row_bytes + coordinate_scratch)
                    points = np.column_stack([chunk[name].astype(np.float64, copy=False) for name in ("x", "y", "z")])
                    if not np.isfinite(points).all():
                        raise ValueError("PLY coordinates must be finite.")
                    for axis, name in enumerate(("x", "y", "z")):
                        declared = np.asarray(chunk[name], dtype=_type(next(p["type"] for p in element["properties"] if p["name"] == name))[0])
                        if not np.array_equal(declared, points[:, axis].astype(declared.dtype)):
                            raise ValueError("PLY coordinates cannot be represented exactly as vtkPoints float64 values.")
                if element["name"] == "face":
                    for cell in chunk[face_prop["name"]]:
                        if not retain:
                            count, minimum, maximum = cell
                            invalid = count < 3 or (count and (minimum < 0 or maximum >= vertex_count))
                        elif isinstance(cell, tuple):
                            invalid = len(cell) < 3 or min(cell) < 0 or max(cell) >= vertex_count
                        else:
                            invalid = len(cell) < 3 or np.any(cell < 0) or np.any(cell >= vertex_count)
                        if invalid:
                            raise ValueError("A PLY face has invalid vertex indices.")
                if element["name"] == "edge":
                    for left, right in zip(chunk["vertex1"], chunk["vertex2"]):
                        if left < 0 or right < 0 or left >= vertex_count or right >= vertex_count:
                            raise ValueError("A PLY edge has invalid vertex indices.")
                if retain:
                    for name, values in chunk.items():
                        is_list = any(p["name"] == name and p.get("list") for p in element["properties"])
                        if is_list:
                            records[element["name"]][name].extend(values)
                        else:
                            records[element["name"]][name].append(values)
        check_cancelled(cancel_check)
    if before is not None and before != file_revision(revision_path):
        raise OSError("The PLY source changed while it was being read.")

    return fmt, elements, comments, obj_info, records


def validate_ply(path, cancel_check=None, *, memory_reserved=False):
    """Validate PLY declarations and payload without constructing or retaining geometry."""
    _parse_ply(path, cancel_check, memory_reserved=memory_reserved, retain=False)


def read_ply(path, cancel_check=None, *, memory_reserved=False):
    """Read PLY into detached vtkPolyData, retaining ordered properties and metadata."""
    fmt, elements, comments, obj_info, records = _parse_ply(
        path, cancel_check, memory_reserved=memory_reserved, retain=True)

    vertex_element = next((e for e in elements if e["name"] == "vertex"), None)
    def scalar_values(element_name, prop_name, prop_type):
        chunks = records[element_name][prop_name]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=_type(prop_type)[0])

    points = np.empty((vertex_element["count"], 3), dtype=np.float64) if vertex_element else np.empty((0, 3), dtype=np.float64)
    if vertex_element:
        for axis, name in enumerate(("x", "y", "z")):
            prop_type = next(p["type"] for p in vertex_element["properties"] if p["name"] == name)
            points[:, axis] = scalar_values("vertex", name, prop_type)
    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    poly.SetPoints(vtk_points)
    faces, edges = records.get("face", {}).get("vertex_indices", []), []
    edge_data = records.get("edge", {})
    if edge_data:
        edge_element = next(e for e in elements if e["name"] == "edge")
        left_type = next(p["type"] for p in edge_element["properties"] if p["name"] == "vertex1")
        right_type = next(p["type"] for p in edge_element["properties"] if p["name"] == "vertex2")
        edges = list(zip(scalar_values("edge", "vertex1", left_type),
                         scalar_values("edge", "vertex2", right_type)))
    for cells, setter in ((faces, poly.SetPolys), (edges, poly.SetLines)):
        if cells:
            offsets = np.zeros(len(cells) + 1, dtype=np.int64)
            offsets[1:] = np.cumsum([len(c) for c in cells])
            conn = np.fromiter((int(i) for cell in cells for i in cell), dtype=np.int64)
            vtk_cells = vtk.vtkCellArray()
            vtk_cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True), numpy_to_vtkIdTypeArray(conn, deep=True))
            setter(vtk_cells)
    schema = {"version": 1, "format": fmt, "comments": comments, "obj_info": obj_info,
              "elements": []}
    used = {prop["name"] for element in elements for prop in element["properties"]}
    for element in elements:
        out_element = {"name": element["name"], "count": element["count"], "properties": []}
        for prop in element["properties"]:
            desc = dict(prop)
            ename, pname = element["name"], prop["name"]
            if ename == "vertex" and pname in {"x", "y", "z"}:
                desc["mapping"] = "geometry_coordinate"
            elif ename == "face" and pname == "vertex_indices":
                desc["mapping"] = "geometry_faces"
            elif ename == "edge" and pname in {"vertex1", "vertex2"}:
                desc["mapping"] = "geometry_edge"
            else:
                arr_name = _array_name(len(used), used)
                desc["array"] = arr_name
                if prop.get("list"):
                    values = [v for row in records[ename][pname] for v in row]
                    offsets = np.zeros(len(records[ename][pname]) + 1, dtype=np.int64)
                    offsets[1:] = np.cumsum([len(row) for row in records[ename][pname]])
                    poly.GetFieldData().AddArray(_numpy_to_vtk_array(offsets, arr_name + "_offsets"))
                    poly.GetFieldData().AddArray(_numpy_to_vtk_array(np.asarray(values, dtype=_type(prop["type"])[0]), arr_name + "_values"))
                    desc["mapping"] = "list"
                elif ename == "vertex":
                    vals = scalar_values(ename, pname, prop["type"])
                    poly.GetPointData().AddArray(_numpy_to_vtk_array(vals, arr_name))
                    desc["mapping"] = "point_scalar"
                else:
                    vals = scalar_values(ename, pname, prop["type"])
                    poly.GetFieldData().AddArray(_numpy_to_vtk_array(vals, arr_name))
                    desc["mapping"] = "field_scalar"
            out_element["properties"].append(desc)
        schema["elements"].append(out_element)
    meta = vtk.vtkStringArray()
    meta.SetName(_schema_array_name({prop["name"] for element in elements for prop in element["properties"]}))
    meta.InsertNextValue(json.dumps(schema, separators=(",", ":"), ensure_ascii=True))
    poly.GetFieldData().AddArray(meta)
    return poly


def _geometry(poly):
    points = poly.GetPoints()
    if points is None:
        coords = np.empty((0, 3), dtype=np.float64)
    else:
        coords = np.asarray(vtk_to_numpy(points.GetData()), dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(coords).all():
        raise ValueError("PLY coordinates must be finite.")
    def cells_array(cells):
        if cells is None or cells.GetNumberOfCells() == 0:
            return []
        offsets = vtk_to_numpy(cells.GetOffsetsArray()).astype(np.int64, copy=False)
        conn = vtk_to_numpy(cells.GetConnectivityArray()).astype(np.int64, copy=False)
        return [conn[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]
    polys, lines = cells_array(poly.GetPolys()), cells_array(poly.GetLines())
    for cell in polys + lines:
        if any(i < 0 or i >= len(coords) for i in cell):
            raise ValueError("vtkPolyData contains an invalid point index.")
    if any(len(cell) < 3 for cell in polys):
        raise ValueError("PLY polygons must contain at least three point indices.")
    if any(len(cell) != 2 for cell in lines):
        raise ValueError("PLY edge elements require vtkPolyData line cells with exactly two endpoints.")
    return coords, polys, lines


def _encode_scalar(value, typ):
    dtype, code = _type(typ)
    try:
        if dtype[0] in "iu":
            integer = int(value)
            if integer != value:
                raise ValueError
            limits = np.iinfo(np.dtype(dtype))
            if integer < limits.min or integer > limits.max:
                raise ValueError
            scalar = integer
        else:
            scalar = np.asarray(value, dtype=dtype).item()
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"Value cannot be represented by PLY type {typ!r}.") from None
    if dtype[0] == "f" and math.isfinite(float(value)) and not math.isfinite(float(scalar)):
        raise ValueError(f"Value cannot be represented by PLY type {typ!r}.")
    return scalar


def write_ply(
    polydata, path, encoding="binary_little_endian", cancel_check=None, *,
    memory_reserved=False,
):
    """Write PLY, atomically publishing only a complete file."""
    if encoding not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported PLY encoding: {encoding!r}.")
    if not polydata.IsA("vtkPolyData"):
        raise TypeError("write_ply requires vtkPolyData.")
    if polydata.GetNumberOfStrips():
        raise ValueError("PLY cannot represent vtkPolyData triangle strips.")
    coords, polys, lines = _geometry(polydata)
    field = polydata.GetFieldData()
    schema = ply_schema(polydata)
    records = {}
    if schema:
        _validate_schema_arrays(polydata, schema)
        if polydata.GetCellData().GetNumberOfArrays():
            raise ValueError("PLY cannot represent additional vtkCellData arrays.")
        if polydata.GetNumberOfVerts():
            raise ValueError("PLY cannot represent explicit vtkPolyData vertex cells.")
        if schema.get("version") != 1:
            raise ValueError("Unsupported PLY attribute schema.")
        for element in schema["elements"]:
            ename = element["name"]
            rows = [{} for _ in range(_element_count(ename, coords, polys, lines, schema))]
            for prop in element["properties"]:
                mapping = prop.get("mapping")
                if mapping == "geometry_coordinate":
                    axis = {"x": 0, "y": 1, "z": 2}[prop["name"]]
                    values = coords[:, axis]
                elif mapping == "geometry_faces":
                    values = polys
                elif mapping == "geometry_edge":
                    axis = {"vertex1": 0, "vertex2": 1}[prop["name"]]
                    values = [cell[axis] for cell in lines]
                elif mapping in {"point_scalar", "generic_point"}:
                    values = _get_array(polydata.GetPointData(), prop["array"])
                elif mapping in {"field_scalar", "list"}:
                    if mapping == "list":
                        offsets = _get_array(field, prop["array"] + "_offsets").astype(np.int64, copy=False)
                        flat = _get_array(field, prop["array"] + "_values")
                        if len(offsets) != len(rows) + 1 or offsets[0] != 0 or offsets[-1] != len(flat) or np.any(np.diff(offsets) < 0):
                            raise ValueError("PLY list attribute storage is malformed.")
                        values = [flat[offsets[i]:offsets[i + 1]].tolist() for i in range(len(rows))]
                    else:
                        values = _get_array(field, prop["array"])
                else:
                    raise ValueError("PLY schema contains an unsupported attribute mapping.")
                if len(values) != len(rows):
                    raise ValueError(f"PLY attribute {ename}.{prop['name']} has the wrong tuple count.")
                for i, value in enumerate(values):
                    rows[i][prop["name"]] = value
            records[ename] = rows
        comments, obj_info = schema.get("comments", []), schema.get("obj_info", [])
        elements = schema["elements"]
    else:
        comments, obj_info = [], []
        elements = []
        elements.append({"name": "vertex", "properties": [
            {"name": axis, "type": "double", "mapping": "geometry_coordinate"} for axis in "xyz"]})
        records["vertex"] = [{"x": p[0], "y": p[1], "z": p[2]} for p in coords]
        normals = polydata.GetPointData().GetNormals()
        for arr in _numeric_point_arrays(polydata):
            data = _vtk_array_values(arr)
            typ = _CANONICAL_TYPE.get(data.dtype.str[1:])
            if typ is None:
                raise ValueError(f"Unsupported point array {arr.GetName()!r}.")
            if arr is normals and arr.GetNumberOfComponents() == 3:
                # VTK stores cache normals as a vector; PLY names their scalar
                # components nx/ny/nz. Other vectors remain unsupported.
                properties = zip(("nx", "ny", "nz"), data.T)
            elif arr.GetNumberOfComponents() == 1:
                properties = ((arr.GetName(), data),)
            else:
                raise ValueError(f"Unsupported point array {arr.GetName()!r}.")
            for name, values in properties:
                if not name or any(prop["name"] == name for prop in elements[0]["properties"]):
                    raise ValueError(f"Point array {name!r} cannot be represented as a unique PLY property.")
                elements[0]["properties"].append({
                    "name": name, "type": typ, "mapping": "generic_point", "array": arr.GetName(),
                })
                for row, value in zip(records["vertex"], values):
                    row[name] = value
        for element_name, cells, geometry in (("face", polys, "geometry_faces"), ("edge", lines, "geometry_edge")):
            if not cells:
                continue
            if element_name == "face":
                elements.append({"name": "face", "properties": [{"name": "vertex_indices", "list": True,
                    "count_type": "uint", "type": "int", "mapping": geometry}]})
                records["face"] = [{"vertex_indices": c} for c in cells]
            else:
                elements.append({"name": "edge", "properties": [
                    {"name": "vertex1", "type": "int", "mapping": geometry},
                    {"name": "vertex2", "type": "int", "mapping": geometry}]})
                records["edge"] = [{"vertex1": c[0], "vertex2": c[1]} for c in cells]
        cell_arrays = _numeric_polygon_cell_arrays(polydata, polys)
        if cell_arrays:
            face = next(element for element in elements if element["name"] == "face")
            face_rows = records["face"]
            for arr, values, typ in cell_arrays:
                name = arr.GetName()
                face["properties"].append({
                    "name": name, "type": typ, "mapping": "generic_face", "array": name,
                })
                for row, value in zip(face_rows, values):
                    row[name] = value.item()
        if (_unsupported_field_arrays(polydata, None)
                or (polydata.GetNumberOfVerts() and (polys or lines))):
            raise ValueError("Generic vtkPolyData contains arrays that PLY cannot represent without a PLY schema.")
    if hasattr(path, "write"):
        _write_stream(path, encoding, comments, obj_info, elements, records, cancel_check)
        return
    with staged_file(path, cancel_check=cancel_check) as stage:
        with open(stage, "wb") as stream:
            _write_stream(stream, encoding, comments, obj_info, elements, records, cancel_check)


def _write_stream(stream, encoding, comments, obj_info, elements, records, cancel_check):
    endian = "<" if encoding == "binary_little_endian" else ">"
    stream.write(b"ply\n")
    stream.write(f"format {encoding} 1.0\n".encode("ascii"))
    for text in comments:
        stream.write(("comment " + str(text) + "\n").encode("ascii"))
    for text in obj_info:
        stream.write(("obj_info " + str(text) + "\n").encode("ascii"))
    for element in elements:
        element_name = element["name"]
        rows = records[element_name]
        stream.write(f"element {element_name} {len(rows)}\n".encode("ascii"))
        for prop in element["properties"]:
            if prop.get("list"):
                stream.write(f"property list {prop['count_type']} {prop['type']} {prop['name']}\n".encode("ascii"))
            else:
                stream.write(f"property {prop['type']} {prop['name']}\n".encode("ascii"))
    stream.write(b"end_header\n")
    for element in elements:
        for row_index, row in enumerate(records[element["name"]]):
            if row_index % _CHECK_EVERY == 0:
                check_cancelled(cancel_check)
            ascii_values = [] if encoding == "ascii" else None
            for prop in element["properties"]:
                value = row[prop["name"]]
                if prop.get("list"):
                    if len(value) > _MAX_LIST_VALUES:
                        raise ValueError("A PLY list exceeds the supported resource limit.")
                    count = _encode_scalar(len(value), prop["count_type"])
                    if ascii_values is not None:
                        ascii_values.append(str(count))
                    else:
                        stream.write(struct.pack(endian + _type(prop["count_type"])[1], count))
                    for item_index, item in enumerate(value):
                        if item_index % _CHECK_EVERY == 0:
                            check_cancelled(cancel_check)
                        scalar = _encode_scalar(item, prop["type"])
                        if ascii_values is not None:
                            ascii_values.append(str(scalar))
                        else:
                            stream.write(struct.pack(endian + _type(prop["type"])[1], scalar))
                else:
                    scalar = _encode_scalar(value, prop["type"])
                    if ascii_values is not None:
                        ascii_values.append(str(scalar))
                    else:
                        stream.write(struct.pack(endian + _type(prop["type"])[1], scalar))
            if ascii_values is not None:
                stream.write((" ".join(ascii_values) + "\n").encode("ascii"))


def encode_ply(
    polydata, encoding="binary_little_endian", cancel_check=None, *,
    memory_reserved=False,
):
    """Encode PLY to bytes for bounded in-memory project resources."""
    import io
    stream = io.BytesIO()
    write_ply(polydata, stream, encoding=encoding, cancel_check=cancel_check,
              memory_reserved=memory_reserved)
    return stream.getvalue()


def _element_count(name, coords, polys, lines, schema):
    if name == "vertex":
        return len(coords)
    if name == "face":
        return len(polys)
    if name == "edge":
        return len(lines)
    # Unknown elements are represented by aligned FieldData tuples.
    for element in schema["elements"]:
        if element["name"] == name:
            return int(element.get("count", 0))
    return 0


def _get_array(attributes, name):
    arr = attributes.GetArray(name) if attributes else None
    if arr is None:
        raise ValueError(f"PLY attribute storage {name!r} is missing.")
    return _vtk_array_values(arr).reshape(-1)


def _numeric_point_arrays(polydata):
    data = polydata.GetPointData()
    arrays = [data.GetAbstractArray(i) for i in range(data.GetNumberOfArrays())
              if data.GetAbstractArray(i).GetName() not in {_SCHEMA_NAME, _BRUSH_MASK}]
    for array in arrays:
        _vtk_array_values(array)
    return arrays


def _numeric_polygon_cell_arrays(polydata, polygons):
    data = polydata.GetCellData()
    arrays = [data.GetAbstractArray(i) for i in range(data.GetNumberOfArrays())]
    if not arrays:
        return ()
    if (not polygons or polydata.GetNumberOfVerts() or polydata.GetNumberOfLines()
            or polydata.GetNumberOfStrips()):
        raise ValueError(
            "Generic PLY cell attributes require polygon-only geometry."
        )
    used_names = {"vertex_indices"}
    result = []
    for array in arrays:
        name = array.GetName()
        if not name or name in used_names:
            raise ValueError(
                f"Cell array {name!r} cannot be represented as a unique PLY face property."
            )
        used_names.add(name)
        if array.GetNumberOfComponents() != 1:
            raise ValueError(f"Unsupported cell array {name!r}: expected one component.")
        if array.GetNumberOfTuples() != len(polygons):
            raise ValueError(f"Cell array {name!r} has the wrong tuple count.")
        values = _vtk_array_values(array)
        typ = _CANONICAL_TYPE.get(values.dtype.str[1:])
        if typ is None:
            raise ValueError(f"Unsupported cell array {name!r} dtype.")
        result.append((array, values, typ))
    return tuple(result)


def _unsupported_field_arrays(polydata, schema):
    field = polydata.GetFieldData()
    ignored = {_SCHEMA_NAME, _EXPORT_MARKER}
    if schema:
        for element in schema.get("elements", []):
            for prop in element["properties"]:
                if prop.get("array"):
                    ignored.update({prop["array"], prop["array"] + "_offsets", prop["array"] + "_values"})
    return any(field.GetAbstractArray(i).GetName() not in ignored for i in range(field.GetNumberOfArrays())) if field else False


def _validate_schema_arrays(polydata, schema):
    point_names, field_names = set(), {_SCHEMA_NAME, _EXPORT_MARKER}
    for element in schema.get("elements", []):
        for prop in element.get("properties", []):
            name = prop.get("array")
            if not name:
                continue
            if prop.get("mapping") == "point_scalar":
                point_names.add(name)
            elif prop.get("mapping") in {"field_scalar", "list"}:
                field_names.add(name)
                if prop.get("mapping") == "list":
                    field_names.update({name + "_offsets", name + "_values"})
    point = polydata.GetPointData()
    for index in range(point.GetNumberOfArrays()):
        array = point.GetAbstractArray(index)
        # The declaration owns source attributes. Cache-generated display
        # normals are derived and must not replace its recorded nx/ny/nz.
        if array.GetName() == _BRUSH_MASK or array is point.GetNormals():
            continue
        if array.GetName() not in point_names:
            raise ValueError(f"PLY polydata has an unmapped point array {array.GetName()!r}.")
    field = polydata.GetFieldData()
    if field:
        for index in range(field.GetNumberOfArrays()):
            array = field.GetAbstractArray(index)
            if _is_schema_array(array):
                continue
            if array.GetName() not in field_names:
                raise ValueError(f"PLY polydata has an unmapped field array {array.GetName()!r}.")


def _is_schema_array(array):
    name = array.GetName() if array is not None else None
    return bool(name and (name == _SCHEMA_NAME or name.startswith(_SCHEMA_NAME + "_")))


def ply_schema(polydata):
    """Return a detached copy of the embedded PLY declaration schema, if present."""
    if polydata is None or not hasattr(polydata, "IsA") or not polydata.IsA("vtkPolyData"):
        return None
    field = polydata.GetFieldData()
    if not field:
        return None
    for index in range(field.GetNumberOfArrays()):
        value = field.GetAbstractArray(index)
        if (not _is_schema_array(value) or not value.IsA("vtkStringArray")
                or value.GetNumberOfValues() != 1):
            continue
        try:
            schema = json.loads(value.GetValue(0))
        except (TypeError, ValueError):
            continue
        if isinstance(schema, dict) and schema.get("version") == 1 and isinstance(schema.get("elements"), list):
            return schema
    return None


def _replace_schema(polydata, schema):
    field = polydata.GetFieldData()
    for index in range(field.GetNumberOfArrays()):
        array = field.GetAbstractArray(index)
        if _is_schema_array(array) and array.IsA("vtkStringArray"):
            replacement = vtk.vtkStringArray()
            replacement.SetName(array.GetName())
            replacement.InsertNextValue(json.dumps(schema, separators=(",", ":"), ensure_ascii=True))
            field.AddArray(replacement)
            return
    raise ValueError("PLY source schema is missing.")


def is_ply_polydata(polydata):
    return ply_schema(polydata) is not None


def ply_texture_path(polydata, source_path):
    """Return a contained relative TextureFile reference, or None when absent/unsafe."""
    schema = ply_schema(polydata)
    if schema is None:
        return None
    references = [comment[len("TextureFile "):].strip() for comment in schema.get("comments", [])
                  if comment.startswith("TextureFile ")]
    if len(references) != 1 or not references[0]:
        return None
    relative = Path(references[0])
    if relative.is_absolute() or ".." in relative.parts:
        return None
    base = Path(source_path).resolve().parent
    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        return None
    return resolved


def relink_ply_texture(polydata, filename):
    """Change only the texture reference on a detached export copy."""
    schema = ply_schema(polydata)
    if schema is None:
        raise ValueError("PLY texture relinking requires source PLY schema.")
    references = [index for index, comment in enumerate(schema.get("comments", ()))
                  if comment.startswith("TextureFile ")]
    if (len(references) != 1 or not filename or Path(filename).name != filename
            or "\r" in filename or "\n" in filename):
        raise ValueError("PLY has an ambiguous or unsafe texture reference.")
    schema["comments"][references[0]] = f"TextureFile {filename}"
    _replace_schema(polydata, schema)


def realized_ply_coordinates(polydata, points, cancel_check=None, *, memory_reserved=False):
    """Realize new coordinates while borrowing immutable topology and attributes."""
    schema = ply_schema(polydata)
    if schema is None:
        raise ValueError("PLY coordinate realization requires embedded PLY schema.")
    mapped = np.asarray(points)
    source_points = polydata.GetPoints() if polydata is not None else None
    point_count = source_points.GetNumberOfPoints() if source_points is not None else 0
    if mapped.dtype != np.dtype(np.float64) or mapped.shape != (point_count, 3):
        raise ValueError("Realized PLY coordinates must be matching Nx3 float64 values.")
    if not np.isfinite(mapped).all():
        raise ValueError("Realized PLY coordinates must be finite.")

    safely_invariant = {"red", "green", "blue", "alpha", "label", "confidence",
                        "intensity", "quality", "region", "s", "t", "u", "v"}
    vertex = None
    for element in schema["elements"]:
        name = element.get("name")
        if name not in {"vertex", "face", "edge"}:
            raise ValueError("PLY coordinate realization cannot interpret custom elements.")
        if name == "vertex":
            vertex = element
        for prop in element.get("properties", []):
            mapping = prop.get("mapping")
            if mapping in {"geometry_coordinate", "geometry_faces", "geometry_edge"}:
                continue
            if name == "vertex" and prop.get("name") in {"nx", "ny", "nz"}:
                continue
            if name == "face" and prop.get("name") == "texcoord" and mapping == "list":
                continue
            if prop.get("name") not in safely_invariant or mapping not in {"point_scalar", "field_scalar"}:
                raise ValueError(
                    f"PLY coordinate realization cannot preserve {name} property {prop.get('name')!r}."
                )
    coordinate_properties = {
        prop.get("name") for prop in (vertex or {}).get("properties", [])
        if prop.get("mapping") == "geometry_coordinate"
    }
    if coordinate_properties != {"x", "y", "z"}:
        raise ValueError("PLY coordinate realization requires x, y, and z coordinate properties.")

    point_data = polydata.GetPointData()
    active_point_normals = point_data.GetNormals()
    active_cell_normals = polydata.GetCellData().GetNormals()
    cell_data = polydata.GetCellData()
    if cell_data.GetNumberOfArrays() > (1 if active_cell_normals is not None else 0):
        raise ValueError("PLY coordinate realization cannot preserve unmapped cell arrays.")
    if not memory_reserved:
        check_resources(memory_bytes=mapped.nbytes)
    check_cancelled(cancel_check)

    result = vtk.vtkPolyData()
    result.ShallowCopy(polydata)
    schema = ply_schema(result)
    vertex = next(element for element in schema["elements"] if element["name"] == "vertex")
    point_data = result.GetPointData()
    normal_names = {"nx", "ny", "nz"}
    retained_properties = []
    for prop in vertex["properties"]:
        if prop.get("name") in normal_names and prop.get("mapping") == "point_scalar":
            point_data.RemoveArray(prop["array"])
        else:
            retained_properties.append(prop)
    vertex["properties"] = retained_properties

    # Renderer-generated normals are disposable display data; other unknown arrays
    # remain errors under the embedded schema contract.
    for data, normals in ((point_data, active_point_normals),
                          (result.GetCellData(), active_cell_normals)):
        for index in range(data.GetNumberOfArrays() - 1, -1, -1):
            array = data.GetAbstractArray(index)
            name = array.GetName()
            if array is normals or (name and name.casefold() == "normals"):
                data.RemoveArray(name)
    if point_data.GetNormals() is not None:
        point_data.SetNormals(None)
    if result.GetCellData().GetNormals() is not None:
        result.GetCellData().SetNormals(None)

    _validate_schema_arrays(result, schema)
    if result.GetCellData().GetNumberOfArrays():
        raise ValueError("PLY coordinate realization cannot preserve unmapped cell arrays.")
    for prop in vertex["properties"]:
        if prop.get("mapping") == "geometry_coordinate":
            prop["type"] = "double"
    target_points = vtk.vtkPoints()
    target_points.SetData(numpy_to_vtk(np.ascontiguousarray(mapped), deep=True))
    result.SetPoints(target_points)
    _geometry(result)
    check_cancelled(cancel_check)
    _replace_schema(result, schema)
    return result


def transformed_ply(polydata, matrix, cancel_check=None, *, memory_reserved=False):
    """Copy a PLY mesh and apply an affine transform while preserving its schema."""
    schema = ply_schema(polydata)
    if schema is None:
        raise ValueError("PLY transform requires embedded PLY schema.")
    if hasattr(matrix, "GetElement"):
        transform = np.asarray([[matrix.GetElement(row, col) for col in range(4)] for row in range(4)],
                               dtype=np.float64)
    else:
        transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("PLY transform must be a finite 4 by 4 matrix.")
    if not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError("PLY transform must be affine.")
    if not memory_reserved:
        check_resources(memory_bytes=polydata.GetActualMemorySize() * 1024)
    result = vtk.vtkPolyData()
    result.DeepCopy(polydata)
    if np.array_equal(transform, np.eye(4)):
        return result
    safely_invariant = {"red", "green", "blue", "alpha", "label", "confidence",
                        "intensity", "quality", "region", "s", "t", "u", "v"}
    for element in schema["elements"]:
        if element["name"] not in {"vertex", "face", "edge"}:
            raise ValueError("Baked PLY export cannot interpret custom elements under a transform; use native coordinates.")
        for prop in element["properties"]:
            if prop.get("mapping") in {"geometry_coordinate", "geometry_faces", "geometry_edge"}:
                continue
            if element["name"] == "vertex" and prop["name"] in {"nx", "ny", "nz"}:
                continue
            if element["name"] == "face" and prop["name"] == "texcoord":
                continue
            if prop["name"] not in safely_invariant or prop.get("mapping") == "list":
                raise ValueError(f"Baked PLY export cannot interpret {element['name']} property {prop['name']!r} under a transform; use native coordinates.")
    check_cancelled(cancel_check)
    coords = np.asarray(vtk_to_numpy(result.GetPoints().GetData()), dtype=np.float64).reshape(-1, 3)
    if not memory_reserved:
        check_resources(memory_bytes=coords.nbytes)
    linear = transform[:3, :3]
    determinant_sign, _ = np.linalg.slogdet(linear)
    if determinant_sign == 0:
        raise ValueError("PLY transform has a singular or numerically unstable linear component.")
    try:
        inverse = np.linalg.inv(linear)
    except np.linalg.LinAlgError:
        raise ValueError("PLY transform has a singular linear component.") from None
    if not np.isfinite(inverse).all():
        raise ValueError("PLY transform inverse is not finite.")
    transformed = coords @ linear.T + transform[:3, 3]
    if not np.isfinite(transformed).all():
        raise ValueError("Transformed PLY coordinates are not finite.")
    result.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(transformed), deep=True))

    vertex = next((element for element in schema["elements"] if element["name"] == "vertex"), None)
    for prop in (vertex or {}).get("properties", []):
        if prop.get("mapping") == "geometry_coordinate":
            prop["type"] = "double"
    normal_props = {prop["name"]: prop for prop in (vertex or {}).get("properties", [])
                    if prop.get("mapping") == "point_scalar"}
    if {"nx", "ny", "nz"}.issubset(normal_props):
        normals = np.column_stack([_get_array(result.GetPointData(), normal_props[name]["array"])
                                   for name in ("nx", "ny", "nz")]).astype(np.float64)
        normals = normals @ inverse
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 0
        normals[valid] /= lengths[valid, None]
        if not np.isfinite(normals).all():
            raise ValueError("Transformed PLY normals are not finite.")
        for col, name in enumerate(("nx", "ny", "nz")):
            prop = normal_props[name]
            replacement = _numpy_to_vtk_array(normals[:, col], prop["array"])
            result.GetPointData().RemoveArray(prop["array"])
            result.GetPointData().AddArray(replacement)
            prop["type"] = "double"

    if determinant_sign < 0:
        faces = _geometry(result)[1]
        reversed_faces = [list(reversed(cell)) for cell in faces]
        offsets = np.zeros(len(reversed_faces) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(cell) for cell in reversed_faces])
        conn = np.asarray([i for cell in reversed_faces for i in cell], dtype=np.int64)
        cells = vtk.vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True), numpy_to_vtkIdTypeArray(conn, deep=True))
        result.SetPolys(cells)
        for element in schema["elements"]:
            if element["name"] != "face":
                continue
            for prop in element["properties"]:
                if prop.get("mapping") != "list" or prop["name"] != "texcoord":
                    continue
                offsets = _get_array(result.GetFieldData(), prop["array"] + "_offsets").astype(np.int64, copy=False)
                values = _get_array(result.GetFieldData(), prop["array"] + "_values")
                reordered = values.copy()
                for index in range(len(offsets) - 1):
                    start, end = int(offsets[index]), int(offsets[index + 1])
                    face_values = values[start:end]
                    if len(face_values) != 2 * len(faces[index]):
                        raise ValueError("PLY face texcoord list does not contain a pair for each corner.")
                    reordered[start:end] = face_values.reshape(-1, 2)[::-1].reshape(-1)
                array = _numpy_to_vtk_array(reordered, prop["array"] + "_values")
                result.GetFieldData().RemoveArray(prop["array"] + "_values")
                result.GetFieldData().AddArray(array)
    check_cancelled(cancel_check)
    _replace_schema(result, schema)
    return result
