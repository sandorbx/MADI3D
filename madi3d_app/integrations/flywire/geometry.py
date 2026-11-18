"""Bounded writers for detached FlyWire mesh and skeleton geometry."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

from madi3d_app.io_utils import check_cancelled, staged_file
from madi3d_app.tools.swc.model import SwcNode
from madi3d_app.mesh.export import staged_polydata
from madi3d_app.mesh.cache import _read_vtp, generate_cache_file


@dataclass(frozen=True)
class GeometryLimits:
    max_vertices: int | None = None
    max_primitives: int | None = None
    max_payload_bytes: int | None = None
    max_artifact_bytes: int | None = None

    def __post_init__(self):
        for name in ("max_vertices", "max_primitives", "max_payload_bytes", "max_artifact_bytes"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")


class GeometryUnavailable(ValueError):
    """Raised when supplied geometry cannot be represented without loss or invention."""


DEFAULT_LIMITS = GeometryLimits()
_CHUNK_ROWS = 16_384


def _array(value, name, shape_tail, kinds):
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array.") from exc
    if arr.ndim != 2 or arr.shape[1:] != shape_tail or arr.dtype.kind not in kinds:
        raise ValueError(f"{name} must have shape (n, {shape_tail[0]}) and a supported numeric dtype.")
    return arr


def _limits_check(vertices, primitives, limits):
    if limits.max_vertices is not None and vertices > limits.max_vertices:
        raise ValueError("Geometry exceeds the vertex limit.")
    if limits.max_primitives is not None and primitives > limits.max_primitives:
        raise ValueError("Geometry exceeds the primitive limit.")


def _artifact_limit_check(size, limits):
    if limits.max_artifact_bytes is not None and size > limits.max_artifact_bytes:
        raise ValueError("Artifact exceeds the artifact size limit.")


def verification_memory_bytes(representation, vertices, primitives):
    """Known overlapping arrays for verification, including bounded scratch."""
    for name, value in (("vertices", vertices), ("primitives", primitives)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Artifact summary has an invalid {name} count.")
    if representation == "mesh":
        # Canonical VTP: float64 XYZ + float32 point normals, and int64
        # triangle connectivity + offsets. NumPy views borrow these buffers.
        persistent = 36 * vertices + 32 * primitives + 8
        scratch = max(9 * min(primitives, _CHUNK_ROWS),
                      3 * min(vertices, _CHUNK_ROWS))
        return persistent + scratch
    if representation == "swc":
        # Union/rank arrays are known; the exact-ID index and text parser's
        # Python allocations remain unknown (the caller retains unknown scopes).
        return 9 * (vertices + 1)
    raise ValueError("Unsupported geometry representation.")


def _summary(vertices, primitives, low, high, *, components=None):
    bounds = None if vertices == 0 else tuple(float(x) for x in (*low, *high))
    summary = {"vertices": int(vertices), "primitives": int(primitives), "bounds_um": bounds}
    if components is not None:
        summary["components"] = int(components)
    return summary


def write_mesh(path, vertices_nm, faces, limits=DEFAULT_LIMITS):
    """Write finite triangle geometry as bounded VTP in micrometre coordinates."""
    vertices = _array(vertices_nm, "vertices_nm", (3,), "iuf")
    triangles = _array(faces, "faces", (3,), "iu")
    n_vertices, n_faces = len(vertices), len(triangles)
    if n_vertices == 0 or n_faces == 0:
        raise GeometryUnavailable("Mesh vertices and triangle faces are unavailable.")
    _limits_check(n_vertices, n_faces, limits)
    if triangles.size and (triangles.min() < 0 or triangles.max() >= n_vertices):
        raise ValueError("Triangle indices must refer to existing vertices.")
    converted = np.array(vertices, dtype=np.float64, copy=True)
    if not np.isfinite(converted).all():
        raise ValueError("Mesh coordinates must be finite.")
    converted *= 0.001
    if not np.isfinite(converted).all():
        raise ValueError("Converted mesh coordinates must be finite.")
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(converted, deep=True))
    connectivity = np.asarray(triangles, dtype=np.int64).reshape(-1).copy()
    offsets = np.arange(n_faces + 1, dtype=np.int64) * 3
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(connectivity, deep=True))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    polydata = generate_cache_file(polydata, None)
    actual = vtk_to_numpy(polydata.GetPoints().GetData())
    low, high = actual.min(axis=0), actual.max(axis=0)
    target = Path(path)
    with staged_polydata(polydata, target) as stage:
        _artifact_limit_check(Path(stage).stat().st_size, limits)
        # staged_polydata publishes only after serialization and this size check.
    return _summary(polydata.GetNumberOfPoints(), polydata.GetNumberOfPolys(), low, high)


def _skeleton_transform(transform):
    try:
        matrix = np.asarray(transform, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Skeleton transform must be a finite 3 by 4 matrix.") from exc
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
        raise ValueError("Skeleton transform must be a finite 3 by 4 matrix.")
    linear = matrix[:, :3]
    if not np.allclose(linear.T @ linear, np.eye(3), rtol=0.0, atol=1e-6):
        raise GeometryUnavailable("Skeleton radii cannot represent transforms containing scale or shear.")
    return matrix


def _forest_parents(vertex_count, edges):
    """Validate an undirected forest and return its deterministic BFS parents."""
    degree = np.zeros(vertex_count, dtype=np.int64)
    representative = np.arange(vertex_count, dtype=np.int64)
    rank = np.zeros(vertex_count, dtype=np.uint8)

    for a, b in edges:
        a, b = int(a), int(b)
        if a == b or _find_root(representative, a) == _find_root(representative, b):
            raise GeometryUnavailable("Skeleton edges must form a forest without self or duplicate edges.")
        ra, rb = _find_root(representative, a), _find_root(representative, b)
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        representative[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        degree[a] += 1
        degree[b] += 1

    offsets = np.empty(vertex_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(degree, out=offsets[1:])
    adjacent = np.empty(int(offsets[-1]), dtype=np.int64)
    cursor = offsets[:-1].copy()
    for a, b in edges:
        a, b = int(a), int(b)
        adjacent[cursor[a]] = b
        adjacent[cursor[b]] = a
        cursor[a] += 1
        cursor[b] += 1
    for vertex in range(vertex_count):
        adjacent[offsets[vertex]:offsets[vertex + 1]].sort()

    parents = np.full(vertex_count, -2, dtype=np.int64)
    queue = np.empty(vertex_count, dtype=np.int64)
    for root in range(vertex_count):
        if parents[root] != -2:
            continue
        parents[root] = -1
        head = tail = 0
        queue[tail] = root
        tail += 1
        while head < tail:
            current = int(queue[head])
            head += 1
            for neighbor in adjacent[offsets[current]:offsets[current + 1]]:
                neighbor = int(neighbor)
                if parents[neighbor] == -2:
                    parents[neighbor] = current
                    queue[tail] = neighbor
                    tail += 1
    return parents


def write_skeleton(path, vertices_nm, edges, radii_nm, transform, limits=DEFAULT_LIMITS):
    """Write a deterministic SWC forest from detached nanometre geometry."""
    vertices = _array(vertices_nm, "vertices_nm", (3,), "iuf")
    edge_array = _array(edges, "edges", (2,), "iu")
    if radii_nm is None:
        raise GeometryUnavailable("Skeleton radii are unavailable.")
    try:
        radii = np.asarray(radii_nm, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeometryUnavailable("Skeleton radii are unavailable or invalid.") from exc
    if radii.ndim != 1 or len(radii) != len(vertices) or not np.isfinite(radii).all() or (radii < 0).any():
        raise GeometryUnavailable("Skeleton radii are unavailable or invalid.")
    n_vertices, n_edges = len(vertices), len(edge_array)
    if n_vertices == 0:
        raise GeometryUnavailable("Skeleton vertices are unavailable.")
    _limits_check(n_vertices, n_edges, limits)
    if edge_array.size and (edge_array.min() < 0 or edge_array.max() >= n_vertices):
        raise ValueError("Skeleton edge indices must refer to existing vertices.")
    matrix = _skeleton_transform(transform)
    parents = _forest_parents(n_vertices, edge_array)
    component_count = int(np.count_nonzero(parents == -1))
    target = Path(path)
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    # Stream rows directly and enforce only an explicitly configured size cap.
    with target.open("wb") as stream:
        for start in range(0, n_vertices, _CHUNK_ROWS):
            stop = min(start + _CHUNK_ROWS, n_vertices)
            points = vertices[start:stop].astype(np.float64, copy=True)
            if not np.isfinite(points).all():
                raise ValueError("Skeleton coordinates must be finite.")
            points = points @ matrix[:, :3].T + matrix[:, 3]
            points *= 0.001
            scaled_radii = radii[start:stop] * 0.001
            if not np.isfinite(points).all() or not np.isfinite(scaled_radii).all():
                raise ValueError("Converted skeleton coordinates and radii must be finite.")
            if len(points):
                low = np.minimum(low, points.min(axis=0))
                high = np.maximum(high, points.max(axis=0))
            for local, point in enumerate(points):
                index = start + local
                parent_id = int(parents[index]) + 1 if parents[index] >= 0 else -1
                row = (f"{index + 1} 0 {point[0]:.17g} {point[1]:.17g} {point[2]:.17g} "
                       f"{scaled_radii[local]:.17g} {parent_id}\n")
                encoded = row.encode("ascii")
                if (limits.max_artifact_bytes is not None
                        and stream.tell() + len(encoded) > limits.max_artifact_bytes):
                    raise ValueError("SWC artifact exceeds the artifact size limit.")
                stream.write(encoded)
    return _summary(n_vertices, n_edges, low, high, components=component_count)


def _summary_count(summary, key):
    if not isinstance(summary, dict):
        raise ValueError("Artifact summary must be a mapping.")
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Artifact summary has an invalid {key} count.")
    return value


def _check_bounds(summary, low, high):
    bounds = summary.get("bounds_um")
    if (not isinstance(bounds, (tuple, list)) or len(bounds) != 6
            or not np.isfinite(np.asarray(bounds, dtype=np.float64)).all()):
        raise ValueError("Artifact summary has invalid micrometre bounds.")
    actual = np.concatenate((low, high))
    if not np.allclose(actual, np.asarray(bounds, dtype=np.float64), rtol=1e-12, atol=1e-12):
        raise ValueError("Artifact coordinates do not match the summary bounds.")


def _find_root(parent, vertex):
    root = vertex
    while parent[root] != root:
        root = int(parent[root])
    while parent[vertex] != vertex:
        following = int(parent[vertex])
        parent[vertex] = root
        vertex = following
    return root


def _swc_record(line) -> tuple[int, SwcNode] | None:
    """Parse one complete source record without repairs or float-form IDs."""
    if not line.strip() or line.lstrip().startswith(b"#"):
        return None
    try:
        fields = line.decode("ascii").split()
        if len(fields) != 7 or any(re.fullmatch(r"[+-]?[0-9]+", fields[i]) is None for i in (0, 1, 6)):
            raise ValueError
        node_id, node_type, parent_id = (int(fields[i]) for i in (0, 1, 6))
        x, y, z, radius = (float(value) for value in fields[2:6])
    except (UnicodeDecodeError, ValueError, OverflowError):
        raise ValueError("SWC contains an invalid seven-column record.") from None
    if node_id <= 0 or (parent_id != -1 and parent_id <= 0):
        raise ValueError("SWC contains an invalid node or parent ID.")
    if not all(math.isfinite(value) for value in (x, y, z, radius)) or radius < 0:
        raise ValueError("SWC contains non-finite coordinates or an invalid radius.")
    return node_id, (node_type, x, y, z, radius, parent_id)


def normalize_swc(path, source, source_unit_nm, limits=DEFAULT_LIMITS, *, cancel_check=None):
    """Normalize strict source SWC to µm, retaining supplied node semantics.

    Source comments are quoted as source evidence so their original unit claims
    cannot be mistaken for the normalized output's units. Topology is checked
    before the staged artifact replaces its destination.
    """
    if not math.isfinite(source_unit_nm) or source_unit_nm <= 0:
        raise ValueError("SWC source units must be finite and positive.")
    scale = source_unit_nm * 0.001
    count = edges = components = 0
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    with staged_file(path, cancel_check=cancel_check) as stage:
        with Path(stage).open("wb") as output:
            output.write(b"# MADI3D normalized SWC; XYZ and radius units: um\n")
            for line in source:
                check_cancelled(cancel_check)
                parsed = _swc_record(line)
                if parsed is None:
                    if line.lstrip().startswith(b"#"):
                        output.write(b"# Source comment: " + line.strip() + b"\n")
                else:
                    node_id, (node_type, x, y, z, radius, parent_id) = parsed
                    point = (x * scale, y * scale, z * scale)
                    radius *= scale
                    if not all(math.isfinite(value) for value in (*point, radius)):
                        raise ValueError("Converted SWC coordinates and radii must be finite.")
                    count += 1
                    edges += parent_id != -1
                    components += parent_id == -1
                    _limits_check(count, edges, limits)
                    low = np.minimum(low, point)
                    high = np.maximum(high, point)
                    row = (f"{node_id} {node_type} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g} "
                           f"{radius:.17g} {parent_id}\n")
                    output.write(row.encode("ascii"))
                _artifact_limit_check(output.tell(), limits)
        if count == 0:
            raise GeometryUnavailable("Skeleton vertices are unavailable.")
        summary = _summary(count, edges, low, high, components=components)
        verify_artifact(stage, "swc", summary, limits, cancel_check=cancel_check)
    return summary


def _verify_swc(stream, vertex_count, edge_count, component_count, summary, cancel_check):
    # Index exact IDs independently of row order. Only IDs are retained; XYZ,
    # radii and comments are streamed. A second pass validates directed parents.
    indices = {}
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    for line in stream:
        check_cancelled(cancel_check)
        if not line.endswith(b"\n"):
            raise ValueError("SWC contains an incomplete record.")
        parsed = _swc_record(line)
        if parsed is None:
            continue
        node_id, record = parsed
        if node_id in indices:
            raise ValueError("SWC contains duplicate node IDs.")
        indices[node_id] = len(indices)
        if len(indices) > vertex_count:
            raise ValueError("SWC vertex count does not match the summary.")
        low = np.minimum(low, record[1:4])
        high = np.maximum(high, record[1:4])
    if len(indices) != vertex_count:
        raise ValueError("SWC vertex count does not match the summary.")
    parent = np.arange(vertex_count + 1, dtype=np.int64)
    rank = np.zeros(vertex_count + 1, dtype=np.uint8)
    found_edges = 0
    components = vertex_count
    stream.seek(0)
    for line in stream:
        check_cancelled(cancel_check)
        parsed = _swc_record(line)
        if parsed is None:
            continue
        node_id, record = parsed
        swc_parent = record[5]
        if swc_parent == -1:
            pass
        elif swc_parent not in indices:
            raise ValueError("SWC contains an invalid parent ID.")
        else:
            root, other = _find_root(parent, indices[node_id]), _find_root(parent, indices[swc_parent])
            if root == other:
                raise ValueError("SWC parent links contain a cycle or duplicate edge.")
            if rank[root] < rank[other]:
                root, other = other, root
            parent[other] = root
            if rank[root] == rank[other]:
                rank[root] += 1
            found_edges += 1
            components -= 1
    if found_edges != edge_count or components != component_count:
        raise ValueError("SWC edge or component counts do not match the summary.")
    _check_bounds(summary, low, high)


def verify_artifact(path, representation, summary, limits=DEFAULT_LIMITS, *, cancel_check=None):
    """Verify one writer-produced VTP mesh or SWC artifact before publication."""
    check_cancelled(cancel_check)
    target = Path(path)
    try:
        artifact_size = target.stat().st_size
    except OSError as exc:
        raise ValueError("Artifact is unavailable for verification.") from exc
    _artifact_limit_check(artifact_size, limits)

    vertex_count = _summary_count(summary, "vertices")
    primitive_count = _summary_count(summary, "primitives")
    _limits_check(vertex_count, primitive_count, limits)
    if representation == "mesh":
        if vertex_count == 0 or primitive_count == 0:
            raise ValueError("VTP artifact cannot be empty.")
        check_cancelled(cancel_check)
        polydata = _read_vtp(target, cancel_check=cancel_check)
        if (polydata.GetNumberOfPoints() != vertex_count
                or polydata.GetNumberOfPolys() != primitive_count
                or polydata.GetNumberOfVerts() or polydata.GetNumberOfLines()
                or polydata.GetNumberOfStrips()):
            raise ValueError("VTP point or triangle counts do not match the summary.")
        points = vtk_to_numpy(polydata.GetPoints().GetData())
        for start in range(0, vertex_count, _CHUNK_ROWS):
            check_cancelled(cancel_check)
            block = points[start:start + _CHUNK_ROWS]
            if block.shape != (len(block), 3) or not np.isfinite(block).all():
                raise ValueError("VTP contains invalid or non-finite vertex coordinates.")
        cells = polydata.GetPolys()
        offsets = vtk_to_numpy(cells.GetOffsetsArray())
        connectivity = vtk_to_numpy(cells.GetConnectivityArray())
        if len(offsets) != primitive_count + 1 or len(connectivity) != primitive_count * 3:
            raise ValueError("VTP contains invalid triangle topology.")
        for start in range(0, primitive_count, _CHUNK_ROWS):
            check_cancelled(cancel_check)
            stop = min(start + _CHUNK_ROWS, primitive_count)
            if not (np.diff(offsets[start:stop + 1]) == 3).all():
                raise ValueError("VTP contains invalid triangle topology.")
            block = connectivity[start * 3:stop * 3]
            if (block < 0).any() or (block >= vertex_count).any():
                raise ValueError("VTP contains invalid triangle topology.")
        low = points.min(axis=0)
        high = points.max(axis=0)
        _check_bounds(summary, low, high)
    elif representation == "swc":
        component_count = _summary_count(summary, "components")
        if vertex_count == 0 or component_count == 0 or component_count > vertex_count:
            raise ValueError("SWC summary has invalid vertex or component counts.")
        with target.open("rb") as stream:
            _verify_swc(stream, vertex_count, primitive_count, component_count, summary, cancel_check)
    else:
        raise ValueError("Unsupported geometry representation for artifact verification.")
    return None
