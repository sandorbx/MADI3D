"""GUI-independent SWC skeleton model and topology operations."""
from __future__ import annotations

from madi3d_app.io_utils import check_cancelled

import hashlib
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, TypeAlias


SwcNode: TypeAlias = tuple[int, float, float, float, float, int]
SwcNodeMap: TypeAlias = Mapping[int, SwcNode]


def _swc_integer(token: str) -> int:
    """Keep integer tokens exact; retain the editor's historical float forms."""
    try:
        return int(token)
    except ValueError:
        return int(float(token))


def _parse_swc_lines(lines, *, cancel_check=None) -> dict[int, SwcNode]:
    nodes: dict[int, SwcNode] = {}
    for line in lines:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("SWC reading cancelled.")
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            node_id = _swc_integer(parts[0])
            node_type = _swc_integer(parts[1])
            x, y, z, radius = (float(parts[index]) for index in range(2, 6))
            parent_id = _swc_integer(parts[6])
        except Exception:
            continue
        nodes[node_id] = (node_type, x, y, z, radius, parent_id)
    return nodes


def parse_swc(swc_filename, *, cancel_check=None) -> dict[int, SwcNode]:
    """Parse valid seven-column SWC records, ignoring comments and bad rows."""
    with Path(swc_filename).open("r", encoding="utf-8", errors="replace") as stream:
        return _parse_swc_lines(stream, cancel_check=cancel_check)


def parse_swc_with_sha256(swc_filename, *, cancel_check=None) -> tuple[dict[int, SwcNode], str]:
    """Parse SWC records and hash the exact source bytes consumed by the parser."""
    digest = hashlib.sha256()

    def decoded_lines(stream):
        for raw_line in stream:
            digest.update(raw_line)
            yield raw_line.decode("utf-8", errors="replace")

    with Path(swc_filename).open("rb") as stream:
        nodes = _parse_swc_lines(decoded_lines(stream), cancel_check=cancel_check)
    return nodes, digest.hexdigest()


def build_children(nodes: SwcNodeMap, *, cancel_check=None):
    """Return the parent-to-children adjacency map for valid parent links."""
    children = defaultdict(list)
    for node_id, record in nodes.items():
        check_cancelled(cancel_check)
        parent_id = record[5]
        if parent_id >= 0 and parent_id in nodes:
            children[parent_id].append(node_id)
    return children


def collect_subtree(children, root_id: int, *, cancel_check=None) -> set[int]:
    """Collect a root and every descendant reachable through child links."""
    result: set[int] = set()
    pending = deque([root_id])
    while pending:
        check_cancelled(cancel_check)
        node_id = pending.popleft()
        if node_id in result:
            continue
        result.add(node_id)
        pending.extend(children.get(node_id, ()))
    return result


def split_at_node(
    nodes: SwcNodeMap, split_id: int, *, cancel_check=None) -> tuple[dict[int, SwcNode], dict[int, SwcNode]]:
    """Detach the subtree rooted at a non-root node without changing node IDs."""
    if split_id not in nodes:
        raise ValueError(f"Node {split_id} does not exist.")
    if nodes[split_id][5] < 0:
        raise ValueError(
            "The selected split node is a root node. "
            "Choose a non-root node or click on a branch segment."
        )

    branch_ids = collect_subtree(build_children(nodes), split_id)
    main_ids = set(nodes) - branch_ids
    if not main_ids:
        raise ValueError("The selected split node would leave no main tree behind.")

    main_nodes = {node_id: tuple(nodes[node_id]) for node_id in main_ids}
    branch_nodes: dict[int, SwcNode] = {}
    for node_id in branch_ids:
        check_cancelled(cancel_check)
        node_type, x, y, z, radius, parent_id = nodes[node_id]
        branch_nodes[node_id] = (
            node_type, x, y, z, radius, -1 if node_id == split_id else parent_id
        )
    return main_nodes, branch_nodes


def point_segment_distance(point, start, end):
    """Return distance, clamped parameter, and closest point on a segment."""
    px, py, pz = point
    ax, ay, az = start
    bx, by, bz = end
    abx, aby, abz = bx - ax, by - ay, bz - az
    apx, apy, apz = px - ax, py - ay, pz - az
    denominator = abx * abx + aby * aby + abz * abz
    if denominator <= 1e-18:
        distance = math.sqrt((px - ax) ** 2 + (py - ay) ** 2 + (pz - az) ** 2)
        return distance, 0.0, start

    parameter = (apx * abx + apy * aby + apz * abz) / denominator
    parameter = max(0.0, min(1.0, parameter))
    closest = (
        ax + parameter * abx,
        ay + parameter * aby,
        az + parameter * abz,
    )
    distance = math.sqrt(sum((p - c) ** 2 for p, c in zip(point, closest)))
    return distance, parameter, closest


def nearest_split_node_from_point(nodes: SwcNodeMap, point, *, cancel_check=None):
    """Resolve a picked point to the child of its nearest valid SWC segment."""
    best = None
    for child_id, record in nodes.items():
        check_cancelled(cancel_check)
        parent_id = record[5]
        if parent_id < 0 or parent_id not in nodes:
            continue
        parent = nodes[parent_id]
        start = (parent[1], parent[2], parent[3])
        end = (record[1], record[2], record[3])
        distance, parameter, closest = point_segment_distance(point, start, end)
        if best is None or distance < best[1]:
            best = (child_id, distance, parent_id, parameter, closest)
    if best is not None:
        return best

    nearest_id = None
    nearest_distance = float("inf")
    for node_id, record in nodes.items():
        check_cancelled(cancel_check)
        if record[5] < 0:
            continue
        distance = math.sqrt(
            (record[1] - point[0]) ** 2
            + (record[2] - point[1]) ** 2
            + (record[3] - point[2]) ** 2
        )
        if distance < nearest_distance:
            nearest_id, nearest_distance = node_id, distance
    if nearest_id is None:
        return None, float("nan"), None, 0.0, None
    record = nodes[nearest_id]
    return nearest_id, nearest_distance, record[5], 1.0, tuple(record[1:4])


def reindex_nodes_for_write(nodes: SwcNodeMap, *, cancel_check=None) -> dict[int, SwcNode]:
    """Return a deterministic, contiguous, parent-preserving node map."""
    if not nodes:
        return {}
    children = build_children(nodes)
    roots = sorted(
        node_id
        for node_id, record in nodes.items()
        if record[5] < 0 or record[5] not in nodes
    )
    order: list[int] = []
    visited: set[int] = set()
    for root_id in roots:
        check_cancelled(cancel_check)
        pending = deque([root_id])
        while pending:
            check_cancelled(cancel_check)
            node_id = pending.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            order.append(node_id)
            pending.extend(sorted(children.get(node_id, ())))
    order.extend(node_id for node_id in sorted(nodes) if node_id not in visited)

    id_map = {old_id: new_id for new_id, old_id in enumerate(order, start=1)}
    result: dict[int, SwcNode] = {}
    for old_id in order:
        check_cancelled(cancel_check)
        node_type, x, y, z, radius, old_parent = nodes[old_id]
        result[id_map[old_id]] = (
            node_type, x, y, z, radius, id_map.get(old_parent, -1)
        )
    return result


def merge_pieces(pieces: Iterable[tuple[object, SwcNodeMap]], *, cancel_check=None) -> dict[int, SwcNode]:
    """Merge independent SWC pieces with fresh IDs and independent roots."""
    merged: dict[int, SwcNode] = {}
    next_id = 1
    for _source_path, nodes in pieces:
        check_cancelled(cancel_check)
        if not nodes:
            continue
        local_ids = {}
        for old_id in sorted(nodes):
            check_cancelled(cancel_check)
            local_ids[old_id] = next_id
            next_id += 1
        for old_id in sorted(nodes):
            check_cancelled(cancel_check)
            node_type, x, y, z, radius, old_parent = nodes[old_id]
            merged[local_ids[old_id]] = (
                node_type, x, y, z, radius, local_ids.get(old_parent, -1)
            )
    return merged


def connected_components(nodes: SwcNodeMap, *, cancel_check=None) -> list[set[int]]:
    """Return undirected components without assuming a single SWC root."""
    adjacency = defaultdict(set)
    for node_id, record in nodes.items():
        check_cancelled(cancel_check)
        parent_id = record[5]
        adjacency[node_id]
        if parent_id >= 0 and parent_id in nodes:
            adjacency[node_id].add(parent_id)
            adjacency[parent_id].add(node_id)

    remaining = set(nodes)
    components = []
    while remaining:
        check_cancelled(cancel_check)
        component: set[int] = set()
        pending = deque([next(iter(remaining))])
        while pending:
            check_cancelled(cancel_check)
            node_id = pending.popleft()
            if node_id in component:
                continue
            component.add(node_id)
            pending.extend(
                neighbor for neighbor in adjacency[node_id] if neighbor not in component
            )
        components.append(component)
        remaining.difference_update(component)
    return components


def node_to_component_map(nodes: SwcNodeMap, *, cancel_check=None) -> dict[int, frozenset[int]]:
    result = {}
    for component in connected_components(nodes, cancel_check=cancel_check):
        check_cancelled(cancel_check)
        frozen = frozenset(component)
        for node_id in component:
            check_cancelled(cancel_check)
            result[node_id] = frozen
    return result


def extract_components(nodes: SwcNodeMap, touched_node_ids, *, cancel_check=None):
    """Partition whole touched graph components, retaining IDs and records verbatim."""
    selected = set()
    for component in connected_components(nodes, cancel_check=cancel_check):
        check_cancelled(cancel_check)
        if component.intersection(touched_node_ids):
            selected.update(component)
    extracted = {node_id: nodes[node_id] for node_id in selected if node_id in nodes}
    remaining = {node_id: record for node_id, record in nodes.items() if node_id not in selected}
    return extracted, remaining


def nearest_node_or_segment_node(nodes: SwcNodeMap, point, *, cancel_check=None):
    """Return the child of the nearest edge, falling back to the nearest node."""
    nearest_id = None
    nearest_distance = float("inf")
    for child_id, record in nodes.items():
        check_cancelled(cancel_check)
        parent_id = record[5]
        if parent_id < 0 or parent_id not in nodes:
            continue
        parent = nodes[parent_id]
        distance, _parameter, _closest = point_segment_distance(
            point, (parent[1], parent[2], parent[3]), tuple(record[1:4])
        )
        if distance < nearest_distance:
            nearest_id, nearest_distance = child_id, distance
    if nearest_id is not None:
        return nearest_id, nearest_distance

    for node_id, record in nodes.items():
        check_cancelled(cancel_check)
        distance = math.sqrt(
            (record[1] - point[0]) ** 2
            + (record[2] - point[1]) ** 2
            + (record[3] - point[2]) ** 2
        )
        if distance < nearest_distance:
            nearest_id, nearest_distance = node_id, distance
    return nearest_id, nearest_distance


@dataclass(frozen=True)
class SwcSkeleton:
    """Immutable topology snapshot used by editing and GUI controllers."""

    nodes: SwcNodeMap

    def __post_init__(self):
        copied = {int(node_id): tuple(record) for node_id, record in self.nodes.items()}
        object.__setattr__(self, "nodes", MappingProxyType(copied))

    @classmethod
    def from_file(cls, swc_filename):
        return cls(parse_swc(swc_filename))

    @classmethod
    def merge(cls, pieces: Iterable[tuple[object, SwcNodeMap]]):
        return cls(merge_pieces(pieces))

    def split(self, split_id: int):
        main, branch = split_at_node(self.nodes, split_id)
        return type(self)(main), type(self)(branch)

    def reindexed(self):
        return type(self)(reindex_nodes_for_write(self.nodes))

    def components(self):
        return connected_components(self.nodes)

    def nearest_split_node(self, point):
        return nearest_split_node_from_point(self.nodes, point)
