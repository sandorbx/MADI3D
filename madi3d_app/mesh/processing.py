"""Detached mesh topology algorithms; no Qt objects or renderer access."""
import vtk
import numpy as np
from dataclasses import dataclass, field
from vtk.util import numpy_support
from madi3d_app.io_utils import check_cancelled, check_resources

SURFACE_SMOOTHING_ITERATIONS = 20
SURFACE_SMOOTHING_PASS_BAND = 0.1


def update_algorithm(algorithm, *, cancel_check=None):
    observer = algorithm.AddObserver(vtk.vtkCommand.ProgressEvent,
        lambda obj, _event: obj.SetAbortExecute(True) if cancel_check and cancel_check() else None)
    try:
        check_cancelled(cancel_check)
        algorithm.Update()
        check_cancelled(cancel_check)
    finally:
        algorithm.RemoveObserver(observer)


def _reject_ply_topology_edit(polydata):
    from .ply import is_ply_polydata
    if polydata is not None and is_ply_polydata(polydata):
        raise ValueError("This geometry edit cannot preserve PLY source attributes and topology. Export the unchanged PLY or use a non-PLY derivative explicitly.")


def transformed_polydata(source, matrix, *, cancel_check=None, memory_reserved=False):
    from .ply import is_ply_polydata, transformed_ply
    if is_ply_polydata(source):
        return transformed_ply(
            source, matrix, cancel_check=cancel_check, memory_reserved=memory_reserved
        )
    if not memory_reserved:
        check_resources(memory_bytes=source.GetActualMemorySize() * 1024)
    check_cancelled(cancel_check)
    poly = vtk.vtkPolyData()
    poly.DeepCopy(source)
    point_normals = poly.GetPointData().GetNormals()
    cached_normals = point_normals is not None and point_normals.GetNumberOfTuples() == poly.GetNumberOfPoints()
    if cached_normals and matrix.IsIdentity():
        return poly
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    geometry = vtk.vtkTransformPolyDataFilter()
    geometry.SetTransform(transform)
    geometry.SetInputData(poly)
    update_algorithm(geometry, cancel_check=cancel_check)
    # The transform filter already applies the inverse-transpose to cached
    # point normals. Rebuilding SWC topology/normals here wastes the loaded cache.
    if cached_normals:
        return geometry.GetOutput()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(geometry.GetOutput())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.ConsistencyOn()
    normals.SplittingOff()
    update_algorithm(normals, cancel_check=cancel_check)
    return normals.GetOutput()


def reformatted_polydata(source, points, matrix, *, cancel_check=None):
    """Realize captured Reformat points with their detached topology and arrays."""
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    if source.GetNumberOfPoints() != len(points):
        raise ValueError("Reformat coordinates do not match the captured mesh.")
    matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
    from .ply import is_ply_polydata, transformed_ply
    if is_ply_polydata(source):
        # PLY owns corner UV lists and component normal properties, which must
        # follow the affine as well as the VTK points and connectivity.
        return transformed_ply(source, matrix, cancel_check=cancel_check)
    check_resources(memory_bytes=source.GetActualMemorySize() * 1024 + points.nbytes)
    check_cancelled(cancel_check)
    poly = vtk.vtkPolyData()
    poly.DeepCopy(source)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points, deep=True))
    poly.SetPoints(vtk_points)
    if np.linalg.det(matrix[:3, :3]) < 0:
        reverse = vtk.vtkReverseSense()
        reverse.SetInputData(poly)
        reverse.ReverseCellsOn()
        reverse.ReverseNormalsOff()
        update_algorithm(reverse, cancel_check=cancel_check)
        poly.DeepCopy(reverse.GetOutput())
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.SplittingOff()
    normals.ConsistencyOff()
    normals.AutoOrientNormalsOff()
    normals.ComputePointNormalsOn()
    normals.SetComputeCellNormals(source.GetCellData().GetNormals() is not None)
    update_algorithm(normals, cancel_check=cancel_check)
    result = vtk.vtkPolyData()
    result.DeepCopy(normals.GetOutput())
    return result


def smooth_surface_geometry(polydata, *, cancel_check=None):
    """Return detached, windowed-sinc smoothed geometry without changing topology."""
    check_cancelled(cancel_check)
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        result = vtk.vtkPolyData()
        if polydata is not None:
            result.DeepCopy(polydata)
        return result
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(polydata)
    smoother.SetNumberOfIterations(SURFACE_SMOOTHING_ITERATIONS)
    smoother.SetPassBand(SURFACE_SMOOTHING_PASS_BAND)
    smoother.NormalizeCoordinatesOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOff()
    smoother.NonManifoldSmoothingOff()
    update_algorithm(smoother, cancel_check=cancel_check)
    result = vtk.vtkPolyData()
    result.DeepCopy(smoother.GetOutput())
    return result


def volume_surface(
    image, matrix, iso_value, *, smooth_normals=False, reverse_cells=False,
    cancel_check=None, memory_reserved=False, cutoff_range=None,
    surface_smoothing=False,
):
    """Extract the displayed contour without updating the renderer's pipeline."""
    check_cancelled(cancel_check)
    if not np.isfinite(iso_value):
        raise ValueError("The surface iso value must be finite.")
    x, y, z = image.GetDimensions()
    scalars = image.GetPointData().GetScalars()
    if min(x, y, z) < 2 or scalars is None or scalars.GetNumberOfComponents() != 1:
        raise ValueError("Surface extraction requires a 3D scalar volume with at least two samples per axis.")
    if not memory_reserved:
        check_resources(memory_bytes=image.GetActualMemorySize() * 1024)
    owned = vtk.vtkImageData()
    owned.DeepCopy(image)
    mask_bytes = 0
    padded_mask_bytes = 0
    if cutoff_range is not None:
        from madi3d_app.volume.rendering import surface_cutoff_mask_peak_bytes
        mask_bytes = surface_cutoff_mask_peak_bytes((x, y, z))
        padded_mask_bytes = (x + 2) * (y + 2) * (z + 2)
    pixels = numpy_support.vtk_to_numpy(owned.GetPointData().GetScalars()).reshape(z, y, x)
    for plane in pixels:
        check_cancelled(cancel_check)
        if not np.isfinite(plane).all():
            raise ValueError("Surface extraction requires finite voxel values.")
    del pixels
    if not memory_reserved:
        # FlyingEdges allocates XCases and six vtkIdType values per x-row.
        # Output topology and native filter overhead remain unknown.
        id_bytes = vtk.vtkIdTypeArray().GetDataTypeSize()
        if cutoff_range is None:
            contour_bytes = (x - 1) * y * z + y * z * 6 * id_bytes
        else:
            # The shared contour helper pads one voxel on every side before
            # FlyingEdges, so its row/case arrays follow the padded dimensions.
            contour_bytes = (
                (x + 1) * (y + 2) * (z + 2)
                + (y + 2) * (z + 2) * 6 * id_bytes
            )
        known_peak = max(mask_bytes, padded_mask_bytes + contour_bytes)
        check_resources(memory_bytes=known_peak)
    contour = vtk.vtkFlyingEdges3D()
    from madi3d_app.volume.rendering import configure_surface_contour
    threshold = configure_surface_contour(
        contour, owned, iso_value, cutoff_range=cutoff_range
    )
    contour.ComputeNormalsOn()
    contour.ComputeScalarsOff()
    contour.ComputeGradientsOff()
    update_algorithm(contour, cancel_check=cancel_check)
    poly = contour.GetOutput()
    from madi3d_app.volume.rendering import normalize_surface_smoothing
    effective_surface_smoothing = (
        normalize_surface_smoothing(surface_smoothing)
        and cutoff_range is not None
    )
    # Keep only the output, releasing the private volume and contour pipeline
    # before allocating smoothing, reversed connectivity or transformed points.
    del contour, threshold, owned
    if effective_surface_smoothing:
        poly = smooth_surface_geometry(poly, cancel_check=cancel_check)
    if smooth_normals or reverse_cells or effective_surface_smoothing:
        from .cache import compute_smooth_point_normals
        poly = compute_smooth_point_normals(poly, cancel_check=cancel_check)
    if reverse_cells:
        reverse = vtk.vtkReverseSense()
        reverse.SetInputData(poly)
        reverse.ReverseCellsOn()
        reverse.ReverseNormalsOff()
        update_algorithm(reverse, cancel_check=cancel_check)
        poly = reverse.GetOutput()
        del reverse
    # vtkTransformPolyDataFilter shares topology and copies point arrays.
    if not memory_reserved:
        check_resources(memory_bytes=poly.GetNumberOfPoints() * 6 * 4)
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    transformed = vtk.vtkTransformPolyDataFilter()
    transformed.SetTransform(transform)
    transformed.SetInputData(poly)
    update_algorithm(transformed, cancel_check=cancel_check)
    return transformed.GetOutput()

def build_cell_adjacency_maps(polydata, *, cancel_check=None):
    """
        Builds:
          - cell_to_points: dict[cell_id] -> tuple(point_ids)
          - point_to_cells: dict[point_id] -> list(cell_ids)

        This lets us compute connected components manually and reliably.
        """
    cell_to_points = {}
    point_to_cells = {}
    n_cells = polydata.GetNumberOfCells()
    for cid in range(n_cells):
        check_cancelled(cancel_check)
        cell = polydata.GetCell(cid)
        if cell is None:
            continue
        pts = cell.GetPointIds()
        point_ids = tuple((pts.GetId(i) for i in range(pts.GetNumberOfIds())))
        cell_to_points[cid] = point_ids
        for pid in point_ids:
            check_cancelled(cancel_check)
            point_to_cells.setdefault(pid, []).append(cid)
    return (cell_to_points, point_to_cells)

def collect_connected_cells_from_seed(seed_cell_id, allowed_cells, cell_to_points, point_to_cells, *, cancel_check=None):
    """
        Collects the connected cell component containing seed_cell_id.
        Connectivity is based on shared mesh points.
        """
    if seed_cell_id not in allowed_cells:
        return set()
    visited = set()
    stack = [seed_cell_id]
    while stack:
        check_cancelled(cancel_check)
        cid = stack.pop()
        if cid in visited:
            continue
        if cid not in allowed_cells:
            continue
        visited.add(cid)
        for pid in cell_to_points.get(cid, ()):
            check_cancelled(cancel_check)
            for neighbor_cid in point_to_cells.get(pid, ()):
                check_cancelled(cancel_check)
                if neighbor_cid not in visited and neighbor_cid in allowed_cells:
                    stack.append(neighbor_cid)
    return visited

def connected_components_from_cells(cell_ids, cell_to_points, point_to_cells, *, cancel_check=None):
    """
        Splits a set of cell IDs into connected components.
        """
    remaining = set(cell_ids)
    components = []
    while remaining:
        check_cancelled(cancel_check)
        seed = next(iter(remaining))
        comp = collect_connected_cells_from_seed(
            seed_cell_id=seed,
            allowed_cells=remaining,
            cell_to_points=cell_to_points,
            point_to_cells=point_to_cells,
            cancel_check=cancel_check,
        )
        if not comp:
            remaining.remove(seed)
            continue
        components.append(comp)
        remaining.difference_update(comp)
    check_cancelled(cancel_check)
    return components

def cell_centroid(polydata, cell_id, *, cancel_check=None):
    """
        Returns the centroid of a cell.
        """
    cell = polydata.GetCell(cell_id)
    pts = cell.GetPointIds()
    n = pts.GetNumberOfIds()
    if n <= 0:
        return None
    sx = sy = sz = 0.0
    for i in range(n):
        check_cancelled(cancel_check)
        p = polydata.GetPoint(pts.GetId(i))
        sx += p[0]
        sy += p[1]
        sz += p[2]
    return (sx / n, sy / n, sz / n)

def extract_cells_to_polydata(source_polydata, cell_ids, *, cancel_check=None):
    """
        Builds a new vtkPolyData from selected source cells.

        Copies geometry and point-data arrays where possible.
        Then cleans the result and recomputes normals.
        """
    _reject_ply_topology_edit(source_polydata)
    new_points = vtk.vtkPoints()
    new_polys = vtk.vtkCellArray()
    old_to_new_pid = {}
    source_point_data = source_polydata.GetPointData()
    new_point_data = vtk.vtkPointData()
    new_point_data.CopyAllocate(source_point_data)
    sorted_cell_ids = sorted(cell_ids)
    for cid in sorted_cell_ids:
        check_cancelled(cancel_check)
        cell = source_polydata.GetCell(cid)
        if cell is None:
            continue
        old_pts = cell.GetPointIds()
        new_ids = vtk.vtkIdList()
        for i in range(old_pts.GetNumberOfIds()):
            check_cancelled(cancel_check)
            old_pid = old_pts.GetId(i)
            if old_pid not in old_to_new_pid:
                new_pid = new_points.InsertNextPoint(source_polydata.GetPoint(old_pid))
                old_to_new_pid[old_pid] = new_pid
                new_point_data.CopyData(source_point_data, old_pid, new_pid)
            new_ids.InsertNextId(old_to_new_pid[old_pid])
        if new_ids.GetNumberOfIds() >= 3:
            new_polys.InsertNextCell(new_ids)
    out = vtk.vtkPolyData()
    out.SetPoints(new_points)
    out.SetPolys(new_polys)
    out.GetPointData().ShallowCopy(new_point_data)
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(out)
    update_algorithm(cleaner, cancel_check=cancel_check)
    cleaned = cleaner.GetOutput()
    cleaned.GetPointData().SetNormals(None)
    cleaned.GetCellData().SetNormals(None)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(cleaned)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.ConsistencyOn()
    normals.SplittingOff()
    update_algorithm(normals, cancel_check=cancel_check)
    final_poly = vtk.vtkPolyData()
    final_poly.DeepCopy(normals.GetOutput())
    return final_poly

def filter_polydata_components_by_cell_count(polydata, min_cells, *, cancel_check=None):
    """Keep point-connected regions using VTK's C++ connectivity implementation."""
    _reject_ply_topology_edit(polydata)
    if polydata is None or polydata.GetNumberOfCells() <= 0:
        return (None, 0, 0, 0)
    min_cells = max(1, int(min_cells))
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(polydata)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    update_algorithm(connectivity, cancel_check=cancel_check)
    region_sizes_vtk = connectivity.GetRegionSizes()
    if region_sizes_vtk is None or region_sizes_vtk.GetNumberOfTuples() <= 0:
        return (polydata, 1, 0, 0)
    region_sizes = numpy_support.vtk_to_numpy(region_sizes_vtk).astype(np.int64, copy=False)
    n_components = int(region_sizes.size)
    keep_regions = np.flatnonzero(region_sizes >= min_cells)
    removed_mask = region_sizes < min_cells
    removed_components = int(np.count_nonzero(removed_mask))
    removed_cells = int(region_sizes[removed_mask].sum()) if removed_components else 0
    if keep_regions.size == 0:
        empty = vtk.vtkPolyData()
        empty.SetPoints(vtk.vtkPoints())
        empty.SetPolys(vtk.vtkCellArray())
        return (empty, n_components, removed_components, removed_cells)
    if keep_regions.size == n_components:
        return (polydata, n_components, 0, 0)
    extract = vtk.vtkPolyDataConnectivityFilter()
    extract.SetInputData(polydata)
    extract.SetExtractionModeToSpecifiedRegions()
    for region_id in keep_regions:
        check_cancelled(cancel_check)
        extract.AddSpecifiedRegion(int(region_id))
    update_algorithm(extract, cancel_check=cancel_check)
    cleaned = vtk.vtkCleanPolyData()
    cleaned.SetInputConnection(extract.GetOutputPort())
    update_algorithm(cleaned, cancel_check=cancel_check)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(cleaned.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.ConsistencyOn()
    normals.SplittingOff()
    update_algorithm(normals, cancel_check=cancel_check)
    out = vtk.vtkPolyData()
    out.DeepCopy(normals.GetOutput())
    return (out, n_components, removed_components, removed_cells)


def painted_geometry(polydata, painted_ids, *, cancel_check=None):
    _reject_ply_topology_edit(polydata)
    cell_set = set()
    cell_ids = vtk.vtkIdList()
    for pid in painted_ids:
        check_cancelled(cancel_check)
        polydata.GetPointCells(pid, cell_ids)
        for i in range(cell_ids.GetNumberOfIds()):
            check_cancelled(cancel_check)
            cell_set.add(cell_ids.GetId(i))
    return _brush_geometry_subset(polydata, cell_set, cancel_check)


def _brush_geometry_subset(source, cell_ids, cancel_check):
    if not cell_ids:
        return None
    points = vtk.vtkPoints()
    points.SetDataType(source.GetPoints().GetDataType())
    result = vtk.vtkPolyData()
    result.SetPoints(points)
    result.Allocate(len(cell_ids))
    point_ids = {}
    for cell_id in sorted(cell_ids):
        check_cancelled(cancel_check)
        cell = source.GetCell(cell_id)
        subset_cell = vtk.vtkIdList()
        for index in range(cell.GetNumberOfPoints()):
            check_cancelled(cancel_check)
            point_id = cell.GetPointId(index)
            if point_id not in point_ids:
                point_ids[point_id] = points.InsertNextPoint(source.GetPoint(point_id))
            subset_cell.InsertNextId(point_ids[point_id])
        result.InsertNextCell(cell.GetCellType(), subset_cell)
    return result


def unpainted_geometry(pd, painted_ids, *, cancel_check=None):
    _reject_ply_topology_edit(pd)
    n_pts = int(pd.GetNumberOfPoints())
    painted_mask = np.zeros(n_pts, dtype=bool)
    painted_mask[painted_ids[(painted_ids >= 0) & (painted_ids < n_pts)]] = True
    unp_ids = np.flatnonzero(~painted_mask)
    if unp_ids.size == 0:
        return None
    cell_ids = vtk.vtkIdList()
    cells_to_keep = set()
    for pid in unp_ids:
        check_cancelled(cancel_check)
        pd.GetPointCells(pid, cell_ids)
        for i in range(cell_ids.GetNumberOfIds()):
            check_cancelled(cancel_check)
            cid = cell_ids.GetId(i)
            cell = pd.GetCell(cid)
            pts = cell.GetPointIds()
            all_unp = True
            for j in range(pts.GetNumberOfIds()):
                check_cancelled(cancel_check)
                p2 = pts.GetId(j)
                if painted_mask[p2]:
                    all_unp = False
                    break
            if all_unp:
                cells_to_keep.add(cid)
    return _brush_geometry_subset(pd, cells_to_keep, cancel_check)


def prepare_mask(poly, *, cancel_check=None):
    """Prepare a closed manifold mask without filling holes or repairing intersections.

    All disconnected components are retained. This validates edge topology,
    not self-intersections or intersections between components.
    """
    check_cancelled(cancel_check)
    if poly.GetNumberOfPoints() == 0 or poly.GetNumberOfCells() == 0:
        raise ValueError("The mask is empty. Select a closed mesh surface.")
    points = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())
    if not np.isfinite(points).all():
        raise ValueError("The mask contains non-finite coordinates.")
    prepared = fix_faces(poly, cancel_check=cancel_check)
    if prepared.GetNumberOfPolys() == 0:
        raise ValueError("The mask contains no surface faces.")
    vertices = numpy_support.vtk_to_numpy(prepared.GetPoints().GetData())
    triangles = numpy_support.vtk_to_numpy(prepared.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    for start in range(0, len(triangles), 65536):
        check_cancelled(cancel_check)
        chunk = vertices[triangles[start:start + 65536]].astype(np.float64)
        areas = np.cross(chunk[:, 1] - chunk[:, 0], chunk[:, 2] - chunk[:, 0])
        if not np.isfinite(areas).all() or np.any(~np.any(areas, axis=1)):
            raise ValueError("The mask contains degenerate faces. Repair the mesh before masking.")
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(prepared)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    update_algorithm(edges, cancel_check=cancel_check)
    if edges.GetOutput().GetNumberOfCells():
        raise ValueError(
            "The mask has open boundaries or non-manifold edges. "
            "Repair the mesh in a mesh editor before masking; holes are not filled automatically."
        )
    return prepared


def fix_faces(poly, *, cancel_check=None):
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.PassLinesOff()
    tri.PassVertsOff()
    update_algorithm(tri, cancel_check=cancel_check)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.PointMergingOn()
    clean.ConvertLinesToPointsOff()
    clean.ConvertPolysToLinesOff()
    clean.ConvertStripsToPolysOff()
    update_algorithm(clean, cancel_check=cancel_check)
    norms = vtk.vtkPolyDataNormals()
    norms.SetInputConnection(clean.GetOutputPort())
    norms.SplittingOff()
    norms.ConsistencyOn()
    norms.AutoOrientNormalsOn()
    norms.ComputePointNormalsOn()
    norms.ComputeCellNormalsOff()
    update_algorithm(norms, cancel_check=cancel_check)
    return norms.GetOutput()


def cells_inside_sphere(polydata, candidate_cells, center, radius, point_to_cells=None, actor_matrix=None, world_center=None, world_radius=None, *, cancel_check=None):
    """Return cutter cells using a local broad phase and exact world-space tests."""
    r = float(radius)
    cells_to_remove = set()
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()
    ids = vtk.vtkIdList()
    locator.FindPointsWithinRadius(r, center, ids)
    exact_world = actor_matrix is not None and world_center is not None and (world_radius is not None)
    if exact_world:
        matrix = actor_matrix
        wc = np.asarray(world_center, dtype=float)
        wr2 = float(world_radius) * float(world_radius) * (1.0 + 1e-10)
    else:
        matrix = None
        wc = np.asarray(center, dtype=float)
        wr2 = r * r
    if point_to_cells is not None and ids.GetNumberOfIds() > 0:
        broad_ids = np.fromiter((ids.GetId(i) for i in range(ids.GetNumberOfIds())), dtype=np.int64, count=ids.GetNumberOfIds())
        if exact_world:
            points_data = polydata.GetPoints().GetData() if polydata.GetPoints() is not None else None
            if points_data is not None:
                local = numpy_support.vtk_to_numpy(points_data)[broad_ids, :3]
                world = local @ matrix[:3, :3].T + matrix[:3, 3]
                delta = world - wc.reshape(1, 3)
                keep = np.einsum('ij,ij->i', delta, delta) <= wr2
                broad_ids = broad_ids[keep]
        for pid in broad_ids:
            for cid in point_to_cells.get(int(pid), []):
                if cid in candidate_cells:
                    cells_to_remove.add(cid)
    for cid in candidate_cells:
        check_cancelled(cancel_check)
        if cid in cells_to_remove:
            continue
        c = cell_centroid(polydata, cid)
        if c is None:
            continue
        if exact_world:
            c_test = np.asarray(c, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]
        else:
            c_test = np.asarray(c, dtype=float)
        d = c_test - wc
        if float(np.dot(d, d)) <= wr2:
            cells_to_remove.add(cid)
    return cells_to_remove


@dataclass
class MeshCutSelection:
    """Cutter outcome; cell IDs always refer to the unchanged input geometry."""

    status: str
    main_cells: set[int] = field(default_factory=set)
    branch_cells: set[int] = field(default_factory=set)
    removed_cells: set[int] = field(default_factory=set)


def select_mesh_cut(polydata, picked_cell_id, center, radius, actor_matrix,
                    world_center, world_radius, *, cancel_check=None):
    """Cut the picked component, retaining unrelated components with the largest piece.

    The local sphere is a conservative broad phase; vertex and centroid distances
    use the explicit world matrix. Equal-sized pieces retain primitive traversal
    order, as in the original editing action.
    """
    all_cells = set(range(polydata.GetNumberOfCells()))
    cell_to_points, point_to_cells = build_cell_adjacency_maps(polydata, cancel_check=cancel_check)
    affected = collect_connected_cells_from_seed(
        picked_cell_id, all_cells, cell_to_points, point_to_cells, cancel_check=cancel_check)
    if not affected:
        return MeshCutSelection("no_component")
    removed = cells_inside_sphere(
        polydata, affected, center, radius, point_to_cells,
        actor_matrix, world_center, world_radius, cancel_check=cancel_check)
    if not removed:
        return MeshCutSelection("no_hit")
    remaining = affected.difference(removed)
    if not remaining:
        return MeshCutSelection("whole_component", removed_cells=removed)
    components = connected_components_from_cells(
        remaining, cell_to_points, point_to_cells, cancel_check=cancel_check)
    if len(components) < 2:
        return MeshCutSelection("not_split", removed_cells=removed)
    components.sort(key=lambda component: len(component), reverse=True)
    main = all_cells.difference(affected)
    main.update(components[0])
    branch = set()
    for component in components[1:]:
        branch.update(component)
    status = "split" if main and branch else "unusable_pieces"
    return MeshCutSelection(status, main, branch, removed)


def decimate_geometry(source, *, feature_angle, maximum_iterations, preserve_topology,
                      error_is_absolute, absolute_error, target_reduction, cancel_check=None):
    """Apply the editing decimator and normals settings to a detached copy."""
    _reject_ply_topology_edit(source)
    check_resources(memory_bytes=source.GetActualMemorySize() * 1024)
    check_cancelled(cancel_check)
    decimated_polydata = vtk.vtkPolyData()
    decimated_polydata.DeepCopy(source)
    # Remove existing normals
    decimated_polydata.GetPointData().RemoveArray("Normals")
    decimated_polydata.GetCellData().RemoveArray("Normals")

    # Set up the decimator
    decimator = vtk.vtkDecimatePro()
    decimator.SetInputData(decimated_polydata)
    decimator.SetFeatureAngle(feature_angle)
    decimator.MaximumIterations = (maximum_iterations)
    if preserve_topology:
        decimator.PreserveTopologyOn()
    else:
        decimator.PreserveTopologyOff()
    if error_is_absolute:
        decimator.SetErrorIsAbsolute(1)
        decimator.SetAbsoluteError(absolute_error)
    else:
        decimator.SetErrorIsAbsolute(0)
    decimator.SetTargetReduction(target_reduction)
    update_algorithm(decimator, cancel_check=cancel_check)

    decimated_output = decimator.GetOutput()

    # Recalculate normals
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(decimated_output)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOff()
    normals_filter.SplittingOff()  # Prevent normals splitting
    update_algorithm(normals_filter, cancel_check=cancel_check)

    final_polydata = normals_filter.GetOutput()

    return final_polydata


def measure_geometry(polydata, *, measure_volume, fill_holes):
    """Return dataset-space mass/area and boundary-edge count with editing defaults."""

    # --- Pre-Processing: Fill Holes if requested ---
    if fill_holes:
        fill_filter = vtk.vtkFillHolesFilter()
        fill_filter.SetInputData(polydata)
        fill_filter.SetHoleSize(1000.0)  # Adjust threshold as needed.
        fill_filter.Update()
        polydata = fill_filter.GetOutput()

    # --- Check Mesh Closure using vtkFeatureEdges ---
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(polydata)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.Update()
    boundary_count = feature_edges.GetOutput().GetNumberOfCells()
    # --- Convert Mesh to a Triangulated (Closed) Surface ---
    # Use a vtkTriangleFilter if polydata is already a vtkPolyData,
    # otherwise fall back to a vtkGeometryFilter.
    if polydata.IsA("vtkPolyData"):
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputData(polydata)
        triangle_filter.SetPassLines(0)  # Do not pass non-surface elements.
        triangle_filter.Update()
        filtered_polydata = triangle_filter.GetOutput()
    else:
        geometry_filter = vtk.vtkGeometryFilter()
        geometry_filter.SetInputData(polydata)
        geometry_filter.Update()
        filtered_polydata = geometry_filter.GetOutput()

    # --- Compute Mesh Properties using vtkMassProperties ---
    if filtered_polydata.GetNumberOfCells() > 0:
        mass_props = vtk.vtkMassProperties()
        mass_props.SetInputData(filtered_polydata)
        mass_props.Update()
        if measure_volume:
            result = mass_props.GetVolume()
        else:
            result = mass_props.GetSurfaceArea()
    else:
        result = 0.0

    return result, boundary_count


def combine_geometry(polydata_inputs, *, compute_normals=False, cancel_check=None):
    """Append dataset geometry without welding or baking actor transforms."""
    append = vtk.vtkAppendPolyData()
    for polydata in polydata_inputs:
        _reject_ply_topology_edit(polydata)
        check_cancelled(cancel_check)
        append.AddInputData(polydata)
    update_algorithm(append, cancel_check=cancel_check)
    combined = append.GetOutput()
    if compute_normals:
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(combined)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.ConsistencyOn()
        normals.SplittingOff()
        update_algorithm(normals, cancel_check=cancel_check)
        combined = normals.GetOutput()
    return combined


def unpainted_geometry_with_normals(polydata, painted_ids, *, cancel_check=None):
    """Extract unpainted cells and apply the existing extraction normals settings."""
    selected = unpainted_geometry(polydata, painted_ids, cancel_check=cancel_check)
    if selected is None:
        return None
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(selected)
    normals.ComputePointNormalsOn()
    normals.ConsistencyOn()
    normals.SplittingOff()
    update_algorithm(normals, cancel_check=cancel_check)
    return normals.GetOutput()


def enclosed_geometry_indices(polydata_inputs, mask_polydata, *, cancel_check=None):
    """Return inputs with any enclosed point, using their existing dataset coordinates."""
    selector = vtk.vtkSelectEnclosedPoints()
    selector.SetSurfaceData(mask_polydata)
    selector.CheckSurfaceOn()
    selected = []
    for index, polydata in enumerate(polydata_inputs):
        selector.SetInputData(polydata)
        update_algorithm(selector, cancel_check=cancel_check)
        array = selector.GetOutput().GetPointData().GetArray("SelectedPoints")
        if array is not None and (numpy_support.vtk_to_numpy(array) > 0).any():
            selected.append(index)
    return selected


def clip_geometry(polydata, mask_polydata, *, extract_mode, cancel_check=None):
    """Apply the existing mask clipping polarity without changing its scientific policy."""
    _reject_ply_topology_edit(polydata)
    distance = vtk.vtkImplicitPolyDataDistance()
    distance.SetInput(mask_polydata)
    clip = vtk.vtkClipPolyData()
    clip.SetInputData(polydata)
    clip.SetClipFunction(distance)
    clip.InsideOutOn() if not extract_mode else clip.InsideOutOff()
    update_algorithm(clip, cancel_check=cancel_check)
    return clip.GetOutput()
