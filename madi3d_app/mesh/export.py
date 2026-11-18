"""Detached geometry serialization and atomic mesh/SWC output."""
from contextlib import contextmanager
import errno
import os

import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
from madi3d_app.tools.swc import reindex_nodes_for_write

from madi3d_app.io_utils import (
    PUBLICATION_MARKER_SUFFIX, check_cancelled, check_resources, staged_file, publish_file_pair,
)


def _write_plain_obj(polydata, filename, cancel_check, export_id=None):
    """Stream untextured geometry in bounded, cancellable Python batches.

    vtkOBJWriter.Write holds the Python GIL throughout its native text loop,
    starving Qt callbacks even when called from a QThread.
    """
    point_data = polydata.GetPointData()
    normals = point_data.GetNormals()
    if normals is not None and normals.GetNumberOfTuples() != polydata.GetNumberOfPoints():
        raise ValueError("Mesh point normals do not match the point count.")
    chunk_size = 4096
    point_count = polydata.GetNumberOfPoints()
    with open(filename, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("# MADI3D mesh\n")
        if export_id:
            stream.write(f"# MADI3D export generation: {export_id}\n")
        for prefix, values in (("v", polydata.GetPoints().GetData()), ("vn", normals)):
            if values is None:
                continue
            array = vtk_to_numpy(values)
            for start in range(0, len(array), chunk_size):
                check_cancelled(cancel_check)
                chunk = array[start:start + chunk_size]
                if chunk.shape[1] != 3 or not np.isfinite(chunk).all():
                    raise ValueError("Mesh coordinates and normals must be finite triples.")
                np.savetxt(stream, chunk, fmt=prefix + " %.17g %.17g %.17g")
        for prefix, cells in (("f", polydata.GetPolys()), ("l", polydata.GetLines())):
            offsets = vtk_to_numpy(cells.GetOffsetsArray())
            indices = vtk_to_numpy(cells.GetConnectivityArray())
            for start in range(0, cells.GetNumberOfCells(), chunk_size):
                check_cancelled(cancel_check)
                end = min(start + chunk_size, cells.GetNumberOfCells())
                chunk_ids = indices[offsets[start]:offsets[end]]
                if np.any(chunk_ids < 0) or np.any(chunk_ids >= point_count):
                    raise ValueError("Mesh connectivity references an invalid point.")
                rows = []
                for cell in range(start, end):
                    ids = indices[offsets[cell]:offsets[cell + 1]]
                    tokens = (str(int(index) + 1) for index in ids)
                    if prefix == "f" and normals is not None:
                        tokens = (index + "//" + index for index in tokens)
                    rows.append(prefix + " " + " ".join(tokens) + "\n")
                stream.writelines(rows)


@contextmanager
def staged_polydata(polydata, filename, *, binary_vtk=False,
                    vtp_data_mode="binary", obj_writer="auto", ply_encoding="binary_little_endian",
                    cancel_check=None, publish=True, export_id=None,
                    memory_reserved=False):
    """Write detached geometry; publish after the caller's checks complete.

    Yield the written stage for cache verification or companion preparation.
    Any error or cancellation before publication removes this stage and leaves
    the previous destination intact. All mesh writers share this boundary.

    ``auto`` streams plain OBJ points, normals, polygons and lines; textured or
    strip geometry still uses VTK. Editing actions that previously used VTK's
    OBJ serialization request ``vtk`` explicitly. VTP is inline binary except
    for the combine action's appended data. Both VTP modes use encoded ZLib
    compression; legacy VTK remains ASCII unless explicitly requested binary.

    No transform, cleaning, triangulation or normals filters are added here.
    Format limitations (including OBJ's omission of vertex cells and
    arbitrary point/cell arrays) are unchanged.
    """
    filename = os.fspath(filename)
    if vtp_data_mode not in ("binary", "appended"):
        raise ValueError(f"Unsupported VTP data mode: {vtp_data_mode}")
    if obj_writer not in ("auto", "vtk"):
        raise ValueError(f"Unsupported OBJ writer: {obj_writer}")
    extension = os.path.splitext(filename)[1].lower()
    plain_obj = (extension == ".obj" and obj_writer == "auto" and polydata.GetPoints() is not None
            and polydata.GetNumberOfStrips() == 0
            and polydata.GetPointData().GetTCoords() is None
            and polydata.GetFieldData().GetAbstractArray("MaterialNames") is None
            and polydata.GetFieldData().GetAbstractArray("MaterialLibraries") is None)
    writer_type = {".obj": vtk.vtkOBJWriter, ".vtp": vtk.vtkXMLPolyDataWriter,
                   ".vtk": vtk.vtkPolyDataWriter}.get(extension)
    if writer_type is None and extension != ".ply":
        raise ValueError(f"Unsupported mesh output: {extension}")
    if export_id:
        owned = vtk.vtkPolyData()
        owned.ShallowCopy(polydata)
        marker = vtk.vtkStringArray()
        marker.SetName("madi3d_export_generation")
        marker.InsertNextValue(export_id)
        owned.GetFieldData().AddArray(marker)
        polydata = owned
    with staged_file(filename, cancel_check=cancel_check, publish=publish) as stage:
        if extension == ".ply":
            from .ply import write_ply
            with open(stage, "wb") as stream:
                write_ply(
                    polydata, stream, encoding=ply_encoding,
                    cancel_check=cancel_check, memory_reserved=memory_reserved,
                )
        elif plain_obj:
            _write_plain_obj(polydata, stage, cancel_check, export_id)
        else:
            writer = writer_type()
            if extension == ".vtp":
                if vtp_data_mode == "binary":
                    writer.SetDataModeToBinary()
                else:
                    writer.SetDataModeToAppended()
                writer.SetCompressor(vtk.vtkZLibDataCompressor())
                writer.SetEncodeAppendedData(True)
            elif extension == ".vtk":
                if binary_vtk:
                    writer.SetFileTypeToBinary()
                else:
                    writer.SetFileTypeToASCII()
            writer.SetInputData(polydata)
            writer.SetFileName(stage)
            errors = []
            error_observer = writer.AddObserver(vtk.vtkCommand.ErrorEvent, lambda *_: errors.append(True))
            observer = writer.AddObserver(vtk.vtkCommand.ProgressEvent,
                lambda obj, _event: obj.SetAbortExecute(True) if cancel_check and cancel_check() else None)
            try:
                result = writer.Write()
                if not result or writer.GetErrorCode() or errors:
                    raise OSError(f"Could not write geometry: {filename}")
            finally:
                writer.RemoveObserver(observer)
                writer.RemoveObserver(error_observer)
                writer.SetFileName(filename)
            if extension == ".obj" and export_id:
                with open(stage, "a", encoding="utf-8") as stream:
                    stream.write(f"\n# MADI3D export generation: {export_id}\n")
        yield stage


def write_polydata(polydata, filename, *, binary_vtk=False,
                   vtp_data_mode="binary", obj_writer="auto", cancel_check=None):
    """Publish geometry without additional pre-publication checks or companions."""
    with staged_polydata(polydata, filename, binary_vtk=binary_vtk,
                         vtp_data_mode=vtp_data_mode, obj_writer=obj_writer,
                         cancel_check=cancel_check):
        pass
    return True


def mesh_transform_allocation_bytes(source, matrix):
    """Known active and retained backing added by the baked processing route.

    Native filter scratch, data-dependent indexing and Python face lists remain
    unquantified. The copy uses VTK's reported storage; transformed numeric
    arrays use their actual tuple counts and scalar widths.
    """
    from .ply import ply_schema

    copied = int(source.GetActualMemorySize()) * 1024
    points = source.GetPoints()
    if points is None:
        return copied, copied
    coordinates = points.GetData()
    coordinate_bytes = coordinates.GetDataSize() * coordinates.GetDataTypeSize()
    schema = ply_schema(source)
    normals = source.GetPointData().GetNormals()
    cached_normals = normals is not None and normals.GetNumberOfTuples() == source.GetNumberOfPoints()
    if matrix.IsIdentity() and (schema is not None or cached_normals):
        return copied, copied
    if schema is not None:
        # transformed_ply materializes a double coordinate expression and a
        # deep VTK result, plus a conversion when the source is not double.
        doubles = source.GetNumberOfPoints() * 3 * 8
        conversion = doubles if coordinates.GetDataType() != vtk.VTK_DOUBLE else 0
        peak = copied + conversion + 2 * doubles
        retained = copied - coordinate_bytes + doubles
        # coords and transformed stay live while result owns its new points.
        normal_stage = retained + 2 * doubles
        vertex = next((e for e in schema["elements"] if e["name"] == "vertex"), {})
        normals = {p["name"]: p for p in vertex.get("properties", ())
                   if p.get("mapping") == "point_scalar" and p["name"] in {"nx", "ny", "nz"}}
        if len(normals) == 3:
            arrays = [source.GetPointData().GetArray(normals[name]["array"])
                      for name in ("nx", "ny", "nz")]
            dtype = np.result_type(*(vtk_to_numpy(a).dtype for a in arrays))
            stack = source.GetNumberOfPoints() * 3 * dtype.itemsize
            component = source.GetNumberOfPoints() * 8
            valid = source.GetNumberOfPoints()  # NumPy bool
            peak = max(
                peak,
                normal_stage + stack + doubles,  # column_stack -> astype copy
                normal_stage + 2 * doubles + 2 * component,  # norm product, reduction, sqrt
                normal_stage + doubles + component + valid + 3 * valid,  # finite check
            )
            for array in arrays:
                # A strided component is made contiguous and then deep-copied
                # into VTK before the old component is removed from result.
                peak = max(peak, normal_stage + doubles + component + valid + 2 * component)
                growth = component - array.GetDataSize() * array.GetDataTypeSize()
                normal_stage += growth
                retained += growth
            normal_stage += doubles + component + valid  # locals survive into reflection
        linear = np.array([[matrix.GetElement(r, c) for c in range(3)] for r in range(3)])
        if np.linalg.slogdet(linear)[0] < 0:
            peak = max(peak, normal_stage + source.GetNumberOfPoints() * 3)
            polys = source.GetPolys()
            offsets = (polys.GetNumberOfCells() + 1) * 8
            connectivity = polys.GetConnectivityArray().GetDataSize() * 8
            # cumsum converts the face-length list; SetData deep-copies both
            # arrays before replacing the old polygon storage.
            peak = max(peak, normal_stage + offsets + 2 * polys.GetNumberOfCells() * 8,
                       normal_stage + 2 * (offsets + connectivity))
            old_topology = sum(a.GetDataSize() * a.GetDataTypeSize()
                               for a in (polys.GetOffsetsArray(), polys.GetConnectivityArray()))
            growth = offsets + connectivity - old_topology
            retained += growth
            normal_stage += growth
            for element in schema["elements"]:
                for prop in element["properties"]:
                    if element["name"] == "face" and prop["name"] == "texcoord" and prop.get("mapping") == "list":
                        field = source.GetFieldData()
                        values = field.GetArray(prop["array"] + "_values")
                        uv_offsets = field.GetArray(prop["array"] + "_offsets")
                        cast = (uv_offsets.GetDataSize() * 8
                                if vtk_to_numpy(uv_offsets).dtype != np.dtype(np.int64) else 0)
                        uv_bytes = values.GetDataSize() * values.GetDataTypeSize()
                        peak = max(peak, normal_stage + connectivity + cast + 2 * uv_bytes)
        return max(peak, retained), retained
    # vtkTransformPolyDataFilter replaces coordinates and active normals/vectors;
    # topology and other arrays alias the private processing copy.
    arrays = {coordinates.GetAddressAsString(""): coordinates}
    for attributes in (source.GetPointData(), source.GetCellData()):
        for array in (attributes.GetNormals(), attributes.GetVectors()):
            if array is not None:
                arrays[array.GetAddressAsString("")] = array
    transformed = sum(a.GetDataSize() * a.GetDataTypeSize() for a in arrays.values())
    generated = source.GetNumberOfPoints() * 3 * 4 if not cached_normals else 0
    return copied + transformed + generated, copied + generated


def export_mesh(source, matrix, path, metadata, *, cancel_check=None, native_matrix=None,
                coordinates=None, ply_encoding="binary_little_endian", texture_bytes=None,
                return_geometry=False, memory_reserved=False, transform_resources=None,
                geometry_ownership="borrowed"):
    """Publish stable geometry and evidence.

    Borrowed input is copied before processing. ``private`` transfers exclusive
    use of detached input through completion: callers must not read or mutate it
    concurrently. Unbaked writing reuses that backing (including texture relinks).
    ``return_geometry`` returns the written backing only after publication; the
    caller may then adopt it. Baked processing retains its existing copy policy.
    """
    from pathlib import Path
    from madi3d_app.project.scientific_records import write_object_metadata_sidecar, object_metadata_sidecar_path
    from madi3d_app.project.scientific_records import COORDINATES_KEY, export_coordinates, new_identity, validate_export_coordinates
    from .processing import transformed_polydata
    from .ply import ply_schema, relink_ply_texture, _unsupported_field_arrays

    if geometry_ownership not in {"borrowed", "private"}:
        raise ValueError(f"Unsupported geometry ownership: {geometry_ownership}")
    destination_format = Path(path).suffix.lower()
    schema = ply_schema(source)
    source_is_ply = schema is not None
    # A declaration containing only points and faces adds no attributes for OBJ
    # to lose. Rich PLY declarations still require a preserving container.
    geometry_only = (
        bool(schema) and not schema.get("comments") and not schema.get("obj_info")
        and all(prop.get("mapping") in {"geometry_coordinate", "geometry_faces"}
                for element in schema["elements"] for prop in element["properties"])
        and source.GetPointData().GetNumberOfArrays() == 0
        and source.GetCellData().GetNumberOfArrays() == 0
        and not _unsupported_field_arrays(source, schema)
    )
    if source_is_ply and destination_format not in {".ply", ".vtp"} and not (
        destination_format == ".obj" and geometry_only
    ):
        raise ValueError("This output format would omit PLY source attributes. Choose PLY or VTP.")
    texture_reference = None
    if source_is_ply:
        references = [comment[len("TextureFile "):].strip()
                      for comment in ply_schema(source).get("comments", ())
                      if comment.startswith("TextureFile ")]
        if references:
            if len(references) != 1 or not references[0]:
                raise ValueError("PLY texture references are ambiguous; export cannot preserve the texture.")
            if destination_format != ".ply":
                raise ValueError("Textured PLY requires PLY export to retain its image asset.")
            if not texture_bytes:
                raise ValueError("The PLY texture image is unavailable; export cannot publish an incomplete asset.")
            texture_reference = references[0]
    if matrix is not None:
        poly = transformed_polydata(
            source, matrix, cancel_check=cancel_check, memory_reserved=memory_reserved
        )
    elif geometry_ownership == "private":
        check_cancelled(cancel_check)
        poly = source
    else:
        check_cancelled(cancel_check)
        if not memory_reserved:
            check_resources(memory_bytes=source.GetActualMemorySize() * 1024)
        poly = vtk.vtkPolyData()
        poly.DeepCopy(source)
    if poly.GetNumberOfPoints() <= 0:
        raise ValueError("Mesh has no exportable geometry.")
    companion = None
    export_id = new_identity()
    metadata = dict(metadata or {})
    previous = metadata.get(COORDINATES_KEY, {})
    metadata[COORDINATES_KEY] = validate_export_coordinates(coordinates) if coordinates is not None else export_coordinates(_matrix_rows(matrix if matrix is not None else native_matrix),
        baked=matrix is not None, units=previous.get("units", "unknown"),
        coordinate_space=previous.get("coordinate_space", "unknown"))
    coordinates = metadata[COORDINATES_KEY]
    independent = (coordinates["units"] == "unknown" and
                   coordinates["import_matrix4x4"] == export_coordinates()["import_matrix4x4"])
    warning = None
    asset_path = None
    asset_published = False
    pair_committed = [False]
    try:
        if texture_reference is not None:
            texture_suffix = Path(texture_reference).suffix.lower()
            if texture_suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                raise ValueError("The PLY texture image format is unsupported for portable export.")
            asset_path = Path(path).with_name(f"madi3d-{export_id}{texture_suffix}")
            if asset_path.exists():
                raise FileExistsError(f"Texture output already exists: {asset_path}")
            relink_ply_texture(poly, asset_path.name)
        with staged_polydata(poly, path, cancel_check=cancel_check, publish=False,
                             export_id=export_id, ply_encoding=ply_encoding,
                             memory_reserved=memory_reserved) as stage:
            companion = object_metadata_sidecar_path(stage)
            try:
                write_object_metadata_sidecar(stage, metadata, cancel_check=cancel_check, export_id=export_id,
                                              transform_resources=transform_resources)
            except InterruptedError:
                raise
            except OSError as exc:
                if not independent:
                    raise
                Path(companion).unlink(missing_ok=True)
                companion = None
                warning = f"Data saved; provenance companion failed: {exc}"
            if asset_path is not None:
                check_resources(disk_bytes=len(texture_bytes))
                with staged_file(asset_path, cancel_check=cancel_check) as asset_stage:
                    with open(asset_stage, "wb") as stream:
                        view = memoryview(texture_bytes)
                        for offset in range(0, len(view), 1024 * 1024):
                            check_cancelled(cancel_check)
                            stream.write(view[offset:offset + 1024 * 1024])
                        stream.flush()
                        os.fsync(stream.fileno())
                asset_published = True
            warning = publish_file_pair(stage, path, companion, object_metadata_sidecar_path(path),
                cancel_check=cancel_check, optional_history=independent and asset_path is None,
                on_commit=(lambda: pair_committed.__setitem__(0, True)) if asset_path is not None else None) or warning
        return (warning, poly) if return_geometry else warning
    except BaseException as failure:
        if asset_published and not pair_committed[0]:
            barrier = Path(object_metadata_sidecar_path(path) + PUBLICATION_MARKER_SUFFIX)
            marker_name_failed = (
                isinstance(failure, OSError) and failure.errno == errno.ENAMETOOLONG
                and failure.filename == str(barrier)
            )
            if marker_name_failed or not barrier.exists():
                asset_path.unlink(missing_ok=True)
        raise
    finally:
        if companion is not None:
            Path(companion).unlink(missing_ok=True)


def _matrix_rows(matrix):
    if matrix is None:
        return None
    return ([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)]
            if hasattr(matrix, "GetElement") else matrix)


def export_swc(source_path, path, metadata, *, source_revision, source_checksum,
               matrix=None, native_matrix=None, cancel_check=None):
    """Bind exported nodes to the exact captured skeleton bytes."""
    from madi3d_app.io_utils import file_revision
    from madi3d_app.tools.swc import parse_swc_with_sha256
    before = file_revision(source_path)
    if not source_revision or tuple(source_revision) != before:
        raise ValueError("The source skeleton changed after export was prepared. Retry exporting the current skeleton.")
    nodes, parsed_checksum = parse_swc_with_sha256(source_path, cancel_check=cancel_check)
    if file_revision(source_path) != before:
        raise ValueError("The source skeleton changed while reading it. Retry exporting.")
    if not source_checksum or parsed_checksum != source_checksum:
        raise ValueError("The source skeleton changed after export was prepared. Retry exporting the current skeleton.")
    return write_swc_file(path, nodes, metadata=metadata, matrix=matrix, native_matrix=native_matrix, cancel_check=cancel_check)


def export_volume_surface(image, matrix, iso_value, path, *, smooth_normals,
                          reverse_cells, cancel_check=None, metadata=None,
                          memory_reserved=False, cutoff_range=None,
                          surface_smoothing=False):
    """Extract privately, apply the captured matrix once, and publish a surface.

    ``volume_surface`` copies the input image and retains its spacing, origin
    and direction. No live surface filter, mapper or actor enters this worker.
    Preserve the surface action's collision policy: resolve a unique filename
    after extraction, leaving existing exports untouched.
    """
    from madi3d_app.scene.export import unique_export_path
    from .processing import volume_surface

    poly = volume_surface(image, matrix, iso_value, smooth_normals=smooth_normals,
                          reverse_cells=reverse_cells, cancel_check=cancel_check,
                          memory_reserved=memory_reserved,
                          cutoff_range=cutoff_range,
                          surface_smoothing=surface_smoothing)
    if poly.GetNumberOfPoints() == 0 or poly.GetNumberOfCells() == 0:
        if cutoff_range is None:
            raise RuntimeError(f"The surface is empty at iso value {iso_value:g}.")
        raise RuntimeError(
            "The surface is empty for cutoff band "
            f"[{cutoff_range[0]:g}, {cutoff_range[1]:g}]."
        )
    path = unique_export_path(path)
    from madi3d_app.project.scientific_records import export_coordinates
    context = (metadata or {}).get("export_coordinates", {})
    warning = export_mesh(poly, None, path, metadata, cancel_check=cancel_check,
                memory_reserved=memory_reserved,
                coordinates=export_coordinates(_matrix_rows(matrix), baked=True,
                    units=context.get("units", "unknown"), coordinate_space=context.get("coordinate_space", "unknown")))
    return path, warning


def write_swc_file(
    path,
    nodes_dict,
    source_path=None,
    source_paths=None,
    split_node_id=None,
    piece_name=None,
    note=None,
    cancel_check=None,
    metadata=None,
    matrix=None,
    native_matrix=None,
):
    """Serialize canonical skeleton nodes, never reconstructed render tubes.

    Baked SWC supports similarity transforms: one scalar radius cannot encode
    shear or anisotropic scaling. Reject those rather than inventing radii.
    """
    from madi3d_app.project.scientific_records import COORDINATES_KEY, export_coordinates, new_identity
    from madi3d_app.project.scientific_records import object_metadata_sidecar_path, write_object_metadata_sidecar
    from madi3d_app.volume.geometry import invertible_affine4
    if metadata is not None or matrix is not None:
        if not nodes_dict:
            raise ValueError("The source skeleton contains no nodes.")
        from numbers import Integral
        from madi3d_app.project.scientific_records import validate_acyclic
        edges = {}
        for nid, (typ, x, y, z, radius, parent) in nodes_dict.items():
            check_cancelled(cancel_check)
            if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (nid, typ, parent)):
                raise ValueError("SWC node IDs, types and parents must be integers.")
            if not np.isfinite((x, y, z, radius)).all() or radius < 0:
                raise ValueError("SWC coordinates and radii must be finite; radii cannot be negative.")
            edges[nid] = [parent] if parent >= 0 and parent in nodes_dict else []
        validate_acyclic(edges)
    if matrix is not None:
        affine = np.asarray(invertible_affine4(_matrix_rows(matrix), "SWC export transform"))
        gram = affine[:3, :3].T @ affine[:3, :3]
        scale = float(np.sqrt(np.trace(gram) / 3))
        if not np.isfinite(gram).all() or not np.isfinite(scale) or scale <= 0:
            raise ValueError("SWC transform exceeds the supported numerical range.")
        if not np.allclose(gram, np.eye(3) * scale**2, rtol=1e-10, atol=1e-12):
            raise ValueError("SWC scalar radii cannot represent shear or unequal axis scales. Export native SWC coordinates instead.")
        transformed = {}
        for nid, (typ, x, y, z, radius, parent) in nodes_dict.items():
            check_cancelled(cancel_check)
            xyz = affine @ np.array([x, y, z, 1.])
            if not np.isfinite((*xyz[:3], radius * scale)).all():
                raise ValueError("Transformed SWC coordinates or radii exceed the supported numerical range.")
            transformed[nid] = (typ, *xyz[:3], radius * scale, parent)
        nodes_dict = transformed
    nodes_to_write = reindex_nodes_for_write(nodes_dict, cancel_check=cancel_check)
    export_id = new_identity() if metadata is not None else None
    if metadata is not None:
        metadata = dict(metadata)
        previous = metadata.get(COORDINATES_KEY, {})
        metadata[COORDINATES_KEY] = export_coordinates(_matrix_rows(matrix if matrix is not None else native_matrix),
            baked=matrix is not None, units=previous.get("units", "unknown"),
            coordinate_space=previous.get("coordinate_space", "unknown"))
        metadata["swc_semantics"] = {"representation": "skeleton_nodes", "radius_transform": "uniform_similarity_scale" if matrix is not None else "unchanged",
                                     "node_ids": "reindexed_parent_first", "missing_parent_handling": "root_minus_1", "precision_decimal_places": 6}

    all_sources = []

    if source_path:
        all_sources.append(source_path)

    if source_paths:
        all_sources.extend(source_paths)

    companion = None
    warning = None
    independent = metadata is None or (metadata[COORDINATES_KEY]["units"] == "unknown" and
        metadata[COORDINATES_KEY]["import_matrix4x4"] == export_coordinates()["import_matrix4x4"])
    try:
        with staged_file(path, cancel_check=cancel_check, publish=metadata is None) as stage:
            _serialize_swc(stage, nodes_to_write, all_sources, piece_name, note, split_node_id, export_id, cancel_check)
            if metadata is not None:
                companion = object_metadata_sidecar_path(stage)
                try:
                    write_object_metadata_sidecar(stage, metadata, cancel_check=cancel_check, export_id=export_id)
                except InterruptedError:
                    raise
                except OSError as exc:
                    if not independent:
                        raise
                    from pathlib import Path
                    Path(companion).unlink(missing_ok=True)
                    companion = None
                    warning = f"Data saved; provenance companion failed: {exc}"
                warning = publish_file_pair(stage, path, companion, object_metadata_sidecar_path(path),
                    cancel_check=cancel_check, optional_history=independent) or warning
        return warning
    finally:
        if companion is not None:
            from pathlib import Path
            Path(companion).unlink(missing_ok=True)


def _serialize_swc(stage, nodes_to_write, all_sources, piece_name, note, split_node_id, export_id, cancel_check):
    with open(stage, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Created by MADI3D\n")
        if export_id:
            f.write(f"# MADI3D export generation: {export_id}\n")

        if piece_name:
            f.write(f"# Piece: {piece_name}\n")

        if note:
            f.write(f"# Note: {note}\n")

        for src in all_sources:
            f.write(f"# Source: {src}\n")

        if split_node_id is not None:
            f.write(f"# Original split node ID: {split_node_id}\n")

        f.write("# Columns: id type x y z radius parent\n")

        for nid in sorted(nodes_to_write.keys()):
            check_cancelled(cancel_check)
            typ, x, y, z, r, parent = nodes_to_write[nid]
            f.write(
                f"{int(nid)} "
                f"{int(typ)} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{z:.6f} "
                f"{r:.6f} "
                f"{int(parent)}\n"
            )
