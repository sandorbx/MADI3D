"""Validated on-disk mesh cache and normal generation."""

from __future__ import annotations

import os
import json
import hashlib
from contextlib import contextmanager
from threading import BoundedSemaphore, RLock
from weakref import WeakValueDictionary

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from madi3d_app.io_utils import check_cancelled, check_resources, file_revision

from .export import staged_polydata
from .processing import update_algorithm

_source_locks = WeakValueDictionary()
_registry_lock = RLock()
# Only cache misses take this slot. VTP reads retain the loader's concurrency.
_swc_build_slot = BoundedSemaphore(1)
_SWC_IDENTITY_FIELD = "madi3d_swc_render_cache"
_PLY_IDENTITY_FIELD = "madi3d_ply_cache_identity"
_PLY_DIGEST_FIELD = "madi3d_ply_cache_digest"
# Bump when the PLY codec's decoded representation or derivative contract changes.
_PLY_DERIVATIVE_VERSION = 1
_DIGEST_CHUNK_ROWS = 65536


class _SourceRevisionError(OSError):
    """The source revision is unavailable or changed mid-operation."""


def _source_lock(path):
    key = os.path.normcase(os.path.realpath(path))
    with _registry_lock:
        lock = _source_locks.get(key)
        if lock is None:
            lock = RLock()
            _source_locks[key] = lock
        return lock


@contextmanager
def _cancellable_lock(lock, cancel_check):
    check_cancelled(cancel_check)
    while not lock.acquire(timeout=0.05):
        check_cancelled(cancel_check)
    try:
        check_cancelled(cancel_check)
        yield
    finally:
        lock.release()


def _source_revision(path):
    try:
        return list(file_revision(path))
    except OSError as exc:
        raise _SourceRevisionError(f"Could not inspect mesh source revision: {path}") from exc


def get_cache_dir(original_file_path):
    """
    Returns (and creates if necessary) the .madicache folder in the same directory as the original file.
    """
    folder = os.path.dirname(original_file_path)
    cache_dir = os.path.join(folder, ".madicache")
    # Create the cache directory if it doesn't exist.
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_cache_filename(original_file_path):
    """
    Keep PLY/SWC derivatives distinct from same-stem OBJ caches.
    """
    base_name = os.path.splitext(os.path.basename(original_file_path))[0]
    ext = os.path.splitext(original_file_path)[1].lower()
    suffix = {".swc": ".swc.vtp", ".ply": ".ply.vtp"}.get(ext, ".vtp")
    return base_name + suffix


def _ply_identity_field(polydata, name):
    field = polydata.GetFieldData().GetAbstractArray(name)
    if field is None or field.GetNumberOfValues() != 1:
        return None
    try:
        return json.loads(field.GetValue(0))
    except (TypeError, ValueError):
        return None


def _update_array_digest(digest, array, cancel_check):
    """Hash an array's complete declared representation in bounded chunks."""
    metadata = (array.GetClassName(), array.GetName(), array.GetNumberOfComponents(),
                array.GetNumberOfTuples(), array.GetNumberOfValues())
    digest.update(json.dumps(metadata, separators=(",", ":"), ensure_ascii=True).encode("ascii"))
    digest.update(b"\0")
    if isinstance(array, vtk.vtkStringArray):
        for start in range(0, array.GetNumberOfValues(), _DIGEST_CHUNK_ROWS):
            check_cancelled(cancel_check)
            end = min(start + _DIGEST_CHUNK_ROWS, array.GetNumberOfValues())
            for index in range(start, end):
                value = array.GetValue(index).encode("utf-8", errors="surrogatepass")
                digest.update(len(value).to_bytes(8, "little"))
                digest.update(value)
        return
    values = vtk_to_numpy(array)
    for start in range(0, len(values), _DIGEST_CHUNK_ROWS):
        check_cancelled(cancel_check)
        chunk = values[start:start + _DIGEST_CHUNK_ROWS]
        digest.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))


def _update_connectivity_digest(digest, array, cancel_check):
    """Hash topology values while ignoring VTK's writer-specific array names."""
    values = vtk_to_numpy(array)
    metadata = (values.dtype.str, array.GetNumberOfComponents(), array.GetNumberOfTuples(),
                array.GetNumberOfValues())
    digest.update(json.dumps(metadata, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    for start in range(0, len(values), _DIGEST_CHUNK_ROWS):
        check_cancelled(cancel_check)
        digest.update(memoryview(np.ascontiguousarray(values[start:start + _DIGEST_CHUNK_ROWS])).cast("B"))


def _ply_polydata_digest(polydata, *, cancel_check=None):
    """Digest geometry, topology, arrays, schema, and active attribute roles."""
    digest = hashlib.sha256()
    digest.update(b"madi3d-ply-polydata-v1\0")
    digest.update(json.dumps((polydata.GetNumberOfPoints(), polydata.GetNumberOfCells(),
                              polydata.GetNumberOfVerts(), polydata.GetNumberOfLines(),
                              polydata.GetNumberOfPolys(), polydata.GetNumberOfStrips()),
                             separators=(",", ":")).encode("ascii"))
    points = polydata.GetPoints()
    if points is None:
        digest.update(b"no-points\0")
    else:
        digest.update(b"points\0")
        _update_array_digest(digest, points.GetData(), cancel_check)
    for association, attributes in (("point", polydata.GetPointData()),
                                    ("cell", polydata.GetCellData()),
                                    ("field", polydata.GetFieldData())):
        check_cancelled(cancel_check)
        digest.update(association.encode("ascii") + b"\0")
        roles = []
        for role in ("Scalars", "Vectors", "Normals", "TCoords", "Tensors", "GlobalIds", "PedigreeIds"):
            getter = getattr(attributes, "Get" + role, None)
            active = getter() if getter else None
            roles.append((role, active.GetName() if active is not None else None))
        digest.update(json.dumps(roles, separators=(",", ":")).encode("utf-8"))
        for index in range(attributes.GetNumberOfArrays()):
            array = attributes.GetAbstractArray(index)
            if association == "field" and array.GetName() in {_PLY_IDENTITY_FIELD, _PLY_DIGEST_FIELD}:
                continue
            _update_array_digest(digest, array, cancel_check)
    for kind, cells in (("verts", polydata.GetVerts()), ("lines", polydata.GetLines()),
                        ("polys", polydata.GetPolys()), ("strips", polydata.GetStrips())):
        digest.update(kind.encode("ascii") + b"\0")
        if cells is None:
            digest.update(b"none\0")
            continue
        digest.update(cells.GetClassName().encode("ascii") + b"\0")
        for array in (cells.GetOffsetsArray(), cells.GetConnectivityArray()):
            if array is None:
                digest.update(b"none\0")
            else:
                _update_connectivity_digest(digest, array, cancel_check)
    return digest.hexdigest()


def _remove_ply_cache_fields(polydata):
    field = polydata.GetFieldData()
    field.RemoveArray(_PLY_IDENTITY_FIELD)
    field.RemoveArray(_PLY_DIGEST_FIELD)
    return polydata


def load_cached_ply(path, *, cancel_check=None, memory_reserved=False):
    """Load an exact PLY decode, reusing only a verified lossless VTP derivative."""
    path = os.fspath(path)
    if os.path.splitext(path)[1].lower() != ".ply":
        raise ValueError(f"Expected a PLY source: {path}")
    check_cancelled(cancel_check)
    try:
        cache = get_cache_file_path(path)
    except OSError:
        cache = None
    with _cancellable_lock(_source_lock(path), cancel_check):
        revision = _source_revision(path)
        identity = {"source_revision": revision, "derivative_version": _PLY_DERIVATIVE_VERSION}
        if cache is not None and os.path.isfile(cache):
            try:
                cached = _read_vtp(cache, cancel_check=cancel_check)
                stored_identity = _ply_identity_field(cached, _PLY_IDENTITY_FIELD)
                stored_digest = _ply_identity_field(cached, _PLY_DIGEST_FIELD)
                if (stored_identity == identity and isinstance(stored_digest, str)
                        and _ply_polydata_digest(cached, cancel_check=cancel_check) == stored_digest):
                    if revision != _source_revision(path):
                        raise _SourceRevisionError("The PLY source changed while loading its cache. Retry loading it.")
                    check_cancelled(cancel_check)
                    return _remove_ply_cache_fields(cached)
            except (InterruptedError, _SourceRevisionError):
                raise
            except (OSError, ValueError, RuntimeError, TypeError, OverflowError, MemoryError):
                pass  # Cache corruption or unsupported VTP representation falls back to source decoding.
            cached = None

        from .ply import read_ply
        polydata = read_ply(path, cancel_check=cancel_check, memory_reserved=memory_reserved)
        if revision != _source_revision(path):
            raise _SourceRevisionError("The PLY source changed while decoding. Retry loading it.")
        if cache is not None:
            try:
                source_digest = _ply_polydata_digest(polydata, cancel_check=cancel_check)
                if not memory_reserved:
                    check_resources(memory_bytes=polydata.GetActualMemorySize() * 1024 * 2)
                check_resources(disk_bytes=polydata.GetActualMemorySize() * 1024 * 2,
                                directory=os.path.dirname(cache))
                identity_array = vtk.vtkStringArray()
                identity_array.SetName(_PLY_IDENTITY_FIELD)
                identity_array.InsertNextValue(json.dumps(identity, sort_keys=True, separators=(",", ":")))
                digest_array = vtk.vtkStringArray()
                digest_array.SetName(_PLY_DIGEST_FIELD)
                digest_array.InsertNextValue(json.dumps(source_digest))
                polydata.GetFieldData().AddArray(identity_array)
                polydata.GetFieldData().AddArray(digest_array)
                try:
                    with staged_polydata(polydata, cache, cancel_check=cancel_check) as stage:
                        verified = _read_vtp(stage, cancel_check=cancel_check)
                        if (_ply_identity_field(verified, _PLY_IDENTITY_FIELD) != identity
                                or _ply_identity_field(verified, _PLY_DIGEST_FIELD) != source_digest
                                or _ply_polydata_digest(verified, cancel_check=cancel_check) != source_digest):
                            raise ValueError("The written PLY cache does not preserve the source representation.")
                        del verified
                        if revision != _source_revision(path):
                            raise _SourceRevisionError("The PLY source changed while writing its cache. Retry loading it.")
                finally:
                    polydata.GetFieldData().RemoveArray(_PLY_IDENTITY_FIELD)
                    polydata.GetFieldData().RemoveArray(_PLY_DIGEST_FIELD)
            except (InterruptedError, _SourceRevisionError):
                raise
            except (OSError, ValueError, RuntimeError, TypeError, OverflowError, MemoryError):
                pass  # Cache storage is optional; retain the exact source decode.
        if revision != _source_revision(path):
            raise _SourceRevisionError("The PLY source changed while loading. Retry loading it.")
        check_cancelled(cancel_check)
        return polydata


def get_cache_file_path(original_file_path):
    """
    Returns the full path to the cache file.
    """
    cache_dir = get_cache_dir(original_file_path)
    return os.path.join(cache_dir, get_cache_filename(original_file_path))


def is_cache_valid(original_file_path, cache_file):
    """
    Validates the cache by comparing file modification times.
    Returns True if the cache file exists and is newer than (or as new as) the original.
    """
    if not os.path.exists(cache_file):
        return False
    return os.path.getmtime(cache_file) >= os.path.getmtime(original_file_path)


def has_valid_point_normals(polydata: vtk.vtkPolyData) -> bool:
    n = polydata.GetPointData().GetNormals()
    return bool(n) and n.GetNumberOfTuples() == polydata.GetNumberOfPoints()


def _validate_polydata_connectivity(polydata, *, cancel_check=None, label="polydata"):
    """Reject empty geometry and out-of-range cell point ids before VTK filters.

    vtkOBJReader can emit cells whose point ids exceed the point array. Those
    meshes raise here as ordinary ValueError instead of crashing later inside
    CleanPolyData / FeatureEdges / PolyDataNormals.
    """
    npts = int(polydata.GetNumberOfPoints())
    if npts <= 0:
        raise ValueError(f"Empty or corrupted {label}")
    for getter in (polydata.GetVerts, polydata.GetLines, polydata.GetPolys, polydata.GetStrips):
        check_cancelled(cancel_check)
        cells = getter()
        if cells is None or cells.GetNumberOfCells() == 0:
            continue
        if hasattr(cells, "IsValid") and not cells.IsValid():
            raise ValueError(f"Empty or corrupted {label}")
        ids = vtk_to_numpy(cells.GetConnectivityArray())
        if ids.size == 0:
            continue
        for start in range(0, ids.size, 1 << 20):
            check_cancelled(cancel_check)
            chunk = ids[start:start + (1 << 20)]
            if int(chunk.min()) < 0 or int(chunk.max()) >= npts:
                raise ValueError(f"Empty or corrupted {label}")


def _read_vtp(path, *, cancel_check=None):
    """Read detached VTP from a file or embedded bytes through one reader."""
    reader = vtk.vtkXMLPolyDataReader()
    if isinstance(path, (bytes, bytearray, memoryview)):
        reader.ReadFromInputStringOn()
        reader.SetInputString(bytes(path))
    else:
        reader.SetFileName(os.fspath(path))
    errors = []
    observer = reader.AddObserver(vtk.vtkCommand.ErrorEvent, lambda *_: errors.append(True))
    try:
        update_algorithm(reader, cancel_check=cancel_check)
        result = reader.GetOutput()
        if errors or reader.GetErrorCode() or result.GetNumberOfPoints() == 0:
            raise ValueError("Empty or invalid VTP geometry.")
        _validate_polydata_connectivity(result, cancel_check=cancel_check, label="VTP geometry")
        detached = vtk.vtkPolyData()
        detached.ShallowCopy(result)
        return detached
    finally:
        reader.RemoveObserver(observer)
        reader.SetFileName("")
        reader.SetInputString("")


def _validate_swc_surface(polydata, *, cancel_check=None):
    """Reject incomplete render output, including VTK's plausible partial arrays."""
    if (polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfPolys() == 0
            or not has_valid_point_normals(polydata)):
        raise ValueError("Incomplete SWC render surface.")
    for array in (polydata.GetPoints().GetData(), polydata.GetPointData().GetNormals()):
        if array.GetNumberOfComponents() != 3:
            raise ValueError("Invalid SWC surface coordinates or normals.")
        values = vtk_to_numpy(array)
        for start in range(0, len(values), 65536):
            check_cancelled(cancel_check)
            if not np.isfinite(values[start:start + 65536]).all():
                raise ValueError("Non-finite SWC surface coordinates or normals.")
    cells = polydata.GetPolys()
    if not cells.IsValid():
        raise ValueError("Incomplete SWC surface connectivity.")
    ids = vtk_to_numpy(cells.GetConnectivityArray())
    if ids.size == 0 or ids.min() < 0 or ids.max() >= polydata.GetNumberOfPoints():
        raise ValueError("Invalid SWC surface connectivity.")


def _swc_cache_identity(path, renderer_identity, *, cancel_check=None):
    revision = _source_revision(path)
    check_cancelled(cancel_check)
    if revision != _source_revision(path):
        raise _SourceRevisionError("The SWC source changed while capturing its file revision. Retry loading it.")
    return {"source_revision": revision, "renderer": renderer_identity}


def _swc_cache_matches(polydata, identity):
    field = polydata.GetFieldData().GetAbstractArray(_SWC_IDENTITY_FIELD)
    if field is None or field.GetNumberOfValues() != 1:
        return False
    try:
        stored = json.loads(field.GetValue(0))
        return stored == {"identity": identity, "points": polydata.GetNumberOfPoints(),
                          "polys": polydata.GetNumberOfPolys()}
    except (ValueError, TypeError):
        return False


def load_cached_swc(original_file_path, generate_surface, renderer_identity, *, cancel_check=None,
                    memory_reserved=False):
    """Cache display geometry only; the source SWC remains the morphology authority.

    The embedded source fingerprint and renderer contract travel atomically with
    the geometry. Per-source locking coalesces same-process requests; unique
    staged files and atomic replacement also keep independent writers complete.
    """
    path = os.fspath(original_file_path)
    if os.path.splitext(path)[1].lower() != ".swc":
        raise ValueError(f"Expected an SWC source: {path}")
    check_cancelled(cancel_check)
    try:
        cache = get_cache_file_path(path)
    except OSError:
        cache = None
    with _cancellable_lock(_source_lock(path), cancel_check):
        identity = _swc_cache_identity(path, renderer_identity, cancel_check=cancel_check)
        cache_available = cache is not None and os.path.isfile(cache)
        if cache_available:
            try:
                cached = _read_vtp(cache, cancel_check=cancel_check)
                if _swc_cache_matches(cached, identity):
                    _validate_swc_surface(cached, cancel_check=cancel_check)
                    if identity["source_revision"] != _source_revision(path):
                        raise _SourceRevisionError("The SWC source changed while loading its cache. Retry loading it.")
                    cached.GetFieldData().RemoveArray(_SWC_IDENTITY_FIELD)
                    return cached
            except (InterruptedError, _SourceRevisionError):
                raise
            except OSError:
                pass  # A cache read failure does not prevent rendering the source.
            except ValueError:
                pass  # Preserve the old file until a complete replacement exists.
            cached = None  # Release stale geometry before building its replacement.

        with _cancellable_lock(_swc_build_slot, cancel_check):
            polydata = generate_surface(path, cancel_check=cancel_check)
            _validate_swc_surface(polydata, cancel_check=cancel_check)
            field = vtk.vtkStringArray()
            field.SetName(_SWC_IDENTITY_FIELD)
            field.InsertNextValue(json.dumps({"identity": identity,
                "points": polydata.GetNumberOfPoints(), "polys": polydata.GetNumberOfPolys()},
                sort_keys=True))
            polydata.GetFieldData().AddArray(field)
            if cache is not None:
                if not memory_reserved:
                    check_resources(memory_bytes=polydata.GetActualMemorySize() * 1024 * 2)
                try:
                    check_resources(disk_bytes=polydata.GetActualMemorySize() * 1024 * 2,
                                    directory=os.path.dirname(cache))
                    with staged_polydata(polydata, cache, cancel_check=cancel_check) as stage:
                        verified = _read_vtp(stage, cancel_check=cancel_check)
                        _validate_swc_surface(verified, cancel_check=cancel_check)
                        if (not _swc_cache_matches(verified, identity)
                                or verified.GetBounds() != polydata.GetBounds()):
                            raise ValueError("The written SWC cache does not match the completed surface.")
                        del verified
                        if identity != _swc_cache_identity(path, renderer_identity, cancel_check=cancel_check):
                            raise _SourceRevisionError("The SWC source changed during rendering. Retry loading it.")
                except (InterruptedError, _SourceRevisionError):
                    raise
                except OSError:
                    pass  # Cache storage is optional; keep the validated source rendering.
            if identity != _swc_cache_identity(path, renderer_identity, cancel_check=cancel_check):
                raise _SourceRevisionError("The SWC source changed during rendering. Retry loading it.")
            check_cancelled(cancel_check)
            polydata.GetFieldData().RemoveArray(_SWC_IDENTITY_FIELD)
            return polydata


def compute_smooth_point_normals(polydata: vtk.vtkPolyData, *, cancel_check=None,
                                 preserve_topology=False) -> vtk.vtkPolyData:
    """Discard imported normals and compute smooth point normals.

    Ordinary cache input is triangulated, welded and oriented. Material mappings
    can preserve accepted vertex identities, connectivity and winding instead.
    """

    check_cancelled(cancel_check)
    _validate_polydata_connectivity(polydata, cancel_check=cancel_check)
    # Remove possibly bad/imported normals
    if polydata.GetPointData():
        polydata.GetPointData().SetNormals(None)
        polydata.GetPointData().RemoveArray("Normals")

    if polydata.GetCellData():
        polydata.GetCellData().SetNormals(None)
        polydata.GetCellData().RemoveArray("Normals")

    if not preserve_topology:
        # Triangulate and weld ordinary render-cache input. Material coordinate
        # mappings already have accepted topology and must retain vertex identity.
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(polydata)
        update_algorithm(tri, cancel_check=cancel_check)

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(tri.GetOutputPort())
        clean.PointMergingOn()
        clean.ConvertLinesToPointsOff()
        clean.ConvertPolysToLinesOff()
        clean.ConvertStripsToPolysOn()
        update_algorithm(clean, cancel_check=cancel_check)

    # Remove normals again after cleaning, just in case they survived
    cleaned = vtk.vtkPolyData()
    cleaned.ShallowCopy(polydata if preserve_topology else clean.GetOutput())

    if cleaned.GetPointData():
        cleaned.GetPointData().SetNormals(None)
        cleaned.GetPointData().RemoveArray("Normals")

    if cleaned.GetCellData():
        cleaned.GetCellData().SetNormals(None)
        cleaned.GetCellData().RemoveArray("Normals")

    # Compute smooth vertex normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(cleaned)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.SplittingOff()          # important: smooth across all shared vertices
    normals.SetConsistency(not preserve_topology)
    normals.SetAutoOrientNormals(not preserve_topology)

    try:
        normals.SetNonManifoldTraversal(1)
    except AttributeError:
        pass

    update_algorithm(normals, cancel_check=cancel_check)

    out = vtk.vtkPolyData()
    out.ShallowCopy(normals.GetOutput())
    return out


def generate_cache_file(original_file_path, cache_file, *, cancel_check=None):
    """Generate the usual smooth mesh cache from a file or detached polydata."""
    file_backed = not isinstance(original_file_path, vtk.vtkPolyData)
    revision = _source_revision(original_file_path) if file_backed else None
    if file_backed:
        ext = os.path.splitext(original_file_path)[1].lower()
        if ext == ".obj":
            reader = vtk.vtkOBJReader()
        elif ext == ".vtp":
            reader = vtk.vtkXMLPolyDataReader()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        reader.SetFileName(original_file_path)
        update_algorithm(reader, cancel_check=cancel_check)
        polydata = reader.GetOutput()
    else:
        # Normal generation discards imported normals. Keep the caller's data
        # untouched while using the same triangulation/welding pipeline.
        polydata = vtk.vtkPolyData()
        polydata.ShallowCopy(original_file_path)
    label = f"polydata in {original_file_path}" if file_backed else "mesh cache input"
    _validate_polydata_connectivity(polydata, cancel_check=cancel_check, label=label)
    polydata_with_normals = compute_smooth_point_normals(polydata, cancel_check=cancel_check)

    if file_backed:
        mod_time = vtk.vtkStringArray()
        mod_time.SetName("source_mod_time")
        mod_time.InsertNextValue(str(os.path.getmtime(original_file_path)))
        identity = vtk.vtkStringArray()
        identity.SetName("madi3d_source_revision")
        identity.InsertNextValue(json.dumps(revision))
        polydata_with_normals.GetFieldData().AddArray(mod_time)
        polydata_with_normals.GetFieldData().AddArray(identity)
    if file_backed and revision != _source_revision(original_file_path):
        raise _SourceRevisionError("The mesh source changed during cache generation. Retry loading it.")
    check_cancelled(cancel_check)
    if cache_file is not None:
        try:
            with staged_polydata(polydata_with_normals, cache_file, cancel_check=cancel_check) as stage:
                verified = _read_vtp(stage, cancel_check=cancel_check)
                if (verified.GetNumberOfPoints() != polydata_with_normals.GetNumberOfPoints()
                        or verified.GetNumberOfCells() != polydata_with_normals.GetNumberOfCells()):
                    raise ValueError("The written mesh cache does not match the completed geometry.")
                del verified
                if file_backed and revision != _source_revision(original_file_path):
                    raise _SourceRevisionError("The mesh source changed during cache generation. Retry loading it.")
        except (InterruptedError, _SourceRevisionError):
            raise
        except OSError:
            pass  # Cache storage is optional; the decoded geometry is still usable.
    check_cancelled(cancel_check)
    if file_backed and revision != _source_revision(original_file_path):
        raise _SourceRevisionError("The mesh source changed during cache generation. Retry loading it.")
    polydata_with_normals.GetFieldData().RemoveArray("source_mod_time")
    polydata_with_normals.GetFieldData().RemoveArray("madi3d_source_revision")
    return polydata_with_normals


def load_cached_mesh(original_file_path, *, cancel_check=None):
    """Read a complete cache, or generate it and reuse the decoded geometry."""
    if isinstance(original_file_path, (bytes, bytearray, memoryview)):
        result = _read_vtp(original_file_path, cancel_check=cancel_check)
        return (result if has_valid_point_normals(result) else
                generate_cache_file(result, None, cancel_check=cancel_check))
    if os.path.splitext(original_file_path)[1].lower() == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(original_file_path)
        errors = []
        observer = reader.AddObserver(vtk.vtkCommand.ErrorEvent, lambda *_: errors.append(True))
        try:
            update_algorithm(reader, cancel_check=cancel_check)
            if errors or reader.GetErrorCode() or reader.GetOutput().GetNumberOfPoints() == 0:
                raise ValueError("Empty or corrupted legacy VTK polydata.")
            _validate_polydata_connectivity(
                reader.GetOutput(), cancel_check=cancel_check, label="legacy VTK polydata"
            )
            return reader.GetOutput()
        finally:
            reader.RemoveObserver(observer)
    native_vtp = os.path.splitext(original_file_path)[1].lower() == ".vtp"
    try:
        cache = original_file_path if native_vtp else get_cache_file_path(original_file_path)
    except OSError:
        cache = None
    with _cancellable_lock(_source_lock(original_file_path), cancel_check):
        check_cancelled(cancel_check)
        if native_vtp:
            result = _read_vtp(cache, cancel_check=cancel_check)
            return (result if has_valid_point_normals(result) else
                    generate_cache_file(result, None, cancel_check=cancel_check))
        if cache is not None:
            try:
                cache_is_current = is_cache_valid(original_file_path, cache)
            except OSError:
                cache_is_current = False
        else:
            cache_is_current = False
        if cache_is_current:
            try:
                result = _read_vtp(cache, cancel_check=cancel_check)
                identity = result.GetFieldData().GetAbstractArray("madi3d_source_revision")
                if identity is not None and identity.GetNumberOfValues() == 1:
                    try:
                        if json.loads(identity.GetValue(0)) == _source_revision(original_file_path):
                            result.GetFieldData().RemoveArray("source_mod_time")
                            result.GetFieldData().RemoveArray("madi3d_source_revision")
                            return result
                    except (ValueError, TypeError):
                        pass
            except (InterruptedError, _SourceRevisionError):
                raise
            except OSError:
                pass
            except ValueError:
                pass
            result = None  # Release stale geometry before replacing its cache.
        return generate_cache_file(original_file_path, cache, cancel_check=cancel_check)
