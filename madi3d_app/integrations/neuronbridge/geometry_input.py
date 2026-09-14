"""Bounded, explicit binary rasterization of captured mesh and skeleton geometry."""
from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkIdTypeArray, vtkPoints, vtkVersion
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkImagingHybrid import vtkVoxelModeller

from madi3d_app.volume.geometry import VolumeWorkingGrid, invertible_affine4
from .cdm import checkpoint
from .cdm_records import CDMGeometry, CDMParameters, CDMSelection, canonical_json
from .mapping import exploratory_mapping
from .query import QueryInput, QueryLaunch, digest
from .records import ChannelSelection, SourceIdentity


SCENE_SOURCE_KEY = "neuronbridge_cdm_source_id"
MAX_GEOMETRY_BYTES = 128 * 1024 * 1024
RASTER_WARNING = (
    "Mesh/SWC geometry is rasterized as binary occupancy, not fluorescence. "
    "Thin features depend on the chosen raster resolution; template alignment is unverified."
)


@dataclass(frozen=True)
class GeometryCapture:
    source_id: str
    points: np.ndarray
    cells: tuple[np.ndarray, ...]
    pose: tuple
    preparation_json: str
    parameters: CDMParameters
    placement: str
    resolution: int

    @classmethod
    def capture(cls, source_id, polydata, pose, parameters, placement, resolution=256, evidence=None):
        if type(resolution) is not int or not 16 <= resolution <= 512:
            raise ValueError("Geometry raster resolution must be an integer in 16..512.")
        if polydata.GetPoints() is None or not polydata.GetNumberOfPoints():
            raise ValueError("The selected object has no geometry to project.")
        matrix = invertible_affine4(pose)
        points = vtk_to_numpy(polydata.GetPoints().GetData())
        cells = (polydata.GetVerts(), polydata.GetLines(), polydata.GetPolys(), polydata.GetStrips())
        if points.size * 8 + sum(c.GetNumberOfConnectivityIds() + c.GetNumberOfCells() for c in cells) * 8 > MAX_GEOMETRY_BYTES:
            raise ValueError("Geometry capture exceeds 128 MiB. Simplify the mesh before generating its MIP.")
        arrays = []
        for cell_array in cells:
            ids = vtk_to_numpy(cell_array.GetConnectivityArray())
            if ids.size and (ids.min() < 0 or ids.max() >= len(points)):
                raise ValueError("The selected geometry contains invalid point references.")
            packed = vtkIdTypeArray()
            cell_array.ExportLegacyFormat(packed)
            arrays.append(vtk_to_numpy(packed))
        if not np.isfinite(points).all():
            raise ValueError("The selected geometry contains non-finite points.")
        def frozen(array, dtype):
            return np.frombuffer(np.asarray(array, dtype=dtype).tobytes(), dtype=dtype).reshape(array.shape)
        points = frozen(points, "<f8")
        cells = tuple(frozen(a, "<i8") for a in arrays)
        return cls(source_id, points, cells, tuple(map(tuple, matrix)),
                   canonical_json(evidence or {}), parameters, placement, resolution)

    def geometry_checksum(self):
        checksum = hashlib.sha256()
        for array in (self.points,) + self.cells:
            checksum.update(canonical_json(array.shape).encode("ascii"))
            checksum.update(memoryview(array).cast("B"))
        return checksum.hexdigest()

    def prepare(self, cancel=None):
        """Run on a worker with private geometry; never modify a scene object."""
        checkpoint(cancel)
        if not self.parameters.allow_exploratory:
            raise ValueError("Enable generation without verified alignment to project mesh/SWC geometry.")
        if self.placement == "isotropic":
            raise ValueError("The isotropic template preset is for volumes. Choose automatic or current placement for geometry.")
        matrix = np.asarray(self.pose)
        world = self.points @ matrix[:3, :3].T + matrix[:3, 3]
        if not np.isfinite(world).all():
            raise ValueError("The transformed geometry contains non-finite points.")
        low, high = world.min(axis=0), world.max(axis=0)
        span = high - low
        if not np.isfinite(span).all():
            raise ValueError("The geometry bounds exceed finite working coordinates.")
        step = float(span.max()) / (self.resolution - 3) if span.max() > 0 else 1.
        dimensions = np.clip(np.ceil(span / step).astype(int) + 3, 3, self.resolution)
        bounds_low = low - step
        bounds_high = bounds_low + step * (dimensions - 1)
        if not np.isfinite((bounds_low, bounds_high)).all() or np.any(bounds_high <= bounds_low):
            raise ValueError("The geometry cannot be represented on a finite raster grid.")
        bounds = tuple(np.column_stack((bounds_low, bounds_high)).ravel())
        points = vtkPoints()
        points.SetData(numpy_to_vtk(world, deep=True))
        polydata = vtkPolyData()
        polydata.SetPoints(points)
        for setter, packed in zip((polydata.SetVerts, polydata.SetLines, polydata.SetPolys, polydata.SetStrips), self.cells):
            cells = vtkCellArray()
            cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(packed, deep=True))
            setter(cells)
        if not polydata.GetNumberOfCells():
            cells = vtkCellArray()
            for i in range(len(world)):
                if i % 4096 == 0:
                    checkpoint(cancel)
                cells.InsertNextCell(1)
                cells.InsertCellPoint(i)
            polydata.SetVerts(cells)
        voxelizer = vtkVoxelModeller()
        voxelizer.SetInputData(polydata)
        voxelizer.SetModelBounds(bounds)
        voxelizer.SetSampleDimensions(tuple(int(v) for v in dimensions))
        voxelizer.SetScalarTypeToUnsignedChar()
        voxelizer.SetForegroundValue(1)
        voxelizer.SetBackgroundValue(0)
        # Search only near cells, rather than across the full object bounds.
        distance = 2. / self.resolution
        voxelizer.SetMaximumDistance(distance)
        voxelizer.AddObserver("ProgressEvent", lambda *_: voxelizer.SetAbortExecute(bool(cancel and cancel())))
        voxelizer.Update()
        checkpoint(cancel)
        image = voxelizer.GetOutput()
        scalars = image.GetPointData().GetScalars()
        if scalars is None:
            raise ValueError("The geometry could not be rasterized.")
        pixels = vtk_to_numpy(scalars).reshape(tuple(image.GetDimensions())[::-1]).copy()
        pixels.setflags(write=False)
        evidence = json.loads(self.preparation_json)
        warnings = [RASTER_WARNING]
        if evidence.get("input_kind") == "swc-rendered-surface":
            renderer = evidence.get("swc_renderer")
            detail = (f"The surface was built with SWC radii multiplied by {renderer['radius_multiplier']}, "
                      f"{renderer['sphere_segments']} sphere segments, {renderer['sphere_rings']} sphere rings "
                      f"and {renderer['branch_segments']} branch segments."
                      if renderer else "The loaded surface's radius and tessellation settings are unknown.")
            warnings.append("Experimental SWC rendered-surface query uses the displayed mesh; "
                            "it is not a canonical skeleton query. " + detail)
        grid = VolumeWorkingGrid(dimensions=image.GetDimensions(), spacing=image.GetSpacing(),
            origin=image.GetOrigin(), direction=np.eye(3), physical_units=None,
            source_coordinate_space_id=self.source_id, coordinate_mode="working-grid",
            geometry_basis="partially-assumed", physical_grid_state="incomplete-geometry",
            assumed_fields=("units", "coordinate frame"), warnings=tuple(warnings))
        geometry = CDMGeometry(grid, digest(grid.to_dict()), np.eye(4))
        preparation = dict(evidence, algorithm="vtkVoxelModeller-binary/1",
            vtk_version=vtkVersion.GetVTKVersion(), resolution=self.resolution,
            maximum_distance=distance, pose=self.pose, grid=grid.to_dict(), warning=" ".join(warnings),
            world_bounds=[low.tolist(), high.tolist()])
        selection = CDMSelection(self.source_id, self.geometry_checksum(), SourceIdentity(),
            ChannelSelection(self.source_id), 0, "ZYX", preparation_json=canonical_json(preparation))
        value = QueryInput(selection, geometry, pixels)
        mapping = exploratory_mapping(geometry, self.placement, world_bounds=(low, high), profile_id=self.parameters.profile)
        mapping = replace(mapping, assumptions=tuple(warnings) + mapping.assumptions)
        parameters = replace(self.parameters, mode="binary_mask", display_range=None,
                             scalar_range=None, interpolation="nearest")
        return QueryLaunch(value, value, mapping, parameters)
