"""Stellarmesh mesh.

name: mesh.py
author: Alex Koen

desc: Mesh class wraps Gmsh functionality for geometry meshing.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol

import gmsh
import meshio
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IMeshTools import (
    IMeshTools_MeshAlgoType_Delabella,
    IMeshTools_MeshAlgoType_Watson,
    IMeshTools_Parameters,
)
from OCP.TopAbs import TopAbs_FACE, TopAbs_FORWARD
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Builder, TopoDS_Compound

from ._core import PathLike
from .geometry import Geometry

logger = logging.getLogger(__name__)


class GmshSurfaceAlgo(Enum):
    """Algorithm used by Gmsh for surface meshing."""

    MESH_ADAPT = 1
    AUTOMATIC = 2
    INITIAL_ONLY = 3
    DELAUNAY = 5
    FRONTAL_DELAUNAY = 6
    BAMG = 7
    FRONTAL_DELAUNAY_QUADS = 8
    PACKING_OF_PARALLELOGRAMS = 9
    QUASI_STRUCTURED_QUAD = 11


class GmshVolumeAlgo(Enum):
    """Algorithm used by Gmsh for volume meshing."""

    DELAUNAY = 1
    INITIAL_ONLY = 3
    FRONTAL = 4
    MMG3D = 7
    R_TREE = 9
    HXT = 10


@dataclass
class GmshMeshingOptions(Protocol):
    """Gmsh generic meshing options.

    See https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes for
    parameter descriptions.

    Attributes:
        min_mesh_size: Min element size
        max_mesh_size: Max element size
    """

    min_mesh_size: float
    max_mesh_size: float


@dataclass
class GmshSurfaceOptions(GmshMeshingOptions):
    """Gmsh surface meshing options.

    See https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options.

    Attributes:
        min_mesh_size: Min mesh element size. Defaults to 50.
        max_mesh_size: Max mesh element size. Defaults to 50.
        algorithm: Gmsh meshing algorithm.
    """

    min_mesh_size: float
    max_mesh_size: float
    algorithm: GmshSurfaceAlgo = GmshSurfaceAlgo.AUTOMATIC


@dataclass
class GmshVolumeOptions(GmshMeshingOptions):
    """Gmsh volume meshing options.

    Attributes:
        algorithm: Gmsh volume meshing algorithm.
    """

    min_mesh_size: float
    max_mesh_size: float
    algorithm: GmshVolumeAlgo = GmshVolumeAlgo.DELAUNAY


class OCCSurfaceAlgo(Enum):
    """OCC surface meshing algorithm."""

    WATSON = IMeshTools_MeshAlgoType_Watson
    DELABELLA = IMeshTools_MeshAlgoType_Delabella


@dataclass
class OCCSurfaceOptions:
    """OCC surface meshing options."""

    algorithm = OCCSurfaceAlgo.WATSON
    tol_angular: float = 0.5
    tol_linear: Optional[float] = None
    min_mesh_size: Optional[float] = None
    relative: bool = False

    def build_params(self) -> IMeshTools_Parameters:
        """Build IMeshTools Parameters struct from values."""
        params = IMeshTools_Parameters()

        params.MeshAlgo = self.algorithm.value

        if self.tol_angular:
            params.Angle = self.tol_angular
        if self.tol_linear:
            params.Deflection = self.tol_linear

        params.Relative = self.relative
        params.InParallel = True

        return params


class Mesh:
    """A Gmsh mesh.

    As Gmsh allows for only a single process, this class provides a context manager to
    set the Gmsh API to operate on this mesh.
    """

    _mesh_filename: str

    def __init__(self, mesh_filename: Optional[PathLike] = None):
        """Initialize a mesh from a .msh file.

        Args:
            mesh_filename: Optional .msh filename. If not provided defaults to a
            temporary file. Defaults to None.
        """
        if not mesh_filename:
            with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as mesh_file:
                mesh_filename = mesh_file.name
        self._mesh_filename = str(mesh_filename)

    def __enter__(self):
        """Enter mesh context, setting gmsh commands to operate on this mesh."""
        if not gmsh.is_initialized():
            gmsh.initialize()

        gmsh.option.set_number(
            "General.Terminal",
            1 if logger.getEffectiveLevel() <= logging.INFO else 0,
        )
        gmsh.open(self._mesh_filename)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup (finalize) gmsh."""
        gmsh.finalize()

    def _save_changes(self, *, save_all: bool = True):
        gmsh.option.set_number("Mesh.SaveAll", 1 if save_all else 0)
        gmsh.write(self._mesh_filename)

    def write(self, filename: PathLike, *, save_all: bool = True):
        """Write mesh to a .msh file.

        Args:
            filename: Path to write file.
            save_all: Whether to save all entities (or just physical groups). See
            Gmsh documentation for Mesh.SaveAll. Defaults to True.
        """
        with self:
            gmsh.option.set_number("Mesh.SaveAll", 1 if save_all else 0)
            try:
                gmsh.write(filename)
            except Exception:
                with tempfile.NamedTemporaryFile(suffix=".msh") as tmp_mesh:
                    gmsh.write(tmp_mesh.name)
                    mesh = meshio.read(tmp_mesh.name)
                    meshio.write(filename, mesh)

    def render(
        self,
        output_filename: Optional[str] = None,
        rotation_xyz: tuple[float, float, float] = (0, 0, 0),
        normals: int = 0,
        *,
        clipping: bool = True,
    ) -> str:
        """Render mesh as an image.

        Args:
            output_filename: Optional output filename. Defaults to None.
            rotation_xyz: Rotation in Euler angles. Defaults to (0, 0, 0).
            normals: Normal render size. Defaults to 0.
            clipping: Whether to enable mesh clipping. Defaults to True.

        Returns:
            Path to image file, either passed output_filename or a temporary file.
        """
        with self:
            gmsh.option.set_number("Mesh.SurfaceFaces", 1)
            gmsh.option.set_number("Mesh.Clip", 1 if clipping else 0)
            gmsh.option.set_number("Mesh.Normals", normals)
            gmsh.option.set_number("General.Trackball", 0)
            gmsh.option.set_number("General.RotationX", rotation_xyz[0])
            gmsh.option.set_number("General.RotationY", rotation_xyz[1])
            gmsh.option.set_number("General.RotationZ", rotation_xyz[2])
            if not output_filename:
                with tempfile.NamedTemporaryFile(
                    delete=False, mode="w", suffix=".png"
                ) as temp_file:
                    output_filename = temp_file.name

            try:
                gmsh.fltk.initialize()
                gmsh.write(output_filename)
            finally:
                gmsh.fltk.finalize()
            return output_filename

    @staticmethod
    def _check_is_initialized():
        if not gmsh.is_initialized():
            raise RuntimeError("Gmsh not initialized.")

    @contextmanager
    def _stash_physical_groups(self):
        self._check_is_initialized()
        physical_groups: dict[tuple[int, int], tuple[list[int], str]] = {}
        dim_tags = gmsh.model.get_physical_groups()
        for dim_tag in dim_tags:
            tags = gmsh.model.get_entities_for_physical_group(*dim_tag)
            name = gmsh.model.get_physical_name(*dim_tag)
            physical_groups[dim_tag] = (tags, name)
        gmsh.model.remove_physical_groups(dim_tags)

        try:
            yield
        except Exception as e:
            raise RuntimeError("Cannot unstash physical groups due to error.") from e
        else:
            if len(gmsh.model.get_physical_groups()) > 0:
                raise RuntimeError(
                    "Not overwriting existing physical groups on stash restore."
                )
            for physical_group in physical_groups.items():
                dim, tag = physical_group[0]
                tags, name = physical_group[1]
                gmsh.model.add_physical_group(dim, tags, tag, name)


class SurfaceMesh(Mesh):
    """A surface mesh."""

    @staticmethod
    def _mesh_gmsh(options: GmshSurfaceOptions | OCCSurfaceOptions):
        assert gmsh.is_initialized()
        gmsh.option.set_number("Mesh.MeshSizeMin", options.min_mesh_size)
        gmsh.option.set_number("Mesh.MeshSizeMax", options.max_mesh_size)  # type: ignore
        gmsh.option.set_number("Mesh.Algorithm", options.algorithm.value)
        gmsh.model.mesh.generate(2)

    @staticmethod
    def _mesh_occ(geometry: Geometry, options: GmshSurfaceOptions | OCCSurfaceOptions):
        assert gmsh.is_initialized
        cmp = TopoDS_Compound()
        cmp_builder = TopoDS_Builder()
        cmp_builder.MakeCompound(cmp)

        for shape in geometry.solids:
            cmp_builder.Add(cmp, shape)

        params = options.build_params()  # type: ignore

        BRepMesh_IncrementalMesh(theShape=cmp, theParameters=params)

        loc = TopLoc_Location()
        explorer = TopExp_Explorer(cmp, TopAbs_FACE)
        faces = []
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            faces.append(face)
            explorer.Next()

        # NOTE: Gmsh import logic is at
        # https://github.com/live-clones/gmsh/blob/a20dc70a8bb9115185dd6a3b519f6bb3a1aec261/src/geo/GModelIO_OCC.cpp#L715
        known_surface_tags = []
        for face in faces:
            dim_tags = gmsh.model.occ.import_shapes_native_pointer(face._address())
            surface_tag = dim_tags[0][1]
            if surface_tag in known_surface_tags:
                continue
            known_surface_tags.append(surface_tag)
            ocp_mesh_vertices = []
            triangles = []
            offset = 0

            poly_triangulation = BRep_Tool.Triangulation_s(face, loc)
            trsf = loc.Transformation()
            # Store vertices
            node_count = poly_triangulation.NbNodes()
            for j in range(1, node_count + 1):
                gp_pnt = poly_triangulation.Node(j).Transformed(trsf)
                pnt = (gp_pnt.X(), gp_pnt.Y(), gp_pnt.Z())
                ocp_mesh_vertices.extend(pnt)

            # Store triangles
            order = (
                [1, 2, 3]
                if face.Orientation().value == TopAbs_FORWARD.value
                else [3, 2, 1]
            )
            for tri in poly_triangulation.Triangles():
                triangles.extend([tri.Value(i) + offset - 1 for i in order])
            offset += node_count
            gmsh.model.mesh.add_nodes(
                2,
                surface_tag,
                [],
                ocp_mesh_vertices,
            )
            nodes, _, _ = gmsh.model.mesh.get_nodes(
                2, surface_tag, includeBoundary=True
            )
            node_start = nodes[0]
            gmsh.model.mesh.add_elements_by_type(
                surface_tag, 2, [], [node_start + i for i in triangles]
            )

    @classmethod
    def from_geometry(
        cls, geometry: Geometry, options: GmshSurfaceOptions | OCCSurfaceOptions
    ) -> SurfaceMesh:
        """Mesh solids with Gmsh.

        See Gmsh documentation on mesh sizes:
        https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes

        Args:
            geometry: Geometry to be meshed.
            options: Meshing options.
        """
        with cls() as mesh:
            gmsh.model.add("stellarmesh_model")

            material_solid_map = {}
            for s, m in zip(geometry.solids, geometry.material_names, strict=True):
                dim_tags = gmsh.model.occ.import_shapes_native_pointer(s._address())
                if dim_tags[0][0] != 3:
                    raise TypeError("Importing non-solid geometry.")

                solid_tag = dim_tags[0][1]
                if m not in material_solid_map:
                    material_solid_map[m] = [solid_tag]
                else:
                    material_solid_map[m].append(solid_tag)

            gmsh.model.occ.synchronize()

            for material, solid_tags in material_solid_map.items():
                gmsh.model.add_physical_group(3, solid_tags, name=f"mat:{material}")

            if type(options).__name__ == GmshSurfaceOptions.__name__:
                cls._mesh_gmsh(options)
            elif type(options).__name__ == OCCSurfaceOptions.__name__:
                cls._mesh_occ(geometry, options)
            else:
                raise RuntimeError(
                    "Unreachable code. May be caused by module hot-reloading."
                )

            mesh._save_changes(save_all=True)
            return mesh

    def refine(  # noqa: PLR0913
        self,
        *,
        min_mesh_size: Optional[float] = None,
        max_mesh_size: Optional[float] = None,
        const_mesh_size: Optional[float] = None,
        hausdorff_value: float = 0.01,
        gradation_value: float = 1.3,
        optim: bool = False,
    ) -> Mesh:
        """Refine mesh using mmgs.

        See mmgs documentation:
        https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-tutorials/mmg-remesher-mmg2d/mesh-adaptation-to-a-solution
        for more info.

        Pay particular attention to the hausdorff value, which overrides most of the
        other options and is typically set too low. Set to a large value, on the order
        of the size of your bounding box, to disable completely.

        Args:
            min_mesh_size: -hmin: Min size of output mesh elements. Defaults to None.
            max_mesh_size: -hmax: Max size of output mesh elements. Defaults to None.
            const_mesh_size: -hsize: Constant size map
            hausdorff_value: -hausd: Hausdorff value. Defaults to 0.01, which is
            suitable for a circle of radius 1. Set to a large value to disable effect.
            gradation_value: -hgrad Gradation value. Defaults to 1.3.
            optim: -optim Do not change elements sizes. Defaults to False.

        Raises:
            RuntimeError: If refinement fails.

        Returns:
            New refined mesh with filename <original-filename>.refined.msh.
        """
        with (
            self,
            self._stash_physical_groups(),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            filename = f"{tmpdir}/model.mesh"
            gmsh.write(filename)

            refined_filename = str(Path(filename).with_suffix(".o.mesh").resolve())
            command = ["mmgs"]

            params = {
                "-hmin": min_mesh_size,
                "-hmax": max_mesh_size,
                "-hsiz": const_mesh_size,
                "-hausd": hausdorff_value,
                "-hgrad": gradation_value,
            }

            for param in params.items():
                if param[1]:
                    command.extend([param[0], str(param[1])])
            if optim:
                command.append("-optim")

            command.extend(
                [
                    "-in",
                    filename,
                    "-out",
                    refined_filename,
                ]
            )

            # TODO(akoen): log subprocess realtime
            # https://github.com/Thea-Energy/stellarmesh/issues/13
            try:
                logger.info(
                    f"Refining mesh {filename} with mmgs, output to {refined_filename}."
                )
                output = subprocess.run(
                    command,
                    text=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )

                if output.stdout:
                    logger.info(output.stdout)

            except subprocess.CalledProcessError as e:
                logger.exception(
                    "Command failed with error code %d\nSTDERR:%s",
                    e.returncode,
                    e.stdout,
                )
                raise RuntimeError("Command failed to run. See output above.") from e

            gmsh.model.mesh.clear()
            gmsh.merge(refined_filename)

            new_filename = str(
                Path(self._mesh_filename).with_suffix(".refined.msh").resolve()
            )
            gmsh.option.set_number("Mesh.SaveAll", 1)
            gmsh.write(new_filename)
            return type(self)(new_filename)


class VolumeMesh(Mesh):
    """Volume Mesh."""

    @classmethod
    def from_geometry(
        cls, geometry: Geometry, options: GmshVolumeOptions
    ) -> VolumeMesh:
        """Mesh solids with Gmsh.

        See Gmsh documentation on mesh sizes:
        https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes

        Args:
            geometry: Geometry to be meshed.
            options: Meshing options.
        """
        with cls() as mesh:
            gmsh.model.add("stellarmesh_model")

            for s in geometry.solids:
                dim_tags = gmsh.model.occ.import_shapes_native_pointer(s._address())
                if dim_tags[0][0] != 3:
                    raise TypeError("Importing non-solid geometry.")

            gmsh.model.occ.synchronize()

            gmsh.option.set_number("Mesh.MeshSizeMin", options.min_mesh_size)
            gmsh.option.set_number("Mesh.MeshSizeMax", options.max_mesh_size)
            gmsh.option.set_number("Mesh.Algorithm", options.algorithm.value)
            gmsh.model.mesh.generate(3)

            mesh._save_changes(save_all=True)
            return mesh

    def skin(self) -> SurfaceMesh:
        """Transform a tetrahedral volume mesh into a triangular surface mesh.

        Returns: the new surface mesh.
        """
        surface_mesh = SurfaceMesh()
        self.write(surface_mesh._mesh_filename)

        with surface_mesh:
            dim_tags = gmsh.model.get_entities(3)
            gmsh.model.mesh.clear(dim_tags)
            surface_mesh._save_changes()

        return surface_mesh
