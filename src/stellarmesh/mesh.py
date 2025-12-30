"""Stellarmesh mesh.

name: mesh.py
author: Alex Koen

desc: Mesh class wraps Gmsh functionality for geometry meshing.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Optional, Union, get_type_hints
from urllib.parse import parse_qs, urlencode

import meshio
import numpy as np

from ._core import PathLike
from .geometry import Geometry

try:
    import gmsh
except ImportError as e:
    raise ImportError(
        "Gmsh not found. See Stellarmesh installation instructions."
    ) from e

try:
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.IMeshTools import (
        IMeshTools_MeshAlgoType_Delabella,
        IMeshTools_MeshAlgoType_Watson,
        IMeshTools_Parameters,
    )
    from OCP.ShapeFix import ShapeFix_ShapeTolerance
    from OCP.TopAbs import TopAbs_FACE, TopAbs_FORWARD
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS, TopoDS_Builder, TopoDS_Compound, TopoDS_Shape
except ImportError as e:
    raise ImportError(
        "OCP not found. See Stellarmesh installation instructions."
    ) from e


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


@dataclass(kw_only=True)
class GmshSurfaceOptions:
    """Gmsh surface meshing options.

    See https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options.

    Attributes:
        min_mesh_size: Min mesh element size.
        max_mesh_size: Max mesh element size.
        curvature_target: Target number of elements per 2pi radians.
        algorithm2d: Gmsh meshing algorithm.
        num_threads: Max number of threads to use when GMSH compiled with OpenMP
        support. 0 for system default i.e. OMP_NUM_THREADS. Defaults to None.
    """

    min_mesh_size: Optional[float] = None
    max_mesh_size: Optional[float] = None
    curvature_target: Optional[float] = None
    algorithm2d: GmshSurfaceAlgo = GmshSurfaceAlgo.AUTOMATIC
    num_threads: Optional[int] = None
    _recombine: bool = False

    def set_options(self):
        """Set corresponding Gmsh options."""
        assert gmsh.is_initialized()

        if self.min_mesh_size:
            gmsh.option.set_number("Mesh.MeshSizeMin", self.min_mesh_size)

        if self.max_mesh_size:
            gmsh.option.set_number("Mesh.MeshSizeMax", self.max_mesh_size)

        if self.curvature_target:
            gmsh.option.set_number("Mesh.MeshSizeFromCurvature", self.curvature_target)

        if self.num_threads:
            gmsh.option.set_number("General.NumThreads", self.num_threads)

        gmsh.option.set_number("Mesh.Algorithm", self.algorithm2d.value)


@dataclass(kw_only=True)
class GmshVolumeOptions(GmshSurfaceOptions):
    """Gmsh volume meshing options.

    See See https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options.

    Attributes:
        algorithm3d: Gmsh volume meshing algorithm.
    """

    algorithm3d: GmshVolumeAlgo = GmshVolumeAlgo.DELAUNAY

    def set_options(self):
        """Set corresponding Gmsh options."""
        super().set_options()
        gmsh.option.set_number("Mesh.Algorithm3D", self.algorithm3d.value)


class OCCSurfaceAlgo(Enum):
    """OCC surface meshing algorithm."""

    WATSON = IMeshTools_MeshAlgoType_Watson
    DELABELLA = IMeshTools_MeshAlgoType_Delabella


@dataclass(kw_only=True)
class OCCSurfaceOptions:
    """OCC surface meshing options.

    Attributes:
        algorithm: The meshing algorithm to use. Defaults to WATSON.
        tol_angular: Angular tolerance for meshing. Defaults to 0.5.
        tol_linear: Linear tolerance for meshing. Defaults to None.
        min_mesh_size: Minimum mesh size. Defaults to None.
        relative: Whether to use relative tolerances. Defaults to False.
        parallel: Whether to use parallel meshing. Defaults to True.
    """

    tol_angular_deg: Optional[float] = 0.5
    tol_linear: Optional[float] = None
    min_mesh_size: Optional[float] = None
    algorithm: OCCSurfaceAlgo = OCCSurfaceAlgo.WATSON
    relative: bool = False
    parallel: bool = True

    def _build_params(self) -> IMeshTools_Parameters:
        """Build IMeshTools Parameters struct from values."""
        # NOTE(akoen)
        # OCC parsing logic at https://github.com/Open-Cascade-SAS/OCCT/blob/783c3440b242277b52f822fde07515bb3aa7c49f/src/ModelingAlgorithms/TKMesh/BRepMesh/BRepMesh_IncrementalMesh.hxx#L105
        # Intro article: https://unlimited3d.wordpress.com/2024/03/17/brepmesh-intro/
        if self.tol_linear is None and self.tol_angular_deg is None:
            raise ValueError(
                "At least one of tol_linear or tol_angular_deg must be set."
            )

        params = IMeshTools_Parameters()
        params.MeshAlgo = self.algorithm.value

        if self.tol_angular_deg:
            params.Angle = self.tol_angular_deg
        else:
            params.Angle = float("inf")

        if self.tol_linear:
            # DeflectionInterior is set to Deflection in OCCT
            params
            # DeflectionInterior is set to Deflection in OCCT
            params.Deflection = self.tol_linear
        else:
            params.Deflection = float("inf")

        # NOTE(akoen): MinSize is by default 0.1 of Deflection---see OCC parsing logic
        # above. If set to less than 1e-7 OCC treats as 0 and ignores.
        if self.min_mesh_size:
            params.MinSize = self.min_mesh_size
        else:
            params.MinSize = 1e-7

        params.Relative = self.relative
        params.InParallel = self.parallel

        return params


@dataclass(kw_only=True)
class EntityMetadata:
    """Metadata for a Mesh elementary entity."""

    _mesh: Mesh = field(init=False)
    _dim: int = field(init=False)
    tag: int = field(init=False)

    def _attach(self, mesh: Mesh, dim: int, tag: int):
        object.__setattr__(self, "_mesh", mesh)
        object.__setattr__(self, "_dim", dim)
        object.__setattr__(self, "tag", tag)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if not name.startswith("_") and getattr(self, "_mesh", None):
            self._mesh.write_metadata(self._dim, self.tag, self)

    def to_str(self) -> str:
        """Converts the dataclass instance to a url string."""
        data = asdict(self)
        public_data = {k: v for k, v in data.items() if not k.startswith("_")}
        return urlencode(public_data)

    @classmethod
    def from_str(cls, url_str: str):
        """Create an EntityMetadata from an encoded string."""
        data = {k: v[0] for k, v in parse_qs(url_str).items()}
        # get_type_hints captures fields from User AND any subclass
        type_hints = get_type_hints(cls)
        init_fields = {f.name for f in fields(cls) if f.init}

        # Validation logic
        for key, expected_type in type_hints.items():
            if key.startswith("_") or key not in init_fields:
                continue
            # Handle Optional types
            is_optional = False
            if (
                hasattr(expected_type, "__origin__")
                and expected_type.__origin__ is Union
            ):
                args = expected_type.__args__
                if type(None) in args:
                    is_optional = True
                    # Assuming Optional[T] is Union[T, NoneType]
                    # We strictly check against the non-None type if present, but for now
                    # just skipping missing check is enough.

            if key not in data:
                if is_optional:
                    continue
                raise ValueError(f"Missing field: {key}")

            # Simple type check (skip for Optional for simplicity or improve logic)
            if data[key] is not None and not is_optional:
                if not isinstance(data[key], expected_type):
                    try:
                        data[key] = expected_type(data[key])
                    except Exception as e:
                        raise TypeError(
                            f"Field '{key}' expected {expected_type}, got {type(data[key])}"
                        ) from e

        return cls(**{k: v for k, v in data.items() if k in init_fields})

    ...


@dataclass(kw_only=True)
class SurfaceMetadata(EntityMetadata):
    """Metadata for a Mesh elementary surface."""

    forward_volume_tag: Optional[int] = None
    reverse_volume_tag: Optional[int] = None

    @property
    def forward_volume(self) -> Optional[int]:
        return self.forward_volume_tag

    @forward_volume.setter
    def forward_volume(self, value: Optional[int]):
        self.forward_volume_tag = value

    @property
    def reverse_volume(self) -> Optional[int]:
        return self.reverse_volume_tag

    @reverse_volume.setter
    def reverse_volume(self, value: Optional[int]):
        self.reverse_volume_tag = value


@dataclass(kw_only=True)
class VolumeMetadata(EntityMetadata):
    """Metadata for a Mesh elementary volume."""


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

    def write(
        self, filename: PathLike, *, save_all: bool = True, use_meshio: bool = False
    ):
        """Write mesh to a file.

        If Gmsh cannot handle the file format, writing will be deferred to meshio.

        Args:
            filename: Path to write file.
            save_all: Whether to save all entities (or just physical groups). See
            Gmsh documentation for Mesh.SaveAll. Defaults to True.
            use_meshio: Write mesh with meshio instead of the builtin exporters.
            Meshio supports more unstructured mesh formats.
        """
        with self:
            gmsh.option.set_number("Mesh.SaveAll", 1 if save_all else 0)
            if use_meshio:
                with tempfile.NamedTemporaryFile(suffix=".msh") as tmp_mesh:
                    gmsh.write(tmp_mesh.name)
                    mesh = meshio.read(tmp_mesh.name)
                    meshio.write(filename, mesh)
            else:
                gmsh.write(str(filename))

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
            gmsh.option.set_number("Mesh.VolumeFaces", 1)
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

    def scaled(self, factor: float) -> Mesh:
        """Return a new mesh scaled by factor.

        Args:
            factor: Scale factor.

        Raises:
            ValueError: If negative scale factor.

        Returns:
            New scaled mesh.
        """
        if factor <= 0:
            raise ValueError("Scale factor must be positive.")
        new_filename = str(
            Path(self._mesh_filename).with_suffix(".scaled.msh").resolve()
        )
        with self:
            gmsh.option.set_number("Mesh.SaveAll", 1)
            gmsh.write(new_filename)
        mesh_scaled = type(self)(new_filename)
        with mesh_scaled:
            dim_tags = gmsh.model.get_entities(2)
            for dim, tag in dim_tags:
                node_tags, coords, _ = gmsh.model.mesh.get_nodes(dim, tag)
                element_types, element_tags, node_tags2 = gmsh.model.mesh.get_elements(
                    dim, tag
                )
                gmsh.model.mesh.clear([(dim, tag)])
                gmsh.model.mesh.add_nodes(dim, tag, node_tags, coords * factor)
                gmsh.model.mesh.add_elements(
                    dim, tag, element_types, element_tags, node_tags2
                )
            mesh_scaled._save_changes()
        return mesh_scaled

    def _get_metadata_group(
        self, dim: int, tag: int, create: bool = True
    ) -> Optional[int]:
        physical_groups = gmsh.model.get_physical_groups_for_entity(dim, tag)
        for pg in physical_groups:
            entities = gmsh.model.get_entities_for_physical_group(dim, pg)
            if len(entities) == 1 and entities[0] == tag:
                return pg
        if not create:
            raise RuntimeError(f"Entity ({dim}, {tag}) has no metadata group")

        pg = gmsh.model.add_physical_group(dim, [tag], tag)
        return pg

    def write_metadata(self, dim: int, tag: int, metadata: EntityMetadata):
        metadata_str = metadata.to_str()
        try:
            metadata_group = self._get_metadata_group(dim, tag, create=False)
            gmsh.model.remove_physical_groups([(dim, metadata_group)])
        except RuntimeError:
            pass

        metadata_group = self._get_metadata_group(dim, tag, create=True)
        gmsh.model.set_physical_name(dim, metadata_group, metadata_str)
        self._save_changes()

    def entity_metadata(self, dim: int, tag: int) -> EntityMetadata:
        """Get metadata for an entity.

        Args:
            dim: Entity dimension.
            tag: Entity tag.

        Returns:
            Entity metadata.
        """
        self._check_is_initialized()
        # physical_groups = gmsh.model.get_physical_groups_for_entity(dim, tag)

        pg = self._get_metadata_group(dim, tag)
        name = gmsh.model.get_physical_name(dim, pg)

        if name == "":
            if dim == 2:
                metadata = SurfaceMetadata()
            elif dim == 3:
                metadata = VolumeMetadata()
            else:
                metadata = EntityMetadata()
        else:
            if dim == 2:
                metadata = SurfaceMetadata.from_str(name)
            elif dim == 3:
                metadata = VolumeMetadata.from_str(name)
            else:
                metadata = EntityMetadata.from_str(name)

        metadata._attach(self, dim, tag)
        return metadata

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
    def _mesh_gmsh(options: GmshSurfaceOptions):
        options.set_options()

        if options._recombine:
            tags = [e[1] for e in gmsh.model.get_entities(2)]
            for tag in tags:
                gmsh.model.mesh.set_recombine(2, tag)

        gmsh.model.mesh.generate(2)

    @classmethod
    def _mesh_occ(cls, geometry: Geometry, options: OCCSurfaceOptions):
        assert gmsh.is_initialized()
        cmp = TopoDS_Compound()
        cmp_builder = TopoDS_Builder()
        cmp_builder.MakeCompound(cmp)

        tolerance_tool = ShapeFix_ShapeTolerance()
        params = options._build_params()
        # These operations must be completed before IncrementalMesh is initialized
        for shape in geometry.solids + geometry.faces:
            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                face = TopoDS.Face_s(explorer.Current())
                # OCC ignores the deflection if the shape tolerance is less than the
                # deflection
                tolerance_tool.LimitTolerance(face, 0, params.Deflection)
                # Remove any existing triangulation on the shape
                BRepTools.Clean_s(face)
                explorer.Next()

            cmp_builder.Add(cmp, shape)

        # NOTE: Gmsh import logic is at
        # https://github.com/live-clones/gmsh/blob/a20dc70a8bb9115185dd6a3b519f6bb3a1aec261/src/geo/GModelIO_OCC.cpp#L715
        BRepMesh_IncrementalMesh(theShape=cmp, theParameters=params)
        loc = TopLoc_Location()
        known_surface_tags = []
        explorer = TopExp_Explorer(cmp, TopAbs_FACE)
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())

            # This returns the existing dim_tag if face is already bound
            dim_tags = cls._import_occ(face, native=True)
            surface_tag = dim_tags[0][1]
            if surface_tag in known_surface_tags:
                logger.debug(f"Surface {surface_tag} already meshed. Skipping.")
                explorer.Next()
                continue
            known_surface_tags.append(surface_tag)

            poly_triangulation = BRep_Tool.Triangulation_s(face, loc)
            trsf = loc.Transformation()

            # Store vertices
            node_count = poly_triangulation.NbNodes()
            ocp_mesh_vertices = np.empty((node_count) * 3)
            offset = 0
            for j in range(node_count):
                gp_pnt = poly_triangulation.Node(j + 1).Transformed(trsf)
                pnt = (gp_pnt.X(), gp_pnt.Y(), gp_pnt.Z())
                ocp_mesh_vertices[3 * j : 3 * (j + 1)] = pnt

            # Store triangles
            order = (
                [1, 2, 3]
                if face.Orientation().value == TopAbs_FORWARD.value
                else [3, 2, 1]
            )
            triangles = np.empty(len(poly_triangulation.Triangles()) * 3)
            for j, tri in enumerate(poly_triangulation.Triangles()):
                tri_pnts = tuple(tri.Value(o) + offset - 1 for o in order)
                triangles[3 * j : 3 * (j + 1)] = tri_pnts
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
                surface_tag, 2, [], triangles + node_start
            )

            explorer.Next()

    @staticmethod
    def _import_occ(shape: TopoDS_Shape, *, native: bool = True) -> list[tuple]:
        if native:
            dim_tags = gmsh.model.occ.import_shapes_native_pointer(shape._address())
            return dim_tags
        with tempfile.NamedTemporaryFile(suffix=".brep") as tmp_file:
            BRepTools.Write_s(shape, tmp_file.name)
            dim_tags = gmsh.model.occ.import_shapes(tmp_file.name)
            return dim_tags

    @classmethod
    def from_geometry(
        cls, geometry: Geometry, options: GmshSurfaceOptions | OCCSurfaceOptions
    ) -> SurfaceMesh:
        """Mesh geometry.

        Args:
            geometry: Geometry to be meshed.
            options: Meshing options.
        """
        with cls() as mesh:
            gmsh.model.add("stellarmesh_model")

            # Solids
            material_solid_map = defaultdict(list)
            for s, m in zip(geometry.solids, geometry.material_names, strict=True):
                dim_tags = cls._import_occ(s)
                if dim_tags[0][0] != 3:
                    raise TypeError("Importing non-solid geometry.")

                solid_tag = dim_tags[0][1]
                material_solid_map[m].append(solid_tag)

            # Faces
            surface_bc_map = defaultdict(list)
            for f, bc in zip(
                geometry.faces, geometry.face_boundary_conditions, strict=True
            ):
                dim_tags = cls._import_occ(f)
                if dim_tags[0][0] != 2:
                    raise TypeError("Importing non-surface geometry.")

                surface_tag = dim_tags[0][1]
                surface_bc_map[bc].append(surface_tag)

            gmsh.model.occ.synchronize()

            assert len(gmsh.model.get_entities(3)) == len(geometry.solids)

            for material, solid_tags in material_solid_map.items():
                gmsh.model.add_physical_group(3, solid_tags, name=f"mat:{material}")
            for bc, surface_tags in surface_bc_map.items():
                gmsh.model.add_physical_group(2, surface_tags, name=f"boundary:{bc}")

            gmsh.model.occ.synchronize()

            known_surfaces = set()
            volume_dimtags = gmsh.model.get_entities(3)
            volume_tags = [v[1] for v in volume_dimtags]
            for i, volume_tag in enumerate(volume_tags):
                _, surface_tags = gmsh.model.get_adjacencies(3, volume_tag)
                for surface_tag in surface_tags:
                    if surface_tag not in known_surfaces:
                        known_surfaces.add(surface_tag)
                        mesh.entity_metadata(2, surface_tag).forward_volume = volume_tag
                    else:
                        mesh.entity_metadata(2, surface_tag).reverse_volume = volume_tag

            if type(options).__name__ == GmshSurfaceOptions.__name__:
                cls._mesh_gmsh(options)  # type: ignore
            elif type(options).__name__ == OCCSurfaceOptions.__name__:
                cls._mesh_occ(geometry, options)  # type: ignore
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

            options.set_options()
            gmsh.model.mesh.generate(3)

            mesh._save_changes(save_all=True)
            return mesh

    def skin(self) -> SurfaceMesh:
        """Transform a tetrahedral volume mesh into a triangular surface mesh."""
        surface_mesh = SurfaceMesh()
        self.write(surface_mesh._mesh_filename)

        with surface_mesh:
            dim_tags = gmsh.model.get_entities(3)
            gmsh.model.mesh.clear(dim_tags)
            surface_mesh._save_changes()

        return surface_mesh
