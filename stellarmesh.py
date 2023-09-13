"""GMSH wrapper and DAGMC geometry creator."""
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Self, Sequence, Union

import build123d as bd
import gmsh
import numpy as np
from deprecated import deprecated
from pymoab import core, types  # types: ignore

logger = logging.getLogger(__name__)


@dataclass
class MOABSurface:
    """Represents a MOAB surface."""

    handle: np.uint64
    forward_volume: np.uint64 = field(default=np.uint64(0))
    reverse_volume: np.uint64 = field(default=np.uint64(0))

    def sense_data(self):
        """Get MOAB tag sense data.

        Returns:
            Sense data.
        """
        return [self.forward_volume, self.reverse_volume]


class Geometry:
    """Geometry, representing a list of solids, to be meshed."""

    solids: Sequence[bd.Solid]

    def __init__(self, solids: Sequence[bd.Solid]):
        """Construct geometry from solids.

        Args:
            solids: Solids.
        """
        logger.info(f"Importing {len(solids)} to geometry")
        self.solids = solids

    @classmethod
    def import_step(
        cls,
        filename: str,
    ) -> Self:
        """Import model from a step file.

        Args:
            filename: File path to import.

        Returns:
            Model.
        """
        geometry = bd.import_step(filename)
        solids = geometry.solids()
        logger.info(f"Importing {len(solids)} from {filename}")
        return cls(solids)

    @classmethod
    def import_brep(
        cls,
        filename: str,
    ) -> Self:
        """Import model from a brep (cadquery, build123d native) file.

        Args:
            filename: File path to import.

        Returns:
            Model.
        """
        geometry = bd.import_brep(filename)
        solids = geometry.solids()
        logger.info(f"Importing {len(solids)} from {filename}")
        return cls(solids)


class Mesh:
    """Mesh."""

    mesh_filename: str

    def __init__(self, mesh_filename: str):
        """Initialize a mesh from a .msh file.

        Args:
            mesh_filename: Gmsh .msh filename.
        """
        self.mesh_filename = mesh_filename

    @classmethod
    def mesh_geometry(
        cls,
        geometry: Geometry,
        mesh_filename: Optional[str] = None,
        min_mesh_size: float = 50,
        max_mesh_size: float = 50,
    ):
        """Mesh solids with Gmsh.

        Args:
            geometry: Geometry to be meshed.
            mesh_filename: Optional filename to store .msh file. Defaults to None.
            min_mesh_size: Min mesh element size. Defaults to 50.
            max_mesh_size: Max mesh element size. Defaults to 50.
        """
        logger.info(f"Meshing solids with mesh size {min_mesh_size}, {max_mesh_size}")

        try:
            gmsh.initialize()
            gmsh.option.setNumber(
                "General.Terminal",
                1 if logger.getEffectiveLevel() <= logging.INFO else 0,
            )
            gmsh.model.add("stellarmesh_model")

            cmp = bd.Compound.make_compound(geometry.solids)
            gmsh.model.occ.import_shapes_native_pointer(cmp.wrapped._address())
            gmsh.model.occ.synchronize()

            gmsh.option.set_number("Mesh.MeshSizeMin", min_mesh_size)
            gmsh.option.set_number("Mesh.MeshSizeMax", max_mesh_size)

            gmsh.model.mesh.generate(2)

            if not mesh_filename:
                with tempfile.NamedTemporaryFile(
                    suffix=".msh", delete=False
                ) as mesh_file:
                    mesh_filename = mesh_file.name

            gmsh.write(mesh_filename)

            return cls(mesh_filename)
        finally:
            gmsh.finalize()

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
        try:
            gmsh.initialize()
            gmsh.fltk.initialize()

            gmsh.option.setNumber(
                "General.Terminal",
                1 if logger.getEffectiveLevel() <= logging.INFO else 0,
            )
            gmsh.merge(self.mesh_filename)
            gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
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

            gmsh.write(output_filename)
            return output_filename

        finally:
            gmsh.fltk.finalize()
            gmsh.finalize()


class DAGMCGeometry:
    """MOAB DAGMC geometry."""

    h5m_filename: str

    def __init__(self, h5m_filename: str):
        """Initialize a mesh from a .msh file.

        Args:
            h5m_filename: DAGMC .h5m filename.
        """
        self.h5m_filename = h5m_filename

    @deprecated(reason="Prefer STL export/import.")
    def _moab_add_vertices_manual_inefficient(
        self, moab_core, all_node_coords, gmsh_surface_tag: int, surface: MOABSurface
    ):
        """Add vertices (very inefficiently) to MOAB.

        Poorly implemented as all vertices are added to each surface.
        """
        _, _, node_tags = gmsh.model.mesh.get_elements(2, gmsh_surface_tag)
        # We only have a single element type
        node_tags = node_tags[0]
        # node_tags is of form [t11, t12, t13, t21, t22, t23, ...]
        # gmsh uses 1-based indexing for tags
        node_tags = [tag - 1 for tag in node_tags]

        # Add vertices to MOAB core
        # coords can be 1D array with 3*n or 2d array
        moab_verts = moab_core.create_vertices(all_node_coords)
        moab_core.add_entity(surface.handle, moab_verts)

        # Add triangles to MOAB core
        triangles = np.array(node_tags).reshape(-1, 3)
        for triangle in triangles:
            tri = [moab_verts[int(triangle[i])] for i in range(3)]
            moab_triangle = moab_core.create_element(types.MBTRI, tri)
            moab_core.add_entity(surface.handle, moab_triangle)

    @staticmethod
    def _make_watertight(
        input_filename: str,
        output_filename: str,
        binary_path: str = "make_watertight",
    ):
        subprocess.run(
            [binary_path, input_filename, "-o", output_filename],  # noqa
            check=True,
        )

    @classmethod
    def make_from_mesh(  # noqa: PLR0915
        cls,
        mesh: Mesh,
        material_names: Sequence[str],
        filename: str = "dagmc.h5m",
        *,
        make_watertight: Union[bool, str] = False,
    ):
        """Compose DAGMC MOAB .h5m file from mesh.

        Args:
            mesh: Mesh from which to build DAGMC geometry.
            material_names: Ordered list of material names matching number of
            solids/volumes in Geometry/Mesh.
            filename: Filename of the output .h5m file.
            make_watertight: Whether to run make_watertight on the produced file. If
            True find make_watertight in PATH. If string use the provided
            make_watertight binary path. Defaults to False.
        """
        moab_core = core.Core()

        tag_handles = {}

        sense_tag_name = "GEOM_SENSE_2"
        sense_tag_size = 2
        tag_handles["surf_sense"] = moab_core.tag_get_handle(
            sense_tag_name,
            sense_tag_size,
            types.MB_TYPE_HANDLE,
            types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

        tag_handles["category"] = moab_core.tag_get_handle(
            types.CATEGORY_TAG_NAME,
            types.CATEGORY_TAG_SIZE,
            types.MB_TYPE_OPAQUE,
            types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

        tag_handles["name"] = moab_core.tag_get_handle(
            types.NAME_TAG_NAME,
            types.NAME_TAG_SIZE,
            types.MB_TYPE_OPAQUE,
            types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

        # TODO(akoen): C2D and C2O set tag type to SPARSE, while cubit plugin is DENSE
        # https://github.com/Thea-Energy/stellarmesh/issues/1
        geom_dimension_tag_size = 1
        tag_handles["geom_dimension"] = moab_core.tag_get_handle(
            types.GEOM_DIMENSION_TAG_NAME,
            geom_dimension_tag_size,
            types.MB_TYPE_INTEGER,
            types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

        faceting_tol_tag_name = "FACETING_TOL"
        faceting_tol_tag_size = 1
        tag_handles["faceting_tol"] = moab_core.tag_get_handle(
            faceting_tol_tag_name,
            faceting_tol_tag_size,
            types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

        # Default tag, does not need to be created
        tag_handles["global_id"] = moab_core.tag_get_handle(types.GLOBAL_ID_TAG_NAME)

        known_surfaces: dict[int, MOABSurface] = {}
        try:
            gmsh.initialize()
            gmsh.option.setNumber(
                "General.Terminal",
                1 if logger.getEffectiveLevel() <= logging.INFO else 0,
            )
            gmsh.open(mesh.mesh_filename)

            volume_dimtags = gmsh.model.get_entities(3)
            volume_tags = [v[1] for v in volume_dimtags]
            if len(volume_dimtags) != len(material_names):
                raise ValueError(
                    "Number of volumes does not match number of material names"
                )
            for i, volume_tag in enumerate(volume_tags):
                volume_set_handle = moab_core.create_meshset()

                moab_core.tag_set_data(tag_handles["global_id"], volume_set_handle, i)

                moab_core.tag_set_data(
                    tag_handles["geom_dimension"], volume_set_handle, 3
                )
                moab_core.tag_set_data(
                    tag_handles["category"], volume_set_handle, "Volume"
                )

                # Add the group set, which stores metadata about the volume
                group_set = moab_core.create_meshset()
                moab_core.tag_set_data(tag_handles["category"], group_set, "Group")
                # TODO(akoen): support other materials
                # https://github.com/Thea-Energy/neutronics-cad/issues/3
                moab_core.tag_set_data(
                    tag_handles["name"], group_set, f"mat:{material_names[i]}"
                )
                moab_core.tag_set_data(tag_handles["geom_dimension"], group_set, 4)
                # TODO(akoen): should this be a parent-child relationship?
                # https://github.com/Thea-Energy/neutronics-cad/issues/2
                moab_core.add_entity(group_set, volume_set_handle)

                adjacencies = gmsh.model.get_adjacencies(3, volume_tag)
                surface_tags = adjacencies[1]
                for surface_tag in surface_tags:
                    if surface_tag not in known_surfaces:
                        surface_set_handle = moab_core.create_meshset()
                        surface = MOABSurface(handle=surface_set_handle)
                        surface.forward_volume = volume_set_handle
                        known_surfaces[surface_tag] = surface

                        moab_core.tag_set_data(
                            tag_handles["global_id"], surface.handle, surface_tag
                        )
                        moab_core.tag_set_data(
                            tag_handles["geom_dimension"], surface.handle, 2
                        )
                        moab_core.tag_set_data(
                            tag_handles["category"], surface.handle, "Surface"
                        )
                        moab_core.tag_set_data(
                            tag_handles["surf_sense"],
                            surface.handle,
                            surface.sense_data(),
                        )

                        with tempfile.NamedTemporaryFile(
                            suffix=".stl", delete=True
                        ) as stl_file:
                            gmsh.model.add_physical_group(2, [surface_tag])
                            gmsh.write(stl_file.name)
                            gmsh.model.remove_physical_groups()
                            moab_core.load_file(stl_file.name, surface_set_handle)

                    else:
                        # Surface already has a forward volume, so this must be the
                        # reverse volume.
                        surface = known_surfaces[surface_tag]
                        surface.reverse_volume = volume_set_handle
                        moab_core.tag_set_data(
                            tag_handles["surf_sense"],
                            surface.handle,
                            surface.sense_data(),
                        )

                    moab_core.add_parent_child(volume_set_handle, surface.handle)

            all_entities = moab_core.get_entities_by_handle(0)
            file_set = moab_core.create_meshset()
            # TODO(akoen): faceting tol set to a random value
            # https://github.com/Thea-Energy/neutronics-cad/issues/5
            # faceting_tol required to be set for make_watertight, although its
            # significance is not clear
            moab_core.tag_set_data(tag_handles["faceting_tol"], file_set, 1e-3)
            moab_core.add_entities(file_set, all_entities)

            with tempfile.NamedTemporaryFile(suffix=".tmp.h5m") as tmp_file:
                moab_core.write_file(tmp_file.name)
                if make_watertight:
                    watertight_path = (
                        make_watertight
                        if isinstance(make_watertight, str)
                        else "make_watertight"
                    )
                    cls._make_watertight(tmp_file.name, filename, watertight_path)
                else:
                    shutil.copy(tmp_file.name, filename)

                logger.info(f"Wrote MOAB mesh to {filename}")

                return cls(filename)
        finally:
            gmsh.finalize()
