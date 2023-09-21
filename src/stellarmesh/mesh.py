"""Stellarmesh mesh.

name: mesh.py
author: Alex Koen

desc: Mesh class wraps Gmsh functionality for geometry meshing.
"""
import logging
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import gmsh

from ._core import logger
from .geometry import Geometry


class Mesh:
    """A Gmsh mesh.

    As gmsh allows for only a single process, this class provides a context manager to
    set the gmsh api to operate on this mesh.
    """

    _mesh_filename: str

    def __init__(self, mesh_filename: Optional[str] = None):
        """Initialize a mesh from a .msh file.

        Args:
            mesh_filename: Optional .msh filename. If not provided defaults to a
            temporary file. Defaults to None.
        """
        if not mesh_filename:
            with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as mesh_file:
                mesh_filename = mesh_file.name
        self._mesh_filename = mesh_filename

    def __enter__(self):
        """Enter mesh context, setting gmsh commands to operate on this mesh."""
        if not gmsh.is_initialized():
            gmsh.initialize()

        gmsh.option.setNumber(
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

    def write(self, filename: str, *, save_all: bool = True):
        """Write mesh to a .msh file.

        Args:
            filename: Path to write file.
            save_all: Whether to save all entities (or just physical groups). See
            Gmsh documentation for Mesh.SaveAll. Defaults to True.
        """
        with self:
            gmsh.option.set_number("Mesh.SaveAll", 1 if save_all else 0)
            gmsh.write(filename)

    @classmethod
    def mesh_geometry(
        cls,
        geometry: Geometry,
        min_mesh_size: float = 50,
        max_mesh_size: float = 50,
    ):
        """Mesh solids with Gmsh.

        See Gmsh documentation on mesh sizes:
        https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes

        Args:
            geometry: Geometry to be meshed.
            mesh_filename: Optional filename to store .msh file. Defaults to None.
            min_mesh_size: Min mesh element size. Defaults to 50.
            max_mesh_size: Max mesh element size. Defaults to 50.
        """
        logger.info(f"Meshing solids with mesh size {min_mesh_size}, {max_mesh_size}")

        with cls() as mesh:
            gmsh.model.add("stellarmesh_model")

            material_solid_map = {}
            for s, m in zip(geometry.solids, geometry.material_names):
                dim_tags = gmsh.model.occ.import_shapes_native_pointer(
                    s.wrapped._address()
                )
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

            gmsh.option.set_number("Mesh.MeshSizeMin", min_mesh_size)
            gmsh.option.set_number("Mesh.MeshSizeMax", max_mesh_size)
            gmsh.model.mesh.generate(2)

            mesh._save_changes(save_all=True)
            return mesh

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
        finally:
            for physical_group in physical_groups.items():
                dim, tag = physical_group[0]
                tags, name = physical_group[1]
                gmsh.model.add_physical_group(dim, tags, tag, name)

    # TODO(akoen): support mmgs optim
    def refine(
        self,
        min_mesh_size: float,
        max_mesh_size: float,
        hausdorff_value: float,
        gradation_value: float = 1.3,
    ) -> "Mesh":
        """Refine mesh using mmgs.

        See mmgs documentation:
        https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-tutorials/mmg-remesher-mmg2d/mesh-adaptation-to-a-solution
        for more info.

        Args:
            min_mesh_size: Min size of output mesh.
            max_mesh_size: Max size of output mesh.
            hausdorff_value: Hausdorff value.
            gradation_value: Gradation value. Defaults to 1.3.

        Raises:
            RuntimeError: _description_

        Returns:
            _description_
        """
        with self:
            with self._stash_physical_groups():
                surface_dimtags = gmsh.model.get_entities(2)
                surface_tags = [v[1] for v in surface_dimtags]
                for surface_tag in surface_tags:
                    # triangles = node_tags[0].reshape(-1, 3)
                    # sorted_triangles = np.sort(triangles, axis=1)
                    # edges = sorted_triangles[:, [0, 1, 1, 2, 0, 2]].reshape(-1, 2)
                    # unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
                    # required_edges = unique_edges[counts == 1, :]
                    # print(len(required_edges))

                    gmsh.model.add_physical_group(2, [surface_tag])
                    edge_tags = gmsh.model.get_adjacencies(2, surface_tag)[1]
                    gmsh.model.add_physical_group(1, edge_tags)

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        filename = f"{tmp_dir}/surface_{surface_tag}.mesh"
                        print(filename)
                        gmsh.write(filename)
                        with open(filename, "r+") as f:
                            lines = f.readlines()

                            num_edges = 0
                            for i, line in enumerate(lines):
                                if line.strip() == "Edges":
                                    num_edges = int(lines[i + 1])

                            if num_edges < 1:
                                raise RuntimeError("No Edges.")

                            new_lines = (
                                "RequiredEdges\n"
                                + str(num_edges)
                                + "\n"
                                + "\n".join([str(i + 1) for i in range(num_edges)])
                                + "\n"
                            )

                            lines.insert(-1, new_lines)
                            f.seek(0)
                            f.writelines(lines)

                        refined_filename = str(
                            Path(filename).with_suffix(".o.mesh").resolve()
                        )
                        result = subprocess.run(
                            [
                                "mmgs",
                                "-hmin",
                                str(min_mesh_size),
                                "-hmax",
                                str(max_mesh_size),
                                "-hausd",
                                str(hausdorff_value),
                                "-hgrad",
                                str(gradation_value),
                                "-in",
                                filename,
                                "-out",
                                refined_filename,
                            ],
                            check=False,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                        )
                        print(result.stdout)

                        print(refined_filename)
                        gmsh.model.mesh.clear([(2, surface_tag)])
                        # # gmsh.model.mesh.clear([])
                        gmsh.merge(refined_filename)
                        print(surface_tag)
                        gmsh.model.remove_physical_groups([])
                        # gmsh.model.remove_entities([(2, surface_tag)])

                    # Remove all physical groups

                    # node_tags, _, _ = gmsh.model.mesh.get_edges()
                    # meshutils.write_dotmesh()
                    #     gmsh.model.mesh.set_compound()
                    #     gmsh.merge()
                    ...
                    # gmsh.model.reparametrize_on_surface()

            new_filename = Path(self._mesh_filename).with_suffix(".refined.msh").name
            gmsh.option.set_number("Mesh.SaveAll", 1)
            gmsh.write(new_filename)
            return type(self)(new_filename)
