"""Stellarmesh MOAB/DAGMC models.

name: moab.py
author: Alex Koen, Paul Romano

desc: MOABModel class represents a MOAB model.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import warnings
from functools import cached_property
from typing import Final, Literal, Optional, Sequence, Union

import h5py
import numpy as np

from ._core import PathLike
from .mesh import Mesh, SurfaceMetadata

try:
    import gmsh
except ImportError as e:
    raise ImportError(
        "Gmsh not found. See Stellarmesh installation instructions."
    ) from e


try:
    import pymoab.core
    import pymoab.tag
    import pymoab.types
    from pymoab.rng import Range
except ImportError as e:
    raise ImportError(
        "PyMOAB not found. See Stellarmesh installation instructions."
    ) from e

logger = logging.getLogger(__name__)


class EntitySet:
    """A MOAB entity set."""

    model: MOABModel
    handle: np.uint64

    def __init__(self, model: MOABModel, handle: np.uint64):
        """Initialize entity set.

        Args:
            model: MOAB model
            handle: Handle of entity set
        """
        self.model = model
        self.handle = handle

    def __eq__(self, other) -> bool:
        """Compare this entity with another."""
        return self.handle == other.handle

    def __hash__(self) -> int:
        """Return hash of entity set's handle."""
        return hash(self.handle)

    def __repr__(self) -> str:
        """String representation of entity set."""
        return f"<{type(self).__name__}(id={self.global_id})>"

    def _tag_get_data(self, tag: pymoab.tag.Tag):
        return self.model._core.tag_get_data(tag, self.handle, flat=True)[0]

    def _tag_set_data(self, tag: pymoab.tag.Tag, value):
        self.model._core.tag_set_data(tag, self.handle, value)

    @property
    def category(self) -> str:
        """Category for entity set."""
        return self._tag_get_data(self.model.category_tag)

    @category.setter
    def category(self, category: str):
        self._tag_set_data(self.model.category_tag, category)

    @property
    def global_id(self) -> int:
        """Global ID."""
        return self._tag_get_data(self.model.id_tag)

    @global_id.setter
    def global_id(self, value: int):
        self._tag_set_data(self.model.id_tag, value)

    @property
    def geom_dimension(self) -> int:
        """Geometry dimension."""
        return self._tag_get_data(self.model.geom_dimension_tag)

    @geom_dimension.setter
    def geom_dimension(self, dimension: int):
        self._tag_set_data(self.model.geom_dimension_tag, dimension)


class DAGMCGroup(EntitySet):
    """A DAGMC Group (used to assign material metadata)."""

    def __contains__(self, entity_set: EntitySet) -> bool:
        """Determine whether group contains a given entity set."""
        return any(vol.handle == entity_set.handle for vol in self.volumes)

    def __repr__(self) -> str:
        """String representation of group."""
        return f"<DAGMCGroup: {self.name})>"

    @property
    def name(self) -> str:
        """Name of the group."""
        model = self.model
        return model._core.tag_get_data(model.name_tag, self.handle, flat=True)[0]

    @name.setter
    def name(self, value: str):
        self.model._core.tag_set_data(self.model.name_tag, self.handle, value)

    @property
    def volumes(self) -> list[DAGMCVolume]:
        """Get list of volumes contained in this group."""
        handles: Range = self.model._core.get_entities_by_type_and_tag(
            self.handle, pymoab.types.MBENTITYSET, [self.model.category_tag], ["Volume"]
        )
        return [DAGMCVolume(self.model, handle) for handle in handles]

    @property
    def surfaces(self) -> list[DAGMCSurface]:
        """Get list of surfaces contained in this group."""
        handles: Range = self.model._core.get_entities_by_type_and_tag(
            self.handle,
            pymoab.types.MBENTITYSET,
            [self.model.category_tag],
            ["Surface"],
        )
        return [DAGMCSurface(self.model, handle) for handle in handles]

    def add(self, entity_set: EntitySet):
        """Add entity set to the group.

        Args:
            entity_set: Entity set to add
        """
        self.model._core.add_entity(self.handle, entity_set.handle)

    def remove(self, entity_set: EntitySet):
        """Remove entity set from the group.

        Args:
            entity_set: Entity set to remove
        """
        self.model._core.remove_entity(self.handle, entity_set.handle)


class DAGMCEntitySet(EntitySet):
    """An entity set for a DAGMC topological surface or volume."""

    model: DAGMCModel

    @property
    def groups(self) -> list[DAGMCGroup]:
        """Get list of groups containing this volume."""
        return [group for group in self.model.groups if self in group]


class DAGMCCurve(DAGMCEntitySet):
    """DAGMC curve entity."""

    @property
    def adjacent_surfaces(self) -> list[DAGMCSurface]:
        """Get adjacent surfaces.

        Returns:
            Adjacent surfaces.
        """
        parent_entities = self.model._core.get_parent_meshsets(self.handle)
        return [DAGMCSurface(self.model, e) for e in parent_entities]


class DAGMCSurface(DAGMCEntitySet):
    """DAGMC surface entity."""

    @property
    def forward_volume(self) -> Optional[DAGMCVolume]:
        """Volume with forward sense with respect to the surface."""
        return self.surf_sense[0]

    @forward_volume.setter
    def forward_volume(self, volume: DAGMCVolume):
        self.surf_sense = (volume, self.reverse_volume)

    @property
    def reverse_volume(self) -> Optional[DAGMCVolume]:
        """Volume with reverse sense with respect to the surface."""
        return self.surf_sense[1]

    @reverse_volume.setter
    def reverse_volume(self, volume: DAGMCVolume):
        self.surf_sense = (self.forward_volume, volume)

    @property
    def surf_sense(self) -> list[Optional[DAGMCVolume]]:
        """Surface sense data."""
        try:
            handles = self.model._core.tag_get_data(
                self.model.surf_sense_tag, self.handle, flat=True
            )
        except RuntimeError:
            return [None, None]

        return [
            DAGMCVolume(self.model, handle) if handle != 0 else None
            for handle in handles
        ]

    @surf_sense.setter
    def surf_sense(self, volumes: tuple[Optional[DAGMCVolume], Optional[DAGMCVolume]]):
        sense_data = [
            vol.handle if vol is not None else np.uint64(0) for vol in volumes
        ]
        self._tag_set_data(self.model.surf_sense_tag, sense_data)

        parents = self.model._core.get_parent_meshsets(self.handle)
        for parent in parents:
            if parent not in sense_data:
                # REVIEW (akoen): pymoab seems not to have a remove_parent method.
                logger.warning(
                    f"Surface has existing parent {parent} that cannot be removed."
                )
        # Establish parent-child relationships
        for vol in volumes:
            if vol is not None:
                self.model._core.add_parent_child(vol.handle, self.handle)

    @property
    def boundary(self) -> Optional[str]:
        """Name of the boundary condition assigned to this surface."""
        for group in self.groups:
            if group.name.startswith("boundary:"):
                return group.name[9:]
        return None

    @boundary.setter
    def boundary(self, name: str):
        existing_group = False
        for group in self.model.groups:
            if f"boundary:{name}" == group.name:
                # Add volume to group matching specified name, unless the volume
                # is already in it
                if self in group:
                    return
                group.add(self)
                existing_group = True

            elif self in group and group.name.startswith("boundary:"):
                # Remove volume from existing group
                group.remove(self)

        if not existing_group:
            # Create new group and add entity
            new_group = self.model.create_group(f"boundary:{name}")
            new_group.global_id = (
                max((g.global_id for g in self.model.groups), default=0) + 1
            )
            new_group.add(self)

    @property
    def adjacent_volumes(self) -> list[DAGMCVolume]:
        """Get adjacent volumes.

        Returns:
            Adjacent volumes.
        """
        parent_entities = self.model._core.get_parent_meshsets(self.handle)
        return [DAGMCVolume(self.model, e) for e in parent_entities]

    @property
    def triangles(self) -> Range:
        """Get range of triangle elements."""
        return self.model._core.get_entities_by_type(self.handle, pymoab.types.MBTRI)


class DAGMCVolume(DAGMCEntitySet):
    """DAGMC volume entity."""

    @property
    def adjacent_surfaces(self) -> list[DAGMCSurface]:
        """Get adjacent surfaces.

        Returns:
            Adjacent surfaces.
        """
        child_entities = self.model._core.get_child_meshsets(self.handle)
        return [DAGMCSurface(self.model, e) for e in child_entities]

    @property
    def material(self) -> Optional[str]:
        """Name of the material assigned to this volume."""
        for group in self.groups:
            if self in group and group.name.startswith("mat:"):
                return group.name[4:]
        return None

    @material.setter
    def material(self, name: str):
        existing_group = False
        for group in self.model.groups:
            if f"mat:{name}" == group.name:
                # Add volume to group matching specified name, unless the volume
                # is already in it
                if self in group:
                    return
                group.add(self)
                existing_group = True

            elif self in group and group.name.startswith("mat:"):
                # Remove volume from existing group
                group.remove(self)

        if not existing_group:
            # Create new group and add entity
            new_group = self.model.create_group(f"mat:{name}")
            new_group.global_id = (
                max((g.global_id for g in self.model.groups), default=0) + 1
            )
            new_group.add(self)


class MOABModel:
    """MOAB Model.

    This class holds a generic MOAB mesh, which could be a 2D surface mesh used
    in DAGMC, a 3D tetrahedral mesh, etc.
    """

    _core: pymoab.core.Core

    def __init__(self, core_or_file: Union[PathLike, pymoab.core.Core]):
        """Initialize model from a file or existing pymoab Core object.

        Args:
            core_or_file: path-like or Core object.
        """
        if isinstance(core_or_file, (str, os.PathLike)):
            core = pymoab.core.Core()
            core.load_file(str(core_or_file))
        else:
            core = core_or_file
        self._core = core

    @cached_property
    def category_tag(self) -> pymoab.tag.Tag:
        """Category tag."""
        return self._core.tag_get_handle(
            pymoab.types.CATEGORY_TAG_NAME,
            pymoab.types.CATEGORY_TAG_SIZE,
            pymoab.types.MB_TYPE_OPAQUE,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

    @cached_property
    def name_tag(self) -> pymoab.tag.Tag:
        """Name tag."""
        return self._core.tag_get_handle(
            pymoab.types.NAME_TAG_NAME,
            pymoab.types.NAME_TAG_SIZE,
            pymoab.types.MB_TYPE_OPAQUE,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

    @cached_property
    def id_tag(self) -> pymoab.tag.Tag:
        """Global ID tag."""
        # Default tag, does not need to be created
        return self._core.tag_get_handle(pymoab.types.GLOBAL_ID_TAG_NAME)
        #   rval = mdbImpl->tag_get_handle(GLOBAL_ID_TAG_NAME, 1, moab::MB_TYPE_INTEGER,
        #                                  id_tag, moab::MB_TAG_DENSE | moab::MB_TAG_ANY, &zero);
        # return self._core.tag_get_handle(
        #     pymoab.types.GLOBAL_ID_TAG_NAME,
        #     1,
        #     pymoab.types.MB_TYPE_INTEGER,
        #     pymoab.types.MB_TAG_DENSE,
        #     create_if_missing=True,
        # )

    @cached_property
    def geom_dimension_tag(self) -> pymoab.tag.Tag:
        """Geometry dimension tag."""
        return self._core.tag_get_handle(
            pymoab.types.GEOM_DIMENSION_TAG_NAME,
            1,
            pymoab.types.MB_TYPE_INTEGER,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

    @cached_property
    def surf_sense_tag(self) -> pymoab.tag.Tag:
        """Surface sense tag."""
        return self._core.tag_get_handle(
            "GEOM_SENSE_2",
            2,
            pymoab.types.MB_TYPE_HANDLE,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

    @cached_property
    def faceting_tol_tag(self) -> pymoab.tag.Tag:
        """Faceting tolerance tag."""
        return self._core.tag_get_handle(
            "FACETING_TOL",
            1,
            pymoab.types.MB_TYPE_DOUBLE,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

    @property
    def root_set(self) -> np.uint64:
        """Get handle of MOAB root entity set."""
        return self._core.get_root_set()

    @property
    def tets(self) -> Range:
        """Get range of tetrahedral elements."""
        return self._core.get_entities_by_type(
            self.root_set, pymoab.types.MBTET, recur=True
        )

    @property
    def triangles(self) -> Range:
        """Get range of triangle elements."""
        return self._core.get_entities_by_type(
            self.root_set, pymoab.types.MBTRI, recur=True
        )

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> MOABModel:
        """Create MOAB model from mesh.

        Args:
            mesh: Mesh from which to build MOAB mesh.

        Returns:
            Initialized model.
        """
        with tempfile.NamedTemporaryFile(suffix=".vtk", delete=True) as mesh_file:
            with mesh:
                gmsh.write(mesh_file.name)
            return cls(mesh_file.name)

    @classmethod
    def read_file(cls, h5m_file: PathLike) -> MOABModel:
        """Initialize model from .h5m file.

        Args:
            h5m_file: File to load.

        Returns:
            Initialized model.
        """
        warnings.warn(
            f"The read_file method is deprecated. Use {cls.__name__}(...) instead.",
            FutureWarning,
            stacklevel=2,
        )
        return cls(h5m_file)

    def write(self, filename: PathLike):
        """Write MOAB model to .h5m, .vtk, or other file.

        Args:
            filename: Filename with format-appropriate extension.
        """
        logger.info(f"Writing DAGMC mesh to {filename!s}")
        self._core.write_file(str(filename))


class DAGMCModel(MOABModel):
    """DAGMC Model."""

    def create_group(self, group_name: str) -> DAGMCGroup:
        """Create new group.

        Args:
            group_name: Name assigned to the new group

        Returns:
            Group object.
        """
        group = DAGMCGroup(self, self._core.create_meshset())
        group.geom_dimension = 4
        group.category = "Group"
        group.name = group_name
        return group

    def create_volume(self, global_id: Optional[int] = None) -> DAGMCVolume:
        """Create new volume.

        Args:
            global_id: Global ID.

        Returns:
            Volume object.
        """
        volume = DAGMCVolume(self, self._core.create_meshset())
        volume.geom_dimension = 3
        volume.category = "Volume"
        if global_id is not None:
            volume.global_id = global_id
        return volume

    def create_surface(self, global_id: Optional[int] = None) -> DAGMCSurface:
        """Create new surface.

        Args:
            global_id: Global ID.

        Returns:
            Surface object.
        """
        surface = DAGMCSurface(self, self._core.create_meshset())
        surface.geom_dimension = 2
        surface.category = "Surface"
        if global_id is not None:
            surface.global_id = global_id
        return surface

    def create_curve(self, global_id: Optional[int] = None) -> DAGMCCurve:
        """Create new curve.

        Args:
            global_id: Global ID.

        Returns:
            curve object.
        """
        curve = DAGMCCurve(self, self._core.create_meshset())
        curve.geom_dimension = 1
        curve.category = "Curve"
        if global_id is not None:
            curve.global_id = global_id
        return curve

    @property
    def groups(self) -> list[DAGMCGroup]:
        """Get list of groups."""
        group_handles: Range = self._core.get_entities_by_type_and_tag(
            self.root_set,
            pymoab.types.MBENTITYSET,
            [self.category_tag],
            ["Group"],
        )
        return [DAGMCGroup(self, handle) for handle in group_handles]

    @staticmethod
    def make_watertight(
        input_filename: PathLike,
        output_filename: PathLike,
        binary_path: str = "make_watertight",
    ) -> bool:
        """Make mesh watertight.

        Args:
            input_filename: Input .h5m filename.
            output_filename: Output watertight .h5m filename.
            binary_path: Path to make_watertight or default to find in path. Defaults to
            "make_watertight".
        """
        subprocess.run(
            [binary_path, str(input_filename), "-o", str(output_filename)],
            check=True,
        )

    @staticmethod
    def check_overlap(
        input_filename: PathLike,
        binary_path: str = "overlap_check",
        points_per_edge: int = 0,
        num_threads: int = 1,
    ) -> bool:
        """Check mesh for overlaps.

        Args:
            input_filename: Input .h5m filename.
            binary_path: Path to overlap_check or default to find in path. Defaults to
            "overlap_check".
            points_per_edge: Number of evenly-spaced points to test on each triangle
                edge. If points_per_edge=0, only triangle vertex locations are checked.
                Defaults to 0.
            num_threads: Number of threads.

        Returns:
            True if no overlaps are found, else False.
        """
        out = subprocess.run(
            [
                binary_path,
                str(input_filename),
                "-p",
                str(points_per_edge),
                "-t",
                str(num_threads),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return "No overlaps were found." in out.stdout.splitlines()

    @staticmethod
    def check_watertight(
        input_filename: PathLike,
        binary_path: str = "check_watertight",
    ):
        """Check mesh for watertightness.

        Args:
            input_filename: Input .h5m filename.
            binary_path: Path to overlap_check or default to find in path. Defaults to
            "check_watertight".

        Returns:
            True if mesh is watertight, else False.
        """
        out = subprocess.run(
            [binary_path, str(input_filename)],
            capture_output=True,
            text=True,
            check=True,
        )
        output_lines = out.stdout.splitlines()
        for line in output_lines:
            if ("leaky surface ids=" in line) or ("leaky volume ids=" in line):
                return line.strip().endswith("=")
        return False

    @classmethod
    def from_mesh(
        cls,
        mesh: Mesh,
    ) -> DAGMCModel:
        """Compose DAGMC MOAB .h5m file from mesh.

        Args:
            mesh: Mesh from which to build DAGMC geometry.
        """
        core = pymoab.core.Core()
        model = cls(core)

        with mesh:
            # Warn about volume elements being discarded
            if gmsh.model.mesh.get_elements(3)[1]:
                logger.warning("Discarding volume elements from mesh.")

            gmsh.model.mesh.removeDuplicateNodes()

            # 1. Add nodes
            node_tags, coords, _ = gmsh.model.mesh.get_nodes()
            assert len(node_tags) == len(np.unique(node_tags))
            # print(len(coords.reshape(-1, 3)))
            # print(len(np.unique(node_tags.reshape(-1, 3), axis=0)))
            if np.isnan(coords).any():
                raise ValueError("Mesh coordinates contain NaNs.")
            if np.isinf(coords).any():
                raise ValueError("Mesh coordinates contain infinite values.")

            moab_vertices = core.create_vertices(coords)
            core.tag_set_data(model.id_tag, moab_vertices, node_tags.astype(np.int32))
            debug_unused_nodes = np.array(moab_vertices)
            node_tag_map: dict[int, int] = dict(
                zip(node_tags, moab_vertices, strict=True)
            )

            # 2. Add surface elements and boundary conditions
            surface_map: Final[dict[int, DAGMCSurface]] = {}
            surface_dimtags = gmsh.model.get_entities(2)
            surface_tags: Final[list[int]] = [s[1] for s in surface_dimtags]
            logger.debug(f"Mesh has {len(surface_tags)} surfaces")
            for i, surface_tag in enumerate(surface_tags):
                element_types, _, node_tags_list = gmsh.model.mesh.get_elements(
                    2, surface_tag
                )

                if len(node_tags_list[0]) == 0:
                    raise RuntimeError(f"Surface {surface_tag} has no elements")

                surface_set = model.create_surface(surface_tag)
                surface_map[surface_tag] = surface_set

                if (
                    bc := mesh.entity_metadata(2, surface_tag).boundary_condition
                ) is not None:
                    surface_set.boundary = bc

                volume_adjacencies_dimtags = gmsh.model.get_adjacencies(2, surface_tag)[
                    0
                ]
                if len(volume_adjacencies_dimtags) == 0:
                    logger.warning(
                        "DAGMC does not support surfaces without attached "
                        "volumes. Creating a void volume for surface "
                        f"{surface_tag}."
                    )
                    existing_vol_tags = [v[1] for v in gmsh.model.get_entities(3)]
                    new_id = max(existing_vol_tags, default=0) + 1
                    volume_set = model.create_volume(new_id)
                    volume_set.material = "void"
                    surface_set.forward_volume = volume_set

                # Add elements to MOAB
                if len(element_types) != 1 or element_types[0] != 2:
                    raise RuntimeError(
                        f"Non-triangular element in surface {surface_tag}: "
                        "{element_types}"
                    )

                moab_conn = np.array(
                    [node_tag_map[t] for t in node_tags_list[0]], dtype=np.uint64
                )

                debug_unused_nodes = np.setdiff1d(debug_unused_nodes, moab_conn)

                # reshaped_conn = moab_conn.reshape(-1, 3)
                # if (
                #     np.any(reshaped_conn[:, 0] == reshaped_conn[:, 1])
                #     or np.any(reshaped_conn[:, 1] == reshaped_conn[:, 2])
                #     or np.any(reshaped_conn[:, 2] == reshaped_conn[:, 0])
                # ):
                #     raise RuntimeError(
                #         f"Surface {surface_tag} contains degenerate triangles (duplicate vertices). "
                #         "This may cause OBB tree construction to fail."
                #     )

                # ---------------------------------------------------------
                # ROBUST FIX: Manual Element Loop
                # ---------------------------------------------------------

                # 1. Create the flat array and reshape (Standard setup)
                moab_conn_flat = np.array(
                    [node_tag_map[t] for t in node_tags_list[0]], dtype=np.uint64
                )
                moab_conn_2d = moab_conn_flat.reshape(-1, 3)

                # 2. Pre-allocate array for the new handles
                num_elements = len(moab_conn_2d)
                new_handles = np.zeros(num_elements, dtype=np.uint64)

                # 3. Loop and create elements individually
                #    This bypasses the broken bulk wrapper by using the method
                #    we KNOW works (create_element).
                try:
                    for i, conn in enumerate(moab_conn_2d):
                        new_handles[i] = core.create_element(pymoab.types.MBTRI, conn)
                        adj = core.get_adjacencies(Range(new_handles[i]), 0, False)
                        if len(adj) != 3:
                            logger.error(
                                "Triangle {i} has no vertex adjacencies.", exc_info=True
                            )

                        adj = core.get_adjacencies(Range(new_handles[i]), 1, True)
                        if len(adj) != 3:
                            logger.error(
                                "Triangle {i} has no edge adjacencies.", exc_info=True
                            )
                except Exception as e:
                    logger.error(f"Failed at element {i}: {e}")
                    raise

                # 4. Convert NumPy array of handles to a MOAB Range
                triangles = Range(new_handles)

                # 5. Add to the Surface Set
                core.add_entities(surface_set.handle, triangles)
                core.add_entities(surface_set.handle, np.unique(moab_conn_flat))
                ...

            assert len(debug_unused_nodes) == 0

            # 4. Add volume sets and surface sense
            volume_dimtags = gmsh.model.get_entities(3)
            volume_tags = [v[1] for v in volume_dimtags]
            volume_map: Final[dict[int, DAGMCVolume]] = {}
            logger.debug(f"Mesh has {len(volume_tags)} volumes")
            for i, volume_tag in enumerate(volume_tags):
                volume_set = model.create_volume(volume_tag)
                volume_map[volume_tag] = volume_set

                mat_name = mesh.entity_metadata(3, volume_tag).material
                volume_set.material = mat_name

            for surface_tag, surface in surface_map.items():
                metadata = mesh.entity_metadata(2, surface_tag)
                if (forward_vol_tag := metadata.forward_volume) is not None:
                    assert (forward_vol := volume_map.get(forward_vol_tag)) is not None
                    surface.forward_volume = forward_vol
                if (reverse_vol_tag := metadata.reverse_volume) is not None:
                    assert (reverse_vol := volume_map.get(reverse_vol_tag)) is not None
                    surface.reverse_volume = reverse_vol

            # 5. Delete empty volumes
            for volume in volume_map.values():
                if not core.get_child_meshsets(volume.handle):
                    # logger.warning(f"Volume {volume.global_id} has no assigned edges. ")
                    ...
                    # logger.warning(
                    #     f"Volume {volume.global_id} has no assigned surfaces. "
                    #     "Removing from MOAB model."
                    # )
                    # core.delete_entity(volume.handle)

            for surface in surface_map.values():
                if not core.get_child_meshsets(surface.handle):
                    ...
                    # logger.warning(
                    #     f"Suface {surface.global_id} has no assigned edges. "
                    # )
                    # logger.warning(
                    #     f"Suface {surface.global_id} has no assigned edges. "
                    #     "Removing from MOAB model."
                    # )
                    # core.delete_entity(surface.handle)

            all_entities = core.get_entities_by_handle(0)
            file_set = core.create_meshset()
            # TODO(akoen): faceting tol set to a random value
            # https://github.com/Thea-Energy/neutronics-cad/issues/5
            # faceting_tol required to be set for make_watertight, although its
            # significance is not clear
            # core.tag_set_data(model.faceting_tol_tag, file_set, 1e-3)
            core.tag_set_data(model.faceting_tol_tag, file_set, 0.1)
            # core.tag_set_data(model.faceting_tol_tag, file_set, 1e-5)
            core.add_entities(file_set, all_entities)

            # core.tag_set_data(model.faceting_tol_tag, model.root_set)

            vertices = core.get_entities_by_dimension(0, 0, recur=True)
            edges = core.get_entities_by_dimension(0, 1, recur=True)
            triangles = core.get_entities_by_dimension(0, 2, recur=True)

            n_adjacencies = np.empty(len(vertices), dtype=np.uint32)
            for i, v in enumerate(vertices):
                n_adjacencies[i] = len(core.get_adjacencies(v, 2))
            bins = {}
            for i, c in enumerate(np.bincount(n_adjacencies)):
                bins[i] = int(c)
            logger.info(f"Vertex-Triangle # adjacencies = {bins}")

            n_adjacencies = np.empty(len(edges), dtype=np.uint32)
            for i, v in enumerate(edges):
                n_adjacencies[i] = len(core.get_adjacencies(v, 2))
            bins = {}
            for i, c in enumerate(np.bincount(n_adjacencies)):
                bins[i] = int(c)
            logger.info(f"Edge-Triangle bins = {bins}")

            n_adjacencies = np.empty(len(triangles), dtype=np.uint32)
            for i, t in enumerate(triangles):
                adjacencies = core.get_adjacencies(t, 2)
                connectivity = core.get_connectivity(t)
                n_adjacencies[i] = len(adjacencies)
            bins = {}
            for i, c in enumerate(np.bincount(n_adjacencies)):
                bins[i] = int(c)
            logger.info(f"Triangle-Triangle adjacencies = {bins}")

            return model

    @classmethod
    def make_from_mesh(cls, mesh: Mesh) -> DAGMCModel:
        """Compose DAGMC MOAB .h5m file from mesh.

        Args:
            mesh: Mesh from which to build DAGMC geometry.
        """
        warnings.warn(
            "The make_from_mesh method is deprecated. Use from_mesh instead.",
            FutureWarning,
            stacklevel=2,
        )
        return cls.from_mesh(mesh)

    def _get_entities_of_geom_dimension(self, dim: int) -> list[np.uint64]:
        return self._core.get_entities_by_type_and_tag(
            0, pymoab.types.MBENTITYSET, [self.geom_dimension_tag], [dim]
        )

    @property
    def curves(self) -> list[DAGMCCurve]:
        """Get curves in this model.

        Returns:
            Curve.
        """
        curve_handles = self._get_entities_of_geom_dimension(1)
        return [DAGMCCurve(self, h) for h in curve_handles]

    @property
    def surfaces(self) -> list[DAGMCSurface]:
        """Get surfaces in this model.

        Returns:
            Surfaces.
        """
        surface_handles = self._get_entities_of_geom_dimension(2)
        return [DAGMCSurface(self, h) for h in surface_handles]

    @property
    def volumes(self) -> list[DAGMCVolume]:
        """Get volumes in this model.

        Returns:
            Volumes.
        """
        volume_handles = self._get_entities_of_geom_dimension(3)
        return [DAGMCVolume(self, h) for h in volume_handles]
