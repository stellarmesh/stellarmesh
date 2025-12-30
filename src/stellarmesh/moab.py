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
from typing import Literal, Optional, Sequence, Union

import h5py
import numpy as np

from ._core import PathLike
from .mesh import Mesh

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
    def curve_sense(
        self,
    ) -> Optional[Sequence[tuple[DAGMCSurface, Literal[-1] | Literal[1]]]]:
        """Curve sense data."""
        try:
            if (
                hasattr(self.model, "_deferred_curve_senses")
                and self.handle in self.model._deferred_curve_senses
            ):
                ents, senses = self.model._deferred_curve_senses[self.handle]
                surfaces = [DAGMCSurface(self.model, ent) for ent in ents]
                return list(zip(surfaces, senses, strict=True))

            ents = self.model._core.tag_get_data(
                self.model.curve_sense_tags[0], self.handle, flat=True
            )
            surfaces = [DAGMCSurface(self.model, ent) for ent in ents]
            senses = self.model._core.tag_get_data(
                self.model.curve_sense_tags[1], self.handle, flat=True
            )
        except RuntimeError:
            return None

        return list(zip(surfaces, senses, strict=True))

    @curve_sense.setter
    def curve_sense(
        self, curve_senses: Sequence[tuple[DAGMCSurface, Literal[-1] | Literal[1]]]
    ):
        ents_data = np.array(
            [curve_sense[0].handle for curve_sense in curve_senses], dtype=np.uint64
        )
        sense_data = np.array(
            [curve_sense[1] for curve_sense in curve_senses], dtype=np.int32
        )

        self.model._deferred_curve_senses[self.handle] = (ents_data, sense_data)

        parents = self.model._core.get_parent_meshsets(self.handle)
        for parent in parents:
            if parent not in ents_data:
                # REVIEW (akoen): pymoab seems not to have a remove_parent method.
                logger.warning(
                    f"Curve has existing parent {parent} that cannot be removed."
                )
        # Establish parent-child relationships
        for surf_handle in ents_data:
            self.model._core.add_parent_child(surf_handle, self.handle)

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
    def curve_sense_tags(self) -> tuple[pymoab.tag.Tag, pymoab.tag.Tag]:
        """Curve sense tags."""
        return (
            self._core.tag_get_handle(
                "GEOM_SENSE_N_ENTS",
                0,
                pymoab.types.MB_TYPE_HANDLE,
                pymoab.types.MB_TAG_SPARSE | pymoab.types.MB_TAG_VARLEN,
                create_if_missing=True,
            ),
            self._core.tag_get_handle(
                "GEOM_SENSE_N_SENSES",
                0,
                pymoab.types.MB_TYPE_INTEGER,
                pymoab.types.MB_TAG_SPARSE | pymoab.types.MB_TAG_VARLEN,
                create_if_missing=True,
            ),
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

    def __init__(self, core_or_file: Union[PathLike, pymoab.core.Core]):
        super().__init__(core_or_file)
        self._deferred_curve_senses: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def write(self, filename: PathLike):
        """Write DAGMC model to .h5m, .vtk, or other file.

        Args:
            filename: Filename with format-appropriate extension.
        """
        super().write(filename)

        if str(filename).endswith(".h5m") and self._deferred_curve_senses:
            self._write_deferred_tags(filename)

    def _write_deferred_tags(self, filename: PathLike):
        """Write deferred tags to HDF5 file."""
        with h5py.File(filename, "r+") as f:
            handles = sorted(self._deferred_curve_senses.keys())

            # GEOM_SENSE_N_ENTS (handles)
            ents_values = []
            ents_indices = [0]

            # GEOM_SENSE_N_SENSES (integers)
            senses_values = []
            senses_indices = [0]

            for h in handles:
                ents, senses = self._deferred_curve_senses[h]

                ents_values.extend(ents)
                ents_indices.append(ents_indices[-1] + len(ents))

                senses_values.extend(senses)
                senses_indices.append(senses_indices[-1] + len(senses))

            self._write_single_tag(
                f,
                "GEOM_SENSE_N_ENTS",
                handles,
                ents_values,
                ents_indices,
                np.uint64,
            )
            self._write_single_tag(
                f,
                "GEOM_SENSE_N_SENSES",
                handles,
                senses_values,
                senses_indices,
                np.int32,
            )

    def _write_single_tag(self, f, tag_name, handles, values, indices, dtype):
        if "tstt/tags" not in f:
            raise RuntimeError("No tstt/tags group")

        tags_grp = f["tstt/tags"]
        if tag_name not in tags_grp:
            logger.warning(f"Tag {tag_name} not found in HDF5 file. Skipping.")
            return

        tag_grp = tags_grp[tag_name]

        # Create/Overwrite datasets
        if "id_list" in tag_grp:
            del tag_grp["id_list"]
        tag_grp.create_dataset("id_list", data=np.array(handles, dtype=np.uint64))

        if "values" in tag_grp:
            del tag_grp["values"]
        tag_grp.create_dataset("values", data=np.array(values, dtype=dtype))

        if "indices" in tag_grp:
            del tag_grp["indices"]
        tag_grp.create_dataset("indices", data=np.array(indices, dtype=np.uint64))

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
    ):
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
    def overlap_check(
        input_filename: PathLike,
        binary_path: str = "overlap_check",
    ):
        """Check mesh for overlaps.

        Args:
            input_filename: Input .h5m filename.
            binary_path: Path to overlap_check or default to find in path. Defaults to
            "overlap_check".
        """
        subprocess.run(
            [binary_path, str(input_filename)],
            check=True,
        )

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

            # 1. Add nodes
            node_tags, coords, _ = gmsh.model.mesh.get_nodes()
            moab_nodes = core.create_vertices(coords)

            # Create a mapping from Gmsh tag to MOAB handle
            # Note: node_tags is int32 or int64, moab_nodes is uint64
            tag_map = dict(zip(node_tags, moab_nodes, strict=True))

            # 2. Add surface elements and boundary conditions
            surfaces: dict[int, DAGMCSurface] = {}
            surface_dimtags = gmsh.model.get_entities(2)
            surface_tags = [s[1] for s in surface_dimtags]
            for i, surface_tag in enumerate(surface_tags):
                surface_set = model.create_surface(surface_tag)
                surfaces[surface_tag] = surface_set

                surface_groups = gmsh.model.get_physical_groups_for_entity(
                    2, surface_tag
                )
                if (num_groups := len(surface_groups)) not in [0, 1]:
                    raise ValueError(
                        f"Surface with tag {surface_tag} and global_id {i}"
                        f"belongs to {num_groups} physical groups, should be 0 "
                        f"or 1."
                    )
                elif num_groups == 1:
                    boundary_name = gmsh.model.get_physical_name(2, surface_groups[0])
                    surface_set.boundary = boundary_name[9:]

                volume_adjacencies_dimtags = gmsh.model.get_adjacencies(2, surface_tag)[
                    0
                ]
                if len(volume_adjacencies_dimtags) == 0:
                    logger.warning(
                        "DAGMC does not support surfaces without attached "
                        "volumes. Creating a void volume for surface "
                        f"{surface_tag}."
                    )
                    volume_set = model.create_volume(i)
                    volume_set.material = "void"
                    surface_set.forward_volume = volume_set

                # Add elements to MOAB
                element_types, _, node_tags_list = gmsh.model.mesh.get_elements(
                    2, surface_tag
                )
                if len(element_types) != 1 or element_types[0] != 2:
                    raise RuntimeError(
                        f"Non-triangular element in surface {surface_tag}: {element_types}"
                    )

                moab_conn = np.array(
                    [tag_map[t] for t in node_tags_list[0]], dtype=np.uint64
                )

                triangles = core.create_elements(
                    pymoab.types.MBTRI, moab_conn.reshape(-1, 3)
                )
                core.add_entities(surface_set.handle, triangles)

            # 3. Add curves
            curve_dimtags = gmsh.model.get_entities(1)
            curve_tags = [c[1] for c in curve_dimtags]
            for i, curve_tag in enumerate(curve_tags):
                curve_set = model.create_curve(curve_tag)

                # Add elements to MOAB
                element_types, _, node_tags_list = gmsh.model.mesh.get_elements(
                    1, curve_tag
                )

                # Set curve senses
                upward_adjacencies, _ = gmsh.model.get_adjacencies(1, curve_tag)
                curve_sense = tuple((surfaces[a], 1) for a in upward_adjacencies)
                # FIXME(akoen): This sets all curve senses to positive, which is
                # incorrect, but possible non-fatal
                curve_set.curve_sense = curve_sense

                if len(node_tags_list) != 0:
                    # TODO (this may only have sequential edges)
                    moab_conn = np.array(
                        [tag_map[t] for t in node_tags_list[0]], dtype=np.uint64
                    )

                    edges = core.create_elements(
                        pymoab.types.MBEDGE, moab_conn.reshape(-1, 2)
                    )
                    core.add_entities(curve_set.handle, edges)
                else:
                    logger.warning(f"Curve {curve_tag} has no elements.")

            # 4. Add volume sets and surface sense
            known_surfaces: set[int] = set()
            volume_dimtags = gmsh.model.get_entities(3)
            volume_tags = [v[1] for v in volume_dimtags]
            for i, volume_tag in enumerate(volume_tags):
                volume_set = model.create_volume(i)

                # Add volume to its physical group, which stores metadata incl. material
                # TODO(akoen): should this be a parent-child relationship?
                # https://github.com/Thea-Energy/neutronics-cad/issues/2
                vol_groups = gmsh.model.get_physical_groups_for_entity(3, volume_tag)
                if (num_groups := len(vol_groups)) != 1:
                    raise ValueError(
                        f"Volume with tag {volume_tag} and global_id {i} "
                        f"belongs to {num_groups} physical groups, should be 1."
                    )
                mat_name = gmsh.model.get_physical_name(3, vol_groups[0])
                volume_set.material = mat_name[4:]

                # Add surfaces to MOAB core, respecting surface sense.
                # Logic: Gmsh meshes volumes in order. When it gets to the first volume,
                # it points all adjacent surfaces normals outward. For each subsequent
                # volume, it points the surface normals outwards iff the surface hasn't
                # yet been encountered. Thus, the first time a surface is encountered,
                # the current volume has a forward sense and the second time a reverse
                # sense.
                adjacencies = gmsh.model.get_adjacencies(3, volume_tag)

                surface_tags = adjacencies[1]
                for surface_tag in surface_tags:
                    surface_set = surfaces[surface_tag]
                    if surface_tag not in known_surfaces:
                        known_surfaces.add(surface_tag)
                        surface_set.forward_volume = volume_set
                    else:
                        logger.debug(
                            f"Surface {surface_tag} has been seen before"
                            ", setting reverse volume"
                        )
                        surface_set.reverse_volume = volume_set

            all_entities = core.get_entities_by_handle(0)
            file_set = core.create_meshset()
            # TODO(akoen): faceting tol set to a random value
            # https://github.com/Thea-Energy/neutronics-cad/issues/5
            # faceting_tol required to be set for make_watertight, although its
            # significance is not clear
            core.tag_set_data(model.faceting_tol_tag, file_set, 1e-3)
            core.add_entities(file_set, all_entities)

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
