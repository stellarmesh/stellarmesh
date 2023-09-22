"""Stellarmesh MOAB model.

name: moab.py
author: Alex Koen

desc: MOABModel class represents a MOAB model.
"""
import subprocess
import tempfile
from dataclasses import dataclass, field

import gmsh
import numpy as np
import pymoab.core
import pymoab.types

from .mesh import Mesh


class _MOABEntity:
    _core: pymoab.core.Core
    handle: np.uint64

    def __init__(self, core: pymoab.core.Core, handle: np.uint64):
        self._core = core
        self.handle = handle


class MOABSurface(_MOABEntity):
    """MOAB surface entity."""

    @property
    def adjacent_volumes(self) -> list["MOABVolume"]:
        """Get adjacent volumes.

        Returns:
            Adjacent volumes.
        """
        parent_entities = self._core.get_parent_meshsets(self.handle)
        return [MOABVolume(self._core, e) for e in parent_entities]


class MOABVolume(_MOABEntity):
    """MOAB volume entity."""

    @property
    def adjacent_surfaces(self) -> list["MOABSurface"]:
        """Get adjacent surfaces.

        Returns:
            Adjacent surfaces.
        """
        child_entities = self._core.get_child_meshsets(self.handle)
        return [MOABSurface(self._core, e) for e in child_entities]


@dataclass
class _Surface:
    """Internal class for surface sense handling."""

    handle: np.uint64
    forward_volume: np.uint64 = field(default=np.uint64(0))
    reverse_volume: np.uint64 = field(default=np.uint64(0))

    def sense_data(self) -> list[np.uint64]:
        """Get MOAB tag sense data.

        Returns:
            Sense data.
        """
        return [self.forward_volume, self.reverse_volume]


class MOABModel:
    """MOAB Model."""

    # h5m_filename: str
    _core: pymoab.core.Core

    def __init__(self, core: pymoab.core.Core):
        """Initialize model from a pymoab core object.

        Args:
            core: Pymoab core.
        """
        self._core = core

    @classmethod
    def read_file(cls, h5m_file: str) -> "MOABModel":
        """Initialize model from .h5m file.

        Args:
            h5m_file: File to load.

        Returns:
            Initialized model.
        """
        core = pymoab.core.Core()
        core.load_file(h5m_file)
        return cls(core)

    def write(self, filename: str):
        """Write MOAB model to .h5m, .vtk, or other file.

        Args:
            filename: Filename with format-appropriate extension.
        """
        self._core.write_file(filename)

    @staticmethod
    def make_watertight(
        input_filename: str,
        output_filename: str,
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
            [binary_path, input_filename, "-o", output_filename],
            check=True,
        )

    @staticmethod
    def _get_moab_tag_handles(core: pymoab.core.Core) -> dict[str, np.uint64]:
        tag_handles = {}

        sense_tag_name = "GEOM_SENSE_2"
        sense_tag_size = 2
        tag_handles["surf_sense"] = core.tag_get_handle(
            sense_tag_name,
            sense_tag_size,
            pymoab.types.MB_TYPE_HANDLE,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

        tag_handles["category"] = core.tag_get_handle(
            pymoab.types.CATEGORY_TAG_NAME,
            pymoab.types.CATEGORY_TAG_SIZE,
            pymoab.types.MB_TYPE_OPAQUE,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

        tag_handles["name"] = core.tag_get_handle(
            pymoab.types.NAME_TAG_NAME,
            pymoab.types.NAME_TAG_SIZE,
            pymoab.types.MB_TYPE_OPAQUE,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

        geom_dimension_tag_size = 1
        tag_handles["geom_dimension"] = core.tag_get_handle(
            pymoab.types.GEOM_DIMENSION_TAG_NAME,
            geom_dimension_tag_size,
            pymoab.types.MB_TYPE_INTEGER,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

        faceting_tol_tag_name = "FACETING_TOL"
        faceting_tol_tag_size = 1
        tag_handles["faceting_tol"] = core.tag_get_handle(
            faceting_tol_tag_name,
            faceting_tol_tag_size,
            pymoab.types.MB_TYPE_DOUBLE,
            pymoab.types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

        # Default tag, does not need to be created
        tag_handles["global_id"] = core.tag_get_handle(pymoab.types.GLOBAL_ID_TAG_NAME)

        return tag_handles

    @classmethod
    def make_from_mesh(  # noqa: PLR0915
        cls,
        mesh: Mesh,
    ):
        """Compose DAGMC MOAB .h5m file from mesh.

        Args:
            mesh: Mesh from which to build DAGMC geometry.
            filename: Filename of the output .h5m file.
        """
        core = pymoab.core.Core()

        tag_handles = cls._get_moab_tag_handles(core)

        known_surfaces: dict[int, _Surface] = {}
        known_groups: dict[int, np.uint64] = {}

        with mesh:
            volume_dimtags = gmsh.model.get_entities(3)
            volume_tags = [v[1] for v in volume_dimtags]
            for i, volume_tag in enumerate(volume_tags):
                # Add volume set
                volume_set_handle = core.create_meshset()
                global_id = volume_set_handle
                core.tag_set_data(tag_handles["global_id"], global_id, i)
                core.tag_set_data(tag_handles["geom_dimension"], volume_set_handle, 3)
                core.tag_set_data(tag_handles["category"], volume_set_handle, "Volume")

                # Add volume to its physical group, which stores metadata incl. material
                # TODO(akoen): should this be a parent-child relationship?
                # https://github.com/Thea-Energy/neutronics-cad/issues/2
                vol_groups = gmsh.model.get_physical_groups_for_entity(3, volume_tag)
                if (num_groups := len(vol_groups)) != 1:
                    raise ValueError(
                        f"Volume with tag {volume_tag} and global_id {global_id} "
                        f"belongs to {num_groups} physical groups, should be 1"
                    )

                if (vol_group := vol_groups[0]) not in known_groups:
                    mat_name = gmsh.model.get_physical_name(3, vol_group)
                    group_set = core.create_meshset()
                    core.tag_set_data(tag_handles["category"], group_set, "Group")
                    core.tag_set_data(tag_handles["name"], group_set, f"{mat_name}")
                    core.tag_set_data(tag_handles["global_id"], group_set, vol_group)
                    known_groups[vol_group] = group_set
                else:
                    group_set = known_groups[vol_group]

                core.add_entity(group_set, volume_set_handle)

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
                    if surface_tag not in known_surfaces:
                        surface_set_handle = core.create_meshset()
                        surface = _Surface(handle=surface_set_handle)
                        surface.forward_volume = volume_set_handle
                        known_surfaces[surface_tag] = surface

                        core.tag_set_data(
                            tag_handles["global_id"], surface.handle, surface_tag
                        )
                        core.tag_set_data(
                            tag_handles["geom_dimension"], surface.handle, 2
                        )
                        core.tag_set_data(
                            tag_handles["category"], surface.handle, "Surface"
                        )
                        core.tag_set_data(
                            tag_handles["surf_sense"],
                            surface.handle,
                            surface.sense_data(),
                        )

                        # Write surface to MOAB. STL export/import is very efficient.
                        with tempfile.NamedTemporaryFile(
                            suffix=".stl", delete=True
                        ) as stl_file:
                            group_tag = gmsh.model.add_physical_group(2, [surface_tag])
                            gmsh.write(stl_file.name)
                            gmsh.model.remove_physical_groups([(2, group_tag)])
                            core.load_file(stl_file.name, surface_set_handle)

                    else:
                        # Surface already has a forward volume, so this must be the
                        # reverse volume.
                        surface = known_surfaces[surface_tag]
                        surface.reverse_volume = volume_set_handle
                        core.tag_set_data(
                            tag_handles["surf_sense"],
                            surface.handle,
                            surface.sense_data(),
                        )

                    core.add_parent_child(volume_set_handle, surface.handle)

            all_entities = core.get_entities_by_handle(0)
            file_set = core.create_meshset()
            # TODO(akoen): faceting tol set to a random value
            # https://github.com/Thea-Energy/neutronics-cad/issues/5
            # faceting_tol required to be set for make_watertight, although its
            # significance is not clear
            core.tag_set_data(tag_handles["faceting_tol"], file_set, 1e-3)
            core.add_entities(file_set, all_entities)

            return cls(core)

    def _get_entities_of_geom_dimension(self, dim: int) -> list[np.uint64]:
        dim_tag = self._core.tag_get_handle(pymoab.types.GEOM_DIMENSION_TAG_NAME)
        return self._core.get_entities_by_type_and_tag(
            0, pymoab.types.MBENTITYSET, dim_tag, [dim]
        )

    @property
    def surfaces(self):
        """Get surfaces in this model.

        Returns:
            Surfaces.
        """
        surface_handles = self._get_entities_of_geom_dimension(2)
        return [MOABSurface(self._core, h) for h in surface_handles]

    @property
    def volumes(self):
        """Get volumes in this model.

        Returns:
            Volumes.
        """
        volume_handles = self._get_entities_of_geom_dimension(3)
        return [MOABVolume(self._core, h) for h in volume_handles]
