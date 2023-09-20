"""Stellarmesh geometry.

name: geometry.py
author: Alex Koen

desc: Geometry class represents a CAD geometry to be meshed.
"""
from typing import Sequence

import build123d as bd
from OCP.BOPAlgo import BOPAlgo_MakeConnected

from ._core import logger


class Geometry:
    """Geometry, representing an ordered list of solids, to be meshed."""

    solids: Sequence[bd.Solid]
    material_names: Sequence[str]

    def __init__(self, solids: Sequence[bd.Solid], material_names: Sequence[str]):
        """Construct geometry from solids.

        Args:
            solids: Solids.
            material_names: List of materials. Must match length of solids.
        """
        logger.info(f"Importing {len(solids)} solids to geometry")
        if len(material_names) != len(solids):
            raise ValueError("Length of material_names must match length of solids.")
        self.solids = solids
        self.material_names = material_names

    # TODO(akoen): import_step and import_brep are not DRY
    # https://github.com/Thea-Energy/stellarmesh/issues/2
    @classmethod
    def import_step(
        cls,
        filename: str,
        material_names: str,
    ) -> "Geometry":
        """Import model from a step file.

        Args:
            filename: File path to import.
            material_names: Ordered list of material names matching solids in file.

        Returns:
            Model.
        """
        geometry = bd.import_step(filename)
        solids = geometry.solids()
        if len(material_names) != len(solids):
            raise ValueError(
                "Length of material_names must match number of solids in file."
            )
        logger.info(f"Importing {len(solids)} from {filename}")
        return cls(solids, material_names)

    @classmethod
    def import_brep(
        cls,
        filename: str,
        material_names: str,
    ) -> "Geometry":
        """Import model from a brep (cadquery, build123d native) file.

        Args:
            filename: File path to import.
            material_names: Ordered list of material names matching solids in file.

        Returns:
            Model.
        """
        geometry = bd.import_brep(filename)
        solids = geometry.solids()
        if len(material_names) != len(solids):
            raise ValueError(
                "Length of material_names must match number of solids in file."
            )
        logger.info(f"Importing {len(solids)} from {filename}")
        return cls(solids, material_names)

    def imprint(self) -> "Geometry":
        """Imprint faces of current geometry.

        Returns:
            A new geometry with the imprinted and merged geometry.
        """
        bldr = BOPAlgo_MakeConnected()
        bldr.SetRunParallel(theFlag=True)
        bldr.SetUseOBB(theUseOBB=True)

        for new_solid in self.solids:
            if new_solid.wrapped is not None:
                bldr.AddArgument(new_solid.wrapped)

        bldr.Perform()
        res = bd.Shape(bldr.Shape())
        new_solids = res.solids()

        # Track bd attributes (material, name, label) across imprinting
        old_wrapped = [n.wrapped for n in self.solids]
        for new_solid in new_solids:
            old_matching = bldr.GetOrigins(new_solid.wrapped)
            if (matching_len := old_matching.Size()) != 1:
                raise RuntimeError(f"New solid derives from {matching_len} solids")
            matching = [o.IsSame(old_matching.First()) for o in old_wrapped if o]
            i = matching.index(True)
            for attr in ["label", "material", "color"]:
                new_solid.__setattr__(attr, self.solids[i].__getattribute__(attr))

        return type(self)(new_solids, self.material_names)
