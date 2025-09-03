"""Stellarmesh geometry.

name: geometry.py
author: Alex Koen

desc: Geometry class represents a CAD geometry to be meshed.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import warnings
from typing import Protocol, Sequence, Union, overload, runtime_checkable

from build123d import Optional

try:
    from OCP.BOPAlgo import BOPAlgo_MakeConnected
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Solid, TopoDS_Face, TopoDS_Shell
except ImportError as e:
    raise ImportError(
        "OCP not found. See Stellarmesh installation instructions."
    ) from e

logger = logging.getLogger(__name__)


@runtime_checkable
class Face(Protocol):
    wrapped: TopoDS_Face | None


@runtime_checkable
class Shell(Protocol):
    wrapped: TopoDS_Shell | None


@runtime_checkable
class Solid(Protocol):
    wrapped: TopoDS_Solid | None


class Geometry:
    """Geometry, representing an ordered list of solids, to be meshed."""

    solids: list[TopoDS_Solid]
    material_names: list[str]
    surfaces: list[TopoDS_Face | TopoDS_Shell]
    surface_boundary_conditions: list[str]

    def __init__(
        self,
        solids: Optional[Sequence[Solid | TopoDS_Solid]],
        material_names: Optional[Sequence[str]],
        surfaces: Optional[Sequence[Face | Shell | TopoDS_Face | TopoDS_Shell]] = None,
        surface_boundary_conditions: Optional[Sequence[str]] = None,
    ):
        """Construct geometry from solids.

        Args:
            solids: List of solids, where each solid is a build123d Solid, cadquery
            Solid, or OCP TopoDS_Solid.
            material_names: List of materials. Must match length of solids.
        """
        if (
            (solids and not material_names)
            or (material_names and not solids)
            or (solids and material_names and len(solids) != len(material_names))
        ):
            raise ValueError(
                "If solids or material_names are provided, both must be provided and match in length."
            )

        if (
            (surfaces and not surface_boundary_conditions)
            or (surface_boundary_conditions and not surfaces)
            or (
                surfaces
                and surface_boundary_conditions
                and len(surfaces) != len(surface_boundary_conditions)
            )
        ):
            raise ValueError(
                "If surfaces or surface_boundary_conditions are provided, both must be provided and match in length."
            )

        self.solids = []
        if solids:
            for i, s in enumerate(solids):
                if isinstance(s, TopoDS_Solid):
                    self.solids.append(s)
                elif isinstance(s, Solid):
                    if s.wrapped is None:
                        raise ValueError(
                            f"{s} {i} has no wrapped TopoDS_Solid. Is it valid?"
                        )
                    self.solids.append(s.wrapped)
                else:
                    raise TypeError(f"Solid {i} is of invalid type {type(s).__name__}")

            self.material_names = list(material_names)

        self.surfaces = []
        if surfaces:
            for i, s in enumerate(surfaces):
                if isinstance(s, (TopoDS_Face, TopoDS_Shell)):
                    self.surfaces.append(s)
                elif isinstance(s, (Face, Shell)):  # type: ignore
                    if s.wrapped is None:
                        raise ValueError(
                            f"{s} {i} has no wrapped TopoDS_Face or TopoDS_Shell. Is it valid?"
                        )
                    self.surfaces.append(s.wrapped)
                else:
                    raise TypeError(
                        f"Surface {i} is of invalid type {type(s).__name__}"
                    )
            self.surface_boundary_conditions = list(surface_boundary_conditions)

    @staticmethod
    def _solids_from_shape(shape: TopoDS_Shape) -> list[TopoDS_Solid]:
        """Return all the solids in this shape."""
        solids = []
        if shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_SOLID:
            solids.append(TopoDS.Solid_s(shape))
        if shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_COMPOUND:
            explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_SOLID)
            while explorer.More():
                assert explorer.Current().ShapeType() == TopAbs_ShapeEnum.TopAbs_SOLID
                solids.append(TopoDS.Solid_s(explorer.Current()))
                explorer.Next()
        return solids

    # TODO(akoen): from_step and from_brep are not DRY
    # https://github.com/Thea-Energy/stellarmesh/issues/2
    @classmethod
    def from_step(
        cls,
        filename: str,
        material_names: Sequence[str],
    ) -> Geometry:
        """Import model from a step file.

        Args:
            filename: File path to import.
            material_names: Ordered list of material names matching solids in file.

        Returns:
            Model.
        """
        logger.info(f"Importing {filename}")

        reader = STEPControl_Reader()
        read_status = reader.ReadFile(filename)
        if read_status != IFSelect_RetDone:
            raise ValueError(f"STEP File {filename} could not be loaded")
        for i in range(reader.NbRootsForTransfer()):
            reader.TransferRoot(i + 1)

        solids = []
        for i in range(reader.NbShapes()):
            shape = reader.Shape(i + 1)
            solids.extend(cls._solids_from_shape(shape))

        return cls(solids, material_names)

    @classmethod
    def import_step(
        cls,
        filename: str,
        material_names: Sequence[str],
    ) -> Geometry:
        """Import model from a step file.

        Args:
            filename: File path to import.
            material_names: Ordered list of material names matching solids in file.

        Returns:
            Model.
        """
        warnings.warn(
            "The import_step method is deprecated. Use from_step instead.",
            FutureWarning,
            stacklevel=2,
        )
        return cls.from_step(filename, material_names)

    @classmethod
    def from_brep(
        cls,
        filename: str,
        material_names: Sequence[str],
    ) -> Geometry:
        """Import model from a brep (cadquery, build123d native) file.

        Args:
            filename: File path to import.
            material_names: Ordered list of material names matching solids in file.

        Returns:
            Model.
        """
        logger.info(f"Importing {filename}")

        shape = TopoDS_Shape()
        builder = BRep_Builder()
        BRepTools.Read_s(shape, filename, builder)

        if shape.IsNull():
            raise ValueError(f"Could not import {filename}")
        solids = cls._solids_from_shape(shape)

        logger.info(f"Importing {len(solids)} from {filename}")
        return cls(solids, material_names)

    @classmethod
    def import_brep(
        cls,
        filename: str,
        material_names: Sequence[str],
    ) -> Geometry:
        """Import model from a brep (cadquery, build123d native) file.

        Args:
            filename: File path to import.
            material_names: Ordered list of material names matching solids in file.

        Returns:
            Model.
        """
        warnings.warn(
            "The import_brep method is deprecated. Use from_brep instead.",
            FutureWarning,
            stacklevel=2,
        )
        return cls.from_brep(filename, material_names)

    def imprint(self) -> Geometry:
        """Imprint faces of current geometry.

        Returns:
            A new geometry with the imprinted and merged geometry.
        """
        bldr = BOPAlgo_MakeConnected()
        bldr.SetRunParallel(theFlag=True)
        bldr.SetUseOBB(theUseOBB=True)

        for solid in self.solids:
            bldr.AddArgument(solid)

        bldr.Perform()
        res = bldr.Shape()
        res_solids = self._solids_from_shape(res)

        if (l0 := len(res_solids)) != (l1 := len(self.solids)):
            raise RuntimeError(
                f"Length of imprinted solids {l0} != length of original solids {l1}"
            )

        return type(self)(res_solids, self.material_names)
