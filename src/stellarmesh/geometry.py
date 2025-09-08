"""Stellarmesh geometry.

name: geometry.py
author: Alex Koen

desc: Geometry class represents a CAD geometry to be meshed.
"""

from __future__ import annotations

import logging
import warnings
from typing import (
    Protocol,
    Sequence,
    overload,
)

from build123d import Optional, Type

try:
    from OCP.BOPAlgo import BOPAlgo_MakeConnected
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shape, TopoDS_Shell, TopoDS_Solid
except ImportError as e:
    raise ImportError(
        "OCP not found. See Stellarmesh installation instructions."
    ) from e

logger = logging.getLogger(__name__)


class Face(Protocol):
    """Interface for a CadQuery or Build123d Face."""

    wrapped: TopoDS_Face | None


class Shell(Protocol):
    """Interface for a CadQuery or Build123d Shell."""

    wrapped: TopoDS_Shell | None


class Solid(Protocol):
    """Interface for a CadQuery or Build123d Solid."""

    wrapped: TopoDS_Solid | None


class Geometry:
    """Geometry, representing an ordered list of solids, to be meshed."""

    solids: list[TopoDS_Solid]
    material_names: list[str]
    faces: list[TopoDS_Face]
    face_boundary_conditions: list[str]

    def __init__(
        self,
        solids: Optional[Sequence[Solid | TopoDS_Solid]] = None,
        material_names: Optional[Sequence[str]] = None,
        surfaces: Optional[Sequence[Face | Shell | TopoDS_Face | TopoDS_Shell]] = None,
        surface_boundary_conditions: Optional[Sequence[str]] = None,
    ):
        """Construct geometry from solids.

        Args:
            solids: List of solids, where each solid is a build123d Solid, CadQuery
            Solid, or OCP TopoDS_Solid.
            material_names: List of materials. Must match length of solids.
            surfaces: List of surfaces, where each surface is a build123d or Cadquery
            Face or Shell, or an OCP TopoDS_Face or TopoDS_Shell.
            surface_boundary_conditions: List of boundary condition names. Must match
            length of surfaces.
        """
        if (solids and not material_names) or (material_names and not solids):
            raise ValueError(
                "If solids or material_names are provided"
                ", both must be provided and match in length."
            )

        if (surfaces and not surface_boundary_conditions) or (
            surface_boundary_conditions and not surfaces
        ):
            raise ValueError(
                "If surfaces or surface_boundary_conditions are provided"
                ", both must be provided and match in length."
            )

        self.solids = []
        self.material_names = []
        if solids and material_names:
            for i, (s, mat_name) in enumerate(zip(solids, material_names, strict=True)):
                s_wrapped = (
                    s if isinstance(s, TopoDS_Shape) else getattr(s, "wrapped", None)
                )

                if s_wrapped is None:
                    raise ValueError(
                        f"{s} {i} has no wrapped TopoDS_Shape. Is it valid?"
                    )

                self.solids.append(s_wrapped)
                self.material_names.append(mat_name)

        self.faces = []
        self.face_boundary_conditions = []
        if surfaces and surface_boundary_conditions:
            for i, (s, bc) in enumerate(
                zip(surfaces, surface_boundary_conditions, strict=True)
            ):
                s_wrapped = (
                    s
                    if isinstance(s, (TopoDS_Face, TopoDS_Shell))
                    else getattr(s, "wrapped", None)
                )

                if s.wrapped is None:  # type: ignore
                    raise ValueError(
                        f"{s} {i} has no wrapped TopoDS_Shape. Is it valid?"
                    )

                if isinstance(s_wrapped, TopoDS_Face):
                    self.faces.append(s_wrapped)
                    self.face_boundary_conditions.append(bc)
                elif isinstance(s_wrapped, TopoDS_Shell):
                    child_faces = self._get_child_shapes(s_wrapped, TopoDS_Face)
                    self.faces.extend(child_faces)
                    self.face_boundary_conditions.extend([bc] * len(child_faces))

                else:
                    raise TypeError(
                        f"Surface {i} is of invalid type {type(s).__name__}"
                    )

    @staticmethod
    @overload
    def _get_child_shapes(
        parent: TopoDS_Shape, shape_type: Type[TopoDS_Face]
    ) -> list[TopoDS_Face]: ...

    @staticmethod
    @overload
    def _get_child_shapes(
        parent: TopoDS_Shape, shape_type: Type[TopoDS_Shell]
    ) -> list[TopoDS_Shell]: ...

    @staticmethod
    @overload
    def _get_child_shapes(
        parent: TopoDS_Shape, shape_type: Type[TopoDS_Solid]
    ) -> list[TopoDS_Solid]: ...

    @staticmethod
    def _get_child_shapes(
        parent: TopoDS_Shape, shape_type: Type[TopoDS_Shape]
    ) -> Sequence[TopoDS_Shape]:
        """Return all the child shapes of this shape."""
        type_map = {
            "TopoDS_Face": TopAbs_ShapeEnum.TopAbs_FACE,
            "TopoDS_Shell": TopAbs_ShapeEnum.TopAbs_SHELL,
            "TopoDS_Solid": TopAbs_ShapeEnum.TopAbs_SOLID,
        }

        top_abs_type = type_map.get(shape_type.__name__)
        if top_abs_type is None:
            raise ValueError(f"Unsupported shape_type: {shape_type}")

        shapes = []
        explorer = TopExp_Explorer(parent, top_abs_type)
        while explorer.More():
            assert explorer.Current().ShapeType() == top_abs_type
            shapes.append(explorer.Current())
            explorer.Next()
        return shapes

    @classmethod
    def _get_solids_from_shape(cls, shape: TopoDS_Shape) -> list[TopoDS_Solid]:
        """Return all the solids in this shape."""
        solids: list[TopoDS_Solid] = []
        if shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_SOLID:
            solids.append(TopoDS.Solid_s(shape))
        elif shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_COMPOUND:
            solids = cls._get_child_shapes(shape, TopoDS_Solid)
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
            solids.extend(cls._get_solids_from_shape(shape))

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
        solids = cls._get_solids_from_shape(shape)

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
        res_solids = self._get_solids_from_shape(res)

        if (l0 := len(res_solids)) != (l1 := len(self.solids)):
            raise RuntimeError(
                f"Length of imprinted solids {l0} != length of original solids {l1}"
            )

        return type(self)(res_solids, self.material_names)
