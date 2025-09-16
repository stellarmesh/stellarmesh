"""Gmsh wrapper and DAGMC geometry creator."""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from .geometry import Geometry
from .mesh import (
    GmshSurfaceAlgo,
    GmshSurfaceOptions,
    GmshVolumeAlgo,
    GmshVolumeOptions,
    Mesh,
    OCCSurfaceAlgo,
    OCCSurfaceOptions,
    SurfaceMesh,
    VolumeMesh,
)
from .moab import DAGMCModel, DAGMCSurface, DAGMCVolume, MOABModel

__all__ = [
    "DAGMCModel",
    "DAGMCSurface",
    "DAGMCVolume",
    "Geometry",
    "GmshSurfaceAlgo",
    "GmshSurfaceOptions",
    "GmshVolumeAlgo",
    "GmshVolumeOptions",
    "MOABModel",
    "Mesh",
    "OCCSurfaceAlgo",
    "OCCSurfaceOptions",
    "SurfaceMesh",
    "VolumeMesh",
]


with contextlib.suppress(PackageNotFoundError):
    __version__ = version("stellarmesh")
