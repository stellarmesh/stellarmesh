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
    VolumeReport,
)
from .moab import (
    DAGMCModel,
    DAGMCSurface,
    DAGMCVolume,
    MOABModel,
    MOABVolumeModel,
    OverlapReport,
    WatertightReport,
)

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
    "MOABVolumeModel",
    "Mesh",
    "OCCSurfaceAlgo",
    "OCCSurfaceOptions",
    "OverlapReport",
    "SurfaceMesh",
    "VolumeMesh",
    "VolumeReport",
    "WatertightReport",
]


with contextlib.suppress(PackageNotFoundError):
    __version__ = version("stellarmesh")
