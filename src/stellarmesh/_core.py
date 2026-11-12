import os
from typing import Union

try:
    from OCP.TopoDS import TopoDS  # pyright: ignore[reportMissingModuleSource]
except ImportError as e:
    raise ImportError(
        "OCP not found. See Stellarmesh installation instructions."
    ) from e

PathLike = Union[str, bytes, os.PathLike]

# https://github.com/stellarmesh/stellarmesh/issues/85
# NOTE(akoen) OCC 7.9 turned the TopoDS class into a namespace, earlier OCP exposes
# them as static methods with an `_s` suffix.
topods_face = getattr(TopoDS, "Face_s", None) or TopoDS.Face
topods_solid = getattr(TopoDS, "Solid_s", None) or TopoDS.Solid
