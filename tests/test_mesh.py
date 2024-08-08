"""Meshing tests."""

import filecmp
from pathlib import Path

import gmsh
import pytest
import stellarmesh as sm

from .test_geometry import (
    geom_bd_imprintedboxes,
    geom_bd_layered_torus,
    geom_bd_nestedspheres,
    geom_bd_sphere,
)


@pytest.mark.parametrize(
    "model_name",
    [
        geom_bd_layered_torus.__name__,
        geom_bd_nestedspheres.__name__,
        geom_bd_sphere.__name__,
        geom_bd_imprintedboxes.__name__,
    ],
)
def test_mesh_output(model_name: str, request: pytest.FixtureRequest, tmp_path: Path):
    """Ensure that generated meshes match benchmark .msh files."""
    model = request.getfixturevalue(model_name)
    msh_name = f"mesh_{model_name}.msh"
    mesh = sm.SurfaceMesh.from_geometry(
        model,
        sm.OCCSurfaceOptions(0.1, 10),
    )
    msh_file = tmp_path / msh_name
    mesh.write(str(msh_file))
    cmp_file = Path(__file__).parent / "data" / msh_name

    # For initial benmark (check meshes!)
    init = False
    if init:
        mesh.write(str(cmp_file))
        raise AssertionError()

    assert filecmp.cmp(msh_file, cmp_file, shallow=False), "Mesh files differ"


def test_mesh_geom_bd_imprintedboxes(geom_bd_imprintedboxes):
    """Test imprinting."""
    mesh = sm.SurfaceMesh.from_geometry(
        geom_bd_imprintedboxes,
        sm.OCCSurfaceOptions(0.1, 10),
    )
    with mesh:
        surface_tags = gmsh.model.get_entities(2)
        assert len(surface_tags) == 13
