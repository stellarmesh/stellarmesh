"""Meshing tests."""

import importlib.resources
import subprocess
from pathlib import Path

import build123d as bd
import gmsh
import pytest

import stellarmesh as sm

from . import resources
from .test_geometry import (
    geom_bd_capped_torus,  # noqa: F401
    geom_bd_torus_single_surface,  # noqa: F401
    model_bd_layered_torus,
    model_bd_nestedspheres,
    model_bd_offsetboxes,  # noqa: F401
    model_bd_sphere,
    model_cq_layered_torus,
    model_cq_nestedspheres,
    model_cq_sphere,
)


def test_entity_metadata(model_bd_sphere):
    geom = sm.Geometry(model_bd_sphere, material_names=[""])
    mesh = sm.SurfaceMesh.from_geometry(
        geom,
        sm.OCCSurfaceOptions(tol_angular_deg=0.5),
    )

    mesh.entity_metadata(2, 1).forward_volume  # noqa: B018
    mesh.entity_metadata(2, 1).forward_volume = 10
    assert mesh.entity_metadata(2, 1).forward_volume == 10
    with pytest.raises(AttributeError, match=r".*has no attribute invalid_attrib.*"):
        mesh.entity_metadata(2, 1).invalid_attrib  # noqa: B018
    with pytest.raises(AttributeError, match=r".*has no attribute invalid_attrib.*"):
        mesh.entity_metadata(2, 1).invalid_attrib = 10


@pytest.mark.parametrize(
    "model_name,num_elements_occ,num_elements_gmsh",
    [
        (model_bd_sphere.__name__, 306, 850),
        (model_cq_sphere.__name__, 306, 850),
        (model_bd_layered_torus.__name__, 5408, 2814),
        (model_cq_layered_torus.__name__, 5408, 2814),
        (model_bd_nestedspheres.__name__, 612, 1086),
        (model_cq_nestedspheres.__name__, 612, 1086),
    ],
)
def test_surface_mesh_num_elements(
    model_name: str,
    num_elements_occ: int,
    num_elements_gmsh: int,
    request: pytest.FixtureRequest,
):
    model = request.getfixturevalue(model_name)
    geom = sm.Geometry(model, material_names=[""] * len(model))

    mesh_occ = sm.SurfaceMesh.from_geometry(
        geom,
        sm.OCCSurfaceOptions(tol_angular_deg=0.5),
    )
    with mesh_occ:
        num_elements = len(gmsh.model.mesh.get_elements(2, -1)[1][0])
    assert num_elements == num_elements_occ, (
        f"Number of elements in OCC mesh ({num_elements}) does not match expected "
        f"({num_elements_occ}) for model {model_name}. Mesh: {mesh_occ._mesh_filename}"
    )

    mesh_gmsh = sm.SurfaceMesh.from_geometry(
        geom,
        sm.GmshSurfaceOptions(min_mesh_size=0.5, max_mesh_size=2),
    )
    with mesh_gmsh:
        num_elements = len(gmsh.model.mesh.get_elements(2, -1)[1][0])
    assert num_elements == num_elements_gmsh, (
        f"Number of elements in GMSH mesh ({num_elements}) does not match expected "
        f"({num_elements_gmsh}) for model {model_name}. Mesh: {mesh_gmsh._mesh_filename}"
    )


@pytest.mark.parametrize(
    "model_name,num_elements_gmsh",
    [
        (model_bd_sphere.__name__, 2648),
        (model_bd_layered_torus.__name__, 6501),
        (model_bd_nestedspheres.__name__, 2828),
    ],
)
def test_volume_mesh_num_elements(
    model_name: str,
    num_elements_gmsh: int,
    request: pytest.FixtureRequest,
):
    model = request.getfixturevalue(model_name)
    geom = sm.Geometry(model, material_names=[""] * len(model))

    mesh = sm.VolumeMesh.from_geometry(
        geom,
        sm.GmshVolumeOptions(min_mesh_size=0.5, max_mesh_size=2),
    )
    with mesh:
        num_elements = len(gmsh.model.mesh.get_elements(3, -1)[1][0])
    assert num_elements_gmsh == num_elements, (
        f"Number of elements in volume mesh ({num_elements}) does not match expected "
        f"({num_elements_gmsh}) for model {model_name}. Mesh: {mesh._mesh_filename}"
    )


@pytest.fixture
def geom_imprintedboxes(model_bd_offsetboxes):
    geom = sm.Geometry(
        model_bd_offsetboxes, material_names=[""] * len(model_bd_offsetboxes)
    )
    geom_imprinted = geom.imprint()

    return geom_imprinted


def test_mesh_geom_imprintedboxes(geom_imprintedboxes):
    options = [
        ("OCC", sm.OCCSurfaceOptions(tol_angular_deg=1)),
        ("GMSH", sm.GmshSurfaceOptions(min_mesh_size=0.5, max_mesh_size=2)),
    ]
    for backend, option in options:
        mesh = sm.SurfaceMesh.from_geometry(
            geom_imprintedboxes,
            option,
        )
        with mesh:
            surface_tags = gmsh.model.get_entities(2)
            assert len(surface_tags) == 13, (
                f"Number of surfaces in {backend} mesh ({len(surface_tags)}) does not match expected (13). Mesh: {mesh._mesh_filename}"
            )


def test_mesh_volume_imprintedboxes(geom_imprintedboxes):
    mesh = sm.VolumeMesh.from_geometry(
        geom_imprintedboxes, sm.GmshVolumeOptions(min_mesh_size=0.5, max_mesh_size=2)
    )
    with mesh:
        volume_tags = gmsh.model.get_entities(3)
        assert len(volume_tags) == 2, (
            f"Number of volumes ({len(volume_tags)}) does not match expected (2). Mesh: {mesh._mesh_filename}"
        )
        surface_tags = gmsh.model.get_entities(2)
        assert len(surface_tags) == 13, (
            f"Number of surfaces ({len(surface_tags)}) does not match expected (13). Mesh: {mesh._mesh_filename}"
        )


@pytest.fixture
def model_bd_stellarator_plasma():
    with importlib.resources.path(resources, "NCSX-LCFS.brep") as path:
        plasma = bd.import_brep(path)
        return plasma


def test_mesh_overlap(model_bd_stellarator_plasma):
    plasma = model_bd_stellarator_plasma
    b1: bd.Solid = bd.thicken(plasma, 5).solid()  # pyright: ignore[reportAssignmentType]
    b2: bd.Solid = bd.thicken(b1.faces()[0], 5).solid()  # pyright: ignore[reportAssignmentType]

    def check_overlap(tol_linear):
        geom = sm.Geometry(solids=[b1, b2], material_names=[""] * 2)
        mesh = sm.SurfaceMesh.from_geometry(
            geom, sm.OCCSurfaceOptions(tol_linear=tol_linear, tol_angular_deg=None)
        )

        mesh.write("tmp.msh")
        dagmc_model = sm.DAGMCModel.from_mesh(mesh)
        dagmc_model.write("dagmc.h5m")

        result = subprocess.run(
            ["overlap_check", "-p", "5", "dagmc.h5m"],
            check=False,
            cwd=".",
            capture_output=True,
            text=True,
        )

        return result.stdout

    assert "No overlaps were found" in check_overlap(2.5)
    assert "Overlap Location:" in check_overlap(5)


def test_mesh_surface_capped_torus_bcs(geom_bd_capped_torus):
    """Test material and BC assignments for a capped torus."""
    # NOTE: In this test all surfaces are also members of solids
    for options in [sm.GmshSurfaceOptions(), sm.OCCSurfaceOptions()]:
        mesh = sm.SurfaceMesh.from_geometry(geom_bd_capped_torus, options)
        with mesh:
            for dim, tag in gmsh.model.get_entities(2):
                if tag in [2, 3, 5, 6, 8, 9, 11, 12]:
                    assert (
                        mesh.entity_metadata(dim, tag).boundary_condition
                        == "reflecting"
                    )


def test_mesh_torus_single_surface_bc(geom_bd_torus_single_surface):
    for options in [sm.GmshSurfaceOptions(), sm.OCCSurfaceOptions()]:
        mesh = sm.SurfaceMesh.from_geometry(geom_bd_torus_single_surface, options)
        with mesh:
            for dim, tag in gmsh.model.get_entities(2):
                if tag in [1]:
                    assert mesh.entity_metadata(dim, tag).boundary_condition == "vacuum"


def test_mesh_export_exodus(model_bd_layered_torus, tmp_path: Path):
    geom = sm.Geometry(
        model_bd_layered_torus, material_names=[""] * len(model_bd_layered_torus)
    )
    mesh = sm.VolumeMesh.from_geometry(
        geom, sm.GmshVolumeOptions(min_mesh_size=0.5, max_mesh_size=2)
    )
    mesh.write(tmp_path / "out.exo", use_meshio=True)


def test_gmsh_threads(model_bd_layered_torus, tmp_path: Path):
    geom = sm.Geometry(
        model_bd_layered_torus, material_names=[""] * len(model_bd_layered_torus)
    )
    mesh = sm.SurfaceMesh.from_geometry(
        geom,
        sm.GmshSurfaceOptions(min_mesh_size=0.5, max_mesh_size=2, num_threads=8),
    )
