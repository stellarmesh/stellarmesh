import build123d as bd
import gmsh
import pytest
import stellarmesh as sm


@pytest.fixture
def geom_bd_sphere():
    solids = [bd.Solid.make_sphere(10.0)]
    return sm.Geometry(solids, ["a"])


def test_mesh_geometry_2d(geom_bd_sphere):
    sm.Mesh.from_geometry(geom_bd_sphere, 5, 5, dim=2)


def test_threads(geom_bd_sphere):
    sm.Mesh.from_geometry(geom_bd_sphere, 5, 5, dim=2, num_threads=4)


def test_mesh_geometry_3d(geom_bd_sphere):
    sm.Mesh.from_geometry(geom_bd_sphere, 5, 5, dim=3)


def test_scale_factor(geom_bd_sphere):
    mesh = sm.Mesh.from_geometry(geom_bd_sphere, 5, 5, scale_factor=0.1)

    # Get coordinates of mesh nodes and shape into (N, 3)
    with mesh:
        _, coords, _ = gmsh.model.mesh.getNodes()
    coords.shape = (-1, 3)

    # Check that coordinates are within a sphere of radius 1
    assert (coords.min(axis=0) >= -1.0).all()
    assert (coords.max(axis=0) <= 1.0).all()
