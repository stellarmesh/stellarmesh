import build123d as bd
import pytest
import stellarmesh as sm


@pytest.fixture
def geom_bd_sphere():
    solids = [bd.Solid.make_sphere(10.0)]
    return sm.Geometry(solids, ["a"])


def test_mesh_geometry_2d(geom_bd_sphere):
    sm.Mesh.from_geometry(geom_bd_sphere, 5, 5, dim=2)


def test_mesh_geometry_3d(geom_bd_sphere):
    sm.Mesh.from_geometry(geom_bd_sphere, 5, 5, dim=3)
