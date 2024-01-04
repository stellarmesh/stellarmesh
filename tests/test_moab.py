import build123d as bd
import pytest
import stellarmesh as sm
from pymoab.rng import Range


@pytest.fixture(scope="module")
def model():
    solids = [bd.Solid.make_sphere(10.0)]
    geom = sm.Geometry(solids, ["iron"])
    mesh = sm.Mesh.from_geometry(geom, max_mesh_size=5, dim=2)
    return sm.DAGMCModel.from_mesh(mesh)


def test_surfaces(model):
    assert isinstance(model.surfaces, list)
    assert len(model.surfaces) == 1
    assert isinstance(model.surfaces[0], sm.DAGMCSurface)


def test_volumes(model):
    assert isinstance(model.volumes, list)
    assert len(model.volumes) == 1
    assert isinstance(model.volumes[0], sm.DAGMCVolume)


def test_id(model):
    assert model.surfaces[0].id == 1
    assert model.volumes[0].id == 0


def test_adjacent_surfaces(model):
    vol = model.volumes[0]
    surfaces = vol.adjacent_surfaces
    assert len(surfaces) == 1
    assert surfaces == [model.surfaces[0]]


def test_adjacent_volumes(model):
    surf = model.surfaces[0]
    volumes = surf.adjacent_volumes
    assert len(volumes) == 1
    assert volumes == [model.volumes[0]]


def test_tets(model):
    assert model.tets.empty()


def test_triangles(model):
    all_tris = model.triangles
    assert isinstance(all_tris, Range)
    surf_tris = model.surfaces[0].triangles
    assert isinstance(surf_tris, Range)
    assert all_tris.contains(surf_tris)


def test_group_name(model):
    vol = model.volumes[0]
    assert vol.group_name == "mat:iron"

    vol.group_name = "mat:plastic"
    assert vol.group_name == "mat:plastic"

    group_names = {x[1] for x in model.groups.keys()}
    assert "mat:plastic" in group_names
