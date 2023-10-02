import build123d as bd
import stellarmesh as sm
import pytest


@pytest.fixture
def geom_layered_torus():
    solids = [bd.Solid.make_torus(1000, 100)]
    for _ in range(3):
        solids.append(solids[-1].faces()[0].thicken(100))
    solids = solids[1:]
    return solids


def test_geometry_init(geom_layered_torus):
    solids = geom_layered_torus
    material_names = ["material"] * len(solids)
    geom = sm.Geometry(solids, material_names)
    assert geom.solids == solids
    assert geom.material_names == material_names


def test_geometry_init_wrong_materials(geom_layered_torus):
    solids = geom_layered_torus
    material_names = ["material"] * (len(solids) - 1)
    with pytest.raises(ValueError):
        sm.Geometry(solids, material_names)


@pytest.mark.skip(reason="Not working, will soon refactor.")
def test_geometry_imprint(geom_layered_torus):
    solids = geom_layered_torus
    material_names = ["material"] * len(solids)
    geom = sm.Geometry(solids, material_names)
    imprinted_geom = geom.imprint()
    assert len(imprinted_geom.solids) == len(solids)
    assert all(isinstance(s, bd.Solid) for s in imprinted_geom.solids)
