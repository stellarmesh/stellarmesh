import build123d as bd
import cadquery as cq
import pytest
import stellarmesh as sm


@pytest.fixture
def geom_bd_layered_torus():
    solids = [bd.Solid.make_torus(1000, 100)]
    for _ in range(3):
        solids.append(solids[-1].faces()[0].thicken(100))
    solids = solids[1:]
    return solids


@pytest.fixture
def geom_cq_layered_torus():
    solids = [cq.Solid.makeTorus(1000, 100)]
    for _ in range(3):
        solids.append(cq.Workplane(solids[-1]).faces().vals()[0].thicken(100))
    solids = solids[1:]
    return solids


@pytest.fixture
def geom_ocp_layered_torus(geom_bd_layered_torus):
    return [s.wrapped for s in geom_bd_layered_torus]


@pytest.mark.parametrize(
    "fixture",
    [("geom_bd_layered_torus"), ("geom_cq_layered_torus"), ("geom_ocp_layered_torus")],
)
def test_geometry_init(fixture, request):
    solids = request.getfixturevalue(fixture)
    material_names = ["material"] * len(solids)
    geom = sm.Geometry(solids, material_names)
    if hasattr(solids[0], "wrapped"):
        assert geom.solids == [s.wrapped for s in solids]
    else:
        assert geom.solids == solids
    assert geom.material_names == material_names


def test_geometry_init_wrong_materials(geom_bd_layered_torus):
    solids = geom_bd_layered_torus
    material_names = ["material"] * (len(solids) - 1)
    with pytest.raises(ValueError):
        sm.Geometry(solids, material_names)


def test_step_import_compound(geom_bd_layered_torus):
    bd.Compound.make_compound(geom_bd_layered_torus).export_step("model.step")
    sm.Geometry.from_step("model.step", material_names=[""] * 3)
    with pytest.raises(ValueError):
        sm.Geometry.from_step("model.step", material_names=[""] * 2)


def test_step_import_solid(geom_bd_layered_torus):
    geom_bd_layered_torus[0].export_step("layer.step")
    sm.Geometry.from_step("layer.step", material_names=[""])


def test_brep_import_compound(geom_bd_layered_torus):
    bd.Compound.make_compound(geom_bd_layered_torus).export_brep("model.brep")
    sm.Geometry.from_brep("model.brep", material_names=[""] * 3)
    with pytest.raises(ValueError):
        sm.Geometry.from_brep("model.brep", material_names=[""] * 2)


def test_brep_import_solid(geom_bd_layered_torus):
    geom_bd_layered_torus[0].export_brep("layer.brep")
    sm.Geometry.from_brep("layer.brep", material_names=[""])


def test_geometry_imprint(geom_bd_layered_torus):
    solids = geom_bd_layered_torus
    material_names = ["material"] * len(solids)
    geom = sm.Geometry(solids, material_names)
    imprinted_geom = geom.imprint()
