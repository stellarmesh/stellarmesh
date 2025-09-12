import cadquery as cq
import pytest
from cadquery import exporters

import stellarmesh as sm


@pytest.fixture
def model_cq_sphere():
    return [cq.Workplane("XY").sphere(10.0).val()]


@pytest.fixture
def model_cq_nestedspheres():
    s0 = cq.Workplane("XY").sphere(5.0).val()
    s1 = cq.Workplane("XY").sphere(10.0).val()
    return [s0, s1]


@pytest.fixture
def model_cq_layered_torus():
    return [cq.Solid.makeTorus(10, r) for r in (2, 3, 4)]


@pytest.fixture
def model_ocp_layered_torus_cq(model_cq_layered_torus):
    return [s.wrapped for s in model_cq_layered_torus]


@pytest.fixture
def model_cq_offsetboxes():
    b1 = cq.Workplane("XY").box(10, 10, 10).val()
    b2 = cq.Workplane("XY").transformed(offset=(0, 5, 10)).box(10, 10, 10).val()
    return [b1, b2]


@pytest.fixture
def geom_imprintedboxes_cq(model_cq_offsetboxes):
    geom = sm.Geometry(
        model_cq_offsetboxes,
        material_names=[""] * len(model_cq_offsetboxes),
    )
    return geom.imprint()


@pytest.mark.parametrize(
    "fixture",
    ["model_cq_layered_torus", "model_ocp_layered_torus_cq"],
)
def test_geometry_init_cq(fixture, request):
    solids = request.getfixturevalue(fixture)
    material_names = ["material"] * len(solids)
    geom = sm.Geometry(solids, material_names)
    if hasattr(solids[0], "wrapped"):
        assert geom.solids == [s.wrapped for s in solids]
    else:
        assert geom.solids == solids
    assert geom.material_names == material_names


def test_geometry_init_wrong_materials_cq(model_cq_layered_torus):
    solids = model_cq_layered_torus
    material_names = ["material"] * (len(solids) - 1)
    with pytest.raises(ValueError):
        sm.Geometry(solids, material_names)


def test_step_import_compound_cq(model_cq_layered_torus):
    cmp = cq.Compound.makeCompound(model_cq_layered_torus)
    exporters.export(cmp, "model_cq.step")
    sm.Geometry.from_step("model_cq.step", material_names=[""] * 3)
    with pytest.raises(ValueError):
        sm.Geometry.from_step("model_cq.step", material_names=[""] * 2)


def test_step_import_solid_cq(model_cq_layered_torus):
    exporters.export(model_cq_layered_torus[0], "layer_cq.step")
    sm.Geometry.from_step("layer_cq.step", material_names=[""])


def test_brep_import_compound_cq(model_cq_layered_torus):
    cmp = cq.Compound.makeCompound(model_cq_layered_torus)
    exporters.export(cmp, "model_cq.brep")
    sm.Geometry.from_brep("model_cq.brep", material_names=[""] * 3)
    with pytest.raises(ValueError):
        sm.Geometry.from_brep("model_cq.brep", material_names=[""] * 2)


def test_brep_import_solid_cq(model_cq_layered_torus):
    exporters.export(model_cq_layered_torus[0], "layer_cq.brep")
    sm.Geometry.from_brep("layer_cq.brep", material_names=[""])


def test_geometry_imprint_cq(model_cq_layered_torus):
    geom = sm.Geometry(
        model_cq_layered_torus,
        material_names=[""] * len(model_cq_layered_torus),
    )
    geom.imprint()
