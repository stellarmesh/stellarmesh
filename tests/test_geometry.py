import build123d as bd
import pytest

import stellarmesh as sm


@pytest.fixture
def model_bd_sphere():
    return [bd.Solid.make_sphere(10.0)]


@pytest.fixture
def model_bd_nestedspheres():
    s0 = bd.Solid.make_sphere(5.0)
    s1 = bd.Solid.thicken(s0.faces()[0], 5.0)
    return [s0, s1]


@pytest.fixture
def model_bd_layered_torus():
    solids = [bd.Solid.make_torus(10, 1)]
    for _ in range(3):
        solids.append(bd.Solid.thicken(solids[-1].faces()[0], 1))
    solids = solids[1:]
    return solids


@pytest.fixture
def model_ocp_layered_torus(model_bd_layered_torus):
    return [s.wrapped for s in model_bd_layered_torus]


@pytest.fixture
def model_bd_offsetboxes():
    b1 = bd.Solid.make_box(10, 10, 10)
    b2 = b1.transformed(offset=(0, 5, 10))
    # b2 = bd.Solid.make_box(10, 10, 10, plane=bd.Plane.XY.offset(11))

    return [b1, b2]


@pytest.fixture
def geom_imprintedboxes(model_bd_offsetboxes):
    geom = sm.Geometry(
        model_bd_offsetboxes, material_names=[""] * len(model_bd_offsetboxes)
    )
    geom_imprinted = geom.imprint()
    return geom_imprinted


@pytest.fixture
def geom_bd_capped_torus():
    faces = bd.ShapeList()
    solids = [bd.Solid.make_torus(10, 1, major_angle=90)]
    for _ in range(3):
        s = bd.Solid.thicken(solids[-1].faces()[0], 1)
        solids.append(s)

    for s in solids:
        faces.extend(s.faces())

    xz_faces = faces.filter_by(bd.Plane.XZ)
    yz_faces = faces.filter_by(bd.Plane.YZ)
    surfaces = list(xz_faces + yz_faces)

    geom = sm.Geometry(
        solids=solids,
        material_names=[""] * len(solids),
        surfaces=surfaces,
        surface_boundary_conditions=["reflecting"] * len(surfaces),
    )
    return geom


@pytest.fixture
def geom_bd_torus_single_surface():
    face: bd.Face = bd.Solid.make_torus(10, 1).face()  # type: ignore
    geom = sm.Geometry(surfaces=[face], surface_boundary_conditions=["vacuum"])
    return geom


@pytest.mark.parametrize(
    "fixture",
    [("model_bd_layered_torus"), ("model_ocp_layered_torus")],
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


def test_geometry_init_wrong_materials(model_bd_layered_torus):
    solids = model_bd_layered_torus
    material_names = ["material"] * (len(solids) - 1)
    with pytest.raises(ValueError):
        sm.Geometry(solids, material_names)


def test_step_import_compound(model_bd_layered_torus):
    cmp = bd.Compound(model_bd_layered_torus)
    bd.export_step(cmp, "model.step")
    sm.Geometry.from_step("model.step", material_names=[""] * 3)
    with pytest.raises(ValueError):
        sm.Geometry.from_step("model.step", material_names=[""] * 2)


def test_step_import_solid(model_bd_layered_torus):
    bd.export_step(model_bd_layered_torus[0], "layer.step")
    sm.Geometry.from_step("layer.step", material_names=[""])


def test_brep_import_compound(model_bd_layered_torus):
    cmp = bd.Compound(model_bd_layered_torus)
    bd.export_brep(cmp, "model.brep")
    sm.Geometry.from_brep("model.brep", material_names=[""] * 3)
    with pytest.raises(ValueError):
        sm.Geometry.from_brep("model.brep", material_names=[""] * 2)


def test_brep_import_solid(model_bd_layered_torus):
    bd.export_brep(model_bd_layered_torus[0], "layer.brep")
    sm.Geometry.from_brep("layer.brep", material_names=[""])


def test_geometry_imprint(model_bd_layered_torus):
    geom = sm.Geometry(
        model_bd_layered_torus, material_names=[""] * len(model_bd_layered_torus)
    )
    geom.imprint()
