import build123d as bd
import cadquery as cq
import pytest

import stellarmesh as sm


@pytest.fixture
def model_bd_sphere():
    return [bd.Solid.make_sphere(10.0)]


@pytest.fixture
def model_cq_sphere():
    return [cq.Workplane("XY").sphere(10.0).val()]


@pytest.fixture
def model_bd_nestedspheres():
    s0 = bd.Solid.make_sphere(5.0)
    s1 = bd.Solid.thicken(s0.faces()[0], 5.0)
    return [s0, s1]


@pytest.fixture
def model_cq_nestedspheres():
    s0 = cq.Workplane("XY").sphere(5.0).val()
    s1 = cq.Workplane("XY").sphere(10.0).val()
    return [s0, s1]


@pytest.fixture
def model_bd_layered_torus():
    solids = [bd.Solid.make_torus(10, 1)]
    for _ in range(3):
        solids.append(bd.Solid.thicken(solids[-1].faces()[0], 1))
    return solids[1:]


@pytest.fixture
def model_cq_layered_torus():
    solids = [cq.Solid.makeTorus(10, 1)]
    for _ in range(3):
        solids.append(solids[-1].Faces()[0].thicken(1))
    return solids[1:]


@pytest.fixture
def model_ocp_layered_torus(model_bd_layered_torus):
    return [s.wrapped for s in model_bd_layered_torus]


@pytest.fixture
def model_ocp_layered_torus_cq(model_cq_layered_torus):
    return [s.wrapped for s in model_cq_layered_torus]


@pytest.fixture
def model_bd_offsetboxes():
    b1 = bd.Solid.make_box(10, 10, 10)
    b2 = b1.transformed(offset=(0, 5, 10))
    return [b1, b2]


@pytest.fixture
def geom_imprintedboxes(model_bd_offsetboxes):
    # Removed duplication: this was defined in both test_geometry.py and test_mesh.py
    geom = sm.Geometry(
        model_bd_offsetboxes, material_names=[""] * len(model_bd_offsetboxes)
    )
    return geom.imprint()


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


def pytest_collection_modifyitems(config, items):
    """Automatically applies markers based on test location and name.

    Called after test collection, before execution.
    """
    for item in items:
        if "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        if "/benchmark/" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
