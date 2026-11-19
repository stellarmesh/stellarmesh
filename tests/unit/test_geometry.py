import build123d as bd
import pytest
import stellarmesh as sm


class TestGeometryInitialization:
    @pytest.mark.parametrize(
        "fixture_name",
        [
            "model_bd_layered_torus",
            "model_cq_layered_torus",
            "model_ocp_layered_torus",
        ],
    )
    def test_geometry_init(self, fixture_name, request):
        solids = request.getfixturevalue(fixture_name)
        material_names = ["material"] * len(solids)
        geom = sm.Geometry(solids, material_names)

        if hasattr(solids[0], "wrapped"):
            assert geom.solids == [s.wrapped for s in solids]
        else:
            assert geom.solids == solids
        assert geom.material_names == material_names

    def test_geometry_init_wrong_materials(self, model_bd_layered_torus):
        solids = model_bd_layered_torus
        material_names = ["material"] * (len(solids) - 1)
        with pytest.raises(ValueError):
            sm.Geometry(solids, material_names)

    def test_geometry_init_ocp_face(self):
        face = bd.Face.make_rect(10, 10).wrapped
        geom = sm.Geometry(surfaces=[face], surface_boundary_conditions=["vacuum"])
        assert geom.faces == [face]


class TestGeometryImportExport:
    def test_step_import_compound(self, model_bd_layered_torus):
        cmp = bd.Compound(model_bd_layered_torus)
        bd.export_step(cmp, "model.step")
        sm.Geometry.from_step("model.step", material_names=[""] * 3)
        with pytest.raises(ValueError):
            sm.Geometry.from_step("model.step", material_names=[""] * 2)

    def test_step_import_solid(self, model_bd_layered_torus):
        bd.export_step(model_bd_layered_torus[0], "layer.step")
        sm.Geometry.from_step("layer.step", material_names=[""])

    def test_brep_import_compound(self, model_bd_layered_torus):
        cmp = bd.Compound(model_bd_layered_torus)
        bd.export_brep(cmp, "model.brep")
        sm.Geometry.from_brep("model.brep", material_names=[""] * 3)
        with pytest.raises(ValueError):
            sm.Geometry.from_brep("model.brep", material_names=[""] * 2)

    def test_brep_import_solid(self, model_bd_layered_torus):
        bd.export_brep(model_bd_layered_torus[0], "layer.brep")
        sm.Geometry.from_brep("layer.brep", material_names=[""])


class TestGeometryOperations:
    def test_geometry_imprint(self, geom_bd_layered_torus):
        geom_bd_layered_torus.imprint()

    def test_geometry_imprint_keeps_order_and_faces(self):
        small = bd.Pos(0, 5, 10) * bd.Solid.make_box(4, 4, 6)
        large = bd.Solid.make_box(10, 10, 10)
        face = bd.Pos(50, 0, 0) * bd.Face.make_rect(10, 10)
        geom = sm.Geometry([small, large], ["m_small", "m_large"], [face], ["vacuum"])
        imprinted = geom.imprint()

        for before, after in zip(geom.solids, imprinted.solids, strict=True):
            assert bd.Solid(after).volume == pytest.approx(
                bd.Solid(before).volume, rel=1e-6
            )
        assert imprinted.material_names == ["m_small", "m_large"]
        assert len(imprinted.faces) == 1
        assert imprinted.face_boundary_conditions == ["vacuum"]
        assert len(bd.Solid(imprinted.solids[1]).faces()) > 6
