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
