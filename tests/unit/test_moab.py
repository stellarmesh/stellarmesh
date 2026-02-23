import build123d as bd
import pytest
import stellarmesh as sm
from pymoab.rng import Range


@pytest.fixture(scope="module")
def dagmc_model():
    solid1 = bd.Solid.make_sphere(10.0)
    solid2 = bd.thicken(solid1.faces()[0], 10.0).solid()
    geom = sm.Geometry([solid1, solid2], ["iron", "iron"])
    mesh = sm.SurfaceMesh.from_geometry(geom, sm.GmshSurfaceOptions(max_mesh_size=5))
    return sm.DAGMCModel.from_mesh(mesh)


class TestDAGMCModel:
    def test_surfaces(self, dagmc_model):
        assert isinstance(dagmc_model.surfaces, list)
        assert len(dagmc_model.surfaces) == 2
        assert isinstance(dagmc_model.surfaces[0], sm.DAGMCSurface)

    def test_volumes(self, dagmc_model):
        assert isinstance(dagmc_model.volumes, list)
        assert len(dagmc_model.volumes) == 2
        assert isinstance(dagmc_model.volumes[0], sm.DAGMCVolume)

    def test_global_id(self, dagmc_model):
        assert dagmc_model.surfaces[0].global_id == 1
        assert dagmc_model.volumes[0].global_id == 1

    def test_adjacent_surfaces(self, dagmc_model):
        vol = dagmc_model.volumes[0]
        surfaces = vol.adjacent_surfaces
        assert len(surfaces) == 1
        assert surfaces == [dagmc_model.surfaces[0]]

    def test_adjacent_volumes(self, dagmc_model):
        surf = dagmc_model.surfaces[0]
        volumes = surf.adjacent_volumes
        assert len(volumes) == 2
        assert volumes == [dagmc_model.volumes[0], dagmc_model.volumes[1]]
        assert surf.forward_volume == dagmc_model.volumes[0]
        assert surf.reverse_volume == dagmc_model.volumes[1]

    def test_tets(self, dagmc_model):
        assert dagmc_model.tets.empty()

    def test_triangles(self, dagmc_model):
        all_tris = dagmc_model.triangles
        assert isinstance(all_tris, Range)
        surf_tris = dagmc_model.surfaces[0].triangles
        assert isinstance(surf_tris, Range)
        assert all_tris.contains(surf_tris)

    def test_material(self, dagmc_model):
        vol = dagmc_model.volumes[0]
        assert vol.material == "iron"
        assert "mat:iron" in {group.name for group in vol.groups}

        vol.material = "plastic"
        assert vol.material == "plastic"
        vol_group_names = {group.name for group in vol.groups}
        assert "mat:iron" not in vol_group_names
        assert "mat:plastic" in vol_group_names

        all_group_names = {group.name for group in dagmc_model.groups}
        assert "mat:plastic" in all_group_names

    def test_group(self, dagmc_model):
        vol = dagmc_model.volumes[0]
        surf = dagmc_model.surfaces[0]

        group = dagmc_model.create_group("test_group")
        assert group.name == "test_group"

        group.name = "funny group"
        assert group.name == "funny group"

        group.add(vol)
        assert vol in group
        assert group.volumes == [vol]

        group.remove(vol)
        assert vol not in group
        assert group.volumes == []

        group.add(surf)
        assert group.surfaces == [surf]
        group.remove(surf)
        assert group.surfaces == []

    def test_repr(self, dagmc_model):
        surf = dagmc_model.surfaces[0]
        repr(surf)

        vol = dagmc_model.volumes[0]
        repr(vol)

        group = dagmc_model.groups[0]
        repr(group)

    def test_hash(self, dagmc_model):
        objects = {dagmc_model.surfaces[0], dagmc_model.volumes[0]}
        assert len(objects) == 2


class TestMOABModel:
    def test_moabmodel_from_h5m(self, geom_bd_sphere):
        mesh = sm.VolumeMesh.from_geometry(
            geom_bd_sphere, sm.GmshVolumeOptions(max_mesh_size=5)
        )
        model = sm.MOABModel.from_mesh(mesh)


class TestMOABVolumeModel:
    @pytest.fixture(scope="class")
    def volume_model(self):
        solid = bd.Solid.make_sphere(10.0)
        geom = sm.Geometry([solid], [""])
        mesh = sm.VolumeMesh.from_geometry(geom, sm.GmshVolumeOptions(max_mesh_size=5))
        return sm.MOABVolumeModel.from_mesh(mesh)

    def test_has_tets(self, volume_model):
        tets = volume_model.tets
        assert isinstance(tets, Range)
        assert not tets.empty()

    def test_no_triangles_in_root(self, volume_model):
        tris = volume_model.triangles
        assert tris.empty()

    def test_write_and_read(self, volume_model, tmp_path):
        out = tmp_path / "volume.h5m"
        volume_model.write(out)
        assert out.exists()
        reloaded = sm.MOABVolumeModel(out)
        assert not reloaded.tets.empty()

    def test_tet_count_reasonable(self, volume_model):
        assert len(volume_model.tets) > 100
