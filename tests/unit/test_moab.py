import build123d as bd
import gmsh
import numpy as np
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

    def test_volume_mesh_adds_surface_vertices_only(self, geom_bd_sphere):
        mesh = sm.VolumeMesh.from_geometry(
            geom_bd_sphere, sm.GmshVolumeOptions(max_mesh_size=5)
        )
        model = sm.DAGMCModel.from_mesh(mesh)
        with mesh:
            nodes = gmsh.model.mesh.get_nodes()[0]
            surface_nodes = np.unique(gmsh.model.mesh.get_elements_by_type(2)[1])
        assert len(surface_nodes) < len(nodes)
        assert len(model._core.get_entities_by_dimension(0, 0)) == len(surface_nodes)

    def test_adds_nodes_of_embedded_points(self):
        # Nodes of a point embedded in a surface are used by its triangles but are
        # not classified on the surface or its boundary.
        mesh = sm.Mesh()
        with mesh:
            gmsh.model.occ.add_box(0, 0, 0, 10, 10, 10)
            point = gmsh.model.occ.add_point(5, 5, 0)
            gmsh.model.occ.synchronize()
            bottom = gmsh.model.get_entities_in_bounding_box(-1, -1, -1, 11, 11, 1, 2)
            gmsh.model.mesh.embed(0, [point], 2, bottom[0][1])
            gmsh.option.set_number("Mesh.MeshSizeMax", 4)
            gmsh.model.mesh.generate(2)
            surface_nodes = np.unique(gmsh.model.mesh.get_elements_by_type(2)[1])
            mesh._dirty = True
        model = sm.DAGMCModel.from_mesh(mesh)
        assert len(model._core.get_entities_by_dimension(0, 0)) == len(surface_nodes)

    @pytest.mark.parametrize("extra", ["orphan_node", "quad"])
    def test_skips_surface_nodes_without_triangles(self, geom_bd_sphere, extra):
        mesh = sm.SurfaceMesh.from_geometry(
            geom_bd_sphere, sm.GmshSurfaceOptions(max_mesh_size=5)
        )
        with mesh:
            surface = gmsh.model.get_entities(2)[0][1]
            surface_nodes = np.unique(gmsh.model.mesh.get_elements_by_type(2)[1])
            first = int(gmsh.model.mesh.get_max_node_tag()) + 1
            if extra == "orphan_node":
                gmsh.model.mesh.add_nodes(2, surface, [first], [0.0, 0.0, 0.0])
            else:
                quad = list(range(first, first + 4))
                coords = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
                gmsh.model.mesh.add_nodes(2, surface, quad, coords)
                gmsh.model.mesh.add_elements_by_type(surface, 3, [], quad)
            mesh._dirty = True
        model = sm.DAGMCModel.from_mesh(mesh)
        assert len(model._core.get_entities_by_dimension(0, 0)) == len(surface_nodes)

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

    def test_skips_nodes_without_elements(self, geom_bd_sphere):
        mesh = sm.VolumeMesh.from_geometry(
            geom_bd_sphere, sm.GmshVolumeOptions(max_mesh_size=5)
        )
        with mesh:
            tet_nodes = np.unique(gmsh.model.mesh.get_elements_by_type(4)[1])
            orphan = int(gmsh.model.mesh.get_max_node_tag()) + 1
            gmsh.model.add_discrete_entity(0, 999)
            gmsh.model.mesh.add_nodes(0, 999, [orphan], [100.0, 0, 0])
            mesh._dirty = True
        model = sm.MOABVolumeModel.from_mesh(mesh)
        assert len(model._core.get_entities_by_dimension(0, 0)) == len(tet_nodes)
