import build123d as bd
import pytest
import stellarmesh as sm
from pymoab.rng import Range
from stellarmesh.moab import _is_group_for, _parse_group_value


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


class TestParseGroupValue:
    """Tests for the physical-group name parser (per docs/format.rst).

    Stellarmesh's writer emits canonical DAGMC groups as ``<prefix>:<slug>``
    (e.g. ``mat:iron``), but conforming ``.msh`` producers like basalt
    emit URL-encoded names like ``tag=N&material=<slug>``. The parser must
    handle both forms without corrupting the slug.
    """

    # --- Legacy short-prefix form ---

    def test_legacy_mat_simple(self):
        assert _parse_group_value("mat:iron", "mat", "material") == "iron"

    def test_legacy_mat_with_dot(self):
        # Slugs may contain dots (e.g. body-suffix notation like ``foo.b1``).
        assert (
            _parse_group_value("mat:PLASMA_1.b1", "mat", "material")
            == "PLASMA_1.b1"
        )

    def test_legacy_boundary_simple(self):
        assert (
            _parse_group_value("boundary:vacuum", "boundary", "boundary") == "vacuum"
        )

    def test_legacy_boundary_with_punctuation(self):
        assert (
            _parse_group_value("boundary:reflecting-xy", "boundary", "boundary")
            == "reflecting-xy"
        )

    # --- URL-encoded form ---

    def test_url_encoded_material(self):
        assert (
            _parse_group_value("tag=1&material=foo", "mat", "material") == "foo"
        )

    def test_url_encoded_material_with_dot(self):
        assert (
            _parse_group_value(
                "tag=1&material=PLASMA_1.b1", "mat", "material"
            )
            == "PLASMA_1.b1"
        )

    def test_url_encoded_material_key_first(self):
        # Key order should not matter for parse_qs.
        assert (
            _parse_group_value("material=iron&tag=3", "mat", "material") == "iron"
        )

    def test_url_encoded_surface(self):
        # Surface boundary groups carry forward_volume / reverse_volume, not
        # a ``boundary=`` key, so a spec-conforming surface group has no
        # boundary value to extract.
        assert (
            _parse_group_value(
                "tag=12&forward_volume=3&reverse_volume=4",
                "boundary",
                "boundary",
            )
            is None
        )

    # --- URL-encoded form smuggled behind the legacy prefix ---
    # This is the empirical corruption pattern the bug introduces: if a
    # downstream write of a buggy ``mat_name`` ever lands in a group, we
    # should still recover the real slug.

    def test_prefix_plus_url_encoded(self):
        # ``mat:tag=1&material=PLASMA_1.b1`` -- defensively recover the slug.
        assert (
            _parse_group_value(
                "mat:tag=1&material=PLASMA_1.b1", "mat", "material"
            )
            == "PLASMA_1.b1"
        )

    # --- Negative / boundary cases ---

    def test_wrong_prefix_returns_none(self):
        assert _parse_group_value("boundary:vacuum", "mat", "material") is None

    def test_unrelated_group_returns_none(self):
        assert _parse_group_value("graveyard", "mat", "material") is None

    def test_empty_returns_none(self):
        assert _parse_group_value("", "mat", "material") is None

    def test_is_group_for_legacy(self):
        assert _is_group_for("mat:iron", "mat", "material")
        assert _is_group_for("boundary:vacuum", "boundary", "boundary")

    def test_is_group_for_url_encoded(self):
        assert _is_group_for("tag=1&material=foo", "mat", "material")

    def test_is_group_for_rejects_wrong_prefix(self):
        assert not _is_group_for("graveyard", "mat", "material")
        assert not _is_group_for("mat:iron", "boundary", "boundary")
