"""Meshing tests."""

import copy
import gc
import importlib.resources
import math
import subprocess
from pathlib import Path

import build123d as bd
import gmsh
import pytest

import stellarmesh as sm
from .. import resources


class TestMeshMetadataAndImmutability:
    @pytest.fixture
    def mesh_sphere_occ(self, geom_bd_sphere):
        mesh = sm.SurfaceMesh.from_geometry(
            geom_bd_sphere,
            sm.OCCSurfaceOptions(tol_angular_deg=math.degrees(0.5)),
        )
        return copy.deepcopy(mesh)

    def test_entity_metadata(self, mesh_sphere_occ):
        mesh = mesh_sphere_occ
        mesh.entity_metadata(2, 1).forward_volume  # noqa: B018
        mesh.entity_metadata(2, 1).forward_volume = 10
        assert mesh.entity_metadata(2, 1).forward_volume == 10
        with pytest.raises(
            AttributeError, match=r".*has no attribute invalid_attrib.*"
        ):
            mesh.entity_metadata(2, 1).invalid_attrib  # noqa: B018
        with pytest.raises(
            AttributeError, match=r".*has no attribute invalid_attrib.*"
        ):
            mesh.entity_metadata(2, 1).invalid_attrib = 10

    def test_mesh_immutability(self, mesh_sphere_occ, tmp_path):
        mesh_sphere_occ.write(tmp_path / "mesh.msh")

        mesh1 = sm.Mesh(tmp_path / "mesh.msh")
        mesh1.entity_metadata(2, 1).forward_volume  # noqa: B018
        mesh1.entity_metadata(2, 1).forward_volume = 10
        assert mesh1.entity_metadata(2, 1).forward_volume == 10

        mesh2 = sm.Mesh(tmp_path / "mesh.msh")
        assert mesh2.entity_metadata(2, 1).forward_volume == 1

    def test_from_geometry_writes_mesh_once(self, geom_bd_nestedspheres, monkeypatch):
        """Metadata setters inside from_geometry must not each rewrite the file."""
        writes = []
        original_write = gmsh.write

        def counting_write(filename):
            writes.append(filename)
            original_write(filename)

        monkeypatch.setattr(gmsh, "write", counting_write)
        sm.SurfaceMesh.from_geometry(geom_bd_nestedspheres, sm.OCCSurfaceOptions())
        assert len(writes) == 1

    def test_metadata_discarded_on_exception(self, mesh_sphere_occ):
        with pytest.raises(ValueError, match="abort"), mesh_sphere_occ:
            mesh_sphere_occ.entity_metadata(2, 1).forward_volume = 10
            raise ValueError("abort")
        assert mesh_sphere_occ.entity_metadata(2, 1).forward_volume == 1


class TestSurfaceMesh:
    @pytest.mark.parametrize(
        "geom_name,num_elements_occ,num_elements_gmsh",
        [
            ("geom_bd_sphere", 306, 850),
            ("geom_cq_sphere", 306, 850),
            ("geom_bd_layered_torus", 5408, 2814),
            ("geom_cq_layered_torus", 5408, 2814),
            ("geom_bd_nestedspheres", 612, 1086),
            ("geom_cq_nestedspheres", 612, 1086),
        ],
    )
    def test_surface_mesh_num_elements(
        self,
        geom_name: str,
        num_elements_occ: int,
        num_elements_gmsh: int,
        request: pytest.FixtureRequest,
    ):
        geom = request.getfixturevalue(geom_name)

        mesh_occ = sm.SurfaceMesh.from_geometry(
            geom,
            sm.OCCSurfaceOptions(tol_angular_deg=math.degrees(0.5)),
        )
        with mesh_occ:
            num_elements = len(gmsh.model.mesh.get_elements(2, -1)[1][0])
        assert num_elements == num_elements_occ, (
            f"Number of elements in OCC mesh ({num_elements}) does not match expected "
            f"({num_elements_occ}) for model {geom_name}. Mesh: {mesh_occ._mesh_filename}"
        )

        mesh_gmsh = sm.SurfaceMesh.from_geometry(
            geom,
            sm.GmshSurfaceOptions(min_mesh_size=0.5, max_mesh_size=2),
        )
        with mesh_gmsh:
            num_elements = len(gmsh.model.mesh.get_elements(2, -1)[1][0])
        assert num_elements == num_elements_gmsh, (
            f"Number of elements in GMSH mesh ({num_elements}) does not match expected "
            f"({num_elements_gmsh}) for model {geom_name}. Mesh: {mesh_gmsh._mesh_filename}"
        )

    def test_mesh_geom_imprintedboxes(self, geom_imprintedboxes):
        options = [
            ("OCC", sm.OCCSurfaceOptions(tol_angular_deg=math.degrees(1))),
            ("GMSH", sm.GmshSurfaceOptions(min_mesh_size=0.5, max_mesh_size=2)),
        ]
        for backend, option in options:
            mesh = sm.SurfaceMesh.from_geometry(geom_imprintedboxes, option)
            with mesh:
                surface_tags = gmsh.model.get_entities(2)
                assert len(surface_tags) == 13, (
                    f"Number of surfaces in {backend} mesh ({len(surface_tags)}) does not "
                    f"match expected (13). Mesh: {mesh._mesh_filename}"
                )

    def test_mesh_surface_capped_torus_bcs(self, geom_bd_capped_torus):
        """Test material and BC assignments for a capped torus."""
        for options in [sm.GmshSurfaceOptions(), sm.OCCSurfaceOptions()]:
            mesh = sm.SurfaceMesh.from_geometry(geom_bd_capped_torus, options)
            with mesh:
                for dim, tag in gmsh.model.get_entities(2):
                    if tag in [2, 3, 5, 6, 8, 9, 11, 12]:
                        assert (
                            mesh.entity_metadata(dim, tag).boundary_condition
                            == "reflecting"
                        )

    def test_mesh_torus_single_surface_bc(self, geom_bd_torus_single_surface):
        for options in [sm.GmshSurfaceOptions(), sm.OCCSurfaceOptions()]:
            mesh = sm.SurfaceMesh.from_geometry(geom_bd_torus_single_surface, options)
            with mesh:
                for dim, tag in gmsh.model.get_entities(2):
                    if tag in [1]:
                        assert (
                            mesh.entity_metadata(dim, tag).boundary_condition
                            == "vacuum"
                        )


class TestVolumeMesh:
    @pytest.mark.parametrize(
        "geom_name,num_elements_gmsh",
        [
            ("geom_bd_sphere", 2648),
            ("geom_bd_layered_torus", 6501),
            ("geom_bd_nestedspheres", 2828),
        ],
    )
    def test_volume_mesh_num_elements(
        self,
        geom_name: str,
        num_elements_gmsh: int,
        request: pytest.FixtureRequest,
    ):
        geom = request.getfixturevalue(geom_name)

        mesh = sm.VolumeMesh.from_geometry(
            geom,
            sm.GmshVolumeOptions(min_mesh_size=0.5, max_mesh_size=2),
        )
        with mesh:
            num_elements = len(gmsh.model.mesh.get_elements(3, -1)[1][0])
        assert num_elements_gmsh == num_elements, (
            f"Number of elements in volume mesh ({num_elements}) does not match expected "
            f"({num_elements_gmsh}) for model {geom_name}. Mesh: {mesh._mesh_filename}"
        )

    def test_mesh_volume_imprintedboxes(self, geom_imprintedboxes):
        mesh = sm.VolumeMesh.from_geometry(
            geom_imprintedboxes,
            sm.GmshVolumeOptions(min_mesh_size=0.5, max_mesh_size=2),
        )
        with mesh:
            volume_tags = gmsh.model.get_entities(3)
            assert len(volume_tags) == 2, (
                f"Number of volumes ({len(volume_tags)}) does not match expected (2). "
                f"Mesh: {mesh._mesh_filename}"
            )
            surface_tags = gmsh.model.get_entities(2)
            assert len(surface_tags) == 13, (
                f"Number of surfaces ({len(surface_tags)}) does not match expected (13). "
                f"Mesh: {mesh._mesh_filename}"
            )

    def test_mesh_export_exodus(self, geom_bd_layered_torus, tmp_path: Path):
        mesh = sm.VolumeMesh.from_geometry(
            geom_bd_layered_torus,
            sm.GmshVolumeOptions(min_mesh_size=0.5, max_mesh_size=2),
        )
        mesh.write(tmp_path / "out.exo", use_meshio=True)


class TestMeshMerge:
    @pytest.fixture
    def sphere_meshes(self, model_bd_sphere):
        options = sm.OCCSurfaceOptions(tol_angular_deg=math.degrees(0.5))
        left = sm.SurfaceMesh.from_geometry(
            sm.Geometry(model_bd_sphere, ["a"], solid_names=["s1"]), options
        )
        moved = [bd.Pos(100, 0, 0) * model_bd_sphere[0]]
        right = sm.SurfaceMesh.from_geometry(
            sm.Geometry(moved, ["b"], solid_names=["s2"]), options
        )
        return left, right

    def test_merge(self, sphere_meshes, tmp_path: Path):
        left, right = sphere_meshes
        merged = sm.Mesh.merge({"left": left, "right": right})

        with left:
            max_left_tag = max(t for _, t in gmsh.model.get_entities())
        right_volume = merged.merge_map["right"][(3, 1)]
        assert merged.merge_map["left"][(3, 1)] == 1
        assert right_volume == 10 ** len(str(max_left_tag)) + 1

        with merged:
            volumes = [t for _, t in gmsh.model.get_entities(3)]
            surfaces = [t for _, t in gmsh.model.get_entities(2)]
        assert sorted(volumes) == [1, right_volume]
        expected = {1: ("left", "s1", "a"), right_volume: ("right", "s2", "b")}
        for tag, (source, name, material) in expected.items():
            md = merged.entity_metadata(3, tag).as_dict()
            assert (md["source"], md["name"], md["material"]) == (
                source,
                name,
                material,
            )
        for tag in surfaces:
            assert merged.entity_metadata(2, tag).forward_volume in volumes

        model = sm.DAGMCModel.from_mesh(merged)
        assert len(model.volumes) == 2
        model.write(tmp_path / "merged.h5m")
        report = sm.DAGMCModel.check_watertight(tmp_path / "merged.h5m")
        assert report
        assert report.leaky_surfaces == []
        assert report.leaky_volumes == []

    def test_merge_material_map(self, sphere_meshes):
        left, right = sphere_meshes
        merged = sm.Mesh.merge(
            {"left": left, "right": right},
            material_map=lambda label, tag, material: (
                "c" if material == "b" else material
            ),
        )
        tag = merged.merge_map["right"][(3, 1)]
        assert merged.entity_metadata(3, tag).material == "c"

    def test_merge_errors(self, sphere_meshes, geom_bd_sphere):
        left, right = sphere_meshes
        unnamed = sm.SurfaceMesh.from_geometry(geom_bd_sphere, sm.OCCSurfaceOptions())
        with pytest.raises(ValueError, match="without a material"):
            sm.Mesh.merge({"left": left, "unnamed": unnamed})
        with pytest.raises(ValueError, match="longer than"):
            sm.Mesh.merge(
                {"left": left}, material_map=lambda label, tag, material: "x" * 29
            )
        merged = sm.Mesh.merge({"left": left})
        with pytest.raises(ValueError, match="already has source metadata"):
            sm.Mesh.merge({"merged": merged, "right": right})
        with pytest.raises(ValueError, match="at least one mesh"):
            sm.Mesh.merge({})
        with left, pytest.raises(RuntimeError, match="open context"):
            sm.Mesh.merge({"left": left, "right": right})

    def test_volume_report(self):
        radius, centre = 10.0, (5.0, -3.0, 2.0)
        sphere = bd.Pos(*centre) * bd.Solid.make_sphere(radius)
        mesh = sm.SurfaceMesh.from_geometry(
            sm.Geometry([sphere], ["a"]), sm.OCCSurfaceOptions(tol_angular_deg=5)
        )
        (report,) = mesh.volume_report()
        assert report.metadata["material"] == "a"
        assert report.volume == pytest.approx(4 / 3 * math.pi * radius**3, rel=0.01)
        assert math.dist(report.centroid, centre) < 0.01 * radius

    def test_temp_file_deleted(self):
        mesh = sm.Mesh()
        path = Path(mesh._mesh_filename)
        assert path.exists()
        del mesh
        gc.collect()
        assert not path.exists()


class TestMeshUtilities:
    @pytest.fixture
    def model_bd_stellarator_plasma(self):
        with importlib.resources.path(resources, "NCSX-LCFS.brep") as path:
            return bd.import_brep(path)

    def test_mesh_overlap(self, model_bd_stellarator_plasma):
        plasma = model_bd_stellarator_plasma
        b1: bd.Solid = bd.thicken(plasma, 5).solid()  # pyright: ignore[reportAssignmentType]
        b2: bd.Solid = bd.thicken(b1.faces()[0], 5).solid()  # pyright: ignore[reportAssignmentType]

        def check_overlap(tol_linear):
            geom = sm.Geometry(solids=[b1, b2], material_names=[""] * 2)
            mesh = sm.SurfaceMesh.from_geometry(
                geom, sm.OCCSurfaceOptions(tol_linear=tol_linear, tol_angular_deg=None)
            )

            mesh.write("tmp.msh")
            dagmc_model = sm.DAGMCModel.from_mesh(mesh)
            dagmc_model.write("dagmc.h5m")

            result = subprocess.run(
                ["overlap_check", "-p", "5", "dagmc.h5m"],
                check=False,
                cwd=".",
                capture_output=True,
                text=True,
            )
            report = sm.DAGMCModel.check_overlap("dagmc.h5m", points_per_edge=5)
            return result.stdout, report

        stdout, report = check_overlap(2.5)
        assert "No overlaps were found" in stdout
        assert report
        stdout, report = check_overlap(5)
        assert "Overlap Location:" in stdout
        assert not report
        assert report.overlaps
        assert all(len(volumes) >= 2 for _, volumes in report.overlaps)

    def test_gmsh_threads(self, geom_bd_layered_torus, tmp_path: Path):
        mesh = sm.SurfaceMesh.from_geometry(
            geom_bd_layered_torus,
            sm.GmshSurfaceOptions(min_mesh_size=0.5, max_mesh_size=2, num_threads=8),
        )
