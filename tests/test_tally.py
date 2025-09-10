"""Continuous integration tests with OpenMC."""

import logging
import random
import tempfile
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Literal, Sequence, Type

import build123d as bd
import numpy as np
import openmc
import openmc.stats
import pytest

import stellarmesh as sm

RELATIVE_TOLERANCE_PERCENT = 2.5 / 100

logger = logging.getLogger(__name__)


class Model(ABC):
    """CSG and DAGMC model representations with materials, tallies, and source."""

    bounded: bool = True
    fail_if_identical = True

    @cached_property
    @abstractmethod
    def materials(self) -> openmc.Materials | None:
        """Generate the neutron source."""
        ...

    @cached_property
    def settings(self) -> openmc.Settings:
        settings = openmc.Settings()
        settings.batches = 20
        settings.inactive = 0
        settings.particles = 500
        settings.run_mode = "fixed source"
        settings.source = self.source
        settings.seed = random.randint(0, 2**31 - 1)
        return settings

    @abstractmethod
    def get_tallies(self, type: Literal["CSG", "CAD"]) -> openmc.Tallies:
        """Generate the tallies."""
        ...

    @cached_property
    @abstractmethod
    def source(self) -> openmc.IndependentSource:
        """Generate the neutron source."""
        ...

    @cached_property
    @abstractmethod
    def geom_csg(self) -> openmc.Geometry:
        """Generate the CSG model."""
        ...

    @cached_property
    @abstractmethod
    def geom_cad(self) -> openmc.Geometry:
        with tempfile.TemporaryDirectory(delete=False) as tmp_path:  # pyright: ignore[reportCallIssue]
            self.dagmc.write(str(Path(tmp_path) / "dagmc.h5m"))
            universe = openmc.DAGMCUniverse(Path(tmp_path) / "dagmc.h5m")
            if self.bounded:
                universe = universe.bounded_universe()
            geometry = openmc.Geometry(universe)
        return geometry

    @cached_property
    def solids(self) -> Sequence[bd.Solid] | None:
        """Generate the CAD model solids."""
        return None

    @cached_property
    def surfaces(self) -> tuple[Sequence[bd.Face] | None, Sequence[str] | None]:
        """Generate the CAD model surfaces. Return (surfaces, boundary_conditions)."""
        return None, None

    @cached_property
    def sm_geom(self) -> sm.Geometry:
        """Generate the Stellarmesh Geometry."""
        mat_names: list[str] | None = (
            [m.name for m in self.materials] if self.materials else None
        )
        solids = self.solids
        surfaces, bcs = self.surfaces
        geom = sm.Geometry(solids, mat_names, surfaces, bcs)
        return geom

    @cached_property
    def mesh(self) -> sm.SurfaceMesh:
        """Generate the mesh."""
        geom = self.sm_geom
        msh = sm.SurfaceMesh.from_geometry(
            geom, sm.OCCSurfaceOptions(tol_angular_deg=0.2)
        )
        return msh

    @cached_property
    def dagmc(self) -> sm.DAGMCModel:
        sm_dagmc_model = sm_dagmc_model = sm.DAGMCModel.from_mesh(self.mesh)
        return sm_dagmc_model

    def build_model(self, type: Literal["CSG", "CAD"]) -> openmc.Model:
        geometry = self.geom_cad if type == "CAD" else self.geom_csg
        model = openmc.Model(geometry=geometry)
        if self.materials:
            model.materials = self.materials
        model.settings = self.settings
        model.tallies = self.get_tallies(type)
        return model


class NestedSpheres(Model):
    """Two nested spheres."""

    radius1 = 20
    radius2 = 5

    @cached_property
    def materials(self):
        mat1 = openmc.Material(name="1")
        mat1.add_nuclide("Fe56", 1)
        mat1.set_density("g/cm3", 1)

        mat2 = openmc.Material(name="2")
        mat2.add_nuclide("Be9", 1)
        mat2.set_density("g/cm3", 1)
        return openmc.Materials([mat1, mat2])

    def get_tallies(self, type):
        mat_filter = openmc.MaterialFilter(self.materials)
        tally = openmc.Tally(name="flux_tally")
        tally.filters = [mat_filter]
        tally.scores = ["flux"]

        return openmc.Tallies([tally])

    @cached_property
    def source(self):
        source = openmc.IndependentSource()
        source.space = openmc.stats.Point((0, 0, 0))
        source.angle = openmc.stats.Isotropic()
        source.energy = openmc.stats.Discrete([14e6], [1])
        return source

    @cached_property
    def geom_csg(self):
        surface1 = openmc.Sphere(r=self.radius1)
        surface2 = openmc.Sphere(r=self.radius1 + self.radius2, boundary_type="vacuum")
        region1 = -surface1
        region2 = +surface1 & -surface2
        cell1 = openmc.Cell(fill=self.materials[0], region=region1)
        cell2 = openmc.Cell(fill=self.materials[1], region=region2)
        csg_geometry = openmc.Geometry([cell1, cell2])
        return csg_geometry

    @cached_property
    def solids(self):
        s1 = bd.Solid.make_sphere(self.radius1)
        s2 = bd.Solid.thicken(s1.faces()[0], self.radius2)
        return [s1, s2]


class NestedCylinders(Model):
    """Two nested cylinders."""

    radius1 = 15
    radius2 = 8
    height1 = 15
    height2 = 8

    @cached_property
    def materials(self):
        mat1 = openmc.Material(name="1")
        mat1.add_nuclide("Fe56", 1)
        mat1.set_density("g/cm3", 1)

        mat2 = openmc.Material(name="2")
        mat2.add_nuclide("Be9", 1)
        mat2.set_density("g/cm3", 1)
        return openmc.Materials([mat1, mat2])

    def get_tallies(self, type):
        mat_filter = openmc.MaterialFilter(self.materials)
        tally = openmc.Tally(name="flux_tally")
        tally.filters = [mat_filter]
        tally.scores = ["flux"]

        return openmc.Tallies([tally])

    @cached_property
    def source(self):
        source = openmc.IndependentSource()
        source.space = openmc.stats.Point((0, 0, 0))
        source.angle = openmc.stats.Isotropic()
        source.energy = openmc.stats.Discrete([14e6], [1])
        return source

    @cached_property
    def geom_csg(self):
        surface1 = openmc.ZCylinder(
            r=self.radius1,
        )
        zmin1 = openmc.ZPlane(
            z0=-self.height1 / 2,
        )
        zmax1 = openmc.ZPlane(
            z0=self.height1 / 2,
        )
        surface2 = openmc.ZCylinder(
            r=self.radius1 + self.radius2, boundary_type="vacuum"
        )
        zmin2 = openmc.ZPlane(
            z0=-(self.height1 + self.height2) / 2, boundary_type="vacuum"
        )
        zmax2 = openmc.ZPlane(
            z0=(self.height1 + self.height2) / 2, boundary_type="vacuum"
        )

        region1 = -surface1 & +zmin1 & -zmax1
        region2 = ~region1 & (-surface2 & +zmin2 & -zmax2)

        cell1 = openmc.Cell(fill=self.materials[0], region=region1)
        cell2 = openmc.Cell(fill=self.materials[1], region=region2)

        csg_geometry = openmc.Geometry([cell1, cell2])
        return csg_geometry

    @cached_property
    def solids(self):
        s1 = bd.Cylinder(self.radius1, self.height1, align=bd.Align.CENTER).solid()
        s2 = bd.Cylinder(
            self.radius1 + self.radius2,
            self.height1 + self.height2,
            align=bd.Align.CENTER,
        ).solid()  # type: ignore
        s2: bd.Solid = s2.cut(s1)  # type: ignore
        return [s1, s2]


class Torus(Model):
    """Torus."""

    major_radius: float = 10.0
    minor_radius: float = 1.0

    @cached_property
    def materials(self):
        mat1 = openmc.Material(name="1")
        mat1.add_nuclide("Fe56", 1)
        mat1.set_density("g/cm3", 1)
        return openmc.Materials([mat1])

    def get_tallies(self, type):
        mat_filter = openmc.MaterialFilter(self.materials[0])
        tally = openmc.Tally(name="mat1_flux_tally")
        tally.filters = [mat_filter]
        tally.scores = ["flux"]

        return openmc.Tallies([tally])

    @cached_property
    def source(self):
        source = openmc.IndependentSource()
        source.space = openmc.stats.CylindricalIndependent(
            openmc.stats.Discrete([self.major_radius], [1]),
            openmc.stats.Uniform(0, 2 * np.pi),
            openmc.stats.Discrete([0], [1]),
        )
        source.energy = openmc.stats.Discrete([14e6], [1])

        return source

    @cached_property
    def geom_csg(self):
        torus_surface = openmc.ZTorus(
            a=self.major_radius,
            b=self.minor_radius,
            c=self.minor_radius,
        )
        torus_cell = openmc.Cell(fill=self.materials[0], region=-torus_surface)

        # Create a bounding box to allow torus re-entry
        box_half_side = self.major_radius + self.minor_radius + 5.0  # Add a buffer
        xmin = openmc.XPlane(-box_half_side, boundary_type="vacuum")
        xmax = openmc.XPlane(box_half_side, boundary_type="vacuum")
        ymin = openmc.YPlane(-box_half_side, boundary_type="vacuum")
        ymax = openmc.YPlane(box_half_side, boundary_type="vacuum")
        zmin = openmc.ZPlane(-box_half_side, boundary_type="vacuum")
        zmax = openmc.ZPlane(box_half_side, boundary_type="vacuum")
        bounding_box_region = +xmin & -xmax & +ymin & -ymax & +zmin & -zmax

        void_region = bounding_box_region & +torus_surface
        void_cell = openmc.Cell(fill=None, region=void_region)

        csg_geometry = openmc.Geometry([torus_cell, void_cell])
        return csg_geometry

    @cached_property
    def solids(self):
        s1 = bd.Solid.make_torus(self.major_radius, self.minor_radius)
        return [s1]


class TorusSurface(Model):
    """Torus."""

    major_radius: float = 10.0
    minor_radius: float = 1.0
    dagmc_bounded = False
    fail_if_identical = False

    @cached_property
    def materials(self):
        return None

    def get_tallies(self, type):
        surface_ids = (
            list(self.geom_csg.get_all_surfaces())
            if type == "CSG"
            else [s.global_id for s in self.dagmc.surfaces]
        )
        current_tally = openmc.Tally(name="current")
        current_tally.scores = ["current"]
        surface_filter = openmc.SurfaceFilter(surface_ids)
        current_tally.filters = [surface_filter]
        return openmc.Tallies([current_tally])

    @cached_property
    def source(self):
        """14 MeV ring source centered at major_radius."""
        source = openmc.IndependentSource()
        source.space = openmc.stats.CylindricalIndependent(
            openmc.stats.Discrete([self.major_radius], [1]),
            openmc.stats.Uniform(0, 2 * np.pi),
            openmc.stats.Discrete([0], [1]),
        )
        source.energy = openmc.stats.Discrete([14e6], [1])

        return source

    @cached_property
    def geom_csg(self):
        """CSG geometry: torus cell bounded by a vacuum box."""
        torus_surface = openmc.ZTorus(
            a=self.major_radius,
            b=self.minor_radius,
            c=self.minor_radius,
        )
        torus_surface.boundary_type = "vacuum"
        torus_cell = openmc.Cell(fill=None, region=-torus_surface)

        csg_geometry = openmc.Geometry([torus_cell])
        return csg_geometry

    @cached_property
    def surfaces(self):
        f1: bd.Face = bd.Solid.make_torus(self.major_radius, self.minor_radius).face()  # type: ignore
        return [f1], ["vacuum"]


@pytest.mark.parametrize(
    "model_cls", [NestedSpheres, NestedCylinders, Torus, TorusSurface]
)
def test_model(model_cls: Type[Model], tmp_path: Path):
    """Run CSG and DAGMC models and compare tallies for agreement."""
    openmc.reset_auto_ids()
    model = model_cls()

    csg_model = model.build_model("CSG")
    cad_model = model.build_model("CAD")

    logger.info("Running CSG model")
    csg_model.export_to_model_xml(path=str(tmp_path))
    output_file_from_csg = csg_model.run(cwd=str(tmp_path / "csg"))
    logger.info(output_file_from_csg)

    logger.info("Running DAGMC model")
    cad_model.export_to_model_xml(path=str(tmp_path))
    output_file_from_cad = cad_model.run(cwd=str(tmp_path / "cad"))
    logger.info(output_file_from_cad)

    with (
        openmc.StatePoint(output_file_from_csg) as sp_from_csg,
        openmc.StatePoint(output_file_from_cad) as sp_from_cad,
    ):
        for csg_tally, dagmc_tally in zip(
            csg_model.tallies, cad_model.tallies, strict=True
        ):
            csg_tally_result = sp_from_csg.get_tally(name=csg_tally.name)
            cad_tally_result = sp_from_cad.get_tally(name=dagmc_tally.name)
            logger.info(f"Comparing tally: {csg_tally.name}")
            logger.info(f"CSG result: {csg_tally_result.mean}")
            logger.info(f"CAD result: {cad_tally_result.mean}")

            assert np.allclose(
                csg_tally_result.mean,
                cad_tally_result.mean,
                rtol=RELATIVE_TOLERANCE_PERCENT,
            ), f"CSG and CAD tallies do not match within {RELATIVE_TOLERANCE_PERCENT}%"

            if model_cls.fail_if_identical:
                assert not np.allclose(
                    csg_tally_result.mean, cad_tally_result.mean, rtol=0.000001
                ), (
                    "CSG and CAD tallies are too close, have we loaded the same data twice?"
                )
