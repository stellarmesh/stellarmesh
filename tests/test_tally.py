"""Continuous integration tests with OpenMC."""

import logging
import random
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Sequence, Type

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

    @cached_property
    @abstractmethod
    def materials(self) -> openmc.Materials:
        """Generate the neutron source."""
        ...

    @abstractmethod
    def tallies(self) -> openmc.Tallies:
        """Generate the tallies."""
        ...

    @abstractmethod
    def source(self) -> openmc.IndependentSource:
        """Generate the neutron source."""
        ...

    @abstractmethod
    def csg(self) -> openmc.Model:
        """Generate the CSG model."""
        ...

    @abstractmethod
    def cad(self) -> Sequence[bd.Solid]:
        """Generate the CAD model."""
        ...

    def mesh(self) -> sm.SurfaceMesh:
        """Generate the mesh."""
        ...
        mat_names: list[str] = [m.name for m in self.materials]
        solids = self.cad()
        geom = sm.Geometry(solids, mat_names)
        msh = sm.SurfaceMesh.from_geometry(
            geom, sm.OCCSurfaceOptions(tol_angular_deg=0.2)
        )
        return msh


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

    def tallies(self):
        mat_filter = openmc.MaterialFilter(self.materials)
        tally = openmc.Tally(name="flux_tally")
        tally.filters = [mat_filter]
        tally.scores = ["flux"]

        return openmc.Tallies([tally])

    def source(self):
        source = openmc.IndependentSource()
        source.space = openmc.stats.Point((0, 0, 0))
        source.angle = openmc.stats.Isotropic()
        source.energy = openmc.stats.Discrete([14e6], [1])
        return source

    def csg(self):
        surface1 = openmc.Sphere(r=self.radius1)
        surface2 = openmc.Sphere(r=self.radius1 + self.radius2, boundary_type="vacuum")
        region1 = -surface1
        region2 = +surface1 & -surface2
        cell1 = openmc.Cell(fill=self.materials[0], region=region1)
        cell2 = openmc.Cell(fill=self.materials[1], region=region2)
        csg_geometry = openmc.Geometry([cell1, cell2])
        return openmc.Model(geometry=csg_geometry)

    def cad(self):
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

    def tallies(self):
        mat_filter = openmc.MaterialFilter(self.materials)
        tally = openmc.Tally(name="flux_tally")
        tally.filters = [mat_filter]
        tally.scores = ["flux"]

        return openmc.Tallies([tally])

    def source(self):
        source = openmc.IndependentSource()
        source.space = openmc.stats.Point((0, 0, 0))
        source.angle = openmc.stats.Isotropic()
        source.energy = openmc.stats.Discrete([14e6], [1])
        return source

    def csg(self):
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
        return openmc.Model(geometry=csg_geometry)

    def cad(self):
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

    def tallies(self):
        mat_filter = openmc.MaterialFilter(self.materials[0])
        tally = openmc.Tally(name="mat1_flux_tally")
        tally.filters = [mat_filter]
        tally.scores = ["flux"]

        return openmc.Tallies([tally])

    def source(self):
        source = openmc.IndependentSource()
        source.space = openmc.stats.CylindricalIndependent(
            openmc.stats.Discrete([self.major_radius], [1]),
            openmc.stats.Uniform(0, 2 * np.pi),
            openmc.stats.Discrete([0], [1]),
        )
        source.energy = openmc.stats.Discrete([14e6], [1])

        return source

    def csg(self):
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

        return openmc.Model(geometry=csg_geometry)

    def cad(self):
        s1 = bd.Solid.make_torus(self.major_radius, self.minor_radius)
        return [s1]


@pytest.mark.parametrize("model_cls", [NestedSpheres, NestedCylinders, Torus])
def test_model(model_cls: Type[Model], tmp_path: Path):
    openmc.reset_auto_ids()
    model = model_cls()

    settings = openmc.Settings()
    settings.batches = 20
    settings.inactive = 0
    settings.particles = 500
    settings.run_mode = "fixed source"
    settings.source = model.source()

    settings.seed = random.randint(0, 2**31 - 1)

    tallies = model.tallies()
    csg_model = model.csg()
    csg_model.materials = model.materials
    csg_model.tallies = tallies
    csg_model.settings = settings

    msh = model.mesh()
    sm_dagmc_model = sm.DAGMCModel.from_mesh(msh)
    sm_dagmc_model.write(str(tmp_path / "dagmc.h5m"))
    logger.info(f"DAGMC model written to {tmp_path / 'dagmc.h5m'}")
    universe = openmc.DAGMCUniverse(tmp_path / "dagmc.h5m").bounded_universe()
    dagmc_geometry = openmc.Geometry(universe)
    dagmc_model = openmc.Model(geometry=dagmc_geometry)
    dagmc_model.materials = model.materials
    dagmc_model.tallies = tallies
    dagmc_model.settings = settings

    logger.info("Running CSG model")
    csg_model.export_to_model_xml(path=str(tmp_path))
    output_file_from_csg = csg_model.run(cwd=str(tmp_path / "csg"))
    logger.info(output_file_from_csg)

    logger.info("Running DAGMC model")
    dagmc_model.export_to_model_xml(path=str(tmp_path))
    output_file_from_cad = dagmc_model.run(cwd=str(tmp_path / "cad"))
    logger.info(output_file_from_cad)

    with (
        openmc.StatePoint(output_file_from_csg) as sp_from_csg,
        openmc.StatePoint(output_file_from_cad) as sp_from_cad,
    ):
        for tally in tallies:
            csg_tally = sp_from_csg.get_tally(name=tally.name)
            cad_tally = sp_from_cad.get_tally(name=tally.name)
            logger.info(f"Comparing tally: {tally.name}")
            logger.info(f"CSG result: {csg_tally.mean}")
            logger.info(f"CAD result: {cad_tally.mean}")

            logger.debug(msh._mesh_filename)
            assert np.allclose(
                csg_tally.mean, cad_tally.mean, rtol=RELATIVE_TOLERANCE_PERCENT
            )
            # We've loaded the same data twice if they're too close
            assert not np.allclose(csg_tally.mean, cad_tally.mean, rtol=0.000001)
