"""Continuous integration tests with OpenMC."""

import logging
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
    def dagmc(self) -> sm.DAGMCModel:
        """Generate the DAGMC model."""
        ...

    def _make_dagmc_model(
        self,
        solids: Sequence[bd.Solid],
        materials: openmc.Materials,
        min_mesh_size: float,
        max_mesh_size,
    ) -> sm.DAGMCModel:
        mat_names: list[str] = [m.name for m in self.materials]
        geom = sm.Geometry(solids, mat_names)
        msh = sm.Mesh.from_geometry(geom, min_mesh_size, max_mesh_size)
        return sm.DAGMCModel.from_mesh(msh)


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

    def dagmc(self):
        s1 = bd.Solid.make_sphere(self.radius1)
        s2 = s1.faces()[0].thicken(self.radius2)
        return self._make_dagmc_model([s1, s2], self.materials, 1, 1)


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
        surface = openmc.ZTorus(
            a=self.major_radius,
            b=self.minor_radius,
            c=self.minor_radius,
            boundary_type="vacuum",
        )
        cell = openmc.Cell(fill=self.materials[0], region=-surface)
        csg_geometry = openmc.Geometry([cell])
        return openmc.Model(geometry=csg_geometry)

    def dagmc(self):
        s1 = bd.Solid.make_torus(self.major_radius, self.minor_radius)
        return self._make_dagmc_model([s1], self.materials, 0.01, 0.5)


@pytest.mark.parametrize("model_cls", [NestedSpheres, Torus])
def test_model(model_cls: Type[Model], tmp_path: Path):
    openmc.reset_auto_ids()
    model = model_cls()

    settings = openmc.Settings()
    settings.batches = 20
    settings.inactive = 0
    settings.particles = 500
    settings.run_mode = "fixed source"
    settings.source = model.source()

    tallies = model.tallies()
    csg_model = model.csg()
    csg_model.materials = model.materials
    csg_model.tallies = tallies
    csg_model.settings = settings

    model.dagmc().write(str(tmp_path / "dagmc.h5m"))
    universe = openmc.DAGMCUniverse(tmp_path / "dagmc.h5m").bounded_universe()
    dagmc_geometry = openmc.Geometry(universe)
    dagmc_model = openmc.Model(geometry=dagmc_geometry)
    dagmc_model.materials = model.materials
    dagmc_model.tallies = tallies
    dagmc_model.settings = settings

    csg_model.export_to_model_xml(path=str(tmp_path))
    output_file_from_csg = csg_model.run(cwd=str(tmp_path / "csg"))
    logger.info(output_file_from_csg)

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
            assert np.allclose(
                csg_tally.mean, cad_tally.mean, rtol=RELATIVE_TOLERANCE_PERCENT
            )
            # We've loaded the same data twice if they're too close
            assert not np.allclose(csg_tally.mean, cad_tally.mean, rtol=0.000001)
