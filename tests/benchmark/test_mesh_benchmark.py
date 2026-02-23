import logging

import gmsh
import pytest

import stellarmesh as sm

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "mesh_options",
    [
        sm.OCCSurfaceOptions(tol_angular_deg=0.1),
        sm.GmshSurfaceOptions(curvature_target=45, num_threads=8),
    ],
    ids=["OCC", "Gmsh"],
)
def test_benchmark_mesh(geom_bd_layered_torus, benchmark, mesh_options):
    mesh = benchmark.pedantic(
        sm.SurfaceMesh.from_geometry,
        args=(geom_bd_layered_torus, mesh_options),
        rounds=3,
    )
    with mesh:
        num_nodes = len(gmsh.model.mesh.get_nodes()[0])
        logger.info(f"Model has {num_nodes} nodes")
