"""Test of a simple layered torus. Outputs refined mesh to refined.msh."""
import logging

import build123d as bd
import numpy as np
import stellarmesh as sm

logging.getLogger("stellarmesh").setLevel(logging.INFO)

with bd.BuildPart() as my_part:
    with bd.BuildLine() as my_line:
        bd.Bezier(
            *[
                (i, 0, 2 * x)
                for i, x in enumerate(np.sin(np.arange(2 * np.pi, step=0.5)))
            ]
        )
    with bd.BuildSketch() as my_sketch:
        bd.Rectangle(1, 5)
    bd.sweep()

geom = sm.Geometry(my_part.solids(), material_names=["hi"])
mesh = sm.Mesh.mesh_geometry(geom, min_mesh_size=0.5, max_mesh_size=0.5)
refined_mesh = mesh.refine(const_mesh_size=1, hausdorff_value=100)
refined_mesh.write("refined.msh")
