"""Making the stellarmesh logo."""
import logging

import build123d as bd
import stellarmesh as sm

logging.getLogger("stellarmesh").setLevel(logging.INFO)

cmp = bd.Compound.make_text("Stellarmesh", 14, font="Arial Black")
solids = [f.thicken(10) for f in cmp.faces()]

geometry = sm.Geometry(solids, [""] * len(solids))
mesh = sm.Mesh.mesh_geometry(geometry, min_mesh_size=1, max_mesh_size=2)
mesh.render("doc/logo.png", rotation_xyz=(0, -2, 0), clipping=False)
