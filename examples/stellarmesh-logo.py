"""Making the stellarmesh logo."""

import logging

import build123d as bd
import stellarmesh as sm

logging.getLogger("stellarmesh").setLevel(logging.INFO)

cmp = bd.Compound.make_text("S", 14, font="Arial Black")
solids = [f.thicken(0.01) for f in cmp.faces()]

geometry = sm.Geometry(solids, [""] * len(solids))
mesh = sm.SurfaceMesh.from_geometry(geometry, sm.GmshSurfaceOptions(1, 2))
mesh.render("doc/logo.png", rotation_xyz=(0, 0, 0), clipping=False)
