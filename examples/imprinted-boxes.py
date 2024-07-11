"""Making the stellarmesh logo."""

import logging

import build123d as bd
import stellarmesh as sm

logging.getLogger("stellarmesh").setLevel(logging.INFO)

b1 = bd.Solid.make_box(10, 10, 10)
b2 = b1.transformed(offset=(0, 10, 0))
b3 = b1.transformed(offset=(0, 5, 10))

cmp_initial = bd.Compound.make_compound([b1, b2, b3])
solids = cmp_initial.solids()
geom = sm.Geometry(solids, material_names=[""] * len(solids))
geom_imprinted = geom.imprint()
