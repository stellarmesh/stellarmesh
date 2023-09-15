"""Stellarmesh examples."""
# %% Module imports and initialization
# These examples are best run with the OCP CAD Viewer VSCode extension
# to view CAD geometry.
import logging
import sys

import build123d as bd

import stellarmesh as sm

try:
    from ocp_vscode import show
except:  # noqa
    print("ocp_vscode not installed, not showing geometry.")


def show_or_skip(*args, **kwargs):
    if "ocp_vscode" in sys.modules:
        try:
            show(*args, **kwargs)
        except:
            print("OCP viewer not available, skipping.")


# Required to show logging in Jupyter
logging.basicConfig()
logging.getLogger("stellarmesh").setLevel(logging.INFO)
# %% Simple torus geometry
solids = [bd.Solid.make_torus(1000, 100)]
for i in range(3):
    solids.append(solids[-1].faces()[0].thicken(100))
solids = solids[1:]
show_or_skip(solids, transparent=True)

geometry = sm.Geometry(solids, material_names=["a", "a", "c"])
mesh = sm.Mesh.mesh_geometry(geometry, min_mesh_size=50, max_mesh_size=50)
mesh.write("test.msh")
mesh.render("doc/torus-mesh.png", rotation_xyz=(90, 0, -90), normals=15)

h5m = sm.MOABModel.make_from_mesh(mesh)
h5m.write("dagmc.h5m")
h5m.write("dagmc.vtk")

# %% Stellarmesh logo
cmp = bd.Compound.make_text("Stellarmesh", 14, font="Arial Black")
solids = [f.thicken(10) for f in cmp.faces()]
show_or_skip(solids)

geometry = sm.Geometry(solids, [""] * len(solids))
mesh = sm.Mesh.mesh_geometry(geometry, min_mesh_size=1, max_mesh_size=2)
mesh.render("doc/logo.png", rotation_xyz=(0, -2, 0), clipping=False)
