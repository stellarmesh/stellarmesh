<p align="center">
<img src="doc/logo.png" width="80%">
</p>

:warning: This library is in development. Expect breaking changes and bugs, and feel free to contribute.

Stellarmesh is a Gmsh wrapper and DAGMC geometry creator for fusion neutronics workflows, building on other libraries such as [cad-to-dagmc](https://github.com/fusion-energy/cad_to_dagmc) and [cad-to-openmc](https://github.com/openmsr/CAD_to_OpenMC). The goal is to reach feature parity with the [Cubit plugin](https://github.com/svalinn/Cubit-plugin) to enable a fully-featured and open-source workflow.

**Progress**:

- [x] Correct implementation of surface-sense
- [x] Imprinting and merging of conformal geometry
- [ ] Programatic manipulation of .h5m tags e.g. materials
- [ ] Mesh refinement

# Contents
- [Contents](#contents)
- [Installation](#installation)
- [Examples](#examples)
  - [Simple torus geometry](#simple-torus-geometry)
- [Comparison to other libraries](#comparison-to-other-libraries)

# Installation

Stellarmesh is not yet available on PyPi, for now install with:
```sh
pip install https://github.com/Thea-Energy/stellarmesh.git
```

# Examples

For more examples see `examples.py`

## Simple torus geometry

<details>
<summary>Module imports and configuration</summary>

```python
%load_ext autoreload
%autoreload 2
import build123d as bd
import stellarmesh as sm
import logging

# Required to show logging in Jupyter
logging.basicConfig()
logging.getLogger("stellarmesh").setLevel(logging.INFO)
```

</details>

```python
solids = [bd.Solid.make_torus(1000, 100)]
for i in range(3):
    solids.append(solids[-1].faces()[0].thicken(100))
solids = solids[1:]

geometry = sm.Geometry(solids, material_names=["a", "a", "c"])
mesh = sm.Mesh.mesh_geometry(geometry, min_mesh_size=50, max_mesh_size=50)
mesh.write("test.msh")
mesh.render("doc/torus-mesh.png", rotation_xyz=(90, 0, -90), normals=15)

h5m = sm.MOABModel.make_from_mesh(mesh)
h5m.write("dagmc.h5m")
h5m.write("dagmc.vtk")
```

<p align="center">
<img width="80%" src="doc/torus-mesh.png"> <br> <em>Rendered mesh with normals.</em>
</p>

<details>
<summary>Check overlaps</summary>

```{bash}
❯ overlap_check dagmc.h5m

NOTICE:
     Performing overlap check using triangle vertex locations only.
     Use the '-p' option to check more points on the triangle edges.
     Run '$ overlap_check --help' for more information.

Running overlap check:
100% |===============================================================>|+
No overlaps were found.
```

</details>

<details>
<summary>Check materials</summary>

```{bash}
❯ mbsize -ll dagmc.h5m | grep mat:

NAME = mat:a
NAME = mat:c
```

</details>

<details>
<summary>Check watertight</summary>

```{bash}
❯ check_watertight dagmc.h5m

number of surfaces=4
number of volumes=3

0/0 (nan%) unmatched edges
0/4 (0%) unsealed surfaces
0/3 (0%) unsealed volumes
leaky surface ids=
leaky volume ids=
0.173068 seconds
```

</details>


# Comparison to other libraries

|| Stellarmesh | [CAD-to-DAGMC](https://github.com/fusion-energy/cad_to_dagmc) |  [CAD-to-OpenMC](https://github.com/openmsr/CAD_to_OpenMC) | Cubit |
|---| --- | --- | --- | --- |
| Developer | Thea Energy | Jonathan Shimwell | Erik B. Knudsen | Coreform
| Meshing backend | Gmsh | Gmsh | Gmsh/CQ |  Cubit |
| In development | ✓ | ✓ | ✓ | ✓ |
| Open-source | ✓ | ✓ | ✓ |   |
| Surface-sense handling | ✓ |   | <sup>1</sup> | ✓ |
| Mesh refinement |  |   | ✓ | ✓ |
| Manipulation of .h5m files | <sup>2</sup> |   | | |

<em>Note: Please file an issue if this table is out-of-date.</em>

<sup>1</sup> In development on a personal branch

<sup>2</sup> In development
