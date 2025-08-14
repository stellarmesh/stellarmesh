<p align="center">
<img src="https://github.com/Thea-Energy/stellarmesh/raw/main/doc/logo.png" width="80%">
</p>

[![Tests](https://github.com/stellarmesh/stellarmesh/actions/workflows/test.yml/badge.svg)](https://github.com/stellarmesh/stellarmesh/actions/workflows/test.yml)

:warning: See [logging](#logging) to enable logging output when using Jupyter.

Stellarmesh is a meshing library for nuclear workflows. Principally, it supports the creation of DAGMC geometry from CAD models.


**Features**:

- [x] Import of [CadQuery](https://github.com/CadQuery/cadquery), [build123d](https://github.com/gumyr/build123d), STEP and BREP geometry
- [x] Surface and volume meshing
- [x] Both Gmsh and OpenCASCADE meshing backends
- [x] Linear and angular mesh tolerances
- [x] Imprinting and merging of conformal geometry
- [x] Mesh refinement
- [x] Automated testing and integration
- [x] Programatic manipulation of .h5m tags

# Contents
- [Contents](#contents)
- [Installation](#installation)
- [Usage](#usage)
  - [Geometry construction](#geometry-construction)
  - [Examples](#examples)
    - [Simple torus geometry](#simple-torus-geometry)
    - [Other](#other)
  - [Logging](#logging)
  - [Mesh refinement](#mesh-refinement)

# Installation
```sh
pip install stellarmesh
```

or install the development version with:

```sh
pip install git+https://github.com/stellarmesh/stellarmesh
```

Stellarmesh requires several dependencies that must be installed either from source or from conda-forge. At a minimum, these are `moab`, `python-gmsh`, and `OCP`. However, most users will likely also want `build123d` for geometry construction and `openmc` for simulation. See [environment.yml](environment.yml) for a sample conda environment with these additional dependencies.

# Usage
## Geometry construction
Stellarmesh supports both [build123d](https://github.com/gumyr/build123d) (recommended) and [CadQuery](https://github.com/CadQuery/cadquery) for geometry construction but does not depend on either.

The included examples use build123d. To install, run:

```python
pip install build123d
```

For documentation and usage examples, see [Read the Docs](https://build123d.readthedocs.io/en/latest/).

## Meshing backend

Stellarmesh supports surface meshing with both OCC and Gmsh backends.

### OOC



## Examples

### Simple torus geometry
```python
import build123d as bd
import stellarmesh as sm

solids = [bd.Solid.make_torus(1000, 100)]
for _ in range(3):
    solids.append(solids[-1].faces()[0].thicken(100))
solids = solids[1:]

geometry = sm.Geometry(solids, material_names=["a", "a", "c"])
mesh = sm.Mesh.from_geometry(geometry, min_mesh_size=50, max_mesh_size=50)
mesh.write("test.msh")
mesh.render("doc/torus-mesh-reversed.png", rotation_xyz=(90, 0, -90), normals=15)

h5m = sm.DAGMCModel.from_mesh(mesh)
h5m.write("dagmc.h5m")
h5m.write("dagmc.vtk")
```

<p align="center">
<img width="80%" src="https://github.com/Thea-Energy/stellarmesh/raw/main/doc/torus-mesh.png"> <br> <em>Rendered mesh with normals.</em>
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

### Other
- [Mesh refinement](examples/mesh-refinement.py)
- [Stellarmesh logo](examples/stellarmesh-logo.py)
- [Imprinted boxes](examples/imprinted-boxes.py)

## Logging
Stellarmesh uses the logging library for debug, info and warning messages. Set the level with:

```python
import logging

logging.basicConfig() # Required in Jupyter to correctly set output stream
logging.getLogger("stellarmesh").setLevel(logging.INFO)
```

## Mesh refinement
> [!NOTE]
> Given CAD geometry, Gmsh often produces high-quality meshes that do not benefit from remeshing.

Stellarmesh supports mesh refinement using the [mmg](https://www.mmgtools.org/) library. Refine a mesh with:

```python
refined_mesh = mesh.refine(
  ...
)
```
and consult the `Mesh.refine` and [mmgs](https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-tutorials/mmg-remesher-mmg2d/mesh-adaptation-to-a-solution) documentations for parameter values.

<p align="center">
    <img width="40%" src="https://github.com/Thea-Energy/stellarmesh/assets/43913902/f3440b6b-3e11-476a-9fae-ab9708f8f2b2"/>
  <br>
  <img width="40%" src="https://github.com/Thea-Energy/stellarmesh/assets/43913902/29acbdb3-24a2-419d-9f3f-237aec475369" />
  <br>
  <em>The refined mesh has more triangles in regions with high curvature thanks to the <a href="https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-options/mmg-remesher-option-hausd">hausdorff parameter</a>.</em>
</p>

Many thanks to [Erik B. Knudsen](https://github.com/ebknudsen) for his work on remeshing for [CAD-to-OpenMC](https://github.com/openmsr/CAD_to_OpenMC).

# Acknowledgements
Stellarmesh is originally a project of Thea Energy, who are building the world's first planar coil stellarator.

[<img width="200" src="https://github.com/user-attachments/assets/37b9ba1c-b22c-4837-b226-a6212854127e">](https://thea.energy/)
