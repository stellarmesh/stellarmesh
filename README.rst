.. figure:: https://github.com/Thea-Energy/stellarmesh/raw/main/docs/logo.png
   :width: 100%
   :align: center


|Tests| |PyPI Version| |Conda Version|

Stellarmesh is a meshing library for nuclear workflows. Principally, it
supports the creation of DAGMC geometry from CAD models.

**Features**:

* ✅ Import of `CadQuery <https://github.com/CadQuery/cadquery>`__,
  `build123d <https://github.com/gumyr/build123d>`__, STEP and BREP
  geometry
* ✅ Surface and volume meshing
* ✅ Gmsh and OpenCASCADE meshing backends
* ✅ Linear and angular mesh tolerances
* ✅ Surface boundary conditions
* ✅ Imprinting and merging of conformal geometry
* ✅ Mesh refinement
* ✅ Programmatic manipulation of .h5m tags
* ✅ Automated testing and integration

---------------
Getting Started
---------------

* `Installation <https://stellarmesh.readthedocs.io/en/latest/install.html>`__
* `Tutorials <https://stellarmesh.readthedocs.io/en/latest/tutorials.html>`__
* `API <https://stellarmesh.readthedocs.io/en/latest/api>`__

-------
Example
-------

.. code:: python

   import build123d as bd
   import stellarmesh as sm

   solids = [bd.Solid.make_torus(1000, 100)]
   for _ in range(3):
       solids.append(bd.Solid.thicken(solids[-1].faces()[0], 100))
   solids = solids[1:]

   geometry = sm.Geometry(solids[::-1], material_names=["a", "a", "c"])
   mesh = sm.SurfaceMesh.from_geometry(
       geometry, sm.GmshSurfaceOptions(min_mesh_size=50, max_mesh_size=200)
   )
   mesh.write("test.msh")
   mesh.render("docs/torus-mesh-reversed.png", rotation_xyz=(90, 0, -90), normals=15)

   h5m = sm.DAGMCModel.from_mesh(mesh)
   h5m.write("dagmc.h5m")
   h5m.write("dagmc.vtk")


.. figure:: https://github.com/Thea-Energy/stellarmesh/raw/main/docs/torus-mesh.png
   :width: 80%
   :align: center
   :alt: Rendered mesh with normals.

   Rendered mesh with normals.


.. note::
   Stellarmesh uses the logging library for debug, info and warning messages. Set the level with:


   .. code:: python

      import logging

      logging.basicConfig() # Required in Jupyter to correctly set output stream
      logging.getLogger("stellarmesh").setLevel(logging.INFO)



----------------
Acknowledgements
----------------

Stellarmesh is originally a project of Thea Energy, who are building the
world’s first planar coil stellarator.

.. raw:: html

   <img src="https://github.com/user-attachments/assets/37b9ba1c-b22c-4837-b226-a6212854127e" width="200px">

.. raw:: html

.. |Tests| image:: https://github.com/stellarmesh/stellarmesh/actions/workflows/test.yml/badge.svg
   :target: https://github.com/stellarmesh/stellarmesh/actions/workflows/test.yml
.. |PyPI Version| image:: https://img.shields.io/pypi/v/stellarmesh.svg
   :target: https://pypi.org/project/stellarmesh/
.. |Conda Version| image:: https://img.shields.io/conda/vn/conda-forge/stellarmesh.svg
   :target: https://anaconda.org/conda-forge/stellarmesh
