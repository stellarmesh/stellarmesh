.. module:: stellarmesh
=========
Tutorials
=========

Most Stellarmesh workflows will resemble the following. Here:

#. A :py:class:`Geometry` object is created from a list of build123d and CadQuery solids with associated material names.
#. A :py:class:`SurfaceMesh` object is created from the geometry with either the `OCC or Gmsh <api/mesh.rst#Backends>`__ backends.
#. A :py:class:`DAGMCModel` object is created from the surface mesh and written to a ``.h5m`` file for analysis in OpenMC.

.. code:: python

   import stellarmesh as sm

   solids: list[Solid] # Build123d or CadQuery solids

   geometry = sm.Geometry(solids, material_names = ["l1", "l2", "l3"])
   mesh = sm.SurfaceMesh.from_geometry(
      geometry, sm.OCCSurfaceOptions(tol_angular_deg = 0.5)
   )
   mesh.write("test.msh")

   h5m = sm.DAGMCModel.from_mesh(mesh)
   h5m.write("dagmc.vtk")
   h5m.write("dagmc.h5m")


.. note::
   Stellarmesh uses the logging library for debug, info and warning messages. Set the level with:


   .. code:: python

      import logging

      logging.basicConfig() # Required in Jupyter to correctly set output stream
      logging.getLogger("stellarmesh").setLevel(logging.INFO)


---------
Tutorials
---------

The included tutorials use `build123d <https://github.com/gumyr/build123d>`__ for geometry construction. See :doc:`install`.

.. toctree::
   :maxdepth: 1

   notebooks/tutorials/surface_meshing.ipynb
   notebooks/tutorials/volume_meshing.ipynb