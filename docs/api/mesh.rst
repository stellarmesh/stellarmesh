----
Mesh
----
.. module:: stellarmesh


Stellarmesh supports surface meshing with both OCC and Gmsh backends.

OCC
~~~

The OpenCASCADE (OCC) meshing backend is the preferred backend when
linear or angular mesh tolerances are required.

Gmsh
~~~~

The Gmsh meshing backend offers a number of meshing algorithms for both
surface and volume meshes. For more detailed documentation, see
`gmsh.info <https://gmsh.info/doc/texinfo/gmsh.html>`__.


Mesh refinement
---------------

.. note:: Given CAD geometry, Gmsh often produces high-quality meshes
   that do not benefit from remeshing.

Stellarmesh supports mesh refinement using the
`mmg <https://www.mmgtools.org/>`__ library. Refine a mesh with:

.. code:: python

   refined_mesh = mesh.refine(
     ...
   )

and consult the ``Mesh.refine`` and
`mmgs <https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-tutorials/mmg-remesher-mmg2d/mesh-adaptation-to-a-solution>`__
documentations for parameter values.




.. figure:: https://github.com/Thea-Energy/stellarmesh/assets/43913902/f3440b6b-3e11-476a-9fae-ab9708f8f2b2
   :width: 40%
   :align: center


.. figure:: https://github.com/Thea-Energy/stellarmesh/assets/43913902/29acbdb3-24a2-419d-9f3f-237aec475369
   :width: 40%
   :align: center

   The refined mesh has more triangles in regions with high curvature thanks to the hausdorff parameter.

Many thanks to `Erik B. Knudsen <https://github.com/ebknudsen>`__ for
his work on remeshing for
`CAD-to-OpenMC <https://github.com/openmsr/CAD_to_OpenMC>`__.

API
---------------------

.. autosummary::
    :toctree: generated
    :template: class.rst

    SurfaceMesh
    VolumeMesh
    OCCSurfaceOptions
    OCCSurfaceAlgo
    GmshSurfaceOptions
    GmshSurfaceAlgo
    GmshVolumeOptions
    GmshVolumeAlgo
