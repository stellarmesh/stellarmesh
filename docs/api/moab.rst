====
MOAB
====
.. module:: stellarmesh

---------------------
MOAB and DAGMC Models
---------------------

Both the :py:class:`MOABModel` and the inherited :py:class:`DAGMCModel` classes can be instantiated from a Stellarmesh :py:class:`Mesh` object using the :py:meth:`from_mesh() <DAGMCModel.from_mesh>` constructor.

Use the :py:meth:`write() <MOABModel.write>` method to write a ``.h5m`` file for import in OpenMC.

See the `Surface Meshing <../notebooks/tutorials/surface_meshing.html#Run-an-OpenMC-Tally>`__ tutorial for a complete example.


API
---------------------

.. autosummary::
    :toctree: generated
    :template: class.rst

    MOABModel
    DAGMCModel