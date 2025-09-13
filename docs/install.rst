============
Installation
============

|PyPI Version| |Conda Version|

Stellarmesh is available on both `conda-forge <https://pypi.org/project/stellarmesh/>`__ and `PyPI <https://pypi.org/project/stellarmesh/>`__.



.. tabs ::

    .. tab:: Conda

        .. code:: sh

            conda install stellarmesh
    
    .. tab:: PyPI

        .. code:: sh

            pip install stellarmesh

        Note that the PyPI distribution does not include some essential dependencies, and as such the following must be installed from source or from ``conda-forge``:

        * ``moab``
        * ``python-gmsh``
        * ``OCP``

        .. warning::  While OCP and Gmsh can both be installed from PyPI, they are not ABI compatible. Usage of these packages will result in errors for some geometries.

..

Stellarmesh supports both `build123d <https://github.com/gumyr/build123d>`__ (recommended) and `CadQuery <https://github.com/CadQuery/cadquery>`__ for geometry construction but does not depend on either. The included examples use build123d.


.. tabs ::

    .. tab:: build123d

        .. code:: sh

            conda install build123d


    .. tab:: CadQuery

        .. code:: sh

            conda install cadquery

.. |PyPI Version| image:: https://img.shields.io/pypi/v/stellarmesh.svg
   :target: https://pypi.org/project/stellarmesh/
.. |Conda Version| image:: https://img.shields.io/conda/vn/conda-forge/stellarmesh.svg
   :target: https://anaconda.org/conda-forge/stellarmesh