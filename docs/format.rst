======
Format
======

.. module:: stellarmesh
   :no-index:

Stellarmesh consumes Gmsh ``.msh`` v4.1 ASCII files annotated with
physical groups using a URL-style ``key=value`` encoding.

-----------------------
Physical-group encoding
-----------------------

Per-entity metadata is stored as the **name** of a Gmsh physical
group covering exactly one Gmsh entity. The name encodes a URL-style
query string of key/value pairs, parsed by stellarmesh as follows.

Volume groups
=============

::

   tag=<integer>&material=<string>

* ``tag`` — the volume's Gmsh tag (matches the discrete entity tag in
  the same file).
* ``material`` — a slug identifying the volume's material region.
  Slugs are bounded to **28 characters** (a hard MOAB limit downstream
  in the DAGMC pipeline).

Every volume entity in the file must carry exactly one such physical
group.

Surface groups
==============

::

   tag=<integer>&forward_volume=<integer>&reverse_volume=<integer>

* ``tag`` — the surface's Gmsh tag.
* ``forward_volume`` — the volume tag on the surface's outward-normal
  side.
* ``reverse_volume`` — the volume tag on the other side. Use ``None``
  (or omit the key) for exterior surfaces. 

Edges and vertices
==================

Discrete entities only; **no physical groups**. Stellarmesh ignores
edge and vertex annotations if present.

-------
Merging
-------

``Mesh.merge({label: mesh, ...})`` combines meshes without imprinting.
The first input keeps its tags. Each later input's entity tags are
offset by the next power of ten above the largest entity tag merged so
far (e.g. a first mesh with tags up to 87 puts the second at 100 +
tag). This helps the user to quickly identify which source mesh an entity
came from. Node and element tags are offset by the running maxima.

------------------
Consumer behaviour
------------------

When stellarmesh reads a conforming ``.msh`` file:

* Each volume's ``material`` slug becomes a DAGMC ``mat:<slug>`` group
  in the output ``.h5m`` file.
* Surface ``forward_volume`` / ``reverse_volume`` populate DAGMC's
  surface-sense relationships.
* Downstream tooling (e.g. OpenMC) maps the ``mat:<slug>`` groups to
  ``openmc.Material`` instances. Stellarmesh does not perform that
  mapping itself.
