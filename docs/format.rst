======
Format
======

.. module:: stellarmesh
   :no-index:

Stellarmesh consumes Gmsh ``.msh`` v4.1 ASCII files annotated with
physical groups using a URL-style ``key=value`` encoding. Any tool
that emits this format is supported as a producer; ``basalt`` is the
reference implementation. This page is the canonical schema spec.

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
  in the DAGMC pipeline). Slug semantics are producer-defined; see
  `basalt's producer-side reference <https://basalt.readthedocs.io/en/latest/format.html>`__
  for one convention.

Every volume entity in the file must carry exactly one such physical
group.

Surface groups
==============

::

   tag=<integer>&forward_volume=<integer>&reverse_volume=<integer>

* ``tag`` — the surface's Gmsh tag.
* ``forward_volume`` — the volume tag on the surface's outward-normal
  side.
* ``reverse_volume`` — the volume tag on the other side. Use ``0`` for
  exterior surfaces (boundary of the model with vacuum).

Edges and vertices
==================

Discrete entities only; **no physical groups**. Stellarmesh ignores
edge and vertex annotations if present.

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

--------------------
Stability commitment
--------------------

Stellarmesh treats this encoding as a **public stability boundary**.
Changes to the key set (``tag``, ``material``, ``forward_volume``,
``reverse_volume``), the scheme, or the consumer semantics described
above constitute breaking changes and require a major-version bump.

----------------
Coordinate units
----------------

The encoding does not pin coordinate units. Stellarmesh passes through
whatever the producer wrote. Consumers that have unit conventions
(OpenMC expects centimetres) typically rely on the producer to scale
the output appropriately. See ``basalt``'s ``Mesh.write_msh``
``scale_factor`` parameter for one convention.

---------
Producers
---------

* `basalt <https://basalt.readthedocs.io/>`__ — Parasolid → mesh
  toolchain wrapping Simmetrix SimModSuite; the reference producer.

Any tool implementing the schema above interoperates with stellarmesh.
Inclusion in this list is informational only — there is no
registration mechanism.
