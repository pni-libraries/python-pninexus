Working with groups
===================

Groups are the fundamental containers in Nexus. They can hold other groups as
well as fields. 

Creation
--------

There are several ways how to create groups and fields. The most natural one is
to use the :py:meth:`create_group` method of a group instance.  The former one is
rather easy to use

.. code-block:: python

    g = root_group.create_group("entry_1")
    e = root_group.create_group("entry_2","NXentry")

The first call creates a group of name ``entry_1`` while the second 
one create a group of name ``entry_2`` of type ``NXentry``. 
The type of a group is determined by an attribute name ``NX_class``
attached to the group (we will learn more about attributes latter).

The :py:meth:`create_group` method can only create groups which are direct
children of the their parent group. So using 

.. code-block:: python

    g = root_group.create_group("entry/instrument/detector")

will fail if `entry` and/or `instrument` do not exist. It may sounds strange
that one cannot create intermediate groups automatically a feature the HDF5
library and the :py:mod:`h5py` wrapper support. However, we cannot use this
feature as it creates groups only by name and does not add type support which 
we would have to use in case of



where the first argument is the parent group, the second the group to create as
a Nexus path. If the keyword argument ``intermediates`` is set to ``True`` all
the intermediate groups will be created if they do not exist.  By default the
``intermediates`` argument is set to ``False`` so that the ``create_group``
function behaves like its class member counterpart.

Inquery
-------

In order to determine the number of (direct) children of a group we can 
either use the :py:attr:`size` attribute of an instance of 
:py:class:`nxgroup` or use the build-in function :py:func:`len` with an 
instance of :py:class:`nxgroup` as its argument. 

.. code-block:: python

    g = ....
    g.size == len(g)


:py:class:`nxgroup` exposes some more read-only attributes to obtain 
more information about a particular instance

=====================  ====================================================
Attribute name         Description 
=====================  ====================================================
:py:attr:`name`        returns the name of the group 
:py:attr:`parent`      returns the groups parent group 
:py:attr:`size`        the number of children a group has 
:py:attr:`filename`    name of the file the group belongs to 
:py:attr:`attributes`  property providing access to the groups' attributes
:py:attr:`path`        provides the path for the group
=====================  ====================================================

Iteration
---------

As containers, instances of :py:class:`nxgroup` expose two different iterator
interfaces

* a simple one directly provided by :py:class:`nxgroup` which iterates only 
  over the direct children of a group
* and a recursive iterator provided by the :py:attr:`nxgroup.recursive` of 
  an instance of :py:class:`nxgroup`.

The latter one iterates over all children of a group and the children of its
subgroups. Simple iteration can be done with

.. code-block:: python

    for child in group: print(child.path)

while the recursive iterator can be accessed via the 
:py:attr:`recursive` attribute of an instance of :py:class:`nxgroup`

.. code-block:: python

    for child in group.recursive: print(child.path)

Recursive iteration is a quite usefull feature along with list comprehension to
generate lists of particular object types.  
A typical application for recursive iteration would be to find all the fields
that have been involved in a measurement. We assume here that for all fields
the first dimension indicates the number of scan points. We thus can simply use
the following list comprehension 

.. code-block:: python 

    from __future__ import print_function
    import pni.io.nx.h5 as nx 

    f = nx.open_file("test.nxs")

    def scanned_field(obj):
        return is_instance(obj,nx.nxfield) and obj.shape[0]>1

    scanned_fields = [ obj for obj in f.root().recursive if scanned_field(obj)]

    for field in scanned_fields:
        print field.path


