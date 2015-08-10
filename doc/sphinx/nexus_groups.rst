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

will fail if ``entry`` and/or ``instrument`` do not exist. As NeXus data trees
can become rather complicated 


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
:py:attr:`attributes`  property providing access to the groups attributes
:py:attr:`path`        provides the path for the group
=====================  ====================================================

Iteration
---------


As containers, instances of :py:class:`nxgroup` can be iterated. 
Two iteration schemes are supported

* direct iteration
* recursive iteration

In the former case iteration is only done over the direct children of a
group (those children for which the particular group is the parent). 
The latter scheme provides access to all children stored below a group and its 
subgroups.
To iterate only over the direct children of a group the common Python 
syntax can be used

.. code-block:: python

    for child in group: print(child.path)

In order to iterate recursively use the :py:meth:`recursive` method to obtain 
a recursive iterator to the instance

.. code-block:: python

    for child in group.recursive: print(child.path)

