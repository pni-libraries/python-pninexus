==================================
Addressing objects: the NeXus path
==================================

.. automodule:: pninexus.nexus

An introduction
===============

One of the major contributions added by *libpninexus* to the standard HDF5 library
is its *Path* class which is provided by :py:mod:`pninexus.nexus` via the 
:py:class:`Path`. The most obvious difference between a plain HDF5 path and a
NeXus path as introduced by *libpninexus* is 

1. a NeXus path can include the name of the file the object referenced by the
   path 
2. a NeXus path can also address an attribute.

However, the major difference between a NeXus path and an HDF5 path is the 
fact that a NeXus path can address object not only by their links name 
but also by type (base class). This feature allows, under certain circumstances, 
the construction of path's which can address objects without knowing their 
names and thus remain generic.   

The best way to understand thus path's is by example. So lets start with 
a simple one:

.. code-block:: bash

    detector_1.nxs://scan_1/instrument/detector/transformation/phi 

This looks pretty much like a common HDF5 path but with the name of the 
file where to find the dataset prepended (separated from the rest of the path 
by `:/`). If we would like to address an attribute by a path we could do so
with NeXus paths by appending the name of the attribute to the path 
of the object the attribute is attached to 

.. code-block:: bash

    detector_1.nxs://scan_1/instrument/detector/transformation/phi@units 

wher the attributes name is separated from its parent node path by a `@` 
character. With the concept of base classes in place we can further determine 
the path of an object by adding the base classes the intermediate objects belong
to: 

.. code-block:: bash

    detector_1.nxs://scan_1:NXentry/instrument:NXinstrument/ \
                    detector:NXdetector/transformation:NXtransformation/phi@units 

This would mean that the first element of name `scan_1` must be a group 
belonging to the base class `NXentry`. The second one of name `instrument` 
and of type `NXinstrument`. As the last element in the path is a dataset 
(or *field* in NeXus terminology) there is no means of further specifying it. 

If we now assume that, in the above example, each base class appears only 
once below its parent group we could even omit the names of the links 
which leaves us for this example with 

.. code-block:: bash

    detector_1.nxs://:NXentry/:NXinstrument/:NXdetector/:NXtransformation/phi@units 

Being independent of any link name, such a path can be used on every file 
which satisfies the above conditions. 

Which would still work if every type appears only once below its parent.
However, as we have removed the group names the path becomes generic in a sense
that the paths no longer depends on the particular names given to the
individual groups. It thus gives more freedom to the beamline scientist 
naming groups according to his needs and even change the naming over the coarse
of time. As long as the overall structure is not altered this approach will
still work. It is important to note that the leading ``:`` before the group
types is requied to indicate the parser that the following string refers to a
type rather than to a name.

The anatomy of a Nexus path
---------------------------

A NeXus path consists of three sections 

* the *file-section* determining the file within which a particular object is
  stored. The *file-section* preceds the entire path and is terminated by
  ``://``. 
* the *object-section* pointing to the object within the file
* and the *attribute-section* which refers to an attribute stored a the object
  referenced by the previous *object-section*. The *attribute section* consists
  of the name of the referenced attaribute and is preceded by the ``@`` symbol. 
  The *attribute-section* follows immediately after the last element of the
  *object-section*.

The *file-* as well as the *attribute-section* are optional while the
*object-section* is the onmy mandatory path of a path. 


The major part of the path is the *object-section*. The elements of the object
section are separated by a ``/`` like file- or directory-names on a unix
system. For groups an element can have the form *name:type* while for a field
it is only the *name*. As we have seen in the examples above the name can be
omitted group in which case the element reduces to *:type*.


Path equality and match
-----------------------

Two paths are equal if all their elements (including file and attribute
section) are equal. Equality is thus a rather strict criteria when comparing
two paths. A more relaxed version is a *match*. Two paths are matching if they
would be in principle able to reference the same object. The problem can be
reduced to define when the *elements* of the *object-section* are matching. 
Three rules can be defined which determine whether two elements match

1. Two elements, *a* and *b* are matching, if each element they are equal in
   the strict sense. For instance `detector:NXdetector` and
   `detector:NXdetector` would match, as would `data` and `data`. 
2. Two elements, *a* and *b* are considered matching, if both have the same 
   type and for only one the `name` field is set. For instance `:NXdetector`
   and `mythen:NXdetector` would match but `pilatus:NXdetector` and
   `mythen:NXdetector` would not. 

3. If both have either their `name` or `type` field set and those are equal. 
   This is again simple. For instance `:NXdetector` and `:NXdetector` would
   match as well as `data` and `data` would. 

Using paths
-----------

All that theory sounds rather complicated. However, it is quite simple in
practice. 
Nearly all objects in the NeXus API have a :py:attr:`path` attribute
(read-only) which returns the absolute path of an object. It should be
mentioned that this path does not include the file name. The latter one could
be obtained via the, read-only, :py:attr:`filename` attribute.

Functions and classes concerning NeXus paths are available from the 
:py:mod:`pni.io.nx` package.
A NeXus path is represented by an instance of :py:class:`nxpath`.  
The :py:func:`Path.from_string` utiltiy function creates a new instance of a path

.. code-block:: python 

    from pninexus import nexus

    p = nexus.Path.from_string("/:NXentry/:NXinstrument/:NXdetector/data")

An empyt instance can be obtained by usint the default constructor 
of :py:class:`nxpath`

.. code-block:: python

    from pninexus import nexus

    p = nexus.Path()
    

The *file-* and *attribute-section* of the path is accessible via the 
:py:attr:`filename` and :py:attr:`attribute` attributes of a path instance
which are both read- and writeable 

.. code-block:: python
    
    from __future__ import print_function
    from pninexus import nexus

    path = nexus.Path.from_string("test.nxs://:NXentry/:NXinstrument/:NXsample/:NXtransformation/x@units")

    print(path.filename)
    path.filename = "test.nxs"

    print(path.attribute)
    path.attribute = "units"


For the *object-section* a path acts like a stack whose first element is
accessible via the :py:attr:`front` and the last element vai the 
:py:attr:`back` attribute. These attributes are only readable. 

.. code-block:: python

    from __future__ import print_function
    from pninexus import nexus

    path =  nexus.Path.from_string("/:NXentry/:NXinstrument")

    print(path.front) #output: {'name':'/','base_class':'NXroot'}
    print(path.back)  #output: {'name':'','base_class':'NXinstrument'}


The elements of the *object-section* are represented as Python dictionaries
with keys ``name`` and ``base_class`` where the former one is the name of the group
or field and the latter one, in the case of a group, is the value of the
``NX_class`` attribute of the object. In the case of fields the ``base_class`` 
entry is always an empty string.

Appending and prepending elements to the *object-section*

.. code-block:: python

    from __future__ import print_function
    from pninexus import nexus

    path = nexus.Path.from_string("/")

    path.push_back("scan_1:NXentry")
    path.push_back(name="instrument",base_class="NXinstrument")

    print(path)
    #output: scan_1:NXentry/instrument:NXinstrument

    path = make_path("scan_1:NXentry/:NXinstrument")
    path.push_front("/:NXroot")
    print(path)
    #output: /scan_1:NXentry/:NXinstrument

Using a path as a stack 

.. code-block:: python

    from __future__ import print_function
    from pninexus import nexus

    path = nexus.Path.from_string("/:NXentry/:NXinstrument/mythen:NXdetector/data")
    print(path)
    #output: /:NXentry/:NXinstrument/mythen:NXdetector/data

    print(path.front)
    #output: {name:"/",type:"NXroot"}

    #remove the root group
    path.pop_front()
    print(path)
    #output: ":NXentry/:NXinstrument/mythen:NXdetector/data"

    #remove the back entry
    path.pop_back()
    print(path)
    #output: ":NXentry/:NXinstrument/mythen:NXdetector"



One can iterate of the *object-section* of a path 

.. code-block:: python

    from __future__ import print_function
    from pninexus import nexus
    
    path = nexus.Path.from_string(":NXentry/:NXinstrument/:NXdetector/data")

    for e in path:
        print(e["name"], p["base_class"])

To check if two paths are matching use the :py:func:`match` function

.. code-block:: python

    from __future__ import print_function
    from pninexus import nexus

    det_path = nexus.Path.from_string("/:NXentry/:NXinstrument/:NXdetector")
    p =  nexus.Path.from_string("/scan_1:NXentry/p08:NXinstrument/mythen:NXdetector")

    print(nexus.match(det_path, p))
    #output: True

