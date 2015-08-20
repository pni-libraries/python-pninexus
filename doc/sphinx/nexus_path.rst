Addressing objects: the NeXus path
==================================

An introduction
---------------

Every object within a NeXus file can be addressed by a so called *NeXus path*.
*Nexus paths* can become rather complex and a good starting point is to have a
look on some examples. Lets start with a very simple one

.. code-block:: bash

    detector_1.nxs://scan_1/instrument/detector/transformation/phi 

which refers to the field holding the data of motor *phi* in the transformation
group of a detector. To access the `units` attribute we could use 

.. code-block:: bash

    detector_1.nxs://scan_1/instrument/detector/transformation/phi@units 

However, this path is merely more than a standard HDF5 path with a 
filename prepended and an attribute name appended to it. To make this path more
NeXus-style we can add the group types 


.. code-block:: bash

    detector_1.nxs://scan_1:NXentry/instrument:NXinstrument/ \
                    detector:NXdetector/transformation:NXtransformation/phi@units 

in which case we do not only specify the names of the groups but also their
types. In a final step we could make this path more generic by omitting the 
group names 

.. code-block:: bash

    detector_1.nxs://:NXentry/:NXinstrument/:NXdetector/:NXtransformation/phi@units 

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
An instance can be constructed from the string representation of a path 

.. code-block:: python 

    from pni.io.nx import nxpath 

    p = nxpath("/:NXentry/:NXinstrument/:NXdetector/data")
    

The *file-* and *attribute-section* of the path is accessible via the 
:py:attr:`filename` and :py:attr:`attribute` attributes of a path instance. 

.. code-block:: python

    path = .... 

    print(path.filename)
    path.filename = "test.nxs"

    print(path.attribute)
    path.attribute = "units"

For the *object-section* a path acts like a stack. One can append elements to
the beginning and the end of a path, pop elements from the path, access the 
first and last element via the :py:attr:`front` and :py:attr:`back` attribute. 

The elements of the *object-section* are represented as Python dictionaries
with keys ``name`` and ``base_class`` where the former one is the name of the group
or field and the latter one, in the case of a group, is the value of the
``NX_class`` attribute of the object.

Appending and prepending elements to the *object-section*

.. code-block:: python

    path = nxpath()

    path.append("scan_1:NXentry")
    path.append(name="instrument",base_class="NXinstrument")

    print(path)
    #output: scan_1:NXentry/instrument:NXinstrument

    path.prepend("/:NXroot")
    print(path)
    #output: /scan_1:NXentry/instrument:NXinstrument

Using a path as a stack 

.. code-block:: python

    path = nxpath("/:NXentry/:NXinstrument/mythen:NXdetector/data")
    print(path)
    #output: /:NXentry/:NXinstrument/mythen:NXdetector/data

    print(path.back)
    #output: {name:"/",type:"NXroot"}

    #remove the root group
    path.pop_back()
    print(":NXentry/:NXinstrument/mythen:NXdetector/data")


Iteration of path elements

.. code-block:: python

    path = ....

    for e in path:
        print(e["name"],p["type"])

Paths can be joined using the ``+`` operator 

.. code-block:: python

    a = nxpath(...)
    b = nxpath(...)

    c = a+b

However, the two paths have to follow some rules in order to be joined 

* :py:data:`a` must not have a *attribute-section*
* :py:data:`b` must not have a *file-section*
* :py:data:`b` must not be an absolute path

Though these rules seem to be complicated at the first glimpse they are totally
natural if you think about them for a moment. For instance, how would you join
two paths if the left hand side of the operator has an attribute section which
is most naturally the terminal element of every path. 
If any of these rules is violated a :py:exc:`ValueError` exception will be
thrown.
