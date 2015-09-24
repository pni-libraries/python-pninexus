==================
Working with links
==================

Nexus files support two kinds of links 

* *internal* links between objects within the same file
* and *external* links between objects in different files

Both types are supported by this Python wrapper. From the point of the library
interface, internal and external links work exactly the same, just the path is
different. 

.. code-block:: python
    
    from __future__ import print_function
    import pni.io.nx.h5 as nexus

    f = nexus.open_file("master.nxs",False)
    r = f.root()
    data = nx.get_object(r,"/:NXentry/:NXdata")

    nexus.link("/entry/instrument/detector/data",data,"data")

The :py:func:`link` function creates now a link to the *data* field below
*NXdata* with name *data*. The important thing to note here is that the path
pointing to the link target must not contain elements that consist only of
types. This due to the fact, that the linking feature is provided by the HDF5
library which has no idea about NeXus semantics. 
The object the HDF5 path referes to must not necessarily  exist at link time. 
Alternatively, within a file we can do a link of an existing object. In this
case the above example would look like this

.. code-block:: python
    
    from __future__ import print_function
    import pni.io.nx.h5 as nexus

    f = nexus.open_file("master.nxs",False)
    r = f.root()
    detector = nexus.get_object(r,"/:NXentry/:NXinstrument/:NXdetector/data")
    data = nx.get_object(r,"/:NXentry/:NXdata")

    nexus.link(detector,data,"data")

In this case the object we are linking to must obviously exist in order to make
the call to :py:func:`get_object` successful.

External links can be done like in the first example but with the filename
section at the beginning of the path 

.. code-block:: python
    
    from __future__ import print_function
    import pni.io.nx.h5 as nexus

    f = nexus.open_file("master.nxs",False)
    r = f.root()
    detector = nexus.get_object(r,"/:NXentry/:NXinstrument/:NXdetector")

    nexus.link("detector.nxs://entry/instrument/detector/data",detector,"data")
