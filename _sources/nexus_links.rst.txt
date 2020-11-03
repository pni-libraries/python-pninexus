==================
Working with links
==================

Nexus files support two kinds of links 

* *internal* links between objects within the same file
* and *external* links between objects in different files

Links are an appropriate tool to avoid data duplication. A typical application
for links are the fields in the *NXdata* instance below *NXentry*. *NXdata*
provides data recorded during a measurement intended for easy plotting.  In
virtually all cases the data which should go to *NXdata* is already stored in
other places within the same NeXus file.  Without links we would have to copy
this data to *NXdata* which would be an awefull waste of time and space in
particular when the amount of data is getting large (think of a 3D image block
recorded with a 2D detector). 

With links we can simply provide references to the original data below
*NXdata*. 

.. only:: latex

    .. figure:: link.pdf
        :align: center
        :width: 75%
        
        The internal links in *NXdata* point to already existing data in 
        other locations within the same file. This avoids data duplication
        within a single file.

.. only:: html

    .. figure:: link.svg
        :align: center
        :width: 100%
        
        The internal links in *NXdata* point to already existing data in 
        other locations within the same file. This avoids data duplication
        within a single file.

In the above example the *data* and *position* field are members of the 
*NXdetector* and *NXsample* group respectively. However, by means of links 
we can make these two fields available as children of the *NXdata* group
without copying data. In this case internal links have been used as all data
resides in the same file. The red lines in the sketch denote the links

For *external*-links the canonical use case is a detector which writes its own
NeXus file with data (we will refer to this file as the *detector*-file). 
All additional data is writen by the control system to a second NeXus file
which we will call *master*-file. To make the detector data available from the
*master*-file we have two possibilities 

* either copy the entire data to the *master*-file 
* or use an *external*-link.

The drawbacks of the former solution are obvious: we would need to copy all the
data to a different file. However, with *external*-links we can do something
like this 

.. only:: latex
    
    .. figure:: external_link.pdf
        :align: center
        :width: 100%

        With an external link we can avoid copying data between two individual
        NeXus files. The data in the detector file can be accessed easily from
        within the master file.

.. only:: html

    .. figure:: external_link.svg
        :align: center
        :width: 100%
        
        With an external link we can avoid copying data between two individual
        NeXus files. The data in the detector file can be accessed easily from
        within the master file.


Here the detector group can be accessed from the *master*-file as if it would
be a part of it. The link is totally transparent to the user. 

Creating links 
==============

Internal links
--------------
Links are created using the :py:func:`link` function.
The signature of the function is the same for internal and external links. 
Only the path to the link target differs.
An internal links can be created linke this

.. code-block:: python
    
    from __future__ import print_function
    import pninexus.nexus as nexus

    f = nexus.open_file("master.nxs",False)
    r = f.root()
    data = nx.get_object(r,"/:NXentry/:NXdata")

    nexus.link("/entry/instrument/detector/data",data,"data")

The :py:func:`link` function creates now a link to the *data* field below
*NXdata* with name *data*. The important thing to note here is that the path
pointing to the link target must not contain elements that consist only of
types (path elements like ``/:NXentry/`` for instance. This due to the fact,
that the linking feature is provided by the HDF5 library which has no idea
about NeXus semantics.  The object the HDF5 path referes to must not
necessarily  exist at link time.  Alternatively, within a file we can do a link
of an existing object. In this case the above example would look like this

.. code-block:: python
    
    from __future__ import print_function
    import pninexus.nexus as nexus

    f = nexus.open_file("master.nxs",False)
    r = f.root()
    detector = nexus.get_object(r,"/:NXentry/:NXinstrument/:NXdetector/data")
    data = nx.get_object(r,"/:NXentry/:NXdata")

    nexus.link(detector,data,"data")

In this case the object we are linking to must obviously exist in order to make
the call to :py:func:`get_object` successful.

External links
--------------

Creating an external links is quite similar to the first example shown above 
for internal links. The only difference is that the path to the link target
starts with a filename

.. code-block:: python
    
    from __future__ import print_function
    import pninexus.nexus as nexus

    f = nexus.open_file("master.nxs",False)
    r = f.root()
    detector = nexus.get_object(r,"/:NXentry/:NXinstrument/:NXdetector")

    nexus.link("detector.nxs://entry/instrument/detector/data",detector,"data")

For external links relative paths to the file should be used. Otherwise moving
the files to a different file system can cause unresolvable links!


Using links in existing files
=============================

Links can also be used to create more robust parsers for NeXus files. 
In many cases files may contain unresolvable links (typically external ones). 
As an alternative to iterating over a group one could use either 
:py:func:`get_links` or :py:func:`get_links_recursive` in order to obtain 
a list of links in the file and investigate their status before accessing the 
objects the links refer to. 

.. code-block:: python

    import pninexus.nexus as nexus

    f = nexus.open_file("scan_000001.nxs")
    entry = nexus.get_object(f.root(),"/:NXentry")
    links = nexus.get_links_recursive(entry)

    broken_links = [link for link in links if link.status == nexus.nxlink_status.INVALID]
    objects = [link.resolve() for link in links if link.status == nexus.nxlink_status.VALID]

:py:func:`get_links_recursive` returns a list of all links (instances of
:py:class:`nxlink`) below :py:obj:`entry` and its subgroups. Since links do not
access the objects they are refering to no exception will be thrown if one of
the links cannot be resolved due to a missing target.  The list
:py:obj:`borken_links` holds all links which are not resolveable in the file
while :py:obj:`objects` the objects refered to by all valid links. 

:py:func:`get_links` returns only a list of the direct children of a group. 

