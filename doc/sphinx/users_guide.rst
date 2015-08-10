Users Guide
===========

Introduction
------------

The PNI Python package provides bindings to the PNI C++ libraries

* ``libpnicore``
* ``libpniio``

The term binding might not be entirely correct as the PNI libraries are mainly
C++ templates. One can consider this package as an implementation providing 
the functionality of the templates. In particular not all the features exposed
by the C++ libraries are imported into Python as equivalent native Python 
solutions exist. For instance ``libpnicore`` provides templates for
multidimensional arrays which are not required in Python as we have ``numpy``
arrays. 

From ``libpnicore`` only the exceptions are imported as they are used by 
``libpniio``. Most of the code accessed by this Python packages comes from
``libpniio`` and addresses the following problems

* reading legacy ASCII and binary (mainly image) files
* provide access to NeXus files


The top level node of the package
---------------------------------

Reading legacy ASCII and binary files
-------------------------------------

Basic Nexus support
-------------------
At the current status (version 1.0.0) the ``libpniio`` does only support
Nexus using the HDF5 storage backend. In order to use the provided Nexus
functionality import the package with something like this

.. code-block:: python
    
    import pni.io.nx.h5 as nexus


.. toctree::
    :maxdepth: 1
    
    nexus_files
    nexus_groups
    nexus_fields
    nexus_attributes


Advanced NeXus
--------------

.. toctree::
    :maxdepth: 1

    nexus_xml

