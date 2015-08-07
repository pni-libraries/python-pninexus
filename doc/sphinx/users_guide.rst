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

Reading non nexus files
-----------------------

Basic Nexus support
-------------------

NeXus algorithms
----------------

