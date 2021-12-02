Users Guide
===========

Introduction
------------

The PNI Python package provides bindings to the h5cpp and PNI C++ libraries

* ``libh5cpp``
* ``libpninexus``

The term binding might not be entirely correct as the PNI libraries are mainly
C++ templates. One can consider this package as an implementation providing 
the functionality of the templates. In particular not all the features exposed
by the C++ libraries are imported into Python as equivalent native Python 
solutions exist. For instance ``libpninexus`` provides templates for
multidimensional arrays which are not required in Python as we have ``numpy``
arrays. 

The code accessed by this Python packages comes from
``libh5cpp`` and ``libpninexus`` and addresses the following problems

* reading legacy ASCII and binary (mainly image) files
* provide access to NeXus/HDF5 files


The top level node of the package
---------------------------------

Reading legacy ASCII and binary files
-------------------------------------

Nexus support
-------------

NeXus is supported with the HDF5 file format as the physical storage format. 
:py:mod:`pninexus` provides NeXus support via two packages 

* :py:mod:`pninexus.h5cpp` which is a low level wrapper around the *h5cpp* 
  C++ library for HDF5 
* and :py:mod:`pninexus.nexus` providing high level functions related to the 
  NeXus file format. 
  
.. attention::

   It is strongly encouraged to read the following two chapters in the correct 
   order: start with the uses guide for the :py:mod:`pninexus.h5cpp` package
   and than read the chapter about :py:mod:`pninexus.h5cpp`. 

.. toctree::
   :maxdepth: 1
   
   using_h5cpp
   using_nexus



