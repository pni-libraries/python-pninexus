
API documentation
=================

The package consists of two major subpackages 

* the :py:mod:`pninexus.h5cpp` which is a thin wrapper around the *h5cpp* library
* and :py:mod:`pninexus.nexus` which wraps classes and functions from the 
  :cpp:any:`pni::io::nexus` namespace of *libpniio*. 

The ``pninexus.h5cpp`` package
==============================

.. automodule:: pninexus.h5cpp
   :members:
   
.. toctree::
   :maxdepth: 1
   
   h5cpp/attributes
   h5cpp/dataspace
   h5cpp/datatype
   h5cpp/file
   h5cpp/properties
   h5cpp/node






The ``pninexus.nexus`` package
==============================

.. automodule:: pninexus.nexus
   :members:
   :undoc-members:

.. toctree::
   :maxdepth: 1
   
   nexus/files
