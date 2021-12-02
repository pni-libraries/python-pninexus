==========================================
Using the :py:mod:`pninexus.h5cpp` package
==========================================

The :py:mod:`pninexus.h5cpp` package is a thin wrapper around the *h5cpp* 
C++ wrapper for HDF5. The functionality provided by this package is not 
related to NeXus but rather to HDF5 in general. However, as the NeXus 
support package :py:mod:`pninexus.nexus` is built on top of this package 
a firm understanding of the functionality provided by :py:mod:`pninexus.h5cpp`
is mandatory to work with the NeXus package. 

This users guide does not provide a detailed documentation of the design 
concepts of *h5cpp* but rather focuses on its usage. For more details
consult the `h5cpp documentation`_.
Unlike other wrappers for HDF5 (for instance `h5py`_) the :py:mod:`pninexus.h5cpp` 
package has a different focus on completeness rather than on simple usage (
though it is not extremely hard to use it). 


.. _h5py: https://www.h5py.org/
.. _h5cpp documentation: https://ess-dmsc.github.io/h5cpp/index.html

.. toctree::
   :maxdepth: 1
   
   h5cpp_files
   h5cpp_groups
   h5cpp_datatypes
   h5cpp_dataspace
   h5cpp_dataset
   h5cpp_attributes
