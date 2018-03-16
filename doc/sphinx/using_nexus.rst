==========================================
Using the :py:mod:`pninexus.nexus` package
==========================================

.. code-block:: python
    
    import pninexus as nexus

As :py:mod:`pninexus.h5cpp` is a wrapper for the *h5cpp* C++ library, 
:py:mod:`pninexus.nexus` is a thin wrapper around the `pni::io::nexus` namespace 
of *libpniio*. With the new API a lot has changed. While with the old interface
all functionality was provided by *libpniio* this has now moved to *h5cpp*. 
Naturally also the Python wrapper has changed. 



.. toctree::
    :maxdepth: 2 
   
    nexus_files
    nexus_path
    
    nexus_groups
    nexus_fields
    nexus_attributes
    nexus_links
    nexus_xml
