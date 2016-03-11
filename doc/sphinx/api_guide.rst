
API documentation
=================

For a Python programmer this is most probably the most important part of this
very short documentation. ``libpnicore`` provides a set of exception types
which are used by itself and other libraries like ``libpniio``. 
``pni.core`` provides wrappers to some of these exceptions and include
translator functions which ensure that the appropriate Python exception is
raised for every C++ exception. 

Some of the exceptions provided by ``libpnicore`` can be mapped directly on
default exceptions already provided by Python. These are the following ones

=======================================  ======================
C++ exception                            Python exception
=======================================  ======================
`pni::core::memory_allocation_error`     `MemoryError`
`pni::core::memory_not_allocated_error`  `MemoryError`
`pni::core::index_error`                 `IndexError`
`pni::core::key_error`                   `KeyError`
`pni::core::not_implemented_error`       `NotImplementedError`
`pni::core::value_error`                 `ValueError`
`pni::core::type_error`                  `type_error`
=======================================  ======================

However, some of the exceptions provided by `libpnicore` do not have a proper
counterpart in Python. In these cases new exceptions have been provided by
`pni.core` which to which the original C++ exceptions get translated. 
These exceptions are

=======================================  =============================
C++ exception                            Python exception
=======================================  =============================
`pni::core::file_error`                  `pni.core.FileError`
`pni::core::shape_mismatch_error`        `pni.core.ShapeMismatchError`
`pni::core::size_mismatch_error`         `pni.core.SizeMismatcherror`
`pni::core::range_error`                 `pni.core.RangeError`
`pni::core::iterator_error`              `pni.core.IteratorError`
`pni::core::cli_argument_error`          `pni.core.CliArgumentError`
`pni::core::cli_error`                   `pni.core.CliError`
=======================================  =============================

For more detailed information about under which conditions these exceptions are 
thrown, consult the `libpnicore` users guide.


.. automodule:: pni.core
   :members:
   :undoc-members:

The ``pni.io`` package
======================

.. automodule:: pni.io
   :members:
   :undoc-members:


The ``pni.io.nx`` package
=========================

.. py:module:: pni.io.nx

.. autoclass:: pni.io.nx.nxpath
   :members:
   :undoc-members:

.. autofunction:: pni.io.nx.make_relative
.. autofunction:: pni.io.nx.make_path
.. autofunction:: pni.io.nx.match
.. autofunction:: pni.io.nx.join
.. autofunction:: pni.io.nx.is_root_element
.. autofunction:: pni.io.nx.is_empty
.. autofunction:: pni.io.nx.has_name
.. autofunction:: pni.io.nx.has_class
.. autofunction:: pni.io.nx.is_absolute



The ``pni.io.nx.h5`` package
============================

.. py:module:: pni.io.nx.h5

.. autoclass:: pni.io.nx.h5.nxfile
   :members:
   :undoc-members:

.. autofunction:: pni.io.nx.h5.create_file
.. autofunction:: pni.io.nx.h5.create_files
.. autofunction:: pni.io.nx.h5.open_file

.. autoclass:: pni.io.nx.h5.nxgroup
   :members:

.. autoclass:: pni.io.nx.h5.nxfield
   :members:
   :undoc-members:

.. autoclass:: pni.io.nx.h5.nxattribute
   :members:
   :undoc-members:

.. autoclass:: pni.io.nx.h5.deflate_filter
   :members:
   :undoc-members:

.. autoclass:: pni.io.nx.h5.nxlink
   :members:
   :undoc-members:

.. autofunction:: pni.io.nx.h5.xml_to_nexus
.. autofunction:: pni.io.nx.h5.get_size
.. autofunction:: pni.io.nx.h5.get_name
.. autofunction:: pni.io.nx.h5.get_rank
.. autofunction:: pni.io.nx.h5.get_unit
.. autofunction:: pni.io.nx.h5.get_class
.. autofunction:: pni.io.nx.h5.get_object
.. autofunction:: pni.io.nx.h5.get_path
.. autofunction:: pni.io.nx.h5.link
.. autofunction:: pni.io.nx.h5.get_links
.. autofunction:: pni.io.nx.h5.get_links_recursive

