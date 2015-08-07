
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

.. automodule:: pni.io.nx
   :members: 
   :undoc-members:



The ``pni.io.nx.h5`` package
============================

.. automodule:: pni.io.nx
   :members: 
   :undoc-members:
