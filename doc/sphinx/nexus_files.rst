========================
Working with Nexus files
========================

.. automodule:: pninexus.nexus
   :noindex:

NeXus files are technically nothing else than HDF5 files. However, for an
HDF5 file to become a standard compliant NeXus file it must have some
special attributes attached to its root group. It would be possible
to create a NeXus file with the functions provided by
:py:mod:`pninexus.h5cpp.file` package. However, the :py:mod:`pninexus.nexus`
package provides a :py:func:`create_file` function.
Concerning its signature this function is nothing else than a simple
wrapper around :py:func:`pninexus.h5cpp.file.create` with the benefit
that it creates all the required standard attributes on the root group
of the file once it created it.

.. code-block:: python

   from pninexus import h5cpp
   from pninexus import nexus

   nxfile = nexus.create_file("nexus_file.nxs")

Like its counterpart in the :py:mod:`pninexus.h5cpp` this function will
throw an exception if a file of the given name already exists and like
its HDF5 counterpart this behavior can be controlled with HDF5 control
flags

.. code-block:: python

   from pninexus import h5cpp
   from pninexus.h5cpp.file import AccessFlags
   from pninexus import nexus

   nxfile = nexus.create_file("nexus_file.nxs",AccessFlags.TRUNCATE)


The same is true for opening files for which the :py:mod:`pninexus.nexus`
package provides the :py:func:`open_file` function

.. code-block:: python

   from pninexus import nexus

   nxfile = nexus.open_file("nexus_file.nxs")

which opens a file in read-only mode. Again, using HDF5 file access flags can
be used to change this default behavior

.. code-block:: python

   from pninexus.h5cpp.file import AccessFlags
   from pninexus import nexus

   nxfile = nexus.open_file("test_file.nxs",AccessFlags.READWRITE)

In all cases these functions return an instance of
:py:class:`pninexus.h5cpp.file.File`.

In order to check whether a file is a NeXus compliant HDF5 file the package
provides the :py:func:`is_nexus_file` function

.. code-block:: python

   import sys
   from pninexus.nexus import is_nexus_file

   if not is_nexus_file("a_file.nxs"):
      print("This file is not a nexus file")
      sys.exit(1)

This function performs two tasks

1. it checks if the file is an HDF5 file
2. it checks if the all attributes required by the standard are available at
   the root group of the file.

.. attention::

   It should be mentioned that this function is rather strict. To be a valid
   NeXus file all required attributes must be available at the root group of
   the file. If this function fails you may want to fallback to
   :py:func:`pninexus.h5cpp.file.is_hdf5_file` to check if the file is
   least an HDF5 file.
