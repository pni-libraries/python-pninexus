==================
Working with files
==================

The first thing to do when working with HDF5 is to create new or opening 
existing files. Classes and functions related to file operations can be 
found in the :py:mod:`pninexus.h5cpp.file` package. 

Creating a file
===============

.. automodule:: pninexus.h5cpp.file

In the :py:func:`create` function is used to create new files. In 
the simplest case you can use this function with a single argument: the name 
of the new file  

.. code-block:: python

   from pninexus import h5cpp
   
   h5file = h5cpp.file.create("test.h5")
   
This function returns an instance of :py:class:`File` which can be used 
for all further operations on files. Unlike with other HDF5 Python wrappers 
the :py:class:`File` class cannot be used as an entry point to the node 
hierarchy in the file. For that purpose one has to obtain the root group 
with 

.. code-block:: python 

   h5file = .... 
   
   root = h5file.root() 
   
This sounds like a inconvenience at first but has the major advantage that the 
semantic of classes remains clear: a file is not a group and thus should not 
behave as one. 

A subsequent call to :py:func:`create` with the same filename would raise 
a :py:exc:`RuntimeError` exception as the file already exists. In order 
to overwrite existing files the second argument must be set with

.. code-block:: python

   h5file = h5cpp.file.create("test.h5",h5cpp.file.AccessFlags.TRUNCATE)
   
See the API documentation of :py:class:`AccessFlags` for all 
available flags. 
Additional fine tuning of the file creation process is possible using 
two more keyword arguments to this function: `fcpl` and `fapl` 
which accept instances of :py:class:`pninexus.h5cpp.property.FileCreationList` and 
:py:class:`pninexus.h5cpp.property.FileAccessList`. These classes refer to a 
*file creation property list* and a *file access property list* respectively.  


Opening files
=============

To open an existing file use the :py:func:`open` function. 

.. code-block:: python

   from pninexus import h5cpp
   
   h5file = h5cpp.file.open("test.h5")
   
which, by default, opens the file in read-only mode. To get read-write 
access to a file use 

.. code-block:: python

   h5file = h5cpp.file.open("test.h5",flags=h5cpp.file.AccessFlags.READWRITE)
   
:py:func:`open` would raise an exception if the filename refers not 
to an HDF5 file. To avoid such situations use the :py:func:`is_hdf5_file`
to check if the requested file is an HDF5 file. 

.. code-block:: python

   if not h5cpp.file.is_hdf5_file("test.h5"):
      raise RuntimeError("not an HDF5 file")
      
   h5file = h5cpp.file.open("test.h5")

   
Working with files
==================

At first: a file can be closed by calling its :py:func:`close` method. 

.. code-block:: python

   h5file.close()
   
However, this is only necessary if the file should be closed deliberately. 
Usually all objects imported by :py:mod:`pninexus.h5cpp` are automatically 
destroyed when they're losing their scope. 