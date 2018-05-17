======================
File related functions
======================

.. automodule:: pninexus.nexus

The :py:mod:`pninexus.nexus` package adds some useful wrapper functions 
around the native HDF5 functions. 

.. autofunction:: create_file(path,flags=AccessFlags.EXCLUSIVE,fcpl=FileCreationList(),fapl=FileAccessList())

   A thin wrapper function around :py:func:`pninexus.h5cpp.file.create` HDF5 
   function. It not only creates the file but also creates the required 
   attributes no the root group of the file. Using this function ensures
   that a file created with it will pass a test with :py:func:`is_nexus_file`. 

   :param str path: path to the file
   :param pninexus.h5cpp.file.AccessFlags flags: HDF5 flags used to open the file 
   :param pninexus.h5cpp.property.FileCreationList lcpl: optional file creation list
   :param pninexus.h5cpp.property.FileAccessList lapl: optional file access list

.. autofunction:: open_file(path,flags=AccessFlags.READONLY,fapl=FileAccessList())

   A thin wrapper function around :py:func:`pninexus.h5cpp.file.open`. 

   :param str path: path to the file
   :param pninexus.h5cpp.file.AccessFlags flags: access flags used to open the file
   :param pninexus.h5cpp.property.FileAccessList lapl: optional file access property list

.. autofunction:: is_nexus_file(path)

   Checks if the file referenced by `path` is a NeXus file. This function 
   is more restrictive than :py:func:`pninexus.h5cpp.file.is_hdf5_file`. 
   It not only checks whether or not the file is an HDF5 file but also 
   whether all required attributes are set at the root group of the HDF5 file. 
   
   :param str path: path to the file on the file system
