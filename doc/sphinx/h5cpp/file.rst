=============================
:py:mod:`pninexus.h5cpp.file`
=============================

.. automodule:: pninexus.h5cpp.file

.. autosummary::

   AccessFlags
   Scope
   File
   create
   open
   is_hdf5_file


.. autoclass:: AccessFlags

   Enumeration type determining the constraits applied to a file when creating
   or opening it. 
   
   .. autoattribute:: TRUNCATE
      :annotation: = truncate a file during creation if a file of same name already exists
   .. autoattribute:: EXCLUSIVE
      :annotation: = raise exception during creation if a file of same name already exists 
   .. autoattribute:: READWRITE
      :annotation: = open a file in read-write mode
   .. autoattribute:: READONLY
      :annotation: = open a file in read-only mode 
   
.. autoclass:: Scope 

   Enumeration type determining the scope to use when flushing a file
   
   .. autoattribute:: LOCAL
      :annotation: = local scope 
   .. autoattribute:: GLOBAL
      :annotation: = global scope
   
.. autoclass:: File
   
   Class representing an HDF5 file. 
   
   .. autoattribute:: intent
   
      read-only property returning the intent used when opening the file. 
      This would typically be read-write or read-only. 
      
      :return: file access flags
      :rtype: AccessFlags
      
   .. autoattribute:: is_valid
   
      Read-only property returning :py:const:`True` if the file is a valid 
      HDF5 object, :py:const:`False` otherwise. 
      
      :rtype: boolean 
   
   .. autoattribute:: path 
   
      Read-only property returning the path to the file. 
      
      :rtype: str
   
   .. autoattribute:: size 
   
      Read only property returning the size of the file in bytes. 
      
      :rtype: integer
   
   
   .. automethod:: flush(scope=Scope.GLOBAL)
   
      Flushes all buffers to the OS. 
      
      :param Scope scope: the scope of the flush operation
   
   .. automethod:: close
   
      Close the file. After calling this function :py:attr:`is_valid` should 
      be :py:const:`False`. 
   
   .. automethod:: root
   
      Return an instance of the root group of the file. 
      
      :return: root group 
      :rtype: :py:class:`pni.io.h5cpp.node.Group`

.. autofunction:: create(path,flags=AccessFlags.EXCLUSIVE)

   Create a new HDf5 file. 
   
   :param str path: the path to the new file 
   :param AccessFlags flags: access flags to use for the file creation
   :return: new file object
   :rtype: :py:class:`File`
   :raise RuntimeError: in case of a failure

.. autofunction:: open(path,flags=AccessFlags.READONLY)

   Open an existing HDF5 file. By default the file will be opened in read-only 
   mode.  
   
   :param str path: path to the file to open 
   :param AccessFlags flags: the flags to use for opening the file 
   :return: new file object 
   :rtype: :py:class:`File`
   :raise RuntimeError: in case of a failure

.. autofunction:: is_hdf5_file(path)

   :param str path: the path to the file to check 
   :return: :py:const:`True` if the file is an HDF5 file, :py:const:`Flase` otherwise
   :rtype: boolean