===============================
:py:mod:`pni.io.h5cpp.datatype`
===============================

.. automodule:: pni.io.h5cpp.datatype

Enumerations
============

.. autoclass:: Class

   Enumeration determining the class to which a type belongs. 
   
   .. autoattribute:: NONE
      :annotation: = type belongs to no class (not a type at all)
   .. autoattribute:: INTEGER
      :annotation: = type is an integer type
   .. autoattribute:: FLOAT
      :annotation: = floating point type
   .. autoattribute:: TIME
      :annotation: = time type
   .. autoattribute:: STRING
      :annotation: = string type
   .. autoattribute:: BITFIELD
      :annotation: = bitfield type 
   .. autoattribute:: OPAQUE
      :annotation: = an opaque type
   .. autoattribute:: COMPOUND
      :annotation: = a compound type 
   .. autoattribute:: REFERENCE
      :annotation: = a reference to a dataset region
   .. autoattribute:: ENUM
      :annotation: = an enumeration type
   .. autoattribute:: VARLENGTH
      :annotation: = a variable length datat type
   .. autoattribute:: ARRAY
      :annotation: = an array datatype

.. autoclass:: Order

   Enumeration type determining the byte order used for types 
   
   .. autoattribute:: LE
      :annotation: = little endian 
   .. autoattribute:: BE
      :annotation: = big endian

.. autoclass:: Sign

   Enumeration determining the sign convention
   
   .. autoattribute:: TWOS_COMPLEMENT 
      :annotation: = signed type using twos complement 
   .. autoattribute:: UNSIGNED 
      :annotation: = unsigned type 

.. autoclass:: Norm
   
   Enumeration determining the normalization 
   
   .. autoattribute:: IMPLIED 
      :annotation: = ????
   .. autoattribute:: MSBSET
      :annotation: = ????
   .. autoattribute:: NONE
      :annotation: = no normalization used 

.. autoclass:: Pad

   Enumeration determining the general padding used for datatypes. 
   
   .. autoattribute:: ZERO
      :annotation: = 0 padding is used 
   .. autoattribute:: ONE 
      :annotation: = 1 padding
   .. autoattribute:: BACKGROUND
      :annotation: = ????

.. autoclass:: StringPad

   Enumeration determining the padding used for strings 
   
   .. autoattribute:: NULLTERM 
      :annotation: = strings are terminated with a \0
   .. autoattribute:: NULLPAD
      :annotation: = strings are padded with \0
   .. autoattribute:: SPACEPAD
      :annotation: = strings are padded with whitespaces 

.. autoclass:: Direction

   .. autoattribute:: ASCEND
      :annotation: = ascending order
   .. autoattribute:: DESCEND
      :annotation: = descending order

.. autoclass:: CharacterEncoding

   Enumeration determining the character encoding used for strings 
   
   .. autoattribute:: ASCII
      :annotation: using standard ASCII encoding
   .. autoattribute:: UTF8
      :annotation: using Unicode encoding (UTF8)

Classes
=======

.. autoclass:: Datatype

   This is the base class for all other type classes. 
   
   .. autoattribute:: type
      
      Read-only property providing the type of the datatype. 
      
      :return: datatype type
      :rtype: Class
      
   .. autoattribute:: super
   
      Read-only attribute returning the super-type. The meaning of what is 
      returned highly depends on the class the datatype belongs to. 
      Consult the HDF5 users guide for more information about this. 
      
      :rtype: Datatype
   
   .. autoattribute:: size 
   
      Read-only property returning the total size of te type in bytes. 
      
      :return: byte size of a type
      :rtype: integer
      
   .. autoattribute:: is_valid 
   
      Returns :py:const:`True` if the datatype is a valid HDF5 object, 
      :py:const:`False` otherwise. 
      
      :rtype: boolean
      
   .. automethod:: native_type(dir) 
   
      Return the native type of this particular type. 
      
      :param Direction dir: the search direction
      :return: native type 
      :rtype: Datatype
      
   .. automethod:: has_class

.. autoclass:: Float

   Type used for flaoting point numbers. 

.. autoclass:: Integer

   Type used for integer numbers. 

.. autoclass:: String
   
   String type. Usually, an instance of :py:class:`String` is constructed 
   using one of the two static factory functions of the class. 
   
   For variable length strings use 
   
   .. code-block:: python 
   
      type = String.variable()
      
   and for fixed size string types 
   
   .. code-block:: python
   
      type = String.fixed(size=10) #string type with 10 characters 
      
   .. autoattribute:: is_variable_length
   
      Returns :py:const:`True` if the datatype is a variable length string type,
      false otherwise. 
      
      :rtype: boolean
      
   .. autoattribute:: encoding 
   
      Read-write property to set or get the character encoding used for the 
      string type. 
      
      .. code-block:: python
      
         type = String.variable()         
         type.encoding = CharacterEncoding.UTF8
   
   .. autoattribute:: padding 
   
      Read-write propety to set or get the padding used for a string type. 
      
      .. code-block:: python
      
         type = String.fixed(size=20)
         type.padding = StringPad.NULLTERM
   
   .. autoattribute:: size 
   
      Read-write property to get and set the size of a string type. 
    

.. autoclass:: Factory
   :members:

.. autofunction:: to_numpy
   

Predefined types
================

We provide some predefined type instances for common types. 

.. autodata:: kUInt8
   :annotation: = HDF5 datatype for 8Bit unsigned integer
   
.. autodata:: kInt8
   :annotation: = HDF5 datatype for signed 8Bit integers
   
.. autodata:: kUInt16 
   :annotation: = HDF5 datatype for unsigned 16Bit integers
   
.. autodata:: kInt16
   :annotation: = HDF5 datatype for signed 16Bit integers
   
.. autodata:: kUInt32
   :annotation: = HDF5 datatype for unsigned 32Bit integers 
   
.. autodata:: kInt32
   :annotation: = HDF5 datatype for signed 32Bit integers
   
.. autodata:: kUInt64
   :annotation: = HDF5 datatype for unsigned 64Bit integers

.. autodata:: kInt64
   :annotation: = HDF5 datatype for singed 64Bit integers
   
.. autodata:: kFloat32
   :annotation: = HDf5 datatype for 32Bit floating point numbers 
   
.. autodata:: kFloat64
   :annotation: = HDF5 datatype for 64Bit floating point numbers 
   
.. autodata:: kFloat128
   :annotation: = HDF5 datatype for 128Bit floating point numbers 
   
.. autodata:: kVariableString
   :annotation: = HDF5 datatype for variable length strings  
