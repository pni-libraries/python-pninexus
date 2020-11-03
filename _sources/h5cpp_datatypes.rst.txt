==============
HDF5 datatypes
==============

.. automodule:: pninexus.h5cpp.datatype

Beneath HDF5s IO model there are two fundamental concepts 

* *datatypes* 
* and *dataspaces*. 

We will deal with the latter one in :ref:`hdf5-dataspaces-and-selections` and focus
in this section on datatypes. 

Datatypes describe the smallest unit of data written to an HDF5 file. This 
can be such simple things a primitive numbers like integer or floating point 
numbers, strings but also complex compound data types like heterogeneous 
records of a database table. Currently, this wrapper only supports three 
types of data types 

* *integers* via :py:class:`Integer`
* *floats* via :py:class:`Float`
* and *stings* via :py:class:`String`

which can be found in the :py:mod:`pninexus.h5cpp.datatype` package and are 
all decendents of the :py:class:`Datatype` class. This 
package currently not provides the full functionality of the corresponding 
C++ namespace :cpp:any:`hdf5::datatype` of *h5cpp* but enough to get started 
from within Python. 
Usually you do not want to create a datatype manually. Instead there is is a 
factory class :py:class:`Factory` which provides a method to 
create HDF5 datatypes from numpy dtypes or their string representation.
For conveniance there is a module global instance :py:const:`kFactory` available
to make life a bit easier. 

So in order to create an HDF5 data type for a 64Bit floating point type you 
could use something like this 

.. code-block:: python
   
   import numpy
   from pninexus.h5cpp import datatype

   type = datatype.kFactory.create(numpy.dtype("float64"))
   
There exists also an inverse function which converts an HDF5 datatype to an 
appropriate numpy type: :py:func:`to_numpy`. This function returns the 
string representation of the appropriate numpy data type  

.. code-block:: python

   import numpy
   from pninexus.h5cpp.datatype import to_numpy
   
   
   dtype = numpy.dtype(to_numpy(dataset.datatype()))
   
The :py:mod:`pninexus.h5cpp.datatype` package also provides a set of predefined 
datatypes

+--------------------------------------+-----------------------------+
| datatype constant                    | description                 |
+======================================+=============================+
| :py:const:`datatype.kUInt8`          | 8Bit unsigned integer type  |
+--------------------------------------+-----------------------------+
| :py:const:`datatype.kInt8`           | 8Bit singed integer type    |
+--------------------------------------+-----------------------------+
| :py:const:`datatype.kUInt16`         | 16Bit unsigned integer type |
+--------------------------------------+-----------------------------+
| :py:const:`datatype.kInt16`          | 16Bit signed integer type   |
+--------------------------------------+-----------------------------+
| :py:const:`datatype.kUInt32`         | 32Bit unsigned integer type |
+--------------------------------------+-----------------------------+
| :py:const:`datatype.kInt32`          | 32Bit signed integer type   |
+--------------------------------------+-----------------------------+
| :py:const:`datatype.kUInt64`         | 64Bit unsigned integer type |
+--------------------------------------+-----------------------------+
| :py:const:`datatype.kInt64`          | 64Bit signed integer type   |
+--------------------------------------+-----------------------------+
| :py:const:`datatype.kFloat32`        | 32Bit floating point type   |
+--------------------------------------+-----------------------------+
| :py:const:`datatype.kFloat64`        | 64Bit floating point type   |
+--------------------------------------+-----------------------------+
| :py:const:`datatype.kFloat128`       | 128Bit floating point type  |
+--------------------------------------+-----------------------------+
| :py:const:`datatype.kVariableString` | variable length string type |
+--------------------------------------+-----------------------------+
   
Datatype are only important when creating datasets or attributes. During IO 
operations the appropriate type for the data is obtained automatically by the 
IO functions. 
   
String types
============

String types are particularly difficult in HDF5. This is true for virtually 
every wrapper or programming interface currently available. Though *h5py* seems
to make things easy (at least for the user side) there are corner cases where 
even *h5py* fails to read string data. 
Similar to *h5py* we focus on the string storage scheme of numpy. When storing
strings in a numpy array they are always stored as fixed size strings with 
NULL padding. The size of the strings is determined by the longest string in 
the array. In terms of HDF5 that would be 

.. code-block:: python

   from pninexus.h5cpp.datatype import String
   from pninexus.h5cpp.datatype import StringPad
   from pninexus.h5cpp.datatype import CharacterEncoding

   datatype = String.fixed(str_len)
   datatype.padding  = StringPad.NULLPAD
   datatype.encoding = CharachterEncoding.UTF8
