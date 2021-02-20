Working with attributes
=======================

Fields and groups can have attributes attached which provide additional
metadata. In fact Nexus makes heavy use of attributes to store additional
information about a field or group. It is thus not recommended to use
attributes too excessively as one may runs into name conflicts with future Nexus
development.
Attributes can be accessed from fields and groups via their :py:attr:`attributes`
member. Attributes behave basically like fields with some restrictions

* they cannot grow
* one cannot apply compression to attributes.

Due to this restrictions one should not use attributes to store large amounts of
data.

To create attributes use the :py:meth:`create` method of the
:py:obj:`attributes` member

.. code-block:: python

    import pninexus.nexus as nx

    field = ....

    a = field.attributes.create("temperature",type="float32")

which would create a scalar attribute of name ``temperature`` and with a 32-bit
floating point type. Multidimensional attributes can be created with the
optional ``shape`` keyword argument

.. code-block:: python

    a = field.attributes.create("temperature",type="float32",shape=(4,4))

If an attribute already exists it can be overwritten using the ``overwrite``
keyword argument (:py:const:`False` by default).

.. code-block:: python

    a = field.attributes.create("temperature",type="float32",shape=(4,4))
    a = field.attributes.create("temperature",type="int32",shape=(3,5),
                                overwrite=True)

Attribute inquiry
-----------------

Like fields and groups attributes have a couple of properties which can be
queried to obtain metadata for a particular attribute instance


===================  =====================================================
property             description
===================  =====================================================
:py:attr:`shape`     a tuple with the number of elements along each
                     dimension of the attribute
:py:attr:`dtype`     the string representation of the attributes data type
:py:attr:`is_valid`  true if the attribute is a valid object
:py:attr:`name`      the name of the attribute (the key which can be used
                     to retrieve the attribute from its parent)
:py:attr:`value`     provides access to the attributes data
:py:attr:`path`      returns the path of the attribute
===================  =====================================================

The following code

.. literalinclude:: ../examples/attributes_properties.py

will produce

.. code-block:: bash

    HDF5_Version         : /@HDF5_Version       type=string     size=1
    NX_class             : /@NX_class           type=string     size=1
    file_name            : /@file_name          type=string     size=1
    file_time            : /@file_time          type=string     size=1
    file_update_time     : /@file_update_time   type=string     size=1



Attribute retrieval and iteration
---------------------------------

There are basically three ways to to access the attributes attached to a group
or a field

* by name via :py:meth:`__getitem__` (using `[]`)
* by the attributes index
* by an iterator.

The :py:attr:`attributes` attribute of groups and fields exposes an iterable
interface. The following example shows all three ways how to access the
attributes of a files root group

.. literalinclude:: ../examples/attribute_access.py

The output is


Reading and writing data from and to an attribute
-------------------------------------------------

Concerning IO operations attributes heave pretty much like fields as shown in
the next example

.. literalinclude:: ../examples/attribute_io.py

.. code-block:: bash

    r= [-1.  0.  0.]
    x= -1.0
    y= 0.0
    z= 0.0

There is not too much more to day about that. When reading and writing
multidimensional data numpy arrays must be used in any case (also for strings).
