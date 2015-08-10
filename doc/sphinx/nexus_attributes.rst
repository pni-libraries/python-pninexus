Working with attributes
=======================

Fields and groups can have attributes attached which provide additional
metadata. In fact Nexus makes heavy use of attributes to store additional 
information about a field or group. It is thus not recommended to use 
attributes too excessively as one may runs into name conflicts with future Nexus
development. 
Attributes can be accessed for fields  and groups via the ``attributes``
member of the field and group classes. Attributes behave basically like fields
with some restrictions

* they cannot grow
* one cannot apply compression to attributes.

Due to this restrictions one should not use attributes to store large amounts of
data. 

To create attributes use the :py:meth:`create` method of the
:py:obj:`attributes` member

.. code-block:: python
    
    import pni.io.nx.h5 as nx

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
===================  =====================================================
