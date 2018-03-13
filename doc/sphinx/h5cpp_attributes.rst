
.. _working with attributes: 

===============
HDF5 attributes
===============

.. automodule:: pninexus.h5cpp.attribute

Nodes (datasets and groups) have attributes attached to them which can 
be used to store meta-data or other related data close to the group or 
dataset they are associated with. 

.. note::

   Though attributes in many ways behave like datasets, unlike those, they 
   are not intended to store large data!
   
The attributes of a node can be accessed and managed via the 
:py:class:`AttributeManager` interface accessible from every node via its 
:py:attr:`attributes` property. 
   
Creating attributes
===================

Creating attributes follows more or less the same scheme than creating 
datasets. You first have to create and HDF5 datataype and dataspace which 
is then passed to the :py:func:`create` function of the :py:class:`AttributeManager` 
instance of a node 

.. code-block:: python

   from pninexus.h5cpp.datatype import kVariableString
   from pninexus.h5cpp.dataspace import Scalar

   dataset = ...
   
   attr = dataset.attributes.create("unit",type=kVariableString,space=Scalar())

Though this seems to be a bit verbose and there is definitely more typing
work to do than in the case of other wrappers, however, on the other side 
this gives you full control of how an attribute is created (and *full control* 
is the intention of this wrapper). 
You can also pass an instance of :py:class:`pninexus.h5cpp.property.AttributeCreationList` 
as a fourth argument which would give you the same level of control over 
attribute creation as the C-API. 

Accessing and managing attributes
=================================

Attributes can be accessed either by their numerical index or by their name 
using the :py:meth:`__getitem__` method of the :py:class:`AttributeManager` 
instance of a node. 

.. code-block:: python

   attr = dataset.attributes["unit"]
   
   #or 
   
   attr = dataset.attributes[0] #get the first attribute


To check whether an existing attribute exists us the :py:meth:`exists` method
provided by :py:class:`AttributeManager`. 

.. code-block:: python

   if dataset.attributes.exists("unit"):
      attr = dataset.attributes["unit"]

:py:class:`AttributeManager` provides also an interator interface for attributes. 
So in order to iterate over all attributes attached to a node we could do 

.. code-block:: python

   for attr in dataset.attributes:
      print(attr.name)
      
Existing attributes can be removed from a node using the :py:meth:`remove` method
of :py:class:`AttributeManager` which takes either a numerical index or the 
name of the attribute to delete. Finally the :py:meth:`size` method returns 
the number of attributes attached to a node. 

.. code-block:: python

   dataset = ...
   
   print("This node has {} attributes".format(dataset.attributes.size))

Reading and writing data
========================

Reading data from and writing data to an attribute is quite simple. The 
:py:class:`Attribute` class provides two methods for this :py:meth:`read` 
and :py:meth:`write`. 

.. code-block:: python

   attr = dataset.attributes["unit"]
   attr.write("degree")
   print(attr.read())

Like for datasets the :py:meth:`read` method returns an instance of 
:py:class:`numpy.ndarray` and the :py:class:`write` function internally 
converts every argument to a numpy array before writing (unless it is not 
already a numpy array). 