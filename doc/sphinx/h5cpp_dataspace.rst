
.. _hdf5-dataspaces-and-selections: 

==============================
HDF5 dataspaces and selections
==============================

.. automodule:: pninexus.h5cpp.dataspace

While datatype declare what is stored the dataspace defines how it is stored. 
At the current state of the HDF5 library there are only two storage schemes
available 

* *scalar* which means that only a single value of a particular datatype is 
  stored 
* *simple* which is a simple n-dimensional regular array of data elements of a
  given type. 
  
Simple and scalar dataspaces are represented by the :py:class:`Scalar` and 
:py:class:`Simple` classes in the :py:mod:`pninexus.h5cpp.dataspace` package. 
Dataspaces are required when creating attributes and datasets (we will 
discuss this in more detail in :ref:`working with attributes` and :ref:`hdf5 datasets`). 

Scalar dataspaces are simple to create

.. code-block:: python

   from pninexus.h5cpp.dataspace import Scalar
   
   dataspace = Scalar()
   
as its constructor does not require any additional arguments. For simple 
dataspaces a bit more effort has to be made. 
There are three different configurations for a simple dataspace 

* a fixed size dataspace whose size (and thus the size of the dataset 
  constructed with it) cannot be changed once it has been created 
* an extensible dataspace of finite size 
* and an extensible dataspace of infinite size. 

The first case is fairly simple 

.. code-block:: python

   from pninexus.h5cpp.dataspace import Simple 
   
   dataspace = Simple((12,3))
   
which will create a 2-dimensional dataspace with 12 elemnts along the first 
and 3 elements along the second dimension. To cover the second sitatuation 
we have to pass a second argument with the maximum number of elements along 
each dimension

.. code-block:: python

   dataspace = Simple((12,3),(24,6))
   
The newly created dataspace has 12 elements along the first and 3 along the 
second dimension. However, it would be possible to extend it up to 24 and 6
dimensions along the two dimensions (we will see later in :ref:`hdf5 datasets` 
how this is done practically). 

The last case finally where we can extend a dataset indefinitely along one 
or more of its dimensions requires a special constant :py:const:`UNLIMITED` from
the :py:mod:`dataspace` package 

.. code-block:: python

   from pninexus.h5cpp.dataspace import Simple, UNLIMITED
   
   dataspace = Simple((0,10),(UNLIMITED,10))
   
This code snippet shows a typical use case for such a dataspace. We start 
with no elements along the first dimension but with the option to extend the 
dataspace indefinitely along this first dimension. As a result we are able 
to extend a dataset using such a dataspace indefinitely along this dimension
and thus append data as it drops in. This has the nice advantage that we do 
not have to know the number of recorded data points in advance. 


Selections
==========

A topic closely related to dataspaces are selections which is related to 
HDF5s partial IO feature. In many cases the data stored to disk is far 
larger than the memory available on the machine the data should be analyzed. 
Or, not the entire data is required but only a relatively small part of it. 
HDF5 allows to apply selections on a dataspace and subsequently read only the
selected data from a dataset. 

.. note::

   Another missconception about HDF5 is that a selection is applied to a dataset.
   This is wrong. The selection is applied to a dataspace which is then used to 
   describe the data stored on disk. 
   
HDF5 supports two kinds of selections

* *point selections* where an arbitrary set of data elements can be selected 
* and *hyperslab selections* where a regular pattern of data elements is 
  selected. 
  
Currently only the *hyperslab selection* is implemented in *h5cpp* and thus 
in this Python wrapper. The class in charge is :py:class:`Hyperslab`. 

A Hyperslab has 4 parameters 

* an *offset* determining the start of the selection in the dataspace
* a *block* array which determines how many elements are selected in each 
  individual block
* a *stride* giving the distance between the blocks
* and a *count* value which determines how many blocks to read. 



