==================================
:py:mod:`pninexus.h5cpp.dataspace`
==================================

.. automodule:: pninexus.h5cpp.dataspace
   :noindex:

:py:class:`Type`
================

.. autoclass:: Type

   Enumeration type used to determine the type of a dataspace. 

   .. autoattribute:: SCALAR
      :annotation: = Denotes a scalar dataspace
    
      
   .. autoattribute:: SIMPLE
      :annotation: = Denotes a simple dataspace
      
:py:class:`SelectionType`
=========================

.. autoclass:: SelectionType

   Enumeration type determining the type of selection. 
   
   .. autoattribute:: NONE 
      :annotation: = nothing is selected. 
      
   .. autoattribute:: ALL
      :annotation: = all elements in a dataspace are selected 
      
   .. autoattribute:: POINTS 
      :annotation: = the selection is a point selection
      
   .. autoattribute:: HYPERSLAB
      :annotation: = the selection is a hyperslab selection
      
:py:class:`SelectionOperation`
==============================

.. autoclass:: SelectionOperation

   Enumeration type determining how a particular selection is applied on a 
   dataspace (and on possible previous selections). 
   
   .. autoattribute:: SET
      :annotation: replace all previously applied with this selection
   .. autoattribute:: OR
      :annotation: use a logical or relation
   .. autoattribute:: AND
      :annotation: logical and relation
   .. autoattribute:: XOR
      :annotation: logical xor operation
   .. autoattribute:: NOTB
      :annotation: ???
   .. autoattribute:: NOTA
      :annotation: ???
   .. autoattribute:: APPEND
      :annotation: ???
   .. autoattribute:: PREPEND
      :annotation: ??? 
      

:py:class:`SelectionManager`
============================

.. autoclass:: SelectionManager
   
   :py:class:`SelectionManager` controls the application of selections on a 
   particular dataspace. 
   
   .. automethod:: all
   
      select all elements in the dataset 
   
   .. automethod:: none
   
      deselect all elements in the dataset 
   
   .. autoattribute:: size
   
      Read-only property returning the number of elements currently selected 
      in the dataspace. 
      
      :return: number of selected elements 
      :rtype: integer 
   
   .. autoattribute:: type
   
      Read only property returning the particular type of selections applied. 
      
      :return: type of applied selections
      :rtype: :py:class:`SelectionType`
      
   .. automethod:: __call__
   
      Apply a selection to a dataset. 
      
      .. code-block:: python
      
         hyperslab = Hyperslab()
         
         dataset.selection(SelectionOperation.SET,hyperslab)
      
      :param SelectionOperation operation: how the selection should be applied
      :param Selection selection: the selection to apply
      :raises RuntimError: in case of a failure


:py:class:`Dataspace`
=====================

.. autoclass:: Dataspace
   
   Base class of all dataspaces. 
   
   .. autoattribute:: Dataspace.is_valid
   
      Returns :py:const:`True` if the dataspace is a valid HDF5 object, 
      :py:const:`False` otherwise. 
      
      :return: :py:const:`True` if valid, :py:const:`False` otherwise
      :rtype: boolean
   
   .. autoattribute:: Dataspace.selection
   
      Read only property providing access to the :py:class:`SelectionManager`
      instance of the dataspace. 
   
   .. autoattribute:: Dataspace.size
   
      Read-only property returning the number of data elements in the current
      dataspace. This will always return the number of elements for the total 
      dataspace independent of whether a selection has been applied to that 
      dataspace or not. 
      
      :return: number of data elements
      :rtype: integer
   
   .. autoattribute:: type
   
      Read only attribute returning the type of the dataspace. 
      
      :return: dataspace type
      :rtype: pninexus.h5cpp.dataspace.Type


:py:class:`Simple`
==================

.. autoclass:: Simple
   
   Simple dataspace describing a regular array of data elements. In order to 
   construct an instance of :py:class:`Simple` use either 
   
   .. code-block:: python
   
      dataspace = Simple((10,20))
      
   Which will produce a dataspace where the current and maximum dimensions are 
   equal. Or, for extensible datasets one could use 
   
   .. code-block:: python
   
      space = Simple((0,1024),(pni.io.h5cpp.dataspace.UNLIMITED,1024))
      
   where the second tuple provides the maximum dimensions for the dataspace.
   
   .. autoattribute:: current_dimensions
   
      read-only property returning the current dimensions of the simple dataspace 
      
      :return: tuple with dimensions
      :rtype: tuple of integers
      
   .. autoattribute:: maximum_dimensions
   
      read-only attribute returning the current maximum dimensions of the 
      simple dataspace
      
      :return: tuple with dimensions
      :rtype: tuple of integers
   
   .. autoattribute:: rank
   
      read-only property returning the number of dimensions of the dataspace
   
   .. automethod:: dimensions
   
      Set the current and maximum dimensions of a dataspace 
      
      :param tuple/list current: tuple or list with the current dimensions
      :param tuple/list maximum: tuple or list with the maximum dimensions

:py:class:`Scalar`
==================

.. autoclass:: Scalar
  
   This class describes a dataspace for a single scalar value of a particular 
   element type. 
   
   .. code-block:: python
   
      dataspace = Scalar()
      
   It provides no additional methods or properties.
