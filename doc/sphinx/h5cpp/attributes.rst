================================
:py:mod:`pni.io.h5cpp.attribute`
================================

.. automodule:: pninexus.h5cpp.attribute

:py:class:`AttributeManager`
============================

.. autoclass:: AttributeManager
   
   Every instance of :py:class:`Node` has an instance of :py:class:`AttributeManager` 
   attached to it which provides access to the attributes attached to the 
   particular node.
   
   .. autoattribute:: AttributeManager.size
   
      Read-only property returning the number of attributs of a node. 
      
      :return: number of attributes
      :rtype: integer
      
   .. automethod:: AttributeManager.create
   
      Creates a new attribute on the node to which the manager instance is 
      attached. 
      
      :param str name: the name of the new attribute
      :param pni.io.h5cpp.datatype.Datatype type: a datatype instance 
      :param pni.io.h5cpp.dataspace.Dataspace space: the dataspace for the attribute
      :param pni.io.h5cpp.property.AttributeCreationList: an optional attribute creation property list
      :return: a new instance of :py:class:`Attribute`
      :rtype: :py:class:`pni.io.h5cpp.attribute.Attribute`
      :raises RuntimeError: in case of an error
      
   .. automethod:: AttributeManager.remove
      
      Removes an attribute from a node. Raises :py:exc:`RuntimeError` if the 
      node does not exist. 
      
      :param object index: the index of the node to remove. This can either be 
                           a numerical index or the name of the attribute. 
      :raises RuntimeError: in case of a failure

   .. automethod:: AttributeManager.__len__
   
      The attribute manager supports the `__len__` protocol so one can use the 
      :py:func:`len` function to determine the number of attributes attached 
      to a node. 
      
      .. code-block:: python
      
         root = file.root()         
         print("Number of attributes: {}".format(len(root.attributes)))

   .. automethod:: AttributeManager.exists
   
      Takes the name of an attribute as its sole argument and returns :py:const:`True`
      if an attribute of that name exists, otherwise it returns :py:const:`False`. 
      
      :param str name: the name of an attribute
      :return: :py:const:`True` if the attribute exists, :py:const:`False` otherwise
      :rtype: :py:type:`bool`
  
   .. automethod:: AttributeManager.__getitem__
   
      Reads data from an attribute and allows slicing. 
      
      .. code-block:: python
      
         data = dataset.attributes["tensor"][1,2,2]
         
      A numpy array is returned. 
      
      :param slice index: the selection for the data to read
      :return: numpy array with the data
      :rtype: :py:class:`numpy.ndarray`
      
   
   
:py:class:`Attribute`
=====================
   
.. autoclass:: Attribute

   .. code-block:: python
   
      a = dataset.attribtues.create()
      a.write("m")
      
   Attributes can be obtained with 
   
   .. code-block:: python
   
      a = dataset.attributes["temperature"]
      print(a.read())
      
   
   .. autoattribute:: Attribute.dataspace
      
      read-only property returning the dataspace for the attribute 
      
   .. autoattribute:: Attribute.datatype
      
      read-only property returning the HDF5 datatype of the attribute 
      
   .. autoattribute:: Attribute.name
      
      read-only property returning the name of the attribute as an instance of 
      :py:class:`str`
      
   .. autoattribute:: Attribute.is_valid
      
      read-only property returning :py:const:`True` if the attribute is a valid 
      HDF5 object, and :py:const:`False` otherwise
              
   .. autoattribute:: Attribute.parent_link
      
      read-only property returning a link to the parent node of the attribute
      as an instance of :py:class:`pni.io.h5cpp.node.Link`.
                   
   .. automethod:: Attribute.close
     
      closes the current attribute 
      
   .. automethod:: Attribute.read 
   
      Reads attribute data and returns it as an instance of :py:class:`numpy.ndarray`. 
   
   .. automethod:: Attribute.write
   
      Takes an instance of a Python object and writes it to the attribute. 
      
