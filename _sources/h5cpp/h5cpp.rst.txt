========================
:py:mod:`pninexus.h5cpp`
========================

.. automodule:: pninexus.h5cpp

.. autoclass:: Path
   :members:   
   
   This class represents an HDF5 path. A path in HDF5 is a list of link 
   names which should be followed starting from a particular node to access 
   another node. A path is *absolute* if the first node should be the root 
   group of a particular file. In its string representation a absolute path 
   can be identified by a leading ```/``. 
   
   There are two passiblities how to construct a path: 
   
   * either use the default constructor 
   
      .. code-block:: python
         
         path = Path()
         
   * or construct from a string representation
   
      .. code-block:: python
        
         path = Path("sensors/temperature/data")
   
   .. autoattribute:: absolute
   
      Read-write property either set a path to absolute or to check whether a 
      path is absolute. 
      
      .. code-block:: python 
      
         from pninexus import h5cpp
         
         path = h5cpp.Path("/run_001/sensors/temperature")
         print(path)
         #output: /run_0001/sensors/temperature
         
         print(path.absolute) #this should print True
         
         path.absolute = False
         print(path)
         #output: run_0001/sensors/temperature
   
   .. autoattribute:: name
   
      Read-only property returning the last element of a path (basically 
      the name of the last link in the chain)
      
      :rtype: str
      
   .. autoattribute:: parent
   
      Read-only property returning a path except for the last last element 
      
      .. code-block:: python
      
         path = h5cpp.Path("/run_0001/sensors/temperature")
         print(path.parent)
         #output: /run_0001/sensors 
   
   .. autoattribute:: size
   
      Read-only property returning the number of elements (link names) in 
      this path. 
      
      :rtype: int 
      
   .. automethod:: append(name)
   
      Append a an additional name to the end of the path
      
      :param str name: the link name to add
      
   .. automethod:: __add__(other)
   
      Paths can be concatenated using the + operator. 
      
      .. code-block:: python
      
         p1 = Path("/run_0001")
         p2 = Path("sensors/temperature/data")
         print(p1+p2)
         #output: /run_0001/sensors/temperature/data
         
   .. automethod:: __eq__(other)
   
      Paths can be checked for equality. Two paths are considered equal if 
      they store the same link names in the same order. 
      
   
.. autoclass:: IteratorConfig
   :members:
   
   Class providing configuration information for link, node and attribute 
   iteration. 
   
   .. autoattribute:: index 
   
      read-write property to set and retrieve the index to use for the 
      iteration. 
      
      :param IterationIndex index: index to use for iteration
      :rtype: IterationIndex
      
   .. autoattribute:: order
   
      read-write property to set and retrieve the iteration order
      
      :param IterationOrder order: order value
      :rtype: IterationOrder
      
   .. autoattribute:: link_access_list 
   
      read-write attribute to set and retrieve the link access property list 
      which should be used for iteration. 
      
      :param h5cpp.property.LinkAccessList lapl: link access property list
      :rtype: :py:class:`pninexus.h5cpp.property.LinkAccessList`
   
.. autoclass:: IterationOrder
   :members:
   
   Enumeration type determining in which order a particular iteration index 
   should be traversed. 
   
   .. autoattribute:: NATIVE
      :annotation: platform and/or library depending native order
   
   .. autoattribute:: DECREASING
      :annotation: traverse the index in decreasing order
      
   .. autoattribute:: INCREASING
      :annotation: traverse the index in increasing order
   
.. autoclass:: IterationIndex
   :members:
   
   Enumeration type determining which index to use for iteration over the 
   node tree.
   
   .. autoattribute:: NAME
      :annotation: use the link name of a node as an iteration index
   
   .. autoattribute:: CREATION_ORDER
      :annotation: use the order of creation as a node index
      
