=============================
HDF5 nodes, links, and groups
=============================

.. automodule:: pninexus.h5cpp.node

In HDF5 data is stored as a tree of nodes. 

.. figure:: _static/hdf5_basic_tree.svg
   :width: 50%
   :align: center

Currently two kinds of nodes are supported 

* *groups* (:py:class:`Group`)
* and *datasets* (:py:class:`Dataset`) 

which both are subclasses of :py:class:`Node`. The nodes are connected 
via *links* (:py:class:`Link`). Every node can have *attributes* attached to 
it (:py:class:`pninexus.h5cpp.attribute.Attribute`) which can be used 
to store metadata about a particular node. The attributes of a node 
can be accessed via the :py:attr:`Node.attributes` property. We will discuss 
attributes later (see :ref:`working with attributes` for more details).
Aside from attributes the most important property of :py:class:`Node` is the 
:py:attr:`link` property. It provides information about the link via which 
the node was accessed. 

In this section we will discuss *groups* and *links*. 


Links
=====
 


Groups
======
Groups are the containers
which can store links to other nodes (groups and datasets).

Groups are the basic containers for nodes in HDF5. Thus all related functions 
concerning groups can be found in the :py:mod:`pninexus.h5cpp.node` package. 
HDF5 groups are represented by instances of :py:class:`Group`. 


Accessing child nodes
=====================

Access to the child nodes of a particular group is provided via its 
:py:attr:`Group.nodes` 
property which returns an instance of :py:class:`NodeView`. 
To access a particular child one could use 

.. code-block:: python 

   group = ...
   
   if group.nodes.exists("data"):
      data = group.nodes["data"]
      
The :py:class:`NodeView` instance only provides access to the *immediate* 
children of a group. :py:meth:`NodeView.__getitem__` returns either an 
instance of :py:class:`Group` or :py:class:`Dataset`. 

The child nodes of a group instance can be accessed via its :py:attr:`Group.nodes`
attribute which exposes an instance of :py:class:`NodeView`. The latter is 
basically an  iterable over all *immediate* child nodes of a group. 

.. code-block:: python

   group = ...
   
   for node in group.nodes;
      print(node.link.path)
      
In case that on wants to iterate not only over the *immediate* child nodes 
but recursively over the entire hierarchy below a particular group (including
all its subgroups and their children), the 
:py:class:`NodeView` instance provides also the :py:attr:`NodeView.recursive`
property 

.. code-block:: python

   group = ....
   
   for node in group.nodes.recursive;
      print(node.link.path)


Accessing links
===============
