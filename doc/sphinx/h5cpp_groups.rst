=============================
HDF5 nodes, links, and groups
=============================

.. automodule:: pninexus.h5cpp.node

In HDF5 data is stored as a tree of nodes connected via links 

.. figure:: _static/hdf5_basic_tree.svg
   :width: 40%
   :align: center
   
The root of the tree is the root group which is itself a *group node*.
Currently two kinds of nodes are supported 

* *groups* (:py:class:`Group`)
* and *datasets* (:py:class:`Dataset`) 

which both are subclasses of :py:class:`Node`. The nodes are connected 
via *links* (:py:class:`Link`). There is a common misconception about HDF5 
that nodes (datasets and groups) have names. It is not the node which 
has a name but rather the link which connects a particular node with its 
parent node. It is possible that several links with different name point to 
the same node within a file. Thus the concept of a name for a node 
does not make too much sense. What makes sense instead is the concept of a 
*path* used to access a particular node (see below). 

Every node can have *attributes* attached to it 
(:py:class:`pninexus.h5cpp.attribute.Attribute`) which can be used to store 
meta-data about a particular node. The attributes of a node can be accessed via 
the :py:attr:`Node.attributes` property. 
We will discuss attributes later (see :ref:`working with attributes` for more 
details). Aside from attributes the most important property of 
:py:class:`Node` is the :py:attr:`link` property. It provides information about 
the link via which the node was accessed. 

In this section we will discuss *groups* and *links*. 

Links and paths
===============
 
Links can be considered the glue which sticks the nodes within an HDF5 
tree together. A link connects a particular node (dataset or group) with its 
parent group. Every link has a name which must be unique for all  links 
below the same parent group.

A concept closely related to links is a *path* (:py:class:`pninexus.h5cpp.Path`) 
which is technically nothing else than a list of link names used to access 
a particular object. *Paths* have a string representation which resembles 
that of a Unix file-system path. So for instance the path  
```/sensors/temperature``` would be the list of link names to access a 
dataset (most probably storing temperature data) in the HDF5 tree depicted 
above.  The :py:class:`pninexus.h5cpp.Path` class is used throughout the 
code to represent such a path. See the API documentation for more information 
about how to use this class. 

Links are usually not created directly by a user but rather returned by several
methods and class properties. The relevant property for this section is the 
:py:attr:`Node.link` property of the :py:class:`Node` class. This read 
only property returns an instance of :py:class:`Link` describing the link
used to access a particular node. 
In HDF5 links come in three flavors  

* *hard links* which are created when a new node is created
* *soft links* which can be used to reference already existing nodes within 
  the same file (they are thus similar to symbolic links on a UNIX file system)
* *external links* which allow referencing nodes in a different HDF5 file.  

The :py:class:`Link` class provides several useful properties and methods 
to query the status of a link. We will restrict ourselves here to the most 
important ones. Consult the API documentation for :py:class:`Link` for 
more detailed information. 

In order to determine what kind of link we are dealing with we can query the 
:py:meth:`type` method of :py:class:`Link` which returns an enumeration 
of :py:class:`LinkType`. 

.. code-block:: python

   link = ... #we somehow get a reference to a link 
   
   if link.type() == LinkType.HARD:
      print("we got a hard link")
   elif link.type() == LinkType.SOFT:
      print("we got a soft link")
   elif link.type() == LinkType.EXTERNAL:
      print("we got an external link")


Since in HDF5 *soft-* and *external-links* can be created even if the target 
the link refers to does not exist (yet) we have to know whether or not a link 
can be resolved. This is where the :py:attr:`is_resolvable` property of 
:py:class:`Link` comes in handy 

.. code-block:: python

   link = ...
   
   if not link.is_resolvable:
      raise ValueError("link cannot be resolved")  
 
Once we know that that a link is resolvable we can use the :py:attr:`node` 
property to retrieve the node referenced by the link 

.. code-block:: python

   link == ...
   
   if not link.is_resolvable:
      raise ValueError("link cannot be resolved")
      
   node = link.node 
   
which returns either an instance of :py:class:`Group` or :py:class:`Dataset`. 
If a link is not resolvable a good address to ask why is to use the 
:py:class:`target` method (see API docs for details) which allows you to 
identify the node the link would refer to. 

Groups
======

Groups are containers for links which refer to the *child nodes*  of a group. 
The :py:class:`Group` class provides an interface to work with groups 
and which will be discussed in more detail in this section. 

Creating groups
---------------

Groups are created by calling their constructor: 

.. code-block:: python

   from pninexus import h5cpp
   from pninexus.h5cpp.node import Group
   from pninexus.h5cpp import Path
   
   h5file = h5cpp.file.create(...)
   root   = h5file.root()
   
   run = Group(root,"run_0001") 

There are two notable things taking place in this example. 

1. the entry point to the HDF5 node hierarchy is the root group which can be 
   obtained from the :py:meth:`root` method of the file class. It returns an 
   instance of :py:class:`Group` as any other group would be. Unlike for many 
   HDF5 wrappers and the C-API itself the file object does not have any 
   group semantics. 
2. the constructor of :py:class:`Group` takes two parameters 
    
   * the first one is a reference to the parent group of the new group
   * and the second is path to the new group relative to the base.
   
If the path to the new group comprises more than one element all the 
intermediate groups must exist. If you want to create also these groups 
collectively you have to provide a special link creation property list 

.. code-block:: python

   from pninexus.h5cpp.property import LinkCreationList
   
   lcpl = LinkCreationList()
   lcpl.intermediate_group_creation = True
   temperature = Group(root,"run_0001/sensors/temperature",lcpl=lcpl)
     
There are three distinct things you can do with an instance of :py:class:`Group`

1. you can use it as a parent group for new nodes (datasets and groups)
2. you can iterate over its child nodes
3. you can iterate over the links attached to it.  

Accessing child nodes
---------------------

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

In order to iterate over the child nodes of a group one has two choices. 
For iteration over the *immediate* child nodes only, the :py:class:`NodeView`
instance provides a Python iterator interface 

.. code-block:: python

   group = ...
   
   for node in group.nodes;
      print(node.link.path)

The :py:attr:`path` property of the :py:class:`Link` class returns the 
HDF5 path of the link (and hence, in this case, to the node it dereferences). 
Alternatively, if you want to iterate recursively over all the nodes below 
a group including that of its subgroups, the :py:class:`NodeView` class 
provides a recursive node iterator via its :py:attr:`recursive` property

.. code-block:: python

   group = ....
   
   for node in group.nodes.recursive;
      print(node.link.path)
      
.. attention::

   Though this iterator approach is quite simple it has a significant downside: 
   it assumes that all the links accessed via the iteration can be resolved. 
   If a link does not exist and cannot be resolved to an object a 
   :py:exc:`RuntimeError` exception will be thrown. 
   So in the case of a file whose structure is mainly unknown rather follow 
   the approach in  the next subsection and iterate over links rather than 
   over nodes. 

Accessing links
---------------

Similar to the :py:class:`NodeView` interface there is also a :py:class:`LinkView` 
interface attached to :py:class:`Group` which can be accessed via its 
:py:attr:`links` property. 

To access a particular link attached to a group you could use 

.. code-block:: python

   group = ...
   
   if group.links.exists("data"):
      link = group.links["data"]
      
      if not link.is_resolvable:
         raise RuntimeError("Unresolvable link: "+link.path)
         
      data = link.node #get the object referenced by the link 
         
Like for :py:class:`NodeView`, :py:class:`LinkView` provides an iterator 
interface over the link *immediately* attached to a group while  
:py:attr:`LinkView.recursive` returns a recursive link iterator. 

Utility functions
=================

There are several utility functions available for working with groups and links. 
Two of them shall be presented here. 

Retrieving nodes by path
------------------------

The :py:class:`NodeView` interface allows only access by name to *immediate* child
nodes of a group. The idea was that a group should have a similar semantic 
as a Python *dictionary*. However, if you want to access nodes via a path 
you may consider using the :py:func:`get_node` function 

.. code-block:: python

   from pninexus.h5cpp.node import get_node
   from pninexus.h5cpp import Path
   
   temperature = get_node(root,Path("run_0001/sensors/temperature"))

Creating links
--------------

Another important topic is to create links. This can be done with the 
:py:func:`link` function. This function allows you to create two types 
of links: soft links and external links. For a detailed description 
about the arguments of :py:func:`link` see the API documentation. We will 
show here only the function in action. 

The easiest way to create a soft link to an existing node would be this  

.. code-block:: python

   from pninexus.h5cpp import Path
   from pninexus.h5cpp.node import link

   group = ...
   link(group,root,Path("link_to_group"))

Where the first argument is the target for the link, in this case a group. 
The second is the base relative to which the new link should be created, and 
the third argument is the path to the new link. 
When using the above syntax the target node obviously must already exist at the 
time the link is created. 
However, there is also an alternative approach for situations where you want to 
create the link in advance of the target. In this case, the above example 
would read somehow like 

.. code-block:: python

   link(Path("/absolute/path/to/target"),base,Path("link_to_group"))
   
Here the first argument is an HDF5 path rather than a node instance. Since 
the existence of the target path is not checked when the link is created, 
you can safely create a link in advance of its target. However, when you 
try to access the node referenced by the new link the target must exist 
for the operation to succeed. 

If you want to create a link to a node in a different file you could use

.. code-block:: python

   link(target = Path("/path/to/node")
        link_base   = base_group,
        link_path   = Path("link_name"),
        target_file = "external_data.h5")
        
The important argument here is the `target_file` argument. When provided 
:py:func:`link` creates an external link assuming that `target` refers 
to a node in the file determined by `target_file`.



