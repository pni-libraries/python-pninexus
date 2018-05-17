===================
Working with groups
===================

Groups are the fundamental containers in NeXus. Technically NeXus groups are
nothing else than ordinary HDF5 groups (:py:class:`pninexus.h5cpp.node.Group`)
with two additional features 

* in order to be NeXus compliant a the name of a groups link must adhere 
  to the NeXus naming convention 
* the group can represent a particular *base class* if it has a string 
  attribute `NX_class` attached to it and set to the name of the desired 
  *base class*. 

The task of creating a NeXus compliant group or *base class* can be split 
into two subtasks

* first check if the name of the new group obeys the NeXus naming rules
* attach a `NX_class` attribute to the new group and set it to the name 
  of the desired base class. 
  
In principle one could create a group simply by using the constructor of the 
:py:class:`h5cpp.node.Group` class and perform the two takes mentioned above 
manually. However to make life easier the :py:mod:`nexus` package provides 
a utility class :py:class:`pninexus.nexus.BaseClassFactory` which provides a single static 
method :py:meth:`create` which takes away the burden to write all the boiler 
plate code yourself. 

.. code-block:: python

   from pninexus.nexus import BaseClassFactory
   
   #open a file an obtain the root group 
   
   entry = nexus.BaseClassFactory.create(parent = root_group,
                                         name = "scan_00001",
                                         base_class = "NXentry")
                                         
In order to customize the group creation this static method can be provided 
with three additional named arguments 

+----------+-------------------------------------------------------+------------------------------+
| argument | type                                                  | description                  |
+==========+=======================================================+==============================+
| `lcpl`   | :py:class:`pninexus.h5cpp.property.LinkCreationList`  | link creation property list  |
+----------+-------------------------------------------------------+------------------------------+
| `gcpl`   | :py:class:`pninexus.h5cpp.property.GroupCreationList` | group creation property list |
+----------+-------------------------------------------------------+------------------------------+
| `gapl`   | :py:class:`pninexus.h5cpp.property.GroupAccessList`   | group access property list   |
+----------+-------------------------------------------------------+------------------------------+


