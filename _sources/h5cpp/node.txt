=============================
:py:mod:`pninexus.h5cpp.node`
=============================

.. automodule:: pninexus.h5cpp.node
   :noindex:

Enumerations
============

.. autoclass:: Type

   Enumeration identifying the type of node.

   .. autoattribute:: DATASET
      :annotation: the node is a dataset

   .. autoattribute:: GROUP
      :annotation: the node is a group

   .. autoattribute:: DATATYPE
      :annotation: the node is a committed datatype


.. autoclass:: LinkType

   Enumeration determining the type of a particular link.

   .. autoattribute:: ERROR
      :annotation: invalid link

   .. autoattribute:: HARD
      :annotation: a hard link

   .. autoattribute:: EXTERNAL
      :annotation: an external link to a node in a different file

   .. autoattribute:: SOFT
      :annotation: soft (symbolic) link to a node within the same file


Classes
=======

.. autosummary::

   Node
   Group
   Dataset

.. autoclass:: Node
   :members:

   Base class for all node types.

   hello world this is some stupid text

   .. autoattribute:: attributes

      Read-only property providing access to the
      :py:class:`pninexus.h5cpp.attribute.AttributeManager` instance associated
      with the node

   .. autoattribute:: is_valid

      Read-only property returning true if the node is a valid HDF5 object.
      False otherweise.

      :return: :py:const:`True` if node is valid object, :py:const:`False` otherwise
      :rtype: Boolean

   .. autoattribute:: link

      Read-only attribute providing access to :py:class:`Link` used to
      access this node in the first place.

   .. autoattribute:: type

      Read only attribute returning the :py:class:`Type` of the particular node.

.. autoclass:: Group
   :members:

   HDF5 group class.

   :param Group parent: the parent group below which to construct the new group
   :param str name: the name for the new group
   :param pninexus.h5cpp.property.LinkCreationList lcpl: optional link creation property list
   :param pninexus.h5cpp.property.GroupCreationList gcpl: optional group creation property list
   :param pninexus.h5cpp.property.GroupAccessList gapl: optional group access property list

   .. autoattribute:: links

      Read only property returning the link view instance associated with this
      group instance.

      :rtype: :py:class:`LinkView`

   .. autoattribute:: nodes

      Read only property returning the node view instance associated with this
      group.

      :rtype: :py:class:`NodeView`

   .. automethod:: close

      Closes the group. After a call to this method the :py:attr:`is_valid`
      property returns :py:const:`False`.

      .. code-block:: python

         group = Group(...)
         group.is_valid   #returns True

         group.close()
         group.is_valid   #returns False


.. autoclass:: Dataset
   :members:

   Datasets are the primary data storing nodes in HDF5.

   The constructor for a dataset takes the following arguments:

   :param Group parent: the parent group for the new dataset
   :param str name: the name for the new dataset
   :param pninexus.h5cpp.datatype.Datatype type: the datatype for the new dataset
   :param pninexus.h5cpp.dataspace.Dataspace space: the dataspace for the dataset
   :param pninexus.h5cpp.property.LinkCreationList lcpl: an optional link creation property list
   :param pninexus.h5cpp.property.DatasetCreationList dcpl: an optional dataset creation property list
   :param pninexus.h5cpp.property.DatasetAccessList dapl: an optional dataset access property list

   .. autoattribute:: access_list

      Read-only attribute returning the access property list used when opening
      the dataset.

      :rtype: :py:class:`pninexus.h5cpp.property.DatasetAccessList`

   .. autoattribute:: creation_list

      Read-only attribute returning the creation property list used during
      dataset creation.

      :rtype: :py:class:`pninexus.h5cpp.property.DatasetCreationList`

   .. autoattribute:: dataspace

      Read only attribute returning the associated with this dataset

      :rtype: :py:class:`pninexus.h5cpp.dataspace.Dataspace`

   .. autoattribute:: datatype

      Read only attribute returning the datatype associated with the dataset.

      :rtype: :py:class:`pninexus.h5cpp.datatype.Datatype`

   .. automethod:: close

      Closes the dataset. After calling this function :py:attr:`is_valid`
      should return :py:const:`False`.

   .. automethod:: refresh

      When built with HDF5 > 1.10 this method can be used in SWMR mode to
      obtain updates to the dataset from the writer.

   .. automethod:: extent(new_shape)

      Set the extent (number of elements along each dimension) for a dataset.
      `new_shape` is a tuple or list with the new number of elements.
      In order for this method to succeed

      * the number of dimensions (or *rank*) must remain the same
      * the dataset must be constructed with a chunked storage layout

      :param tuple new_shape: new number of elements
      :raises RuntimeError: in case of a failure

   .. automethod:: extent(dim_index,dim_delta)
      :noindex:

      Change the number of elements of a dataset along a particular dimensions
      by a delta.

      .. code-block:: python

         from pninexus.h5cpp.dataspace import Simple,UNLIMITED
         from pninexus.h5cpp.node import Dataset

         current_space = Simple((10,10),(UNLIMITED,10))
         dataset = Dataset(...,current_space)

         #do something with the dataset here


         dataset.extent(0,10)
         current_space = dataset.dataspace
         print(current_space.current_dimensions)
         #output (15,10)


      :param int dim_index: index of the dimension for which to reset the extent
      :param int dim_delta: change in number of dimensions
      :raises RuntimeError: in case of a failure

   .. automethod:: write
      :noindex:

      Write data to the dataset. The `data` argument is a reference to the
      Python object storing the data to write. Internally this object will be
      converted to a numpy array which is then written to disk. If `data`
      is already an instance of :py:class:`numpy.ndarray` this conversion
      is omitted.

      In addition to the data an optinal reference to an HDF5 selection
      :py:class:`pninexus.h5cpp.dataspace.Selection` can be passed to the
      method determining where in the dataset the data will be written.
      If no selection is passed the data will be assumed to fit into the entire
      dataset.

      :param object write: reference to the object to write
      :param pninexus.h5cpp.dataspace.Selection selection: optional selection for the dataset
      :raises RuntimeError: in case of a failure

   .. automethod:: read
      :noindex:

      Reading data from a dataset. If called with no argument this method
      reads all data from the dataset and returns it as an instance of
      :py:class:`numpy.ndarray`. With the `selection` parameter an HDF5
      selection can be passed to determine what data from the dataset to read.

      The optional `data` argument must be a reference to an instance of a
      numpy array (:py:class:`numpy.ndarray`). If `data` is not
      :py:const:`None` the method tries to read the data *inplace* to this
      numpy array thus avoiding memory additional memory allocation.
      In order for this to succeed the numpy array has to satisfy the following
      criteria

      * the datatype of the datatype must be convertible to the type of the
        numpy array
      * the number of elements of the numpy array must match those of the
        dataset or the selection if provided.

      :param numpy.ndarray data: optional data reference for inplace reading
      :param pninexus.h5cpp.dataspace.Selection: selection determining what to read
      :raises RuntimeError: in case of a failure
      :raises TypeError: if `data` is not an instance of :py:class:`numpy.ndarray`

.. autoclass:: LinkView
   :members:

   Iterable interface to the links directly attached to a group. The link view
   for a group can be accessed via the :py:attr:`links` attribute of a
   :py:class:`Group` instance.

   Iteration over the links  immediately attached to a group:

   .. code-block:: python

      group = ...

      for link in group.links:
         print(link.path)


   Recursive iteration over all links available below a group and its
   sub-groups

   .. code-block:: python

      group = ...

      for link in group.links.recursive:
         print(link.path)

   .. autoattribute:: recursive

      Read-only property returning a recursive link iterator

   .. automethod:: exists(link_name,lapl=pninexus.h5cpp.property.LinkAccessList())

      Returns :py:const:`True` if a link of name `link_name` exists below
      the associated group, otherwise :py:const:`False` is returned.

      :param str link_name: name of the link to check for
      :param pninexus.h5cpp.property.LinkAccessList lapl: optional link access property list
      :return: :py:const:`True` if link exists, :py:const:`False` otherwise
      :rtype: boolean

.. autoclass:: NodeView
   :members:

   Iterable interface to the nodes below an instance of :py:class:`Group`.
   The instance of :py:class:`NodeView` associated with a particular group
   can be accessed via the :py:attr:`nodes` property of the latter.

   .. code-block:: python

      group = ...

      for node in group.nodes:
         print(node.link.path)

   like for links there is also a recursive iterator available which returns
   also the children of all sub-groups of a group.

   .. code-block:: python

      group = ...

      for node in group.nodes.recursive:
         print(node.link.path)

   .. autoattribute:: recursive

      Read only property returning a recursive node iterator.

   .. automethod:: exists(node_name,lapl=pninexus.h5cpp.property.LinkAccessList())

      Returns true if a node of name `node_name` exists below a group.
      Otherwise :py:const:`False` is returned.
      This check involves two steps under the hood

      1. check if a link of name `node_name` exists
      2. check if the link can be resolved

      only if both checks succeed :py:mod:`True` is returned.

      :param str node_name: the name of the node to check for
      :param pninexus.h5cpp.property.LinkAccessList lapl: optional link access property list
      :return: :py:const:`True` if the node exists, :py:const:`False` otherwise
      :rtype: boolean


   .. automethod:: __getitem__(key)

      Allows access to the *immediate* child nodes of a group.

      .. code-block:: python

         data = group.nodes["data"]

      :param str key: name of the child node's link
      :return: group or dataset instance
      :rtype: :py:class:`Dataset` or :py:class:`Group`
      :raises RuntimeError: in case of a failure

.. autoclass:: Link
   :members:

   Class representing a link. You cannot construct an instance of
   :py:class:`Link` from within Python, but you can obtain them from several
   methods and properties of the node classes.

   .. autoattribute:: exists

      Read-only property returning :py:const:`True` if the link exists,
      :py:const:`False` otherwise.

      :rtype: boolean

      .. autoattribute:: file

      Read-only attribute returning the path to the file within which the link
      exists.

      :rtype: str

   .. autoattribute:: is_resolvable

      Read-only attribute returning :py:const:`True` if the link can be
      resolved, :py:const:`False` otherwise.

      :rtype: boolean

   .. autoattribute:: node

      Read-only property returning the node referenced by this link. This will
      only work if the link itself is resolvable.

      .. code-block:: python

         link = ....

         if link.is_resolvable:
            node = link.node

      :rtype: Group, Dataset
      :raises RuntimeError: if the link cannot be resolved

   .. autoattribute:: parent

      Read only property returning the parent group of the link

      :rtype: Group

   .. autoattribute:: path

      Return the path for the link.

      :rtype: :py:class:`pninexus.h5cpp.Path`

   .. automethod:: type(lapl = pninexus.h5cpp.property.LinkAccessList())

      Return the type.

      :param pninexus.h5cpp.property.LinkAccessList lapl: optional link access proerty list
      :return: link type enumeration
      :rtype: LinkType

   .. automethod:: target(lapl = pninexus.h5cpp.property.LinkAccessList())

      Return the link target

      :param pninexus.h5cpp.property.LinkAccessList lapl: optional link access property list
      :return: the target of the link
      :rtype: LinkTarget

.. autoclass:: LinkTarget
   :members:

   :py:class:`LinkTarget` describes the target for a link. It could be used
   to access this target without dereferencing the link. It is thus useful
   in the case of links which cannot be dereferenced. Using the target
   one could check whether the file or the object in it are missing or both.

   .. autoattribute:: file_path

      Read only property with the path to the file where the link target is
      expected to reside in. In the case of soft links this is empty.

      :rtype: str

   .. autoattribute:: object_path

      Read only property returning the node path to the target.

      :rtype: :py:class:`pninexus.h5cpp.Path`




Node related functions
======================

.. autofunction:: is_dataset(node)

   Returns :py:const:`True` if the node instance represents a dataset,
   :py:const:`False` otherwise.

   :param Node node: reference to a node to check
   :return: :py:const:`True` if `node` is a dataset, :py:const:`False` otherwise
   :rtype: boolean

.. autofunction:: is_group(node)

   Returns :py:const:`True` if the `Node` instance represents a group,
   :py:const:`False` otherwise.

   :param Node node: reference to the node to check
   :return: :py:const:`True` if `node` is a group, :py:const:`False` otherwise
   :rtype: boolean

.. autofunction:: get_node(base,path,lapl=pninexus.h5cpp.property.LinkAccessList())

   Returns a determined by `base` and `path`. If `path` is relative it is
   considered relative to `base`. If `path` is absolute `base` is only used
   to determine the file and root group where to look for the node.

   :param Group base: the base group from which to start the search
   :param Path path: absolute or relative path to the node to obtain
   :param pninexus.h5cpp.property.LinkAccessList lapl: optional link access property list
   :return: new node instance (dataset or group)
   :rtype: :py:class:`Group` or :py:class:`Dataset`
   :raises RuntimeError: in case of a failure

.. autofunction:: copy

.. autofunction:: move

.. autofunction:: remove

.. autofunction:: link
