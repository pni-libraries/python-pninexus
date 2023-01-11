from __future__ import print_function
from pninexus.h5cpp._h5cpp import Path
from pninexus.h5cpp import property
from pninexus.h5cpp import dataspace
from pninexus.h5cpp import datatype
from pninexus.h5cpp.filter import ExternalFilters
import numpy
# from collections import OrderedDict

#
# import enumeration wrappers
#
from pninexus.h5cpp._node import Type
from pninexus.h5cpp._node import LinkType

#
# import node classes
#
from pninexus.h5cpp._node import Node
from pninexus.h5cpp._node import GroupView
from pninexus.h5cpp._node import NodeView
from pninexus.h5cpp._node import LinkView
from pninexus.h5cpp._node import Group
from pninexus.h5cpp._node import Dataset
from pninexus.h5cpp._node import LinkTarget
from pninexus.h5cpp._node import Link
from pninexus.h5cpp._node import RecursiveNodeIterator

#
# import node related functions
#
from pninexus.h5cpp._node import is_dataset
from pninexus.h5cpp._node import is_group
from pninexus.h5cpp._node import get_node

from pninexus.h5cpp._node import _copy
from pninexus.h5cpp._node import _link
from pninexus.h5cpp._node import _move
from pninexus.h5cpp._node import _remove

try:
    from pninexus.h5cpp._node import VirtualDataset
    VDSAvailable = True
except Exception:
    VDSAvailable = False


def copy(node, base, path=None, link_creation_list=property.LinkCreationList(),
         object_copy_list=property.ObjectCopyList()):
    """Copy an object within the HDF5 tree

    Copies an existing object `node` to a new location. The new location is
    determined by `base` and `path` argument. If `path` is not given a
    copy of the original object will be created with the same name under
    `base`. If `path` is given, it determines the new path and thus name
    of the copied object relative to the `base` object.

    The behavior of the copy operation can be controlled by a link and
    object copy property list which can be passed as optional arguments.

    :param Node node: the object to copy
    :param Group base: the base group for the new location
    :param pninexus.h5cpp.Path path: optional HDF5 path determining the final
                                     location of the copied object
    :param LinkCreationList link_creation_list: optional link creation
                property list
    :param ObjectCopyList object_copy_list: optional object copy property list
    :raise RuntimeError: in case of errors
    """

    if path is not None:
        _copy(node, base, path, object_copy_list, link_creation_list)
    else:
        _copy(node, base, object_copy_list, link_creation_list)


def move(node, base, path=None, link_creation_list=property.LinkCreationList(),
         link_access_list=property.LinkAccessList()):
    """Moving a node within the HDF5 tree

    Move an instance of :py:class:`Node` (dataset or group) to a new location.
    The new location is determined by the `base` and `path` argument.
    If `path` is not given `node` is moved below the `base` using the same
    link name as it has under its original group.
    If `path` is given the new location is determined by this path relative
    to the `base` group.

    Technically this function does not move any data but, like on a filesystem,
    only links are altered.

    :param Node node: the node (dataset or group) to move
    :param Group base: the base group where to move the node
    :param Path path: optional HDF5 path determining the new location of the
                      node to be moved relative to the `base` group
    :param LinkCreationList link_creation_list: optional link creation
                      property list
    :param LinkAccessList link_access_list: optional link access property list
    """

    if path is not None:
        _move(node, base, path, link_creation_list, link_access_list)
    else:
        _move(node, base, link_creation_list, link_access_list)


def remove(node=None, base=None, path=None,
           link_access_list=property.LinkAccessList()):
    """Remove a node from the HDF5 node tree

    This function can be used in two modes:

    * either the node to remove is referenced directly by `node`
    * or by `base` and `path`.

    .. attention::

        It is important that this function does not remove any data from the
        file. It only removes the link to the given node. An object is
        considered deleted if no link leading to this node exists. However,
        the file size will not change. In order to remove all the data
        associated with this node `h5repack` has to be used which will
        simply not copy nodes without links to the new file.


    :param Node node: the node to remove
    :param Group base: base group from which to search
    :param Path path: HDF5 path to the object to remove
    :param LinkAccessList link_access_list: optional link access property list
    :raises TypeError: if any of the arguments is not of appropriate type
    :raises RuntimeError: in case of any other error
    """

    if not isinstance(link_access_list, property.LinkAccessList):
        raise TypeError(
            "The 'link_access_list' must be an instance of a link access "
            "property list!")

    if node is not None:
        if not isinstance(node, Node):
            raise TypeError(
                "The 'node' argument must be an instance of `Node`!")

        _remove(node, link_access_list)

    elif base is not None and base is not None:

        if not isinstance(base, Group):
            raise TypeError("The 'base' argument must be a Group instance!")

        if not isinstance(path, Path):
            raise TypeError(
                "The 'path' argument must be an instance of an HDF5 path!")

        _remove(base, path, link_access_list)

    else:
        raise RuntimeError(
            "You have to provide either `node` argument or the `base` "
            "and `path` argument!")


def link(target,
         link_base,
         link_path,
         target_file=None,
         link_creation_list=property.LinkCreationList(),
         link_access_list=property.LinkAccessList()):
    """Create a new link

    Create a new soft link to a node referenced by `target` under `link_path`
    relative to `link_base`. If `target_file` is set an external link is
    provided an external link is created.

    .. attention::

        The target object does not need to exist at the time the link is
        created. Only when the new link should be dereferenced the target
        object has to exist.

    :param Node/Path target: the target for the new link
    :param Group link_base: the base for the new link
    :param Path link_path: the path to the new link relative to the `link_base`
    :param LinkCreationList link_creation_list: optional reference to
               a link creation property list
    :param LinkAccessList link_access_list: optional reference to a link
               access property list
    :raises TypeError: if any of the arguments does not match the required type
    :raises RuntimError: in the case of any other error
    """

    if not isinstance(link_creation_list, property.LinkCreationList):
        raise TypeError(
            "`link_creation_list` must be an instance of a link creation "
            "property list!")

    if not isinstance(link_access_list, property.LinkAccessList):
        raise TypeError(
            "`link_access_list` must be an instance of a link access "
            "property list!")

    if not isinstance(link_base, Group):
        raise TypeError("`link_base` must be an instance of `Gruop`!")

    if not isinstance(link_path, Path):
        raise TypeError("`link_path` must be an instance of an HDF5 path!")

    if target_file is not None:

        _link(target_file, target, link_base, link_path, link_creation_list,
              link_access_list)
    else:

        _link(target, link_base, link_path, link_creation_list,
              link_access_list)


def selection_to_shape(selection):
    """Convert a selection to a numpy array shape

    This utilty function converts an HDF5 selection to a tuple which can
    be used as a numpy array shape. This function performs some kind of index
    reduction: the resulting shape is the minimum shape required to store the
    data referenced by the selection. This means that all unnecessary
    dimensions with only a single element are removed.

    For instance
    [1,1,1,1] -> [1]
    [1,2,3,1] -> [2,3]

    """

    if isinstance(selection, dataspace.Hyperslab):
        shape = []
        size = 1
        for blocks, counts in zip(selection.block(), selection.count()):
            size *= blocks * counts
            shape.append(blocks * counts)

        if size == 1:
            #
            # it the total number of elements in the selection is 1
            # the shape is always (1,)
            # no matter how many dimension are in the selection.
            #
            return (1,)
        elif len(shape) > 1:
            shape = [s for s in shape if s != 1]

        return shape
    elif isinstance(selection, dataspace.Points):
        return [selection.points]
    else:
        raise TypeError(
            "Shape conversion currently only works for Hyperslabs or Points")


def dataset_write(self, data, selection=None):
    """ write data to a dataset

    Writes `data` to a dataset

    :param object data: Python object with data to write
    :param pninexus.h5cpp.dataspace.Selection selection: an optional selection
    :raises RuntimeError: in case of a failure
    """

    #
    # in case that the parameter passed is not an numpy array we
    # have to create one from it
    #
    if not isinstance(data, numpy.ndarray):
        data = numpy.array(data)

    #
    # if the data is a unicode numpy array we have to convert it to a
    # simple string array
    if data.dtype.kind == 'U':
        data = data.astype('S')

    #
    # determine memory datatype and dataspace
    # - if the file type is a variable length string we have to adjust the
    #   memory type accordingly
    memory_space = dataspace.Simple(data.shape)
    memory_type = datatype.kFactory.create(data.dtype)

    if isinstance(self.datatype, datatype.String):
        if self.datatype.is_variable_length:
            memory_type = datatype.String.variable()

    # if the data is bool numpy array we have to convert it to a
    # int array
    #
    if data.dtype == 'bool':
        data = data.astype("int8")

    #
    # get the file dataspace
    #
    file_space = self.dataspace

    if selection is not None:
        file_space.selection(dataspace.SelectionOperation.SET, selection)

    self._write(data, memory_type, memory_space, file_space)


def dataset_read(self, data=None, selection=None):
    """ read data from a dataset

    Reads `data` from a dataset

    :param object data: Python object with data to read
    :param pninexus.h5cpp.dataspace.Selection selection: an optional selection
    :returns: read data
    :raises RuntimeError: in case of a failure
    """

    memory_space = None
    memory_type = None
    file_space = self.dataspace

    if selection is not None:
        file_space.selection(dataspace.SelectionOperation.SET, selection)

    if data is not None:
        #
        # if data has been provided by the user we have to determine the
        # datatype and dataspace for the memory representation
        #
        if not isinstance(data, numpy.ndarray):
            raise TypeError(
                "Inplace reading is only supported for numpy arrays!")

        memory_space = dataspace.Simple(data.shape)
        memory_type = datatype.kFactory.create(data.dtype)

        if isinstance(self.datatype, datatype.String):
            if self.datatype.is_variable_length:
                memory_type = datatype.String.variable()

    else:
        #
        # if no data was provided by the user we can safely take the
        # dataspace and datatype from the dataset in the file
        #
        memory_type = self.datatype

        if selection is not None:
            shape = selection_to_shape(selection)
            memory_space = dataspace.Simple(shape)
        else:
            memory_space = file_space
            shape = (1,)
            if file_space.type == dataspace.Type.SIMPLE:
                shape = dataspace.Simple(file_space).current_dimensions

        #
        # create an empty numpy array to which we read the data
        #
        data = numpy.empty(shape, dtype=datatype.to_numpy(memory_type))

    data = self._read(data, memory_type, memory_space, file_space)

    if data.dtype.kind == 'S':
        try:
            data = data.astype('U')
        except Exception:
            print(data)

    return data


def dataset_filters(self):
    """ read filters from a dataset

    Reads filters from a dataset

    :returns: a list of filters

    :raises RuntimeError: in case of a failure
    """

    efilters = ExternalFilters()
    efilters.fill(self.creation_list)
    return efilters


Dataset.write = dataset_write
Dataset.read = dataset_read
Dataset.filters = dataset_filters


__all__ = ["Type", "LinkType", "Node", "GroupView", "NodeView", "LinkView",
           "Group", "Dataset", "LinkTarget", "Link", "RecursiveNodeIterator",
           "is_dataset", "is_group",
           "get_node", "copy", "move", "remove", "link", "selection_to_shape"]

if VDSAvailable:
    __all__.extend(["VirtualDataset"])
