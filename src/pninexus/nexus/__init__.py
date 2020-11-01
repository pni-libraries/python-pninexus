from __future__ import print_function
from .. import h5cpp
import numpy

from ._nexus import is_nexus_file
from ._nexus import create_file
from ._nexus import open_file

from ._nexus import BaseClassFactory
from ._nexus import FieldFactory

#
# import predicates
#
from ._nexus import NodePredicate
from ._nexus import IsBaseClass
from ._nexus import IsData
from ._nexus import IsDetector
from ._nexus import IsEntry
from ._nexus import IsInstrument
from ._nexus import IsSample
from ._nexus import IsSubentry
from ._nexus import IsTransformation
from ._nexus import search

from ._nexus import create_from_file
from ._nexus import create_from_string

#
# import path related functions and classes
#
from ._nexus import Path
from ._nexus import has_file_section
from ._nexus import has_attribute_section
from ._nexus import is_absolute
from ._nexus import is_unique
from ._nexus import split_path
from ._nexus import split_last
from ._nexus import join
from ._nexus import make_relative
from ._nexus import match
from ._nexus import get_path
from ._nexus import get_objects


def base_class_factory_create(parent, name, base_class,
                              lcpl=None, gcpl=None, gapl=None):
    """create a new base class

    :param Group parent: the parent group for the new base class
    :param str name: the name of the new base class
    :param str base_class: the class the new group belongs to
    :param h5cpp.property.LinkCreationList lcpl: optional reference to
                a link creation property list
    :param h5cpp.property.GroupCreationList gcpl: optional reference to
                a group creation property list
    :param h5cpp.property.GroupAccessList gapl: optional reference to
                a group access property list
    :retrun: new group instance properly setup as a NeXus base class
    :rtype: :py:class:`Group`
    :raises RuntimeError: in case of a failure
    """

    if lcpl is None:
        lcpl = h5cpp.property.LinkCreationList()

    if gcpl is None:
        gcpl = h5cpp.property.GroupCreationList()

    if gapl is None:
        gapl = h5cpp.property.GroupAccessList()

    if isinstance(name, str):
        name = h5cpp.Path(name)
    elif isinstance(name, h5cpp.Path):
        pass
    else:
        raise TypeError(
            "`name` must be either a h5cpp.Path or a str instance!")

    return BaseClassFactory.create_(parent, name, base_class, lcpl, gcpl, gapl)


BaseClassFactory.create = staticmethod(base_class_factory_create)


def field_factory_create(parent, name, dtype, shape=None, max_shape=None,
                         chunk=None, units=None,
                         lcpl=None, dcpl=None, dapl=None):
    """create a new nexus field

    Creates a nexus compliant field (dataset). The name of the field is checked
    against NeXus's naming policy and an exception is thrown if this is not the
    case. See the users manual about how to use this class.

    :param Group parent: the parent group
    :param str name: the name of the field
    :param numpy.dtye,str type: numpy datatype for the field
    :param tuple shape: the shape of the dataset
    :param tuple chunk: the chunk size for the dataset
    :param h5cpp.property.LinkCreationList lcpl: optional reference to
               a link creation property list
    :param h5cpp.property.DatasetCreationList dcpl: optional reference to
               a dataset creation property list
    :param h5cpp.property.DatasetAccessList dapl: optional reference to
               a dataset access property list
    :return: new dataset instance
    :rtype: :py:class:`Dataset`
    :raises RuntimeError: in case of a failure
    """

    if lcpl is None:
        lcpl = h5cpp.property.LinkCreationList()

    if dcpl is None:
        dcpl = h5cpp.property.DatasetCreationList()

    if dapl is None:
        dapl = h5cpp.property.DatasetAccessList()

    #
    # create the datatype
    #
    if isinstance(dtype, str):
        # if the datatype is given as a string we first have to convert it
        # to a numpy dtype instance
        dtype = numpy.dtype(dtype)

    datatype = h5cpp.datatype.Factory().create(dtype)

    #
    # create the dataspace
    #
    if shape is not None:
        # if no maximum shape is given we use the current shape
        if max_shape is None:
            max_shape = shape

        dataspace = h5cpp.dataspace.Simple(shape, max_shape)
    else:
        # if no shape is given we ignore also max_shape
        dataspace = h5cpp.dataspace.Scalar()

    #
    # setup for chunking
    #
    # print(dcpl.layout)
    if chunk is not None:
        dcpl.layout = h5cpp.property.DatasetLayout.CHUNKED
        dcpl.chunk = chunk
    else:
        pass
        # dcpl.layout = h5cpp.property.DatasetLayout.CONTIGUOUS

    #
    # finally we can create the dataset
    #
    dataset = FieldFactory.create_(
        parent, h5cpp.Path(name), datatype, dataspace, lcpl, dcpl, dapl)

    #
    # if units are given we have to attache them as an attribute
    #
    if units is not None:

        if not isinstance(units, str):
            raise TypeError("`units` must be an instance of `str`!")

        unit_attr = dataset.attributes.create(
            "units", h5cpp.datatype.kVariableString)
        unit_attr.write(units)

    return dataset


FieldFactory.create = staticmethod(field_factory_create)


__all__ = ["is_nexus_file", "create_file", "open_file", "BaseClassFactory",
           "FieldFactory", "NodePredicate", "IsBaseClass", "NodePredicate",
           "IsBaseClass", "IsData", "IsDetector", "IsEntry", "IsInstrument",
           "IsSample", "IsSubentry", "IsTransformation", "search",
           "create_from_file", "create_from_string", "Path",
           "has_file_section", "has_attribute_section",
           "is_absolute", "is_unique", "split_path", "split_last", "join",
           "make_relative", "match", "get_path", "get_objects"]
