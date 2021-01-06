#
# Import enumerations
#
from pninexus.h5cpp._datatype import Class
from pninexus.h5cpp._datatype import Order
from pninexus.h5cpp._datatype import Sign
from pninexus.h5cpp._datatype import Norm
from pninexus.h5cpp._datatype import Pad
from pninexus.h5cpp._datatype import StringPad
from pninexus.h5cpp._datatype import Direction
from pninexus.h5cpp._datatype import CharacterEncoding

#
# Import classes
#
from pninexus.h5cpp._datatype import Datatype
from pninexus.h5cpp._datatype import Float
from pninexus.h5cpp._datatype import Integer
from pninexus.h5cpp._datatype import Compound
from pninexus.h5cpp._datatype import String
from pninexus.h5cpp._datatype import Enum

#
# Import predefined constant types
#
from pninexus.h5cpp._datatype import kUInt8
from pninexus.h5cpp._datatype import kInt8
from pninexus.h5cpp._datatype import kUInt16
from pninexus.h5cpp._datatype import kInt16
from pninexus.h5cpp._datatype import kUInt32
from pninexus.h5cpp._datatype import kInt32
from pninexus.h5cpp._datatype import kUInt64
from pninexus.h5cpp._datatype import kInt64
from pninexus.h5cpp._datatype import kFloat16
from pninexus.h5cpp._datatype import kFloat32
from pninexus.h5cpp._datatype import kFloat64
from pninexus.h5cpp._datatype import kFloat128
from pninexus.h5cpp._datatype import kComplex32
from pninexus.h5cpp._datatype import kComplex64
from pninexus.h5cpp._datatype import kComplex128
from pninexus.h5cpp._datatype import kComplex256
from pninexus.h5cpp._datatype import kVariableString
from pninexus.h5cpp._datatype import kEBool

from pninexus.h5cpp._datatype import is_bool


class Factory(object):
    """Construct HDF5 datatypes from numpy types

    """

    type_map = {
        "bool": kEBool,
        "int8": kInt8,
        "uint8": kUInt8,
        "int16": kInt16,
        "uint16": kUInt16,
        "int32": kInt32,
        "uint32": kUInt32,
        "int64": kInt64,
        "uint64": kUInt64,
        "float16": kFloat16,
        "float32": kFloat32,
        "float64": kFloat64,
        "float128": kFloat128,
        "complex32": kComplex32,
        "complex64": kComplex64,
        "complex128": kComplex128,
        "complex256": kComplex256,
    }

    def create(self, dtype):
        """Create HDF5 type from numpy type

        :param numpy.dtype dtype: numpy datat type
        :return: HDF5 datatype
        :rtype: Datatype
        """

        if dtype.kind == 'S' or dtype.kind == 'U':
            if dtype.itemsize != 0:
                type = String.fixed(dtype.itemsize)
                type.padding = StringPad.NULLPAD
                return type
            else:
                return String.variable()
        else:
            return self.type_map[dtype.name]


kFactory = Factory()


def to_numpy(hdf5_datatype):
    """Convert an HDF5 datatype to a numpy type

    Takes an HDF5 datatype and converts it to the string representation
    of its numpy counterpart.

    :param Datatype hdf5_datatype: the HDF5 datatype to convert
    :return: numpy type
    :rtype: str
    """

    if not isinstance(hdf5_datatype, Datatype):
        raise TypeError("Instance of an HDF5 datatype required!")

    if hdf5_datatype == kInt8:
        return "int8"
    elif hdf5_datatype == kUInt8:
        return "uint8"
    elif hdf5_datatype == kInt16:
        return "int16"
    elif hdf5_datatype == kUInt16:
        return "uint16"
    elif hdf5_datatype == kInt32:
        return "int32"
    elif hdf5_datatype == kUInt32:
        return "uint32"
    elif hdf5_datatype == kInt64:
        return "int64"
    elif hdf5_datatype == kUInt64:
        return "uint64"
    elif hdf5_datatype == kFloat16:
        return "float16"
    elif hdf5_datatype == kFloat32:
        return "float32"
    elif hdf5_datatype == kFloat64:
        return "float64"
    elif hdf5_datatype == kFloat128:
        return "float128"
    elif hdf5_datatype == kComplex32:
        return "complex32"
    elif hdf5_datatype == kComplex64:
        return "complex64"
    elif hdf5_datatype == kComplex128:
        return "complex128"
    elif hdf5_datatype == kComplex256:
        return "complex256"
    elif hdf5_datatype == kEBool:
        return "bool"
    elif isinstance(hdf5_datatype, String):
        if hdf5_datatype.is_variable_length:
            return "object"
        else:
            return "S{}".format(hdf5_datatype.size)
    else:
        raise ValueError("Unsupported HDF5 datatype!")


def compound_getitem(self, key):
    """ get comound item datatype

    :param key: field index or field name
    :type key: `obj`:int: or `obj`:str:
    :returns: datatype
    :raises RuntimeError: in case of a failure
    """
    dt = self._getitem(key)
    if dt.type == Class.FLOAT:
        return Float(dt)
    if dt.type == Class.INTEGER:
        return Integer(dt)
    if dt.type == Class.STRING:
        return String(dt)
    if dt.type == Class.ENUM:
        return Enum(dt)
    return dt


Compound.__getitem__ = compound_getitem


__all__ = ["Class", "Order", "Sign", "Norm", "Pad", "StringPad", "Direction",
           "CharacterEncoding", "Datatype", "Float", "Integer", "String",
           "Compound",
           "kUInt8", "kInt8", "kUInt16", "kInt16", "kUInt32", "kInt32",
           "kUInt64", "kInt64",
           "kFloat16", "kFloat32", "kFloat64", "kFloat128",
           "kComplex32", "kComplex64", "kComplex128", "kComplex256",
           "kVariableString",
           "Factory", "kFactory", "to_numpy", "kEBool", "is_bool", "Enum"]
