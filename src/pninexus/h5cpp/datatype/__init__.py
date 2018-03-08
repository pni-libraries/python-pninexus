import numpy
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
from pninexus.h5cpp._datatype import String

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
from pninexus.h5cpp._datatype import kFloat64
from pninexus.h5cpp._datatype import kFloat32
from pninexus.h5cpp._datatype import kFloat128
from pninexus.h5cpp._datatype import kVariableString

class Factory(object):
    """Construct HDF5 datatypes from numpy types
    
    """
    
    type_map = {"int8":kInt8,
           "uint8":kUInt8,
           "int16":kInt16,
           "uint16":kUInt16,
           "int32":kInt32,
           "uint32":kUInt32,
           "int64":kInt64,
           "uint64":kUInt64,
           "float32":kFloat32,
           "float64":kFloat64,
           "float128":kFloat128}
    
    def create(self,dtype):
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
    
    
    if not isinstance(hdf5_datatype,Datatype):
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
    elif hdf5_datatype == kFloat32:
        return "float32"
    elif hdf5_datatype == kFloat64:
        return "float64"
    elif hdf5_datatype == kFloat128:
        return "float128"
    elif isinstance(hdf5_datatype,String):
        if hdf5_datatype.is_variable_length:
            return "object"
        else:
            return "S{}".format(hdf5_datatype.size)
    else:
        raise ValueError("Unsupported HDF5 datatype!")
        
        
        
        
