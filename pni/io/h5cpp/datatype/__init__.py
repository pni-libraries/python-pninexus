import numpy
#
# Import enumerations
#
from .._datatype import Class
from .._datatype import Order
from .._datatype import Sign
from .._datatype import Norm
from .._datatype import Pad
from .._datatype import StringPad
from .._datatype import Direction
from .._datatype import CharacterEncoding

#
# Import classes
# 
from .._datatype import Datatype
from .._datatype import Float
from .._datatype import Integer
from .._datatype import String

#
# Import predefined constant types
#
from .._datatype import kUInt8
from .._datatype import kInt8
from .._datatype import kUInt16
from .._datatype import kInt16
from .._datatype import kUInt32
from .._datatype import kInt32
from .._datatype import kUInt64
from .._datatype import kInt64
from .._datatype import kFloat64
from .._datatype import kFloat32
from .._datatype import kFloat128
from .._datatype import kVariableString

class Factory(object):
    
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
        
        if dtype.kind == 'S':
            type = String.fixed(dtype.itemsize)
            type.padding = StringPad.NULLPAD
            return type
        else:
            return self.type_map[dtype.name]

kFactory = Factory()
        
        
