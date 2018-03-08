from __future__ import print_function
import numpy

from pni.io.h5cpp import property
from pni.io.h5cpp._attribute import AttributeManager
from pni.io.h5cpp._attribute import Attribute


def attribute__getitem__(self,index):
    
    data = self.read()
    
    return data[index]
    
def attribute_write(self,data):

    write_data = data
    if not isinstance(write_data,numpy.ndarray):
        write_data = numpy.array(write_data)
        
    if write_data.dtype.kind == 'U':
        write_data = write_data.astype("S")
        
    try:
        self._write(write_data)
    except RuntimeError:
        print(write_data,write_data.dtype)
        


Attribute.__getitem__ = attribute__getitem__
Attribute.write = attribute_write