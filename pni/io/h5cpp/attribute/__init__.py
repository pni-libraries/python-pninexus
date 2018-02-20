from __future__ import print_function

from pni.io.h5cpp import property
from pni.io.h5cpp._attribute import AttributeManager
from pni.io.h5cpp._attribute import Attribute


def attribute__getitem__(self,index):
    
    data = self.read()
    
    return data[index]


Attribute.__getitem__ = attribute__getitem__