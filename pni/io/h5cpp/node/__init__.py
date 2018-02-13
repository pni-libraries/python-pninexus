from __future__ import print_function
from .. import property
from .. import dataspace
from .. import datatype
import numpy
#
# import enumeration wrappers
#
from .._node import Type
from .._node import LinkType

#
# import node classes
#
from .._node import Node
from .._node import GroupView
from .._node import NodeView
from .._node import LinkView
from .._node import Group
from .._node import Dataset
from .._node import LinkTarget
from .._node import Link

def dataset_write(self,data,selection=None):
    
    #
    # in case that the parameter passed is not an numpy array we 
    # have to create one from it
    #
    if not isinstance(data,numpy.ndarray):
        data = numpy.array([data,])
        
    #
    # determine memory datatype and dataspace
    # - if the file type is a variable length string we have to adjust the 
    #   memory type accordingly
    memory_space = dataspace.Simple(data.shape)
    memory_type  = datatype.kFactory.create(data.dtype)
    
    if isinstance(self.datatype,datatype.String):
        if self.datatype.is_variable_length:
            memory_type = datatype.String.variable()
    
    #
    # get the file dataspace
    #
    file_space = self.dataspace

    if selection != None:
        file_space.selection(dataspace.SelectionOperation.SET,selection)
        
    self._write(data,memory_type,memory_space,file_space)
    
    
    
    
        
Dataset.write = dataset_write

