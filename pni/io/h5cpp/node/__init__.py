from __future__ import print_function
from pni.io.h5cpp import property
from pni.io.h5cpp import dataspace
from pni.io.h5cpp import datatype
import numpy
from collections import OrderedDict
#
# import enumeration wrappers
#
from pni.io.h5cpp._node import Type
from pni.io.h5cpp._node import LinkType

#
# import node classes
#
from pni.io.h5cpp._node import Node
from pni.io.h5cpp._node import GroupView
from pni.io.h5cpp._node import NodeView
from pni.io.h5cpp._node import LinkView
from pni.io.h5cpp._node import Group
from pni.io.h5cpp._node import Dataset
from pni.io.h5cpp._node import LinkTarget
from pni.io.h5cpp._node import Link

def selection_to_shape(selection):
    """Convert a selection to a numpy array shape
    
    This utilty function converts an HDF5 selection to a tuple which can 
    be used as a numpy array shape. This function performs some kind of index
    reduction: the resulting shape is the minimum shape required to store the 
    data referenced by the selection. This means that all unnecessary dimensions
    with only a single element are removed. 
    
    For instance 
    [1,1,1,1] -> [1]
    [1,2,3,1] -> [2,3]
    
    """
    
    if not isinstance(selection,dataspace.Hyperslab):
        raise TypeError("Shape conversion currently only works for Hyperslabs")
    
    shape = []
    size = 1
    for blocks,counts in zip(selection.block(),selection.count()):
        size *= blocks*counts 
        shape.append(blocks*counts)
        

    if size == 1:
        #
        # it the total number of elements in the selection is 1 the shape is 
        # always (1,) no matter how many dimension are in the selection. 
        #
        return (1,)
    elif len(shape)>1:
        shape = [s for s in shape if s!=1]
        
    return shape

def dataset_write(self,data,selection=None):
    
    #
    # in case that the parameter passed is not an numpy array we 
    # have to create one from it
    #
    if not isinstance(data,numpy.ndarray):
        data = numpy.array(data)
        
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
    

def dataset_read(self,data=None,selection=None):
    
    memory_space = None
    memory_type  = None
    file_space   = self.dataspace
    
    if selection != None:
        file_space.selection(dataspace.SelectionOperation.SET,selection)
    
    if data!=None:
        if not isinstance(data,numpy.ndarray):
            raise TypeError("Inplace reading is only supported for numpy arrays!")
        
        memory_space = dataspace.Simple(data.shape)
        memory_type  = datatype.kFactory.create(data.dtype)
        
        if isinstance(self.datatype,datatype.String):
            if self.datatype.is_variable_length:
                memory_type = datatype.String.variable()
                
    else:
        memory_type  = self.datatype
        
        if selection != None:
            shape = selection_to_shape(selection)
            memory_space = dataspace.Simple(shape)
        else:
            memory_space = file_space
            shape = (1,)
            if file_space.type == dataspace.Type.SIMPLE:
                shape = dataspace.Simple(file_space).current_dimensions
            
        
        data = numpy.empty(shape,dtype=datatype.to_numpy(memory_type))
        
        
    data = self._read(data,memory_type,memory_space,file_space)
    
    return data
        

        
Dataset.write = dataset_write
Dataset.read  = dataset_read

