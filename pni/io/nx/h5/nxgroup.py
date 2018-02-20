#
# (c) Copyright 2018 DESY               
#
# This file is part of python-pni.
#
# python-pni is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# python-pni is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with python-pni.  If not, see <http://www.gnu.org/licenses/>.
# ===========================================================================
#
# Created on: Feb 19, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#

import numpy
from pni.io import nexus
from pni.io import h5cpp

from ..nxpath import nxpath
from .nxfield import nxfield

class nxgroup(h5cpp.node.Group):
    
    def __init__(self,base_instance=None):
        if base_instance!=None:
            super(nxgroup,self).__init__(base_instance)
        else:
            super(nxgroup,self).__init__()
            
    @property
    def path(self):
        
        nexus_path = nexus.get_path(self)
        return nxpath(base_instance = nexus_path)
    
    @property
    def size(self):
        
        return super(nxgroup,self).nodes.size
    
    @property
    def name(self):
        
        return super(nxgroup,self).link.path.name
    
    def names(self):
        
        name_list = []
        for node in self.nodes:
            name_list.append(node.link.path.name)
            
        return name_list
    
    def __len__(self):
        
        return super(nxgroup,self).nodes.size
    
    
    def __getitem__(self,index):
        
        object = self.nodes[index]
        
        if isinstance(object,h5cpp.node.Group):
            return nxgroup(base_instance = object)
        elif isinstance(object,h5cpp.node.Dataset):
            return nxfield(base_instance = object)
    
            
    def create_group(self,*args,**kwargs):
        
        if len(args) == 2:
            group = nexus.BaseClassFactory.create(self,h5cpp.Path(args[0]),args[1])
        elif len(args)==1:
    
            nxclass = None
            name    = None
    
            if "nxclass" in kwargs.keys():
                nxclass = kwargs["nxclass"]
    
            if ':' in args[0]:
                (name,nxclass) = args[0].split(":")
            else:
                name = args[0]
               
            if nxclass:
                group = nexus.BaseClassFactory.create(self,h5cpp.Path(name),nxclass)
            else:
                group = h5cpp.node.Group(self,name)
    
        else:
            raise ValueError
        
        return nxgroup(base_instance=group)
    
    
    def create_field(self,name,type,shape=None,chunk=None,filter=None):
        """Create a new field
        
        Create a new field below this group. 
    
        :param str name: the name of the new field
        :param str type: a numpy type string determining the data type of the field
        :param tuple shape: number of elements along each dimension
        :param tuple chunk: the chunk shape of the field
        :param filter: a filter object to compress the data
        :return: instance of a new field
        :rtype: instance of :py:class:`nxfield`
        """
    
        dtype = h5cpp.datatype.kFactory.create(numpy.dtype(type))
        
        if shape != None:
            space = h5cpp.dataspace.Simple(shape)
        else:
            space = h5cpp.dataspace.Scalar()
        
        if chunk != None:
            field = nexus.FieldFactory.create_chunked(self,h5cpp.Path(name),dtype,space,chunk)
        else:
            field = nexus.FieldFactory.create(self,h5cpp.Path(name),dtype,space)
            
        return nxfield(base_instance = field)
        
            

