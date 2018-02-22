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

from pni.io import nexus
from pni.io import h5cpp

from ..nxpath import nxpath
from __builtin__ import property


class nxfield(h5cpp.node.Dataset):
    
    def __init__(self,base_instance=None):
        if base_instance != None:
            super(nxfield,self).__init__(base_instance)
        else:
            super(nxfield,self).__init__()

    @property
    def path(self):
        
        return nxpath(base_instance = nexus.get_path(self))
    
    @property
    def name(self):
        
        return self.link.path.name
    
    @property
    def dtype(self):
        
        return h5cpp.datatype.to_numpy(self.datatype)
    
    @property
    def shape(self):
        
        space = self.dataspace
        if isinstance(space,h5cpp.dataspace.Scalar):
            return (1,)
        else:
            return h5cpp.dataspace.Simple(space).current_dimensions
        
    @property
    def size(self):
        
        return self.dataspace.size
        
    def __getitem__(self,index):
        
        return self.read()