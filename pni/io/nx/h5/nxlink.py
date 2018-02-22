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
# Created on: Feb 21, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#

from pni.io import h5cpp
from pni.io import nexus

class nxlink_type(object):
    HARD = 1
    SOFT = 2
    EXTERNAL = 3

class nxlink_status(object):
    VALID = 1
    INVALID = 2

class nxlink(h5cpp.node.Link):
    
    def __init__(self,base_instance=None):
        
        if base_instance != None:
            super(nxlink,self).__init__(base_instance)
        else:
            super(nxlink,self).__init__()
            
    @property
    def type(self):
        pass
    
    @property
    def status(self):
        pass
            



def link(target,base,link_path):
    
    if isinstance(link_path,str):
        link_path = h5cpp.Path(link_path)
        
    if isinstance(target,str):
        target = h5cpp.Path(target)
    
    h5cpp.node.link(target,base,link_path)