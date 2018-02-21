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

class nxlink(h5cpp.node.Link):
    
    def __init__(self,base_instance=None):
        
        if base_instance != None:
            super(nxlink,self).__init__(base_instance)
        else:
            super(nxlink,self).__init__()
            
