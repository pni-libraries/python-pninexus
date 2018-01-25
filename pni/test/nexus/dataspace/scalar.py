#
# (c) Copyright 2018 DESY, 
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
# Created on: Jan 25, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import os
import numpy

from pni.io.nexus.dataspace import Scalar
from pni.io.nexus.dataspace import Dataspace
from pni.io.nexus.dataspace import Type

class ScalarTest(unittest.TestCase):
    
    def testCreation(self):
        
        space = Scalar()
        self.assertEqual(space.size,1)
        self.assertEqual(space.type,Type.SCALAR)
        
        
        
        
        