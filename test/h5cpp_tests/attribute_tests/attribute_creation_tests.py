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
# Created on: Jan 29, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
from pni.io import h5cpp


class AttributeCreationTest(unittest.TestCase):
    
    filename = 'AttributeCreationTest.h5'
    
    def setUp(self):
        
        self.file = h5cpp.file.create(self.filename,h5cpp.file.AccessFlags.TRUNCATE)
        self.root = self.file.root()
        
    def tearDown(self):
        self.root.close()
        self.file.close()
        
    def testScalarIntAttribute(self):
        
        a = self.root.attributes.create("test",h5cpp.datatype.kInt32)
        self.assertEqual(a.dataspace.type,h5cpp.dataspace.Type.SCALAR)
        self.assertEqual(a.datatype.type,h5cpp.datatype.Class.INTEGER)
        
    def testScalarStringAttribute(self):
        
        a = self.root.attributes.create("test",h5cpp.datatype.kVariableString)
        self.assertEqual(a.dataspace.type,h5cpp.dataspace.Type.SCALAR)
        self.assertEqual(a.datatype.type,h5cpp.datatype.Class.STRING)
        self.assertTrue(h5cpp.datatype.String(a.datatype).is_variable_length)
        
        

