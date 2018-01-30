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
# Created on: Jan 30, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import numpy
from pni.io import h5cpp


class AttributeIOTests(unittest.TestCase):
    
    filename = "AttributeIOTests.h5"
    
    @classmethod
    def setUpClass(cls):
        super(AttributeIOTests, cls).setUpClass()
        
        h5cpp.file.create(cls.filename,h5cpp.file.AccessFlags.TRUNCATE)
        
    def setUp(self):
        self.file = h5cpp.file.open(self.filename,h5cpp.file.AccessFlags.READWRITE)
        self.root = self.file.root()
        
    def tearDown(self):
        self.root.close()
        self.file.close()
        
    def testIntegerScalar(self):
        
        a = self.root.attributes.create("IntegerScalar",h5cpp.datatype.kInt32)
        a.write(42)
        
    def testFloatScalar(self):
        
        a = self.root.attributes.create("FloatScalar",h5cpp.datatype.kFloat32)
        a.write(42.345)
        
    def testFloatNumpyScalar(self):
        
        a = self.root.attributes.create("FloatNumpyScalar",h5cpp.datatype.kFloat32)
        a.write(numpy.float32(34.5323))
        
    def testStringScalar(self):
        
        a = self.root.attributes.create("StringScalar",h5cpp.datatype.kVariableString)
        a.write("hello world")
        
    def testStringArray(self):
        
        data = numpy.array([["hello","world","this"],["is","a","test"]])
        a = self.root.attributes.create("StringArray",h5cpp.datatype.kVariableString,(2,3))
        a.write(data)
        
    def testIntArray(self):
        
        data = numpy.array([1,2,3])
        a = self.root.attributes.create("IntArray",h5cpp.datatype.kInt32,(3,))
        a.write(data)
        
        
