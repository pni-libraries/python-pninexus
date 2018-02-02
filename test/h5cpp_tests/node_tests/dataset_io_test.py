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
# Created on: Jan 31, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
from pni.io import h5cpp
from pni.io.h5cpp.file import AccessFlags
from pni.io.h5cpp.node import Dataset
from pni.io.h5cpp.dataspace import Simple
from pni.io.h5cpp.dataspace import Scalar
import numpy
import numpy.testing as npt

class DatasetAllIOTests(unittest.TestCase):
    
    filename = "DatasetAllIOTests.h5"
    
    @classmethod
    def setUpClass(cls):
        super(DatasetAllIOTests, cls).setUpClass()
        
        h5cpp.file.create(cls.filename,AccessFlags.TRUNCATE)
        
    def setUp(self):
        
        self.file = h5cpp.file.open(self.filename,AccessFlags.READWRITE)
        self.root = self.file.root()
        
    def tearDown(self):
        unittest.TestCase.tearDown(self)
        
        self.root.close()
        self.file.close()
        
    def testWriteIntegerScalar(self):
        
        dataset = Dataset(parent=self.root,
                          path=h5cpp.Path("IntegerScalar"),
                          type=h5cpp.datatype.kInt32,
                          space=Scalar())
        dataset.write(42)
        read = dataset.read()
        self.assertEqual(read,42)
        
    def testWriteFloatScalar(self):
        
        dataset = Dataset(self.root,h5cpp.Path("FloatScalar"),
                          h5cpp.datatype.kFloat32,Scalar())
        dataset.write(23.321)
        read = dataset.read()
        self.assertEqual(read,23.321)
        
    def testWriteStringScalar(self):
        
        data = "hello world"
        dtype = h5cpp.datatype.String.fixed(len(data))
        dtype.padding = h5cpp.datatype.StringPad.NULLPAD
        dataset = Dataset(self.root,h5cpp.Path("StringScalar"),dtype,Scalar())
        dataset.write(data)
        #read = dataset.read()
        #self.assertEqual(read,"hello world")
        
    def testWriteIntegerArray(self):
        
        data = numpy.array([[1,2,3,4],[5,6,7,8]])
        dataset = Dataset(self.root,h5cpp.Path("IntegerArray"),
                          h5cpp.datatype.kInt32,Simple((2,4)))
        dataset.write(data)
        read = dataset.read()
        npt.assert_array_equal(read,data)
        
        #
        # test inplace reading
        #
        read = numpy.zeros(dataset.dataspace.current_dimensions,dtype="int64")
        dataset.read(read)
        npt.assert_array_equal(read,data)
        
        read = numpy.zeros((2,2),dtype="float32")
        self.assertRaises(RuntimeError,dataset.read,read)
        
        
    def testWriteFloatArray(self):
        
        data = numpy.array([[1,2,3,4],[5,6,7,8]],dtype="float64")
        dataset = Dataset(self.root,h5cpp.Path("FloatArray"),
                          h5cpp.datatype.kFloat64,Simple((2,4)))
        dataset.write(data)
        read = dataset.read()
        npt.assert_array_equal(read,data)
        
#    def testWriteStringArray(self):
#        
#        data = numpy.array([["hello","world","this"],["is","a","test"]])
#        dtype = h5cpp.datatype.String.fixed(5)
#        dtype.padding = h5cpp.datatype.StringPad.NULLTERM
#        dataset = Dataset(self.root,h5cpp.Path("StringArray"),
#                          dtype,
#                          Simple((2,3)))
#        dataset.write(data)
#        read = dataset.read()
#        npt.assert_array_equal(read,data)
        
        
                          
