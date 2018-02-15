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
from pni.io.h5cpp.dataspace import Hyperslab
from pni.io.h5cpp.datatype import kInt32
import numpy
import numpy.testing as npt

class DatasetPartialIOTests(unittest.TestCase):
    
    filename = "DatasetPartialIOTests.h5"
    
    @classmethod
    def setUpClass(cls):
        super(DatasetPartialIOTests, cls).setUpClass()
        
        h5cpp.file.create(cls.filename,AccessFlags.TRUNCATE)
        
    def setUp(self):
        
        self.file = h5cpp.file.open(self.filename,AccessFlags.READWRITE)
        self.root = self.file.root()
        
    def tearDown(self):
        unittest.TestCase.tearDown(self)
        
        self.root.close()
        self.file.close()
        
    def testWriteReadStrips(self):
        
        dataspace = Simple((3,5))
        data_base = numpy.ones((5,),dtype="int32")
        dataset = Dataset(self.root,h5cpp.Path("WriteReadStrips"),kInt32,dataspace)
        
        #
        # write data
        #
        selection = Hyperslab(offset=(0,0),block=(1,5))
        dataset.write(data_base,selection=selection)
        selection.offset(0,1)
        dataset.write(2*data_base,selection=selection)
        selection.offset(0,2)
        dataset.write(3*data_base,selection=selection)
        
        #
        # read data back
        #
        selection.offset(0,0)
        npt.assert_array_equal(dataset.read(selection=selection),data_base)
        selection.offset(0,1)
        npt.assert_array_equal(dataset.read(selection=selection),2*data_base)
        selection.offset(0,2)
        npt.assert_array_equal(dataset.read(selection=selection),3*data_base)
        
    def testWriteReadPoints(self):
        
        dataspace = Simple((3,5))
        dataset = Dataset(self.root,h5cpp.Path("WriteReadPoints"),kInt32,dataspace)
        
        value = 0
        selection = Hyperslab(offset=(0,0),block=(1,1))
        for i in range(3):
            selection.offset(0,i)
            for j in range(5):
                selection.offset(1,j)
                dataset.write(data=value,selection=selection)
                self.assertEqual(dataset.read(selection=selection)[0],value)
                value+=1
                
        
