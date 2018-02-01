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
from pni.io.h5cpp.filter import Deflate,Fletcher32,Shuffle
import pni.io.h5cpp as hdf5


class FilterCreationTest(unittest.TestCase):
    
    filename = "FilterCreationTest.h5"
    dataspace = hdf5.dataspace.Simple((10,2))
    datatype = hdf5.datatype.kInt32
    lcpl = hdf5.property.LinkCreationList()
    dcpl = hdf5.property.DatasetCreationList()
    
    
    @classmethod
    def setUpClass(cls):
        super(FilterCreationTest, cls).setUpClass()
        
        hdf5.file.create(cls.filename,hdf5.file.AccessFlags.TRUNCATE)
        
    def setUp(self):
        
        self.file = hdf5.file.open(self.filename,hdf5.file.AccessFlags.READWRITE)
        self.root = self.file.root()
        self.dcpl = hdf5.property.DatasetCreationList()
        self.dcpl.layout = hdf5.property.DatasetLayout.CHUNKED
        self.dcpl.chunk = (10,2)
        
    def tearDown(self):
        self.root.close()
        self.file.close()
        
    def testFletcher32(self):
        
        filter = Fletcher32()
        filter(self.dcpl)
        d = hdf5.node.Dataset(self.root,hdf5.Path("Fletcher32"),self.datatype,self.dataspace,self.lcpl,self.dcpl)
        
    def testShuffle(self):
        
        filter = Shuffle()
        filter(self.dcpl)
        d = hdf5.node.Dataset(self.root,hdf5.Path("Shuffle"),self.datatype,self.dataspace,self.lcpl,self.dcpl)
        
    def testDeflate(self):
        
        filter = Deflate(level=9)
        filter(self.dcpl)
        d = hdf5.node.Dataset(self.root,hdf5.Path("Deflate"),self.datatype,self.dataspace,self.lcpl,self.dcpl)
        
    def testAll(self):
        
        deflate = Deflate()
        deflate.level = 5
        shuffle = Shuffle()
        fletcher = Fletcher32()
        
        fletcher(self.dcpl)
        shuffle(self.dcpl)
        deflate(self.dcpl)
        
        d = hdf5.node.Dataset(self.root,hdf5.Path("AllFilters"),self.datatype,self.dataspace,self.lcpl,self.dcpl)
