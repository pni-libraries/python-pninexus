#
# (c) Copyright 2018 DESY
#
# This file is part of python-pninexus.
#
# python-pninexus is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# python-pninexus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with python-pninexus.  If not, see <http://www.gnu.org/licenses/>.
# ===========================================================================
#
# Created on: Jan 31, 2018
#     Authors:
#             Eugen Wintersberger <eugen.wintersberger@desy.de>
#             Jan Kotanski <jan.kotanski@desy.de>
#
from __future__ import print_function
import unittest
import os
from pninexus.h5cpp.filter import (
    Deflate, Fletcher32, Shuffle, ExternalFilter, is_filter_available)
import pninexus.h5cpp as hdf5


module_path = os.path.dirname(os.path.abspath(__file__))


class FilterCreationTest(unittest.TestCase):

    filename = os.path.join(module_path, "FilterCreationTest.h5")
    dataspace = hdf5.dataspace.Simple((10, 2))
    datatype = hdf5.datatype.kInt32
    lcpl = hdf5.property.LinkCreationList()
    dcpl = hdf5.property.DatasetCreationList()

    @classmethod
    def setUpClass(cls):
        super(FilterCreationTest, cls).setUpClass()

        hdf5.file.create(cls.filename, hdf5.file.AccessFlags.TRUNCATE)

    def setUp(self):

        self.file = hdf5.file.open(
            self.filename, hdf5.file.AccessFlags.READWRITE)
        self.root = self.file.root()
        self.dcpl = hdf5.property.DatasetCreationList()
        self.dcpl.layout = hdf5.property.DatasetLayout.CHUNKED
        self.dcpl.chunk = (10, 2)

    def tearDown(self):
        self.root.close()
        self.file.close()

    def testFletcher32(self):

        filter = Fletcher32()
        filter(self.dcpl)
        hdf5.node.Dataset(self.root, hdf5.Path("Fletcher32"),
                          self.datatype,
                          self.dataspace,
                          self.lcpl,
                          self.dcpl)

    def testShuffle(self):

        filter = Shuffle()
        filter(self.dcpl)
        hdf5.node.Dataset(self.root, hdf5.Path("Shuffle"),
                          self.datatype,
                          self.dataspace,
                          self.lcpl,
                          self.dcpl)

    def testDeflate(self):

        filter = Deflate(level=9)
        filter(self.dcpl)
        hdf5.node.Dataset(self.root, hdf5.Path("Deflate"),
                          self.datatype,
                          self.dataspace,
                          self.lcpl,
                          self.dcpl)

    def testExternalFilter(self):

        filter = ExternalFilter(1, [1])
        filter(self.dcpl)
        hdf5.node.Dataset(self.root, hdf5.Path("ExternalFilter"),
                          self.datatype,
                          self.dataspace,
                          self.lcpl,
                          self.dcpl)

    def testExternalFilter2(self):

        filter = ExternalFilter(32008, [0, 2])
        self.assertEqual(filter.id, 32008)
        self.assertEqual(filter.cd_values, [0, 2])
        if(is_filter_available(32008)):
            filter(self.dcpl)
            hdf5.node.Dataset(self.root, hdf5.Path("ExternalFilter2"),
                              self.datatype,
                              self.dataspace,
                              self.lcpl,
                              self.dcpl)
        else:
            error = False
            try:
                filter(self.dcpl)
            except RuntimeError:
                error = True
            if not error:
                print("filter with 32008 id  created")
            # for newer versions of hdf5 you can constuct the object
            #       on nonexisting filter
            # self.assertTrue(error)

    def testNonExistingExternalFilter(self):

        filter = ExternalFilter(31999, [0, 2])
        self.assertEqual(filter.id, 31999)
        self.assertEqual(filter.cd_values, [0, 2])
        if(is_filter_available(31999)):
            filter(self.dcpl)
            hdf5.node.Dataset(self.root, hdf5.Path("ExternalFilter3"),
                              self.datatype,
                              self.dataspace,
                              self.lcpl,
                              self.dcpl)
        else:
            error = False
            try:
                filter(self.dcpl)
            except RuntimeError:
                error = True
            if not error:
                print("filter with 31999 id  created")
            # for newer versions of hdf5 you can constuct the object
            #       on nonexisting filter
            # self.assertTrue(error)

    def testAll(self):

        deflate = Deflate()
        deflate.level = 5
        shuffle = Shuffle()
        fletcher = Fletcher32()

        fletcher(self.dcpl)
        shuffle(self.dcpl)
        deflate(self.dcpl)

        hdf5.node.Dataset(self.root, hdf5.Path("AllFilters"),
                          self.datatype,
                          self.dataspace,
                          self.lcpl,
                          self.dcpl)
