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
    Deflate, Fletcher32, Shuffle, ExternalFilter, ExternalFilters,
    NBit, SZip, ScaleOffset, SOScaleType, SZipOptionMask,
    is_filter_available, Availability)
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
        self.assertEqual(self.dcpl.nfilters, 1)
        self.assertTrue(filter.is_encoding_enabled())
        self.assertTrue(filter.is_decoding_enabled())

    def testShuffle(self):

        filter = Shuffle()
        filter(self.dcpl)
        hdf5.node.Dataset(self.root, hdf5.Path("Shuffle"),
                          self.datatype,
                          self.dataspace,
                          self.lcpl,
                          self.dcpl)
        self.assertEqual(self.dcpl.nfilters, 1)
        self.assertTrue(filter.is_encoding_enabled())
        self.assertTrue(filter.is_decoding_enabled())

    def testNBit(self):

        filter = NBit()
        filter(self.dcpl)
        hdf5.node.Dataset(self.root, hdf5.Path("NBit"),
                          self.datatype,
                          self.dataspace,
                          self.lcpl,
                          self.dcpl)
        self.assertEqual(self.dcpl.nfilters, 1)
        self.assertTrue(filter.is_encoding_enabled())
        self.assertTrue(filter.is_decoding_enabled())
        self.assertEqual(filter.id, 5)
        filters = ExternalFilters()
        self.assertEqual(len(filters), 0)
        flags = filters.fill(self.dcpl)
        self.assertEqual(len(filters), 1)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0], Availability.OPTIONAL)
        self.assertEqual(filters[0].cd_values, [])
        self.assertEqual(filters[0].id, 5)
        self.assertEqual(filters[0].name, "nbit")

    def testSZip(self):

        filter = SZip(SZipOptionMask.ENTROPY_CODING, 16)
        filter(self.dcpl)
        hdf5.node.Dataset(self.root, hdf5.Path("SZip"),
                          self.datatype,
                          self.dataspace,
                          self.lcpl,
                          self.dcpl)
        self.assertEqual(self.dcpl.nfilters, 1)
        self.assertTrue(filter.is_encoding_enabled())
        self.assertTrue(filter.is_decoding_enabled())
        self.assertEqual(filter.id, 4)
        filters = ExternalFilters()
        self.assertEqual(len(filters), 0)
        flags = filters.fill(self.dcpl)
        self.assertEqual(len(filters), 1)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0], Availability.OPTIONAL)
        self.assertEqual(filters[0].cd_values, [133, 16])
        self.assertEqual(filters[0].id, 4)
        self.assertEqual(filters[0].name, "szip")

    def testScaleOffset(self):

        sfilter = ScaleOffset(SOScaleType.INT, 2)
        sfilter(self.dcpl)
        hdf5.node.Dataset(self.root, hdf5.Path("ScaleOffset"),
                          self.datatype,
                          self.dataspace,
                          self.lcpl,
                          self.dcpl)
        self.assertEqual(self.dcpl.nfilters, 1)
        self.assertTrue(sfilter.is_encoding_enabled())
        self.assertTrue(sfilter.is_decoding_enabled())
        self.assertEqual(sfilter.id, 6)
        filters = ExternalFilters()
        self.assertEqual(len(filters), 0)
        flags = filters.fill(self.dcpl)
        self.assertEqual(len(filters), 1)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0], Availability.OPTIONAL)
        self.assertEqual(filters[0].cd_values, [SOScaleType.INT, 2])
        self.assertEqual(filters[0].id, 6)
        self.assertEqual(filters[0].name, "scaleoffset")

    def testDeflate(self):

        filter = Deflate(level=9)
        filter(self.dcpl)
        hdf5.node.Dataset(self.root, hdf5.Path("Deflate"),
                          self.datatype,
                          self.dataspace,
                          self.lcpl,
                          self.dcpl)
        self.assertEqual(self.dcpl.nfilters, 1)
        self.assertTrue(filter.is_encoding_enabled())
        self.assertTrue(filter.is_decoding_enabled())
        filters = ExternalFilters()
        self.assertEqual(len(filters), 0)
        flags = filters.fill(self.dcpl)
        self.assertEqual(len(filters), 1)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0], Availability.OPTIONAL)
        self.assertEqual(len(filters), 1)
        self.assertEqual(filters[0].cd_values, [9])
        self.assertEqual(filters[0].id, 1)
        self.assertEqual(filters[0].name, "deflate")

    def testExternalFilter(self):

        filter = ExternalFilter(1, [1], "mydeflate")
        filter(self.dcpl)
        hdf5.node.Dataset(self.root, hdf5.Path("ExternalFilter"),
                          self.datatype,
                          self.dataspace,
                          self.lcpl,
                          self.dcpl)
        self.assertEqual(self.dcpl.nfilters, 1)
        self.assertTrue(filter.is_encoding_enabled())
        self.assertTrue(filter.is_decoding_enabled())
        self.assertEqual(filter.name, "mydeflate")

        filters = ExternalFilters()
        self.assertEqual(len(filters), 0)
        flags = filters.fill(self.dcpl)
        self.assertEqual(len(filters), 1)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0], Availability.MANDATORY)
        self.assertEqual(len(filters), 1)
        self.assertEqual(filters[0].cd_values, [1])
        self.assertEqual(filters[0].id, 1)
        self.assertEqual(filters[0].name, "deflate")

    def testExternalFilter2(self):

        filter = ExternalFilter(32008, [0, 2])
        self.assertEqual(filter.id, 32008)
        self.assertEqual(filter.cd_values, [0, 2])
        if is_filter_available(32008):
            filter(self.dcpl)
            hdf5.node.Dataset(self.root, hdf5.Path("ExternalFilter2"),
                              self.datatype,
                              self.dataspace,
                              self.lcpl,
                              self.dcpl)
            filters = ExternalFilters()
            self.assertEqual(len(filters), 0)
            flags = filters.fill(self.dcpl)
            self.assertEqual(len(filters), 1)
            self.assertEqual(len(flags), 1)
            self.assertEqual(flags[0], Availability.MANDATORY)
            self.assertEqual(filters[0].cd_values, [0, 2])
            self.assertEqual(filters[0].id, 32008)
            self.assertEqual(
                filters[0].name,
                "bitshuffle; see https://github.com/kiyo-masui/bitshuffle")
        else:
            raise Exception("Bitshuffle filter is not available")
            error = False
            try:
                filter(self.dcpl)
                self.assertEqual(self.dcpl.nfilters, 1)
            except RuntimeError:
                error = True
            if not error:
                pass
                # print("filter with 32008 id  created")
            # for newer versions of hdf5 you can constuct the object
            #       on nonexisting filter
            # self.assertTrue(error)

    def testNonExistingExternalFilter(self):

        filter = ExternalFilter(31999, [0, 2])
        self.assertEqual(filter.id, 31999)
        self.assertEqual(filter.cd_values, [0, 2])
        if is_filter_available(31999):
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
                pass
                # print("filter with 31999 id  created")
            # for newer versions of hdf5 you can constuct the object
            #       on nonexisting filter
            # self.assertTrue(error)

    def testAll(self):

        # deflate = ExternalFilter(1, [5], "deflate")
        deflate = Deflate()
        deflate.level = 5
        self.assertEqual(deflate.level, 5)
        shuffle = Shuffle()
        fletcher = Fletcher32()

        deflate(self.dcpl)
        fletcher(self.dcpl)
        shuffle(self.dcpl)

        hdf5.node.Dataset(self.root, hdf5.Path("AllFilters"),
                          self.datatype,
                          self.dataspace,
                          self.lcpl,
                          self.dcpl)
        self.assertEqual(self.dcpl.nfilters, 3)
        filters = ExternalFilters()
        self.assertEqual(len(filters), 0)
        flags = filters.fill(self.dcpl)
        self.assertEqual(len(filters), 3)
        self.assertEqual(len(flags), 3)

        self.assertEqual(flags[0], Availability.OPTIONAL)
        self.assertEqual(filters[0].cd_values, [5])
        self.assertEqual(filters[0].id, 1)
        self.assertEqual(filters[0].name, "deflate")

        self.assertEqual(flags[1], Availability.MANDATORY)
        self.assertEqual(filters[1].cd_values, [])
        self.assertEqual(filters[1].id, 3)
        self.assertEqual(filters[1].name, "fletcher32")

        self.assertEqual(flags[2], Availability.OPTIONAL)
        self.assertEqual(filters[2].cd_values, [])
        self.assertEqual(filters[2].id, 2)
        self.assertEqual(filters[2].name, "shuffle")
