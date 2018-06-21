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
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import os
from pninexus import h5cpp
from pninexus.h5cpp.file import AccessFlags
from pninexus.h5cpp.node import Dataset
from pninexus.h5cpp.dataspace import Simple
# from pninexus.h5cpp.dataspace import Scalar
from pninexus.h5cpp.datatype import kFloat32
from pninexus.h5cpp.property import LinkCreationList
from pninexus.h5cpp.property import DatasetCreationList
# import numpy
# import numpy.testing as npt


module_path = os.path.dirname(os.path.abspath(__file__))


class DatasetExtentTest(unittest.TestCase):
    """
    Testing the functionality of the ``extent`` methods.
    """

    filename = os.path.join(module_path, "DatasetExtentTest.h5")
    limited_dataspace = Simple((10, 10), (100, 10))
    unlimited_dataspace = Simple((0, 10), (h5cpp.dataspace.UNLIMITED, 10))
    lcpl = LinkCreationList()
    dcpl = DatasetCreationList()

    @classmethod
    def setUpClass(cls):
        super(DatasetExtentTest, cls).setUpClass()

        h5cpp.file.create(cls.filename, AccessFlags.TRUNCATE)
        cls.dcpl.layout = h5cpp.property.DatasetLayout.CHUNKED
        cls.dcpl.chunk = (1, 10)

    def setUp(self):

        self.file = h5cpp.file.open(self.filename, AccessFlags.READWRITE)
        self.root = self.file.root()

    def tearDown(self):

        self.root.close()
        self.file.close()

    def testSetTotalExtentLimited(self):

        dataset = Dataset(self.root, h5cpp.Path("SetTotalExtentLimited"),
                          kFloat32,
                          self.limited_dataspace,
                          self.lcpl,
                          self.dcpl)
        dataset.extent((90, 10))
        # check the new shape
        self.assertEqual(dataset.dataspace.current_dimensions, (90, 10))

        self.assertRaises(RuntimeError, dataset.extent, (90, 11))
        self.assertRaises(RuntimeError, dataset.extent, (101, 10))
        # dimensions should remain unaltered
        self.assertEqual(dataset.dataspace.current_dimensions, (90, 10))

        #
        # now we shring the dataset again
        #
        dataset.extent((45, 5))
        self.assertEqual(dataset.dataspace.current_dimensions, (45, 5))

    def testSetTotalExtentUnlimited(self):

        dataset = Dataset(self.root, h5cpp.Path("SetTotalExtentUnlimited"),
                          kFloat32,
                          self.unlimited_dataspace,
                          self.lcpl,
                          self.dcpl)
        dataset.extent((1000, 10))
        self.assertEqual(dataset.dataspace.current_dimensions, (1000, 10))

        self.assertRaises(RuntimeError, dataset.extent, (1000, 11))

    def testGrowExtentLimited(self):

        dataset = Dataset(self.root, h5cpp.Path("GrowExtentLimited"),
                          kFloat32,
                          self.limited_dataspace,
                          self.lcpl,
                          self.dcpl)
        dataset.extent(0, 10)
        self.assertEqual(dataset.dataspace.current_dimensions, (20, 10))

        self.assertRaises(RuntimeError, dataset.extent, 0, 90)
        self.assertRaises(RuntimeError, dataset.extent, 1, 1)
        self.assertEqual(dataset.dataspace.current_dimensions, (20, 10))

        #
        # shrink the dataset again
        #
        dataset.extent(0, -2)
        dataset.extent(1, -3)
        self.assertEqual(dataset.dataspace.current_dimensions, (18, 7))

    def testGrowExtentUnlimited(self):

        dataset = Dataset(self.root, h5cpp.Path("GrowExtentUnlimited"),
                          kFloat32,
                          self.unlimited_dataspace,
                          self.lcpl,
                          self.dcpl)
        dataset.extent(0, 10000)
        self.assertEqual(dataset.dataspace.current_dimensions, (10000, 10))

        self.assertRaises(RuntimeError, dataset.extent, 1, 1)
