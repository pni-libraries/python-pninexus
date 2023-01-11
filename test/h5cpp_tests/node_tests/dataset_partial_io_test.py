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
from pninexus.h5cpp.dataspace import Hyperslab, SelectionType, Points
from pninexus.h5cpp.datatype import kInt32
from pninexus.h5cpp.datatype import String
from pninexus.h5cpp.property import LinkCreationList
from pninexus.h5cpp.property import DatasetCreationList
from pninexus.h5cpp.property import DatasetLayout
import numpy
import numpy.testing as npt


module_path = os.path.dirname(os.path.abspath(__file__))


class DatasetPartialIOTests(unittest.TestCase):

    filename = os.path.join(module_path, "DatasetPartialIOTests.h5")

    @classmethod
    def setUpClass(cls):
        super(DatasetPartialIOTests, cls).setUpClass()

        h5cpp.file.create(cls.filename, AccessFlags.TRUNCATE)

    def setUp(self):

        self.file = h5cpp.file.open(self.filename, AccessFlags.READWRITE)
        self.root = self.file.root()

    def tearDown(self):
        unittest.TestCase.tearDown(self)

        self.root.close()
        self.file.close()

    def testWriteVariabelLengthArrayStrips(self):
        """
        The use case here would be a log
        """
        #
        # create file datatype and dataspace
        #
        dtype = String.variable()
        space = Simple((0,), (h5cpp.dataspace.UNLIMITED,))
        #
        # create property lists
        #
        link_creation_list = LinkCreationList()
        dataset_creation_list = DatasetCreationList()
        dataset_creation_list.layout = DatasetLayout.CHUNKED
        dataset_creation_list.chunk = (1,)

        #
        # create the dataset
        #
        dataset = Dataset(
            self.root, h5cpp.Path("Log"), dtype, space,
            link_creation_list, dataset_creation_list)

        # writing data by log
        log_lines = ["hello", "first entry", "second entry"]
        line_selection = Hyperslab(offset=(0,), block=(1,))
        for line in log_lines:
            dataset.extent(0, 1)
            dataset.write(data=line, selection=line_selection)
            line_selection.offset(0, line_selection.offset()[0] + 1)

        #
        # read back data
        #
        for index in range(3):
            line_selection.offset(0, index)
            line = dataset.read(selection=line_selection)
            self.assertEqual(line, log_lines[index])

    def testWriteReadStrips(self):

        dataspace = Simple((3, 5))
        data_base = numpy.ones((5,), dtype="int32")
        dataset = Dataset(self.root,
                          h5cpp.Path("WriteReadStrips"), kInt32, dataspace)

        #
        # write data
        #
        selection = Hyperslab(offset=(0, 0), block=(1, 5))

        self.assertEqual(selection.rank, 2)
        self.assertEqual(selection.size, 5)
        self.assertEqual(selection.dimensions(), (1, 5))
        self.assertEqual(selection.type, SelectionType.HYPERSLAB)

        dataset.write(data_base, selection=selection)
        selection.offset(0, 1)
        dataset.write(2 * data_base, selection=selection)
        selection.offset(0, 2)
        dataset.write(3 * data_base, selection=selection)

        #
        # read data back
        #
        selection.offset(0, 0)
        npt.assert_array_equal(
            dataset.read(selection=selection), data_base)
        selection.offset(0, 1)
        npt.assert_array_equal(
            dataset.read(selection=selection), 2 * data_base)
        selection.offset(0, 2)
        npt.assert_array_equal(
            dataset.read(selection=selection), 3 * data_base)

    def testWriteReadMultiPoints(self):

        dataspace = Simple((5, 3))
        dataset = Dataset(
            self.root, h5cpp.Path("WriteReadMultiPoints"), kInt32, dataspace)

        data_base = numpy.ones((3), dtype="int32")

        for i in range(3):
            selection = Points([[i, 0], [i + 2, 2], [i + 2, 1]])
            self.assertEqual(selection.rank, 2)
            self.assertEqual(selection.size, 3)
            self.assertEqual(selection.dimensions(), (2, 3))
            self.assertEqual(selection.type, SelectionType.POINTS)
            dataset.write(data_base * (i + 1), selection=selection)

        for i in range(3):
            selection = Points([[i, 0], [i + 2, 2], [i + 2, 1]])
            npt.assert_array_equal(
                dataset.read(selection=selection), data_base * (i + 1))

    def testWriteReadHyperPoints(self):

        dataspace = Simple((3, 5))
        dataset = Dataset(
            self.root, h5cpp.Path("WriteReadHyperPoints"), kInt32, dataspace)

        value = 0
        selection = Hyperslab(offset=(0, 0), block=(1, 1))

        self.assertEqual(selection.rank, 2)
        self.assertEqual(selection.size, 1)
        self.assertEqual(selection.dimensions(), (1, 1))
        self.assertEqual(selection.type, SelectionType.HYPERSLAB)

        for i in range(3):
            selection.offset(0, i)
            for j in range(5):
                selection.offset(1, j)
                dataset.write(data=value, selection=selection)
                self.assertEqual(
                    dataset.read(selection=selection)[0], value)
                value += 1

    def testWriteReadPoints(self):

        dataspace = Simple((3, 5))
        dataset = Dataset(
            self.root, h5cpp.Path("WriteReadPoints"), kInt32, dataspace)

        value = 0
        for i in range(3):
            for j in range(5):
                selection = Points([[i, j]])
                self.assertEqual(selection.rank, 2)
                self.assertEqual(selection.size, 1)
                self.assertEqual(selection.dimensions(), (1, 1))
                self.assertEqual(selection.type, SelectionType.POINTS)
                dataset.write(data=value, selection=selection)
                self.assertEqual(
                    dataset.read(selection=selection)[0], value)
                value += 1
