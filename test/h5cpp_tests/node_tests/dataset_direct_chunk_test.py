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
#     Author: Jan Kotanski <jan.kotanski@desy.de>
#
from __future__ import print_function
import unittest
import os
from pninexus import h5cpp
from pninexus.h5cpp.file import AccessFlags
from pninexus.h5cpp.node import Dataset
from pninexus.h5cpp.dataspace import Simple
# from pninexus.h5cpp.dataspace import Scalar
from pninexus.h5cpp.dataspace import Hyperslab
from pninexus.h5cpp.filter import Deflate
from pninexus.h5cpp.datatype import kUInt32
from pninexus.h5cpp.datatype import kUInt16
from pninexus.h5cpp.property import LinkCreationList
from pninexus.h5cpp.property import DatasetCreationList
from pninexus.h5cpp.property import DatasetAccessList
from pninexus.h5cpp.property import DatasetLayout
import numpy
import numpy.testing as npt


try:
    h5cpp.node.Dataset.read_chunk
    HDF5GE102 = True
except Exception:
    HDF5GE102 = False

module_path = os.path.dirname(os.path.abspath(__file__))


class DatasetDirectChunkTests(unittest.TestCase):

    filename = os.path.join(module_path, "DatasetDirectChunkTests.h5")

    @classmethod
    def setUpClass(cls):
        super(DatasetDirectChunkTests, cls).setUpClass()

        h5cpp.file.create(cls.filename, AccessFlags.TRUNCATE)

    def setUp(self):

        self.file = h5cpp.file.open(self.filename, AccessFlags.READWRITE)
        self.root = self.file.root()

    def tearDown(self):
        unittest.TestCase.tearDown(self)

        self.root.close()
        self.file.close()

    def testWriteChunk(self):
        """
        direct chunk writing
        """
        xdim = 867
        ydim = 700
        nframe = 33

        space = Simple((0, xdim, ydim),
                       (h5cpp.dataspace.UNLIMITED,
                        h5cpp.dataspace.UNLIMITED,
                        h5cpp.dataspace.UNLIMITED))

        framespace = Hyperslab(
            offset=(0, 0, 0),
            block=(1, xdim, ydim)
        )

        link_creation_list = LinkCreationList()

        dataset_creation_list = DatasetCreationList()
        dataset_creation_list.layout = DatasetLayout.CHUNKED
        dataset_creation_list.chunk = (1, xdim, ydim)

        dataset_access_list = DatasetAccessList()

        data1 = Dataset(
            self.root, h5cpp.Path("data1"), kUInt16, space,
            link_creation_list, dataset_creation_list,
            dataset_access_list)

        frame = numpy.random.randint(
            0, 65535, size=(xdim * ydim,), dtype="uint16")

        for i in range(nframe):
            data1.extent(0, 1)
            data1.write_chunk(frame, [i, 0, 0])

        read_value = numpy.zeros(shape=[xdim * ydim], dtype="uint16")

        for i in range(nframe):
            framespace.offset([i, 0, 0])
            data1.read(read_value, framespace)
            npt.assert_array_equal(read_value, frame)

    def testReadChunk(self):
        """
        direct chunk reading
        """
        xdim = 867
        ydim = 700
        nframe = 33

        space = Simple((0, xdim, ydim),
                       (h5cpp.dataspace.UNLIMITED,
                        h5cpp.dataspace.UNLIMITED,
                        h5cpp.dataspace.UNLIMITED))

        framespace = Hyperslab(
            offset=(0, 0, 0),
            block=(1, xdim, ydim)
        )

        link_creation_list = LinkCreationList()

        dataset_creation_list = DatasetCreationList()
        dataset_creation_list.layout = DatasetLayout.CHUNKED
        dataset_creation_list.chunk = (1, xdim, ydim)

        dataset_access_list = DatasetAccessList()

        data2 = Dataset(
            self.root, h5cpp.Path("data2"), kUInt16, space,
            link_creation_list, dataset_creation_list,
            dataset_access_list)

        frame = numpy.random.randint(
            0, 65535, size=(xdim * ydim,), dtype="uint16")

        for i in range(nframe):
            data2.extent(0, 1)
            framespace.offset([i, 0, 0])
            data2.write(frame, framespace)

        read_value = numpy.zeros(shape=[xdim * ydim], dtype="uint16")

        for i in range(nframe):
            filter_mask = data2.read_chunk(read_value, [i, 0, 0])
            npt.assert_array_equal(read_value, frame)
            self.assertEqual(filter_mask, 0)

    def testReadChunkDeflate(self):
        """
        direct chunk reading with a deflate filter
        """
        xdim = 17
        nframe = 33

        space = Simple((0, xdim),
                       (h5cpp.dataspace.UNLIMITED,
                        h5cpp.dataspace.UNLIMITED))

        framespace = Hyperslab(
            offset=(0, 0),
            block=(1, xdim)
        )

        dfilter = Deflate(2)
        link_creation_list = LinkCreationList()

        dataset_creation_list = DatasetCreationList()
        dataset_creation_list.layout = DatasetLayout.CHUNKED
        dataset_creation_list.chunk = (1, xdim)
        dfilter(dataset_creation_list)

        dataset_access_list = DatasetAccessList()

        data3 = Dataset(
            self.root, h5cpp.Path("data3"), kUInt16, space,
            link_creation_list, dataset_creation_list,
            dataset_access_list)

        sframe = numpy.array([i // 2 for i in range(xdim)], dtype="uint16")
        tframe = numpy.array([i // 4 for i in range(xdim)], dtype="uint16")

        for i in range(nframe):
            data3.extent(0, 1)
            framespace.offset([i, 0])
            if i % 2:
                data3.write(sframe, framespace)
            else:
                data3.write(tframe, framespace)

        tcxdim = data3.chunk_storage_size([0, 0])
        scxdim = data3.chunk_storage_size([1, 0])

        sread_value = numpy.zeros(shape=[scxdim // 2], dtype="uint16")
        tread_value = numpy.zeros(shape=[tcxdim // 2], dtype="uint16")
        scpvalue = numpy.array(
            [24184, 49677, 4609, 12288, 49156, 6832,
             65478, 44127, 151, 40580, 42322, 55254, 14696, 2563, 16640],
            dtype="uint16")

        tcpvalue = numpy.array(
            [24184, 24675, 128, 1606, 25608, 32866,
             26176, 2054, 24932, 0, 20993, 7424],
            dtype="uint16")

        for i in range(nframe):
            if i % 2:
                filter_mask = data3.read_chunk(sread_value, [i, 0])
                npt.assert_array_equal(sread_value, scpvalue)
                self.assertEqual(filter_mask, 0)
            else:
                filter_mask = data3.read_chunk(tread_value, [i, 0])
                npt.assert_array_equal(tread_value, tcpvalue)
                self.assertEqual(filter_mask, 0)

    def testWriteChunkDeflate(self):
        """
        direct chunk writing with a deflate filter
        """
        xdim = 17
        nframe = 33

        space = Simple((0, xdim),
                       (h5cpp.dataspace.UNLIMITED,
                        h5cpp.dataspace.UNLIMITED))

        framespace = Hyperslab(
            offset=(0, 0),
            block=(1, xdim))

        dfilter = Deflate(2)

        link_creation_list = LinkCreationList()

        dataset_creation_list = DatasetCreationList()
        dataset_creation_list.layout = DatasetLayout.CHUNKED
        dataset_creation_list.chunk = (1, xdim)
        dfilter(dataset_creation_list)

        dataset_access_list = DatasetAccessList()

        data4 = Dataset(
            self.root, h5cpp.Path("data4"), kUInt16, space,
            link_creation_list, dataset_creation_list,
            dataset_access_list)

        scpframe = numpy.array(
            [24184, 49677, 4609, 12288, 49156, 6832,
             65478, 44127, 151, 40580, 42322, 55254, 14696, 2563, 16640],
            dtype="uint16")

        tcpframe = numpy.array(
            [24184, 24675, 128, 1606, 25608, 32866,
             26176, 2054, 24932, 0, 20993, 7424],
            dtype="uint16")

        for i in range(nframe):
            data4.extent(0, 1)
            if i % 2:
                data4.write_chunk(scpframe, [i, 0])
            else:
                data4.write_chunk(tcpframe, [i, 0])

        read_value = numpy.zeros(shape=[xdim], dtype="uint16")
        sframe = numpy.array([i // 2 for i in range(xdim)], dtype="uint16")
        tframe = numpy.array([i // 4 for i in range(xdim)], dtype="uint16")

        for i in range(nframe):
            framespace.offset([i, 0])
            data4.read(read_value, framespace)
            if i % 2:
                npt.assert_array_equal(read_value, sframe)
            else:
                npt.assert_array_equal(read_value, tframe)
        filters = data4.filters()
        self.assertEqual(len(filters), 1)
        self.assertEqual(filters[0].cd_values, [2])
        self.assertEqual(filters[0].id, 1)
        self.assertEqual(filters[0].name, "deflate")


if not HDF5GE102:
    del DatasetDirectChunkTests.testReadChunkDeflate
    del DatasetDirectChunkTests.testReadChunk
