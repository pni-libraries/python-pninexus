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
# Created on: Jun 06, 2019
#     Author: Jan Kontanski <jan.kotanski@desy.de>
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
from pninexus.h5cpp.dataspace import View
from pninexus.h5cpp.datatype import kInt32
from pninexus.h5cpp.datatype import String
from pninexus.h5cpp.property import LinkCreationList
from pninexus.h5cpp.property import DatasetCreationList
from pninexus.h5cpp.property import DatasetLayout
import numpy
import numpy.testing as npt
try:
    from pninexus.h5cpp.node import VirtualDataset
    from pninexus.h5cpp.property import VirtualDataMap
    from pninexus.h5cpp.property import VirtualDataMaps
    VDSAvailable = True
except Exception:
    VDSAvailable = False


module_path = os.path.dirname(os.path.abspath(__file__))

kmodulesize = 30

if VDSAvailable:

    class VDSDatasetPartialIOTests(unittest.TestCase):

        filename = os.path.join(module_path, "VirtualDatasetTests.h5")
        filename2 = os.path.join(module_path, "VirtualDatasetTests_inter.h5")
        vds1name = os.path.join(module_path, "VirtualDataset1Tests.h5")
        vds2name = os.path.join(module_path, "VirtualDataset2Tests.h5")
        vds3name = os.path.join(module_path, "VirtualDataset3Tests.h5")

        @classmethod
        def setUpClass(cls):
            super(VDSDatasetPartialIOTests, cls).setUpClass()

            h5cpp.file.create(cls.filename, AccessFlags.TRUNCATE)
            h5cpp.file.create(cls.filename2, AccessFlags.TRUNCATE)

        @classmethod
        def createSource(cls, fname, data):
            fl = h5cpp.file.create(fname, AccessFlags.TRUNCATE)
            root = fl.root()
            dataset = Dataset(root,
                              h5cpp.Path("module_data"),
                              h5cpp.datatype.kInt32,
                              Simple(tuple(data.shape)))
            dataset.write(data)
            dataset.close()
            root.close()
            fl.close()

        def setUp(self):

            self.datamodule1 = numpy.array([1] * kmodulesize)
            self.datamodule2 = numpy.array([2] * kmodulesize)
            self.datamodule3 = numpy.array([3] * kmodulesize)
            self.datamodule8 = numpy.array([8] * kmodulesize)
            self.createSource(self.vds1name, self.datamodule1)
            self.createSource(self.vds2name, self.datamodule2)
            self.createSource(self.vds3name, self.datamodule3)

        def tearDown(self):
            unittest.TestCase.tearDown(self)

            self.root.close()
            self.file.close()

        def testConcatenation(self):

            self.file = h5cpp.file.open(self.filename, AccessFlags.READWRITE)
            self.root = self.file.root()
            dataspace = Simple((3, kmodulesize))

            vdsmap = VirtualDataMaps()
            vdsmap.add(VirtualDataMap(
                View(dataspace,
                     Hyperslab(offset=(0, 0), block=(1, kmodulesize))),
                self.vds1name,
                h5cpp.Path("/module_data"),
                View(Simple(tuple([kmodulesize])))))
            vdsmap.add(VirtualDataMap(
                View(dataspace,
                     Hyperslab(offset=(1, 0), block=(1, kmodulesize))),
                self.vds2name,
                h5cpp.Path("/module_data"),
                View(Simple(tuple([kmodulesize])))))
            vdsmap.add(VirtualDataMap(
                View(dataspace,
                     Hyperslab(offset=(2, 0), block=(1, kmodulesize))),
                self.vds3name,
                h5cpp.Path("/module_data"),
                View(Simple(tuple([kmodulesize])))))

            dataset = VirtualDataset(
                self.root,
                h5cpp.Path("concatenation"), kInt32, dataspace, vdsmap)

            #
            # read data back
            #
            selection = Hyperslab(offset=(0, 0), block=(3, kmodulesize))
            selection.offset(0, 0)
            allmod = dataset.read(selection=selection)

            selection = Hyperslab(offset=(0, 0), block=(1, kmodulesize))
            selection.offset(0, 0)
            mod1 = dataset.read(selection=selection)
            npt.assert_array_equal(mod1, self.datamodule1)
            npt.assert_array_equal(allmod[0, :], self.datamodule1)
            # selection = Hyperslab(offset=(1, 0), block=(1, kmodulesize))
            # selection.offset(0, 1)
            # dimension offset
            selection.offset((1, 0))
            mod2 = dataset.read(selection=selection)
            npt.assert_array_equal(mod2, self.datamodule2)
            npt.assert_array_equal(allmod[1, :], self.datamodule2)
            # selection = Hyperslab(offset=(2, 0), block=(1, kmodulesize))
            # selection.offset((2, 0))
            # individual index offset
            selection.offset(0, 2)
            mod3 = dataset.read(selection=selection)
            npt.assert_array_equal(mod3, self.datamodule3)
            npt.assert_array_equal(allmod[2, :], self.datamodule3)

        def testGap(self):

            self.file = h5cpp.file.open(self.filename, AccessFlags.READWRITE)
            self.root = self.file.root()
            dataspace = Simple((5, kmodulesize))

            vdsmap = VirtualDataMaps()
            vdsmap.add(VirtualDataMap(
                View(dataspace,
                     Hyperslab(offset=(0, 0), block=(1, kmodulesize))),
                self.vds1name,
                h5cpp.Path("/module_data"),
                View(Simple(tuple([kmodulesize])))))
            vdsmap.add(VirtualDataMap(
                View(dataspace,
                     Hyperslab(offset=(2, 0), block=(1, kmodulesize))),
                self.vds2name,
                h5cpp.Path("/module_data"),
                View(Simple(tuple([kmodulesize])))))
            vdsmap.add(VirtualDataMap(
                View(dataspace,
                     Hyperslab(offset=(4, 0), block=(1, kmodulesize))),
                self.vds3name,
                h5cpp.Path("/module_data"),
                View(Simple(tuple([kmodulesize])))))

            dcpl = DatasetCreationList()
            self.assertEqual(
                dcpl.fill_value_status,
                h5cpp.property.DatasetFillValueStatus.DEFAULT)
            dcpl.set_fill_value(8, kInt32)
            self.assertEqual(dcpl.fill_value(kInt32), 8)
            self.assertEqual(
                dcpl.fill_value_status,
                h5cpp.property.DatasetFillValueStatus.USER_DEFINED)
            dataset = VirtualDataset(
                self.root,
                h5cpp.Path("gap"), kInt32, dataspace, vdsmap, dcpl=dcpl)

            #
            # read data back
            #
            self.assertEqual(dataset.dataspace.size, 150)

            selection = Hyperslab(offset=(0, 0), block=(5, kmodulesize))
            selection.offset(0, 0)
            allmod = dataset.read(selection=selection)
            self.assertEqual(allmod.shape, (5, kmodulesize))

            selection = Hyperslab(offset=(0, 0), block=(1, kmodulesize))
            selection.offset((0, 0))
            mod1 = dataset.read(selection=selection)
            npt.assert_array_equal(mod1, self.datamodule1)
            npt.assert_array_equal(
                allmod[0, :], self.datamodule1)

            selection.offset((1, 0))
            mod8 = dataset.read(selection=selection)
            npt.assert_array_equal(mod8, self.datamodule8)
            npt.assert_array_equal(allmod[1, :],
                                   self.datamodule8)
            selection.offset((2, 0))
            mod2 = dataset.read(selection=selection)
            npt.assert_array_equal(mod2, self.datamodule2)
            npt.assert_array_equal(allmod[2, :],
                                   self.datamodule2)
            selection.offset((3, 0))
            mod8 = dataset.read(selection=selection)
            npt.assert_array_equal(mod8, self.datamodule8)
            npt.assert_array_equal(allmod[3, :],
                                   self.datamodule8)
            selection.offset((4, 0))
            mod3 = dataset.read(selection=selection)
            npt.assert_array_equal(mod3, self.datamodule3)
            npt.assert_array_equal(allmod[4, :], self.datamodule3)

        def testInterleaving(self):

            self.file = h5cpp.file.open(self.filename2, AccessFlags.READWRITE)
            self.root = self.file.root()
            dataspace = Simple((3 * kmodulesize, ))

            vdsmap = VirtualDataMaps()
            vdsmap.add(VirtualDataMap(
                View(dataspace,
                     Hyperslab(offset=(0,), block=(1,),
                               count=(kmodulesize,), stride=(3,))),
                self.vds1name,
                h5cpp.Path("/module_data"),
                View(Simple(tuple([kmodulesize])))))
            vdsmap.add(VirtualDataMap(
                View(dataspace,
                     Hyperslab(offset=(1,), block=(1,),
                               count=(kmodulesize,), stride=(3,))),
                self.vds2name,
                h5cpp.Path("/module_data"),
                View(Simple(tuple([kmodulesize])))))
            vdsmap.add(VirtualDataMap(
                View(dataspace,
                     Hyperslab(offset=(2,), block=(1,),
                               count=(kmodulesize,), stride=(3,))),
                self.vds3name,
                h5cpp.Path("/module_data"),
                View(Simple(tuple([kmodulesize])))))

            dataset = VirtualDataset(
                self.root,
                h5cpp.Path("all"), kInt32, dataspace, vdsmap)

            #
            # read data back
            #
            self.assertEqual(dataset.dataspace.size, 90)

            allmod = dataset.read()

            refdata = numpy.array([1, 2, 3])
            for offset in range(0, kmodulesize * 3, 3):
                selection = Hyperslab(offset=(0,), block=(3,))
                mod1 = dataset.read(selection=selection)
                npt.assert_array_equal(mod1, refdata)
                npt.assert_array_equal(mod1, allmod[offset: (offset + 3)])
