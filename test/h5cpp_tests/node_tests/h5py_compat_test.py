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
from pninexus import h5cpp
from pninexus.h5cpp.file import AccessFlags
# from pninexus.h5cpp.node import Dataset
# from pninexus.h5cpp.dataspace import Simple
# from pninexus.h5cpp.dataspace import Scalar
import numpy
import numpy.testing as npt
import h5py


class H5pyCompatibilityReading(unittest.TestCase):
    """
    Compatability test for reading data generated with h5py.
    There are basically two datatypes which require special care

    * strings
    * boolean values

    """

    filename = "H5pyCompatibilityReading.h5"

    str_data = numpy.array(
        [["hello", "world"], ["a", "text"], ["array", "!"]])

    @classmethod
    def setUpClass(cls):
        super(H5pyCompatibilityReading, cls).setUpClass()

        f = h5py.File(cls.filename, "w")
        f.create_dataset(
            "FixedLengthStringData",
            cls.str_data.shape, data=cls.str_data.astype("S"), dtype="S10")

        dt = h5py.special_dtype(vlen=bytes)
        f.create_dataset(
            "VariableLengthStringData",
            cls.str_data.shape, data=cls.str_data.astype("S"), dtype=dt)
        f.close()

    def setUp(self):

        self.file = h5cpp.file.open(self.filename, AccessFlags.READONLY)
        self.root = self.file.root()

    def tearDown(self):
        self.root.close()
        self.file.close()

    def testReadingFixedLengthStringData(self):

        dataset = self.root.nodes["FixedLengthStringData"]
        read = dataset.read()
        npt.assert_array_equal(read, self.str_data)

    def testReadingVariableLengthStringData(self):

        dataset = self.root.nodes["VariableLengthStringData"]
        read = dataset.read()
        npt.assert_array_equal(read, self.str_data)
