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
from pninexus.h5cpp.dataspace import Scalar
import numpy
import numpy.testing as npt


module_path = os.path.dirname(os.path.abspath(__file__))


class DatasetAllIOTests(unittest.TestCase):

    filename = os.path.join(module_path, "DatasetAllIOTests.h5")

    @classmethod
    def setUpClass(cls):
        super(DatasetAllIOTests, cls).setUpClass()

        h5cpp.file.create(cls.filename, AccessFlags.TRUNCATE)

    def setUp(self):

        self.file = h5cpp.file.open(self.filename, AccessFlags.READWRITE)
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
        self.assertEqual(read, 42)

    def testWriteFloatScalar(self):

        dataset = Dataset(self.root, h5cpp.Path("FloatScalar"),
                          h5cpp.datatype.kFloat32, Scalar())
        dataset.write(23.321)
        read = dataset.read()
        self.assertEqual(read, 23.321)

    def testWriteEBooleanScalar(self):

        dataset = Dataset(self.root, h5cpp.Path("EBoolScalar"),
                          h5cpp.datatype.kEBool, Scalar())
        dataset2 = Dataset(self.root, h5cpp.Path("EBoolScalar2"),
                           h5cpp.datatype.kEBool, Scalar())
        dataset.write(True)
        self.assertTrue(dataset.read())
        dataset2.write(False)
        self.assertTrue(not dataset2.read())

    def testWriteEBoolArray(self):

        data = [True, False, False, True]
        dataset = Dataset(self.root, h5cpp.Path("EBoolArray"),
                          h5cpp.datatype.kFloat64, Simple((4, )))
        dataset.write(data)
        read = dataset.read()
        npt.assert_array_equal(read, numpy.array(data))

    def testWriteFixedLengthStringScalar(self):

        data = "hello world"
        dtype = h5cpp.datatype.String.fixed(len(data))
        dtype.padding = h5cpp.datatype.StringPad.NULLPAD
        dataset = Dataset(
            self.root, h5cpp.Path("FixedLengthStringScalar"), dtype, Scalar())
        dataset.write(data)
        read = dataset.read()
        self.assertEqual(read, "hello world")

    def testWriteVariableLengthScalar(self):
        data = "hello world"
        dtype = h5cpp.datatype.String.variable()
        dataset = Dataset(
            self.root, h5cpp.Path("VariableLengthStringScalar"),
            dtype, Scalar())
        dataset.write(data)
        read = dataset.read()
        self.assertEqual(read, "hello world")

    def testWriteIntegerArray(self):

        data = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        dataset = Dataset(self.root, h5cpp.Path("IntegerArray"),
                          h5cpp.datatype.kInt32, Simple((2, 4)))
        dataset.write(data)
        read = dataset.read()
        npt.assert_array_equal(read, data)

        #
        # test inplace reading
        #
        read = numpy.zeros(dataset.dataspace.current_dimensions, dtype="int64")
        dataset.read(read)
        npt.assert_array_equal(read, data)

        read = numpy.zeros((2, 2), dtype="float32")
        self.assertRaises(RuntimeError, dataset.read, read)

    def testWriteFloatArray(self):

        data = numpy.array(
            [[1, 2, 3, 4], [5, 6, 7, 8]], dtype="float64")
        dataset = Dataset(self.root, h5cpp.Path("FloatArray"),
                          h5cpp.datatype.kFloat64, Simple((2, 4)))
        dataset.write(data)
        read = dataset.read()
        npt.assert_array_equal(read, data)

    def testWriteFloat16Array(self):

        data = numpy.array(
            [[1, 2, 3, 4], [5, 6, 7, 8]], dtype="float16")
        dataset = Dataset(self.root, h5cpp.Path("Float16Array"),
                          h5cpp.datatype.kFloat16, Simple((2, 4)))
        dataset.write(data)
        read = dataset.read()
        npt.assert_array_equal(read, data)

    def testWriteComplex32Array(self):

        data = numpy.array(
            [[1 + 1j, 2 - 1j, 3 + 2j, 4 - 2j],
             [5 + 3j, 6 - 3j, 7 + 4j, 8 - 4j]], dtype="complex64")
        dataset = Dataset(self.root, h5cpp.Path("Complex32Array"),
                          h5cpp.datatype.kComplex64, Simple((2, 4)))
        dataset.write(data)
        read = dataset.read()
        npt.assert_array_equal(read, data)

    def testWriteComplex64Array(self):

        data = numpy.array(
            [[1 + 1j, 2 - 1j, 3 + 2j, 4 - 2j],
             [5 + 3j, 6 - 3j, 7 + 4j, 8 - 4j]], dtype="complex64")
        dataset = Dataset(self.root, h5cpp.Path("Complex64Array"),
                          h5cpp.datatype.kComplex64, Simple((2, 4)))
        dataset.write(data)
        read = dataset.read()
        npt.assert_array_equal(read, data)

    def testWriteComplex128Array(self):

        data = numpy.array(
            [[1 + 1j, 2 - 1j, 3 + 2j, 4 - 2j],
             [5 + 3j, 6 - 3j, 7 + 4j, 8 - 4j]], dtype="complex64")
        dataset = Dataset(self.root, h5cpp.Path("Complex128Array"),
                          h5cpp.datatype.kComplex64, Simple((2, 4)))
        dataset.write(data)
        read = dataset.read()
        npt.assert_array_equal(read, data)

    def testWriteComplex256Array(self):

        data = numpy.array(
            [[1 + 1j, 2 - 1j, 3 + 2j, 4 - 2j],
             [5 + 3j, 6 - 3j, 7 + 4j, 8 - 4j]], dtype="complex64")
        dataset = Dataset(self.root, h5cpp.Path("Complex256Array"),
                          h5cpp.datatype.kComplex64, Simple((2, 4)))
        dataset.write(data)
        read = dataset.read()
        npt.assert_array_equal(read, data)

    def testWriteFixedLengthStringArray(self):

        data = numpy.array(
            [["hello", "world", "this"], ["is", "a", "test"]])
        dtype = h5cpp.datatype.String.fixed(5)
        dtype.padding = h5cpp.datatype.StringPad.NULLPAD
        dataset = Dataset(self.root, h5cpp.Path("FixedLengthStringArray"),
                          dtype,
                          Simple((2, 3)))
        dataset.write(data)
        read = dataset.read()
        npt.assert_array_equal(read, data)

    def testWriteVariableLengthStringArray(self):

        data = numpy.array(
            [["hello", "world", "this"], ["is", "a", "test"]])
        dtype = h5cpp.datatype.String.variable()
        dataset = Dataset(self.root, h5cpp.Path("VariableLengthStringArray"),
                          dtype,
                          Simple((2, 3)))
        dataset.write(data)
        read = dataset.read()
        npt.assert_array_equal(read, data)
