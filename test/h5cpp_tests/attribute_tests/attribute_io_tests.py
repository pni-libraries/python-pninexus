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
# Created on: Jan 30, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import numpy
import os
import numpy.testing as npt
from pninexus import h5cpp
from pninexus.h5cpp.file import AccessFlags
from pninexus.h5cpp.datatype import (
    kInt32, kFloat32, kUInt8, kVariableString, kEBool)
from pninexus.h5cpp.datatype import StringPad, String


module_path = os.path.dirname(os.path.abspath(__file__))


class AttributeIOTests(unittest.TestCase):

    filename = os.path.join(module_path, "AttributeIOTests.h5")

    @classmethod
    def setUpClass(cls):
        super(AttributeIOTests, cls).setUpClass()

        h5cpp.file.create(cls.filename, AccessFlags.TRUNCATE)

    def setUp(self):
        self.file = h5cpp.file.open(self.filename, AccessFlags.READWRITE)
        self.root = self.file.root()

    def tearDown(self):
        self.root.close()
        self.file.close()

    def testBooleanScalar(self):

        a = self.root.attributes.create("BooleanScalar", kUInt8)
        a.write(True)
        self.assertTrue(a.read())
        a2 = self.root.attributes.create("BooleanScalar2", kUInt8)
        a2.write(False)
        self.assertTrue(not a2.read())

    def testBooleanArray(self):

        a = self.root.attributes.create("BooleanArray", kUInt8, (4,))
        data = [True, False, False, True]
        a.write(data)
        npt.assert_array_equal(a.read(), numpy.array(data))

    def testEBooleanScalar(self):

        a = self.root.attributes.create("EBooleanScalar", kEBool)
        a2 = self.root.attributes.create("EBooleanScalar2", kEBool)
        a.write(True)
        self.assertTrue(a.read())
        a2.write(False)
        self.assertTrue(not a2.read())

    def testEBooleanArray(self):

        a = self.root.attributes.create("EBooleanArray", kEBool, (4,))
        data = [True, False, False, True]
        a.write(data)
        npt.assert_array_equal(a.read(), numpy.array(data))

    def testIntegerScalar(self):

        a = self.root.attributes.create("IntegerScalar", kInt32)
        a2 = self.root.attributes.create("IntegerScalar2", kInt32)
        a.write(42)
        r = a.read()
        self.assertEqual(r, 42)
        a2.write(12)
        r = a2.read()
        self.assertEqual(r, 12)

    def testFloatScalar(self):

        a = self.root.attributes.create("FloatScalar", kFloat32)
        a.write(42.345)
        r = a.read()
        self.assertEqual(r, 42.345)

    def testFloatNumpyScalar(self):

        a = self.root.attributes.create("FloatNumpyScalar", kFloat32)
        a.write(numpy.float32(34.5323))
        r = a.read()
        self.assertEqual(r, 34.5323)

    def testStringScalarFixedLength(self):

        data = "hello world"
        dtype = String.fixed(len(data))
        dtype.padding = StringPad.NULLPAD
        a = self.root.attributes.create("StringScalar", dtype)
        a.write("hello world")
        r = a.read()
        self.assertEqual(r, "hello world")

    def testStringScalarVariableLength(self):

        data = "hello world"
        a = self.root.attributes.create("StringScalarVLength", kVariableString)
        a.write(data)
        r = a.read()
        self.assertEqual(r, data)

    def testStringArray(self):

        data = numpy.array([["hello", "world", "this"], ["is", "a", "test"]])
        dtype = String.fixed(5)
        dtype.padding = StringPad.NULLPAD
        a = self.root.attributes.create("StringArray", dtype, (2, 3))
        a.write(data)
        r = a.read()
        npt.assert_array_equal(r, data)

    def testStringArrayVariableLength(self):

        data = numpy.array([["hello", "world", "this"], ["is", "a", "test"]])
        a = self.root.attributes.create(
            "StringArrayVLength", kVariableString, (2, 3))
        a.write(data)
        r = a.read()
        npt.assert_array_equal(r, data)

    def testIntArray(self):

        data = numpy.array([1, 2, 3])
        a = self.root.attributes.create("IntArray", kInt32, (3,))
        a.write(data)
        r = a.read()
        npt.assert_array_equal(r, data)
