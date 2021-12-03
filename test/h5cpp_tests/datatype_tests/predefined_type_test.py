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
# Created on: Jan 29, 2018
#     Authors:
#             Eugen Wintersberger <eugen.wintersberger@desy.de>
#             Jan Kotanski <jan.kotanski@desy.de>
#
from __future__ import print_function
import unittest
from pninexus import h5cpp
from pninexus.h5cpp.datatype import Float, String, Datatype, Integer, Enum


class PredefinedTypeTests(unittest.TestCase):

    enum_types = (Datatype, Enum)
    float_types = (Datatype, Float)
    int_types = (Datatype, Integer)
    string_types = (Datatype, String)

    def testUInt8(self):

        dtype = h5cpp.datatype.Integer(h5cpp.datatype.kUInt8)
        self.assertTrue(isinstance(dtype, self.int_types))
        self.assertEqual(dtype.size, 1)
        self.assertEqual(dtype.is_signed(), False)

        dtype.make_signed(True)
        self.assertEqual(dtype.is_signed(), True)
        dtype.make_signed(False)
        self.assertEqual(dtype.is_signed(), False)

        self.assertEqual(dtype.precision, 8)
        dtype.precision = 16
        self.assertEqual(dtype.precision, 16)
        dtype.precision = 8
        self.assertEqual(dtype.precision, 8)

        self.assertEqual(dtype.offset, 0)
        dtype.offset = 2
        self.assertEqual(dtype.offset, 2)
        dtype.offset = 0
        self.assertEqual(dtype.offset, 0)

        order = dtype.order
        self.assertTrue(dtype.order in [h5cpp.datatype.Order.LE,
                                        h5cpp.datatype.Order.BE])
        dtype.order = h5cpp.datatype.Order.BE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.BE)
        dtype.order = h5cpp.datatype.Order.LE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.LE)
        dtype.order = order

        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        dtype.pad = (h5cpp.datatype.Pad.ONE,
                     h5cpp.datatype.Pad.BACKGROUND)
        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ONE)
        self.assertEqual(mp, h5cpp.datatype.Pad.BACKGROUND)
        dtype.pad = [h5cpp.datatype.Pad.ZERO,
                     h5cpp.datatype.Pad.ZERO]

    def testInt8(self):

        dtype = h5cpp.datatype.Integer(h5cpp.datatype.kInt8)
        self.assertTrue(isinstance(dtype, self.int_types))
        self.assertEqual(dtype.is_signed(), True)

        dtype.make_signed(False)
        self.assertEqual(dtype.is_signed(), False)
        dtype.make_signed(True)
        self.assertEqual(dtype.is_signed(), True)

        self.assertEqual(dtype.precision, 8)
        dtype.precision = 16
        self.assertEqual(dtype.precision, 16)
        dtype.precision = 8
        self.assertEqual(dtype.precision, 8)

        self.assertEqual(dtype.offset, 0)
        dtype.offset = 2
        self.assertEqual(dtype.offset, 2)
        dtype.offset = 0
        self.assertEqual(dtype.offset, 0)

        order = dtype.order
        self.assertTrue(dtype.order in [h5cpp.datatype.Order.LE,
                                        h5cpp.datatype.Order.BE])
        dtype.order = h5cpp.datatype.Order.BE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.BE)
        dtype.order = h5cpp.datatype.Order.LE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.LE)
        dtype.order = order

        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        dtype.pad = (h5cpp.datatype.Pad.ONE,
                     h5cpp.datatype.Pad.BACKGROUND)
        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ONE)
        self.assertEqual(mp, h5cpp.datatype.Pad.BACKGROUND)
        dtype.pad = [h5cpp.datatype.Pad.ZERO,
                     h5cpp.datatype.Pad.ZERO]

    def testUInt16(self):

        dtype = h5cpp.datatype.Integer(h5cpp.datatype.kUInt16)
        self.assertTrue(isinstance(dtype, self.int_types))
        self.assertEqual(dtype.size, 2)
        self.assertEqual(dtype.is_signed(), False)

        dtype.make_signed(True)
        self.assertEqual(dtype.is_signed(), True)

        self.assertEqual(dtype.precision, 16)
        dtype.precision = 12
        self.assertEqual(dtype.precision, 12)
        dtype.precision = 16
        self.assertEqual(dtype.precision, 16)

        self.assertEqual(dtype.offset, 0)
        dtype.offset = 2
        self.assertEqual(dtype.offset, 2)
        dtype.offset = 0
        self.assertEqual(dtype.offset, 0)

        order = dtype.order
        self.assertTrue(dtype.order in [h5cpp.datatype.Order.LE,
                                        h5cpp.datatype.Order.BE])
        dtype.order = h5cpp.datatype.Order.BE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.BE)
        dtype.order = h5cpp.datatype.Order.LE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.LE)
        dtype.order = order

        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        dtype.pad = [h5cpp.datatype.Pad.ONE,
                     h5cpp.datatype.Pad.BACKGROUND]
        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ONE)
        self.assertEqual(mp, h5cpp.datatype.Pad.BACKGROUND)
        dtype.pad = (h5cpp.datatype.Pad.ZERO,
                     h5cpp.datatype.Pad.ZERO)

    def testInt16(self):

        dtype = h5cpp.datatype.Integer(h5cpp.datatype.kInt16)
        self.assertTrue(isinstance(dtype, self.int_types))
        self.assertEqual(dtype.size, 2)
        self.assertEqual(dtype.is_signed(), True)

        dtype.make_signed(False)
        self.assertEqual(dtype.is_signed(), False)

        self.assertEqual(dtype.precision, 16)
        dtype.precision = 12
        self.assertEqual(dtype.precision, 12)
        dtype.precision = 16
        self.assertEqual(dtype.precision, 16)

        self.assertEqual(dtype.offset, 0)
        dtype.offset = 2
        self.assertEqual(dtype.offset, 2)
        dtype.offset = 0
        self.assertEqual(dtype.offset, 0)

        order = dtype.order
        self.assertTrue(dtype.order in [h5cpp.datatype.Order.LE,
                                        h5cpp.datatype.Order.BE])
        dtype.order = h5cpp.datatype.Order.BE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.BE)
        dtype.order = h5cpp.datatype.Order.LE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.LE)
        dtype.order = order

        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        dtype.pad = (h5cpp.datatype.Pad.ONE,
                     h5cpp.datatype.Pad.BACKGROUND)
        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ONE)
        self.assertEqual(mp, h5cpp.datatype.Pad.BACKGROUND)
        dtype.pad = (h5cpp.datatype.Pad.ZERO,
                     h5cpp.datatype.Pad.ZERO)

    def testUInt32(self):

        dtype = h5cpp.datatype.Integer(h5cpp.datatype.kUInt32)
        self.assertTrue(isinstance(dtype, self.int_types))
        self.assertEqual(dtype.size, 4)
        self.assertEqual(dtype.is_signed(), False)

        dtype.make_signed(True)
        self.assertEqual(dtype.is_signed(), True)

        self.assertEqual(dtype.precision, 32)
        dtype.precision = 16
        self.assertEqual(dtype.precision, 16)
        dtype.precision = 32
        self.assertEqual(dtype.precision, 32)

        self.assertEqual(dtype.offset, 0)
        dtype.offset = 2
        self.assertEqual(dtype.offset, 2)
        dtype.offset = 0
        self.assertEqual(dtype.offset, 0)

        order = dtype.order
        self.assertTrue(dtype.order in [h5cpp.datatype.Order.LE,
                                        h5cpp.datatype.Order.BE])
        dtype.order = h5cpp.datatype.Order.BE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.BE)
        dtype.order = h5cpp.datatype.Order.LE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.LE)
        dtype.order = order

        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        dtype.pad = (h5cpp.datatype.Pad.ONE,
                     h5cpp.datatype.Pad.BACKGROUND)
        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ONE)
        self.assertEqual(mp, h5cpp.datatype.Pad.BACKGROUND)
        dtype.pad = (h5cpp.datatype.Pad.ZERO,
                     h5cpp.datatype.Pad.ZERO)

    def testInt32(self):

        dtype = h5cpp.datatype.Integer(h5cpp.datatype.kInt32)
        self.assertTrue(isinstance(dtype, self.int_types))
        self.assertEqual(dtype.size, 4)
        self.assertEqual(dtype.is_signed(), True)

        dtype.make_signed(False)
        self.assertEqual(dtype.is_signed(), False)

        self.assertEqual(dtype.precision, 32)
        dtype.precision = 16
        self.assertEqual(dtype.precision, 16)
        dtype.precision = 32
        self.assertEqual(dtype.precision, 32)

        self.assertEqual(dtype.offset, 0)
        dtype.offset = 2
        self.assertEqual(dtype.offset, 2)
        dtype.offset = 0
        self.assertEqual(dtype.offset, 0)

        order = dtype.order
        self.assertTrue(dtype.order in [h5cpp.datatype.Order.LE,
                                        h5cpp.datatype.Order.BE])
        dtype.order = h5cpp.datatype.Order.BE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.BE)
        dtype.order = h5cpp.datatype.Order.LE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.LE)
        dtype.order = order

        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        dtype.pad = (h5cpp.datatype.Pad.ONE,
                     h5cpp.datatype.Pad.BACKGROUND)
        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ONE)
        self.assertEqual(mp, h5cpp.datatype.Pad.BACKGROUND)
        dtype.pad = (h5cpp.datatype.Pad.ZERO,
                     h5cpp.datatype.Pad.ZERO)

    def testUInt64(self):

        dtype = h5cpp.datatype.Integer(h5cpp.datatype.kUInt64)
        self.assertTrue(isinstance(dtype, self.int_types))
        self.assertEqual(dtype.size, 8)
        self.assertEqual(dtype.is_signed(), False)

        dtype.make_signed(True)
        self.assertEqual(dtype.is_signed(), True)

        self.assertEqual(dtype.precision, 64)
        dtype.precision = 32
        self.assertEqual(dtype.precision, 32)
        dtype.precision = 64
        self.assertEqual(dtype.precision, 64)

        self.assertEqual(dtype.offset, 0)
        dtype.offset = 2
        self.assertEqual(dtype.offset, 2)
        dtype.offset = 0
        self.assertEqual(dtype.offset, 0)

        order = dtype.order
        self.assertTrue(dtype.order in [h5cpp.datatype.Order.LE,
                                        h5cpp.datatype.Order.BE])
        dtype.order = h5cpp.datatype.Order.BE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.BE)
        dtype.order = h5cpp.datatype.Order.LE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.LE)
        dtype.order = order

        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        dtype.pad = (h5cpp.datatype.Pad.ONE,
                     h5cpp.datatype.Pad.BACKGROUND)
        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ONE)
        self.assertEqual(mp, h5cpp.datatype.Pad.BACKGROUND)
        dtype.pad = (h5cpp.datatype.Pad.ZERO,
                     h5cpp.datatype.Pad.ZERO)

    def testInt64(self):

        dtype = h5cpp.datatype.Integer(h5cpp.datatype.kInt64)
        self.assertTrue(isinstance(dtype, self.int_types))
        self.assertEqual(dtype.size, 8)
        self.assertEqual(dtype.is_signed(), True)

        dtype.make_signed(False)
        self.assertEqual(dtype.is_signed(), False)

        self.assertEqual(dtype.precision, 64)
        dtype.precision = 32
        self.assertEqual(dtype.precision, 32)
        dtype.precision = 64
        self.assertEqual(dtype.precision, 64)

        self.assertEqual(dtype.offset, 0)
        dtype.offset = 2
        self.assertEqual(dtype.offset, 2)
        dtype.offset = 0
        self.assertEqual(dtype.offset, 0)

        order = dtype.order
        self.assertTrue(dtype.order in [h5cpp.datatype.Order.LE,
                                        h5cpp.datatype.Order.BE])
        dtype.order = h5cpp.datatype.Order.BE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.BE)
        dtype.order = h5cpp.datatype.Order.LE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.LE)
        dtype.order = order

        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        dtype.pad = (h5cpp.datatype.Pad.ONE,
                     h5cpp.datatype.Pad.BACKGROUND)
        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ONE)
        self.assertEqual(mp, h5cpp.datatype.Pad.BACKGROUND)
        dtype.pad = (h5cpp.datatype.Pad.ZERO,
                     h5cpp.datatype.Pad.ZERO)

    def testFloat32(self):

        dtype = h5cpp.datatype.Float(h5cpp.datatype.kFloat32)
        self.assertTrue(isinstance(dtype, self.float_types))
        self.assertEqual(dtype.size, 4)

        self.assertEqual(dtype.precision, 32)
        dtype.precision = 64
        self.assertEqual(dtype.precision, 64)
        dtype.precision = 32
        self.assertEqual(dtype.precision, 32)

        self.assertEqual(dtype.offset, 0)
        dtype.offset = 2
        self.assertEqual(dtype.offset, 2)
        dtype.offset = 0
        self.assertEqual(dtype.offset, 0)

        order = dtype.order
        self.assertTrue(dtype.order in [h5cpp.datatype.Order.LE,
                                        h5cpp.datatype.Order.BE])
        dtype.order = h5cpp.datatype.Order.BE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.BE)
        dtype.order = h5cpp.datatype.Order.LE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.LE)
        dtype.order = order

        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        dtype.pad = (h5cpp.datatype.Pad.ONE,
                     h5cpp.datatype.Pad.BACKGROUND)
        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ONE)
        self.assertEqual(mp, h5cpp.datatype.Pad.BACKGROUND)
        dtype.pad = [h5cpp.datatype.Pad.ZERO,
                     h5cpp.datatype.Pad.ZERO]

        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.ZERO)
        dtype.inpad = h5cpp.datatype.Pad.ONE
        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.ONE)
        dtype.inpad = h5cpp.datatype.Pad.BACKGROUND
        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.BACKGROUND)
        dtype.inpad = h5cpp.datatype.Pad.ZERO
        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.ZERO)

        norm = dtype.norm
        self.assertTrue(dtype.norm in [
            h5cpp.datatype.Norm.NONE, h5cpp.datatype.Norm.IMPLIED])
        dtype.norm = h5cpp.datatype.Norm.MSBSET
        self.assertEqual(dtype.norm, h5cpp.datatype.Norm.MSBSET)
        dtype.norm = h5cpp.datatype.Norm.NONE
        self.assertEqual(dtype.norm, h5cpp.datatype.Norm.NONE)
        dtype.norm = norm

        ebias = dtype.ebias

        dtype.size = 4
        sz = dtype.size
        self.assertEqual(dtype.size, 4)
        self.assertEqual(dtype.ebias, 127)
        self.assertTrue(dtype.ebias in [2 * sz * sz * sz - 1,
                                        4 * sz * sz * sz - 1])
        dtype.ebias = 63
        self.assertEqual(dtype.ebias, 63)
        dtype.ebias = 31
        self.assertEqual(dtype.ebias, 31)
        dtype.ebias = ebias

        fields = dtype.fields
        self.assertEqual(len(fields), 5)
        fields1 = (15, 10, 5, 0, 5)
        fields2 = (14, 9, 5, 0, 9)
        dtype.fields = fields1
        self.assertEqual(dtype.fields, fields1)
        dtype.fields = fields2
        self.assertEqual(dtype.fields, fields2)
        dtype.fields = fields

    def testFloat64(self):

        dtype = h5cpp.datatype.Float(h5cpp.datatype.kFloat64)
        self.assertTrue(isinstance(dtype, self.float_types))
        self.assertEqual(dtype.size, 8)

        self.assertEqual(dtype.precision, 64)
        dtype.precision = 80
        self.assertEqual(dtype.precision, 80)
        dtype.precision = 64
        self.assertEqual(dtype.precision, 64)

        self.assertEqual(dtype.offset, 0)
        dtype.offset = 2
        self.assertEqual(dtype.offset, 2)
        dtype.offset = 0
        self.assertEqual(dtype.offset, 0)

        order = dtype.order
        self.assertTrue(dtype.order in [h5cpp.datatype.Order.LE,
                                        h5cpp.datatype.Order.BE])
        dtype.order = h5cpp.datatype.Order.BE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.BE)
        dtype.order = h5cpp.datatype.Order.LE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.LE)
        dtype.order = order

        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        dtype.pad = (h5cpp.datatype.Pad.ONE,
                     h5cpp.datatype.Pad.BACKGROUND)
        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ONE)
        self.assertEqual(mp, h5cpp.datatype.Pad.BACKGROUND)
        dtype.pad = [h5cpp.datatype.Pad.ZERO,
                     h5cpp.datatype.Pad.ZERO]

        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.ZERO)
        dtype.inpad = h5cpp.datatype.Pad.ONE
        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.ONE)
        dtype.inpad = h5cpp.datatype.Pad.BACKGROUND
        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.BACKGROUND)
        dtype.inpad = h5cpp.datatype.Pad.ZERO
        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.ZERO)

        norm = dtype.norm
        self.assertTrue(dtype.norm in [
            h5cpp.datatype.Norm.NONE, h5cpp.datatype.Norm.IMPLIED])
        dtype.norm = h5cpp.datatype.Norm.MSBSET
        self.assertEqual(dtype.norm, h5cpp.datatype.Norm.MSBSET)
        dtype.norm = h5cpp.datatype.Norm.NONE
        self.assertEqual(dtype.norm, h5cpp.datatype.Norm.NONE)
        dtype.norm = norm

        ebias = dtype.ebias
        dtype.size = 8
        sz = dtype.size
        self.assertTrue(dtype.ebias in [2 * sz * sz * sz - 1,
                                        4 * sz * sz * sz - 1])
        dtype.ebias = 63
        self.assertEqual(dtype.ebias, 63)
        dtype.ebias = 31
        self.assertEqual(dtype.ebias, 31)
        dtype.ebias = ebias

        fields = dtype.fields
        self.assertEqual(len(fields), 5)
        fields1 = (15, 10, 5, 0, 5)
        fields2 = (14, 9, 5, 0, 9)
        dtype.fields = fields1
        self.assertEqual(dtype.fields, fields1)
        dtype.fields = fields2
        self.assertEqual(dtype.fields, fields2)
        dtype.fields = fields

    def testFloat128(self):

        dtype = h5cpp.datatype.Float(h5cpp.datatype.kFloat128)
        self.assertTrue(isinstance(dtype, self.float_types))
        self.assertTrue(dtype.size in [8, 12, 16])

        prec = dtype.precision
        self.assertTrue(dtype.precision in [64, 80, 128])
        dtype.precision = 80
        self.assertEqual(dtype.precision, 80)
        dtype.precision = prec
        self.assertEqual(dtype.precision, prec)

        self.assertEqual(dtype.offset, 0)
        dtype.offset = 2
        self.assertEqual(dtype.offset, 2)
        dtype.offset = 0
        self.assertEqual(dtype.offset, 0)

        order = dtype.order
        self.assertTrue(dtype.order in [h5cpp.datatype.Order.LE,
                                        h5cpp.datatype.Order.BE])
        dtype.order = h5cpp.datatype.Order.BE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.BE)
        dtype.order = h5cpp.datatype.Order.LE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.LE)
        dtype.order = order

        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        dtype.pad = [h5cpp.datatype.Pad.ONE,
                     h5cpp.datatype.Pad.BACKGROUND]
        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ONE)
        self.assertEqual(mp, h5cpp.datatype.Pad.BACKGROUND)
        dtype.pad = (h5cpp.datatype.Pad.ZERO,
                     h5cpp.datatype.Pad.ZERO)

        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.ZERO)
        dtype.inpad = h5cpp.datatype.Pad.ONE
        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.ONE)
        dtype.inpad = h5cpp.datatype.Pad.BACKGROUND
        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.BACKGROUND)
        dtype.inpad = h5cpp.datatype.Pad.ZERO
        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.ZERO)

        norm = dtype.norm
        self.assertTrue(dtype.norm in [
            h5cpp.datatype.Norm.NONE, h5cpp.datatype.Norm.IMPLIED])
        dtype.norm = h5cpp.datatype.Norm.MSBSET
        self.assertEqual(dtype.norm, h5cpp.datatype.Norm.MSBSET)
        dtype.norm = h5cpp.datatype.Norm.NONE
        self.assertEqual(dtype.norm, h5cpp.datatype.Norm.NONE)
        dtype.norm = norm

        ebias = dtype.ebias
        sz = dtype.size
        self.assertTrue(dtype.ebias in [2 * sz * sz * sz - 1,
                                        4 * sz * sz * sz - 1])
        dtype.ebias = 63
        self.assertEqual(dtype.ebias, 63)
        dtype.ebias = 31
        self.assertEqual(dtype.ebias, 31)
        dtype.ebias = ebias

        fields = dtype.fields
        self.assertEqual(len(fields), 5)
        fields1 = (15, 10, 5, 0, 5)
        fields2 = (14, 9, 5, 0, 9)
        dtype.fields = fields1
        self.assertEqual(dtype.fields, fields1)
        dtype.fields = fields2
        self.assertEqual(dtype.fields, fields2)
        dtype.fields = fields

    def testVariableString(self):

        dtype = h5cpp.datatype.String(h5cpp.datatype.kVariableString)
        self.assertTrue(isinstance(dtype, self.string_types))
        self.assertTrue(dtype.is_variable_length)

    def testEBool(self):

        dtype = h5cpp.datatype.Enum(h5cpp.datatype.kEBool)
        self.assertTrue(isinstance(dtype, self.enum_types))
        self.assertTrue(h5cpp._datatype.is_bool(dtype))
        self.assertEqual(dtype.size, 1)

    def testFloat16(self):

        dtype = h5cpp.datatype.Float(h5cpp.datatype.kFloat16)
        self.assertTrue(isinstance(dtype, self.float_types))

        self.assertEqual(dtype.size, 2)
        self.assertEqual(dtype.ebias, 15)

        prec = dtype.precision
        self.assertTrue(dtype.precision in [16])
        dtype.precision = 80
        self.assertEqual(dtype.precision, 80)
        dtype.precision = prec
        self.assertEqual(dtype.precision, prec)

        self.assertEqual(dtype.offset, 0)
        dtype.offset = 2
        self.assertEqual(dtype.offset, 2)
        dtype.offset = 0
        self.assertEqual(dtype.offset, 0)

        order = dtype.order
        self.assertTrue(dtype.order in [h5cpp.datatype.Order.LE,
                                        h5cpp.datatype.Order.BE])
        dtype.order = h5cpp.datatype.Order.BE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.BE)
        dtype.order = h5cpp.datatype.Order.LE
        self.assertEqual(dtype.order, h5cpp.datatype.Order.LE)
        dtype.order = order

        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        dtype.pad = [h5cpp.datatype.Pad.ONE,
                     h5cpp.datatype.Pad.BACKGROUND]
        lp, mp = dtype.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ONE)
        self.assertEqual(mp, h5cpp.datatype.Pad.BACKGROUND)
        dtype.pad = (h5cpp.datatype.Pad.ZERO,
                     h5cpp.datatype.Pad.ZERO)

        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.ZERO)
        dtype.inpad = h5cpp.datatype.Pad.ONE
        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.ONE)
        dtype.inpad = h5cpp.datatype.Pad.BACKGROUND
        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.BACKGROUND)
        dtype.inpad = h5cpp.datatype.Pad.ZERO
        self.assertEqual(dtype.inpad, h5cpp.datatype.Pad.ZERO)

        norm = dtype.norm
        self.assertTrue(dtype.norm in [
            h5cpp.datatype.Norm.NONE, h5cpp.datatype.Norm.IMPLIED])
        dtype.norm = h5cpp.datatype.Norm.MSBSET
        self.assertEqual(dtype.norm, h5cpp.datatype.Norm.MSBSET)
        dtype.norm = h5cpp.datatype.Norm.NONE
        self.assertEqual(dtype.norm, h5cpp.datatype.Norm.NONE)
        dtype.norm = norm

        ebias = dtype.ebias
        self.assertEqual(dtype.ebias, 15)
        dtype.ebias = 63
        self.assertEqual(dtype.ebias, 63)
        dtype.ebias = 31
        self.assertEqual(dtype.ebias, 31)
        dtype.ebias = ebias

        fields = dtype.fields
        self.assertEqual(len(fields), 5)
        fields1 = (15, 10, 5, 0, 5)
        fields2 = (14, 9, 5, 0, 9)
        dtype.fields = fields1
        self.assertEqual(dtype.fields, fields1)
        dtype.fields = fields2
        self.assertEqual(dtype.fields, fields2)
        dtype.fields = fields

    def testComplex32(self):

        dtype = h5cpp.datatype.Compound(h5cpp.datatype.kComplex32)
        self.assertTrue(isinstance(dtype, self.float_types))
        self.assertEqual(dtype.size, 4)
        self.assertEqual(dtype.number_of_fields, 2)
        self.assertEqual(dtype.field_name(0), "real")
        self.assertEqual(dtype.field_name(1), "imag")
        self.assertEqual(dtype.field_index("real"), 0)
        self.assertEqual(dtype.field_index("imag"), 1)
        self.assertEqual(dtype.field_offset("real"), 0)
        self.assertEqual(dtype.field_offset(0), 0)
        self.assertEqual(dtype.field_offset("imag"), 2)
        self.assertEqual(dtype.field_offset(1), 2)
        self.assertEqual(dtype.field_class("real"),
                         h5cpp._datatype.Class.FLOAT)
        self.assertEqual(dtype.field_class("imag"),
                         h5cpp._datatype.Class.FLOAT)
        self.assertEqual(dtype.field_class(0),
                         h5cpp._datatype.Class.FLOAT)
        self.assertEqual(dtype.field_class(1),
                         h5cpp._datatype.Class.FLOAT)

        real = dtype[0]
        imag = dtype[1]

        self.assertEqual(real.precision, 16)
        self.assertEqual(imag.precision, 16)
        self.assertEqual(real.offset, 0)
        self.assertEqual(imag.offset, 0)
        self.assertTrue(real.order in [h5cpp.datatype.Order.LE,
                                       h5cpp.datatype.Order.BE])
        self.assertTrue(imag.order in [h5cpp.datatype.Order.LE,
                                       h5cpp.datatype.Order.BE])

        lp, mp = real.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        lp, mp = imag.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(real.inpad, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(imag.inpad, h5cpp.datatype.Pad.ZERO)

        self.assertTrue(real.norm in [
            h5cpp.datatype.Norm.NONE, h5cpp.datatype.Norm.IMPLIED])
        self.assertTrue(imag.norm in [
            h5cpp.datatype.Norm.NONE, h5cpp.datatype.Norm.IMPLIED])
        self.assertEqual(real.ebias, 15)
        self.assertEqual(imag.ebias, 15)
        self.assertEqual(real.size, 2)
        self.assertEqual(imag.size, 2)

        fields = real.fields
        self.assertEqual(len(fields), 5)
        fields1 = (15, 10, 5, 0, 10)
        self.assertEqual(fields, fields1)
        fields = imag.fields
        self.assertEqual(len(fields), 5)
        self.assertEqual(fields, fields1)

    def testComplex64(self):

        dtype = h5cpp.datatype.Compound(h5cpp.datatype.kComplex64)
        self.assertTrue(isinstance(dtype, self.float_types))
        self.assertEqual(dtype.size, 8)
        self.assertEqual(dtype.number_of_fields, 2)
        self.assertEqual(dtype.field_name(0), "real")
        self.assertEqual(dtype.field_name(1), "imag")
        self.assertEqual(dtype.field_index("real"), 0)
        self.assertEqual(dtype.field_index("imag"), 1)
        self.assertEqual(dtype.field_offset("real"), 0)
        self.assertEqual(dtype.field_offset(0), 0)
        self.assertEqual(dtype.field_offset("imag"), 4)
        self.assertEqual(dtype.field_offset(1), 4)
        self.assertEqual(dtype.field_class("real"),
                         h5cpp._datatype.Class.FLOAT)
        self.assertEqual(dtype.field_class("imag"),
                         h5cpp._datatype.Class.FLOAT)
        self.assertEqual(dtype.field_class(0),
                         h5cpp._datatype.Class.FLOAT)
        self.assertEqual(dtype.field_class(1),
                         h5cpp._datatype.Class.FLOAT)

        real = dtype[0]
        imag = dtype[1]

        self.assertEqual(real.precision, 32)
        self.assertEqual(imag.precision, 32)
        self.assertEqual(real.offset, 0)
        self.assertEqual(imag.offset, 0)
        self.assertTrue(real.order in [h5cpp.datatype.Order.LE,
                                       h5cpp.datatype.Order.BE])
        self.assertTrue(imag.order in [h5cpp.datatype.Order.LE,
                                       h5cpp.datatype.Order.BE])

        lp, mp = real.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        lp, mp = imag.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(real.inpad, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(imag.inpad, h5cpp.datatype.Pad.ZERO)

        self.assertTrue(real.norm in [
            h5cpp.datatype.Norm.NONE, h5cpp.datatype.Norm.IMPLIED])
        self.assertTrue(imag.norm in [
            h5cpp.datatype.Norm.NONE, h5cpp.datatype.Norm.IMPLIED])
        self.assertEqual(real.ebias, 127)
        self.assertEqual(imag.ebias, 127)
        self.assertEqual(real.size, 4)
        self.assertEqual(imag.size, 4)

        fields = real.fields
        self.assertEqual(len(fields), 5)
        fields1 = (31, 23, 8, 0, 23)
        self.assertEqual(fields, fields1)
        fields = imag.fields
        self.assertEqual(len(fields), 5)
        self.assertEqual(fields, fields1)

    def testComplex128(self):

        dtype = h5cpp.datatype.Compound(h5cpp.datatype.kComplex128)
        self.assertTrue(isinstance(dtype, self.float_types))
        self.assertEqual(dtype.size, 16)
        self.assertEqual(dtype.number_of_fields, 2)
        self.assertEqual(dtype.field_name(0), "real")
        self.assertEqual(dtype.field_name(1), "imag")
        self.assertEqual(dtype.field_index("real"), 0)
        self.assertEqual(dtype.field_index("imag"), 1)
        self.assertEqual(dtype.field_offset("real"), 0)
        self.assertEqual(dtype.field_offset(0), 0)
        self.assertEqual(dtype.field_offset("imag"), 8)
        self.assertEqual(dtype.field_offset(1), 8)
        self.assertEqual(dtype.field_class("real"),
                         h5cpp._datatype.Class.FLOAT)
        self.assertEqual(dtype.field_class("imag"),
                         h5cpp._datatype.Class.FLOAT)
        self.assertEqual(dtype.field_class(0),
                         h5cpp._datatype.Class.FLOAT)
        self.assertEqual(dtype.field_class(1),
                         h5cpp._datatype.Class.FLOAT)

        real = dtype[0]
        imag = dtype[1]

        self.assertEqual(real.precision, 64)
        self.assertEqual(imag.precision, 64)
        self.assertEqual(real.offset, 0)
        self.assertEqual(imag.offset, 0)
        self.assertTrue(real.order in [h5cpp.datatype.Order.LE,
                                       h5cpp.datatype.Order.BE])
        self.assertTrue(imag.order in [h5cpp.datatype.Order.LE,
                                       h5cpp.datatype.Order.BE])

        lp, mp = real.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        lp, mp = imag.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(real.inpad, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(imag.inpad, h5cpp.datatype.Pad.ZERO)

        self.assertTrue(real.norm in [
            h5cpp.datatype.Norm.NONE, h5cpp.datatype.Norm.IMPLIED])
        self.assertTrue(imag.norm in [
            h5cpp.datatype.Norm.NONE, h5cpp.datatype.Norm.IMPLIED])
        self.assertEqual(real.ebias, 1023)
        self.assertEqual(imag.ebias, 1023)
        self.assertEqual(real.size, 8)
        self.assertEqual(imag.size, 8)

        fields = real.fields
        self.assertEqual(len(fields), 5)
        fields1 = (63, 52, 11, 0, 52)
        self.assertEqual(fields, fields1)
        fields = imag.fields
        self.assertEqual(len(fields), 5)
        self.assertEqual(fields, fields1)

    def testComplex256(self):

        dtype = h5cpp.datatype.Compound(h5cpp.datatype.kComplex256)
        self.assertTrue(isinstance(dtype, self.float_types))
        self.assertEqual(dtype.size, 32)
        self.assertEqual(dtype.number_of_fields, 2)
        self.assertEqual(dtype.field_name(0), "real")
        self.assertEqual(dtype.field_name(1), "imag")
        self.assertEqual(dtype.field_index("real"), 0)
        self.assertEqual(dtype.field_index("imag"), 1)
        self.assertEqual(dtype.field_offset("real"), 0)
        self.assertEqual(dtype.field_offset(0), 0)
        self.assertEqual(dtype.field_offset("imag"), 16)
        self.assertEqual(dtype.field_offset(1), 16)
        self.assertEqual(dtype.field_class("real"),
                         h5cpp._datatype.Class.FLOAT)
        self.assertEqual(dtype.field_class("imag"),
                         h5cpp._datatype.Class.FLOAT)
        self.assertEqual(dtype.field_class(0),
                         h5cpp._datatype.Class.FLOAT)
        self.assertEqual(dtype.field_class(1),
                         h5cpp._datatype.Class.FLOAT)

        real = dtype[0]
        imag = dtype[1]

        self.assertTrue(real.precision in [80, 128])
        self.assertTrue(imag.precision in [80, 128])
        self.assertEqual(real.offset, 0)
        self.assertEqual(imag.offset, 0)
        self.assertTrue(real.order in [h5cpp.datatype.Order.LE,
                                       h5cpp.datatype.Order.BE])
        self.assertTrue(imag.order in [h5cpp.datatype.Order.LE,
                                       h5cpp.datatype.Order.BE])

        lp, mp = real.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        lp, mp = imag.pad
        self.assertEqual(lp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(mp, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(real.inpad, h5cpp.datatype.Pad.ZERO)
        self.assertEqual(imag.inpad, h5cpp.datatype.Pad.ZERO)

        self.assertTrue(real.norm in [
            h5cpp.datatype.Norm.NONE, h5cpp.datatype.Norm.IMPLIED])
        self.assertTrue(imag.norm in [
            h5cpp.datatype.Norm.NONE, h5cpp.datatype.Norm.IMPLIED])
        # self.assertEqual(real.ebias, 16383)
        # self.assertEqual(imag.ebias, 16383)
        self.assertEqual(real.size, 16)
        self.assertEqual(imag.size, 16)

        fields = real.fields
        self.assertEqual(len(fields), 5)
        # fields1 = (79, 64, 15, 0, 64)
        # self.assertEqual(fields, fields1)
        fields = imag.fields
        # self.assertEqual(len(fields), 5)
        # self.assertEqual(fields, fields1)
