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

        dtype = h5cpp.datatype.kUInt8
        self.assertTrue(isinstance(dtype, self.int_types))
        self.assertEqual(dtype.size, 1)
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

    def testInt8(self):

        dtype = h5cpp.datatype.kInt8
        self.assertTrue(isinstance(dtype, self.int_types))
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

    def testUInt16(self):

        dtype = h5cpp.datatype.kUInt16
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

        dtype = h5cpp.datatype.kInt16
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

        dtype = h5cpp.datatype.kUInt32
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

        dtype = h5cpp.datatype.kInt32
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

        dtype = h5cpp.datatype.kUInt64
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

        dtype = h5cpp.datatype.kInt64
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

        dtype = h5cpp.datatype.kFloat32
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

        self.assertEqual(dtype.size, 4)
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
        
        self.assertEqual(dtype.size, 4)
        self.assertEqual(dtype.ebias, 127)
        self.assertTrue(dtype.ebias in [2 * sz * sz * sz - 1,
                                        4 * sz * sz * sz - 1])
        dtype.ebias = 63
        self.assertEqual(dtype.ebias, 63)
        dtype.ebias = 31
        self.assertEqual(dtype.ebias, 31)
        dtype.ebias = ebias

    def testFloat64(self):

        dtype = h5cpp.datatype.kFloat64
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
        sz = dtype.size
        self.assertEqual(dtype.ebias, 63)
        self.assertTrue(dtype.ebias in [2 * sz * sz * sz - 1,
                                        4 * sz * sz * sz - 1])
        dtype.ebias = 63
        self.assertEqual(dtype.ebias, 63)
        dtype.ebias = 31
        self.assertEqual(dtype.ebias, 31)
        dtype.ebias = ebias

    def testFloat128(self):

        dtype = h5cpp.datatype.kFloat128
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


    def testVariableString(self):

        dtype = h5cpp.datatype.kVariableString
        self.assertTrue(isinstance(dtype, self.string_types))
        self.assertTrue(dtype.is_variable_length)

    def testEBool(self):

        dtype = h5cpp.datatype.kEBool
        self.assertTrue(isinstance(dtype, self.enum_types))
        self.assertTrue(h5cpp._datatype.is_bool(dtype))
        self.assertEqual(dtype.size, 1)
