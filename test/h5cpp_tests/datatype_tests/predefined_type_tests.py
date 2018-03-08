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
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
from pninexus import h5cpp
from pninexus.h5cpp.datatype import Float,String,Datatype,Integer

class PredefinedTypeTests(unittest.TestCase):
    
    float_types = (Datatype,Float)
    int_types   = (Datatype,Integer)
    string_types = (Datatype,String)
    
    def testUInt8(self):
        
        dtype = h5cpp.datatype.kUInt8
        self.assertTrue(isinstance(dtype,self.int_types))
        self.assertEqual(dtype.size,1)

    def testInt8(self):
        
        dtype = h5cpp.datatype.kInt8
        self.assertTrue(isinstance(dtype,self.int_types))
        self.assertEqual(dtype.size,1)
        
    def testUInt16(self):
        
        dtype = h5cpp.datatype.kUInt16
        self.assertTrue(isinstance(dtype,self.int_types))
        self.assertEqual(dtype.size,2)

    def testInt16(self):
        
        dtype = h5cpp.datatype.kInt16
        self.assertTrue(isinstance(dtype,self.int_types))
        self.assertEqual(dtype.size,2)
        
    def testUInt32(self):
        
        dtype = h5cpp.datatype.kUInt32
        self.assertTrue(isinstance(dtype,self.int_types))
        self.assertEqual(dtype.size,4)

    def testInt32(self):
        
        dtype = h5cpp.datatype.kInt32
        self.assertTrue(isinstance(dtype,self.int_types))
        self.assertEqual(dtype.size,4)
        
    def testUInt64(self):
        
        dtype = h5cpp.datatype.kUInt64
        self.assertTrue(isinstance(dtype,self.int_types))
        self.assertEqual(dtype.size,8)

    def testInt64(self):
        
        dtype = h5cpp.datatype.kInt64
        self.assertTrue(isinstance(dtype,self.int_types))
        self.assertEqual(dtype.size,8)
    
    def testFloat32(self):
        
        dtype = h5cpp.datatype.kFloat32
        self.assertTrue(isinstance(dtype,self.float_types))
        self.assertEqual(dtype.size,4)
    
    def testFloat64(self):
        
        dtype = h5cpp.datatype.kFloat64
        self.assertTrue(isinstance(dtype,self.float_types))
        self.assertEqual(dtype.size,8)
        
    def testFloat128(self):
        
        dtype = h5cpp.datatype.kFloat128
        self.assertTrue(isinstance(dtype,self.float_types))
        self.assertEqual(dtype.size,16)
        
    def testVariableString(self):
        
        dtype = h5cpp.datatype.kVariableString
        self.assertTrue(isinstance(dtype,self.string_types))
        self.assertTrue(dtype.is_variable_length)
