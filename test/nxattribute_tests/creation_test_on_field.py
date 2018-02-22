#
# (c) Copyright 2015 DESY, 
#               2015 Eugen Wintersberger <eugen.wintersberger@desy.de>
#
# This file is part of python-pni.
#
# python-pni is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# python-pni is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with python-pni.  If not, see <http://www.gnu.org/licenses/>.
# ===========================================================================
#
# Created on: Oct 12, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
import unittest
import numpy
import os

import nx as nx
from nx import nxfield
from nx import deflate_filter
from nx import create_file
from nx import open_file
from pni.io import ObjectError

#implementing test fixture
class creation_test_on_field_uint8(unittest.TestCase):
    """
    This test creates a single file as the only artifact. The result for 
    every individual test is written to the same file. This makes 
    investigation in case of errors easier.

    We will test all the inquery properties directly in the creation test. 
    This makes somehow sense and we can spare a single test suite
    """
    _typecode="uint8"
    file_path = os.path.split(__file__)[0]
    file_name = "creation_on_field_test_{tc}.nxs".format(tc=_typecode)
    full_path = os.path.join(file_path,file_name)

    @classmethod
    def setUpClass(self):
        """
        Setup the file where all the tests are performed. 
        """
        self.gf = create_file(self.full_path,overwrite=True)
        self.gf.root().create_field("data",self._typecode) 
        self.gf.close()

    
    def setUp(self):
        self.gf = open_file(self.full_path,readonly=False)
        self.root = self.gf.root()
        self.field = self.root["data"]

    def tearDown(self):
        self.root.close()
        self.field.close()
        self.gf.close()


    def test_scalar_creation(self):
        a = self.field.attributes.create("scalar",self._typecode)
        self.assertTrue(a.is_valid)
        self.assertEqual(a.dtype,self._typecode)
        self.assertEqual(a.shape,(1,))
        self.assertEqual(a.size,1)
        self.assertEqual(a.filename,self.full_path)
        self.assertEqual(a.parent.name,"data")
        self.assertEqual(a.path,"/data@scalar")
        self.assertEqual(a.name,"scalar")

    def test_multidim_creation(self):
        a = self.field.attributes.create("multidim",self._typecode,shape=(2,3))
        self.assertTrue(a.is_valid)
        self.assertEqual(a.dtype,self._typecode)
        self.assertEqual(a.shape,(2,3))
        self.assertEqual(a.size,6)
        self.assertEqual(a.filename,self.full_path)
        self.assertEqual(a.parent.name,"data")
        self.assertEqual(a.path,"/data@multidim")
        self.assertEqual(a.name,"multidim")

    def test_overwrite(self):
        a = self.field.attributes.create("test1",self._typecode)
        self.assertRaises(ObjectError,self.field.attributes.create,
                          "test1",self._typecode)

        a = self.field.attributes.create("test1",self._typecode,
                                        overwrite=True)


class creation_test_on_field_uint16(creation_test_on_field_uint8):
    _typecode = "uint16"

class creation_test_on_field_uint32(creation_test_on_field_uint8):
    _typecode = "uint32"

class creation_test_on_field_uint64(creation_test_on_field_uint8):
    _typecode = "uint64"

class creation_test_on_field_int8(creation_test_on_field_uint8):
    _typecode = "int8"

class creation_test_on_field_int16(creation_test_on_field_uint8):
    _typecode = "int16"

class creation_test_on_field_int32(creation_test_on_field_uint8):
    _typecode = "int32"

class creation_test_on_field_int64(creation_test_on_field_uint8):
    _typecode = "int64"

class creation_test_on_field_float32(creation_test_on_field_uint8):
    _typecode = "float32"

class creation_test_on_field_float64(creation_test_on_field_uint8):
    _typecode = "float64"

class creation_test_on_field_float128(creation_test_on_field_uint8):
    _typecode = "float128"

class creation_test_on_field_complex32(creation_test_on_field_uint8):
    _typecode = "complex32"

class creation_test_on_field_complex64(creation_test_on_field_uint8):
    _typecode = "complex64"

class creation_test_on_field_complex128(creation_test_on_field_uint8):
    _typecode = "complex128"

class creation_test_on_field_bool(creation_test_on_field_uint8):
    _typecode = "bool"

class creation_test_on_field_string(creation_test_on_field_uint8):
    _typecode = "string"
