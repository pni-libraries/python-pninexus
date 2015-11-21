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
import numpy.random as random
import os

import pni.io.nx.h5 as nx
from pni.io.nx.h5 import nxfile
from pni.io.nx.h5 import nxgroup
from pni.io.nx.h5 import nxfield
from pni.io.nx.h5 import deflate_filter
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.core import ShapeMismatchError
from pni.core import SizeMismatchError
from .. data_generator import random_generator_factory

types=["uint8","int8","uint16","int16","uint32","int32","uint64","int64",
       "float32","float64","float128","complex32","complex64","complex128",
       "string","bool"]

scalars={"uint8":numpy.uint8,"int8":numpy.int8,
        "uint16":numpy.uint16,"int16":numpy.int16,
        "uint32":numpy.uint32,"int32":numpy.int32,
        "uint64":numpy.uint64,"int64":numpy.int64,
        "float32":numpy.float32,"float64":numpy.float64,
        "float128":numpy.float128,
        "complex32":numpy.complex64,
        "complex64":numpy.complex128,
        "complex128":numpy.complex256,
        "string":numpy.str_,"bool":numpy.bool_}
         


#implementing test fixture
class mdim_io_test_on_field_uint8(unittest.TestCase):
    """
    Testing multidimensional IO on an attribute attached to a field.
    """
    _typecode = "uint8"
    file_path = os.path.split(__file__)[0]
    shape = (20,11)

    #-------------------------------------------------------------------------
    @classmethod
    def setUpClass(self):
        self.file_name = "mdim_io_test_on_field{tc}.nxs".format(tc=self._typecode)
        self.full_path = os.path.join(self.file_path,self.file_name)
        self.gf = create_file(self.full_path,overwrite=True)
        self.gf.root().create_field("data",self._typecode)
        
        self.gf.close()
    
    #-------------------------------------------------------------------------
    def setUp(self):
        self.gf = open_file(self.full_path,readonly=False)
        self.root = self.gf.root()
        self.generator = random_generator_factory(self._typecode)
        self.scalar_type = scalars[self._typecode]
        self.field = self.root["data"]

    #-------------------------------------------------------------------------
    def tearDown(self):
        self.root.close()
        self.field.close()
        self.gf.close()
    
    #-------------------------------------------------------------------------
    def test_mdim_to_mdim_write_read_method(self):
        """
        Write an attribute entirely from a mdim array using the read and write
        methods.
        """

        a = self.field.attributes.create("mdim_read_write",self._typecode,
                                        shape=self.shape)

        input_data = next(self.generator(shape=self.shape))

        a.write(input_data)
        output_data = a.read()
        self.assertTrue(isinstance(output_data,numpy.ndarray))
        if self._typecode != "string":
            self.assertEqual(output_data.dtype,self.scalar_type)

        for (i,o) in zip(input_data.flat,output_data.flat):
            if self._typecode == "string":
                self.assertEqual(self.scalar_type(o),self.scalar_type(i))
            else:
                self.assertAlmostEqual(self.scalar_type(o),self.scalar_type(i))

        self.assertRaises(SizeMismatchError,
                          a.write,numpy.ones((100,20),dtype=self.scalar_type))


    #-------------------------------------------------------------------------
    def test_mdim_to_mdim_setitem_getitem(self):
        """
        Write an attribute entirely from a mdim array using the __setitem__ and
        __getitem__ approach.
        """

        a = self.field.attributes.create("mdim_setitem_getitem",self._typecode,
                                        shape=self.shape)

        input_data = next(self.generator(shape=self.shape))

        a[...] = input_data
        output_data = a[...]
        self.assertTrue(isinstance(output_data,numpy.ndarray))
        if self._typecode != "string":
            self.assertEqual(output_data.dtype,self.scalar_type)

        for (i,o) in zip(input_data.flat,output_data.flat):
            self.assertAlmostEqual(self.scalar_type(o),self.scalar_type(i))

    #-------------------------------------------------------------------------
    def test_mdim_to_mdim_strip(self):
        """
        Read and write strips of data to an attribute
        """

        a = self.field.attributes.create("mdim_strip",self._typecode,
                                   shape=self.shape)

        g = self.generator(shape=(self.shape[1],))
        for (index,input_data) in zip(range(a.shape[0]),g):
            a[index,...] = input_data
            output_data = a[index,...]
            self.assertEqual(output_data.shape,(a.shape[1],))

            for (o,i) in zip(output_data.flat,input_data.flat):
                self.assertAlmostEqual(self.scalar_type(o),
                                       self.scalar_type(i))


#=============================================================================
class mdim_io_test_on_field_uint16(mdim_io_test_on_field_uint8):
    _typecode = "uint16"

#=============================================================================
class mdim_io_test_on_field_uint32(mdim_io_test_on_field_uint8):
    _typecode = "uint32"

#=============================================================================
class mdim_io_test_on_field_uint64(mdim_io_test_on_field_uint8):
    _typecode = "uint64"

#=============================================================================
class mdim_io_test_on_field_int8(mdim_io_test_on_field_uint8):
    _typecode = "int8"

#=============================================================================
class mdim_io_test_on_field_int16(mdim_io_test_on_field_uint8):
    _typecode = "int16"

#=============================================================================
class mdim_io_test_on_field_int32(mdim_io_test_on_field_uint8):
    _typecode = "int32"

#=============================================================================
class mdim_io_test_on_field_int64(mdim_io_test_on_field_uint8):
    _typecode = "int64"

#=============================================================================
class mdim_io_test_on_field_float32(mdim_io_test_on_field_uint8):
    _typecode = "float32"

#=============================================================================
class mdim_io_test_on_field_float64(mdim_io_test_on_field_uint8):
    _typecode = "float64"

#=============================================================================
class mdim_io_test_on_field_float128(mdim_io_test_on_field_uint8):
    _typecode = "float128"

#=============================================================================
class mdim_io_test_on_field_complex32(mdim_io_test_on_field_uint8):
    _typecode = "complex32"

#=============================================================================
class mdim_io_test_on_field_complex64(mdim_io_test_on_field_uint8):
    _typecode = "complex64"

#=============================================================================
class mdim_io_test_on_field_complex128(mdim_io_test_on_field_uint8):
    _typecode = "complex128"

#=============================================================================
class mdim_io_test_on_field_bool(mdim_io_test_on_field_uint8):
    _typecode = "bool"

#=============================================================================
class mdim_io_test_on_field_string(mdim_io_test_on_field_uint8):
    _typecode = "string"
