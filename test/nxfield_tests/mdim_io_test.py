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
# Created on: Oct 7, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
import unittest
import numpy
import numpy.random as random
import os

from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.core import SizeMismatchError
from .. data_generator import random_generator_factory
from .. import io_test_utils as iotu

#implementing test fixture
class mdim_io_test_uint8(unittest.TestCase):
    """
    Testing IO operations on a field. The possible operations are 
    
    """
    _typecode = "uint8"
    file_path = os.path.split(__file__)[0]
    shape = (20,11)

    #-------------------------------------------------------------------------
    @classmethod
    def setUpClass(self):
        self.file_name = "mdim_io_test_{tc}.nxs".format(tc=self._typecode)
        self.full_path = os.path.join(self.file_path,self.file_name)
        self.gf = create_file(self.full_path,overwrite=True)
        
        self.gf.close()
    
    #-------------------------------------------------------------------------
    def setUp(self):
        self.gf = open_file(self.full_path,readonly=False)
        self.root = self.gf.root()
        self.generator = random_generator_factory(self._typecode)
        self.scalar_type = iotu.scalars[self._typecode]

    #-------------------------------------------------------------------------
    def tearDown(self):
        self.root.close()
        self.gf.close()
    
    #-------------------------------------------------------------------------
    def test_mdim_to_mdim_write_read_method(self):
        """
        Write a field entirely from a mdim array using the read and write 
        methods.
        """

        f = self.root.create_field("mdim_read_write",self._typecode,
                                   shape=self.shape)

        input_data = next(self.generator(shape=self.shape))

        f.write(input_data)
        output_data = f.read()
        self.assertTrue(isinstance(output_data,numpy.ndarray))
        if self._typecode != "string":
            self.assertEqual(output_data.dtype,self.scalar_type)

        for (i,o) in zip(input_data.flat,output_data.flat):
            if self._typecode == "string":
                self.assertEqual(self.scalar_type(o),self.scalar_type(i))
            else:
                self.assertAlmostEqual(self.scalar_type(o),self.scalar_type(i))

        self.assertRaises(SizeMismatchError,
                          f.write,numpy.ones((100,20),dtype=self.scalar_type))
        #self.assertRaises(SizeMismatchError,f.write,
        #                  numpy.ones((11,20),dtype=self.scalar_type))


    #-------------------------------------------------------------------------
    def test_mdim_to_mdim_setitem_getitem(self):
        """
        Write a field entirely from a mdim array using the __setitem__ and 
        __getitem__ approach.
        """

        f = self.root.create_field("mdim_setitem_getitem",self._typecode,
                                   shape=self.shape)

        input_data = next(self.generator(shape=self.shape))

        f[...] = input_data
        output_data = f[...]
        self.assertTrue(isinstance(output_data,numpy.ndarray))
        if self._typecode != "string":
            self.assertEqual(output_data.dtype,self.scalar_type)

        for (i,o) in zip(input_data.flat,output_data.flat):
            self.assertAlmostEqual(self.scalar_type(o),self.scalar_type(i))

    #-------------------------------------------------------------------------
    def test_mdim_to_mdim_strip(self):
        """
        Read and write strips of data to a field
        """

        f = self.root.create_field("mdim_strip",self._typecode,
                                   shape=self.shape)

        g = self.generator(shape=(self.shape[1],))
        for (index,input_data) in zip(range(f.shape[0]),g):
            f[index,...] = input_data
            output_data = f[index,...]
            self.assertEqual(output_data.shape,(f.shape[1],))

            for (o,i) in zip(output_data.flat,input_data.flat):
                self.assertAlmostEqual(self.scalar_type(o),
                                       self.scalar_type(i))

    
    




#=============================================================================
class mdim_io_test_uint16(mdim_io_test_uint8):
    _typecode = "uint16"

#=============================================================================
class mdim_io_test_uint32(mdim_io_test_uint8):
    _typecode = "uint32"

#=============================================================================
class mdim_io_test_uint64(mdim_io_test_uint8):
    _typecode = "uint64"

#=============================================================================
class mdim_io_test_int8(mdim_io_test_uint8):
    _typecode = "int8"

#=============================================================================
class mdim_io_test_int16(mdim_io_test_uint8):
    _typecode = "int16"

#=============================================================================
class mdim_io_test_int32(mdim_io_test_uint8):
    _typecode = "int32"

#=============================================================================
class mdim_io_test_int64(mdim_io_test_uint8):
    _typecode = "int64"

#=============================================================================
class mdim_io_test_float32(mdim_io_test_uint8):
    _typecode = "float32"

#=============================================================================
class mdim_io_test_float64(mdim_io_test_uint8):
    _typecode = "float64"

#=============================================================================
class mdim_io_test_float128(mdim_io_test_uint8):
    _typecode = "float128"

#=============================================================================
class mdim_io_test_complex32(mdim_io_test_uint8):
    _typecode = "complex32"

#=============================================================================
class mdim_io_test_complex64(mdim_io_test_uint8):
    _typecode = "complex64"

#=============================================================================
class mdim_io_test_complex128(mdim_io_test_uint8):
    _typecode = "complex128"

#=============================================================================
class mdim_io_test_bool(mdim_io_test_uint8):
    _typecode = "bool"

#=============================================================================
class mdim_io_test_string(mdim_io_test_uint8):
    _typecode = "string"
