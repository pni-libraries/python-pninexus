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

from nx import create_file
from nx import open_file
from pni.core import SizeMismatchError
from .. data_generator import random_generator_factory
from .. import io_test_utils as iotu


#implementing test fixture
class scalar_io_test_on_group_uint8(unittest.TestCase):
    """
    Testing IO operations on attributes attached to a group.  
    
    """
    _typecode = "uint8"
    file_path = os.path.split(__file__)[0]

    #-------------------------------------------------------------------------
    @classmethod
    def setUpClass(self):
        self.file_name = "scalar_io_test_on_group_{tc}.nxs".format(tc=self._typecode)
        self.full_path = os.path.join(self.file_path,self.file_name)
        self.gf = create_file(self.full_path,overwrite=True)
        self.gf.close()
    
    #-------------------------------------------------------------------------
    def setUp(self):
        self.gf = open_file(self.full_path,readonly=False)
        self.root = self.gf.root()
        self.generator = random_generator_factory(self._typecode)
        self.input_data = next(self.generator())
        self.input_array_data = numpy.array([next(self.generator())])
        self.scalar_type = iotu.scalars[self._typecode]

    #-------------------------------------------------------------------------
    def tearDown(self):
        self.root.close()
        self.gf.close()
    
    #-------------------------------------------------------------------------
    def test_scalar_to_scalar_write_read_method(self):
        """
        Write a scalar to a field and read it back using the write() and 
        read() methods provided by the nxattribute class
        """

        a = self.root.attributes.create("scalar_read_write",self._typecode)

        #reading and writing data from a single native python object
        a.write(self.input_data)
        output_data = a.read()

        self.assertAlmostEqual(self.scalar_type(output_data),
                               self.scalar_type(self.input_data))

    #-------------------------------------------------------------------------
    def test_scalar_array_to_scalar_write_read_method(self):
        """
        Write a numpy array of size 1 to a scalar attribute using the write()
        and read() methods.
        """
        
        a = self.root.attributes.create("scalar_array_read_write",
                                         self._typecode)

        #write data from a numpy array with a single element 
        a.write(self.input_array_data)
        output_data = a.read()
        self.assertAlmostEqual(self.scalar_type(self.input_array_data[0]),
                               self.scalar_type(output_data))

        self.assertRaises(SizeMismatchError,a.write,numpy.ones((10)))

    
    #-------------------------------------------------------------------------
    def test_scalar_to_scalar_setitem_getitem(self):
        """
        Write a scalar value to a scalar attribute using the __getitem__ and 
        __setitem__ methods. 
        """

        a = self.root.attributes.create("scalar_setitem_getitem",
                                        self._typecode)

        a[...] = self.input_data
        output_data = a[...]

        self.assertAlmostEqual(self.scalar_type(output_data),
                               self.scalar_type(self.input_data))

   
    #-------------------------------------------------------------------------
    def test_scalar_array_to_scalar_setitem_getitem(self):
        """
        Write a scalar array value to a scalar attribute using the __getitem__ 
        and __setitem__ methods. 
        """

        a = self.root.attributes.create("scalar_array_setitem_getitem",
                                        self._typecode)

        a[...] = self.input_array_data
        output_data = a[...]

        self.assertAlmostEqual(self.scalar_type(output_data),
                               self.scalar_type(self.input_array_data[0]))
        

    #-------------------------------------------------------------------------
    def test_scalar_to_1Dfield_partial_individual(self):
        """
        Write and read individual data to a 1D attribute using partial IO. 
        """

        a = self.root.attributes.create("scalar_to_1D_partial_individual",
                                   self._typecode,
                                   shape=(10,))

        for (index,input_data) in zip(range(a.size),self.generator()):
            a[index] = input_data
            output_data = a[index]

            self.assertAlmostEqual(self.scalar_type(output_data),
                                   self.scalar_type(input_data))

    #-------------------------------------------------------------------------
    def test_scalar_array_to_1Dfield_partial_individual(self):
        """
        Write and read individual data to a 1D attribute using partial IO. 
        """

        a = self.root.attributes.create("scalar_array_to_1D_partial_individual",
                                   self._typecode,
                                   shape=(10,))

        for (index,input_data) in zip(range(a.size),self.generator()):
            a[index] = numpy.array([input_data])
            output_data = a[index]

            self.assertAlmostEqual(self.scalar_type(output_data),
                                   self.scalar_type(input_data))

    #-------------------------------------------------------------------------
    def test_scalar_to_1Dfield_broadcast(self):
        """
        Broadcast a single data value to a whole attribute.
        """

        a = self.root.attributes.create("scalar_to_1D_broadcast",
                                   self._typecode,shape=(10,))

        a[...] = self.input_data

        for output_data in a[...]:
            self.assertAlmostEqual(self.scalar_type(output_data),
                                   self.scalar_type(self.input_data))

    #-------------------------------------------------------------------------
    def test_scalar_array_to_1Dfield_broadcast(self):
        """
        Broadcast a single data value to a whole attribute.
        """

        a = self.root.attributes.create("scalar_array_to_1D_broadcast",
                                   self._typecode,shape=(10,))

        a[...] = self.input_array_data[...]

        for output_data in a[...]:
            self.assertAlmostEqual(self.scalar_type(output_data),
                                   self.scalar_type(self.input_array_data[0]))

    #-------------------------------------------------------------------------
    def test_scalar_to_2Dfield_partial_individual(self):
        """
        Writing individual scalars using partial IO
        """

        a = self.root.attributes.create("scalar_to_2d_indivdual",
                                   self._typecode,
                                   shape=(3,4))

        for i in range(3):
            for j in range(4):
                input_data = next(self.generator())
                a[i,j] = input_data
                output_data = a[i,j]
                self.assertAlmostEqual(self.scalar_type(output_data),
                                       self.scalar_type(input_data))

    #-------------------------------------------------------------------------
    def test_scalar_from_2Dfield_partial_strips(self):
        """
        Broadcast scalar data to strips and read them back
        """
        
        a = self.root.attributes.create("scalar_to_2d_strips",
                                   self._typecode,
                                   shape=(3,4))

        for index in range(a.shape[0]):
            input_data = next(self.generator())
            a[index,...] = input_data

            for output_data in a[index,...].flat:
                self.assertAlmostEqual(self.scalar_type(output_data),
                                       self.scalar_type(input_data))


    #-------------------------------------------------------------------------
    def test_scalar_from_2Dfield_broadcast(self):
        """
        Broadcast an individual value to a 2D attribute
        """
        
        a = self.root.attributes.create("scalar_to_2d_broadcast",
                                   self._typecode,
                                   shape=(3,4))
        a[...] = self.input_data
        
        for output_data in a[...].flat:
            self.assertAlmostEqual(self.scalar_type(output_data),
                                   self.scalar_type(self.input_data))



#=============================================================================
class scalar_io_test_on_group_uint16(scalar_io_test_on_group_uint8):
    _typecode = "uint16"

#=============================================================================
class scalar_io_test_on_group_uint32(scalar_io_test_on_group_uint8):
    _typecode = "uint32"

#=============================================================================
class scalar_io_test_on_group_uint64(scalar_io_test_on_group_uint8):
    _typecode = "uint64"

#=============================================================================
class scalar_io_test_on_group_int8(scalar_io_test_on_group_uint8):
    _typecode = "int8"

#=============================================================================
class scalar_io_test_on_group_int16(scalar_io_test_on_group_uint8):
    _typecode = "int16"

#=============================================================================
class scalar_io_test_on_group_int32(scalar_io_test_on_group_uint8):
    _typecode = "int32"

#=============================================================================
class scalar_io_test_on_group_int64(scalar_io_test_on_group_uint8):
    _typecode = "int64"

#=============================================================================
class scalar_io_test_on_group_float32(scalar_io_test_on_group_uint8):
    _typecode = "float32"

#=============================================================================
class scalar_io_test_on_group_float64(scalar_io_test_on_group_uint8):
    _typecode = "float64"

#=============================================================================
class scalar_io_test_on_group_float128(scalar_io_test_on_group_uint8):
    _typecode = "float128"

#=============================================================================
class scalar_io_test_on_group_complex32(scalar_io_test_on_group_uint8):
    _typecode = "complex32"

#=============================================================================
class scalar_io_test_on_group_complex64(scalar_io_test_on_group_uint8):
    _typecode = "complex64"

#=============================================================================
class scalar_io_test_on_group_complex128(scalar_io_test_on_group_uint8):
    _typecode = "complex128"

#=============================================================================
class scalar_io_test_on_group_bool(scalar_io_test_on_group_uint8):
    _typecode = "bool"

#=============================================================================
class scalar_io_test_on_group_string(scalar_io_test_on_group_uint8):
    _typecode = "string"
    
    #-------------------------------------------------------------------------
    def test_scalar_array_to_scalar_write_read_method(self):
        """
        Write a numpy array of size 1 to a scalar attribute using the write()
        and read() methods.
        """
        
        a = self.root.attributes.create("scalar_array_read_write",
                                         self._typecode)

        #write data from a numpy array with a single element 
        a.write(self.input_array_data)
        output_data = a.read()
        self.assertAlmostEqual(self.scalar_type(self.input_array_data[0]),
                               self.scalar_type(output_data))

        self.assertRaises(SizeMismatchError,a.write,
                          numpy.array(["hello","world"]))
        self.assertRaises(TypeError,a.write,1)
