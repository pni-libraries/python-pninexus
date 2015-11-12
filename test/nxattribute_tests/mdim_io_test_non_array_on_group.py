import unittest
import numpy
import numpy.random as random
import os
import sys

from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.core import SizeMismatchError
from .. data_generator import random_generator_factory
from .. import io_test_utils as iotu


#implementing test fixture
class mdim_io_test_non_array_on_group_uint8(unittest.TestCase):
    """
    Testing multidimensional IO on an attribute with non-array input data. 
    The attribute is attached to a group.
    """
    _typecode = "uint8"
    file_path = os.path.split(__file__)[0]
    shape = (20,11)

    #-------------------------------------------------------------------------
    @classmethod
    def setUpClass(self):
        self.file_name = "mdim_io_test_non_array_on_group_{tc}.nxs".format(tc=self._typecode)
        self.full_path = os.path.join(self.file_path,self.file_name)
        self.gf = create_file(self.full_path,overwrite=True)
        
        self.gf.close()
        
        if iotu.is_discrete_type(self._typecode):
            self.check_equal = self.assertEqual
        else:
            self.check_equal = self.assertAlmostEqual
    
    
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
        Write an entirey attribute from a nested list using the read and write
        methods.
        """

        a = self.root.attributes.create("mdim_read_write",self._typecode,
                                        shape=self.shape)

        input_data = next(self.generator(shape=self.shape))

        a.write(input_data.tolist())
        output_data = a.read()
        self.assertTrue(isinstance(output_data,numpy.ndarray))
        if self._typecode != "string":
            self.assertEqual(output_data.dtype,self.scalar_type)

        for (i,o) in zip(input_data.flat,output_data.flat):
            self.check_equal(self.scalar_type(o),self.scalar_type(i))

        self.assertRaises(SizeMismatchError,
                          a.write,numpy.ones((100,20),dtype=self.scalar_type))


    #-------------------------------------------------------------------------
    def test_mdim_to_mdim_setitem_getitem(self):
        """
        Write an entire attribute from a nested list using the __setitem__ and
        __getitem__ approach.
        """

        a = self.root.attributes.create("mdim_setitem_getitem",
                                        self._typecode,shape=self.shape)

        input_data = next(self.generator(shape=self.shape))

        a[...] = input_data.tolist()
        output_data = a[...]
        self.assertTrue(isinstance(output_data,numpy.ndarray))
        if self._typecode != "string":
            self.assertEqual(output_data.dtype,self.scalar_type)

        for (i,o) in zip(input_data.flat,output_data.flat):
            self.check_equal(self.scalar_type(o),self.scalar_type(i))

    #-------------------------------------------------------------------------
    def test_mdim_to_mdim_strip(self):
        """
        Read and write strips of data to an attribute rom a nestd list
        """

        a = self.root.attributes.create("mdim_strip",self._typecode,
                                        shape=self.shape)

        g = self.generator(shape=(self.shape[1],))
        for (index,input_data) in zip(range(a.shape[0]),g):
            a[index,...] = input_data.tolist()
            output_data = a[index,...]
            self.assertEqual(output_data.shape,(a.shape[1],))

            for (o,i) in zip(output_data.flat,input_data.flat):
                self.check_equal(self.scalar_type(o),self.scalar_type(i))

    
#=============================================================================
class mdim_io_test_non_array_on_group_uint16(mdim_io_test_non_array_on_group_uint8):
    _typecode = "uint16"

#=============================================================================
class mdim_io_test_non_array_on_group_uint32(mdim_io_test_non_array_on_group_uint8):
    _typecode = "uint32"

#=============================================================================
class mdim_io_test_non_array_on_group_uint64(mdim_io_test_non_array_on_group_uint8):
    _typecode = "uint64"

#=============================================================================
class mdim_io_test_non_array_on_group_int8(mdim_io_test_non_array_on_group_uint8):
    _typecode = "int8"

#=============================================================================
class mdim_io_test_non_array_on_group_int16(mdim_io_test_non_array_on_group_uint8):
    _typecode = "int16"

#=============================================================================
class mdim_io_test_non_array_on_group_int32(mdim_io_test_non_array_on_group_uint8):
    _typecode = "int32"

#=============================================================================
class mdim_io_test_non_array_on_group_int64(mdim_io_test_non_array_on_group_uint8):
    _typecode = "int64"

#=============================================================================
class mdim_io_test_non_array_on_group_float32(mdim_io_test_non_array_on_group_uint8):
    _typecode = "float32"

#=============================================================================
class mdim_io_test_non_array_on_group_float64(mdim_io_test_non_array_on_group_uint8):
    _typecode = "float64"

#=============================================================================
class mdim_io_test_non_array_on_group_float128(mdim_io_test_non_array_on_group_uint8):
    _typecode = "float128"

#=============================================================================
class mdim_io_test_non_array_on_group_complex32(mdim_io_test_non_array_on_group_uint8):
    _typecode = "complex32"

#=============================================================================
class mdim_io_test_non_array_on_group_complex64(mdim_io_test_non_array_on_group_uint8):
    _typecode = "complex64"

#=============================================================================
class mdim_io_test_non_array_on_group_complex128(mdim_io_test_non_array_on_group_uint8):
    _typecode = "complex128"

#=============================================================================
class mdim_io_test_non_array_on_group_bool(mdim_io_test_non_array_on_group_uint8):
    _typecode = "bool"

#=============================================================================
class mdim_io_test_non_array_on_group_string(mdim_io_test_non_array_on_group_uint8):
    _typecode = "string"

#=============================================================================
if sys.version_info[0]<=2:
    class mdim_io_test_non_array_on_group_unicode(mdim_io_test_non_array_on_group_string):
        _type_code="string"

        #-------------------------------------------------------------------------
        def setUp(self):
            self.gf = open_file(self.full_path,readonly=False)
            self.root = self.gf.root()
            self.generator = random_generator_factory('unicode')
            self.scalar_type = iotu.scalars[self._typecode]
