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
class scalar_io_test_uint8(unittest.TestCase):
    """
    Testing IO operations on a field. The possible operations are 
    
    """
    _typecode = "uint8"
    file_path = os.path.split(__file__)[0]

    #-------------------------------------------------------------------------
    @classmethod
    def setUpClass(self):
        self.file_name = "grow_test_{tc}.nxs".format(tc=self._typecode)
        self.full_path = os.path.join(self.file_path,self.file_name)
        self.gf = create_file(self.full_path,overwrite=True)
        self.gf.close()
    
    #-------------------------------------------------------------------------
    def setUp(self):
        self.gf = open_file(self.full_path,readonly=False)
        self.root = self.gf.root()
        self.generator = random_generator_factory(self._typecode)

    #-------------------------------------------------------------------------
    def tearDown(self):
        self.root.close()
        self.gf.close()
    
    #-------------------------------------------------------------------------
    def test_write_scalar_to_scalar(self):
        pass
    
    #-------------------------------------------------------------------------
    def test_read_scalar_from_scalar(self):
        pass

    #-------------------------------------------------------------------------
    def test_write_scalar_to_1Dfield(self):
        pass

    #-------------------------------------------------------------------------
    def test_read_scalar_from_1Dfield(self):
        pass

    #-------------------------------------------------------------------------
    def test_write_scalar_to_2Dfield(self):
        pass

    #-------------------------------------------------------------------------
    def test_read_scalar_from_2Dfield(self):
        pass



#=============================================================================
class scalar_io_test_uint16(scalar_io_test_uint8):
    _typecode = "uint16"

#=============================================================================
class scalar_io_test_uint32(scalar_io_test_uint8):
    _typecode = "uint32"

#=============================================================================
class scalar_io_test_uint64(scalar_io_test_uint8):
    _typecode = "uint64"

#=============================================================================
class scalar_io_test_int8(scalar_io_test_uint8):
    _typecode = "int8"

#=============================================================================
class scalar_io_test_int16(scalar_io_test_uint8):
    _typecode = "int16"

#=============================================================================
class scalar_io_test_int32(scalar_io_test_uint8):
    _typecode = "int32"

#=============================================================================
class scalar_io_test_int64(scalar_io_test_uint8):
    _typecode = "int64"

#=============================================================================
class scalar_io_test_float32(scalar_io_test_uint8):
    _typecode = "float32"

#=============================================================================
class scalar_io_test_float64(scalar_io_test_uint8):
    _typecode = "float64"

#=============================================================================
class scalar_io_test_float128(scalar_io_test_uint8):
    _typecode = "float128"

#=============================================================================
class scalar_io_test_complex32(scalar_io_test_uint8):
    _typecode = "complex32"

#=============================================================================
class scalar_io_test_complex64(scalar_io_test_uint8):
    _typecode = "complex64"

#=============================================================================
class scalar_io_test_complex128(scalar_io_test_uint8):
    _typecode = "complex128"

#=============================================================================
class scalar_io_test_bool(scalar_io_test_uint8):
    _typecode = "bool"

#=============================================================================
class scalar_io_test_string(scalar_io_test_uint8):
    _typecode = "string"
