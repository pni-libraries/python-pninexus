import unittest
import numpy
import os

import pni.io.nx.h5 as nx
from pni.io.nx.h5 import nxfield
from pni.io.nx.h5 import deflate_filter
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file

from .. data_generator import random_generator_factory

#implementing test fixture
class grow_test_uint8(unittest.TestCase):
    """
    This test creates a single file as the only artifact. The result for 
    every individual test is written to the same file. This makes 
    investigation in case of errors easier.

    This testsuite handles 3 standard use cases:
    .) growing a 1D field - this would be storing scalar data 
    .) growing a 2D field - storing 1D MCA data 
    .) growing a 3D field - storing image data on a stack
    """
    _typecode="uint8"
    file_path = os.path.split(__file__)[0]
    file_name = "grow_test_{tc}.nxs".format(tc=_typecode)
    full_path = os.path.join(file_path,file_name)
    npts = 100
    frame_shape_2d=(1024,)
    frame_shape_3d=(1024,512)

    @classmethod
    def setUpClass(self):
        """
        Setup the file where all the tests are performed. 
        """
        self.gf = create_file(self.full_path,overwrite=True)
        self.gf.close()

    #-------------------------------------------------------------------------
    def setUp(self):
        self.gf = open_file(self.full_path,readonly=False)
        self.root = self.gf.root()
        self.randgen = random_generator_factory(self._typecode)

    #-------------------------------------------------------------------------
    def tearDown(self):
        self.root.close()
        self.gf.close()

    #-------------------------------------------------------------------------
    def test_grow_1D_field(self):
        f = self.root.create_field("data_1d",self._typecode,
                              shape=(0,),chunk=(1024,))
        
        ref = []
        for (pts_index,data) in zip(range(self.npts),self.randgen(1,100)):
            f.grow(0,1)
            f[-1] = data
            ref.append(data)

        #nee to add here a comparison

    #-------------------------------------------------------------------------
    def test_grow_2D_field(self):
        f = self.root.create_field("data_2d",self._typecode,
                              shape=(0,1024),chunk=(1,1024))

        for pts_index in range(self.npts):
            pass

    #-------------------------------------------------------------------------
    def test_grow_3D_field(self):

        f = self.root.create_field("data_3d",self._typecode,
                                   shape=(0,1024,512),
                                   chunk=(1,1024,512))

        for pts_index in range(self.npts):
            pass



#=============================================================================
class grow_test_uint16(grow_test_uint8):
    _typecode = "uint16"

#=============================================================================
class grow_test_uint32(grow_test_uint8):
    _typecode = "uint32"

#=============================================================================
class grow_test_uint64(grow_test_uint8):
    _typecode = "uint64"

#=============================================================================
class grow_test_int8(grow_test_uint8):
    _typecode = "int8"

#=============================================================================
class grow_test_int16(grow_test_uint8):
    _typecode = "int16"

#=============================================================================
class grow_test_int32(grow_test_uint8):
    _typecode = "int32"

#=============================================================================
class grow_test_int64(grow_test_uint8):
    _typecode = "int64"

#=============================================================================
class grow_test_float32(grow_test_uint8):
    _typecode = "float32"

#=============================================================================
class grow_test_float64(grow_test_uint8):
    _typecode = "float64"

#=============================================================================
class grow_test_float128(grow_test_uint8):
    _typecode = "float128"

#=============================================================================
class grow_test_complex32(grow_test_uint8):
    _typecode = "complex32"

#=============================================================================
class grow_test_complex64(grow_test_uint8):
    _typecode = "complex64"

#=============================================================================
class grow_test_complex128(grow_test_uint8):
    _typecode = "complex128"

#=============================================================================
class grow_test_bool(grow_test_uint8):
    _typecode = "bool"

#=============================================================================
class grow_test_string(grow_test_uint8):
    _typecode = "string"
