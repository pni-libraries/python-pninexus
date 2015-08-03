import unittest
import numpy
import os

import pni.io.nx.h5 as nx
from pni.io.nx.h5 import nxfile
from pni.io.nx.h5 import nxgroup
from pni.io.nx.h5 import nxfield
from pni.io.nx.h5 import deflate_filter
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file

#implementing test fixture
class nxfield_creation_test_uint8(unittest.TestCase):
    _typecode="uint8"

    def __init__(self,*args,**kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)

        self.filename = "nxfield_creation_test_{tc}.nxs".format(tc=self._typecode)
        self.filename = os.path.join("test",self.filename)
        self.gf = create_file(self.filename,overwrite=True)
        self.gf.close()

    
    def setUp(self):
        self.gf = open_file(self.filename,readonly=False)
        self.root = self.gf.root()

    def tearDown(self):
        self.root.close()
        self.gf.close()


    def test_scalar_creation(self):
        f = self.root.create_field("scalar",self._typecode)
        self.assertTrue(f.is_valid)
        self.assertTrue(f.dtype==self._typecode)
        self.assertTrue(f.shape==(1,))
        self.assertTrue(f.size == 1)

    def test_multidim_creation_without_chunk(self):
        f = self.root.create_field("multidim_without_chunk",
                                   self._typecode,
                                   shape=(0,1024,1024))
        self.assertTrue(f.is_valid)
        self.assertTrue(f.dtype==self._typecode)
        self.assertTrue(f.shape==(0,1024,1024))
        self.assertTrue(f.size == 0)

    def test_multidim_creation_with_chunk(self):
        f = self.root.create_field("multidim_with_chunk",
                                   self._typecode,
                                   shape=(0,1024,1024),
                                   chunk=[1,1024,1024])
        self.assertTrue(f.is_valid)
        self.assertTrue(f.dtype==self._typecode)
        self.assertTrue(f.shape==(0,1024,1024))
        self.assertTrue(f.size == 0)

    def test_multidim_creation_without_chunk_and_filter(self):
        comp = deflate_filter()
        f = self.root.create_field("multidim_withouth_chunk_with_filter",
                                   self._typecode,
                                   shape=(0,1024,1024),
                                   filter=comp)
        self.assertTrue(f.is_valid)
        self.assertTrue(f.dtype==self._typecode)
        self.assertTrue(f.shape==(0,1024,1024))
        self.assertTrue(f.size == 0)


    def test_multdim_creation_with_chunk_and_filter(self):
        comp = deflate_filter()
        f = self.root.create_field("multidim_with_chunk_with_filter",
                                   self._typecode,
                                   shape=(0,1024,1024),
                                   chunk=[1,1024,1024],
                                   filter=comp)
        self.assertTrue(f.is_valid)
        self.assertTrue(f.dtype==self._typecode)
        self.assertTrue(f.shape==(0,1024,1024))
        self.assertTrue(f.size == 0)

    def test_grow_field(self):
        f = self.root.create_field("grow",self._typecode,shape=(3,4))

        self.assertTrue(f.shape == (3,4))
        self.assertTrue(f.size  == 12)

        f.grow(0)
        self.assertTrue(f.shape == (4,4))
        self.assertTrue(f.size  == 16)
        f.grow(0,3)
        self.assertTrue(f.shape == (7,4))
        self.assertTrue(f.size  == 28)

        f.grow(1)
        self.assertTrue(f.shape == (7,5))
        self.assertTrue(f.size  == 35)
        f.grow(1,5)
        self.assertTrue(f.shape == (7,10))
        self.assertTrue(f.size  == 70)

class nxfield_creation_test_uint16(nxfield_creation_test_uint8):
    _typecode = "uint16"

class nxfield_creation_test_uint32(nxfield_creation_test_uint8):
    _typecode = "uint32"

class nxfield_creation_test_uint64(nxfield_creation_test_uint8):
    _typecode = "uint64"

class nxfield_creation_test_int8(nxfield_creation_test_uint8):
    _typecode = "int8"

class nxfield_creation_test_int16(nxfield_creation_test_uint8):
    _typecode = "int16"

class nxfield_creation_test_int32(nxfield_creation_test_uint8):
    _typecode = "int32"

class nxfield_creation_test_int64(nxfield_creation_test_uint8):
    _typecode = "int64"

class nxfield_creation_test_float32(nxfield_creation_test_uint8):
    _typecode = "float32"

class nxfield_creation_test_float64(nxfield_creation_test_uint8):
    _typecode = "float64"

class nxfield_creation_test_float128(nxfield_creation_test_uint8):
    _typecode = "float128"

class nxfield_creation_test_complex32(nxfield_creation_test_uint8):
    _typecode = "complex32"

class nxfield_creation_test_complex64(nxfield_creation_test_uint8):
    _typecode = "complex64"

class nxfield_creation_test_complex128(nxfield_creation_test_uint8):
    _typecode = "complex128"

class nxfield_creation_test_bool(nxfield_creation_test_uint8):
    _typecode = "bool"

class nxfield_creation_test_string(nxfield_creation_test_uint8):
    _typecode = "string"
