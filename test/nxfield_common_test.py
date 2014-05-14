import unittest
import numpy

import pni.io.nx.h5 as nx
from pni.io.nx.h5 import NXFile
from pni.io.nx.h5 import NXGroup
from pni.io.nx.h5 import NXField
from pni.io.nx.h5 import NXDeflateFilter
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file

#implementing test fixture
class nxfield_common_test(unittest.TestCase):

    
    def setUp(self):
        self.gf = create_file("NXFieldTest.h5",overwrite=True)
        self.root = self.gf["/"]

    def tearDown(self):
        self.root.close()
        self.gf.close()


    def test_scalar_creation(self):
        f = self.root.create_field("data","uint16")
        self.assertTrue(f.valid)
        self.assertTrue(f.dtype=="uint16")
        self.assertTrue(f.shape==(1,))
        self.assertTrue(f.size == 1)

    def test_multidim_creation_without_chunk(self):
        f = self.gf.create_field("data","uint16",
                                  shape=(0,1024,1024))
        self.assertTrue(f.valid)
        self.assertTrue(f.dtype=="uint16")
        self.assertTrue(f.shape==(0,1024,1024))
        self.assertTrue(f.size == 0)

    def test_multidim_creation_with_chunk(self):
        f = self.gf.create_field("data","uint16",
                                  shape=(0,1024,1024),
                                  chunk=[1,1024,1024])
        self.assertTrue(f.valid)
        self.assertTrue(f.dtype=="uint16")
        self.assertTrue(f.shape==(0,1024,1024))
        self.assertTrue(f.size == 0)

    def test_multidim_creation_without_chunk_and_filter(self):
        comp = NXDeflateFilter()
        f = self.gf.create_field("data","uint16",
                                  shape=(0,1024,1024),
                                  filter=comp)
        self.assertTrue(f.valid)
        self.assertTrue(f.dtype=="uint16")
        self.assertTrue(f.shape==(0,1024,1024))
        self.assertTrue(f.size == 0)


    def test_multdim_creation_with_chunk_and_filter(self):
        comp = NXDeflateFilter()
        f = self.gf.create_field("data","uint16",
                                  shape=(0,1024,1024),
                                  chunk=[1,1024,1024],
                                  filter=comp)
        self.assertTrue(f.valid)
        self.assertTrue(f.dtype=="uint16")
        self.assertTrue(f.shape==(0,1024,1024))
        self.assertTrue(f.size == 0)

    def test_grow_field(self):
        f = self.gf.create_field("data","uint16",
                                 shape=(3,4))

        self.assertTrue(f.shape == (3,4))

        f.grow(0)
        self.assertTrue(f.shape == (4,4))
        f.grow(0,3)
        self.assertTrue(f.shape == (7,4))

        f.grow(1)
        self.assertTrue(f.shape == (7,5))
        f.grow(1,5)
        self.assertTrue(f.shape == (7,10))
