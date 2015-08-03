import unittest
import numpy

import pni.io.nx.h5 as nx
from pni.io.nx.h5 import nxfile
from pni.io.nx.h5 import nxgroup
from pni.io.nx.h5 import nxfield
from pni.io.nx.h5 import deflate_filter
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file

#implementing test fixture
class nxfield_common_test(unittest.TestCase):

    
    def setUp(self):
        self.gf = create_file("nxfield_test.nxs",overwrite=True)
        self.root = self.gf.root()

    def tearDown(self):
        self.root.close()
        self.gf.close()


    def test_scalar_creation(self):
        f = self.root.create_field("data","uint16")
        self.assertTrue(f.is_valid)
        self.assertTrue(f.dtype=="uint16")
        self.assertTrue(f.shape==(1,))
        self.assertTrue(f.size == 1)

    def test_multidim_creation_without_chunk(self):
        f = self.root.create_field("data","uint16",
                                  shape=(0,1024,1024))
        self.assertTrue(f.is_valid)
        self.assertTrue(f.dtype=="uint16")
        self.assertTrue(f.shape==(0,1024,1024))
        self.assertTrue(f.size == 0)

    def test_multidim_creation_with_chunk(self):
        f = self.root.create_field("data","uint16",
                                  shape=(0,1024,1024),
                                  chunk=[1,1024,1024])
        self.assertTrue(f.is_valid)
        self.assertTrue(f.dtype=="uint16")
        self.assertTrue(f.shape==(0,1024,1024))
        self.assertTrue(f.size == 0)

    def test_multidim_creation_without_chunk_and_filter(self):
        comp = deflate_filter()
        f = self.root.create_field("data","uint16",
                                  shape=(0,1024,1024),
                                  filter=comp)
        self.assertTrue(f.is_valid)
        self.assertTrue(f.dtype=="uint16")
        self.assertTrue(f.shape==(0,1024,1024))
        self.assertTrue(f.size == 0)


    def test_multdim_creation_with_chunk_and_filter(self):
        comp = deflate_filter()
        f = self.root.create_field("data","uint16",
                                  shape=(0,1024,1024),
                                  chunk=[1,1024,1024],
                                  filter=comp)
        self.assertTrue(f.is_valid)
        self.assertTrue(f.dtype=="uint16")
        self.assertTrue(f.shape==(0,1024,1024))
        self.assertTrue(f.size == 0)

    def test_grow_field(self):
        f = self.root.create_field("data","uint16",shape=(3,4))

        self.assertTrue(f.shape == (3,4))

        f.grow(0)
        self.assertTrue(f.shape == (4,4))
        f.grow(0,3)
        self.assertTrue(f.shape == (7,4))

        f.grow(1)
        self.assertTrue(f.shape == (7,5))
        f.grow(1,5)
        self.assertTrue(f.shape == (7,10))
