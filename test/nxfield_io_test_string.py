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
from . import data_generator as data_gen
from .nxfield_io_test import nxfield_io_test_unit8
from .nxfield_io_test import types
from .nxfield_io_test import scalars
from . import config

if config.PY_MAJOR_VERSION >= 3:

    #implementing test fixture
    class nxfield_io_test_string(nxfield_io_test_uint8):
        """
        Testing IO operations on a field. The possible operations are 
        
        memory    nexus
        scalar -> scalar 
        scalar <- scalar
        scalar -> array broadcast
        scalar -> array partial
        
        array  -> array (same shape)
        array  -> array (partial io)
        array  <- array (same shape)
        array  <- array (partial io)

        """

        _typecode = "string"

        def __init__(self,*args,**keyargs):
            self.filename = "nxfield_io_test_{tc}.nxs".format(tc=self._typecode)
            self.filename = os.path.join("test",self.filename)
            unittest.TestCase.__init__(self,*args,**keyargs)
            self.gf = create_file(self.filename,overwrite=True)
            self.gf.close()

        
        def setUp(self):
            self.gf = open_file(self.filename,readonly=False)
            self.root = self.gf.root()
            self.dg = data_gen.create(self._typecode,5,40)

        def tearDown(self):
            self.root.close()
            self.gf.close()

    #    def test_scalar_to_scalar_field_io(self):
    #        """
    #        IO of a scalar with a scalar field
    #        """
    #        f = self.root.create_field("scalar_to_scalar_field_io",
    #                                   self._typecode)
    #
    #        #use the read() and write() methods
    #        s_write = self.dg() #generate new input data
    #        f.write(s_write)
    #
    #        s_read = f.read()
    #        self.assertTrue(s_write == s_read)
    #
    #        #use the broadcast operators - the broadcast operator should also 
    #        #work for scalar fields
    #        s_write = self.dg() #generate new input data
    #        f[...] = s_write
    #        s_read = f[...]
    #        self.assertTrue(s_read == s_write)
    #
    #
    #    def test_scalar_to_mdim_field_broadcast_io(self):
    #        """
    #        Testing IO of a scalar broadcasted on a multidimensional field
    #        """
    #        f = self.root.create_field("scalar_to_mdim_field_broadcast",
    #                                   self._typecode,shape=(3,4))
    #
    #        s_write = self.dg()
    #        f[...] = s_write
    #        s_read = f[...]
    #   
    #        self.assertTrue(all(x==s_write for x in s_read.flat))
    #
    #    def test_scalar_to_mdim_field_partial_io(self):
    #        """
    #        Testing IO from a scalar to a mdim field using partial IO. 
    #        """
    #        f = self.root.create_field("scalar_to_mdim_field_partial",
    #                                   self._typecode,shape=(3,4))
    #        
    #        s1_write = self.dg()
    #        s2_write = self.dg()
    #        s3_write = self.dg()
    #        f[0,:] = s1_write
    #        f[1,:] = s2_write
    #        f[2,:] = s3_write
    #
    #        s1_read = f[0,...]
    #        s2_read = f[1,...]
    #        s3_read = f[2,...]
    #
    #        self.assertTrue(all(x==s1_write for x in s1_read.flat))
    #        self.assertTrue(all(x==s2_write for x in s2_read.flat))
    #        self.assertTrue(all(x==s3_write for x in s3_read.flat))
    #
    #        #writing a single element
    #        s1_write = self.dg()
    #        f[0,1] = s1_write
    #        read =  f[0,1]
    #        self.assertTrue(s1_write == read)
    #
    #    def test_array_to_mdim_field_io(self):
    #        shape = (3,4)
    #        f = self.root.create_field("array_to_mdim_field",
    #                                   self._typecode,shape=shape)
    #        write = self.dg(shape)
    #        f.write(write)
    #        read = f.read()
    #
    #        self.assertTrue(x==y for x,y in zip(read,write))
    #
    #        f[...] = write
    #        read = f[...]
    #
    #        self.assertTrue(x==y for x,y in zip(read,write))

    #    def test_array_to_mdim_field_partial_io(self):
    #        shape = (3,4)
    #        f = self.root.create_field("array_to_mdim_field_partial",
    #                                   self._typecode,shape=shape)
    #
    #        write1 = self.dg((4,))
    #        
    #        f[0,:] = write1
    #        read   = f[0,:]
    #
    #        self.assertTrue(x==y for x,y in zip(read,write1))
    #
    #        write2 = self.dg((3,))
    #        f[:,1] = write2
    #        read   = f[:,1]
    #
    #        self.assertTrue(x==y for x,y in zip(read,write2))

else:
    class nxfield_io_test_string(nxfield_io_test_uint8):
        _typecode="string"
