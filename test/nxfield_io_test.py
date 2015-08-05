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
class nxfield_io_test_uint8(unittest.TestCase):
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

    _typecode = "uint8"

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

    def test_scalar_to_scalar_field_io(self):
        """
        IO of a scalar with a scalar field
        """
        f = self.root.create_field("scalar_to_scalar_field_io",
                                   self._typecode)

        #use the read() and write() methods
        s_write = self.dg() #generate new input data
        f.write(s_write)

        s_read = f.read()
        self.assertTrue(s_write == s_read)

        #use the broadcast operators - the broadcast operator should also 
        #work for scalar fields
        s_write = self.dg() #generate new input data
        f[...] = s_write
        s_read = f[...]
        self.assertTrue(s_read == s_write)


    def test_scalar_to_mdim_field_broadcast_io(self):
        """
        Testing IO of a scalar broadcasted on a multidimensional field
        """
        f = self.root.create_field("scalar_to_mdim_field_broadcast",
                                   self._typecode,shape=(3,4))

        s_write = self.dg()
        f[...] = s_write
        s_read = f[...]
   
        self.assertTrue(all(x==s_write for x in s_read.flat))

    def test_scalar_to_mdim_field_partial_io(self):
        """
        Testing IO from a scalar to a mdim field using partial IO. 
        """
        f = self.root.create_field("scalar_to_mdim_field_partial",
                                   self._typecode,shape=(3,4))
        
        s1_write = self.dg()
        s2_write = self.dg()
        s3_write = self.dg()
        f[0,:] = s1_write
        f[1,:] = s2_write
        f[2,:] = s3_write

        s1_read = f[0,...]
        s2_read = f[1,...]
        s3_read = f[2,...]

        self.assertTrue(all(x==s1_write for x in s1_read.flat))
        self.assertTrue(all(x==s2_write for x in s2_read.flat))
        self.assertTrue(all(x==s3_write for x in s3_read.flat))

        #writing a single element
        s1_write = self.dg()
        f[0,1] = s1_write
        read =  f[0,1]
        self.assertTrue(s1_write == read)

    def test_array_to_mdim_field_io(self):
        shape = (3,4)
        f = self.root.create_field("array_to_mdim_field",
                                   self._typecode,shape=shape)
        write = self.dg(shape)
#        f.write(write)
#        read = f.read()
#
#        self.assertTrue(x==y for x,y in zip(read,write))
#
#        f[...] = write
#        read = f[...]
#
#        self.assertTrue(x==y for x,y in zip(read,write))

    def test_array_to_mdim_field_partial_io(self):
        shape = (3,4)
        f = self.root.create_field("array_to_mdim_field_partial",
                                   self._typecode,shape=shape)

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

class nxfield_io_test_uint16(nxfield_io_test_uint8):
    _typecode = "uint16"

class nxfield_io_test_uint32(nxfield_io_test_uint8):
    _typecode = "uint32"

class nxfield_io_test_uint64(nxfield_io_test_uint8):
    _typecode = "uint64"

class nxfield_io_test_int8(nxfield_io_test_uint8):
    _typecode = "int8"

class nxfield_io_test_int16(nxfield_io_test_uint8):
    _typecode = "int16"

class nxfield_io_test_int32(nxfield_io_test_uint8):
    _typecode = "int32"

class nxfield_io_test_int64(nxfield_io_test_uint8):
    _typecode = "int64"

class nxfield_io_test_float32(nxfield_io_test_uint8):
    _typecode = "float32"

class nxfield_io_test_float64(nxfield_io_test_uint8):
    _typecode = "float64"

class nxfield_io_test_float128(nxfield_io_test_uint8):
    _typecode = "float128"

class nxfield_io_test_complex32(nxfield_io_test_uint8):
    _typecode = "complex32"

class nxfield_io_test_complex64(nxfield_io_test_uint8):
    _typecode = "complex64"

class nxfield_io_test_complex128(nxfield_io_test_uint8):
    _typecode = "complex128"

class nxfield_io_test_string(nxfield_io_test_uint8):
    _typecode = "string"

class nxfield_io_test_bool(nxfield_io_test_uint8):
    _typecode = "bool"
