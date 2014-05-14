import unittest
import numpy
import numpy.random as random

import pni.io.nx.h5 as nx
from pni.io.nx.h5 import NXFile
from pni.io.nx.h5 import NXGroup
from pni.io.nx.h5 import NXField
from pni.io.nx.h5 import NXDeflateFilter
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file

types=["uint8","int8","uint16","int16","uint32","int32","uint64","int64",
       "float32","float64","float128","string"]

scalars={"uint8":numpy.uint8,"int8":numpy.int8,
        "uint16":numpy.uint16,"int16":numpy.int16,
        "uint32":numpy.uint32,"int32":numpy.int32,
        "uint64":numpy.uint64,"int64":numpy.int64,
        "float32":numpy.float32,"float64":numpy.float64,
        "float128":numpy.float128,
        "string":numpy.str_}
         


#implementing test fixture
class nxfield_test(unittest.TestCase):

    _typecode = "uint16"
    
    def setUp(self):
        self.gf = create_file("NXFieldTest.h5",overwrite=True)
        self.root = self.gf["/"]

    def tearDown(self):
        self.root.close()
        self.gf.close()

    def test_scalar_to_scalar_field_io(self):
        """
        IO of a scalar with a scalar field
        """
        f = self.gf.create_field("data",self._typecode)

        s_write = scalars[self._typecode](random.rand())
        f.write(s_write)

        s_read = f.read()
        self.assertTrue(s_write == s_read)

        s_write = scalars[self._typecode](random.rand())
        f[...] = s_write
        s_read = f[...]
        self.assertTrue(s_read == s_write)


    def test_scalar_to_mdim_field_broadcast_io(self):
        """
        Testing IO of a scalar broadcasted on a multidimensional field
        """
        f = self.gf.create_field("data",self._typecode,
                                 shape=(3,4))

        s_write = scalars[self._typecode](random.rand())
        f[...] = s_write
        s_read = f[...]
    
        for x in s_read.flat: 
            self.assertTrue(x==s_write)

    def test_scalar_to_mdim_field_partial_io(self):
        f = self.gf.create_field("data",self._typecode,
                                 shape=(3,4))
        
        s1_write = scalars[self._typecode](random.rand())
        s2_write = scalars[self._typecode](random.rand())
        s3_write = scalars[self._typecode](random.rand())
        f[0,:] = s1_write
        f[1,:] = s2_write
        f[2,:] = s3_write

        s1_read = f[0,...]
        s2_read = f[1,...]
        s3_read = f[2,...]

        self.assertTrue(all(x==s1_write for x in s1_read.flat))
        self.assertTrue(all(x==s2_write for x in s2_read.flat))
        self.assertTrue(all(x==s3_write for x in s3_read.flat))

    def test_array_to_mdim_field_io(self):
        pass

    def test_array_to_mdim_field_partial_io(self):
        pass



    def test_numeric_io(self):
        f1 = self.gf.create_field("data1","float64",shape=(3,1))
        self.assertTrue(f1.valid)
        self.assertTrue(len(f1.shape) == 2)
        self.assertTrue(f1.size == 3)


        #write data
        for i in range(f1.shape[0]):
            f1[i,:] = float(i) # broadcast a single value
            a = numpy.array([float(i)])
            f1[i,:] = a[:] #broadcast an array slice
            f1[i,:] = a  #broadcast an entire array

        #read data back
        for i in range(f1.shape[0]):
            self.assertTrue(float(i)==f1[i,0])

        a = f1[...]
        self.assertTrue(a.shape == (3,))

        f2 = self.gf.create_field("data2","float64",shape=(3,))
        self.assertTrue(f2.shape == (3,))
        self.assertTrue(f2.size == 3)
        self.assertTrue(len(f2.shape) == 1)

        f2[...] = 10.
        a = f2[...]
        self.assertTrue(a.shape == (3,))

    def test_negative_index(self):
        f1 = self.gf.create_field("data1","uint16",shape=(20,))

        f1[...] = numpy.arange(0,20,dtype="uint16")

        #test for a single negative index
        self.assertTrue(f1[-1] == 19)
        self.assertTrue(f1[-2] == 18)
        self.assertTrue(f1[-20] == 0)

        #check for slices with negative indices
        a = f1[-10:-2]
        self.assertTrue(a[0] == 10)
        self.assertTrue(a[-1] == 17)

        f1[-10:-5] = numpy.arange(100,105)
        self.assertTrue(f1[-10] == f1[10])
        self.assertTrue(f1[10] == 100)
    
    def test_io(self):
        f = self.gf.create_field("log","string")
        f.write("hello world this is a text")
        f.write("another text")
        f[0] = "yet another text"

        #try to write unicode
        f.write(u"unicode text")

    def test_string_array(self):
        f = self.gf.create_field("text","string",shape=(2,2))
        data = numpy.array([["hello","world"],["this","is a text"]])
        f.write(data)

        f.close()

