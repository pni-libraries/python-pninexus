import unittest
import numpy

import pni.nx.h5 as nx
from pni.nx.h5 import NXFile
from pni.nx.h5 import NXGroup
from pni.nx.h5 import NXField
from pni.nx.h5 import create_file
from pni.nx.h5 import open_file

#implementing test fixture
class NXFieldTest(unittest.TestCase):
    def setUp(self):
        print "run NXFieldTest.setUp() ......................"
        self.gf = create_file("NXFieldTest.h5",overwrite=True)

    def tearDown(self):
        print "run NXFieldTest.tearDown() .................."
        self.gf.close()

    def test_creation(self):
        print "run NXFieldTest.test_creation() ................."
        f = self.gf.create_field("data1","uint16")
        self.assertTrue(f.valid)
        self.assertTrue(f.dtype=="uint16")
        self.assertTrue(f.shape==(1,))
        self.assertTrue(f.size == 1)

        f = self.gf.create_field("/scan_1/instrument/detector/data",
                "int32",shape=(0,1024,1024))
        self.assertTrue(f.valid)
        self.assertTrue(f.dtype=="int32")
        self.assertTrue(f.shape==(0,1024,1024))
        self.assertTrue(f.name == "data")
        self.assertTrue(f.base == "/scan_1/instrument/detector")
        self.assertTrue(f.path == "/scan_1/instrument/detector/data")
        self.assertTrue(f.size == 0)

        #check for some errors
        #what if chunk shape and data shape do not have same rank
        self.assertRaises(nx.ShapeMissmatchError,self.gf.create_field,
                "data2","float32",shape=(256,412),chunk=(256,))
        #check for unkown data type
        self.assertRaises(nx.TypeError,self.gf.create_field,
                "data2","hallo")


    def test_numeric_io(self):
        print "run NXFieldTest.test_numeric_io() .............."
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

        f2 = self.gf.create_field("data2","float64",shape=(3,1))
        self.assertTrue(f2.shape == (3,1))
        self.assertTrue(f2.size == 3)
        self.assertTrue(len(f2.shape) == 2)

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
