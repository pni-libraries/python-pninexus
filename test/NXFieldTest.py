import unittest
import numpy

from pni.nx.h5 import NXFile
from pni.nx.h5 import NXGroup
from pni.nx.h5 import NXField
from pni.nx.h5 import create_file
from pni.nx.h5 import open_file

from pni.nx.h5 import ShapeMissmatchError


#implementing test fixture
class NXFieldTest(unittest.TestCase):
    def setUp(self):
        self.gf = create_file("NXFieldTest.h5",overwrite=True)

    def tearDown(self):
        self.gf.close()

    def test_creation(self):
        f = self.gf.create_field("data1","uint16")
        self.assertTrue(f.dtype=="uint16")
        self.assertTrue(f.shape==(1,))

        f = self.gf.create_field("/scan_1/instrument/detector/data",
                "int32",shape=(0,1024,1024))
        self.assertTrue(f.dtype=="int32")
        self.assertTrue(f.shape==(0,1024,1024))
        self.assertTrue(f.name == "data")
        self.assertTrue(f.base == "/scan_1/instrument/detector")
        self.assertTrue(f.path == "/scan_1/instrument/detector/data")

        #check for some errors
        #what if chunk shape and data shape do not have same rank
        self.assertRaises(ShapeMissmatchError,self.gf.create_field,
                "data2","float32",shape=(256,412),chunk=(256,))
        #check for unkown data type
        self.assertRaises(TypeError,self.gf.create_field,
                "data2","hallo")

    def test_numeric_io(self):
        f1 = self.gf.create_field("data1","float64",shape=(3,1))

        for i in range(len(f1.shape)):
            f1[i,:] = float(i)
            a = numpy.array([float(i)])
            f1[i,:] = a[:]
            f1[i,:] = a

        for i in range(len(f1.shape)):
            self.assertTrue(float(i)==f1[i,0])

        a = f1[...]
        self.assertTrue(a.shape == (3,1))

        f2 = self.gf.create_field("data2","float64",shape=(3,))
        self.assertTrue(f2.shape == (3,))
        f2[...] = 10.
        a = f2[...]
        self.assertTrue(a.shape == (3,))

    def test_negative_index(self):
        f1 = self.gf.create_field("data1","uint16",shape=(20,))

        f1[...] = numpy.arange(0,20,dtype="uint16")
        self.assertTrue(f1[-1] == 19)
        self.assertTrue(f1[-2] == 18)
        self.assertTrue(f1[-20] == 0)


            

        
    
#    def test_io(self):
#        f = self.gf.create_field("log","string")
#        f.write("hello world this is a text")
#        f.write("another text")
#        f[0] = "yet another text"
#
#        #try to write unicode
#        f.write(u"unicode text")
