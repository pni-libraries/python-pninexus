import unittest
import numpy

from pni.nx.h5 import NXFile
from pni.nx.h5 import NXGroup
from pni.nx.h5 import create_file
from pni.nx.h5 import open_file
from pni.nx.h5 import NXGroupError
from pni.nx.h5 import NXAttributeError
from pni.nx.h5 import ShapeMissmatchError

def write_attribute(a,v):
    a.value = v
    return a


#implementing test fixture
class NXGroupTest(unittest.TestCase):
    def setUp(self):
        self.gf = create_file("NXGroupTest.h5",overwrite=True)

    def tearDown(self):
        self.gf.close()

    def test_creation(self):
        g = self.gf.create_group("metadata")
        g = self.gf.create_group("scan_1",nxclass="NXentry")
        self.assertTrue(g.attr("NX_class").value == "NXentry")

        g = g.create_group("instrument/detector")
        self.assertTrue(g.path == "/scan_1/instrument/detector")
        self.assertTrue(g.name == "detector")
        self.assertTrue(g.base == "/scan_1/instrument")

    def test_open(self):
        #try to open a group that does not exist
        self.assertRaises(NXGroupError,self.gf.open,"/data")

    def test_simple_attributes(self):
        #try to open an attribute that does not exist
        self.assertRaises(NXAttributeError,self.gf.attr,"bla")

        g = self.gf.create_group("metdata")
        a = g.attr("date","string")
        a.value = "12.4.1029"
        self.assertTrue(a.value == "12.4.1029")
        self.assertTrue(a.dtype == "string")
        self.assertTrue(a.shape == ())

        a = g.attr("counter","uint8")
        a.value = 10
        self.assertTrue(a.value == 10)
        self.assertTrue(a.dtype == "uint8")

        #try to write a numpy array to a scalar attribute
        d = numpy.zeros((10,10),"uint8")
        self.assertRaises(ShapeMissmatchError,write_attribute,a,d)

        #create an attribute of unknow type
        self.assertRaises(TypeError,g.attr,"hello","bla")


    def test_array_attributes(self):
        a1 = self.gf.attr("att1","float32",shape=[1,2,3])
        a2 = self.gf.attr("att2","int32",shape=(4,5))
       
        #perform broadcast write operations
        a2.value = 15
        #check for proper return type
        self.assertTrue(isinstance(a2.value,numpy.ndarray))
        self.assertTrue(all(v == 15 for v in a2.value.flat))

        d = numpy.zeros((10,10))
        self.assertRaises(ShapeMissmatchError,write_attribute,a2,d)




        

        


