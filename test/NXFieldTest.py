import unittest

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

    def test_io(self):
        pass
    
        
