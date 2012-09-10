import unittest

from pni.nx.h5 import NXFile
from pni.nx.h5 import create_file
from pni.nx.h5 import open_file
from pni.nx.h5 import NXFileError


#implementing test fixture
class NXFileTest(unittest.TestCase):
    def setUp(self):
        self.gf = create_file("NXFileTest.h5",overwrite=True)

    def tearDown(self):
        self.gf.close()

#    def test_creation(self):
#        f = create_file("test.h5",overwrite=True)
#        self.assertTrue(f.valid)
#        self.assertFalse(f.readonly)
#        f.close()
#    
#        self.assertRaises(NXFileError,create_file,"test.h5",False)
#        f = open_file("test.h5",readonly=False)

    def test_attributes(self):
        s = "a string attribute"
        a = self.gf.attr("text","string")
        self.assertTrue(a.dtype == "string")
        self.assertTrue(a.valid)
        self.assertTrue(a.shape == ())
        self.assertTrue(a.name == "text")
        a.value = s
        self.gf.flush()
        print a.value
        #self.assertTrue(a.value == s)
        a = self.gf.attr("number","uint16")
        a.value = 12
        print a.value



