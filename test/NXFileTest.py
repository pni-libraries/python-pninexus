import unittest

from pni.nx.h5 import NXFile
from pni.nx.h5 import create_file
from pni.nx.h5 import open_file


#implementing test fixture
class NXFileTest(unittest.TestCase):
    def setUp(self):
        self.gf = create_file("NXFileTest.h5",overwrite=True)

    def tearDown(self):
        self.gf.close()

    def test_creation(self):
        f = create_file("test.h5",overwrite=True)
        self.assertTrue(f.valid)
        f.close()
    
        self.assertRaises(IOError,create_file,"test.h5",False)
        f = open_file("test.h5",readonly=False)

    def test_attributes(self):
        s = "a string attribute"
        a = self.gf.attr("text","string")
        a.value = s
        self.assertTrue(a.value == s)


