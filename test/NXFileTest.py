import unittest

from pni.nx.h5 import NXFile
from pni.nx.h5 import create_file
from pni.nx.h5 import open_file
from pni.nx.h5 import NXFileError

from AttributesTest import AttributeTest


#implementing test fixture
class NXFileTest(unittest.TestCase):
    attr_tester = AttributeTest()

    def setUp(self):
        self._file = create_file("NXFileTest.h5",overwrite=True)

    def tearDown(self):
        self._file.flush()
        self._file.close()

    def test_creation(self):
        f = create_file("test.h5",overwrite=True)
        self.assertTrue(f.valid)
        self.assertFalse(f.readonly)
        f.close()
    
        self.assertRaises(NXFileError,create_file,"test.h5",False)
        f = open_file("test.h5",readonly=False)

    def test_scalar_attribute(self):
        self.attr_tester.test_scalar_attribute(self,self._file)

    def test_array_attribute(self):
        self.attr_tester.test_array_attribute(self,self._file)




