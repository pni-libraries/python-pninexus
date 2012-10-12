import unittest

from pni.nx.h5 import NXFile
from pni.nx.h5 import create_file
from pni.nx.h5 import open_file
from pni.nx.h5 import NXFileError
from pni.nx.h5 import NXGroup

from AttributesTest import AttributeTest


#implementing test fixture
class NXFileTest(unittest.TestCase):
    attr_tester = AttributeTest()

    def setUp(self):
        print "setUp ...."
        self._file = create_file("NXFileTest.h5",overwrite=True)

    def tearDown(self):
        print "treaDown ..."
        self._file.flush()
        self._file.close()

    def test_creation(self):
        print "NXFileTest.test_creation() ............................"
        f = create_file("test.h5",overwrite=True)
        self.assertTrue(f.valid)
        self.assertFalse(f.readonly)
        f.close()
    
        self.assertRaises(NXFileError,create_file,"test.h5",False)
        f = open_file("test.h5",readonly=False)
        f.close()

    def test_scalar_attribute(self):
        print "NXFileTest.test_scalar_attributes() ...................."
        self.attr_tester.test_scalar_attribute(self,self._file)

    def test_array_attribute(self):
        print "NXFileTest.test_array_attribute() ......................"
        self.attr_tester.test_array_attribute(self,self._file)

    def test_group_iteration(self):
        print "NXFileTest.test_group_iteration() ......................"
        g=self._file.create_group("data1")
        self.assertTrue(g.valid)
        g=self._file.create_group("data2")
        self.assertTrue(g.valid)
        g=self._file.create_group("data3")
        self.assertTrue(g.valid)
        self.assertTrue(self._file.nchildren == 3)

        for g in self._file.children:
            self.assertTrue(isinstance(g,NXGroup))





