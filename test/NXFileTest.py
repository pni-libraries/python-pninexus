import unittest
import os

from pni.io.nx.h5 import NXFile
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.io.nx.h5 import NXFileError
from pni.io.nx.h5 import NXGroup

from AttributesTest import AttributeTest


#implementing test fixture
class NXFileTest(unittest.TestCase):
    attr_tester = AttributeTest()
    filename = "nxfiletest.nxs"
    filename2 = "nxfiletest2.nxs"

    def setUp(self):
        print "setUp ...."
        self._file = create_file(self.filename,overwrite=True)

    def tearDown(self):
        print "treaDown ..."
        self._file.flush()
        self._file.close()

        try:
            os.remove(self.filename)
        except:
            pass

        try:
            os.remove(self.filename2)
        except:
            pass

    def test_creation(self):
        print "NXFileTest.test_creation() ............................"

        #this should work as the file does not exist yet
        f = create_file(self.filename2)
        self.assertTrue(f.valid)
        self.assertFalse(f.readonly)
        f.close()
   
        #this will throw an exception as the file already exists
        self.assertRaises(NXFileError,create_file,self.filename2,False)
        #this should work now
        f = create_file(self.filename2,overwrite=True)
        self.assertTrue(f.valid)
        f.close()

    def test_open(self):
        print "NXFileTest::test_open() ................................"
        self._file.close() #close the original file

        #open the file in read only mode
        f = open_file(self.filename)
        try:
            f.create_group("group1")
            self.assertTrue(False)
        except:
            self.assertTrue(True)

        f.close()
        #open the file in read/write mode
        f = open_file(self.filename,readonly=False)
        g = f.create_group("group2")
        self.assertTrue(f.valid)
        self.assertTrue(g.valid)
        g.close()
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





