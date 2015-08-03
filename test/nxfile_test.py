import unittest
import os

from pni.io import ObjectError
from pni.io.nx.h5 import nxfile
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.io.nx.h5 import nxgroup


#implementing test fixture
class nxfile_test(unittest.TestCase):
    filename = "nxfiletest.nxs"
    filename2 = "nxfiletest2.nxs"

    def setUp(self):
        self._file = create_file(self.filename,overwrite=True)

    def tearDown(self):
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
        #this should work as the file does not exist yet
        f = create_file(self.filename2)
        self.assertTrue(f.is_valid)
        self.assertFalse(f.readonly)
        f.close()
   
        #this will throw an exception as the file already exists
        self.assertRaises(ObjectError,create_file,self.filename2,False)
        #this should work now
        f = create_file(self.filename2,overwrite=True)
        self.assertTrue(f.is_valid)
        f.close()

    def test_open(self):
        #open the file in read only mode
        f = open_file(self.filename)
        self.assertTrue(f.readonly)
        root = f.root()
        self.assertRaises(ObjectError,root.attributes.create,
                          "temperature","uint16")
        f.close()
        #open the file in read/write mode
        f = open_file(self.filename,readonly=False)
        self.assertTrue(f.is_valid)
        f.close()






