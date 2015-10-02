from __future__ import print_function
import unittest
import os
import numpy

from pni.io       import ObjectError
from pni.io       import InvalidObjectError
from pni.io.nx.h5 import nxfile
from pni.io.nx.h5 import create_file
import time

class simple_creation_test(unittest.TestCase):
    
    file_path = os.path.split(__file__)[0]
    filename = os.path.join(file_path,"nxfile_creation_tests.nxs")

    #-------------------------------------------------------------------------
    def tearDown(self):

        try:
            os.remove(self.filename)
        except:
            pass

    #-------------------------------------------------------------------------
    def test_invalid_file_object_access(self):
        """
        Tests checking the behavior of an instance of nxfile in the case that 
        it is not a valid object (typically after default construction).
        """

        f = nxfile()
        self.assertFalse(f.is_valid)
        self.assertRaises(InvalidObjectError,f.root)

        try:
            f.readonly
            self.assertTrue(False,"Access to the files readonly property did not throw!")
        except InvalidObjectError:
            self.assertTrue(True)

    
    #-------------------------------------------------------------------------
    def test_non_truncating(self):
        """
        Test non-truncating file creation. If a file already exists a
        subsequent create_file should throw an exception instead of 
        overwriting the existing file.
        """
        #create the file
        f = create_file(self.filename) 

        self.assertIsInstance(f,nxfile)
        self.assertTrue(f.is_valid)
        self.assertFalse(f.readonly)

        #close the file
        f.close()
        self.assertFalse(f.is_valid)

        #cannot create the file because it already exists
        self.assertRaises(ObjectError,create_file,self.filename)

    #-------------------------------------------------------------------------
    def test_truncating(self):
        """
        Test truncating file creation. If a file already exists the
        new file will overwrite the existing one. 
        """
        f = create_file(self.filename)
        f.close()
        t1 = os.path.getmtime(self.filename)

        time.sleep(1)
        f = create_file(self.filename,overwrite=True)
        self.assertTrue(f.is_valid)
        self.assertFalse(f.readonly)
        f.close()
        t2 = os.path.getmtime(self.filename)

        self.assertGreater(t2,t1)


        
        
