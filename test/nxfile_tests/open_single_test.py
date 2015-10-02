from __future__ import print_function
import unittest
import os

from pni.io       import ObjectError
from pni.io       import InvalidObjectError
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
import time
import re

class open_single_test(unittest.TestCase):
    file_path = os.path.split(__file__)[0]
    file_name = "open_single_test.nxs"
    full_path = os.path.join(file_path,file_name)

    def setUp(self):
        f = create_file(self.full_path,overwrite=True)
        f.close()


    #-------------------------------------------------------------------------
    def test_read_only(self):
        """
        Open a single file in read only mode.
        """
        f = open_file(self.full_path) 
        self.assertTrue(f.readonly)
        r = f.root()

        self.assertRaises(ObjectError,r.create_group,"entry","NXentry")

    #-------------------------------------------------------------------------
    def test_read_write(self):
        """
        Open a single file in read-write mode.
        """

        f = open_file(self.full_path,readonly=False)
        self.assertFalse(f.readonly)
        r = f.root()

        r.create_group("entry","NXentry")

