from __future__ import print_function
import unittest
import os
import numpy

from pni.io       import ObjectError
from pni.io       import InvalidObjectError
from pni.io.nx.h5 import nxfile
from pni.io.nx.h5 import create_files
import time
import re

fileexp = re.compile(r"nxfile_creation_tests\.\d+\.nxs")
firstexp = re.compile(r"nxfile_creation_tests\.0+\.nxs")

class split_creation_test(unittest.TestCase):
    file_path = os.path.split(__file__)[0]
    filename = os.path.join(file_path,"nxfile_creation_tests.%05i.nxs")

    nx = 1024
    ny = 2048
    nframes = 10
    split_size = 20 #split size in MByte

    #-------------------------------------------------------------------------
    def _fill_with_data(self,f,field):
        """
        Private utility function filling a field within the file with some 
        data.
        """

        frame = numpy.ones((self.nx,self.ny),dtype="uint64")
        for i in range(self.nframes):
            field.grow()
            field[-1,...] = frame
            f.flush()

    #-------------------------------------------------------------------------
    def _get_files(self):
        """
        Generator function returning the created files. 
        """
        for x in os.listdir(self.file_path):
            if fileexp.match(os.path.split(x)[1]):
                yield os.path.join(self.file_path,x)

    #-------------------------------------------------------------------------
    def _create_files(self,ov=False):
        """
        Private utility function which creates the file family filled with
        data. 
        """
        f = create_files(self.filename,self.split_size,overwrite=ov)
        g = f.root();
        d = g.create_field("data","uint64",shape=(0,self.nx,self.ny))

        self._fill_with_data(f,d)

        f.close()

    #-------------------------------------------------------------------------
    def tearDown(self):
        """
        Remove all the temporary files after the test.
        """

        for f in self._get_files():
            try:
                os.remove(f)
            except:
                pass


    #-------------------------------------------------------------------------
    def test_non_truncating(self):
        """
        Test non-truncating creation of split files. The tests creates a set of
        files (filled with some random data). Afterwards it tries to open a new 
        file familiy - which should fail. 
        """
        self._create_files()
        #check size of all files
        print("\n")
        for f in self._get_files():
            print(f)
            self.assertTrue(os.stat(f).st_size != 0)

        self.assertRaises(ObjectError,create_files,
                          self.filename,self.split_size)


    #-------------------------------------------------------------------------
    def test_truncating(self):
        self._create_files()

        f = create_files(self.filename,self.split_size,overwrite=True)

        for f in self._get_files():
            if firstexp.match(os.path.split(f)[1]):
                self.assertTrue(os.stat(f).st_size != 0)
            else:
                self.assertTrue(os.stat(f).st_size == 0)


