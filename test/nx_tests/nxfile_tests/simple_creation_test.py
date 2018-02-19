#
# (c) Copyright 2015 DESY, 
#               2015 Eugen Wintersberger <eugen.wintersberger@desy.de>
#
# This file is part of python-pni.
#
# python-pni is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# python-pni is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with python-pni.  If not, see <http://www.gnu.org/licenses/>.
# ===========================================================================
#
# Created on: Oct 2, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import os
import numpy

from pni.io.nx.h5 import nxfile
from pni.io.nx.h5 import create_file
import time

module_path = os.path.dirname(os.path.abspath(__file__))

class SimpleCreationTest(unittest.TestCase):
    
    filename = os.path.join(module_path,"nxfile_creation_tests.nxs")

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
        self.assertRaises(RuntimeError,f.root)
        self.assertRaises(RuntimeError,lambda : f.readonly)

    
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
        self.assertRaises(RuntimeError,create_file,self.filename)

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


        
        
