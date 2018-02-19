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

from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
import time
import re

module_path = os.path.dirname(os.path.abspath(__file__))

class OpenSingleTest(unittest.TestCase):
    filename = os.path.join(module_path,"open_single_test.nxs")

    def setUp(self):
        f = create_file(self.filename,overwrite=True)
        f.close()


    #-------------------------------------------------------------------------
    def test_read_only(self):
        """
        Open a single file in read only mode.
        """
        f = open_file(self.filename) 
        self.assertTrue(f.readonly)
        r = f.root()

        self.assertRaises(RuntimeError,r.create_group,"entry","NXentry")

    #-------------------------------------------------------------------------
    def test_read_write(self):
        """
        Open a single file in read-write mode.
        """

        f = open_file(self.filename,readonly=False)
        self.assertFalse(f.readonly)
        r = f.root()

        r.create_group("entry","NXentry")

