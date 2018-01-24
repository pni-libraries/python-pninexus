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

from pni.io.nexus import File
from pni.io.nexus import create_file
from pni.io.nexus import open_file
from pni.io.nexus import AccessFlags
import time
import re

class OpenTest(unittest.TestCase):
    file_path = os.path.split(__file__)[0]
    file_name = "OpenTest.nxs"
    full_path = os.path.join(file_path,file_name)

    def setUp(self):
        f = create_file(self.full_path,AccessFlags.TRUNCATE)
        f.close()


    #-------------------------------------------------------------------------
    def test_read_only(self):
        """
        Open a single file in read only mode.
        """
        f = open_file(self.full_path) 
        self.assertEqual(f.intent,AccessFlags.READONLY)

    #-------------------------------------------------------------------------
    def test_read_write(self):
        """
        Open a single file in read-write mode.
        """

        f = open_file(self.full_path,AccessFlags.READWRITE)
        self.assertEqual(f.intent,AccessFlags.READWRITE)
