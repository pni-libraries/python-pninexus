#
# (c) Copyright 2015 DESY,
#               2015 Eugen Wintersberger <eugen.wintersberger@desy.de>
#
# This file is part of python-pninexus.
#
# python-pninexus is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# python-pninexus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with python-pninexus.  If not, see <http://www.gnu.org/licenses/>.
# ===========================================================================
#
# Created on: Oct 2, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import os

# from pninexus.h5cpp.file import File
from pninexus.h5cpp.file import create
from pninexus.h5cpp.file import open
from pninexus.h5cpp.file import AccessFlags
# import time
# import re


class OpenTest(unittest.TestCase):
    file_path = os.path.split(__file__)[0]
    file_name = "OpenTest.h5"
    full_path = os.path.join(file_path, file_name)

    def setUp(self):
        f = create(self.full_path, AccessFlags.TRUNCATE)
        f.close()

    # ------------------------------------------------------------------------
    def test_read_only(self):
        """
        Open a single file in read only mode.
        """
        f = open(self.full_path)
        self.assertEqual(f.intent, AccessFlags.READONLY)

    # ------------------------------------------------------------------------
    def test_read_write(self):
        """
        Open a single file in read-write mode.
        """

        f = open(self.full_path, AccessFlags.READWRITE)
        self.assertEqual(f.intent, AccessFlags.READWRITE)
