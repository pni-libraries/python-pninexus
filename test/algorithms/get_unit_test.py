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
# Created on: Sep 19, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
import unittest
import os
import numpy

from pni.io import ObjectError
from nx import nxfile
from nx import create_file
from nx import create_files
from nx import open_file
from nx import nxgroup
from nx import get_unit


#implementing test fixture
class get_unit_test(unittest.TestCase):
    filename = "get_unit_test.nxs"

    def setUp(self):
        self._file = create_file(self.filename,overwrite=True)

    def tearDown(self):
        self._file.flush()
        self._file.close()


    def test_with_attribute(self):
        #this should work as the file does not exist yet
        root = self._file.root()
        a = root.attributes.create("test_scalar","float32")
        self.assertRaises(TypeError,get_unit,a)

    def test_with_field(self):
        root = self._file.root()

        f = root.create_field("test_scalar","float64")
        f.attributes.create("units","string").write("nm")
    
        self.assertEqual(get_unit(f),"nm")

    def test_with_group(self):
        root = self._file.root()

        self.assertRaises(TypeError,get_unit,root)

            










