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
# Created on: Oct 13, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import numpy

from pni.core import ShapeMismatchError
from pni.io.nx import nxpath
from pni.io.nx import make_path
from pni.io.nx import is_root_element
from pni.io.nx import is_absolute
from pni.io.nx import is_empty
from pni.io.nx import has_name
from pni.io.nx import has_class
from pni.io.nx import match


#implementing test fixture
class nxpath_match_test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_match_path_path(self):
        a = make_path("/:NXentry/:NXinstrument/:NXdetector")
        b = make_path("/scan_1:NXentry/p08:NXinstrument/mythen:NXdetector")
        self.assertTrue(match(a,b))

        b.pop_front()
        self.assertTrue(not match(a,b))
        b.push_front("/")
        b.pop_back()
        self.assertTrue(not match(a,b))


        
