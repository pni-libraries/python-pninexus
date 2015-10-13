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


        
