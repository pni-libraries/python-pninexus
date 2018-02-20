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
# Created on: Oct 3, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import numpy
import os

from pni.io.nx.h5 import nxfile
from pni.io.nx.h5 import nxgroup
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.io.nx.h5 import xml_to_nexus
from pni.io.nx.h5 import get_class
from pni.io.nx.h5 import get_path
from pni.io.nx.h5 import get_object

class creation_test(unittest.TestCase):
    file_path = os.path.split(__file__)[0]
    file_name = "creation_test.nxs"
    full_path = os.path.join(file_path,file_name)

    def setUp(self):
        self.nexus_file = create_file(self.full_path,overwrite=True)
        self.root = self.nexus_file.root()

    def tearDown(self):
        self.root.close()
        self.nexus_file.close()

    def test_default_construction(self):
        g = nxgroup()
        self.assertTrue(isinstance(g,nxgroup))
        self.assertFalse(g.is_valid)

    def test_only_group(self):
        g = self.root.create_group("simple")
        self.assertTrue(g.is_valid)
        self.assertEqual(g.name,"simple")
        self.assertEqual(g.path,"/simple")


    def test_group_with_positional_class(self):
        g = self.root.create_group("entry","NXentry")
        self.assertTrue(g.is_valid)
        self.assertEqual(g.name,"entry")
        self.assertEqual(str(g.path),"/entry:NXentry")

        self.assertEqual(g.attributes["NX_class"][...],"NXentry")

    def test_group_with_keyword_class(self):
        g = self.root.create_group("entry",nxclass="NXentry")
        
        self.assertTrue(g.is_valid)
        self.assertEqual(g.name,"entry")
        self.assertEqual(g.path,"/entry:NXentry")

        self.assertEqual(g.attributes["NX_class"][...],"NXentry")


    def test_group_from_element(self):
        g = self.root.create_group("entry:NXentry")
        self.assertTrue(g.is_valid)
        self.assertEqual(g.name,"entry")
        self.assertEqual(g.path,"/entry:NXentry")

        self.assertEqual(g.attributes["NX_class"][...],"NXentry")

    def test_fails_non_existing_intermediates(self):
        self.assertRaises(RuntimeError,
                          self.root.create_group,"/entry/instrument/detector")

    def test_success_with_existing_intermediates(self):
        self.root.create_group("entry:NXentry").\
                  create_group("instrument:NXinstrument")

        d = self.root.create_group("/entry/instrument/detector","NXdetector")
        self.assertTrue(d.is_valid)
        p = "/entry:NXentry/instrument:NXinstrument/detector:NXdetector"
        self.assertEqual(d.path,p)

