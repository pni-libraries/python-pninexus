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
import os
from pni.io.nx.h5 import nxgroup
from pni.io.nx.h5 import create_file

module_path = os.path.dirname(os.path.abspath(__file__))

class CreationTests(unittest.TestCase):
    
    file_name = os.path.join(module_path,"CreationTests.nxs")
    

    def setUp(self):
        self.nexus_file = create_file(self.file_name,overwrite=True)
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
        self.assertEqual(str(g.path),"{filename}://simple".format(filename=self.file_name))


    def test_group_with_positional_class(self):
        g = self.root.create_group("entry","NXentry")
        self.assertTrue(g.is_valid)
        self.assertEqual(g.name,"entry")
        self.assertEqual(str(g.path),"{filename}://entry:NXentry".format(filename=self.file_name))

        self.assertEqual(g.attributes["NX_class"][...],"NXentry")

    def test_group_with_keyword_class(self):
        g = self.root.create_group("entry",nxclass="NXentry")
        
        self.assertTrue(g.is_valid)
        self.assertEqual(g.name,"entry")
        self.assertEqual(str(g.path),"{filename}://entry:NXentry".format(filename=self.file_name))

        self.assertEqual(g.attributes["NX_class"][...],"NXentry")


    def test_group_from_element(self):
        g = self.root.create_group("entry:NXentry")
        self.assertTrue(g.is_valid)
        self.assertEqual(g.name,"entry")
        self.assertEqual(str(g.path),"{filename}://entry:NXentry".format(filename=self.file_name))

        self.assertEqual(g.attributes["NX_class"][...],"NXentry")

    def test_fails_non_existing_intermediates(self):
        self.assertRaises(RuntimeError,
                          self.root.create_group,"/entry/instrument/detector")

    def test_success_with_existing_intermediates(self):
        self.root.create_group("entry:NXentry").\
                  create_group("instrument:NXinstrument")

        d = self.root.create_group("/entry/instrument/detector","NXdetector")
        self.assertTrue(d.is_valid)
        p = "{filename}://entry:NXentry/instrument:NXinstrument/detector:NXdetector".format(filename=self.file_name)
        self.assertEqual(str(d.path),p)

