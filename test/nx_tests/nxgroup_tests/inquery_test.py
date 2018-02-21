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
# Created on: Oct 5, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import os

from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import get_object

module_path = os.path.dirname(os.path.abspath(__file__))

class InQueryTest(unittest.TestCase):
    
    file_name = os.path.join(module_path,"inquery_test.nxs")

    def setUp(self):
        self.nexus_file = create_file(self.file_name,overwrite=True)
        self.root = self.nexus_file.root()
        e = self.root.create_group("entry:NXentry")
        i = e.create_group("instrument:NXinstrument")
        e.create_group("data:NXdata")
        e.create_group("sample:NXsample")
        e.create_group("control:NXmonitor")

    def tearDown(self):
        self.root.close()
        self.nexus_file.close()


    def test_name_property(self):
        g = get_object(self.root,"/:NXentry")
        self.assertEqual(self.root.name,"/")
        self.assertEqual(get_object(self.root,"/:NXentry").name,"entry")
        self.assertEqual(get_object(self.root,"/").name,"/")
        self.assertEqual(get_object(self.root,"/:NXentry/:NXinstrument").name,
                         "instrument")

    def test_parent_property(self):
        g = get_object(self.root,"/:NXentry/:NXinstrument")
        self.assertEqual(g.parent.name,"entry")
        self.assertEqual(g.parent.parent.name,"/")

    def test_filename_property(self):
        g = get_object(self.root,"/:NXentry/:NXinstrument")
        self.assertEqual(g.filename,self.file_name)

    def test_size_property(self):
        self.assertEqual(self.root.size,1)
        self.assertEqual(get_object(self.root,"/:NXentry").size,4)
        self.assertEqual(get_object(self.root,"/:NXentry/:NXdata").size,0)

    def test_path_property(self):
        g = get_object(self.root,"/:NXentry/:NXinstrument/")
        self.assertEqual(g.path,"{filename}://entry:NXentry/instrument:NXinstrument".format(filename=self.file_name))



