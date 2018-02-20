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
from pni.io.nx import make_relative 

module_path = os.path.dirname(os.path.abspath(__file__))

class ChildAccessTest(unittest.TestCase):
    
    file_name = os.path.join(module_path,"child_access_test.nxs")
    

    def setUp(self):
        self.nexus_file = create_file(self.file_name,overwrite=True)
        self.root = self.nexus_file.root()
        e = self.root.create_group("entry:NXentry")
        i = e.create_group("instrument:NXinstrument")
        e.create_group("data:NXdata")
        e.create_group("sample:NXsample")
        e.create_group("control:NXmonitor")
        e.create_field("title","string")
        e.create_field("experiment_identifier","string")
        e.create_field("experiment_description","string")

        self.names = ["instrument:NXinstrument",
                      "data:NXdata",
                      "sample:NXsample",
                      "control:NXmonitor",
                      "title",
                      "experiment_identifier",
                      "experiment_description"]
        self.names.sort()

    def tearDown(self):
        self.root.close()
        self.nexus_file.close()

    def test_loop_by_index(self):
        e = get_object(self.root,"entry:NXentry")
    
        self.assertEqual(len(e),7)
        for index in range(len(e)):
            child = e[index]
            self.assertEqual(make_relative(e.path,child.path),
                             self.names[index])

    def test_loop_by_name(self):

        e = get_object(self.root,"entry:NXentry")
        
        for (name,nref) in zip(e.names(),self.names):
            child = e[name]
            self.assertEqual(make_relative(e.path,child.path),nref)

    def test_index_error(self):
        e = get_object(self.root,":NXentry")
        self.assertRaises(IndexError,e.__getitem__,len(e)+1)

    def test_key_error(self):
        e = get_object(self.root,":NXentry")
        self.assertRaises(RuntimeError,e.__getitem__,"bla")




