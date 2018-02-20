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

class recursive_iterator_test(unittest.TestCase):
    file_path = os.path.split(__file__)[0]
    file_name = "recursive_iterator_test.nxs"
    full_path = os.path.join(file_path,file_name)

    def setUp(self):
        self.nexus_file = create_file(self.full_path,overwrite=True)
        self.root = self.nexus_file.root()
        self.paths = []
        e = self.root.create_group("entry:NXentry")
        self.paths.append(e.path)
        i = e.create_group("instrument:NXinstrument")
        self.paths.append(i.path)
        self.paths.append(i.create_group("detector:NXdetector").path)
        self.paths.append(i.create_group("monochromator:NXmonochromator").path)
        self.paths.append(i.create_group("source:NXsource").path)
        self.paths.append(e.create_group("data:NXdata").path)
        self.paths.append(e.create_group("sample:NXsample").path)
        self.paths.append(e.create_group("control:NXmonitor").path)
        self.paths.append(e.create_field("title","string").path)
        self.paths.append(e.create_field("experiment_identifier","string").path)
        self.paths.append(e.create_field("experiment_description","string").path)

        #self.paths.sort()

    def tearDown(self):
        self.root.close()
        self.nexus_file.close()

    def test_recursive_iteration(self):
       
        for c in self.root.recursive:
            self.paths.remove(c.path)

        self.assertEqual(len(self.paths),0)
