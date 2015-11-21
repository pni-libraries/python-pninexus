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
# Created on: Oct 21, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
import pni.io.nx.h5 as nexus
import unittest 
import os

class issue_7_regression_test(unittest.TestCase):
    test_dir = os.path.split(__file__)[0]
    file_name = "issue_7_regression_test.nxs"
    full_path = os.path.join(test_dir,file_name)
    data_value = "12235556L"

    def setUp(self):
        self.gf = nexus.create_file(self.full_path,True)
        self.root = self.gf.root()

    def tearDown(self):
        self.root.close()
        self.gf.close()

    def test_field_setitem(self):
        target = self.root.create_field("target","uint64")
        self.assertRaises(TypeError,target.__setitem__,target,Ellipsis,self.data_value)

    def test_attribute_setitem(self):
        target = self.root.attributes.create("target","uint64")
        self.assertRaises(TypeError,target.__setitem__,target,Ellipsis,self.data_value)



