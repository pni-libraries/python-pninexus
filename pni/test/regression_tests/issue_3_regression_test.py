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
import pni.io.nx.h5 as nexus
import numpy
import unittest
import os 

class issue_3_regression_test(unittest.TestCase):
    test_dir = os.path.split(__file__)[0]
    file_name = "issue_3_regression_test.nxs"
    full_path = os.path.join(test_dir,file_name)
    shape = (1,10)

    def setUp(self):
        self.gf = nexus.create_file(self.full_path,True)
        self.data = numpy.ones(self.shape,dtype="uint16")
        self.root = self.gf.root()

    def tearDown(self):
        self.root.close()
        self.gf.close()

    def test_attributes(self):
        a = self.root.attributes.create("test","uint16",shape=self.shape)
        a[...] = self.data

        for (i,o) in zip(self.data.flat,a[...].flat):
            self.assertEqual(i,o)

    def test_fields(self):
        f = self.root.create_field("test","uint16",shape=self.shape)
        f[...] = self.data

        for (i,o) in zip(self.data.flat,f[...].flat):
            self.assertEqual(i,o)




