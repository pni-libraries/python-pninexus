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
#regresssion test for issue 53

import unittest

import nx as nx
from pni.core import SizeMismatchError

class Issue_53_Test(unittest.TestCase):
    def setUp(self):
        self.nxfile = nx.create_file("Issue_53_Test.nx",overwrite=True)

    def tearDown(self):
        self.nxfile.close()

    def test_issue(self):
        deflate = nx.deflate_filter()
        deflate.rate = 5
        deflate.shuffle = True
        root = self.nxfile.root()

        self.assertRaises(SizeMismatchError,
                          root.create_field,"test","string",[],[],deflate)


