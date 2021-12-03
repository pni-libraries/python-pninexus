#
# (c) Copyright 2018 DESY
#
# This file is part of python-pninexus.
#
# python-pninexus is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# python-pninexus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with python-pninexus.  If not, see <http://www.gnu.org/licenses/>.
# ===========================================================================
#
# Created on: Feb 22, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import os

from pninexus import h5cpp
from pninexus import nexus


module_path = os.path.dirname(os.path.abspath(__file__))


class Issue23Regression(unittest.TestCase):

    filename = os.path.join(module_path, "Issue23Regression.nxs")
    group_name = h5cpp.Path("scan_2017-06-19T07:47:35.554207+0200")

    def setUp(self):
        unittest.TestCase.setUp(self)

        self.file = nexus.create_file(
            self.filename, h5cpp.file.AccessFlags.TRUNCATE)
        self.root = self.file.root()

    def tearDown(self):

        self.root.close()
        self.file.close()

    def test_creation_fail_with_base_class_factory(self):
        """
        The creation of an ill named group should not work using
        the BaseClassFactory
        """

        self.assertRaises(RuntimeError, nexus.BaseClassFactory.create,
                          self.root,
                          self.group_name,
                          "NXentry")

    def test_opening(self):
        """
        it should be possible to create such a group with HDF5 -
        still have to handle it for opening
        """

        h5cpp.node.Group(self.root, str(self.group_name))

        g = self.root.nodes[str(self.group_name)]
        self.assertEqual(g.link.path, h5cpp.Path("/") + self.group_name)
