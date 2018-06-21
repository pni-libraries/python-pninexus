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
# Created on: Jan 29, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import os
from pninexus import h5cpp
from pninexus.h5cpp.file import AccessFlags
from pninexus.h5cpp.node import Group
# from pninexus.h5cpp.property import GroupCreationList
from pninexus.h5cpp.property import LinkCreationList
from pninexus.h5cpp import Path


module_path = os.path.dirname(os.path.abspath(__file__))


class GroupCreationTests(unittest.TestCase):

    filename = os.path.join(module_path, "GroupCreationTests.h5")

    def setUp(self):

        self.file = h5cpp.file.create(self.filename, AccessFlags.TRUNCATE)
        self.root = self.file.root()

    def tearDown(self):

        self.root.close()
        self.file.close()

    def testSimpleConstruction(self):
        """The most essential test case

        We simply construct a group with a new name
        """

        g = Group(self.root, "test")
        self.assertEqual(g.link.path.name, "test")
        self.assertEqual(g.link.path, Path("/test"))

    def testWithIntermediateGroups(self):
        """somehow a bit more elaborate

        We construct a group with a custom link creation property list
        """
        lcpl = LinkCreationList()
        lcpl.intermediate_group_creation = True
        g = Group(self.root, "test/sensors/temperature", lcpl)
        self.assertEqual(
            "{}".format(g.link.path), "/test/sensors/temperature")

        self.assertTrue(self.root.nodes.exists("test"))
        g = self.root.nodes["test"]
        self.assertTrue(g.nodes.exists("sensors"))
        g = g.nodes["sensors"]
        self.assertTrue(g.nodes.exists("temperature"))
