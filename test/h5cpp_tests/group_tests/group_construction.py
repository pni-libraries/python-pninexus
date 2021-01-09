#
# (c) Copyright 2018 DESY,
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
# Created on: Jan 25, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
# import os
# import numpy

from pninexus import h5cpp


class GroupConstructionTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        self.file = h5cpp.file.create(
            "GroupTest.h5", h5cpp.file.AccessFlags.TRUNCATE)

    def tearDown(self):

        self.file.close()

    def testDefaultConstruction(self):

        g = h5cpp.node.Group()
        self.assertFalse(g.is_valid)

    def testRootGroupFromFile(self):

        root = self.file.root()
        self.assertEqual(root.attributes.size, 0)

    def testConstructor(self):

        root = self.file.root()
        entry = h5cpp.node.Group(root, "entry")
        self.assertTrue(entry.is_valid)
        self.assertEqual(entry.type, h5cpp.node.Type.GROUP)
        self.assertEqual(entry.link.type(), h5cpp.node.LinkType.HARD)
        self.assertEqual(entry.link.path, h5cpp.Path('/entry'))
