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
# Created on: Jan 26, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
import unittest
from pninexus.h5cpp.property import ObjectCopyList, CopyFlags, CopyFlag


class ObjectCopyListTest(unittest.TestCase):

    def testDefaultConstruction(self):

        list = ObjectCopyList()
        flags = list.flags
        self.assertFalse(flags.shallow_hierarchy)
        self.assertFalse(flags.expand_soft_links)
        self.assertFalse(flags.expand_external_links)
        self.assertFalse(flags.expand_references)
        self.assertFalse(flags.without_attributes)
        self.assertFalse(flags.merge_committed_types)

    def testFlagsAssigment(self):

        list = ObjectCopyList()
        list.flags = CopyFlag.EXPAND_REFERENCES | \
            CopyFlag.MERGE_COMMITTED_TYPES
        self.assertTrue(list.flags.expand_references)
        self.assertTrue(list.flags.merge_committed_types)

    def testSingleFlagAssignment(self):

        list = ObjectCopyList()

        list.flags = CopyFlags() | CopyFlag.EXPAND_REFERENCES
        self.assertTrue(list.flags.expand_references)
