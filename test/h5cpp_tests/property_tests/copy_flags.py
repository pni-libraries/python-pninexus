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

from pninexus.h5cpp.property import CopyFlags, CopyFlag


class CopyFlagsTest(unittest.TestCase):

    def testCopyFlagsSettersAndGetters(self):

        flags = CopyFlags()
        self.assertFalse(flags.shallow_hierarchy)
        flags.shallow_hierarchy = True
        self.assertTrue(flags.shallow_hierarchy)

        self.assertFalse(flags.expand_soft_links)
        flags.expand_soft_links = True
        self.assertTrue(flags.expand_soft_links)

        self.assertFalse(flags.expand_external_links)
        flags.expand_external_links = True
        self.assertTrue(flags.expand_external_links)

        self.assertFalse(flags.expand_references)
        flags.expand_references = True
        self.assertTrue(flags.expand_references)

        self.assertFalse(flags.without_attributes)
        flags.without_attributes = True
        self.assertTrue(flags.without_attributes)

        self.assertFalse(flags.merge_committed_types)
        flags.merge_committed_types = True
        self.assertTrue(flags.merge_committed_types)

    def constructFromFlags(self):

        flags = CopyFlags(
            CopyFlag.SHALLOW_HIERARCHY | CopyFlag.WITHOUT_ATTRIBUTES)
        self.assertTrue(flags.shallow_hierarchy)
        self.assertFalse(flags.expand_soft_links)
        self.assertFalse(flags.expand_external_links)
        self.assertFalse(flags.expand_references)
        self.assertTrue(flags.without_attributes)
        self.assertFalse(flags.merge_committed_types)

    def testUnaryOperationsOnFlags(self):

        flags = CopyFlags()
        flags |= CopyFlag.MERGE_COMMITTED_TYPES
        flags |= CopyFlag.EXPAND_EXTERNAL_LINKS
        self.assertTrue(flags.expand_external_links)
        self.assertTrue(flags.merge_committed_types)

    def testBinaryOperationsOnFlags(self):

        f1 = CopyFlags(
            CopyFlag.SHALLOW_HIERARCHY | CopyFlag.WITHOUT_ATTRIBUTES)
        f2 = CopyFlags(CopyFlag.EXPAND_EXTERNAL_LINKS)
        f3 = f1 | f2

        self.assertTrue(f3.shallow_hierarchy)
        self.assertFalse(f3.expand_soft_links)
        self.assertTrue(f3.expand_external_links)
        self.assertFalse(f3.expand_references)
        self.assertTrue(f3.without_attributes)
        self.assertFalse(f3.merge_committed_types)

    def testBinaryFlagOperation(self):

        flags = CopyFlag.SHALLOW_HIERARCHY | CopyFlag.EXPAND_SOFT_LINKS
        self.assertTrue(isinstance(flags, CopyFlags))
