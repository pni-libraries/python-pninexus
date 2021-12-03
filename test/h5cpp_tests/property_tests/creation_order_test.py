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

from pninexus.h5cpp.property import CreationOrder


class CreationOrderTest(unittest.TestCase):

    def testDefaultConstruction(self):

        order = CreationOrder()

        self.assertFalse(order.tracked)
        self.assertFalse(order.indexed)

    def testTrackedSetter(self):

        order = CreationOrder()
        order.tracked = True
        self.assertTrue(order.tracked)
        self.assertFalse(order.indexed)

    def testIndexedSetter(self):

        order = CreationOrder()
        order.indexed = True

        #
        # setting indexed to true implies tracking order
        #
        self.assertTrue(order.tracked)
        self.assertTrue(order.indexed)
