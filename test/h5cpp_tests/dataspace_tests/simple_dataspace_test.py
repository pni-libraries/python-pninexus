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

from pninexus.h5cpp.dataspace import Simple
from pninexus.h5cpp.dataspace import Dataspace
from pninexus.h5cpp.dataspace import Type


class TestSimple(unittest.TestCase):

    def test_default_construction(self):

        space = Simple()
        self.assertEqual(space.type, Type.SIMPLE)
        self.assertEqual(space.size, 0)
        self.assertEqual(space.current_dimensions, ())
        self.assertEqual(space.maximum_dimensions, ())
        self.assertTrue(isinstance(space, Simple))
        self.assertTrue(isinstance(space, Dataspace))

    def test_current_dimensions_construction(self):

        space = Simple((10, 20))
        self.assertEqual(space.size, 200)
        self.assertEqual(space.current_dimensions, (10, 20))
        self.assertEqual(space.maximum_dimensions, (10, 20))

    def test_all_dimensions_construction(self):

        space = Simple((1, 23), (100, 300))
        self.assertEqual(space.size, 23)
        self.assertEqual(space.current_dimensions, (1, 23))
        self.assertEqual(space.maximum_dimensions, (100, 300))
