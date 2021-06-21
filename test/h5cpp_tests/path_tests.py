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
# Created on: Jan 9, 2021
#     Authors:
#             Eugen Wintersberger <eugen.wintersberger@desy.de>
#             Jan Kotanski <jan.kotanski@desy.de>
#
from __future__ import print_function
import unittest
import os
from pninexus import h5cpp
from pninexus.h5cpp.file import AccessFlags

module_path = os.path.dirname(os.path.abspath(__file__))


class H5cppPathTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(H5cppPathTests, cls).setUpClass()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_hdf5_version(self):

        hdf5ver = h5cpp.current_library_version()
        mj, mn, pa = hdf5ver.split(".")
        imj = int(mj)
        imn = int(mn)
        ipa = int(pa)
        self.assertTrue(imj > 0)
        self.assertTrue(imn > -1)
        self.assertTrue(ipa > -1)

    def test_default_contruction(self):

        p = h5cpp.Path()
        self.assertEqual(p.size, 0)
        self.assertFalse(p.absolute)

    def test_contruction_from_string(self):

        p = h5cpp.Path("/hello/world/data")
        self.assertEqual(p.size, 3)
        self.assertTrue(p.absolute)
        self.assertFalse(p.is_root())

        p = h5cpp.Path("hello/world/data")
        self.assertEqual(p.size, 3)
        self.assertFalse(p.absolute)
        self.assertFalse(p.is_root())
        self.assertEqual("hello/world/data", str(p))
        p.absolute = True
        self.assertTrue(p.absolute)
        self.assertEqual("/hello/world/data", str(p))

        p = h5cpp.Path("hello/world/instrument/data")
        self.assertEqual(p.size, 4)
        self.assertFalse(p.absolute)
        self.assertFalse(p.is_root())

        p = h5cpp.Path(".")
        self.assertEqual(p.size, 0)
        self.assertFalse(p.absolute)
        self.assertFalse(p.is_root())

        p = h5cpp.Path("/")
        self.assertEqual(p.size, 0)
        self.assertTrue(p.absolute)
        self.assertTrue(p.is_root())
