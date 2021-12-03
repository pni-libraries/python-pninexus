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
# Created on: Feb 8, 2018
#     Authors:
#             Eugen Wintersberger <eugen.wintersberger@desy.de>
#             Jan Kotanski <jan.kotanski@desy.de>
#
from __future__ import print_function
import unittest
import os
from pninexus import h5cpp
from pninexus.h5cpp.file import AccessFlags
from pninexus import nexus


module_path = os.path.split(os.path.abspath(__file__))[0]


class IsNexusFileTest(unittest.TestCase):

    hdf5_file_path = os.path.join(module_path, "NoNexusFile.h5")
    nexus_file_path = os.path.join(module_path, "NexusFile.nxs")

    def setUp(self):
        unittest.TestCase.setUp(self)

        h5cpp.file.create(self.hdf5_file_path, AccessFlags.TRUNCATE)
        nexus.create_file(self.nexus_file_path, AccessFlags.TRUNCATE)

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def test_NoNeXusFile(self):

        self.assertFalse(nexus.is_nexus_file(self.hdf5_file_path))

    def test_NeXusFile(self):

        self.assertTrue(nexus.is_nexus_file(self.nexus_file_path))


class CreatFileTest(unittest.TestCase):

    filename = os.path.join(module_path, "CreateFileTest.nxs")
    h5filename = os.path.join(module_path, "CreateFileTest.h5")

    def test(self):

        f = nexus.create_file(self.filename, AccessFlags.TRUNCATE)
        root = f.root()

        self.assertTrue(root.attributes.exists("NX_class"))
        self.assertTrue(root.attributes.exists("file_time"))
        self.assertTrue(root.attributes.exists("file_update_time"))
        self.assertTrue(root.attributes.exists("file_name"))

        self.assertEqual(root.attributes["NX_class"].read(), "NXroot")

    def test_h5create(self):

        fapl = h5cpp.property.FileAccessList()
        self.assertEqual(
            fapl.close_degree, h5cpp._property.CloseDegree.DEFAULT)
        fapl.set_close_degree(h5cpp._property.CloseDegree.WEAK)
        self.assertEqual(
            fapl.close_degree, h5cpp._property.CloseDegree.WEAK)
        fapl.set_close_degree(h5cpp._property.CloseDegree.SEMI)
        self.assertEqual(
            fapl.close_degree, h5cpp._property.CloseDegree.SEMI)
        fapl.set_close_degree(h5cpp._property.CloseDegree.DEFAULT)
        self.assertEqual(
            fapl.close_degree, h5cpp._property.CloseDegree.DEFAULT)
        fapl.set_close_degree(h5cpp._property.CloseDegree.STRONG)
        self.assertEqual(
            fapl.close_degree, h5cpp._property.CloseDegree.STRONG)

        f = h5cpp.file.create(self.h5filename, AccessFlags.TRUNCATE, fapl=fapl)
        f.root()
