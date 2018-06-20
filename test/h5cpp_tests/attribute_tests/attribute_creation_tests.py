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
from pninexus.h5cpp.datatype import kInt32, kVariableString
from pninexus.h5cpp.datatype import Class
from pninexus.h5cpp.dataspace import Type

module_path = os.path.dirname(os.path.abspath(__file__))


class AttributeCreationTest(unittest.TestCase):

    filename = os.path.join(module_path, 'AttributeCreationTest.h5')

    def setUp(self):

        self.file = h5cpp.file.create(self.filename, AccessFlags.TRUNCATE)
        self.root = self.file.root()

    def tearDown(self):
        self.root.close()
        self.file.close()

    def testScalarIntAttribute(self):

        a = self.root.attributes.create("test", kInt32)
        self.assertEqual(a.dataspace.type, Type.SCALAR)
        self.assertEqual(a.datatype.type, Class.INTEGER)

    def testScalarStringAttribute(self):

        a = self.root.attributes.create("test", kVariableString)
        self.assertEqual(a.dataspace.type, Type.SCALAR)
        self.assertEqual(a.datatype.type, Class.STRING)
        self.assertTrue(a.datatype.is_variable_length)
