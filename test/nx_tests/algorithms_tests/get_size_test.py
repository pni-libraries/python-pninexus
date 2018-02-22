#
# (c) Copyright 2015 DESY, 
#               2015 Eugen Wintersberger <eugen.wintersberger@desy.de>
#
# This file is part of python-pni.
#
# python-pni is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# python-pni is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with python-pni.  If not, see <http://www.gnu.org/licenses/>.
# ===========================================================================
#
# Created on: Sep 18, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
import unittest
import os
import numpy

from pni.io import h5cpp
from pni.io.nx.h5 import nxfile
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.io.nx.h5 import nxgroup
from pni.io.nx.h5 import get_size

module_path = os.path.dirname(os.path.abspath(__file__))

#implementing test fixture
class GetSizeTest(unittest.TestCase):
    
    filename = os.path.join(module_path,"get_size_test.nxs")

    def setUp(self):
        self._file = create_file(self.filename,overwrite=True)

    def tearDown(self):
        self._file.flush()
        self._file.close()


    def test_with_attribute(self):
        #this should work as the file does not exist yet
        root = self._file.root()

        a = root.attributes.create("test_scalar",
                                   h5cpp.datatype.kFactory.create(numpy.dtype("float32")))
        self.assertEqual(get_size(a),1)

        a = root.attributes.create("test_multidim",
                                   h5cpp.datatype.kFactory.create(numpy.dtype("ui64")),
                                   shape=(3,4))
        self.assertEqual(get_size(a),12)

    def test_with_field(self):
        root = self._file.root()

        f = root.create_field("test_scalar","float64")
        self.assertEqual(get_size(f),1)

        f = root.create_field("test_multidim","int16",shape=(1024,2048))
        self.assertEqual(get_size(f),1024*2048)

    def test_with_group(self):
        root = self._file.root()
        root.create_group("entry_1","NXentry")
        root.create_group("entry_2","NXentry")

        self.assertEqual(get_size(root),2)

            










