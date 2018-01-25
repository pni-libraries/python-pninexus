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
# Created on: Sep 19, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
import unittest
import os
import numpy

from pni.io.nx.h5 import open_file
from pni.io.nx.h5 import get_object
from pni.io.nx.h5 import xml_to_nexus
from pni.io.nx.h5 import get_name
from pni.io.nx.h5 import get_unit
from pni.io.nx.h5 import get_class
from pni.io.nx.h5 import create_file
from pni.io.nx import nxpath,make_path

file_struct = \
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="detector" type="NXdetector">
            <field name="data" type="uint16" units="cps"/>
        </group>
    </group>

    <group name="sample" type="NXsample">
        <field name="name" type="string"/>
    </group>
</group>
"""


#implementing test fixture
class get_object_test(unittest.TestCase):
    filename = "get_object_test.nxs"

    def setUp(self):
        self._file = create_file(self.filename,overwrite=True)
        root = self._file.root()
        xml_to_nexus(file_struct,root)

    def tearDown(self):
        self._file.flush()
        self._file.close()
    
    def test_get_detector_data(self): 
        p = make_path("/:NXentry/:NXinstrument/:NXdetector/data")
        root = self._file.root()
        o = get_object(root,p)
        self.assertEqual(get_name(o),"data")
        self.assertEqual(get_unit(o),"cps")

    def test_get_detector_group(self):
        root = self._file.root()
        o = get_object(root,"/:NXentry/:NXinstrument/:NXdetector")
        self.assertEqual(get_class(o),"NXdetector")


