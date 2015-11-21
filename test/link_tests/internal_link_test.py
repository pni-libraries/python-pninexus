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
# Created on: Oct 13, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import os

import pni.io.nx.h5 as nexus

file_struct=\
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="detector" type="NXdetector">
            <field name="data" type="uint16">
                <dimensions rank="2">
                    <dim value="0" index="1"/>
                    <dim value="1024" index="2"/>
                </dimensions>
            </field>
        </group>
    </group>
    <group name="data" type="NXdata">
    </group>
</group>
"""

class internal_link_test(unittest.TestCase):
    
    test_dir = os.path.split(__file__)[0]
    file_name = "internal_link_test.nxs"
    full_path = os.path.join(test_dir,file_name)
    
    def setUp(self):
        self.file = nexus.create_file(self.full_path,True)
        self.root = self.file.root()
        nexus.xml_to_nexus(file_struct,self.root)

        self.nxdata = nexus.get_object(self.root,"/:NXentry/:NXdata")

    def tearDown(self):
        self.root.close()
        self.nxdata.close()
        self.file.close()


    def test_link_from_path(self):
        """
        Test link creation from a path. The target is only specified by its
        path. 
        """
        nexus.link("/entry/instrument/detector/data",self.nxdata,"plot_data")
        d = self.nxdata["plot_data"]
        self.assertEqual(d.name,"plot_data")
        self.assertEqual(d.size,0)
        self.assertEqual(d.dtype,"uint16")
        self.assertEqual(d.path,"/entry:NXentry/data:NXdata/plot_data")

    def test_link_from_object(self):
        """
        Test link creation from an object. The target is specified by an
        object.
        """

        det_data = nexus.get_object(self.root,"/:NXentry/:NXinstrument/:NXdetector/data")
        nexus.link(det_data,self.nxdata,"plot_data")
        d = self.nxdata["plot_data"] 
        self.assertEqual(d.name,"plot_data")
        self.assertEqual(d.size,0)
        self.assertEqual(d.dtype,"uint16")
        self.assertEqual(d.path,"/entry:NXentry/data:NXdata/plot_data")

        
