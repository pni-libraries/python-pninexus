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

detector_file_struct=\
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="mca" type="NXdetector">
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

master_file_struct_1=\
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument"/>
    <group name="data" type="NXdata">
        <link name="plot_data" target="../instrument/detector/data"/>
    </group>
</group>
"""

master_file_struct_2=\
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="detector" type="NXdetector"/>
    </group>
    <group name="data" type="NXdata">
        <link name="plot_data" target="../instrument/detector/mca_data"/>
    </group>
</group>
"""

class external_link_test(unittest.TestCase):
    
    test_dir = os.path.split(__file__)[0]
    det_file_name = "external_link_test_detector.nxs"
    master1_file_name = "external_link_test_master1.nxs"
    master2_file_name = "external_link_test_master2.nxs"
    det_full_path = os.path.join(test_dir,det_file_name)
    master1_full_path = os.path.join(test_dir,master1_file_name)
    master2_full_path = os.path.join(test_dir,master2_file_name)
    
    def setUp(self):
        f = nexus.create_file(self.det_full_path,True)
        r = f.root()
        nexus.xml_to_nexus(detector_file_struct,r)
        r.close()
        f.close()

        f = nexus.create_file(self.master1_full_path,True)
        r = f.root()
        nexus.xml_to_nexus(master_file_struct_1,r)
        r.close()
        f.close()
        
        f = nexus.create_file(self.master2_full_path,True)
        r = f.root()
        nexus.xml_to_nexus(master_file_struct_2,r)
        r.close()
        f.close()


    def tearDown(self):
        pass


    def test_link_group_from_path(self):
        """
        Test link creation from a path. The target is only specified by its
        path. 
        """
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        f = nexus.open_file(self.master1_full_path,readonly=False)
        r = f.root()
        instrument_group = nexus.get_object(r,"/:NXentry/:NXinstrument")

        nexus.link("external_link_test_detector.nxs://entry/instrument/mca",
                   instrument_group,"detector")

        d = instrument_group["detector"]
        self.assertEqual(d.name,"mca")
        self.assertEqual(d.path,"/entry:NXentry/instrument:NXinstrument/mca:NXdetector")

        data = nexus.get_object(r,"/:NXentry/:NXdata")
        d = data["plot_data"]

    def test_link_field_from_path(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        f = nexus.open_file(self.master2_full_path,readonly=False)
        r = f.root()
        detector_group = nexus.get_object(r,"/:NXentry/:NXinstrument/:NXdetector")

        nexus.link("external_link_test_detector.nxs://entry/instrument/mca/data",
                   detector_group,"mca_data")

        d = detector_group["mca_data"]
        self.assertEqual(d.name,"data")
        self.assertEqual(d.path,"/entry:NXentry/instrument:NXinstrument/mca:NXdetector/data")

        data = nexus.get_object(r,"/:NXentry/:NXdata")
        d = data["plot_data"]


        


