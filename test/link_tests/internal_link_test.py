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

    def test_unresolvable_link(self):
        data = nexus.get_object(self.root,"/:NXentry/:NXdata")
        nexus.link("/entry/instrument/monochromator/energy",data,"data")
        l = nexus.get_object(self.root,"/:NXentry/:NXdata/data")
        self.assertTrue(isinstance(l,nexus.nxlink))
        self.assertEqual(l.status,nexus.nxlink_status.INVALID)
        self.assertEqual(l.type,nexus.nxlink_type.SOFT)
        self.assertEqual(l.path,"/entry:NXentry/data:NXdata/data")
