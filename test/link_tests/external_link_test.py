from __future__ import print_function
import unittest
import os

import pni.io.nx.h5 as nexus

detector_file_struct=\
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

master_file_struct=\
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="detector" type="NXdetector"/> 
    </group>
    <group name="data" type="NXdata">
        <link name="plot_data" target="../instrument/detector/data"/>
    </group>
</group>
"""

class external_link_test(unittest.TestCase):
    
    test_dir = os.path.split(__file__)[0]
    det_file_name = "external_link_test_detector.nxs"
    master_file_name = "external_link_test_master.nxs"
    det_full_path = os.path.join(test_dir,det_file_name)
    master_full_path = os.path.join(test_dir,master_file_name)
    
    def setUp(self):
        self.file = nexus.create_file(self.det_full_path,True)
        self.root = self.file.root()
        nexus.xml_to_nexus(detector_file_struct,self.root)
        self.root.close()
        self.file.close()

        self.file = nexus.create_file(self.master_full_path,True)
        self.root = self.file.root()
        nexus.xml_to_nexus(master_file_struct,self.root)


    def tearDown(self):
        self.root.close()
        self.file.close()


    def test_link_from_path(self):
        """
        Test link creation from a path. The target is only specified by its
        path. 
        """
        detector_group = nexus.get_object(self.root,
                          "/:NXentry/:NXinstrument/:NXdetector")

        nexus.link("external_link_test_detector.nxs://entry/instrument/detector/data",
                   detector_group,"data")

        d = detector_group["data"]
        self.assertEqual(d.name,"data")
        self.assertEqual(d.dtype,"uint16")
        self.assertEqual(d.path,"/entry:NXentry/instrument:NXinstrument/detector:NXdetector/data")

        data = nexus.get_object(self.root,"/:NXentry/:NXdata")
        d = data["plot_data"]

        


