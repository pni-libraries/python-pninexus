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
        <group name="monochromator" type="NXmonochromator">
        </group>
        <group name="source" type="NXsource"/>
        <group name="undulator" type="NXinsertion_device"/>
    </group>
    <group name="data" type="NXdata">
        <link name="data" target="/entry/instrument/detector/data"/>
        <link name="x" target="/entry/sample/transformations/x"/>
    </group>
</group>
"""

class get_links_test(unittest.TestCase):
    
    test_dir = os.path.split(__file__)[0]
    file_name = "get_links_test.nxs"
    full_path = os.path.join(test_dir,file_name)
    
    def setUp(self):
        self.file = nexus.create_file(self.full_path,True)
        self.root = self.file.root()
        nexus.xml_to_nexus(file_struct,self.root)

        self.instrument = nexus.get_object(self.root,"/:NXentry/:NXinstrument")

    def tearDown(self):
        self.root.close()
        self.instrument.close()
        self.file.close()


    def test_entry(self):
        """
        Test non recursive link list of entry
        """

        l = nexus.get_links(self.instrument)
        self.assertEqual(len(l),4)
        for link in nexus.get_links(self.instrument):
            self.assertEqual(link.status,nexus.nxlink_status.VALID)
            self.assertEqual(link.type,nexus.nxlink_type.HARD)

    def test_nxdata(self):
        data_group = nexus.get_object(self.root,"/:NXentry/:NXdata")
        links = nexus.get_links(data_group)

        self.assertEqual(links[0].status,nexus.nxlink_status.VALID)
        self.assertEqual(links[0].type,nexus.nxlink_type.SOFT)

        self.assertEqual(links[1].status,nexus.nxlink_status.INVALID)
        self.assertEqual(links[1].type,nexus.nxlink_type.SOFT)

        data =  links[0].resolve()
        self.assertTrue(isinstance(data,nexus.nxfield))
        self.assertEqual(data.size,0)
        self.assertEqual(data.dtype,"uint16")
