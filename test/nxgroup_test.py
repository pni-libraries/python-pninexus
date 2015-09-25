from __future__ import print_function
import unittest
import numpy

from pni.io.nx.h5 import nxfile
from pni.io.nx.h5 import nxgroup
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.io.nx.h5 import xml_to_nexus
from pni.io.nx.h5 import get_class
from pni.io.nx.h5 import get_path
from pni.io.nx.h5 import get_object
from pni.core import ShapeMismatchError

from .attributes_test import attributes_test

file_struct = \
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="detector" type="NXdetector">
            <field name="data" type="uint16" units="cps">
                <dimension rank="3">
                    <dim index="1" value="0"/>
                    <dim index="2" value="1024"/>
                    <dim index="3" value="2048"/>
                </dimension>
            </field>
        </group>
    </group>

    <group name="sample" type="NXsample">
        <field name="name" type="string">mysample</field>
        <field name="material" type="string">Si</field>
    </group>

    <group name="data" type="NXdata">
    </group>
</group>
"""

def write_attribute(a,v):
    a.value = v
    return a

#implementing test fixture
class nxgroup_test(unittest.TestCase):
    attr_tester = attributes_test()

    def setUp(self):
        self.gf = create_file("test/nxgroup_test.nxs",overwrite=True)
        self.root = self.gf.root()
        xml_to_nexus(file_struct,self.root)

    def tearDown(self):
        self.root.close()
        self.gf.close()

    def test_group_creation(self):
        g = self.root.create_group("metadata")
        self.assertTrue(g.is_valid)
        self.assertEqual(get_class(g),"")
        self.assertEqual(g.name,"metadata")


        g = self.root.create_group("scan_1",nxclass="NXentry")
        self.assertEqual(get_class(g),"NXentry")
        self.assertEqual(g.name,"scan_1")

        g = self.root.create_group("scan_2","NXentry")
        self.assertTrue(g.is_valid)
        self.assertEqual(g.name,"scan_2")
        self.assertEqual(get_class(g),"NXentry")

    def test_open(self):
        #try to open a group that does not exist
        self.assertRaises(KeyError,self.root.__getitem__,"data")

    def test_simple_attributes(self):
        g = self.root.create_group("dgroup")
        self.assertTrue(g.is_valid)
        self.attr_tester.test_scalar_attribute(self,g)

    def test_array_attributes(self):
        g = self.root.create_group("dgroup")
        self.assertTrue(g.is_valid)
        self.attr_tester.test_array_attribute(self,g)

    def test_group_iteration(self):
        g = self.root.create_group("scan_1","NXentry").\
                      create_group("instrument","NXinstrument").\
                      create_group("detector","NXdetector")
        g.create_group("module_1")
        g.create_group("module_2")
        g.create_group("module_3")

        self.assertTrue(g.size==3)

        i = 1 
        for m in g:
            self.assertTrue(m.name == "module_%i" %(i))
            i += 1

    def test_recursive_iteration(self):

        for g in self.root.recursive:
            print(g.path)

    def test_name_property(self):
        g = get_object(self.root,"/:NXentry")
        self.assertEqual(g.name,"entry")

    def test_parent_property(self):
        g = get_object(self.root,"/:NXentry/:NXinstrument/:NXdetector")
        self.assertEqual(g.parent.name,"instrument")
        self.assertEqual(g.parent.parent.parent.name,"/")

    def test_filename_property(self):
        g = get_object(self.root,"/:NXentry/:NXinstrument/:NXdetector")
        self.assertEqual(g.filename,"test/nxgroup_test.nxs")

    def test_size_property(self):
        self.assertEqual(self.root.size,1)
        self.assertEqual(get_object(self.root,"/:NXentry").size,3)
        self.assertEqual(get_object(self.root,"/:NXentry/:NXdata").size,0)

    def test_path_property(self):
        g = get_object(self.root,"/:NXentry/:NXinstrument/:NXdetector")
        self.assertEqual(g.path,"/entry:NXentry/instrument:NXinstrument/detector:NXdetector")
        



        
