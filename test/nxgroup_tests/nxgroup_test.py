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

    def setUp(self):
        self.gf = create_file("test/nxgroup_test.nxs",overwrite=True)
        self.root = self.gf.root()
        xml_to_nexus(file_struct,self.root)

    def tearDown(self):
        self.root.close()
        self.gf.close()

    def test_open(self):
        #try to open a group that does not exist
        self.assertRaises(KeyError,self.root.__getitem__,"data")

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

        



        
