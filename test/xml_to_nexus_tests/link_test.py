import unittest
import numpy
import os

import pni.io.nx.h5 as nx
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.io.nx.h5 import get_object
from pni.io import ObjectError
from pni.io.nx.h5 import xml_to_nexus
from pni.io.nx.h5 import get_class
from pni.io.nx.h5 import nxlink

internal_link=\
"""
<group name="entry_1" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="detector" type="NXdetector">
            <field name="data" type="uint32" units="cps">
            </field>
        </group>
    </group>

    <group name="data" type="NXdata">
        <link name="data" target="/entry_1/instrument/detector/data"/>
    </group>
</group>
"""

detector_file=\
"""
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="detector" type="NXdetector">
            <field name="data" type="uint32" units="cps">
            </field>
        </group>
    </group>
</group>
"""

master_entry=\
"""
<group name="entry_2" type="NXentry">
    <group name="data" type="NXdata">
        <link name="data"
        target="detector.nxs://entry/instrument/detector/data"/>
    </group>
</group>
"""


def check_field(f,n,t,s):
    """check name, data type, and shape of a field

    Arg:
        f (nxfield) ..... nexus field
        n (string) ...... expected name of the field
        t (string) ...... expected type
        s (list) ........ expected shape of the field

    Return:
        true if f's name, data type, and shape match the 
        expected values

    """

    return f.name == n and \
           f.dtype == t and  \
           len(f.shape) == len(s) and \
           f.shape == s

#implementing test fixture
class link_test(unittest.TestCase):
    """
    Testing link creation
    """
    file_path = os.path.split(__file__)[0]
    file_name = "link_test.nxs"
    full_path = os.path.join(file_path,file_name)
    det_path = os.path.join(file_path,"detector.nxs")
    
    
    @classmethod
    def setUpClass(self):
        f=create_file(self.full_path,overwrite=True)
        f.close()
        f = create_file(self.det_path,overwrite=True)
        r = f.root()
        xml_to_nexus(detector_file,r)
   
    def setUp(self):
        self.gf = open_file(self.full_path,readonly=False)
        self.root = self.gf.root()

    def tearDown(self):
        self.root.close()
        self.gf.close()


    def test_internal_link(self):
        xml_to_nexus(internal_link,self.root)

        f = get_object(self.root,"/entry_1/:NXdata/data")
        self.assertTrue(check_field(f,"data","uint32",(1,)))

    def test_external_link(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        xml_to_nexus(master_entry,self.root)

        f = get_object(self.root,"/entry_2/:NXdata/data")
        self.assertTrue(check_field(f,"data","uint32",(1,)))




