import unittest
import os

from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.io.nx.h5 import get_object
from pni.io.nx.h5 import xml_to_nexus
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

module_path = os.path.dirname(os.path.abspath(__file__))

def check_field(f,n,t,s):
    """check name, data type, and shape of a field

    Arg:
        f (nxfield) ..... h5cpp_tests field
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
class LinkTest(unittest.TestCase):
    """
    Testing link creation
    """
    
    file_name = os.path.join(module_path,"link_test.nxs")
    det_path  = os.path.join(module_path,"detector.nxs")
    
    
    @classmethod
    def setUpClass(self):
        f=create_file(self.file_name,overwrite=True)
        f.close()
        f = create_file(self.det_path,overwrite=True)
        r = f.root()
        xml_to_nexus(detector_file,r)
   
    def setUp(self):
        self.gf = open_file(self.file_name,readonly=False)
        self.root = self.gf.root()

    def tearDown(self):
        self.root.close()
        self.gf.close()


    def test_internal_link(self):
        xml_to_nexus(internal_link,self.root)

        f = get_object(self.root,"/entry_1:NXentry/:NXdata/data")
        self.assertTrue(check_field(f,"data","uint32",(1,)))

    def test_external_link(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        xml_to_nexus(master_entry,self.root)

        f = get_object(self.root,"/entry_2:NXentry/:NXdata/data")
        self.assertTrue(check_field(f,"data","uint32",(1,)))




