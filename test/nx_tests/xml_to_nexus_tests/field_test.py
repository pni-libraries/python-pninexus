import unittest
import os

from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.io.nx.h5 import get_object
from pni.io.nx.h5 import xml_to_nexus


scalar_field=\
"""
<field name="scalar_field" type="float64"/>
"""

mdim_field_default=\
"""
<field name="mdim_field_default" type="uint32">
    <dimensions rank="3">
        <dim index="1" value="1"/>
        <dim index="2" value="1024"/>
        <dim index="3" value="512"/>
    </dimensions>
</field>
"""

mdim_field_custom=\
"""
<field name="mdim_field_custom" type="uint32">
    <dimensions rank="3">
        <dim index="1" value="1"/>
        <dim index="2" value="1024"/>
        <dim index="3" value="512"/>
    </dimensions>
    <chunk rank="3">
        <dim index="1" value="1"/>
        <dim index="2" value="1024"/>
        <dim index="3" value="256"/>
    </chunk>
</field>
"""

mdim_field_compress = \
"""
<field name="mdim_field_compress" type="uint32">
    <dimensions rank="3">
        <dim index="1" value="1"/>
        <dim index="2" value="1024"/>
        <dim index="3" value="512"/>
    </dimensions>
    <strategy compression="true" rate="3" shuffle="true"/>
</field>
"""

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

module_path = os.path.dirname(os.path.abspath(__file__))

#implementing test fixture
class FieldTest(unittest.TestCase):
    """
    Testing group creation
    """
    
    file_name = os.path.join(module_path,"field_test.nxs")
    
    
    @classmethod
    def setUpClass(self):
        f=create_file(self.file_name,overwrite=True)
   
    def setUp(self):
        self.gf = open_file(self.file_name,readonly=False)
        self.root = self.gf.root()

    def tearDown(self):
        self.root.close()
        self.gf.close()


    def test_scalar_field(self):
        xml_to_nexus(scalar_field,self.root)

        f = get_object(self.root,"/scalar_field")
        self.assertTrue(check_field(f,"scalar_field","float64",(1,)))

    def test_mdim_field_default(self):
        xml_to_nexus(mdim_field_default,self.root)
        f = get_object(self.root,"/mdim_field_default")
        self.assertTrue(check_field(f,"mdim_field_default","uint32",
                                    (1,1024,512)))

    def test_mdim_field_compress(self):
        xml_to_nexus(mdim_field_compress,self.root)
        f = get_object(self.root,"/mdim_field_compress")
        self.assertTrue(check_field(f,"mdim_field_compress","uint32",
                                    (1,1024,512)))

    def test_mdim_field_custom(self):
        xml_to_nexus(mdim_field_custom,self.root)
        f = get_object(self.root,"/mdim_field_custom")
        self.assertTrue(check_field(f,"mdim_field_custom","uint32",
                                    (1,1024,512)))





