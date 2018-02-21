import unittest
import os

from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.io.nx.h5 import get_object
from pni.io.nx.h5 import xml_to_nexus


scalar_attribute=\
"""
<attribute name="transformation_type" type="string">rotation</attribute>
"""

mdim_attribute=\
"""
<attribute name="vector" type="float32">
    <dimensions rank="1">
        <dim index="1" value="3"/>
    </dimensions>
    0 0 1
</attribute>
"""


def check_attribute(a,n,t,s):
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

    return a.name == n and \
           a.dtype == t and  \
           len(a.shape) == len(s) and \
           a.shape == s

module_path = os.path.dirname(os.path.abspath(__file__))

#implementing test fixture
class AttributeWriteTest(unittest.TestCase):
    """
    Testing attribute creation
    """

    file_name = os.path.join(module_path,"attribute_test.nxs")
    
    
    @classmethod
    def setUpClass(self):
        f=create_file(self.file_name,overwrite=True)
        r = f.root()
        r.create_field("data","float32")
   
    def setUp(self):
        self.gf = open_file(self.file_name,readonly=False)
        self.root = self.gf.root()
        self.field = self.root["data"]

    def tearDown(self):
        self.root.close()
        self.field.close()
        self.gf.close()


    def test_scalar_attribute(self):
        xml_to_nexus(scalar_attribute,self.field)

        a = get_object(self.root,"/data@transformation_type")
        self.assertTrue(check_attribute(a,"transformation_type","object",(1,)))
        self.assertEqual(a[...],"rotation")

    def test_mdim_attribute(self):
        xml_to_nexus(mdim_attribute,self.field)
        a = get_object(self.root,"/data@vector")
        self.assertTrue(check_attribute(a,"vector","float32", (3,)))
        d = a[...]

        self.assertAlmostEqual(d[0],0.0)
        self.assertAlmostEqual(d[1],0.0)
        self.assertAlmostEqual(d[2],1.0)

