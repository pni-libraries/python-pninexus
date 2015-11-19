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

scalar_attribute=\
"""
<attribute name="transformation_type" type="string"/>
"""

mdim_attribute=\
"""
<attribute name="vector" type="float32">
    <dimensions rank="1">
        <dim index="1" value="3"/>
    </dimensions>
</attribute>
"""


def check_attribute(a,n,t,s):
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

    return a.name == n and \
           a.dtype == t and  \
           len(a.shape) == len(s) and \
           a.shape == s

#implementing test fixture
class attribute_test(unittest.TestCase):
    """
    Testing attribute creation
    """
    file_path = os.path.split(__file__)[0]
    file_name = "attribute_test.nxs"
    full_path = os.path.join(file_path,file_name)
    
    
    @classmethod
    def setUpClass(self):
        f=create_file(self.full_path,overwrite=True)
        r = f.root()
        r.create_field("data","float32")
   
    def setUp(self):
        self.gf = open_file(self.full_path,readonly=False)
        self.root = self.gf.root()
        self.field = self.root["data"]

    def tearDown(self):
        self.root.close()
        self.field.close()
        self.gf.close()


    def test_scalar_attribute(self):
        xml_to_nexus(scalar_attribute,self.field)

        a = get_object(self.root,"/data@transformation_type")
        self.assertTrue(check_attribute(a,"transformation_type","string",(1,)))

    def test_mdim_attribute(self):
        xml_to_nexus(mdim_attribute,self.field)
        a = get_object(self.root,"/data@vector")
        self.assertTrue(check_attribute(a,"vector","float32", (3,)))

