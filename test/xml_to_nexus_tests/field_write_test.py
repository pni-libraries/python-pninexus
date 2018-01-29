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

scalar_field=\
"""
<field name="scalar_field" type="float64">
    100.
</field>
"""

mdim_field_default=\
"""
<field name="mdim_field_default" type="uint32">
    <dimensions rank="2">
        <dim index="1" value="2"/>
        <dim index="2" value="3"/>
    </dimensions>
    1 2 3 
    4 5 6
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

#implementing test fixture
class field_write_test(unittest.TestCase):
    """
    Testing group creation
    """
    file_path = os.path.split(__file__)[0]
    file_name = "field_write_test.nxs"
    full_path = os.path.join(file_path,file_name)
    
    
    @classmethod
    def setUpClass(self):
        f=create_file(self.full_path,overwrite=True)
   
    def setUp(self):
        self.gf = open_file(self.full_path,readonly=False)
        self.root = self.gf.root()

    def tearDown(self):
        self.root.close()
        self.gf.close()


    def test_scalar_field(self):
        xml_to_nexus(scalar_field,self.root,lambda obj: True)

        f = get_object(self.root,"/scalar_field")
        self.assertTrue(check_field(f,"scalar_field","float64",(1,)))
        self.assertAlmostEqual(f[...],100.)

    def test_mdim_field_default(self):
        xml_to_nexus(mdim_field_default,self.root,lambda obj: True)
        f = get_object(self.root,"/mdim_field_default")
        self.assertTrue(check_field(f,"mdim_field_default","uint32",
                                    (2,3)))

        d = f[...]
        for (ref_value,data) in zip(range(1,7),d.flat):
            self.assertEqual(ref_value,data)





