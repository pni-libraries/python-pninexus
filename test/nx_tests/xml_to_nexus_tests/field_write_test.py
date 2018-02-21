import unittest
import os

from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.io.nx.h5 import get_object
from pni.io.nx.h5 import xml_to_nexus

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

module_path = os.path.dirname(os.path.abspath(__file__))

#implementing test fixture
class FieldWriteTest(unittest.TestCase):
    """
    Testing group creation
    """
    file_name = os.path.join(module_path,"field_write_test.nxs")
    
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
        self.assertAlmostEqual(f[...],100.)

    def test_mdim_field_default(self):
        xml_to_nexus(mdim_field_default,self.root)
        f = get_object(self.root,"/mdim_field_default")
        self.assertTrue(check_field(f,"mdim_field_default","uint32",
                                    (2,3)))

        d = f[...]
        for (ref_value,data) in zip(range(1,7),d.flat):
            self.assertEqual(ref_value,data)





