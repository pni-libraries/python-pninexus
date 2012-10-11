import unittest
import numpy

from pni.nx.h5 import NXFile
from pni.nx.h5 import NXGroup
from pni.nx.h5 import create_file
from pni.nx.h5 import open_file
from pni.nx.h5 import NXGroupError
from pni.nx.h5 import NXAttributeError
from pni.nx.h5 import ShapeMissmatchError

from AttributesTest import AttributeTest

def write_attribute(a,v):
    a.value = v
    return a

#implementing test fixture
class NXGroupTest(unittest.TestCase):
    attr_tester = AttributeTest()

    def setUp(self):
        self.gf = create_file("NXGroupTest.h5",overwrite=True)

    def tearDown(self):
        self.gf.close()

    def test_creation(self):
        print "NXGroupTest.test_creation() ........................"
        g = self.gf.create_group("metadata")
        self.assertTrue(g.valid)
        g = self.gf.create_group("scan_1",nxclass="NXentry")
        self.assertTrue(g.attr("NX_class").value == "NXentry")

        g = g.create_group("instrument/detector")
        self.assertTrue(g.valid)
        self.assertTrue(g.path == "/scan_1/instrument/detector")
        self.assertTrue(g.name == "detector")
        self.assertTrue(g.base == "/scan_1/instrument")

    def test_open(self):
        print "NXGroupTest.test_open() ............................"
        #try to open a group that does not exist
        self.assertRaises(NXGroupError,self.gf.open,"/data")

    def test_simple_attributes(self):
        print "NXGroupTest.test_simple_attributes() ..............."
        g = self.gf.create_group("dgroup")
        self.assertTrue(g.valid)
        self.attr_tester.test_scalar_attribute(self,g)

    def test_array_attributes(self):
        print "NXGroupTest.test_array_attributes() ................."
        g = self.gf.create_group("dgroup")
        self.assertTrue(g.valid)
        self.attr_tester.test_array_attribute(self,g)

    def test_group_iteration(self):
        print "NXGroupTest.test_group_iteration() ................."
        g = self.gf.create_group("/scan_1/instrument/detector");
        g.create_group("module_1")
        g.create_group("module_2")
        g.create_group("module_3")

        self.assertTrue(g.nchilds==3)

        i = 1 
        for m in g.childs:
            self.assertTrue(m.name == "module_%i" %(i))
            i += 1

        
