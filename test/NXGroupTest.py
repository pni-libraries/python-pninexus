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
        #try to open a group that does not exist
        self.assertRaises(NXGroupError,self.gf.open,"/data")

    def test_simple_attributes(self):
        g = self.gf.create_group("dgroup")
        self.assertTrue(g.valid)
        self.attr_tester.test_scalar_attribute(self,g)

    def test_array_attributes(self):
        g = self.gf.create_group("dgroup")
        self.assertTrue(g.valid)
        self.attr_tester.test_array_attribute(self,g)

    def test_group_iteration(self):
        pass

