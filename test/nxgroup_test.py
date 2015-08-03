import unittest
import numpy

from pni.io.nx.h5 import nxfile
from pni.io.nx.h5 import nxgroup
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import open_file
from pni.core import ShapeMismatchError

from attributes_test import attributes_test

def write_attribute(a,v):
    a.value = v
    return a

#implementing test fixture
class nxgroup_test(unittest.TestCase):
    attr_tester = attributes_test()

    def setUp(self):
        self.gf = create_file("nxgroup_test.nxs",overwrite=True)
        self.root = self.gf.root()

    def tearDown(self):
        self.root.close()
        self.gf.close()

    def test_creation(self):
        g = self.root.create_group("metadata")
        self.assertTrue(g.is_valid)
        g = self.root.create_group("scan_1",nxclass="NXentry")
        self.assertTrue(g.attributes["NX_class"].value == "NXentry")

        g = g.create_group("instrument","NXinstrument").\
              create_group("detector","NXdetector")
        self.assertTrue(g.is_valid)
        self.assertTrue(g.name == "detector")

    def test_open(self):
        #try to open a group that does not exist
        self.assertRaises(KeyError,self.root.__getitem__,"data")

    def test_simple_attributes(self):
        g = self.root.create_group("dgroup")
        self.assertTrue(g.is_valid)
        self.attr_tester.test_scalar_attribute(self,g)

    def test_array_attributes(self):
        g = self.root.create_group("dgroup")
        self.assertTrue(g.is_valid)
        self.attr_tester.test_array_attribute(self,g)

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

        
