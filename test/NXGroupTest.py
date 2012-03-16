import unittest

from pni.nx.h5 import NXFile
from pni.nx.h5 import NXGroup
from pni.nx.h5 import create_file
from pni.nx.h5 import open_file


#implementing test fixture
class NXGroupTest(unittest.TestCase):
    def setUp(self):
        self.gf = create_file("NXGroupTest.h5",overwrite=True)

    def tearDown(self):
        self.gf.close()

    def test_creation(self):
        g = self.gf.create_group("metadata")
        g = self.gf.create_group("scan_1",nxclass="NXentry")
        self.assertTrue(g.attr("NX_class").value == "NXentry")

        g = g.create_group("instrument/detector")
        self.assertTrue(g.path == "/scan_1/instrument/detector")
        self.assertTrue(g.name == "detector")
        self.assertTrue(g.base == "/scan_1/instrument")

