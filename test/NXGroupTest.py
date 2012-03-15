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
        pass
