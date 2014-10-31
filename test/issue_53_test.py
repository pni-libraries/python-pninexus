#regresssion test for issue 53

import unittest

import pni.io.nx.h5 as nx
from pni.core import size_mismatch_error

class Issue_53_Test(unittest.TestCase):
    def setUp(self):
        self.nxfile = nx.create_file("Issue_53_Test.nx",overwrite=True)

    def tearDown(self):
        self.nxfile.close()

    def test_issue(self):
        deflate = nx.deflate_filter()
        deflate.rate = 5
        deflate.shuffle = True
        root = self.nxfile.root()

        self.assertRaises(size_mismatch_error,
                          root.create_field,"test","string",[],[],deflate)


