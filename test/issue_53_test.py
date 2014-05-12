#regresssion test for issue 53

import unittest

import pni.io.nx.h5 as nx

class Issue_53_Test(unittest.TestCase):
    def setUp(self):
        self.nxfile = nx.create_file("Issue_53_Test.nx",overwrite=True)

    def tearDown(self):
        self.nxfile.close()

    def test_issue(self):
        deflate = nx.NXDeflateFilter()
        deflate.rate = 5
        deflate.shuffle = True
        root = self.nxfile["/"]

        self.assertRaises(nx.SizeMismatchError,
                          root.create_field,"test","string",[],[],deflate)


