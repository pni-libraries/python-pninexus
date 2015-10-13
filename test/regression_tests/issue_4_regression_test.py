import pni.io.nx.h5 as nexus
import unittest 
import os

class issue_4_regression_test(unittest.TestCase):
    test_dir = os.path.split(__file__)[0]
    file_name = "issue_4_regression_test.nxs"
    full_path = os.path.join(test_dir,file_name)

    def setUp(self):
        self.gf = nexus.create_file(self.full_path,True)
        self.root = self.gf.root()

    def tearDown(self):
        self.root.close()
        self.gf.close()

    def test(self):

        self.assertRaises(TypeError,self.root.create_field,"data",dtype="int32")



