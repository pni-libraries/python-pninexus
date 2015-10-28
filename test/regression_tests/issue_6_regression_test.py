import pni.io.nx.h5 as nexus
import unittest 
import os

class issue_6_regression_test(unittest.TestCase):
    test_dir = os.path.split(__file__)[0]
    file_name = "issue_6_regression_test.nxs"
    full_path = os.path.join(test_dir,file_name)
    data_value = 9223372036854775807

    def setUp(self):
        self.gf = nexus.create_file(self.full_path,True)
        self.root = self.gf.root()

    def tearDown(self):
        self.root.close()
        self.gf.close()

    def test_field_setitem(self):
        target = self.root.create_field("target","uint64")
        target[...] = self.data_value

    def test_attribute_setitem(self):
        target = self.root.attributes.create("target","uint64")
        target[...] = self.data_value

    def test_field_write(self):
        target = self.root.create_field("target","uint64")
        target.write(self.data_value)

    def test_attribute_write(self):
        target = self.root.attributes.create("target","uint64")
        target.write(self.data_value)


