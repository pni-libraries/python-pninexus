from __future__ import print_function
import unittest
import os

from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import get_object
from pni.io import ObjectError
from pni.io.nx import make_relative 

class child_access_test(unittest.TestCase):
    file_path = os.path.split(__file__)[0]
    file_name = "child_access_test.nxs"
    full_path = os.path.join(file_path,file_name)

    def setUp(self):
        self.nexus_file = create_file(self.full_path,overwrite=True)
        self.root = self.nexus_file.root()
        e = self.root.create_group("entry:NXentry")
        i = e.create_group("instrument:NXinstrument")
        e.create_group("data:NXdata")
        e.create_group("sample:NXsample")
        e.create_group("control:NXmonitor")
        e.create_field("title","string")
        e.create_field("experiment_identifier","string")
        e.create_field("experiment_description","string")

        self.names = ["instrument:NXinstrument",
                      "data:NXdata",
                      "sample:NXsample",
                      "control:NXmonitor",
                      "title",
                      "experiment_identifier",
                      "experiment_description"]
        self.names.sort()

    def tearDown(self):
        self.root.close()
        self.nexus_file.close()

    def test_loop_by_index(self):
        e = get_object(self.root,"entry:NXentry")
    
        self.assertEqual(len(e),7)
        for index in range(len(e)):
            child = e[index]
            self.assertEqual(make_relative(e.path,child.path),
                             self.names[index])

    def test_loop_by_name(self):

        e = get_object(self.root,"entry:NXentry")
        
        for (name,nref) in zip(e.names(),self.names):
            child = e[name]
            self.assertEqual(make_relative(e.path,child.path),nref)




