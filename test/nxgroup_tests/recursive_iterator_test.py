from __future__ import print_function
import unittest
import os

from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import get_object
from pni.io import ObjectError
from pni.io.nx import make_relative 

class recursive_iterator_test(unittest.TestCase):
    file_path = os.path.split(__file__)[0]
    file_name = "simple_iterator_test.nxs"
    full_path = os.path.join(file_path,file_name)

    def setUp(self):
        self.nexus_file = create_file(self.full_path,overwrite=True)
        self.root = self.nexus_file.root()
        self.paths = []
        e = self.root.create_group("entry:NXentry")
        self.paths.append(e.path)
        i = e.create_group("instrument:NXinstrument")
        self.paths.append(i.path)
        self.paths.append(i.create_group("detector:NXdetector").path)
        self.paths.append(i.create_group("monochromator:NXmonochromator").path)
        self.paths.append(i.create_group("source:NXsource").path)
        self.paths.append(e.create_group("data:NXdata").path)
        self.paths.append(e.create_group("sample:NXsample").path)
        self.paths.append(e.create_group("control:NXmonitor").path)
        self.paths.append(e.create_field("title","string").path)
        self.paths.append(e.create_field("experiment_identifier","string").path)
        self.paths.append(e.create_field("experiment_description","string").path)

        #self.paths.sort()

    def tearDown(self):
        self.root.close()
        self.nexus_file.close()

    def test_recursive_iteration(self):
       
        for c in self.root.recursive:
            self.paths.remove(c.path)

        self.assertEqual(len(self.paths),0)
