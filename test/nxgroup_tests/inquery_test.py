from __future__ import print_function
import unittest
import os

from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import get_object
from pni.io import ObjectError
from pni.io.nx import make_relative 

class inquery_test(unittest.TestCase):
    file_path = os.path.split(__file__)[0]
    file_name = "inquery_test.nxs"
    full_path = os.path.join(file_path,file_name)

    def setUp(self):
        self.nexus_file = create_file(self.full_path,overwrite=True)
        self.root = self.nexus_file.root()
        e = self.root.create_group("entry:NXentry")
        i = e.create_group("instrument:NXinstrument")
        e.create_group("data:NXdata")
        e.create_group("sample:NXsample")
        e.create_group("control:NXmonitor")

    def tearDown(self):
        self.root.close()
        self.nexus_file.close()


    def test_name_property(self):
        g = get_object(self.root,"/:NXentry")
        self.assertEqual(get_object(self.root,"/:NXentry").name,"entry")
        self.assertEqual(get_object(self.root,"/").name,"/")
        self.assertEqual(get_object(self.root,"/:NXentry/:NXinstrument").name,
                         "instrument")

    def test_parent_property(self):
        g = get_object(self.root,"/:NXentry/:NXinstrument")
        self.assertEqual(g.parent.name,"entry")
        self.assertEqual(g.parent.parent.name,"/")

    def test_filename_property(self):
        g = get_object(self.root,"/:NXentry/:NXinstrument")
        self.assertEqual(g.filename,self.full_path)

    def test_size_property(self):
        self.assertEqual(self.root.size,1)
        self.assertEqual(get_object(self.root,"/:NXentry").size,4)
        self.assertEqual(get_object(self.root,"/:NXentry/:NXdata").size,0)

    def test_path_property(self):
        g = get_object(self.root,"/:NXentry/:NXinstrument/")
        self.assertEqual(g.path,"/entry:NXentry/instrument:NXinstrument")



