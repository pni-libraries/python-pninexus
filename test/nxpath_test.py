from __future__ import print_function
import unittest
import numpy

from pni.core import ShapeMismatchError
from pni.io.nx import nxpath
from pni.io.nx import make_path



#implementing test fixture
class nxpath_test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_creation(self):
        str_path = "file.nxs://:NXentry/:NXinstrument/data@units"
        p = make_path(str_path)
        self.assertEqual(p.size,4)
        self.assertEqual(p.__str__(),str_path)

    def test_front(self):
        str_path = "file.nxs://:NXentry/:NXinstrument/data@units"
        p = make_path(str_path)
        
        root = p.front
        self.assertEqual(root["base_class"],"NXroot")
        self.assertEqual(root["name"],"/")

    def test_back(self):
        str_path = "file.nxs://:NXentry/:NXinstrument/data@units"
        p = make_path(str_path)
        
        data = p.back
        self.assertEqual(data["base_class"],"")
        self.assertEqual(data["name"],"data")


    def test_filename(self):
        p = make_path("file.nxs://:NXentry/:NXinstrument")
        self.assertEqual(p.filename,"file.nxs")
        p.filename="hello.nxs"
        self.assertEqual(p.filename,"hello.nxs")

    def test_attribute(self):
        p = make_path(":NXentry/:NXinstrument@NX_class")
        self.assertEqual(p.attribute,"NX_class")
        p.attribute="date"
        self.assertEqual(p.attribute,"date")

    def test_iteration(self):
        s = ":NXentry/:NXinstrument/:NXdetector/data"
        p_dicts = [{"base_class":"NXentry","name":""},
                   {"base_class":"NXinstrument","name":""},
                   {"base_class":"NXdetector","name":""},
                   {"base_class":"","name":"data"}]
        p = make_path(s)

        self.assertEqual(len(p),4)

        for (ref,e) in zip(p_dicts,p):
            self.assertEqual(ref["name"],e["name"])
            self.assertEqual(ref["base_class"],e["base_class"])
        


        
