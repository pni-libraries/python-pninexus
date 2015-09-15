from __future__ import print_function
import unittest
import numpy

from pni.core import ShapeMismatchError
from pni.io.nx import nxpath
from pni.io.nx import make_path
from pni.io.nx import is_root_element
from pni.io.nx import is_absolute
from pni.io.nx import is_empty
from pni.io.nx import has_name
from pni.io.nx import has_class


#implementing test fixture
class nxpath_test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_creation(self):
        str_path = "file.nxs://:NXentry/:NXinstrument/data@units"
        p = make_path(str_path)
        self.assertTrue(is_absolute(p))
        self.assertEqual(p.size,4)
        self.assertEqual(p.__str__(),str_path)

    def test_front(self):
        str_path = "file.nxs://:NXentry/:NXinstrument/data@units"
        p = make_path(str_path)
        
        root = p.front
        self.assertEqual(root["base_class"],"NXroot")
        self.assertEqual(root["name"],"/")
        self.assertTrue(has_name(p.front))
        self.assertTrue(has_class(p.front))

    def test_back(self):
        str_path = "file.nxs://:NXentry/:NXinstrument/data@units"
        p = make_path(str_path)
        
        data = p.back
        self.assertEqual(data["base_class"],"")
        self.assertEqual(data["name"],"data")
        self.assertTrue(has_name(data))
        self.assertTrue(not has_class(data))


    def test_filename(self):
        p = make_path("file.nxs://:NXentry/:NXinstrument")
        self.assertEqual(p.filename,"file.nxs")
        p.filename="hello.nxs"
        self.assertEqual(p.filename,"hello.nxs")

    def test_attribute(self):
        p = make_path(":NXentry/:NXinstrument@NX_class")
        self.assertTrue(not is_absolute(p))
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

    def test_append(self):
        p = nxpath()
        self.assertEqual(len(p),0)
        self.assertTrue(is_empty(p))

        p.append(name="/",base_class="NXroot")
        self.assertEqual(len(p),1)
        print(p.front)
        self.assertTrue(is_root_element(p.front))

        p.append(name="",base_class="NXentry")
        self.assertEqual(len(p),2)

        p.append(name="",base_class="NXinstrument")
        self.assertEqual(len(p),3)

        p.append(name="mythen",base_class="NXdetector")
        self.assertEqual(len(p),4)

        self.assertEqual(p.__str__(),"/:NXentry/:NXinstrument/mythen:NXdetector")

    def test_prepend(self):
        p = nxpath()
        self.assertTrue(is_empty(p))
        
        p.prepend(name="mythen",base_class="NXdetector")
        self.assertEqual(len(p),1)
        
        p.prepend(name="",base_class="NXinstrument")
        self.assertEqual(len(p),2)
        
        p.prepend(name="",base_class="NXentry")
        self.assertEqual(len(p),3)

        p.prepend(name="/",base_class="NXroot")
        self.assertEqual(len(p),4)
        
        self.assertEqual(p.__str__(),"/:NXentry/:NXinstrument/mythen:NXdetector")
        self.assertTrue(is_root_element(p.front))

    def test_pop_front(self):
        p = make_path(":NXentry/:NXinstrument/:NXdetector/data")

        self.assertEqual(len(p),4)
        p.pop_front()
        self.assertEqual(p.__str__(),":NXinstrument/:NXdetector/data")
        p.pop_front()
        self.assertEqual(p.__str__(),":NXdetector/data")
        p.pop_front()
        self.assertEqual(p.__str__(),"data")
        p.pop_front()
        self.assertEqual(p.__str__(),"")
        self.assertRaises(IndexError,p.pop_front)

    def test_pop_back(self):
        p = make_path(":NXentry/:NXinstrument/:NXdetector/data")

        self.assertTrue(not is_root_element(p.front))

        self.assertEqual(len(p),4)
        p.pop_back()
        self.assertEqual(p.__str__(),":NXentry/:NXinstrument/:NXdetector")
        p.pop_back()
        self.assertEqual(p.__str__(),":NXentry/:NXinstrument")
        p.pop_back()
        self.assertEqual(p.__str__(),":NXentry")
        p.pop_back()
        self.assertEqual(p.__str__(),"")
        self.assertRaises(IndexError,p.pop_back)


        
