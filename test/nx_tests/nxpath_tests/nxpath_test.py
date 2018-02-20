#
# (c) Copyright 2015 DESY, 
#               2015 Eugen Wintersberger <eugen.wintersberger@desy.de>
#
# This file is part of python-pni.
#
# python-pni is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# python-pni is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with python-pni.  If not, see <http://www.gnu.org/licenses/>.
# ===========================================================================
#
# Created on: Oct 13, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import numpy

from pni.io import nexus

from pni.io.nx import nxpath
from pni.io.nx import make_path
from pni.io.nx import is_root_element
from pni.io.nx import is_absolute
from pni.io.nx import is_empty
from pni.io.nx import has_name
from pni.io.nx import has_class
from pni.io.nx import match


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
        
    def test_from_nexus_path(self):
        
        nexus_path = nexus.Path.from_string("/scan:NXentry")
        self.assertEqual(str(nexus_path),"/scan:NXentry")
        nxpath_instance = nxpath(base_instance = nexus_path)
        self.assertEqual(str(nxpath_instance),"/scan:NXentry")

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

    def test_push_back(self):
        p = nxpath()
        self.assertEqual(len(p),0)
        self.assertTrue(is_empty(p))

        p.push_back(name="/",base_class="NXroot")
        self.assertEqual(len(p),1)
        print(p.front)
        self.assertTrue(is_root_element(p.front))

        p.push_back(":NXentry")
        self.assertEqual(len(p),2)

        p.push_back("","NXinstrument")
        self.assertEqual(len(p),3)

        p.push_back("mythen","NXdetector")
        self.assertEqual(len(p),4)
        
        p.push_back("data")
        self.assertEqual(len(p),5)

        self.assertEqual(p.__str__(),"/:NXentry/:NXinstrument/mythen:NXdetector/data")

    def test_push_front(self):
        p = nxpath()
        self.assertTrue(is_empty(p))
       
        p.push_front("data")
        self.assertEqual(len(p),1)
        self.assertEqual(p.__str__(),"data")

        p.push_front("mythen:NXdetector")
        self.assertEqual(len(p),2)
        self.assertEqual(p.__str__(),"mythen:NXdetector/data")
        
        p.push_front(name="",base_class="NXinstrument")
        self.assertEqual(len(p),3)
        self.assertEqual(p.__str__(),":NXinstrument/mythen:NXdetector/data")
        
        p.push_front(":NXentry")
        self.assertEqual(len(p),4)
        self.assertEqual(p.__str__(),":NXentry/:NXinstrument/mythen:NXdetector/data")

        p.push_front("/:NXroot")
        self.assertEqual(len(p),5)
        
        self.assertEqual(p.__str__(),"/:NXentry/:NXinstrument/mythen:NXdetector/data")
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

    def test_add_path(self):
        a = make_path("/:NXentry/:NXinstrument")
        b = make_path(":NXdetector/data")
        c = a+b
        self.assertEqual(c.__str__(),"/:NXentry/:NXinstrument/:NXdetector/data")
        self.assertEqual(a.__str__(),"/:NXentry/:NXinstrument")
        self.assertEqual(b.__str__(),":NXdetector/data")
        
        b.push_front("/:NXroot")
        self.assertRaises(ValueError,a.__add__,b)
        b.pop_front()
        b.filename="test.nxs"
        self.assertRaises(ValueError,a.__add__,b)

        self.assertRaises(TypeError,a.__add__,1)

    def test_add_string(self):
        
        a = make_path(":NXentry")
        c = a+":NXinstrument/"
        self.assertEqual(c.__str__(),":NXentry/:NXinstrument")
        c = "/"+a
        self.assertEqual(c.__str__(),"/:NXentry")
        self.assertTrue(is_root_element(c.front))

        a += ":NXinstrument/data"
        self.assertEqual(a.__str__(),":NXentry/:NXinstrument/data")

    def test_match(self):
        self.assertTrue(match(make_path("/:NXentry/:NXinstrument/:NXdetector"),
                              make_path("/scan_1:NXentry/p08:NXinstrument/mythen:NXdetector")))

        self.assertTrue(match("/:NXentry/:NXinstrument/:NXdetector",
                              "/scan_1:NXentry/p08:NXinstrument/mythen:NXdetector"))
        
        
