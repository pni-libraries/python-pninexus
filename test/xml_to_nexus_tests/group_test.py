import unittest
import numpy
import os

import pni.io.nx.h5 as nx
from pni.io.nx.h5 import create_file
from pni.io.nx.h5 import get_object
from pni.io import ObjectError
from pni.io.nx.h5 import xml_to_nexus
from pni.io.nx.h5 import get_class

single_group=\
"""
<group name="entry" type="NXentry"/>
"""

multi_group=\
"""
<group name="scan_1" type="NXentry"/>
<group name="scan_2" type="NXentry"/>
<group name="scan_3" type="NXentry"/>
"""

nested_group=\
"""
<group name="scan" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="detector" type="NXdetector"/>
        <group name="source" type="NXsource"/>
    </group>

    <group name="sample" type="NXsample"/>
    <group name="data" type="NXdata"/>
    <group name="control" type="NXmonitor"/>
</group>
"""

#implementing test fixture
class group_test(unittest.TestCase):
    """
    Testing group creation
    """
    file_path = os.path.split(__file__)[0]
    file_name = "group_test.nxs"
    full_path = os.path.join(file_path,file_name)

    def _check_name_type(self,g,n,t):
        """check name and class of a group

        Checks the name and the class a group belongs to and returns true if it
        satisfies the user provided values.

        Arg:
            g (nxgroup) ..... nexus group
            n (string) ...... expected name of the group
            t (string) ...... expected class of the group

        Return:
            true if g has name n and class t

        """

        return g.name == n and get_class(g)==t
   
    def setUp(self):
        self.gf = create_file(self.full_path,overwrite=True)
        self.root = self.gf.root()

    def tearDown(self):
        self.root.close()
        self.gf.close()


    def test_single_group(self):
        xml_to_nexus(single_group,self.root)

        g = get_object(self.root,"/:NXentry")
        self.assertTrue(self._check_name_type(g,"entry","NXentry"))

    def test_multi_group(self):
        xml_to_nexus(multi_group,self.root)

        self.assertEqual(len(self.root),3)
        self.assertTrue(self._check_name_type(self.root["scan_1"],"scan_1",
                                                                  "NXentry"))
        self.assertTrue(self._check_name_type(self.root["scan_2"],"scan_2",
                                                                  "NXentry"))
        self.assertTrue(self._check_name_type(self.root["scan_3"],"scan_3",
                                                                  "NXentry"))

    def test_nested_group(self):
        xml_to_nexus(nested_group,self.root)

        self.assertTrue(self._check_name_type(get_object(self.root,":NXentry"),
                                              "scan","NXentry"))
        self.assertTrue(self._check_name_type(
             get_object(self.root,":NXentry/:NXinstrument"),
             "instrument","NXinstrument"))
        self.assertTrue(self._check_name_type(
             get_object(self.root,":NXentry/:NXsample"),"sample","NXsample"))
        self.assertTrue(self._check_name_type(
             get_object(self.root,":NXentry/:NXmonitor"),
             "control","NXmonitor"))

        self.assertTrue(self._check_name_type(
             get_object(self.root,":NXentry/:NXinstrument/:NXdetector"),
             "detector","NXdetector"))
        self.assertTrue(self._check_name_type(
             get_object(self.root,":NXentry/:NXinstrument/:NXsource"),
             "source","NXsource"))





