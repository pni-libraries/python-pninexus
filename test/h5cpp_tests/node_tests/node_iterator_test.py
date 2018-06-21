#
# (c) Copyright 2018 DESY
#
# This file is part of python-pninexus.
#
# python-pninexus is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# python-pninexus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with python-pninexus.  If not, see <http://www.gnu.org/licenses/>.
# ===========================================================================
#
# Created on: Mar 9, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import os.path
import unittest

from pninexus import h5cpp
from pninexus import nexus

file_structure = """
<group name="entry" type="NXentry">

  <group name="instrument" type="NXinstrument">

      <group name="detector_1" type="NXdetector">
          <field name="data" type="uint32" units="cps">
              <dimensions rank="3">
                  <dim value="0" index="1"/>
                  <dim value="1024" index="2"/>
                  <dim value="512" index="3"/>
              </dimensions>
              <chunk rank="3">
                  <dim value="100" index="1"/>
                  <dim value="1024" index="2"/>
                  <dim value="512" index="3"/>
              </chunk>
          </field>
      </group>

      <group name="detector_2" type="NXdetector">
            <field name="data" type="uint32" units="cps">
              <dimensions rank="3">
                  <dim value="0" index="1"/>
                  <dim value="512" index="2"/>
                  <dim value="1024" index="3"/>
              </dimensions>
              <chunk rank="3">
                  <dim value="100" index="1"/>
                  <dim value="512" index="2"/>
                  <dim value="1024" index="3"/>
              </chunk>
          </field>
      </group>
  </group>

  <group name="sample" type="NXsample">
      <field name="name" type="string">MySample</field>
  </group>

  <group name="data" type="NXdata">
      <link name="energy" target="/entry/instrument/monochromator/energy"/>
  </group>
</group>

"""

module_path = os.path.abspath(os.path.dirname(__file__))


class NodeIteratorTest(unittest.TestCase):

    filename = os.path.join(module_path, "NodeIteratorTest.h5")

    @classmethod
    def setUpClass(cls):
        super(NodeIteratorTest, cls).setUpClass()

        file = nexus.create_file(
            cls.filename, h5cpp.file.AccessFlags.TRUNCATE)
        nexus.create_from_string(file.root(), file_structure)

    def setUp(self):

        self.file = nexus.open_file(
            self.filename, h5cpp.file.AccessFlags.READONLY)
        self.root = self.file.root()

    def tearDown(self):

        self.root.close()
        self.file.close()

    def test_immediate_children_iteration(self):

        entry = self.root.nodes["entry"]
        self.assertTrue(entry.nodes.exists("instrument"))
        self.assertTrue(entry.nodes.exists("data"))
        self.assertTrue(entry.nodes.exists("sample"))

        nodes = [node.link.path.name for node in entry.nodes]
        refnames = ["data", "instrument", "sample"]
        self.assertListEqual(nodes, refnames,
                             "Comparison of node names: {}".format(nodes))

    def test_recursive_node_iteration(self):

        instrument = h5cpp.node.get_node(
            self.root, h5cpp.Path("/entry/instrument"))

        node_paths = [node.link.path for node in instrument.nodes.recursive]
        paths = [h5cpp.Path("/entry/instrument/detector_1"),
                 h5cpp.Path("/entry/instrument/detector_1/data"),
                 h5cpp.Path("/entry/instrument/detector_2"),
                 h5cpp.Path("/entry/instrument/detector_2/data")
                 ]
        self.assertListEqual(
            node_paths, paths, "Path comparison: {}".format(node_paths))

    def test_exists(self):

        data = h5cpp.node.get_node(self.root, h5cpp.Path("/entry/data"))
        self.assertTrue(data.links.exists("energy"))
        link = data.links["energy"]
        self.assertFalse(link.is_resolvable)
        self.assertTrue(data.is_valid)
        self.assertFalse(data.nodes.exists("energy"))
