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
# Created on: Feb 22, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import os

from pninexus import h5cpp
from pninexus import nexus

module_path = os.path.dirname(os.path.abspath(__file__))

master_file_layout = """
<group name="/" type="NXroot">
    <group name="scan_0001" type="NXentry">
        <group name="instrument" type="NXinstrument">
        </group>

        <group name="data" type="NXdata">
            <link
target="Issue5Regression_detector.nxs://scan_0001/instrument/detector/data"
                  name="data"/>
        </group>

    </group>
</group>
"""

detector_file_layout = """
<group name="/" type="NXroot">
    <group name="scan_0001" type="NXentry">
        <group name="instrument" type="NXinstrument">
            <group name="detector" type="NXdetector">
                <field name="data" type="uint32"/>
            </group>
        </group>

    </group>
</group>
"""


class Issue5Regression(unittest.TestCase):

    master_file = os.path.join(module_path, "Issue5Regression_master.nxs")
    detector_file = os.path.join(module_path, "Issue5Regression_detector.nxs")

    @classmethod
    def setUpClass(cls):
        super(Issue5Regression, cls).setUpClass()

        f = nexus.create_file(
            cls.master_file, h5cpp.file.AccessFlags.TRUNCATE)
        nexus.create_from_string(parent=f.root(), xml=master_file_layout)

        f = nexus.create_file(
            cls.detector_file, h5cpp.file.AccessFlags.TRUNCATE)
        nexus.create_from_string(parent=f.root(), xml=detector_file_layout)

    def test_link_path(self):

        f = nexus.open_file(self.master_file)
        nodes = nexus.get_objects(
            f.root(), nexus.Path.from_string("/:NXentry/:NXdata"))
        self.assertEqual(len(nodes), 1)

        g = nodes[0]
        self.assertTrue(isinstance(g, h5cpp.node.Group))
        self.assertTrue(g.links.exists("data"))
        ln = g.links["data"]
        self.assertEqual(str(ln.path), "/scan_0001/data/data")
        self.assertTrue(isinstance(ln, h5cpp.node.Link))
        self.assertEqual(ln.type(), h5cpp.node.LinkType.EXTERNAL)
