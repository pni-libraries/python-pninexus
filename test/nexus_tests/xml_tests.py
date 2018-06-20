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
# Created on: Feb 19, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import os
from pninexus import h5cpp
from pninexus.h5cpp.file import AccessFlags
from pninexus import nexus

module_path = os.path.dirname(os.path.abspath(__file__))


basic_file = """
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="source" type="NXsource">
        </group>

        <group name="undulator" type="NXinsertion_device">
        </group>


    </group>

    <group name="sample" type="NXsample">
    </group>

    <group name="data" type="NXdata">
    </group>

</group>

"""


class XMLTests(unittest.TestCase):

    filename = os.path.join(module_path, "XMLTests.nxs")
    detector_file = os.path.join(module_path, "detector.xml")

    def setUp(self):

        self.file = nexus.create_file(self.filename, AccessFlags.TRUNCATE)
        self.root = self.file.root()

    def tearDown(self):

        self.root.close()
        self.file.close()

    def test_create_from_string(self):

        nexus.create_from_string(self.root, basic_file)

    def test_create_from_file(self):

        nexus.create_from_string(self.root, basic_file)
        instrument = h5cpp.node.get_node(
            self.root, h5cpp.Path("/entry/instrument"))
        nexus.create_from_file(instrument, self.detector_file)
        detector = h5cpp.node.get_node(
            self.root, h5cpp.Path("/entry/instrument/mythen"))

        self.assertEqual(detector.link.path.name, 'mythen')
        self.assertEqual(detector.nodes["description"].read(),
                         "DECTRIS strip detector")
