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
import unittest
# import sys
import os.path

from pninexus import h5cpp
#
# we use here some nexus functionality to simplify the generation of the
# test file
#
from pninexus import nexus

module_path = os.path.abspath(os.path.dirname(__file__))

file_structure = """
<group name="entry" type="NXentry">
    <group name="instrument" type="NXinstrument">
        <group name="source" type="NXsource">
        </group>

        <link name="detector_module_1"
              target="Lambda_module_1://entry/instrument/detector"/>
        <link name="detector_module_2"
              target="Lambda_module_2://entry/instrument/detector"/>
        <link name="detector_module_3"
              target="Lambda_module_3://entry/instrument/detector"/>
    </group>

    <group name="sample" type="NXsample">
    </group>

    <group name="data" type="NXdata">
    </group>
</group>
"""


class LinkIterationTest(unittest.TestCase):

    filename = os.path.join(module_path, "LinkIterationTest.h5")

    @classmethod
    def setUpClass(cls):
        super(LinkIterationTest, cls).setUpClass()

        file = nexus.create_file(
            cls.filename, h5cpp.file.AccessFlags.TRUNCATE)
        nexus.create_from_string(file.root(), file_structure)

    def setUp(self):
        unittest.TestCase.setUp(self)

        self.file = nexus.open_file(
            self.filename, h5cpp.file.AccessFlags.READONLY)
        self.root = self.file.root()

    def tearDown(self):
        unittest.TestCase.tearDown(self)

        self.root.close()
        self.file.root()

    def test_direct_iteration(self):

        entry = self.root.nodes["entry"]

        links = [link.path.name for link in entry.links]

        self.assertListEqual(links, ["data", "instrument", "sample"],
                             "Comparison of link names {}".format(links))

    def test_recursive_iteration(self):

        entry = self.root.nodes["entry"]

        links = [link.path.name for link in entry.links.recursive]

        refnames = ["data",
                    "instrument",
                    "source",
                    "detector_module_1",
                    "detector_module_2",
                    "detector_module_3",
                    "sample"]
        self.assertListEqual(links, refnames,
                             "Comparison of link names {}".format(links))
