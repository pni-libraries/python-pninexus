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


class NexusPathTests(unittest.TestCase):

    filename = os.path.join(module_path, "NexusPathTests.nxs")
    path_test_structure = os.path.join(module_path, "NexusPathTests.xml")

    @classmethod
    def setUpClass(cls):
        super(NexusPathTests, cls).setUpClass()

        f = nexus.create_file(cls.filename, AccessFlags.TRUNCATE)
        nexus.create_from_file(f.root(), cls.path_test_structure)

    def setUp(self):

        self.file = nexus.open_file(self.filename, AccessFlags.READONLY)
        self.root = self.file.root()

    def tearDown(self):
        self.root.close()
        self.file.close()

    def test_entry_selection(self):

        p = nexus.Path.from_string(":NXentry")

        entries = nexus.get_objects(self.root, p)
        self.assertEqual(len(entries), 3)
        for entry in entries:
            self.assertEqual(entry.attributes["NX_class"].read(), "NXentry")
            self.assertEqual(entry.type, h5cpp.node.Type.GROUP)

    def test_get_experiment_identifier(self):

        p = nexus.Path.from_string(":NXentry/experiment_identifier")

        ids = nexus.get_objects(self.root, p)
        self.assertEqual(len(ids), 3)

        for id in ids:
            self.assertEqual(id.type, h5cpp.node.Type.DATASET)
            self.assertEqual(id.link.path.name, "experiment_identifier")

    def test_detector_search(self):

        p = nexus.Path.from_string(
            "scan_001:NXentry/:NXinstrument/:NXdetector")
        detectors = nexus.get_objects(self.root, p)

        self.assertEqual(len(detectors), 4)
        for detector in detectors:
            self.assertEqual(detector.type, h5cpp.node.Type.GROUP)

    def test_get_path_for_detector(self):

        p = nexus.Path.from_string(
            "scan_001:NXentry/:NXinstrument/detector_01:NXdetector")
        detector = nexus.get_objects(self.root, p)[0]

        self.assertEqual(detector.type, h5cpp.node.Type.GROUP)
        p = nexus.get_path(detector)
        self.assertEqual(
            nexus.Path.to_string(p),
            "{filename}://scan_001:NXentry/instrument:NXinstrument/"
            "detector_01:NXdetector".format(filename=self.filename))
        res = [
            {'name': '/', 'base_class': 'NXroot'},
            {'name': 'scan_001', 'base_class': 'NXentry'},
            {'name': 'instrument', 'base_class': 'NXinstrument'},
            {'name': 'detector_01', 'base_class': 'NXdetector'}
        ]
        for i, element in enumerate(p):
            self.assertEqual(type(element), dict)
            self.assertEqual(element['name'], res[i]['name'])
            self.assertEqual(element['base_class'], res[i]['base_class'])

    def test_get_path(self):

        entry = self.root.nodes["scan_001"]
        p = nexus.get_path(entry)
        self.assertEqual(
            str(p),
            "{filename}://scan_001:NXentry".format(filename=self.filename))
        p2 = nexus.Path(p)
        self.assertEqual(
            str(p2),
            "{filename}://scan_001:NXentry".format(filename=self.filename))
