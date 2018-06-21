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
# Created on: Feb 15, 2018
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
from __future__ import print_function
import unittest
import os

from pninexus import h5cpp
from pninexus.h5cpp import node

module_path = os.path.dirname(os.path.abspath(__file__))


class FunctionTests(unittest.TestCase):

    filename = os.path.join(module_path, "FunctionTests.h5")

    def setUp(self):
        self.file = h5cpp.file.create(
            self.filename, h5cpp.file.AccessFlags.TRUNCATE)
        self.root = self.file.root()

        node.Group(self.root, "plot")
        node.Group(self.root, "sensors")
        self.dataset = node.Dataset(
            self.root, h5cpp.Path("sensors/temperature"),
            h5cpp.datatype.kInt32,
            h5cpp.dataspace.Scalar())
        self.dataset.write(42)

    def tearDown(self):
        self.dataset.close()
        self.root.close()
        self.file.close()


class CopyTests(FunctionTests):

    def testCopyToPath(self):

        h5cpp.node.copy(self.dataset, self.root, h5cpp.Path("plot/data"))
        g = self.root.nodes["plot"]
        self.assertTrue(g.nodes.exists("data"))
        d = g.nodes["data"]
        self.assertEqual(d.type, h5cpp.node.Type.DATASET)

    def testCopyDefault(self):

        h5cpp.node.copy(self.dataset, self.root)
        self.assertTrue(self.root.nodes.exists("temperature"))
        d = self.root.nodes["temperature"]
        self.assertEqual(d.type, h5cpp.node.Type.DATASET)


class MoveTests(FunctionTests):

    def testMoveToPath(self):

        h5cpp.node.move(self.dataset, self.root, h5cpp.Path("plot/data"))
        self.assertFalse(
            self.root.nodes["sensors"].nodes.exists("temperature"))
        self.assertTrue(self.root.nodes["plot"].nodes.exists("data"))

        d = h5cpp.node.get_node(self.root, h5cpp.Path("plot/data"))
        self.assertEqual(d.read(), 42)

    def testMoveDefault(self):

        h5cpp.node.move(self.dataset, self.root)
        self.assertFalse(
            self.root.nodes['sensors'].nodes.exists('temperature'))
        self.assertTrue(self.root.nodes.exists('temperature'))


class RemoveTests(FunctionTests):

    def testRemoveByPath(self):

        self.assertTrue(
            self.root.nodes['sensors'].nodes.exists("temperature"))
        h5cpp.node.remove(
            base=self.root, path=h5cpp.Path("sensors/temperature"))
        self.assertFalse(
            self.root.nodes['sensors'].nodes.exists("temperature"))

    def testRemove(self):

        self.assertTrue(self.root.nodes['sensors'].nodes.exists("temperature"))
        h5cpp.node.remove(self.dataset)
        self.assertFalse(
            self.root.nodes['sensors'].nodes.exists("temperature"))


class LinkTests(FunctionTests):

    def testSymbolicLink(self):

        h5cpp.node.link(target=self.dataset,
                        link_base=self.root,
                        link_path=h5cpp.Path("plot/data"))
        d = h5cpp.node.get_node(self.root, h5cpp.Path("plot/data"))
        self.assertEqual(d.link.type(), h5cpp.node.LinkType.SOFT)

    def testSymbolicLinkFromPath(self):

        h5cpp.node.link(target=h5cpp.Path("/sensors/temperature"),
                        link_base=self.root,
                        link_path=h5cpp.Path("plot/data"))
        d = h5cpp.node.get_node(self.root, h5cpp.Path("plot/data"))
        self.assertEqual(d.link.type(), h5cpp.node.LinkType.SOFT)

    def testExternalLink(self):

        newfilename = os.path.join(module_path, "ExternalLink.h5")
        f = h5cpp.file.create(newfilename, h5cpp.file.AccessFlags.TRUNCATE)
        r = f.root()

        h5cpp.node.link(target_file=self.filename,
                        target=h5cpp.Path("/sensors/temperature"),
                        link_base=r,
                        link_path=h5cpp.Path("data"))
        d = r.nodes["data"]
        self.assertEqual(d.link.path.name, "data")
        self.assertEqual(d.link.type(), h5cpp.node.LinkType.EXTERNAL)
