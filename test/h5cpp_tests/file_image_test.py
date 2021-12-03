#
# (c) Copyright 2015 DESY,
#               2020 Jan Kontaski <jan.kotanski@desy.de>
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
# Created on: Oct 2, 2015
#     Author: Jan Kotanski <jan.kotanski@desy.de>
#
from __future__ import print_function
import unittest
import os

from pninexus.h5cpp.file import File
from pninexus.h5cpp.file import ImageFlags
from pninexus.h5cpp.datatype import kVariableString
from pninexus.h5cpp.file import AccessFlags
from pninexus.h5cpp.file import create, from_buffer
from pninexus.h5cpp.file import open as h5open
import time
import numpy as np


class ImageTest(unittest.TestCase):

    file_path = os.path.split(__file__)[0]
    filename = os.path.join(file_path, "ImageTest.h5")
    filename2 = os.path.join(file_path, "ImageTest2.h5")

    # ------------------------------------------------------------------------
    def tearDown(self):

        try:
            os.remove(self.filename)
        except Exception:
            pass

        try:
            os.remove(self.filename2)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    def test_image_from_buffer(self):
        """
        Tests loading a file from a buffer
        """
        hdf5_version = "1.0.4"
        f1 = create(self.filename, AccessFlags.TRUNCATE)
        r1 = f1.root()
        a1 = r1.attributes.create("HDF5_Version", kVariableString)
        a1.write(hdf5_version)
        a1.close()
        r1.close()
        f1.close()

        ibuffer = np.fromfile(self.filename, dtype='uint8')
        f2 = from_buffer(ibuffer)
        r2 = f2.root()
        a2 = r2.attributes["HDF5_Version"]
        self.assertTrue(a2.read(), hdf5_version)
        a2.close()
        r2.close()
        f2.close()

    def test_image_to_buffer(self):
        """
        Tests copying a file to a buffer
        """
        hdf5_version = "1.0.3"
        f1 = create(self.filename, AccessFlags.TRUNCATE)
        r1 = f1.root()
        a1 = r1.attributes.create("HDF5_Version", kVariableString)
        a1.write(hdf5_version)
        size = f1.buffer_size
        obuffer = np.zeros(shape=[size], dtype='uint8')
        realsize = f1.to_buffer(obuffer)
        self.assertEqual(size, realsize)
        obuffer.tofile(self.filename2)
        a1.close()
        r1.close()
        f1.close()

        f2 = h5open(self.filename2)
        r2 = f2.root()
        a2 = r2.attributes["HDF5_Version"]
        self.assertTrue(a2.read(), hdf5_version)
        a2.close()
        r2.close()
        f2.close()

    def test_image_buffer(self):
        """
        Tests copying a file to a buffer
        """
        hdf5_version = "1.0.2"
        f1 = create(self.filename, AccessFlags.TRUNCATE)
        r1 = f1.root()
        a1 = r1.attributes.create("HDF5_Version", kVariableString)
        a1.write(hdf5_version)

        size = f1.buffer_size
        obuffer = np.zeros(shape=[size], dtype='uint8')
        realsize = f1.to_buffer(obuffer)
        self.assertEqual(size, realsize)
        a1.close()
        r1.close()
        f1.close()

        f2 = from_buffer(obuffer, ImageFlags.READWRITE)
        r2 = f2.root()
        a2 = r2.attributes["HDF5_Version"]
        self.assertTrue(a2.read(), hdf5_version)
        a2.close()
        r2.close()
        f2.close()
