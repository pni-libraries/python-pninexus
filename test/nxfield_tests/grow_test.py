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
# Created on: Oct 6, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
import unittest
import numpy
import os

import nx as nx
from nx import nxfield
from nx import deflate_filter
from nx import create_file
from nx import open_file

from .. data_generator import random_generator_factory
from .. data_generator import _type_desc

#implementing test fixture
class grow_test_uint8(unittest.TestCase):
    """
    This test creates a single file as the only artifact. The result for 
    every individual test is written to the same file. This makes 
    investigation in case of errors easier.

    This testsuite handles 3 standard use cases:
    .) growing a 1D field - this would be storing scalar data 
    .) growing a 2D field - storing 1D MCA data 
    .) growing a 3D field - storing image data on a stack
    """
    _typecode="uint8"
    file_path = os.path.split(__file__)[0]
    file_name = "grow_test_{tc}.nxs".format(tc=_typecode)
    full_path = os.path.join(file_path,file_name)
    npts = 100
    nx = 10
    ny = 20
    frame_shape_2d=(nx,)
    frame_shape_3d=(nx,ny)

    @classmethod
    def setUpClass(self):
        """
        Setup the file where all the tests are performed. 
        """
        self.file_name = "grow_test_{tc}.nxs".format(tc=self._typecode)
        self.full_path = os.path.join(self.file_path,self.file_name)
        self.gf = create_file(self.full_path,overwrite=True)
        self.gf.close()

    #-------------------------------------------------------------------------
    def setUp(self):
        self.gf = open_file(self.full_path,readonly=False)
        self.root = self.gf.root()
        self.generator = random_generator_factory(self._typecode)

    #-------------------------------------------------------------------------
    def tearDown(self):
        self.root.close()
        self.gf.close()

    #-------------------------------------------------------------------------
    def test_grow_1D_field(self):
        f = self.root.create_field("data_1d",self._typecode,
                              shape=(0,),chunk=(self.nx,))
        
        dlist = []
        for (pts_index,data) in zip(range(self.npts),self.generator()):
            f.grow(0,1)
            f[-1] = data
            dlist.append(data)
            self.gf.flush()

        self.assertEqual(f.size,self.npts)
        self.assertEqual(f.shape[0],self.npts)
        
        recorded = f.read()
        dlist = numpy.array(dlist).astype(recorded.dtype)
        for (rec,ref) in zip(recorded.flat,dlist.flat):
            if _type_desc[f.dtype].dtype.kind in ('f','c'):
                self.assertAlmostEqual(rec,ref)
            else:
                self.assertEqual(rec,ref)

    #-------------------------------------------------------------------------
    def test_grow_2D_field(self):
        f = self.root.create_field("data_2d",self._typecode,
                              shape=(0,self.nx),chunk=(1,self.nx))

        dlist = []
        for (pts_index,data) in zip(range(self.npts),self.generator(shape=(self.nx,))):
            f.grow(0,1)
            f[-1,...] = data
            dlist.append(data)
            self.gf.flush()

        self.assertEqual(self.npts*self.nx,f.size)
        self.assertEqual(f.shape[0],self.npts)
        self.assertEqual(f.shape[1],self.nx)

        recorded = f.read()
        dlist = numpy.array(dlist).astype(recorded.dtype)
        for (rec,ref) in zip(recorded.flat,dlist.flat):
            self.assertEqual(rec,ref)

    #-------------------------------------------------------------------------
    def test_grow_3D_field(self):

        f = self.root.create_field("data_3d",self._typecode,
                                   shape=(0,self.nx,self.ny),
                                   chunk=(1,self.nx,self.ny))

        dlist = []
        s = (self.nx,self.ny)
        for (pts_index,data) in zip(range(self.npts),self.generator(shape=s)):
            f.grow(0,1)
            f[-1,...] = data
            dlist.append(data)
            self.gf.flush()

        self.assertEqual(self.npts*self.nx*self.ny,f.size)
        self.assertEqual(f.shape[0],self.npts)
        self.assertEqual(f.shape[1],self.nx)
        self.assertEqual(f.shape[2],self.ny)

        recorded = f.read()
        dlist = numpy.array(dlist).astype(recorded.dtype)
        for (rec,ref) in zip(recorded.flat,dlist.flat):
            self.assertEqual(rec,ref)


#=============================================================================
class grow_test_uint16(grow_test_uint8):
    _typecode = "uint16"

#=============================================================================
class grow_test_uint32(grow_test_uint8):
    _typecode = "uint32"

#=============================================================================
class grow_test_uint64(grow_test_uint8):
    _typecode = "uint64"

#=============================================================================
class grow_test_int8(grow_test_uint8):
    _typecode = "int8"

#=============================================================================
class grow_test_int16(grow_test_uint8):
    _typecode = "int16"

#=============================================================================
class grow_test_int32(grow_test_uint8):
    _typecode = "int32"

#=============================================================================
class grow_test_int64(grow_test_uint8):
    _typecode = "int64"

#=============================================================================
class grow_test_float32(grow_test_uint8):
    _typecode = "float32"

#=============================================================================
class grow_test_float64(grow_test_uint8):
    _typecode = "float64"

#=============================================================================
class grow_test_float128(grow_test_uint8):
    _typecode = "float128"

#=============================================================================
class grow_test_complex32(grow_test_uint8):
    _typecode = "complex32"

#=============================================================================
class grow_test_complex64(grow_test_uint8):
    _typecode = "complex64"

#=============================================================================
class grow_test_complex128(grow_test_uint8):
    _typecode = "complex128"

#=============================================================================
class grow_test_bool(grow_test_uint8):
    _typecode = "bool"

#=============================================================================
class grow_test_string(grow_test_uint8):
    _typecode = "string"
