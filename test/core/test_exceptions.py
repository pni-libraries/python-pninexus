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
# Created on: Aug 3, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
import unittest
import sys

import pni.core
from . import ex_trans_test

class test_exceptions(unittest.TestCase):
    def test_throw_memory_allocation_error(self):
        self.assertRaises(MemoryError,ex_trans_test.throw_memory_allocation_error)

    def test_throw_memory_not_allocated_error(self):
        self.assertRaises(MemoryError,ex_trans_test.throw_memory_not_allocated_error)
    
    def test_throw_shape_mismatch_error(self):
        self.assertRaises(pni.core.ShapeMismatchError,
                          ex_trans_test.throw_shape_mismatch_error)
    
    def test_throw_size_mismatch_error(self):
        self.assertRaises(pni.core.SizeMismatchError,
                          ex_trans_test.throw_size_mismatch_error)

    def test_throw_index_error(self):
        self.assertRaises(IndexError,ex_trans_test.throw_index_error)

    def test_throw_key_error(self):
        self.assertRaises(KeyError,ex_trans_test.throw_key_error)

    def test_throw_file_error(self):
        self.assertRaises(pni.core.FileError,
                          ex_trans_test.throw_file_error)

    def test_throw_type_error(self):
        self.assertRaises(TypeError,ex_trans_test.throw_type_error)

    def test_throw_value_error(self):
        self.assertRaises(ValueError,ex_trans_test.throw_value_error)

    def test_throw_range_error(self):
        self.assertRaises(pni.core.RangeError,
                          ex_trans_test.throw_range_error)

    def test_throw_not_implemented_error(self):
        self.assertRaises(NotImplementedError,
                          ex_trans_test.throw_not_implemented_error)

    def test_throw_iterator_error(self):
        self.assertRaises(pni.core.IteratorError,
                          ex_trans_test.throw_iterator_error)
    
    def test_throw_cli_argument_error(self):
        self.assertRaises(pni.core.CliArgumentError,
                          ex_trans_test.throw_cli_argument_error)
    
    def test_throw_cli_error(self):
        self.assertRaises(pni.core.CliError,
                          ex_trans_test.throw_cli_error)

