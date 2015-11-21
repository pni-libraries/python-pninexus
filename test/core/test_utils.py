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
import pni.core
import unittest
import numpy
import sys

from . import config
from . import utils_test

from . import test_data

class test_utils(unittest.TestCase):
    def test_list_from_vector(self):
        l = utils_test.list_from_vector();
        self.assertEqual(len(l),4)

        index = 1
        for x in l:
            self.assertEqual(x,index)
            index += 1

    def test_list_from_list(self):
        l = utils_test.list_from_list();
        self.assertEqual(len(l),4)

        index = 1
        for x in l:
            self.assertEqual(x,index)
            index += 1

    def test_vector_from_list(self):
        l = [0,1,2,3]
        self.assertTrue(utils_test.vector_from_list(l))

    def test_is_unicode(self):
        self.assertTrue(utils_test.is_unicode(test_data.uc_string))
        self.assertTrue(not utils_test.is_unicode(test_data.str_string))

    def test_unicode2str(self):
        
        u = utils_test.unicode2str(test_data.uc_string)
        self.assertTrue(not utils_test.is_unicode(u))
    
    def test_is_int(self):
        self.assertTrue(utils_test.is_int(test_data.int_number))
        self.assertTrue(not utils_test.is_int(1+2j))
        self.assertTrue(not utils_test.is_int(1.2))
        self.assertTrue(not utils_test.is_int(numpy.arange(1,20)))
        self.assertTrue(not utils_test.is_int(test_data.uc_number))
        self.assertTrue(not utils_test.is_int(test_data.str_number))

    def test_is_long(self):
        #in Python 3 integer and long are the same
        if config.PY_MAJOR_VERSION >= 3:
            self.assertTrue(utils_test.is_long(test_data.int_number))
        else:
            self.assertTrue(not utils_test.is_long(test_data.int_number))

        self.assertTrue(utils_test.is_long(test_data.long_number))
        self.assertTrue(not utils_test.is_long(1.2))
        self.assertTrue(not utils_test.is_long(1+3j))
        self.assertTrue(not utils_test.is_long(test_data.uc_number))
        self.assertTrue(not utils_test.is_long(test_data.str_number))
        self.assertTrue(not utils_test.is_long(numpy.arange(1,20)))

    def test_is_float(self):
        self.assertTrue(utils_test.is_float(1.2))
        self.assertTrue(not utils_test.is_float(1))
        self.assertTrue(not utils_test.is_float(1+3j))
        self.assertTrue(not utils_test.is_float(test_data.uc_number))
        self.assertTrue(not utils_test.is_float(test_data.str_number))
        self.assertTrue(not utils_test.is_float(numpy.arange(1,10)))

    def test_is_bool(self):
        self.assertTrue(utils_test.is_bool(True))
        self.assertTrue(utils_test.is_bool(False))

        self.assertTrue(not utils_test.is_bool(test_data.int_number))
        self.assertTrue(not utils_test.is_bool(test_data.long_number))
        self.assertTrue(not utils_test.is_bool(1.2))
        self.assertTrue(not utils_test.is_bool(1+32j))
        self.assertTrue(not utils_test.is_bool(test_data.uc_number))
        self.assertTrue(not utils_test.is_bool(test_data.str_number))
        self.assertTrue(not utils_test.is_bool(numpy.arange(1,10)))

    def test_is_complex(self):
        self.assertTrue(utils_test.is_complex(1+3j))
        
        self.assertTrue(not utils_test.is_complex(test_data.int_number))
        self.assertTrue(not utils_test.is_complex(test_data.long_number))
        self.assertTrue(not utils_test.is_complex(1.2))
        self.assertTrue(not utils_test.is_complex(test_data.uc_number))
        self.assertTrue(not utils_test.is_complex(test_data.str_number))
        self.assertTrue(not utils_test.is_complex(numpy.arange(1,10)))

    def test_is_string(self):
        self.assertTrue(utils_test.is_string(test_data.str_string))

        self.assertTrue(not utils_test.is_string(test_data.int_number))
        self.assertTrue(not utils_test.is_string(test_data.long_number))
        self.assertTrue(not utils_test.is_string(1.2))
        self.assertTrue(not utils_test.is_string(True))
        self.assertTrue(not utils_test.is_string(test_data.uc_number))
        self.assertTrue(not utils_test.is_string(numpy.arange(1,10)))

    def test_is_scalar(self):
        self.assertTrue(utils_test.is_scalar(test_data.int_number))
        self.assertTrue(utils_test.is_scalar(True))
        self.assertTrue(utils_test.is_scalar(1.2))
        self.assertTrue(utils_test.is_scalar(test_data.long_number))
        self.assertTrue(utils_test.is_scalar(test_data.str_string))
        self.assertTrue(utils_test.is_scalar(test_data.uc_string))
        self.assertTrue(utils_test.is_scalar(1+3j))




