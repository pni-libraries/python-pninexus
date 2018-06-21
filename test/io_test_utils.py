#
# (c) Copyright 2015 DESY,
#               2015 Eugen Wintersberger <eugen.wintersberger@desy.de>
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
# Created on: Oct 12, 2015
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#

"""
Utilities for IO tests.
"""

import numpy


types = ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64",
         "int64", "float32", "float64", "float128", "complex32", "complex64",
         "complex128", "string", "bool"]


scalars = {"uint8": numpy.uint8,
           "int8": numpy.int8,
           "uint16": numpy.uint16,
           "int16": numpy.int16,
           "uint32": numpy.uint32,
           "int32": numpy.int32,
           "uint64": numpy.uint64,
           "int64": numpy.int64,
           "float32": numpy.float32,
           "float64": numpy.float64,
           "float128": numpy.float128,
           "complex32": numpy.complex64,
           "complex64": numpy.complex128,
           "complex128": numpy.complex256,
           "string": numpy.str_,
           "bool": numpy.bool_}
