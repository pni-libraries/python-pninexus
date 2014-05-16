#
#
# (c) Copyright 2014 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
#
# This file is part of python-pniio.
#
# python-pniio is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# python-pniio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
# ===========================================================================
#
# Created on: May 16, 2014
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#
"""
Module provides a data generator for random numbers. 

"""

import numpy as np
import numpy.random as random

#these are the typecodes supported by pniii
_typecodes=["uint8","int8","uint16","int16","uint32","int32","uint64","int64",
            "float32","float64","float128","complex32","complex64","complex128",
            "string","bool"]

class type_desc(object):
    def __init__(self,dt,s):
        self.dtype = dt
        self.scalar = s

_type_desc = {"uint8":type_desc(np.dtype("uint8"),np.uint8),
              "int8": type_desc(np.dtype("int8"),np.int8),
              "uint16": type_desc(np.dtype("uint16"),np.uint16),
              "int16":  type_desc(np.dtype("int16"),np.int16),
              "uint32": type_desc(np.dtype("uint32"),np.uint32),
              "int32":  type_desc(np.dtype("int32"),np.int32),
              "uint64": type_desc(np.dtype("uint64"),np.int64),
              "float32": type_desc(np.dtype("float32"),np.float32),
              "float64": type_desc(np.dtype("float64"),np.float64),
              "float128": type_desc(np.dtype("float128"),np.float128),
              "complex32": type_desc(np.dtype("complex64"),np.complex64),
              "complex64": type_desc(np.dtype("complex128"),np.complex128),
              "complex128": type_desc(np.dtype("complex256"),np.complex256),
              "string": type_desc(np.dtype("string"),np.str_),
              "bool":   type_desc(np.dtype("bool"),np.bool_)}


class data_generator(object):
    def __init__(self,tdesc,func):
        self._desc = tdesc
        self._func = func

    def __call__(self,shape=()):

        #just allocate the data array
        if shape:
            data =  numpy.zeros(shape,dtype=self._desc.dtype)
            for x in numpy.nditer(data,op_flags=['readwrite']):
                x[...] = self._func()

        else: 
            data = self._desc.scalar(self._func())
        
        return data


class generator(object):
    def __init__(self,f,min_val,max_val):
        self._func = f
        self._min = min_val
        self._max = max_val

    def __call__(self):
        return self._func(self._min,self._max)
    


def int_generator_func(min_val=0,max_val=100):
    return random.randint(min_val,max_val)

def float_generator_func(min_val=0,max_val=100):
    delta = max_val-min_val
    return min_val-delta*random.ranf()

def complex_generator_func(min_val=0,max_val=100):
    return complex(float_generator_func(min_val,max_val),
                   float_generator_func(min_val,max_val))


def bool_generator_func(min_val=0,max_val=0):
    return int_generator_func(0,2)



def create_generator(typecode,min_val=0,max_val=100):
    """
     
    """
    tdesc = _type_desc[typecode]

    if tdesc.dtype.kind in ('i','u'):
        g = generator(int_generator_func,min_val,max_val)
    elif tdesc.dtype.kind == 'f':
        g = generator(float_generator_func,min_val,max_val)
    elif tdesc.dtype.kind == 'c':
        g = generator(complex_generator_func,min_val,max_val)
    elif tdesc.dtype.kind == 'b':
        print 'create bool generator'
        g = generator(bool_generator_func,0,1)

    
    return data_generator(tdesc,g)

