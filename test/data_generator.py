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
"""Module provides a data generator for random numbers. 

In many cases testing only makes sense if the data used for testing varies with
each run. For this purpose this module provides a random number data generator 
for all data types supported by pniio. 

"""

import numpy as np
import numpy.random as random
from functools import reduce
import sys



#these are the typecodes supported by pniii
_typecodes=["uint8","int8","uint16","int16","uint32","int32","uint64","int64",
            "float32","float64","float128","complex32","complex64","complex128",
            "bool"]


class type_desc(object):
    """Container for type information

    This class acts rather like a C structure and should just keep type 
    related information together.  
    It has a dtype and a scalar attribute. The former one holds a reference
    to the data type object of the type and the latter a refernece to the 
    appropriate scalar object.
    """
    def __init__(self,dt,s):
        """initialize a type descriptor
        
        Args:
            dt (numpy.dtype): data type object
            s  (numpy.generic): data type scalar
        """
        self.dtype = dt
        self.scalar = s

# a map 
_type_desc = {"uint8":type_desc(np.dtype("uint8"),np.uint8),
              "int8": type_desc(np.dtype("int8"),np.int8),
              "uint16": type_desc(np.dtype("uint16"),np.uint16),
              "int16":  type_desc(np.dtype("int16"),np.int16),
              "uint32": type_desc(np.dtype("uint32"),np.uint32),
              "int32":  type_desc(np.dtype("int32"),np.int32),
              "uint64": type_desc(np.dtype("uint64"),np.uint64),
              "int64": type_desc(np.dtype("int64"),np.int64),
              "float32": type_desc(np.dtype("float32"),np.float32),
              "float64": type_desc(np.dtype("float64"),np.float64),
              "float128": type_desc(np.dtype("float128"),np.float128),
              "complex32": type_desc(np.dtype("complex64"),np.complex64),
              "complex64": type_desc(np.dtype("complex128"),np.complex128),
              "complex128": type_desc(np.dtype("complex256"),np.complex256),
              "bool":   type_desc(np.dtype("bool"),np.bool_)}


if sys.version_info[0]>=3:
    _typecodes += "unicode"
    _type_desc["string"] = type_desc(np.dtype("unicode"),np.unicode_)
else:
    _typecodes += "string"
    _type_desc["string"] = type_desc(np.dtype("string"),np.str_)




#-----------------------------------------------------------------------------
class data_generator(object):
    def __init__(self,tdesc,func):
        self._desc = tdesc
        self._func = func

    def __call__(self,shape=()):

        #just allocate the data array
        if shape:
            size = reduce(lambda x,y: x*y, shape)
            l = [self._func() for i in range(size)]
            data = np.array(l,dtype=self._desc.dtype).reshape(shape)

        else: 
            data = self._desc.scalar(self._func())
        
        return data


#-----------------------------------------------------------------------------
class generator(object):
    """Generator function

    This class acts as a generator function. Everytime an instance of 
    this class is called a new random value will be generated. 
    It wraps the functions below to a single interface. 

    """
    def __init__(self,f,*args,**kargs):
        """Construct a generator function

        The constructor takes the function object and all its positional 
        and keyword arguments. 

        Args:
            f: the function to wrap
            args: all positional arguments for this function
            kargs: all keyword arguments for this function
        """
        self._func = f
        self._args = args
        self._kargs = kargs

    def __call__(self):
        """Execute the wrapped function
    
        Execute the wrapped function with all its arguments passed 
        during construction of this instance.
        """
        return self._func(*self._args,**self._kargs)
    


#-----------------------------------------------------------------------------
def int_generator_func(min_val=0,max_val=100):
    """Generate random integer numbers
    
    Generate a random integer number in the range of min_val to max_val. 

    Args:
        min_val (int): lower bound for random numbers
        max_val (int): upper bound for random numbers

    Return:
        int: random number 
    """

    while True:
        yield random.randint(min_val,max_val)

#-----------------------------------------------------------------------------
def float_generator_func(min_val=0,max_val=100):
    """Generate random float numbers

    Generate a random float number in the range of min_val and max_val.

    Args:
        min_val (float): lower bound for random numbers
        max_val (float): upper bound for random numbers

    Return:
        float: random number

    """
    delta = max_val-min_val

    while True:
        yield min_val-delta*random.ranf()

#-----------------------------------------------------------------------------
def complex_generator_func(min_val=0,max_val=100):
    """Generate random complex numbers
    
    Generate a random complex numbers whose real and imaginary part lie within 
    the bounds min_val and max_val. 

    Args:
        min_val (float): lower bound for imaginary and real part
        max_val (float): upper bound for imaginary and real part

    Return:
        complex: a complex random number
    """

    while True:
        yield complex(float_generator_func(min_val,max_val),
                      float_generator_func(min_val,max_val))


#-----------------------------------------------------------------------------
def bool_generator_func():
    """Generate boolean random numbers

    As boolean values can only take the values True and False no min_val 
    and max_val arguments are required. 

    Return:
        bool: random value
    """

    while True:
        yield int_generator_func(0,2)

#-----------------------------------------------------------------------------
def string_generator_func(min_val=0,max_val=100):
    """Generate a random string

    Generate a random string. The meaning of the arguments min_val and 
    max_val is slightly change for strings. As the length of the string is 
    a random number too, min_val and max_val denote the minimum and 
    maximum string length respectively. 
    The string will contain only printable characters from the ASCII range 
    33 to 126.

    Args:
        min_val (int): minimum length of the random string
        max_val (int): maximum length of the random string

    Return:
        str: random string whose length is within min_val and max_val
    """
    #generate the random length of the string
    size = int_generator_func(min_val,max_val)
   
    while True:
        yield ''.join(map(chr,[ int_generator_func(33,126) for i in range(size)]))

#-----------------------------------------------------------------------------
def random_generator_factory(typecode):
    """Create a data generator  

    This function returns a data generator function for the type determined by 
    typecode which generates random data within a range specified by min_val and 
    max_val.

    Args:
        typecode (str): typecode of the data that shall be produced
        min_val (int|float optional): lower bound of the data to be generated
        max_val (int|float optional): upper bound for the data to be generated

    Returns: 
        data_generator: an instance of class data_generator
    Raises:
        TypeError: if the type code is not  supported by pniio
    """
    tdesc = _type_desc[typecode]

    if tdesc.dtype.kind in ('i','u'):
        return int_generator_func
    elif tdesc.dtype.kind == 'f':
        return float_generator_func
    elif tdesc.dtype.kind == 'c':
        return complex_generator_func
    elif tdesc.dtype.kind == 'b':
        return bool_generator_func
    elif tdesc.dtype.kind == 'S' or tdesc.dtype.kind == 'U':
        return string_generator_func
    else:
        TypeError,'unsupported type code'


