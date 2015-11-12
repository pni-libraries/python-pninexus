"""
Utilities for IO tests.
"""

import numpy

discrete_types = [
        "uint8","int8","uint16","int16","uint32","int32","uint64","int64",        
        "string","bool"
        ]

float_types = ["float32","float64","float128",
               "complex32","complex64","complex128"]

types = discrete_types + float_types

scalars={"uint8":numpy.uint8,"int8":numpy.int8,
        "uint16":numpy.uint16,"int16":numpy.int16,
        "uint32":numpy.uint32,"int32":numpy.int32,
        "uint64":numpy.uint64,"int64":numpy.int64,
        "float32":numpy.float32,"float64":numpy.float64,
        "float128":numpy.float128,
        "complex32":numpy.complex64,
        "complex64":numpy.complex128,
        "complex128":numpy.complex256,
        "string":numpy.str_,"bool":numpy.bool_}


def is_discrete_type(tc):
    if tc in discrete_types:
        return True
    else:
        return False
