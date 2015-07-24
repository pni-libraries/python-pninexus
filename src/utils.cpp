//
// (c) Copyright 2014 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
//
// This file is part of python-pnicore.
//
// python-pnicore is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pnicore is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pnicore.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Oct 21, 2014
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

//helper functions to create wrappers

//#define NO_IMPORT_ARRAY
extern "C"{
#include<Python.h>
}

#include <pni/core/types.hpp>


#include <boost/python/slice.hpp>
#include "utils.hpp"
#include "error_utils.hpp"
#include "numpy_utils.hpp"

using namespace boost::python;

//-----------------------------------------------------------------------------
bool is_unicode(const object &o)
{
    if(PyUnicode_Check(o.ptr())) return true;
    return false;
}

//-----------------------------------------------------------------------------
object unicode2str(const object &o)
{
    PyObject *ptr = PyUnicode_AsUTF8String(o.ptr());
    return object(handle<>(ptr));
}

//----------------------------------------------------------------------------
bool is_int(const object &o)
{
#if PY_MAJOR_VERSION >= 3
    return PyLong_CheckExact(o.ptr()) ? true : false;
#else
    return PyInt_CheckExact(o.ptr()) ? true : false;
#endif
}

//----------------------------------------------------------------------------
bool is_bool(const object &o)
{
    return PyBool_Check(o.ptr()) ? true : false;
}

//----------------------------------------------------------------------------
bool is_long(const object &o)
{
    return PyLong_CheckExact(o.ptr()) ? true : false;
}

//----------------------------------------------------------------------------
bool is_float(const object &o)
{
    return PyFloat_CheckExact(o.ptr()) ? true : false;
}

//----------------------------------------------------------------------------
bool is_complex(const object &o)
{
    return PyComplex_CheckExact(o.ptr()) ? true : false;
}

//----------------------------------------------------------------------------
bool is_string(const object &o)
{
#if PY_MAJOR_VERSION >= 3
    return PyBytes_CheckExact(o.ptr()) ? true : false; 
#else
    return PyString_CheckExact(o.ptr()) ? true : false;
#endif
}

//----------------------------------------------------------------------------
bool is_scalar(const object &o)
{
    return is_unicode(o) || is_int(o) || is_bool(o) || is_long(o) ||
           is_float(o) || ::is_complex(o) || is_string(o) ||
           numpy::is_scalar(o);
}
