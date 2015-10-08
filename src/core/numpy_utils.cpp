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

#include "numpy_utils.hpp"

using namespace boost::python;

namespace numpy
{

    //------------------------------------------------------------------------
    bool is_array(const object &o)
    {
        //if the object is not allocated we assume that it is not an array
        if(o.ptr())
            return PyArray_CheckExact(o.ptr());
        else
            return false;
    }

    //------------------------------------------------------------------------
    bool is_scalar(const object &o)
    {
        if(o.ptr())
            return PyArray_CheckScalar(o.ptr());
        else
            return false;
    }


    //------------------------------------------------------------------------
    pni::core::type_id_t type_id(const object &o)
    {
        using namespace pni::core; 

        if(!numpy::is_array(o) && !numpy::is_scalar(o))
            throw type_error(EXCEPTION_RECORD,
                    "Python object must be a numpy array or scalar!");

        //select the data type to use for writing the array data
        switch(PyArray_TYPE((const PyArrayObject*)o.ptr()))
        {
            case NPY_UINT8:      return type_id_t::UINT8;
            case NPY_INT8:       return type_id_t::INT8;
            case NPY_UINT16:     return type_id_t::UINT16;
            case NPY_INT16:      return type_id_t::INT16;
            case NPY_UINT32:     return type_id_t::UINT32;
            case NPY_INT32:      return type_id_t::INT32;
            case NPY_UINT64:     return type_id_t::UINT64;
            case NPY_INT64:      return type_id_t::INT64;
            case NPY_FLOAT32:    return type_id_t::FLOAT32;
            case NPY_FLOAT64:    return type_id_t::FLOAT64;
            case NPY_LONGDOUBLE: return type_id_t::FLOAT128;
            case NPY_COMPLEX64:      return type_id_t::COMPLEX32;
            case NPY_CDOUBLE:        return type_id_t::COMPLEX64;
            case NPY_CLONGDOUBLE:    return type_id_t::COMPLEX128;
            case NPY_BOOL:           return type_id_t::BOOL;
#if PY_MAJOR_VERSION >= 3
            case NPY_UNICODE: return type_id_t::STRING;
#else
            case NPY_STRING:  return type_id_t::STRING;
#endif
            default:
                throw type_error(EXCEPTION_RECORD,
                "Type of numpy array cannot be handled!");
        };

    }

    //------------------------------------------------------------------------
    pni::core::string type_str(pni::core::type_id_t id)
    {
        return pni::core::str_from_type_id(id);
    }

    //------------------------------------------------------------------------
    pni::core::string type_str(const object &o)
    {
        return type_str(type_id(o));
    }

    //------------------------------------------------------------------------
    size_t get_size(const object &o)
    {
        using namespace pni::core;
        if(!is_array(o))
            throw type_error(EXCEPTION_RECORD,
                    "Argument must be a numpy array!");

        return PyArray_SIZE((PyArrayObject*)o.ptr());
    }
//end of namespace
}
