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
        //select the data type to use for writing the array data
        switch(PyArray_TYPE(o.ptr()))
        {
            case PyArray_UINT8:  return pni::core::type_id_t::UINT8;
            case PyArray_INT8:   return pni::core::type_id_t::INT8;
            case PyArray_UINT16: return pni::core::type_id_t::UINT16;
            case PyArray_INT16:  return pni::core::type_id_t::INT16;
            case PyArray_UINT32: return pni::core::type_id_t::UINT32;
            case PyArray_INT32:  return pni::core::type_id_t::INT32;
            case PyArray_UINT64: return pni::core::type_id_t::UINT64;
            case PyArray_INT64:  return pni::core::type_id_t::INT64;
            case PyArray_FLOAT32:    return pni::core::type_id_t::FLOAT32;
            case PyArray_FLOAT64:    return pni::core::type_id_t::FLOAT64;
            case PyArray_LONGDOUBLE: return pni::core::type_id_t::FLOAT128;
            case NPY_CFLOAT:      return pni::core::type_id_t::COMPLEX32;
            case NPY_CDOUBLE:     return pni::core::type_id_t::COMPLEX64;
            case NPY_CLONGDOUBLE: return pni::core::type_id_t::COMPLEX128;
            case NPY_BOOL:   return pni::core::type_id_t::BOOL;
            case NPY_STRING: return pni::core::type_id_t::STRING;
            default:
                throw pni::core::type_error(pni::core::EXCEPTION_RECORD,
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
        return PyArray_SIZE(o.ptr());
    }
//end of namespace
}
