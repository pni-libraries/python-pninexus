//
// (c) Copyright 2014 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
//
// This file is part of python-pniio.
//
// python-pniio is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pniio is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: May 14, 2014
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include "bool_converter.hpp"

bool_t_to_python_converter::bool_t_to_python_converter()
{
    to_python_converter<bool_t,bool_t_to_python_converter>();
}

//----------------------------------------------------------------------------
PyObject *bool_t_to_python_converter::convert(const bool_t &v)
{
    return incref(object(bool(v)).ptr());
}

//----------------------------------------------------------------------------
python_to_bool_t_converter::python_to_bool_t_converter()
{
    convns::registry::push_back(
    &convertible,
    &construct,
    type_id<bool_t>()
            );
}

//----------------------------------------------------------------------------
void* python_to_bool_t_converter::convertible(PyObject *obj_ptr)
{
    if(!PyBool_Check(obj_ptr)) return nullptr;
    return obj_ptr;
}
   
//----------------------------------------------------------------------------
void python_to_bool_t_converter::construct(PyObject *obj_ptr,rvalue_type *data)
{
    bool value = obj_ptr == Py_True ? true : false;

    void *storage = ((storage_type*)data)->storage.bytes;
    
    new (storage) bool_t(value);
    data->convertible = storage;
}
