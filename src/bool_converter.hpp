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
// Created on: May 7, 2014
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <pni/core/types.hpp>
#include <boost/python.hpp>

using namespace pni::core;
using namespace boost::python;

struct bool_t_to_python_converter
{
    static PyObject *convert(const bool_t &v)
    {
        return incref(object(bool(v)).ptr());
    }

};

struct python_to_bool_t_converter
{
    python_to_bool_t_converter()
    {
        boost::python::converter::registry::push_back(
        &convertible,
        &construct,
        type_id<bool_t>()
                );
    }

    static void* convertible(PyObject *obj_ptr)
    {
        if(!PyBool_Check(obj_ptr)) return nullptr;
        return obj_ptr;
    }

    static void construct(PyObject *obj_ptr,
                          boost::python::converter::rvalue_from_python_stage1_data *data)
    {
        typedef boost::python::converter::rvalue_from_python_storage<bool_t> storage_type;

        bool value = obj_ptr == Py_True ? true : false;

        void *storage = ((storage_type*)data)->storage.bytes;
        
        new (storage) bool_t(value);
        data->convertible = storage;
    }
};

