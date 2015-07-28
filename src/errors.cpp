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

extern "C"{
#include<Python.h>
}

#include <sstream>
#include <pni/core/error.hpp>
#include <boost/python.hpp>

#include "error_utils.hpp"

using namespace pni::core;
using namespace boost::python;

//
// Translation map between PNI and Python exceptions
// 
//   C++ exception             -> Python 3.X           -> Python 2.X
// memory_allocation_error     -> PyExc_MemoryError    -> PyExc_MemoryError
// memory_not_allocated_error  -> PyExc_MemoryError    -> PyExc_MemoryError
// shape_mismatch_error        -> 
// size_mismatch_error         ->
// index_error                 -> PyExc_IndexError     -> PyExc_IndexError
// key_error                   -> PyExc_KeyError       -> PyExc_IndexError
// file_error                  -> 
// type_error                  -> PyExc_TypeError      -> PyExc_TypeError
// value_error                 -> PyExc_ValueError     -> PyExc_ValueError
// range_error                 -> 
// not_implemented_error       -> PyExc_NotImplementedError -> PyExc_NotImplementedError
// iterator_error              -> 
// cli_argument_error          -> 
// cli_error                   -> 


//-----------------------------------------------------------------------------
void memory_allocation_error_translator(memory_allocation_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_MemoryError,stream.str().c_str());
}

//-----------------------------------------------------------------------------
void memory_not_allocated_error_translator(memory_not_allocated_error const
        &error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_MemoryError,stream.str().c_str());
}

//-----------------------------------------------------------------------------
void index_error_translator(index_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_IndexError,stream.str().c_str());
}

//-----------------------------------------------------------------------------
void key_error_translator(key_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_KeyError,stream.str().c_str());
}

//-----------------------------------------------------------------------------
void type_error_translator(type_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_TypeError,stream.str().c_str());
}

//-----------------------------------------------------------------------------
void value_error_translator(value_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_ValueError,stream.str().c_str());
}

//====================General purpose exceptions===============================
//ERR_TRANSLATOR(memory_allocation_error)
//ERR_TRANSLATOR(memory_not_allocated_error)
//ERR_TRANSLATOR(shape_mismatch_error)
ERR_TRANSLATOR(size_mismatch_error)
//ERR_TRANSLATOR(index_error)
//ERR_TRANSLATOR(key_error)
ERR_TRANSLATOR(file_error)
//ERR_TRANSLATOR(type_error)
//ERR_TRANSLATOR(value_error)
ERR_TRANSLATOR(range_error)
ERR_TRANSLATOR(not_implemented_error)
ERR_TRANSLATOR(iterator_error)
ERR_TRANSLATOR(cli_argument_error)
ERR_TRANSLATOR(cli_error)

static PyObject *PyExc_ShapeMismatchError;


void shape_mismatch_error_translator(shape_mismatch_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_ShapeMismatchError,stream.str().c_str());
}


//-----------------------------------------------------------------------------
void exception_registration()
{
    //define the base class for all exceptions
    const string &(exception::*exception_get_description)() const =
        &exception::description;
    class_<exception>("Exception")
        .def(init<>())
        .add_property("description",make_function(exception_get_description,return_value_policy<copy_const_reference>()))
        .def(self_ns::str(self_ns::self))
        ;

    PyExc_ShapeMismatchError = PyErr_NewException("pni.core.ShapeMismatchError",nullptr,nullptr);
    scope().attr("ShapeMismatchError") = object(handle<>(borrowed(PyExc_ShapeMismatchError)));

    //ERR_OBJECT_DECL(memory_allocation_error);
    //ERR_OBJECT_DECL(memory_not_allocated_error);
    //ERR_OBJECT_DECL(shape_mismatch_error);
    ERR_OBJECT_DECL(size_mismatch_error);
    //ERR_OBJECT_DECL(index_error);
    //ERR_OBJECT_DECL(key_error);
    ERR_OBJECT_DECL(file_error);
    //ERR_OBJECT_DECL(type_error);
    //ERR_OBJECT_DECL(value_error);
    ERR_OBJECT_DECL(range_error);
    ERR_OBJECT_DECL(not_implemented_error);
    ERR_OBJECT_DECL(iterator_error);
    ERR_OBJECT_DECL(cli_argument_error);
    ERR_OBJECT_DECL(cli_error);
   
    //ERR_REGISTRATION(memory_allocation_error);
    //ERR_REGISTRATION(memory_not_allocated_error);
    //ERR_REGISTRATION(shape_mismatch_error);
    ERR_REGISTRATION(size_mismatch_error);
    //ERR_REGISTRATION(index_error);
    //ERR_REGISTRATION(key_error);
    ERR_REGISTRATION(file_error);
    //ERR_REGISTRATION(type_error);
    //ERR_REGISTRATION(value_error);
    ERR_REGISTRATION(range_error);
    ERR_REGISTRATION(not_implemented_error);
    ERR_REGISTRATION(iterator_error);
    ERR_REGISTRATION(cli_argument_error);
    ERR_REGISTRATION(cli_error);

    register_exception_translator<memory_allocation_error>(memory_allocation_error_translator);
    register_exception_translator<memory_not_allocated_error>(memory_not_allocated_error_translator);
    register_exception_translator<index_error>(index_error_translator);
    register_exception_translator<key_error>(key_error_translator);
    register_exception_translator<type_error>(type_error_translator);
    register_exception_translator<value_error>(value_error_translator);
    register_exception_translator<shape_mismatch_error>(shape_mismatch_error_translator);

}
