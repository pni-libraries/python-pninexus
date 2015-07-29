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
#include <pni/core/types.hpp>
#include <pni/core/error.hpp>
#include <boost/python.hpp>

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

void not_implemented_error_translator(not_implemented_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_NotImplementedError,stream.str().c_str());
}


//====================General purpose exceptions===============================

static object ShapeMismatchError;
static object SizeMismatchError;
static object FileError;
static object RangeError;
static object IteratorError;
static object CliArgumentError;
static object CliError;

void shape_mismatch_error_translator(shape_mismatch_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(ShapeMismatchError.ptr(),stream.str().c_str());
}

void size_mismatch_error_translator(size_mismatch_error const &error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(SizeMismatchError.ptr(),stream.str().c_str());
}

void file_error_translator(file_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(FileError.ptr(),stream.str().c_str());
}

void range_error_translator(range_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(RangeError.ptr(),stream.str().c_str());
}

void iterator_error_translator(iterator_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(IteratorError.ptr(),stream.str().c_str());
}

void cli_argument_error_translator(cli_argument_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(CliArgumentError.ptr(),stream.str().c_str());
}

void cli_error_translator(cli_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(CliError.ptr(),stream.str().c_str());
}

static char *ShapeMismatchError_Doc=
"Wraps the `pni::core::shape_mismatch_error` C++ exception. This exception "
" is typically thrown iperations where two array or array-like objects are "
"involved which are supposed to have the same shape in order for the "
"operation to succeed.";

static char *SizeMismatchError_Doc=
"Wraps the `pni::core::size_mismatch_error` C++ exception. This exception"
" is typically thrown by operations wher two containers involved are "
"supposed to have the same size (number of elements) in order for the "
"operation to succeed.";

static char *FileError_Doc=
"Wraps the `pni::core::file_error` C++ exception. Thrown by operations from"
" libpnicore and its related libraries in situations where a file cannot"
" be opened or is in any other case faulty or access is denied.";

static char *RangeError_Doc=
"Wraps the `pni::core::range_error` C++ exception. Thrown in situations where"
" a numeric value must be within a particular range.";

static char *IteratorError_Doc=
"Wraps the `pni::core::iterator_error` C++ exception. Thrown by the iterators"
" provided by libpnicore in situations where something went wrong like an"
" element cannot be dereferenced. Do not confuse this with the Python "
" exception used to terminate iteration over a container!";

static char *CliArgumentError_Doc=
"Wraps the `pni::core::cli_argument_error` C++ exception. Thrown when a "
" command line argument has an invalid value or is ill formatted.";

static char *CliError_Doc=
"Wraps the `pni::core::cli_error` C++ exception. Thrown in case of a "
"general command line error.";

object new_exception(char *name,char *doc)
{
    return object(handle<>(PyErr_NewExceptionWithDoc(name,doc,nullptr,nullptr)));
}

//-----------------------------------------------------------------------------
void exception_registration()
{
    ShapeMismatchError = new_exception("pni.core.ShapeMismatchError",
                                       ShapeMismatchError_Doc);
    scope().attr("ShapeMismatchError") = ShapeMismatchError;

    SizeMismatchError = new_exception("pni.core.SizeMismatchError",
                                      SizeMismatchError_Doc);
    scope().attr("SizeMismatchError") = SizeMismatchError;
    
    FileError = new_exception("pni.core.FileError",FileError_Doc);
    scope().attr("FileError") = FileError;

    RangeError = new_exception("pni.core.RangeError",RangeError_Doc);
    scope().attr("RangeError") = RangeError;
    
    IteratorError = new_exception("pni.core.IteratorError",IteratorError_Doc);
    scope().attr("IteratorError") = IteratorError;
    
    CliArgumentError = new_exception("pni.core.CliArgumentError",
                                     CliArgumentError_Doc);
    scope().attr("CliArgumentError") = CliArgumentError;

    CliError = new_exception("pni.core.CliError",CliError_Doc);
    scope().attr("CliError") = CliError;

    register_exception_translator<memory_allocation_error>(memory_allocation_error_translator);
    register_exception_translator<memory_not_allocated_error>(memory_not_allocated_error_translator);
    register_exception_translator<index_error>(index_error_translator);
    register_exception_translator<key_error>(key_error_translator);
    register_exception_translator<type_error>(type_error_translator);
    register_exception_translator<value_error>(value_error_translator);
    register_exception_translator<shape_mismatch_error>(shape_mismatch_error_translator);
    register_exception_translator<size_mismatch_error>(size_mismatch_error_translator);
    register_exception_translator<file_error>(file_error_translator);
    register_exception_translator<range_error>(range_error_translator);
    register_exception_translator<not_implemented_error>(not_implemented_error_translator);
    register_exception_translator<iterator_error>(iterator_error_translator);
    register_exception_translator<cli_argument_error>(cli_argument_error_translator);
    register_exception_translator<cli_error>(cli_error_translator);

}
