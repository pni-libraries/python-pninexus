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

//=============================================================================
// Static variables holding the custom exceptions provided by this module
//=============================================================================

//!
//! \ingroup error_management
//! \brief static ShapeMismatchError instance
static object ShapeMismatchError;
//! \ingroup error_management
//! \brief static SizeMismatchError instance in error_management
static object SizeMismatchError;
//! \ingroup error_management
//! \brief static FileError instance
static object FileError;
//! \ingroup error_management
//! \brief static RangeError instance
static object RangeError;
//! \ingroup error_management
//! \brief static IteratorError instance 
static object IteratorError;
//! \ingroup error_management
//! \brief static CliArgumentError instance
static object CliArgumentError;
//! \ingroup error_management
//! \brief static CliError instance
static object CliError;

//! \ingroup error_management
//! \brief documentation string for `pni.core.ShapeMismatchError`
static char const *ShapeMismatchError_Doc=
"Wraps the `pni::core::shape_mismatch_error` C++ exception. This exception "
" is typically thrown iperations where two array or array-like objects are "
"involved which are supposed to have the same shape in order for the "
"operation to succeed.";

//! \ingroup error_management
//! \brief documentation string for `pni.core.SizeMismatchError`
static char const *SizeMismatchError_Doc=
"Wraps the `pni::core::size_mismatch_error` C++ exception. This exception"
" is typically thrown by operations wher two containers involved are "
"supposed to have the same size (number of elements) in order for the "
"operation to succeed.";

//! \ingroup error_management
//! \brief documentation string for `pni.core.FileError`
static char const *FileError_Doc=
"Wraps the `pni::core::file_error` C++ exception. Thrown by operations from"
" libpnicore and its related libraries in situations where a file cannot"
" be opened or is in any other case faulty or access is denied.";

//! \ingroup error_management
//! \brief documentation string for `pni.core.RangeError`
static char const *RangeError_Doc=
"Wraps the `pni::core::range_error` C++ exception. Thrown in situations where"
" a numeric value must be within a particular range.";

//! \ingroup error_management
//! \brief documentation string for `pni.core.IteratorError`
static char const *IteratorError_Doc=
"Wraps the `pni::core::iterator_error` C++ exception. Thrown by the iterators"
" provided by libpnicore in situations where something went wrong like an"
" element cannot be dereferenced. Do not confuse this with the Python "
" exception used to terminate iteration over a container!";

//! \ingroup error_management
//! \brief documentation string for `pni.core.CliArgumentError`
static char const *CliArgumentError_Doc=
"Wraps the `pni::core::cli_argument_error` C++ exception. Thrown when a "
" command line argument has an invalid value or is ill formatted.";

//! \ingroup error_management
//! \brief documentation string for `pni.core.CliError`
static char const *CliError_Doc=
"Wraps the `pni::core::cli_error` C++ exception. Thrown in case of a "
"general command line error.";


//-----------------------------------------------------------------------------
//! 
//! \ingroup error_management
//! \brief translates `memory_allocation_error` to `MemoryError`
//! 
void memory_allocation_error_translator(memory_allocation_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_MemoryError,stream.str().c_str());
}

//-----------------------------------------------------------------------------
//!
//! \ingroup error_management
//! \brief translates `memory_not_allocated_error` to `MemoryError`
void memory_not_allocated_error_translator(memory_not_allocated_error const &error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_MemoryError,stream.str().c_str());
}

//-----------------------------------------------------------------------------
//!
//! \ingroup error_management
//! \brief translates `index_error` to `IndexError`
void index_error_translator(index_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_IndexError,stream.str().c_str());
}

//-----------------------------------------------------------------------------
//! 
//! \ingroup error_management
//! \brief translates `key_error` to `KeyError`
void key_error_translator(key_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_KeyError,stream.str().c_str());
}

//-----------------------------------------------------------------------------
//! 
//! \ingroup error_management
//! \brief translates `type_error` to `TypeError`
void type_error_translator(type_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_TypeError,stream.str().c_str());
}

//-----------------------------------------------------------------------------
//! 
//! \ingroup error_management
//! \brief translates `value_error` to `ValueError`
void value_error_translator(value_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_ValueError,stream.str().c_str());
}

//-----------------------------------------------------------------------------
//! 
//! \ingroup error_management
//! \brief translates `not_implemented_error` to `NotImplementedError`
void not_implemented_error_translator(not_implemented_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_NotImplementedError,stream.str().c_str());
}



//------------------------------------------------------------------------------
//!
//! \ingroup error_management
//! \brief translates `shape_mismatch_error` to `pni.core.ShapeMismatchError`
void shape_mismatch_error_translator(shape_mismatch_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(ShapeMismatchError.ptr(),stream.str().c_str());
}

//------------------------------------------------------------------------------
//! 
//! \ingroup error_management
//! \brief translates `size_mismatch_error` to `pni.core.SizeMismatchError`
void size_mismatch_error_translator(size_mismatch_error const &error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(SizeMismatchError.ptr(),stream.str().c_str());
}

//-----------------------------------------------------------------------------
//!
//! \ingroup error_management
//! \brief translates `file_error` to `pni.core.FileError`
void file_error_translator(file_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(FileError.ptr(),stream.str().c_str());
}

//-----------------------------------------------------------------------------
//!
//! \ingroup error_managment
//! \brief translates `range_error` to `pni.core.RangeError`
void range_error_translator(range_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(RangeError.ptr(),stream.str().c_str());
}

//-----------------------------------------------------------------------------
//!
//! \ingroup error_management
//! \brief translates `iterator_error` to `pni.core.IteratorError`
void iterator_error_translator(iterator_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(IteratorError.ptr(),stream.str().c_str());
}

//-----------------------------------------------------------------------------
//!
//! \ingroup error_management
//! \brief translates `cli_argument_error` to `pni.core.CliArgumentError`
void cli_argument_error_translator(cli_argument_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(CliArgumentError.ptr(),stream.str().c_str());
}

//-----------------------------------------------------------------------------
//! 
//! \ingroup error_management
//! \brief translates `cli_error` to `pni.core.CliError`
void cli_error_translator(cli_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(CliError.ptr(),stream.str().c_str());
}


//!
//! \ingroup error_management
//! \brief exception creation utility function
//! 
//! This utility function is used inside exception_registration() to create
//! the new exceptions. It was written just to shorten the code within 
//! exception_registration() and does nothing special. 
//!
//! \param name the name of the new exception
//! \param doc the doc string of the new exception
//! \return boost::python::object for the new exception
//!
object new_exception(char const *name,char const *doc)
{
    return object(handle<>(PyErr_NewExceptionWithDoc(const_cast<char*>(name),
                                                     const_cast<char*>(doc),
                                                     nullptr,nullptr)));
}

//-----------------------------------------------------------------------------
//!
//! \ingroup error_management
//! \brief register all translators create exceptions
//!
//! This function is called by `pni.core`s initialization function and 
//! creates all new exceptions and registers the translator functions. 
//! 
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
