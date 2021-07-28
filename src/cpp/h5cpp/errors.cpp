//
// (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
//
// This file is part of python-pninexus.
//
// python-pninexus is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pninexus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pninexus.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Definition and implementation of exception classes and translation functions.
//
// Created on: March 15, 2012
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

extern "C"{
#include<Python.h>
}
#include <pni/error.hpp>
#include <boost/python.hpp>
#include <pni/exceptions.hpp>

#include "errors.hpp"

using namespace boost::python;

//import here the namespace for the nxh5 module
using namespace pni;
using namespace pni;

static object PyExc_LinkError;
static object PyExc_ObjectError;
static object PyExc_ParserError;
static object PyExc_InvalidObjectError;


static char const *LinkError_Doc = 
"Raised in case of errors during link creation or access.";

static char const *ObjectError_Doc = 
"Raised in case of errors during object creation, movement, or copying.";

static char const *ParserError_Doc=
"Raised in case of a parser error during data input.";

static char const *InvalidObjectError_Doc=
"Raised when an instance tries to access an invalid NeXus object.";

//-----------------------------------------------------------------------------
void io_error_translator(io_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_IOError,stream.str().c_str());
}

//-----------------------------------------------------------------------------
void link_error_translator(link_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_LinkError.ptr(),stream.str().c_str());
}

//-----------------------------------------------------------------------------
void object_error_translator(object_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_ObjectError.ptr(),stream.str().c_str());
}

//-----------------------------------------------------------------------------
void parser_error_translator(parser_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_ParserError.ptr(),stream.str().c_str());
}

//-----------------------------------------------------------------------------
void invalid_object_error_translator(invalid_object_error const& error)
{
    std::stringstream stream;
    stream<<error<<std::endl;
    PyErr_SetString(PyExc_InvalidObjectError.ptr(),stream.str().c_str());
}


void stop_iteration_translator(StopIteration const &)
{
  PyErr_SetString(PyExc_StopIteration,"iteration stop");
}

void index_error_translator(IndexError const &)
{
  PyErr_SetString(PyExc_IndexError,"invalid index");
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
#if PY_MAJOR_VERSION >= 3
    return object(handle<>(PyErr_NewExceptionWithDoc(name,doc,
                                                     nullptr,nullptr)));
#else
    return object(handle<>(PyErr_NewExceptionWithDoc(const_cast<char*>(name),
                                                     const_cast<char*>(doc),
                                                     nullptr,nullptr)));
#endif
}


//-----------------------------------------------------------------------------
void exception_registration()
{
    PyExc_LinkError = new_exception("pni.io.LinkError",LinkError_Doc);
    scope().attr("LinkError") = PyExc_LinkError;

    PyExc_ObjectError = new_exception("pni.io.ObjectError",ObjectError_Doc);
    scope().attr("ObjectError") = PyExc_ObjectError;

    PyExc_ParserError = new_exception("pni.io.ParserError",ParserError_Doc);
    scope().attr("ParserError")=PyExc_ParserError;

    PyExc_InvalidObjectError = new_exception("pni.io.InvalidObjectError",
                                             InvalidObjectError_Doc);
    scope().attr("InvalidObjectError") = PyExc_InvalidObjectError;
                            

    register_exception_translator<io_error>(io_error_translator);
    register_exception_translator<link_error>(link_error_translator);
    register_exception_translator<parser_error>(parser_error_translator);
    register_exception_translator<invalid_object_error>(invalid_object_error_translator);
    register_exception_translator<object_error>(object_error_translator);

    register_exception_translator<StopIteration>(stop_iteration_translator);
    register_exception_translator<IndexError>(index_error_translator);

}
