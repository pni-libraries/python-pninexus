//
// (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// Definition and implementation of exception classes and translation functions.
//
// Created on: March 15, 2012
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

extern "C"{
#include<Python.h>
}
#include <pni/core/error.hpp>
#include <pni/core/python/error_utils.hpp>
#include <boost/python.hpp>
#include <pni/io/exceptions.hpp>
#include <pni/io/nx/nx.hpp>

#include "errors.hpp"

using namespace boost::python;

//import here the namespace for the nxh5 module
using namespace pni::io;
using namespace pni::core;

ERR_TRANSLATOR(io_error)
ERR_TRANSLATOR(link_error)
ERR_TRANSLATOR(object_error)
ERR_TRANSLATOR(parser_error)
ERR_TRANSLATOR(invalid_object_error)

//-----------------------------------------------------------------------------
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
void ChildIteratorStop_translator(ChildIteratorStop const &error)
{
    PyErr_SetString(PyExc_StopIteration,"iteration stop");
}
#pragma GCC diagnostic pop

//-----------------------------------------------------------------------------
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
void AttributeIteratorStop_translator(AttributeIteratorStop const &error)
{
    PyErr_SetString(PyExc_StopIteration,"iteration stop");
}
#pragma GCC diagnostic pop


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

    ERR_OBJECT_DECL(io_error);
    ERR_OBJECT_DECL(link_error);
    ERR_OBJECT_DECL(parser_error);
    ERR_OBJECT_DECL(invalid_object_error);
    ERR_OBJECT_DECL(object_error);
   
    ERR_REGISTRATION(io_error);
    ERR_REGISTRATION(link_error);
    ERR_REGISTRATION(parser_error);
    ERR_REGISTRATION(invalid_object_error);
    ERR_REGISTRATION(object_error);


    register_exception_translator<ChildIteratorStop>(ChildIteratorStop_translator);
    register_exception_translator<AttributeIteratorStop>(AttributeIteratorStop_translator);

}
