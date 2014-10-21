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

#include <pni/core/error.hpp>
#include <boost/python.hpp>

#include "error_utils.hpp"

using namespace pni::core;
using namespace boost::python;

//====================General purpose exceptions===============================
ERR_TRANSLATOR(memory_allocation_error)
ERR_TRANSLATOR(memory_not_allocated_error)
ERR_TRANSLATOR(shape_mismatch_error)
ERR_TRANSLATOR(size_mismatch_error)
ERR_TRANSLATOR(index_error)
ERR_TRANSLATOR(key_error)
ERR_TRANSLATOR(file_error)
ERR_TRANSLATOR(type_error)
ERR_TRANSLATOR(value_error)
ERR_TRANSLATOR(range_error)
ERR_TRANSLATOR(not_implemented_error)
ERR_TRANSLATOR(iterator_error)
ERR_TRANSLATOR(cli_argument_error)
ERR_TRANSLATOR(cli_error)


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

    ERR_OBJECT_DECL(memory_allocation_error);
    ERR_OBJECT_DECL(memory_not_allocated_error);
    ERR_OBJECT_DECL(shape_mismatch_error);
    ERR_OBJECT_DECL(size_mismatch_error);
    ERR_OBJECT_DECL(index_error);
    ERR_OBJECT_DECL(key_error);
    ERR_OBJECT_DECL(file_error);
    ERR_OBJECT_DECL(type_error);
    ERR_OBJECT_DECL(value_error);
    ERR_OBJECT_DECL(range_error);
    ERR_OBJECT_DECL(not_implemented_error);
    ERR_OBJECT_DECL(iterator_error);
    ERR_OBJECT_DECL(cli_argument_error);
    ERR_OBJECT_DECL(cli_error);
   
    ERR_REGISTRATION(memory_allocation_error);
    ERR_REGISTRATION(memory_not_allocated_error);
    ERR_REGISTRATION(shape_mismatch_error);
    ERR_REGISTRATION(size_mismatch_error);
    ERR_REGISTRATION(index_error);
    ERR_REGISTRATION(key_error);
    ERR_REGISTRATION(file_error);
    ERR_REGISTRATION(type_error);
    ERR_REGISTRATION(value_error);
    ERR_REGISTRATION(range_error);
    ERR_REGISTRATION(not_implemented_error);
    ERR_REGISTRATION(iterator_error);
    ERR_REGISTRATION(cli_argument_error);
    ERR_REGISTRATION(cli_error);


}
