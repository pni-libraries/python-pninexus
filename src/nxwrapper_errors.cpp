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
#include <boost/python.hpp>
#include <pni/io/exceptions.hpp>
#include <pni/io/nx/nx.hpp>

#include "nxwrapper_errors.hpp"

using namespace pni::core;
using namespace boost::python;

//import here the namespace for the nxh5 module
using namespace pni::io::nx::h5;

ERR_TRANSLATOR(pni::io,io_error)
ERR_TRANSLATOR(pni::io,link_error)
ERR_TRANSLATOR(pni::io,object_error)
ERR_TRANSLATOR(pni::io,parser_error)
ERR_TRANSLATOR(pni::io,invalid_object_error)

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


//====================General purpose exceptions===============================
ERR_TRANSLATOR(pni::core,file_error)
ERR_TRANSLATOR(pni::core,shape_mismatch_error)
ERR_TRANSLATOR(pni::core,index_error)
ERR_TRANSLATOR(pni::core,size_mismatch_error)
ERR_TRANSLATOR(pni::core,memory_not_allocated_error)
ERR_TRANSLATOR(pni::core,memory_allocation_error)
ERR_TRANSLATOR(pni::core,type_error)
ERR_TRANSLATOR(pni::core,key_error)



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

    ERR_OBJECT_DECL(pni::io,io_error);
    ERR_OBJECT_DECL(pni::io,link_error);
    ERR_OBJECT_DECL(pni::io,parser_error);
    ERR_OBJECT_DECL(pni::io,invalid_object_error);
    ERR_OBJECT_DECL(pni::io,object_error);
    ERR_OBJECT_DECL(pni::core,file_error);
    ERR_OBJECT_DECL(pni::core,shape_mismatch_error);
    ERR_OBJECT_DECL(pni::core,index_error);
    ERR_OBJECT_DECL(pni::core,size_mismatch_error);
    ERR_OBJECT_DECL(pni::core,memory_allocation_error);
    ERR_OBJECT_DECL(pni::core,memory_not_allocated_error);
    ERR_OBJECT_DECL(pni::core,type_error);
    ERR_OBJECT_DECL(pni::core,key_error);

   
    ERR_REGISTRATION(pni::io,io_error);
    ERR_REGISTRATION(pni::io,link_error);
    ERR_REGISTRATION(pni::io,parser_error);
    ERR_REGISTRATION(pni::io,invalid_object_error);
    ERR_REGISTRATION(pni::io,object_error);
    ERR_REGISTRATION(pni::core,file_error);
    ERR_REGISTRATION(pni::core,shape_mismatch_error);
    ERR_REGISTRATION(pni::core,index_error);
    ERR_REGISTRATION(pni::core,size_mismatch_error);
    ERR_REGISTRATION(pni::core,memory_allocation_error);
    ERR_REGISTRATION(pni::core,memory_not_allocated_error);
    ERR_REGISTRATION(pni::core,type_error);
    ERR_REGISTRATION(pni::core,key_error);


    register_exception_translator<ChildIteratorStop>(ChildIteratorStop_translator);
    register_exception_translator<AttributeIteratorStop>(AttributeIteratorStop_translator);

}
