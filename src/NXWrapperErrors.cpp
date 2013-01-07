/*
 * (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
 * This file is part of python-pniio.
 *
 * python-pniio is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * python-pniio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************
 *
 * Definition and implementation of exception classes and translation functions.
 *
 * Created on: March 15, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

extern "C"{
#include<Python.h>
}
#include <pni/core/Exceptions.hpp>
#include <boost/python.hpp>
#include <pni/io/nx/NX.hpp>

#include "NXWrapperErrors.hpp"

using namespace pni::core;
using namespace boost::python;

//import here the namespace for the nxh5 module
using namespace pni::io::nx::h5;

ERR_TRANSLATOR(pni::io::nx,NXFileError);
ERR_TRANSLATOR(pni::io::nx,NXGroupError);
ERR_TRANSLATOR(pni::io::nx,NXFieldError);
ERR_TRANSLATOR(pni::io::nx,NXAttributeError);
ERR_TRANSLATOR(pni::io::nx,NXSelectionError);
ERR_TRANSLATOR(pni::io::nx,NXFilterError);

//-----------------------------------------------------------------------------
void ChildIteratorStop_translator(ChildIteratorStop const &error)
{
    PyErr_SetString(PyExc_StopIteration,"iteration stop");
}

//-----------------------------------------------------------------------------
void AttributeIteratorStop_translator(AttributeIteratorStop const &error)
{
    PyErr_SetString(PyExc_StopIteration,"iteration stop");
}


//====================General purpose exceptions===============================
ERR_TRANSLATOR(pni::core,ShapeMissmatchError);
ERR_TRANSLATOR(pni::core,IndexError);
ERR_TRANSLATOR(pni::core,SizeMissmatchError);
ERR_TRANSLATOR(pni::core,MemoryAllocationError);
ERR_TRANSLATOR(pni::core,TypeError);



//-----------------------------------------------------------------------------
void exception_registration()
{
    //define the base class for all exceptions
    const String &(Exception::*exception_get_description)() const =
        &Exception::description;
    class_<Exception>("Exception")
        .def(init<>())
        .add_property("description",make_function(exception_get_description,return_value_policy<copy_const_reference>()))
        .def(self_ns::str(self_ns::self))
        ;

    ERR_OBJECT_DECL(pni::io::nx,NXFileError);
    ERR_OBJECT_DECL(pni::io::nx,NXFieldError);
    ERR_OBJECT_DECL(pni::io::nx,NXGroupError);
    ERR_OBJECT_DECL(pni::io::nx,NXAttributeError);
    ERR_OBJECT_DECL(pni::io::nx,NXSelectionError);
    ERR_OBJECT_DECL(pni::io::nx,NXFilterError);
    ERR_OBJECT_DECL(pni::core,ShapeMissmatchError);
    ERR_OBJECT_DECL(pni::core,IndexError);
    ERR_OBJECT_DECL(pni::core,SizeMissmatchError);
    ERR_OBJECT_DECL(pni::core,MemoryAllocationError);
    ERR_OBJECT_DECL(pni::core,TypeError);

    
    ERR_REGISTRATION(pni::io::nx,NXFileError);
    ERR_REGISTRATION(pni::io::nx,NXFieldError);
    ERR_REGISTRATION(pni::io::nx,NXGroupError);
    ERR_REGISTRATION(pni::io::nx,NXAttributeError);
    ERR_REGISTRATION(pni::io::nx,NXSelectionError);
    ERR_REGISTRATION(pni::io::nx,NXFilterError);
    ERR_REGISTRATION(pni::core,ShapeMissmatchError);
    ERR_REGISTRATION(pni::core,IndexError);
    ERR_REGISTRATION(pni::core,SizeMissmatchError);
    ERR_REGISTRATION(pni::core,MemoryAllocationError);
    ERR_REGISTRATION(pni::core,TypeError);


    register_exception_translator<ChildIteratorStop>(ChildIteratorStop_translator);
    register_exception_translator<AttributeIteratorStop>(AttributeIteratorStop_translator);

}
