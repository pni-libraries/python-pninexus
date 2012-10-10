/*
 * (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
 * This file is part of libpninx-python.
 *
 * libpninx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * libpninx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libpninx.  If not, see <http://www.gnu.org/licenses/>.
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
#include <pni/utils/Exceptions.hpp>
#include <boost/python.hpp>
#include <pni/nx/NX.hpp>

#include "NXWrapperErrors.hpp"

using namespace pni::utils;
using namespace boost::python;

//import here the namespace for the nxh5 module
using namespace pni::nx::h5;

ERR_TRANSLATOR(pni::nx,NXFileError);
ERR_TRANSLATOR(pni::nx,NXGroupError);
ERR_TRANSLATOR(pni::nx,NXFieldError);
ERR_TRANSLATOR(pni::nx,NXAttributeError);
ERR_TRANSLATOR(pni::nx,NXSelectionError);
ERR_TRANSLATOR(pni::nx,NXFilterError);

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
ERR_TRANSLATOR(pni::utils,ShapeMissmatchError);
ERR_TRANSLATOR(pni::utils,IndexError);
ERR_TRANSLATOR(pni::utils,SizeMissmatchError);
ERR_TRANSLATOR(pni::utils,MemoryAllocationError);
ERR_TRANSLATOR(pni::utils,TypeError);



//-----------------------------------------------------------------------------
void exception_registration()
{
    //define the base class for all exceptions
    const String &(Exception::*exception_get_description)() const =
        &Exception::description;
    class_<Exception>("Exception")
        .def(init<>())
        .add_property("description",make_function(exception_get_description,return_value_policy<copy_const_reference>()))
        ;

    ERR_OBJECT_DECL(pni::nx,NXFileError);
    ERR_OBJECT_DECL(pni::nx,NXFieldError);
    ERR_OBJECT_DECL(pni::nx,NXGroupError);
    ERR_OBJECT_DECL(pni::nx,NXAttributeError);
    ERR_OBJECT_DECL(pni::nx,NXSelectionError);
    ERR_OBJECT_DECL(pni::nx,NXFilterError);
    ERR_OBJECT_DECL(pni::utils,ShapeMissmatchError);
    ERR_OBJECT_DECL(pni::utils,IndexError);
    ERR_OBJECT_DECL(pni::utils,SizeMissmatchError);
    ERR_OBJECT_DECL(pni::utils,MemoryAllocationError);
    ERR_OBJECT_DECL(pni::utils,TypeError);

    
    ERR_REGISTRATION(pni::nx,NXFileError);
    ERR_REGISTRATION(pni::nx,NXFieldError);
    ERR_REGISTRATION(pni::nx,NXGroupError);
    ERR_REGISTRATION(pni::nx,NXAttributeError);
    ERR_REGISTRATION(pni::nx,NXSelectionError);
    ERR_REGISTRATION(pni::nx,NXFilterError);
    ERR_REGISTRATION(pni::utils,ShapeMissmatchError);
    ERR_REGISTRATION(pni::utils,IndexError);
    ERR_REGISTRATION(pni::utils,SizeMissmatchError);
    ERR_REGISTRATION(pni::utils,MemoryAllocationError);
    ERR_REGISTRATION(pni::utils,TypeError);


    register_exception_translator<ChildIteratorStop>(ChildIteratorStop_translator);
    register_exception_translator<AttributeIteratorStop>(AttributeIteratorStop_translator);

}
