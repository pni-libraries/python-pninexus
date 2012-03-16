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

//===============exception translators=========================================
void NXFileError_translator(pni::nx::NXFileError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_IOError,estr.str().c_str());
}

//-----------------------------------------------------------------------------
void NXGroupError_translator(pni::nx::NXGroupError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_UserWarning,estr.str().c_str());
}

//-----------------------------------------------------------------------------
void NXAttributeError_translator(pni::nx::NXAttributeError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_UserWarning,estr.str().c_str());
}

//-----------------------------------------------------------------------------
void NXFieldError_translator(pni::nx::NXFieldError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_UserWarning,estr.str().c_str());
}

//-----------------------------------------------------------------------------
void NXSelectionError_translator(pni::nx::NXSelectionError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_UserWarning,estr.str().c_str());
}

//-----------------------------------------------------------------------------
void NXFilterError_translator(pni::nx::NXFilterError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_TypeError,error.description().c_str());
}

//-----------------------------------------------------------------------------
void IndexError_translator(pni::utils::IndexError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_IndexError,estr.str().c_str());
}

//-----------------------------------------------------------------------------
void MemoryAccessError_translator(pni::utils::MemoryAccessError const &error)
{
    std::stringstream estr;
    estr << error;
    std::cerr<<error<<std::endl;
    PyErr_SetString(PyExc_MemoryError,"from me");
}

//-----------------------------------------------------------------------------
void MemoryAllocationError_translator(pni::utils::MemoryAllocationError const
        &error)
{
    std::stringstream estr;
    estr<<error;
    std::cerr<<error<<std::endl;
    PyErr_SetString(PyExc_MemoryError,"from me");
}

//-----------------------------------------------------------------------------
void SizeMissmatchError_translator(pni::utils::SizeMissmatchError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_IndexError,estr.str().c_str());
}


//-----------------------------------------------------------------------------
void TypeError_translator(pni::utils::TypeError const &error)
{
    std::stringstream estr;
    estr<<error;
    PyErr_SetString(PyExc_TypeError,estr.str().c_str());
}

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

PyObject *PyShapeMissmatchErrorPtr = nullptr;
PyObject &PyIndexErrorPtr = nullptr;

//-----------------------------------------------------------------------------
void ShapeMissmatchError_translator(pni::utils::ShapeMissmatchError const
        &error)
{
    assert(PyShapeMissmatchErrorPtr != nullptr);
    object exception(error);
    PyErr_SetObject(PyShapeMissmatchErrorPtr,exception.ptr());
}


//-----------------------------------------------------------------------------
void exception_registration(){

    const String &(Exception::*exception_get_issuer)() const = &Exception::issuer;
    const String &(Exception::*exception_get_description)() const =
        &Exception::description;
    class_<Exception>("Exception")
        .def(init<>())
        .add_property("issuer",make_function(exception_get_issuer,return_internal_reference<1>()))
        .add_property("description",make_function(exception_get_description,return_internal_reference<1>()))
        ;

    object PyShapeMissmatchError = (
            class_<ShapeMissmatchError,bases<Exception> >("ShapeMissmatchError")
                .def(init<>())
            );

    object PyIndexError = (
            class_<IndexError,bases<Exception> >("IndexError")
                .def(init<>())
            );

    PyShapeMissmatchErrorPtr = PyShapeMissmatchError.ptr();



    register_exception_translator<pni::nx::NXFileError>
        (NXFileError_translator);
    register_exception_translator<pni::nx::NXGroupError>
        (NXGroupError_translator);
    register_exception_translator<pni::nx::NXAttributeError>
        (NXAttributeError_translator);
    register_exception_translator<pni::nx::NXFieldError>
        (NXFieldError_translator);
    register_exception_translator<pni::utils::IndexError>
        (IndexError_translator);
    register_exception_translator<pni::utils::MemoryAccessError>
        (MemoryAccessError_translator);
    register_exception_translator<pni::utils::MemoryAllocationError>
        (MemoryAllocationError_translator);
    register_exception_translator<pni::utils::SizeMissmatchError>
        (SizeMissmatchError_translator);
    register_exception_translator<pni::nx::NXSelectionError>
        (NXSelectionError_translator);
    register_exception_translator<pni::utils::ShapeMissmatchError>
        (ShapeMissmatchError_translator);
    register_exception_translator<ChildIteratorStop>(ChildIteratorStop_translator);
    register_exception_translator<AttributeIteratorStop>(AttributeIteratorStop_translator);

}
