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

PyObject *PyNXFileErrorPtr = nullptr;
PyObject *PyNXGroupErrorPtr =  nullptr;
PyObject *PyNXFieldErrorPtr = nullptr;
PyObject *PyNXAttributeErrorPtr = nullptr;
PyObject *PyNXSelectionErrorPtr = nullptr;
PyObject *PyNXFilterErrorPtr = nullptr;


//===============exception translators=========================================
void NXFileError_translator(pni::nx::NXFileError const &error)
{
    assert(PyNXFileErrorPtr != nullptr);
    object exception(error);
    PyErr_SetObject(PyNXFileErrorPtr,exception.ptr());
}

//-----------------------------------------------------------------------------
void NXGroupError_translator(pni::nx::NXGroupError const &error)
{
    assert(PyNXGroupErrorPtr != nullptr);
    object exception(error);
    PyErr_SetObject(PyNXGroupErrorPtr,exception.ptr());
}

//-----------------------------------------------------------------------------
void NXAttributeError_translator(pni::nx::NXAttributeError const &error)
{
    assert(PyNXAttributeError != nullptr);
    object exception(error);
    PyErr_SetObject(PyNXAttributeErrorPtr,exception.ptr());
}

//-----------------------------------------------------------------------------
void NXFieldError_translator(pni::nx::NXFieldError const &error)
{
    assert(PyNXFieldErrorPtr != nullptr);
    object exception(error);
    PyErr_SetObject(PyNXFieldErrorPtr,exception.ptr());
}

//-----------------------------------------------------------------------------
void NXSelectionError_translator(pni::nx::NXSelectionError const &error)
{
    assert(PyNXSelectionErrorPtr != nullptr);
    object exception(error);
    PyErr_SetObject(PyNXSelectionErrorPtr,exception.ptr());
}

//-----------------------------------------------------------------------------
void NXFilterError_translator(pni::nx::NXFilterError const &error)
{
    assert(PyNXFilterErrorPtr != nullptr);
    object exception(error);
    PyErr_SetObject(PyNXFilterErrorPtr,exception.ptr());
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


//====================General purpose exceptions===============================
PyObject *PyShapeMissmatchErrorPtr = nullptr;
PyObject *PyIndexErrorPtr = nullptr;
PyObject *PySizeMissmatchErrorPtr = nullptr;
PyObject *PyMemoryAllocationErrorPtr = nullptr;

//-----------------------------------------------------------------------------
void ShapeMissmatchError_translator(pni::utils::ShapeMissmatchError const
        &error)
{
    assert(PyShapeMissmatchErrorPtr != nullptr);
    object exception(error);
    PyErr_SetObject(PyShapeMissmatchErrorPtr,exception.ptr());
}

//-----------------------------------------------------------------------------
void IndexError_translator(pni::utils::IndexError const &error)
{
    assert(PyIndexErrorPtr != nullptr);
    object exception(error);
    PyErr_SetObject(PyIndexErrorPtr,exception.ptr());
}

//-----------------------------------------------------------------------------
void SizeMissmatchError_translator(pni::utils::SizeMissmatchError const &error)
{
    assert(PySizeMissmatchErrorPtr != nullptr);
    object exception(error);
    PyErr_SetObject(PySizeMissmatchErrorPtr,exception.ptr());
}

//-----------------------------------------------------------------------------
void MemoryAllocationError_translator(pni::utils::MemoryAllocationError const
        &error)
{
    assert(PyMemoryAllocationErrorPtr != nullptr); object exception(error); 
    PyErr_SetObject(PyMemoryAllocationErrorPtr,exception.ptr()); 
}

//-----------------------------------------------------------------------------
void TypeError_translator(pni::utils::TypeError const &error)
{
    object exception(error);
    PyErr_SetObject(PyExc_TypeError,exception.ptr());
}



//-----------------------------------------------------------------------------
void exception_registration()
{

    const String &(Exception::*exception_get_description)() const =
        &Exception::description;
    class_<Exception>("Exception")
        .def(init<>())
        .add_property("description",make_function(exception_get_description,return_value_policy<copy_const_reference>()))
        ;

    object PyShapeMissmatchError = (
            class_<ShapeMissmatchError,bases<Exception> >("ShapeMissmatchError")
                .def(init<>())
            );

    object PyIndexError = (
            class_<IndexError,bases<Exception> >("IndexError")
                .def(init<>())
            );
    
    object PySizeMissmatchError = (
            class_<SizeMissmatchError,bases<Exception> >("SizeMissmatchError")
                .def(init<>())
            );
    
    object PyMemoryAllocationError = (
            class_<MemoryAllocationError,bases<Exception> >("MemoryAllocationError")
                .def(init<>())
            );
    
    class_<TypeError,bases<Exception> >("PNITypeError")
        .def(init<>())
        ;

    PyShapeMissmatchErrorPtr = PyShapeMissmatchError.ptr();
    PyIndexErrorPtr = PyIndexError.ptr();
    PySizeMissmatchErrorPtr = PySizeMissmatchError.ptr();
    PyMemoryAllocationErrorPtr = PyMemoryAllocationError.ptr();


    object PyNXFileError = (
            class_<pni::nx::NXFileError,bases<Exception> >("NXFileError")
                .def(init<>())
            );

    object PyNXGroupError = (
            class_<pni::nx::NXGroupError,bases<Exception> >("NXGroupError")
                .def(init<>())
            );

    object PyNXFilterError = (
            class_<pni::nx::NXFilterError,bases<Exception> >("NXFilterError")
                .def(init<>())
            );

    object PyNXFieldError = (
            class_<pni::nx::NXFieldError,bases<Exception> >("NXFieldError")
                .def(init<>())
            );
    
    object PyNXAttributeError = (
            class_<pni::nx::NXAttributeError,bases<Exception> >("NXAttributeError")
                .def(init<>())
            );
    
    object PyNXSelectionError = (
            class_<pni::nx::NXSelectionError,bases<Exception> >("NXSelectionError")
                .def(init<>())
            );

    PyNXFileErrorPtr        = PyNXFileError.ptr();
    PyNXGroupErrorPtr       = PyNXGroupError.ptr();
    PyNXFieldErrorPtr       = PyNXFieldError.ptr();
    PyNXAttributeErrorPtr   = PyNXAttributeError.ptr();
    PyNXSelectionErrorPtr   = PyNXSelectionError.ptr();
    PyNXFilterErrorPtr      = PyNXFilterError.ptr();



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
    register_exception_translator<pni::utils::MemoryAllocationError>
        (MemoryAllocationError_translator);
    register_exception_translator<pni::utils::SizeMissmatchError>
        (SizeMissmatchError_translator);
    register_exception_translator<pni::nx::NXSelectionError>
        (NXSelectionError_translator);
    register_exception_translator<pni::utils::ShapeMissmatchError>
        (ShapeMissmatchError_translator);
    register_exception_translator<pni::utils::TypeError>(TypeError_translator);
    register_exception_translator<ChildIteratorStop>(ChildIteratorStop_translator);
    register_exception_translator<AttributeIteratorStop>(AttributeIteratorStop_translator);

}
