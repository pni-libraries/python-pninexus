//
// (c) Copyright 2014 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// Created on: May 12, 2014
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

extern "C"
{
#include<Python.h>
#include<numpy/arrayobject.h>
#include<numpy/arrayscalars.h>
}

#include <pni/core/types.hpp>
#include <boost/python.hpp>

using namespace pni::core;
using namespace boost::python;

//converter namespace
namespace convns = boost::python::converter; 

//----------------------------------------------------------------------------
//!
//! \brief converts numpy scalars to POD data
//!
//!
struct numpy_scalar_converter
{
    typedef convns::rvalue_from_python_stage1_data     rvalue_type;
    typedef convns::rvalue_from_python_storage<bool_t> storage_type;
    //!
    //! \brief constructor
    //! 
    //! Registers the converter for the boost::python runtime.
    //!
    numpy_scalar_converter();

    //-----------------------------------------------------------------------
    //!
    //! \brief check convertible 
    //! 
    //! Returns a nullptr if the python object passed via obj_ptr is not a
    //! Python numpy scalar. Otherwise the address to which obj_ptr referes 
    //! to will be returned as void*.
    //!
    //! \param obj_ptr pointer to the python object to check
    //! \return object address or nullptr
    //!
    static void* convertible(PyObject *obj_ptr);

    //------------------------------------------------------------------------
    //!
    //! \brief construct rvalue
    //!
    //! \param obj_ptr pointer to the original python object
    //! \param data pointer to the new rvalue
    //!
    template<typename T>
    static void construct(PyObject *obj_ptr,rvalue_type *data);
};

//-----------------------------------------------------------------------------
template<typename T>
void numpy_scalar_converter::construct(PyObject *obj_ptr,rvalue_type *data)
{
    void *storage = ((storage_type*)data)->storage.bytes;
    
    new (storage) T();
    PyArray_ScalarAsCtype(obj_ptr,storage);
    data->convertible = storage;
}
