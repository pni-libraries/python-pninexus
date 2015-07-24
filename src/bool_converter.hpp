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
#pragma once

extern "C"{
#include <Python.h>
}

#include <pni/core/types.hpp>
#include <boost/python.hpp>

//----------------------------------------------------------------------------
//!
//! \ingroup pnicore_converters
//! \brief convert bool_t to a python object
//! 
//! A converter structure to convert a value of bool_t from the C++ domain
//! to a Python bool object in the Python domain.
//!
struct bool_t_to_python_converter
{
    //! 
    //! \brief constructor
    //! 
    //! When a converter is instantiated the converter is registered. 
    //!
    bool_t_to_python_converter();

    //------------------------------------------------------------------------
    //!
    //! \brief conversion method
    //! 
    //! \param v instance of bool_t
    //! \return Python boolean object
    //!
    static PyObject *convert(const pni::core::bool_t &v);

};

//----------------------------------------------------------------------------
//!
//! \ingroup pnicore_converters
//! \brief convert Python bool object to bool_t
//!
//! Converts a Python boolean object to bool_t.  This is used in cases where 
//! the rvalue of an assignment is a python object of a boolean type. 
//!
struct python_to_bool_t_converter
{
    typedef boost::python::converter::rvalue_from_python_stage1_data     rvalue_type;
    typedef boost::python::converter::rvalue_from_python_storage<pni::core::bool_t> storage_type;
    //!
    //! \brief constructor
    //! 
    //! Registers the converter for the boost::python runtime.
    //!
    python_to_bool_t_converter();

    //-----------------------------------------------------------------------
    //!
    //! \brief check convertible 
    //! 
    //! Returns a nullptr if the python object passed via obj_ptr is not a
    //! Python boolean. Otherwise the address to which obj_ptr referes 
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
    static void construct(PyObject *obj_ptr,rvalue_type *data);
};

