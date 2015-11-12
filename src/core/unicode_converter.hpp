//
// (c) Copyright 2015 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// Created on: Nov 11, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <boost/python.hpp>
#include <pni/core/types.hpp>


//----------------------------------------------------------------------------
//!
//! \ingroup converter_doc
//! \brief convert a Python unicode object to pni::core::string
//!
struct unicode_to_string_converter
{
    typedef boost::python::converter::rvalue_from_python_stage1_data     rvalue_type;
    typedef boost::python::converter::rvalue_from_python_storage<pni::core::string> storage_type;
    //!
    //! \brief constructor
    //! 
    //! Registers the converter at the boost::python runtime.
    //!
    unicode_to_string_converter();

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

