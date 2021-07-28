//
// (c) Copyright 2014 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
//
// This file is part of python-pninexus.
//
// python-pninexus is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pninexus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pninexus.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Oct 21, 2014
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <boost/python.hpp>
#include <pni/types.hpp>
#include <pni/nexus.hpp>

//----------------------------------------------------------------------------
//!
//! \ingroup converter_doc
//! \brief Converts an nxpath::element_type to a dictionary 
//! 
//! Converts a nxpath::element_type (which is basically a pair) to a
//! dictionary. This converter will be used whenever a function 
//! returns a value of type nxpath::element_type. 
//! An instance of nxpath::element_type is converted to a dictionary of 
//! form {"name":"pair.first","base_class":"pair.second"}.
//!
struct nxpath_element_to_dict_converter
{
    //! 
    //! \brief constructor
    //! 
    //! Constructs the converter and registers it at the boost::python 
    //! framework.
    //!
    nxpath_element_to_dict_converter();

    //------------------------------------------------------------------------
    //!
    //! \brief conversion method
    //! 
    //! Convert an instance of nxpath::element_type to a Python dictionary.
    //! 
    //! \param e instance of nxpath::element_type
    //! \return Python dictionary object
    //!
    static PyObject *convert(const pni::nexus::Path::Element &e);

};

//----------------------------------------------------------------------------
//!
//! \ingroup converter_doc
//! \brief convert Python dictionary to an nxpath::element_type instance
//!
//!
struct dict_to_nxpath_element_converter
{
    typedef pni::nexus::Path::Element element_type;
    typedef boost::python::converter::rvalue_from_python_stage1_data     rvalue_type;
    typedef boost::python::converter::rvalue_from_python_storage<element_type> storage_type;
    //!
    //! \brief constructor
    //! 
    //! Registers the converter at the boost::python runtime.
    //!
    dict_to_nxpath_element_converter();

    //-----------------------------------------------------------------------
    //!
    //! \brief check convertible 
    //! 
    //! Returns a nullptr if the python object passed via obj_ptr is not an
    //! appropriate Python dictionary. Otherwise the address to which obj_ptr
    //! referes to will be returned as void*.
    //!
    //! \param obj_ptr pointer to the python dict to check
    //! \return object address or nullptr
    //!
    static void* convertible(PyObject *obj_ptr);

    //------------------------------------------------------------------------
    //!
    //! \brief construct rvalue
    //!
    //! \param obj_ptr pointer to the original python dictionary
    //! \param data pointer to the new rvalue
    //!
    static void construct(PyObject *obj_ptr,rvalue_type *data);
};

