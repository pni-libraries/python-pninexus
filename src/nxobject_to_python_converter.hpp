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
// Created on: Oct 30, 2014
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <pni/core/types.hpp>
#include <boost/python.hpp>

#include <pni/io/nx/algorithms/is_group.hpp>
#include <pni/io/nx/algorithms/is_field.hpp>
#include <pni/io/nx/algorithms/is_attribute.hpp>
#include <pni/io/nx/algorithms/as_group.hpp>
#include <pni/io/nx/algorithms/as_field.hpp>
#include <pni/io/nx/algorithms/as_attribute.hpp>

#include "nxfield_wrapper.hpp"
#include "nxgroup_wrapper.hpp"
#include "nxattribute_wrapper.hpp"

using namespace pni::core;
using namespace pni::io::nx;
using namespace boost::python;

//converter namespace
namespace convns = boost::python::converter; 

//----------------------------------------------------------------------------
//!
//! \ingroup pnicore_converters
//! \brief convert nxobject to a python object
//! 
//! A converter structure to convert an instance of nxobject from the C++ 
//! domain to a Python object by instantiating the appropriate wrapper.
//!
//! \tparam OTYPE object type
//! \tparam GTYPE group type
//! \tparam FTYPE field type
//! \tparam ATYPE attribute type
//! 
template<
         typename OTYPE,
         typename GTYPE,
         typename FTYPE,
         typename ATYPE
        > 
struct nxobject_to_python_converter
{
    typedef OTYPE object_type;
    typedef GTYPE group_type;
    typedef FTYPE field_type;
    typedef ATYPE attribute_type;

    typedef nxfield_wrapper<field_type> field_wrapper_type;
    typedef nxgroup_wrapper<group_type> group_wrapper_type;
    typedef nxattribute_wrapper<attribute_type> attribute_wrapper_type;

    //------------------------------------------------------------------------
    //!
    //! \brief conversion method
    //! 
    //! \param v instance of bool_t
    //! \return Python boolean object
    //!
    static PyObject *convert(const OTYPE &v)
    {
        if(is_group(v))
            return incref(object(group_wrapper_type(as_group(v))).ptr());
        else if(is_field(v))
            return incref(object(field_wrapper_type(as_field(v))).ptr());
        else if(is_attribute(v))
            return incref(object(attribute_wrapper_type(as_attribute(v))).ptr());
        else 
            throw type_error(EXCEPTION_RECORD,
                             "Conversion failed - unknown object type!");
    }

};

