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

#include <boost/python.hpp>

#include "nxfield_wrapper.hpp"

//----------------------------------------------------------------------------
//!
//! \ingroup pnicore_converters
//! \brief convert nxfield instances to their wrapper type
//! 
//! Converts instances of nxfield to their nxfield_wrapper counterpart 
//! on the python side. 
//! 
//! \tparam FTYPE field type
//! 
template< typename FTYPE > 
struct nxfield_to_python_converter
{
    typedef FTYPE field_type;
    typedef nxfield_wrapper<field_type> field_wrapper_type;

    //------------------------------------------------------------------------
    //!
    //! \brief conversion method
    //! 
    //! \param v instance of bool_t
    //! \return Python boolean object
    //!
    static PyObject *convert(const field_type &v)
    {
        using namespace boost::python;

        return incref(object(field_wrapper_type(v)).ptr());
    }

};

