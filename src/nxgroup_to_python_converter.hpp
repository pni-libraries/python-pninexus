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

#include "nxgroup_wrapper.hpp"

using namespace pni::core;
using namespace pni::io::nx;
using namespace boost::python;

//----------------------------------------------------------------------------
//!
//! \ingroup pnicore_converters
//! \brief convert nxgroup instances to their wrapper type
//! 
//! Convert an instance of nxgroup to its corresponding wrapper type.
//!
//! \tparam GTYPE group type
//! 
template< typename GTYPE > 
struct nxgroup_to_python_converter
{
    typedef GTYPE group_type;
    typedef nxgroup_wrapper<group_type> group_wrapper_type;

    //------------------------------------------------------------------------
    //!
    //! \brief conversion method
    //! 
    //! \param v instance of bool_t
    //! \return Python boolean object
    //!
    static PyObject *convert(const group_type &v)
    {
        return incref(object(group_wrapper_type(v)).ptr());
    }

};

