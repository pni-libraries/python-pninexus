//
// (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// along with pyton-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Feb 17, 2012
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#pragma once

#include <pni/core/utilities.hpp>
#include <pni/io/nx/algorithms/get_size.hpp>

template<typename OTYPE> struct algorithms_wrapper
{
    typedef OTYPE object_type; 

    static size_t get_size (const object &o)
    {
        object_type tmp(extract<object_type>(o));
        return pni::io::nx::get_size(tmp);
    }
    
};


//!
//! \ingroup wrappers
//! \brief create NXGroup wrapper
//! 
//! Template function to create a new wrapper for an NXGroup type GType.
//! \param class_name name for the Python class
//!
template<typename GTYPE> void create_algorithms_wrappers()
{
    typedef algorithms_wrapper<GTYPE> wrapper_type;

    def("get_size",&wrapper_type::get_size);

}

