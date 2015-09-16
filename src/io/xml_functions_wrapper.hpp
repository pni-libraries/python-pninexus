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
#include <pni/io/nx/xml.hpp>

template<typename GTYPE> struct xml_functions_wrapper
{
    typedef GTYPE group_type; 

    static void xml_to_nexus(const pni::core::string &xml_data,
                             const group_type &parent)
    {
        using namespace pni::io::nx;
        xml::xml_to_nexus(xml::create_from_string(xml_data),parent);
    }
    
};


//!
//! \ingroup wrappers
//! \brief create NXGroup wrapper
//! 
//! Template function to create a new wrapper for an NXGroup type GType.
//! \param class_name name for the Python class
//!
template<typename GTYPE> void create_xml_function_wrappers()
{
    typedef xml_functions_wrapper<GTYPE> wrapper_type;

    def("xml_to_nexus",&wrapper_type::xml_to_nexus);

#pragma GCC diagnostic pop
}

