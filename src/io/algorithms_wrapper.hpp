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
#include "nxattribute_wrapper.hpp"
#include "nxgroup_wrapper.hpp"
#include "nxfield_wrapper.hpp"

template<
         typename GTYPE,
         typename FTYPE,
         typename ATYPE
        >
struct algorithms_wrapper
{
    typedef GTYPE group_type;
    typedef FTYPE field_type;
    typedef ATYPE attribute_type;
    typedef decltype(GTYPE::parent()) object_type;

    typedef nxfield_wrapper<FTYPE> field_wrapper_type;
    typedef nxgroup_wrapper<GTYPE> group_wrapper_type;
    typedef nxattribute_wrapper<ATYPE> attribute_wrapper_type;

    static size_t get_size (const boost::python::object &o)
    {
        using namespace boost::python; 

        extract<field_wrapper_type>     field_extractor(o);
        extract<attribute_wrapper_type> attribute_extractor(o);

        if(field_extractor.check()) 
        {
            field_type f = field_extractor();
            return pni::io::nx::get_size(f);
        }
        else if(attribute_extractor.check())
        {
            attribute_type a = attribute_extractor();
            return pni::io::nx::get_size(a);
        }
        else
            throw type_error(EXCEPTION_RECORD,
                    "Algorithm accepts only fields and groups!");
    }

    static pni::core::string get_name(const boost::python::object &o)
    {
        using namespace boost::python;
        
        extract<field_wrapper_type>  field_extractor(o);
        extract<group_wrapper_type>  group_extractor(o);
        extract<attribute_wrapper_type> attribute_extractor(o);
        object_type nxo;

        if(field_extractor.check()) nxo = field_extractor();
        else if(group_extractor.check()) nxo = group_extractor();
        else if(attribute_extractor.check()) nxo = attribute_extractor();
        else
            throw type_error(EXCEPTION_RECORD,
                    "Unkown NeXus object type!");

        return get_name(nxo);
            
    }
    
};


//!
//! \ingroup wrappers
//! \brief create NXGroup wrapper
//! 
//! Template function to create a new wrapper for an NXGroup type GType.
//! \param class_name name for the Python class
//!
template<
         typename GTYPE,
         typename FTYPE,
         typename ATYPE
        > 
void create_algorithms_wrappers()
{
    typedef algorithms_wrapper<GTYPE,FTYPE,ATYPE> wrapper_type;

    def("get_size",&wrapper_type::get_size);
    def("get_name",&wrapper_type::get_name);

}

