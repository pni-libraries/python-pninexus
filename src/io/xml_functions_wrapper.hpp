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

#include <boost/python.hpp>
#include <pni/core/utilities.hpp>
#include <pni/io/nx/xml.hpp>
#include <pni/io/nx/algorithms/as_field.hpp>
#include <pni/io/nx/algorithms/as_group.hpp>
#include <pni/core/error.hpp>
#include "nxgroup_wrapper.hpp"
#include "nxfield_wrapper.hpp"

class predicate_wrapper
{
    private:
        boost::python::object _callable; 
    public:
        predicate_wrapper(const boost::python::object &callable):
            _callable(callable)
        {}

        template<typename OTYPE>
        bool operator()(const OTYPE &o) const
        {
            using namespace boost::python; 
            using namespace pni::core;

            object result = _callable(object(o));
            extract<bool> result_extractor(result);

            if(!result_extractor.check())
                throw type_error(EXCEPTION_RECORD,
                        "Predicate functions must return a boolean value!");
            
            return result_extractor();
        }
};

template<typename OTYPE> struct xml_functions_wrapper
{
    //need the 
    typedef decltype(pni::io::nx::as_field(OTYPE())) field_type;
    typedef decltype(pni::io::nx::as_group(OTYPE())) group_type;
    typedef nxgroup_wrapper<group_type> group_wrapper_type;
    typedef nxfield_wrapper<field_type> field_wrapper_type;

    static OTYPE get_nxobject(const boost::python::object &parent)
    {
        using namespace boost::python;
        using namespace pni::core;

        extract<group_wrapper_type> group_ex(parent);
        extract<field_wrapper_type> field_ex(parent);

        if(group_ex.check())
        {
            group_type g = group_ex();
            return OTYPE(g);
        }
        else if(field_ex.check())
        {
            field_type f = field_ex();
            return OTYPE(f);
        }
        else 
            throw type_error(EXCEPTION_RECORD,
                    "Python object is not a field or group!");
    }

    static void xml_to_nexus_no_pred(const pni::core::string &xml_data,
                                     const boost::python::object &parent)
    {
        using namespace pni::io::nx;
        OTYPE p(get_nxobject(parent));
        xml::xml_to_nexus(xml::create_from_string(xml_data),p);
    }

    static void xml_to_nexus_with_pred(const pni::core::string &xml_data,
                                       const boost::python::object &parent,
                                       const boost::python::object &pred)
    {
        using namespace pni::io::nx;

        predicate_wrapper write_predicate(pred);
        OTYPE p(get_nxobject(parent));
        xml::xml_to_nexus(xml::create_from_string(xml_data),p,write_predicate);

    }
    
};


//!
//! \ingroup wrappers
//! \brief create NXGroup wrapper
//! 
//! Template function to create a new wrapper for an NXGroup type GType.
//! \param class_name name for the Python class
//!
template<typename OTYPE> void create_xml_function_wrappers()
{
    using namespace boost::python;
    typedef xml_functions_wrapper<OTYPE> wrapper_type;

    def("xml_to_nexus",&wrapper_type::xml_to_nexus_no_pred);
    def("xml_to_nexus",&wrapper_type::xml_to_nexus_with_pred);

}

