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
#include <pni/io/nx/algorithms/get_name.hpp>
#include <pni/io/nx/algorithms/get_rank.hpp>
#include <pni/io/nx/algorithms/get_unit.hpp>
#include <pni/io/nx/algorithms/get_class.hpp>
#include <pni/io/nx/algorithms/get_object.hpp>
#include <pni/io/nx/algorithms/is_field.hpp>
#include <pni/io/nx/algorithms/is_group.hpp>
#include <pni/io/nx/algorithms/is_attribute.hpp>
#include <pni/io/nx/algorithms/as_field.hpp>
#include <pni/io/nx/algorithms/as_group.hpp>
#include <pni/io/nx/algorithms/as_attribute.hpp>
#include <pni/io/nx/algorithms/set_unit.hpp>
#include <pni/io/nx/algorithms/set_class.hpp>
#include <pni/io/nx/algorithms/get_path.hpp>

#include <pni/io/nx/nximp_code_map.hpp>
#include <pni/io/nx/nxobject_traits.hpp>
#include <pni/io/nx/nxpath/nxpath.hpp>
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
    typedef typename pni::io::nx::nxobject_trait<
            pni::io::nx::nximp_code_map<GTYPE>::icode
            >::object_type object_type;

    typedef nxfield_wrapper<FTYPE> field_wrapper_type;
    typedef nxgroup_wrapper<GTYPE> group_wrapper_type;
    typedef nxattribute_wrapper<ATYPE> attribute_wrapper_type;

    static object_type to_object(const boost::python::object &o)
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

        return nxo;
    }


    //------------------------------------------------------------------------
    static size_t get_size (const boost::python::object &o)
    {
        return pni::io::nx::get_size(to_object(o));
    }

    //------------------------------------------------------------------------
    static pni::core::string get_name(const boost::python::object &o)
    {
        return pni::io::nx::get_name(to_object(o));
    }

    //------------------------------------------------------------------------
    static size_t get_rank(const boost::python::object &o)
    {
        return pni::io::nx::get_rank(to_object(o));
    }

    //------------------------------------------------------------------------
    static pni::core::string get_unit(const boost::python::object &o)
    {
        return pni::io::nx::get_unit(to_object(o));
    }
    
    //------------------------------------------------------------------------
    static pni::core::string get_class(const boost::python::object &o)
    {
        return pni::io::nx::get_class(to_object(o));
    }

    //------------------------------------------------------------------------
    static object_type get_object_nxpath(const boost::python::object &p,
                                         const pni::io::nx::nxpath &path)
    {
        return  pni::io::nx::get_object(to_object(p),path);
    }

    //------------------------------------------------------------------------
    static object_type get_object_string(const boost::python::object &p,
                                         const pni::core::string &path)
    {
        return pni::io::nx::get_object(to_object(p),path);
    }

    //------------------------------------------------------------------------
    static void set_class(const boost::python::object &o,
                          const pni::core::string &c)
    {
        pni::io::nx::set_class(to_object(o),c);
    }

    //------------------------------------------------------------------------
    static void set_unit(const boost::python::object &o,
                         const pni::core::string &u)
    {
        pni::io::nx::set_unit(to_object(o),u);
    }

    //------------------------------------------------------------------------
    static pni::core::string get_path(const boost::python::object &o)
    {
        using namespace pni::io;
        return nx::get_path(to_object(o));
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
    def("get_rank",&wrapper_type::get_rank);
    def("get_object",&wrapper_type::get_object_nxpath);
    def("get_object",&wrapper_type::get_object_string);
    def("get_unit",&wrapper_type::get_unit);
    def("get_class",&wrapper_type::get_class);
    def("set_class",&wrapper_type::set_class);
    def("set_unit",&wrapper_type::set_unit);
    def("get_path",&wrapper_type::get_path);

}

