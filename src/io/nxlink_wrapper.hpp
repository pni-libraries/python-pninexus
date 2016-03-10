//
// (c) Copyright 2015 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// Created on: Sep 23, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <boost/python.hpp>
#include <pni/core/types.hpp>
#include <pni/core/error.hpp>
#include <pni/io/nx/nxlink.hpp>
#include <pni/io/nx/link.hpp>

#include "nxgroup_wrapper.hpp"
#include "nxfield_wrapper.hpp"


template<typename GTYPE,typename FTYPE>
struct link_wrapper
{
    typedef nxgroup_wrapper<GTYPE> group_wrapper_type;
    typedef nxfield_wrapper<FTYPE> field_wrapper_type;


    static void link_object(const boost::python::object &obj,
                            const boost::python::object &g,
                            const pni::core::string &name)
    {
        using namespace boost::python;
        using namespace pni::core;

        extract<string> string_ex(obj);
        extract<group_wrapper_type> parent_ex(g),group_ex(obj);
        extract<field_wrapper_type> field_ex(obj);

        GTYPE parent = parent_ex();
        if(group_ex.check())
        {
            GTYPE object = group_ex();
            pni::io::nx::link(object,parent,name);
        }
        else if(field_ex.check())
        {
            FTYPE object = field_ex();
            pni::io::nx::link(object,parent,name);
        }
        else if(string_ex.check())
        {
            pni::io::nx::link(string_ex(),parent,name);
        }
        else
            throw type_error(EXCEPTION_RECORD,
                    "Target must be a field or a group!");
            
    }

};

template<nximp_code IMPID>
struct get_link_wrapper
{
    using link_type = typename nxobject_trait<IMPID>::link_type; 
    using group_type = typename nxobject_trait<IMPID>::group_type;
    using group_wrapper = nxgroup_wrapper<group_type>;
    using link_vector = std::vector<link_type>;

    static boost::python::list vector_to_list(const link_vector &v)
    {
        using namespace boost::python;
        list l;

        for(auto link: v) l.append(link);

        return l;
    }

    static boost::python::list get_links(const group_wrapper &parent)
    {
        //group_wrapper p_wrapped = extract<group_wrapper>(parent);
        auto links = pni::io::nx::get_links<link_vector>(static_cast<const
                group_type&>(parent));
        return vector_to_list(links);

    }

    static boost::python::list get_links_recursive(const group_wrapper &parent)
    {
        auto links =
            pni::io::nx::get_links_recursive<link_vector>(static_cast<const group_type &>(parent));
        return vector_to_list(links);
    }

};

template<typename GTYPE,typename FTYPE>
void wrap_link()
{
    using namespace boost::python;
    typedef link_wrapper<GTYPE,FTYPE> wrapper_type;

    def("link",&wrapper_type::link_object);
}

static const pni::core::string get_links_doc = 
"Return a list of links to the direct members of parent";

template<nximp_code IMPID> void wrap_nxlink()
{
    using namespace boost::python;
    using nxlink_class = nxlink<IMPID>; 
    using get_link_wrapper_type = get_link_wrapper<IMPID>;

    using namespace pni::io::nx;
    enum_<nxlink_status>("nxlink_status")
        .value("VALID",nxlink_status::VALID)
        .value("INVALID",nxlink_status::INVALID);

    enum_<nxlink_type>("nxlink_type")
        .value("HARD",nxlink_type::HARD)
        .value("SOFT",nxlink_type::SOFT)
        .value("EXTERNAL",nxlink_type::EXTERNAL)
        .value("ATTRIBUTE",nxlink_type::ATTRIBUTE);

    
    def("get_links",&get_link_wrapper_type::get_links,get_links_doc.c_str());
    def("get_links_recursive",&get_link_wrapper_type::get_links_recursive);
    

    class_<nxlink_class>("nxlink")
        .def(init<const nxlink_class &>())
        .add_property("filename",&nxlink_class::filename)
        .add_property("name",&nxlink_class::name)
        .add_property("target_path",&nxlink_class::target_path)
        .add_property("status",&nxlink_class::status)
        .add_property("is_valid",&nxlink_class::is_valid)
        .add_property("type",&nxlink_class::type)
        .add_property("parent",&nxlink_class::parent)
        .def("resolve",&nxlink_class::resolve);


}


