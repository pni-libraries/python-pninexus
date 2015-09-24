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

#include "nxgroup_wrapper.hpp"
#include "nxfield_wrapper.hpp"


template<typename GTYPE,typename FTYPE>
struct link_wrapper
{
    typedef nxgroup_wrapper<GTYPE> group_wrapper_type;
    typedef nxfield_wrapper<FTYPE> field_wrapper_type;

    static void link_path(const pni::core::string &path,
                          const boost::python::object &g,
                          const pni::core::string &name)
    {
        using namespace boost::python;
        extract<group_wrapper_type> e(g);
        GTYPE group = e();
        pni::io::nx::link(path,group,name);
    }

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

template<typename GTYPE,typename FTYPE>
void wrap_link()
{
    using namespace boost::python;
    typedef link_wrapper<GTYPE,FTYPE> wrapper_type;

    def("link",&wrapper_type::link_path);
    def("link",&wrapper_type::link_object);
}



