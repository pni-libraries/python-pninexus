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
            throw pni::core::type_error(EXCEPTION_RECORD,
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
        return pni::io::nx::get_path(to_object(o));
    }
    
};

static const pni::core::string get_size_doc = 
"Returns the size of an object\n"
"\n"
"The semantics of size depends on the object. In the case of a group "
"the number of children is returned. For attributes and fields the "
"number of elements."
"\n"
":param object object: instance of :py:class:`nxattribute`, :py:class:`nxfield`, or :py:class:`nxgroup`\n"
":return: number of elements or children\n"
":rtype: long\n"
;

static const pni::core::string get_name_doc = 
"Returns the name of an object\n"
"\n"
":param object object: instance of :py:class:`nxattribute`, :py:class:`nxlink`, :py:class:`nxfield`, or :py:class:`nxgroup`\n"
":return: the objects name\n"
":rtype: str\n"
;

static const pni::core::string get_rank_doc = 
"Returns the rank of a field or attribute\n"
"\n"
"Returns the number of dimensions a field or attribute has.\n"
"\n"
":param object object: instance of :py:class:`nxattribute` or :py:class:`nxfield`\n"
":return: the rank of the object\n"
":rtype: long\n"
;

static const pni::core::string get_unit_doc = 
"Return the unit of a field\n"
"\n"
"Convenience function reading the `units` attribute of a field and returns "
"its value.\n"
"\n"
":param nxfield field: the field from which to read the unit\n"
":return: value of the units attribute\n"
":rtype: str\n"
;

static const pni::core::string set_unit_doc = 
"Set the unit of a field\n"
"\n"
"Convenience function setting the units attribute of a field.\n"
"\n"
":param nxfield field: the field for which to set the unit\n"
":param unit str: the unit for the field\n"
;

static const pni::core::string get_path_doc = 
"Return the NeXus path of an object\n"
"\n"
"Return the full NeXus path of an object.\n"
"\n"
":param object object: instance of :py:class:`nxfield`, :py:class:`nxattribute`, :py:class:`nxgroup`, or :py:class:`nxlink`\n"
":return: full NeXus path\n"
":rtype: str\n";

static const pni::core::string get_class_doc = 
"Read NXclass attribute from group\n"
"\n"
"Convenience function reading the NX_class attribute of a group and returns "
"and returns its value.\n"
"\n"
":param object group: group from which to read NX_class\n"
":return: value of NX_class\n"
":rtype: str\n";

static const pni::core::string set_class_doc = 
"Set the NX_class attribute of a group\n"
"\n"
"Convenience function to set the base class of a group by writing "
"the NX_class attribute of a group. \n"
"\n"
":param nxgroup group: the group for which to set the class\n"
":param str class: the base class \n";

static  const pni::core::string get_object_doc = 
"Get object by path\n"
"\n"
"Return an object determined by its path from a parent. The path can either "
"be relative to the parent or absolute. In the latter case the parent "
"will only be used to obtain the root group.\n"
"\n"
":param nxgroup parent: parent object \n"
":param str path: relative or absolute path to the object\n"
":return: the requested object\n"
":rtype: an attribute, field, group, or link instance\n"
;


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
    using namespace boost::python;
    typedef algorithms_wrapper<GTYPE,FTYPE,ATYPE> wrapper_type;

    def("get_size",&wrapper_type::get_size,get_size_doc.c_str());
    def("get_name",&wrapper_type::get_name,get_name_doc.c_str());
    def("get_rank",&wrapper_type::get_rank,get_rank_doc.c_str());
    def("get_object",&wrapper_type::get_object_nxpath,get_object_doc.c_str());
    def("get_object",&wrapper_type::get_object_string);
    def("get_unit",&wrapper_type::get_unit,get_unit_doc.c_str());
    def("get_class",&wrapper_type::get_class,get_class_doc.c_str());
    def("set_class",&wrapper_type::set_class,set_class_doc.c_str());
    def("set_unit",&wrapper_type::set_unit,set_unit_doc.c_str());
    def("get_path",&wrapper_type::get_path,get_path_doc.c_str());

}

