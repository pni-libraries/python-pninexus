//
// (c) Copyright 2018 DESY
//
// This file is part of python-pni.
//
// python-pni is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pni is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with pyton-pni.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Jan 24, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include "algorithms_wrapper.hpp"

pni::io::nexus::Path _get_path_from_attribute(const hdf5::node::Attribute &attribute)
{
  return pni::io::nexus::get_path(attribute);
}

pni::io::nexus::Path _get_path_from_group(const hdf5::node::Group &group)
{
  return pni::io::nexus::get_path(group);
}

pni::io::nexus::Path _get_path_from_dataset(const hdf5::node::Dataset &dataset)
{
  return pni::io::nexus::get_path(dataset);
}

pni::io::nexus::Path _get_path_from_link(const hdf5::node::Link &link)
{
  return pni::io::nexus::get_path(link.parent())+link.name();
}


void wrap_algorithms()
{
    using namespace boost::python;
    typedef algorithms_wrapper<GTYPE,FTYPE,ATYPE> wrapper_type;

//    def("get_size",&wrapper_type::get_size,get_size_doc.c_str());
//    def("get_name",&wrapper_type::get_name,get_name_doc.c_str());
//    def("get_rank",&wrapper_type::get_rank,get_rank_doc.c_str());
//    def("get_object",&wrapper_type::get_object_nxpath,get_object_doc.c_str());
//    def("get_object",&wrapper_type::get_object_string);
//    def("get_unit",&wrapper_type::get_unit,get_unit_doc.c_str());
//    def("get_class",&wrapper_type::get_class,get_class_doc.c_str());
//    def("set_class",&wrapper_type::set_class,set_class_doc.c_str());
//    def("set_unit",&wrapper_type::set_unit,set_unit_doc.c_str());
    def("_get_path_from_attribute",&_get_path_from_attribute);
    def("_get_path_from_group",&_get_path_from_group);
    def("_get_path_from_dataset",&_get_path_from_dataset);
    def("_get_path_from_link",&_get_path_from_link);

}
