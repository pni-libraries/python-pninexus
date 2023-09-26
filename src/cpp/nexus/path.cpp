//
// (c) Copyright 2018 DESY
//
// This file is part of python-pninexus.
//
// python-pninexus is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pninexus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pninexus.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
//  Created on: Feb 9, 2018
//      Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <boost/python.hpp>
#include <pni/nexus.hpp>
#include "element_dict_converter.hpp"

using namespace boost::python;
using namespace pni;

nexus::PathObjectList get_objects_path(const hdf5::node::Group &base,
				       const nexus::Path &path)
{
  return nexus::get_objects(base, path);
}



void create_path_wrappers()
{
  void (nexus::Path::*set_filename)(const fs::path &) = &nexus::Path::filename;
  fs::path (nexus::Path::*get_filename)() const = &nexus::Path::filename;
  void (nexus::Path::*set_attribute)(const std::string &) = &nexus::Path::attribute;
  std::string (nexus::Path::*get_attribute)() const = &nexus::Path::attribute;
  nexus::Path::ConstElementIterator (nexus::Path::*cbegin)() const = &nexus::Path::begin;
  nexus::Path::ConstElementIterator (nexus::Path::*cend)() const = &nexus::Path::end;
  class_<nexus::Path>("Path")
      .def(init<const nexus::Path&>())
      .def(init<const hdf5::Path&>())
      .def(init<const std::string&>())
      .def("from_string",&nexus::Path::from_string)
      .staticmethod("from_string")
      .def("to_string",&nexus::Path::to_string)
      .staticmethod("to_string")
      .add_property("has_filename",&nexus::Path::has_filename)
      .add_property("filename",get_filename,set_filename)
      .add_property("has_attribute",&nexus::Path::has_attribute)
      .add_property("attribute",get_attribute,set_attribute)
      .def("push_back",&nexus::Path::push_back)
      .def("push_front",&nexus::Path::push_front)
      .def("pop_back",&nexus::Path::pop_back)
      .def("pop_front",&nexus::Path::pop_front)
      .add_property("front",&nexus::Path::front)
      .add_property("back",&nexus::Path::back)
      .add_property("size",&nexus::Path::size)
      .def("__iter__",boost::python::range(cbegin,cend))
      .def("__str__",nexus::Path::to_string)
      .def(self == self)
      .def(self != self)
      ;

  //
  // wrap utiltiy functions
  //
  def("has_file_section",nexus::has_file_section);
  def("has_attribute_section",nexus::has_attribute_section);
  def("is_absolute",nexus::is_absolute);
  def("is_unique",nexus::is_unique);
  def("split_path",nexus::split_path);
  def("split_last",nexus::split_last);
  def("join",nexus::join);
  def("make_relative",nexus::make_relative);

  bool (*match_function)(const nexus::Path&,const nexus::Path&) = &nexus::match;
  def("match",match_function);

  nexus::Path (*get_path_node)(const hdf5::node::Node &) = &nexus::get_path;
  nexus::Path (*get_path_attr)(const hdf5::attribute::Attribute &) = &nexus::get_path;
  def("get_path",get_path_node);
  def("get_path",get_path_attr);

  def("get_objects_", get_objects_path);

  nxpath_element_to_dict_converter();
  dict_to_nxpath_element_converter();
}
