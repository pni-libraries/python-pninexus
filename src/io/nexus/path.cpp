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
// along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
//  Created on: Feb 9, 2018
//      Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <boost/python.hpp>
#include <pni/io/nexus.hpp>

using namespace boost::python;
using namespace pni::io;

void create_path_wrappers()
{
  void (nexus::Path::*set_filename)(const boost::filesystem::path &) = &nexus::Path::filename;
  boost::filesystem::path (nexus::Path::*get_filename)() const = &nexus::Path::filename;
  void (nexus::Path::*set_attribute)(const std::string &) = &nexus::Path::attribute;
  std::string (nexus::Path::*get_attribute)() const = &nexus::Path::attribute;
  class_<nexus::Path>("Path")
      .def("from_string",&nexus::Path::from_string)
      .staticmethod("from_string")
      .def("to_string",&nexus::Path::to_string)
      .staticmethod("to_string")
      .add_property("has_filename",&nexus::Path::has_filename)
      .add_property("filename",get_filename,set_filename)
      .add_property("has_attribute",&nexus::Path::has_attribute)
      .add_property("attribute",get_attribute,set_attribute)
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

  def("get_objects",nexus::get_objects);
}
