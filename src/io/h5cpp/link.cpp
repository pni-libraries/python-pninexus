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
// Created on: Jan 25, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>


hdf5::node::LinkTarget get_link_target(const hdf5::node::Link &link)
{
  return link.target();
}

hdf5::node::LinkType get_link_type(const hdf5::node::Link &link)
{
  return link.type();
}


void wrap_link()
{
  using namespace boost::python;
  using namespace hdf5::node;

  enum_<LinkType>("LinkType")
      .value("HARD",LinkType::HARD)
      .value("SOFT",LinkType::SOFT)
      .value("EXTERNAL",LinkType::EXTERNAL)
      .value("ERROR",LinkType::ERROR)
      ;

  class_<LinkTarget>("LinkTarget")
      .add_property("file_path",&LinkTarget::file_path)
      .add_property("object_path",&LinkTarget::object_path)
      ;

  class_<Link>("Link")
      .add_property("path",&Link::path)
      .add_property("target",get_link_target)
      .add_property("type",get_link_type)
      .add_property("parent",&Link::parent)
      .add_property("file",make_function(&Link::file,return_internal_reference<>()))
      .add_property("exists",&Link::exists)
      .add_property("is_resolvable",&Link::is_resolvable)
      .add_property("node",&Link::operator*)
      ;
}
