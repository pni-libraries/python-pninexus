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
// Created on: Feb 15, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>

using namespace boost::python;

namespace {

boost::python::object get_node(const hdf5::node::Group &base,
                               const hdf5::Path &path,
                               const hdf5::property::LinkAccessList &lapl)
{
  hdf5::node::Node n = hdf5::node::get_node(base,path,lapl);
  switch(n.type())
  {
    case hdf5::node::Type::Dataset:
      return boost::python::object(hdf5::node::Dataset(n));
    case hdf5::node::Type::Group:
      return boost::python::object(hdf5::node::Group(n));
    default:
      return boost::python::object(n);
  }
}

}

void create_function_wrapper()
{
  //
  // wrapping predicate functions for types
  //
  def("is_group",hdf5::node::is_group);
  def("is_dataset",hdf5::node::is_dataset);

  //
  // get node functions
  //
  def("get_node_",get_node,(arg("base"),
                           arg("path"),
                           arg("lapl")=hdf5::property::LinkAccessList()));

  //
  // wrapping copy functions
  //
  void (*copy_with_path)(const hdf5::node::Node &,
                         const hdf5::node::Group &,
                         const hdf5::Path &,
                         const hdf5::property::ObjectCopyList &,
                         const hdf5::property::LinkCreationList &) = &hdf5::node::copy;
  void (*copy_default)(const hdf5::node::Node &,
                       const hdf5::node::Group &,
                       const hdf5::property::ObjectCopyList &,
                       const hdf5::property::LinkCreationList &) = &hdf5::node::copy;
  def("_copy",copy_with_path);
  def("_copy",copy_default);

  //
  // wrapping move function
  //
  void (*move_with_path)(const hdf5::node::Node &,
                         const hdf5::node::Group &,
                         const hdf5::Path &,
                         const hdf5::property::LinkCreationList &,
                         const hdf5::property::LinkAccessList &) = &hdf5::node::move;
  void (*move_default)(const hdf5::node::Node &,
                       const hdf5::node::Group &,
                       const hdf5::property::LinkCreationList &,
                       const hdf5::property::LinkAccessList&) = &hdf5::node::move;
  def("_move",move_with_path);
  def("_move",move_default);

  //
  // wrapping remove functions
  //
  void (*remove_with_path)(const hdf5::node::Group &,
                           const hdf5::Path &,
                           const hdf5::property::LinkAccessList &) = &hdf5::node::remove;
  void (*remove_node)(const hdf5::node::Node &,
                      const hdf5::property::LinkAccessList &) = &hdf5::node::remove;
  def("_remove",remove_with_path);
  def("_remove",remove_node);

  //
  // wrapping link functions
  //
  void (*link_node_target)(const hdf5::node::Node &,
                           const hdf5::node::Group &,
                           const hdf5::Path &,
                           const hdf5::property::LinkCreationList &,
                           const hdf5::property::LinkAccessList &) = &hdf5::node::link;
  void (*link_path_target)(const hdf5::Path &,
                           const hdf5::node::Group &,
                           const hdf5::Path &,
                           const hdf5::property::LinkCreationList &,
                           const hdf5::property::LinkAccessList &) = &hdf5::node::link;
  void (*link_external)(const fs::path &,
                        const hdf5::Path &,
                        const hdf5::node::Group &,
                        const hdf5::Path &,
                        const hdf5::property::LinkCreationList &,
                        const hdf5::property::LinkAccessList&) = &hdf5::node::link;
  def("_link",link_node_target);
  def("_link",link_path_target);
  def("_link",link_external);

}
