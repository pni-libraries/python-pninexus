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
// Created on: Jan 26, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>

void create_class_wrappers()
{
  using namespace boost::python;
  using namespace hdf5::property;

  class_<List>("List",no_init)
      .add_property("class",&List::get_class)
      ;

  class_<DatasetTransferList,bases<List>>("DatasetTransferList")
      ;

  class_<FileAccessList,bases<List>>("FileAccessList")
      .def("library_version_bounds",&FileAccessList::library_version_bounds)
      .add_property("library_version_bound_high",&FileAccessList::library_version_bound_high)
      .add_property("library_version_bound_low",&FileAccessList::library_version_bound_low)
      ;

  class_<FileMountList,bases<List>>("FileMountList")
      ;


  size_t (LinkAccessList::*get_link_traversals)() const = &LinkAccessList::maximum_link_traversals;
  void (LinkAccessList::*set_link_traversals)(size_t) const= &LinkAccessList::maximum_link_traversals;
  boost::filesystem::path (LinkAccessList::*get_external_link_prefix)() const =
      &LinkAccessList::external_link_prefix;
  void (LinkAccessList::*set_external_link_prefix)(const boost::filesystem::path &) =
      &LinkAccessList::external_link_prefix;
  class_<LinkAccessList,bases<List>>("LinkAccessList")
      .add_property("maximum_link_traversals",get_link_traversals,set_link_traversals)
      .add_property("external_link_prefix",get_external_link_prefix,set_external_link_prefix)
      ;


  class_<ObjectCopyList,bases<List>>("ObjectCopyList")
      ;

  class_<ObjectCreationList,bases<List>>("ObjectCreationList")
      ;

  class_<StringCreationList,bases<List>>("StringCreationList")
      ;

  class_<DatasetAccessList,bases<LinkAccessList>>("DatasetAccessList")
      ;

  class_<DatatypeAccessList,bases<LinkAccessList>>("DatatypeAccessList")
      ;

  class_<GroupAccessList,bases<LinkAccessList>>("GroupAccessList")
      ;

  class_<DatasetCreationList,bases<ObjectCreationList>>("DatasetCreationList")
      ;

  class_<GroupCreationList,bases<ObjectCreationList>>("GroupCreationList")
      ;

  class_<TypeCreationList,bases<ObjectCreationList>>("TypeCreationList")
      ;

  class_<AttributeCreationList,bases<StringCreationList>>("AttributeCreationList")
      ;

  class_<LinkCreationList,bases<StringCreationList>>("LinkCreationList")
      ;

}
