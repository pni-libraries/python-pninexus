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
// Created on: Jan 26, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>

void create_copyflag_wrapper()
{
  using namespace boost::python;
  using namespace hdf5::property;

  enum_<CopyFlag>("CopyFlag")
      .value("SHALLOW_HIERARCHY",CopyFlag::ShallowHierarchy)
      .value("EXPAND_SOFT_LINKS",CopyFlag::ExpandSoftLinks)
      .value("EXPAND_EXTERNAL_LINKS",CopyFlag::ExpandExternalLinks)
      .value("EXPAND_REFERENCES",CopyFlag::ExpandReferences)
      .value("WITHOUT_ATTRIBUTES",CopyFlag::WithoutAttributes)
      .value("MERGE_COMMITTED_TYPES",CopyFlag::MergeCommittedTypes)
      ;


  void (CopyFlags::*set_shallow_hierarchy)(bool) = &CopyFlags::shallow_hierarchy;
  bool (CopyFlags::*get_shallow_hierarchy)() const = &CopyFlags::shallow_hierarchy;
  void (CopyFlags::*set_expand_soft_links)(bool) = &CopyFlags::expand_soft_links;
  bool (CopyFlags::*get_expand_soft_links)() const = &CopyFlags::expand_soft_links;
  void (CopyFlags::*set_expand_external_links)(bool) = &CopyFlags::expand_external_links;
  bool (CopyFlags::*get_expand_external_links)() const = &CopyFlags::expand_external_links;
  void (CopyFlags::*set_expand_references)(bool) = &CopyFlags::expand_references;
  bool (CopyFlags::*get_expand_references)() const = &CopyFlags::expand_references;
  void (CopyFlags::*set_without_attributes)(bool) = &CopyFlags::without_attributes;
  bool (CopyFlags::*get_without_attributes)() const = &CopyFlags::without_attributes;
  void (CopyFlags::*set_merge_committed_types)(bool) = &CopyFlags::merge_committed_types;
  bool (CopyFlags::*get_merge_committed_types)() const = &CopyFlags::merge_committed_types;

  class_<CopyFlags>("CopyFlags")
      .def(init<>())
      .def(init<unsigned>())
      .def(init<CopyFlags>())
      .add_property("shallow_hierarchy",get_shallow_hierarchy,set_shallow_hierarchy)
      .add_property("expand_soft_links",get_expand_soft_links,set_expand_soft_links)
      .add_property("expand_external_links",get_expand_external_links,set_expand_external_links)
      .add_property("expand_references",get_expand_references,set_expand_references)
      .add_property("without_attributes",get_without_attributes,set_without_attributes)
      .add_property("merge_committed_types",get_merge_committed_types,set_merge_committed_types)
      .def(self | CopyFlag())
      .def(CopyFlag() | self)
      .def(self | self)
      ;

}
