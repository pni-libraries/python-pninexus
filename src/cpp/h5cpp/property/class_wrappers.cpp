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
//     Authors:
//             Eugen Wintersberger <eugen.wintersberger@desy.de>
//             Jan Kotanski <jan.kotanski@desy.de>
//

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>

namespace {

bool get_intermediate_group_creation(const hdf5::property::LinkCreationList &list)
{
  return list.intermediate_group_creation();
}

void set_intermediate_group_creation(hdf5::property::LinkCreationList &list,bool value)
{
  if(value)
    list.enable_intermediate_group_creation();
  else
    list.disable_intermediate_group_creation();
}

bool get_time_tracking(const hdf5::property::ObjectCreationList &list)
{
  return list.time_tracking();
}

void set_time_tracking(hdf5::property::ObjectCreationList &list,bool value)
{
  if(value)
    list.enable_time_tracking();
  else
    list.disable_time_tracking();
}

}

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
      .def<void (FileAccessList::*)(CloseDegree) const>("set_close_degree",&FileAccessList::close_degree)
      .add_property<CloseDegree (FileAccessList::*)() const>("close_degree",&FileAccessList::close_degree)
      .add_property("library_version_bound_high",&FileAccessList::library_version_bound_high)
      .add_property("library_version_bound_low",&FileAccessList::library_version_bound_low)
      ;

  class_<FileMountList,bases<List>>("FileMountList")
      ;


  size_t (LinkAccessList::*get_link_traversals)() const = &LinkAccessList::maximum_link_traversals;
  void (LinkAccessList::*set_link_traversals)(size_t) const= &LinkAccessList::maximum_link_traversals;
  boost::filesystem::path (LinkAccessList::*get_external_link_prefix)() const =  &LinkAccessList::external_link_prefix;
  void (LinkAccessList::*set_external_link_prefix)(const boost::filesystem::path &) =  &LinkAccessList::external_link_prefix;

  class_<LinkAccessList,bases<List>>("LinkAccessList")
      .add_property("maximum_link_traversals",get_link_traversals,set_link_traversals)
      .add_property("external_link_prefix",get_external_link_prefix,set_external_link_prefix)
      ;


  CopyFlags (ObjectCopyList::*get_copy_flags)() const = &ObjectCopyList::flags;
  void (ObjectCopyList::*set_copy_flags)(const CopyFlags &) const = &ObjectCopyList::flags;

  class_<ObjectCopyList,bases<List>>("ObjectCopyList")
      .add_property("flags",get_copy_flags,set_copy_flags)
      ;


  CreationOrder (ObjectCreationList::*get_attribute_creation_order)() const = &ObjectCreationList::attribute_creation_order;
  void (ObjectCreationList::*set_attribute_creation_order)(const CreationOrder &) const = &ObjectCreationList::attribute_creation_order;

  class_<ObjectCreationList,bases<List>>("ObjectCreationList")
      .add_property("time_tracking",&get_time_tracking,&set_time_tracking)
      .add_property("attribute_creation_order",get_attribute_creation_order,set_attribute_creation_order)
      .def("attribute_storage_thresholds",&ObjectCreationList::attribute_storage_thresholds)
      .add_property("attribute_storage_maximum_compact",&ObjectCreationList::attribute_storage_maximum_compact)
      .add_property("attribute_storage_minimum_dense",&ObjectCreationList::attribute_storage_minimum_dense)

      ;

  hdf5::datatype::CharacterEncoding (StringCreationList::*get_string_character_encoding)() const =
      &StringCreationList::character_encoding;
  void (StringCreationList::*set_string_character_encoding)(hdf5::datatype::CharacterEncoding) const =
      &StringCreationList::character_encoding;
  class_<StringCreationList,bases<List>>("StringCreationList")
      .add_property("character_encoding",get_string_character_encoding,set_string_character_encoding)
      ;


  ChunkCacheParameters (DatasetAccessList::*get_chunk_cache_parameters)() const =  &DatasetAccessList::chunk_cache_parameters;
  void (DatasetAccessList::*set_chunk_cache_parameters)(const ChunkCacheParameters &) const = &DatasetAccessList::chunk_cache_parameters;

#if H5_VERSION_GE(1, 10, 0)
  VirtualDataView (DatasetAccessList::*get_virtual_view)() const =  &DatasetAccessList::virtual_view;
  void (DatasetAccessList::*set_virtual_view)(VirtualDataView) const = &DatasetAccessList::virtual_view;
#endif

  class_<DatasetAccessList,bases<LinkAccessList>>("DatasetAccessList")
      .add_property("chunk_cache_parameters",get_chunk_cache_parameters,set_chunk_cache_parameters)
#if H5_VERSION_GE(1, 10, 0)
      .add_property("virtual_view",get_virtual_view,set_virtual_view)
#endif

      ;

  class_<DatatypeAccessList,bases<LinkAccessList>>("DatatypeAccessList")
      ;

  class_<GroupAccessList,bases<LinkAccessList>>("GroupAccessList")
      ;


  DatasetLayout (DatasetCreationList::*get_dataset_layout)() const =   &DatasetCreationList::layout;
  void (DatasetCreationList::*set_dataset_layout)(DatasetLayout) const = &DatasetCreationList::layout;
  hdf5::Dimensions (DatasetCreationList::*get_dataset_chunk)() const = &DatasetCreationList::chunk;
  void (DatasetCreationList::*set_dataset_chunk)(const hdf5::Dimensions &) const = &DatasetCreationList::chunk;
  DatasetFillTime (DatasetCreationList::*get_dataset_fill_time)() const =  &DatasetCreationList::fill_time;
  void (DatasetCreationList::*set_dataset_fill_time)(DatasetFillTime) const =  &DatasetCreationList::fill_time;
  DatasetAllocTime (DatasetCreationList::*get_dataset_allocation_time)() const =  &DatasetCreationList::allocation_time;
  void (DatasetCreationList::*set_dataset_allocation_time)(DatasetAllocTime) const =  &DatasetCreationList::allocation_time;

  class_<DatasetCreationList,bases<ObjectCreationList>>("DatasetCreationList")
      .add_property("layout",get_dataset_layout,set_dataset_layout)
      .add_property("chunk",get_dataset_chunk,set_dataset_chunk)
      .add_property("fill_value_status",&DatasetCreationList::fill_value_status)
      .add_property("fill_time",get_dataset_fill_time,set_dataset_fill_time)
      .add_property("allocation_time",get_dataset_allocation_time,set_dataset_allocation_time)
      ;


  size_t (GroupCreationList::*get_local_heap_size_hint)() const = &GroupCreationList::local_heap_size_hint;
  void (GroupCreationList::*set_local_heap_size_hint)(size_t) const = &GroupCreationList::local_heap_size_hint;
  unsigned (GroupCreationList::*get_estimated_number_of_links)() const = &GroupCreationList::estimated_number_of_links;
  void (GroupCreationList::*set_estimated_number_of_links)(unsigned) const = &GroupCreationList::estimated_number_of_links;
  unsigned (GroupCreationList::*get_estimated_link_name_length)() const = &GroupCreationList::estimated_link_name_length;
  void (GroupCreationList::*set_estimated_link_name_length)(unsigned) const = &GroupCreationList::estimated_link_name_length;
  CreationOrder (GroupCreationList::*get_link_creation_order)() const = &GroupCreationList::link_creation_order;
  void (GroupCreationList::*set_link_creation_order)(CreationOrder) const = &GroupCreationList::link_creation_order;

  class_<GroupCreationList,bases<ObjectCreationList>>("GroupCreationList")
      .add_property("local_heap_size_hint",get_local_heap_size_hint,set_local_heap_size_hint)
      .add_property("estimated_number_of_links",get_estimated_number_of_links,set_estimated_number_of_links)
      .add_property("estimated_link_name_length",get_estimated_link_name_length,set_estimated_link_name_length)
      .add_property("link_creation_order",get_link_creation_order,set_link_creation_order)
      .def("link_storage_thresholds",&GroupCreationList::link_storage_thresholds)
      .add_property("link_storage_maximum_compact",&GroupCreationList::link_storage_maximum_compact)
      .add_property("link_storage_minimum_dense",&GroupCreationList::link_storage_minimum_dense)
      ;

  class_<TypeCreationList,bases<ObjectCreationList>>("TypeCreationList")
      ;

  class_<AttributeCreationList,bases<StringCreationList>>("AttributeCreationList")
      ;

  class_<LinkCreationList,bases<StringCreationList>>("LinkCreationList")
      .add_property("intermediate_group_creation",&get_intermediate_group_creation,&set_intermediate_group_creation)
      ;

  hsize_t (FileCreationList::*get_user_block)() const = &FileCreationList::user_block;
  void (FileCreationList::*set_user_block)(hsize_t)const = &FileCreationList::user_block;
  size_t (FileCreationList::*get_object_offset_size)() const = &FileCreationList::object_offset_size;
  void (FileCreationList::*set_object_offset_size)(size_t) const = &FileCreationList::object_offset_size;
  size_t (FileCreationList::*get_object_length_size)() const = &FileCreationList::object_length_size;
  void (FileCreationList::*set_object_length_size)(size_t) const = &FileCreationList::object_length_size;
  unsigned int (FileCreationList::*get_btree_rank)() const = &FileCreationList::btree_rank;
  void (FileCreationList::*set_btree_rank)(unsigned int) = &FileCreationList::btree_rank;
  unsigned int (FileCreationList::*get_btree_symbols)() const = &FileCreationList::btree_symbols;
  void (FileCreationList::*set_btree_symbols)(unsigned int) = &FileCreationList::btree_symbols;
  unsigned int (FileCreationList::*get_chunk_tree_rank)() const = &FileCreationList::chunk_tree_rank;
  void (FileCreationList::*set_chunk_tree_rank)(unsigned int) = &FileCreationList::chunk_tree_rank;
  class_<FileCreationList,bases<GroupCreationList>>("FileCreationList")
      .add_property("user_block",get_user_block,set_user_block)
      .add_property("object_offset_size",get_object_offset_size,set_object_offset_size)
      .add_property("object_length_size",get_object_length_size,set_object_length_size)
      .add_property("btree_rank",get_btree_rank,set_btree_rank)
      .add_property("btree_symbols",get_btree_symbols,set_btree_symbols)
      .add_property("chunk_tree_rank",get_chunk_tree_rank,set_chunk_tree_rank)
      ;
}
