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
// Created on: Sep 18, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include "nxgroup_wrapper.hpp"
#include <core/utils.hpp>
#include <pni/core/types.hpp>

hdf5::datatype::Datatype GroupWrapper::create_datatype(const std::string &type_code)
{
  using namespace pni::core;

  type_id_t tid = type_id_from_str(type_code);

  return pni::io::nexus::DatatypeFactory::create(tid);
}

GroupWrapper::GroupWrapper():
  group_(),
  attributes(group_.attributes)
{}

GroupWrapper::GroupWrapper(const hdf5::node::Group &group):
  group_(group),
  attributes(group_.attributes)
{}

GroupWrapper::GroupWrapper(hdf5::node::Group &&group):
    group_(std::move(group)),
    attributes(group_.attributes)
{}

hdf5::node::Group GroupWrapper::create_group(const std::string &n,
                                             const std::string &nxclass) const
{
    return pni::io::nexus::BaseClassFactory::create(group_,n,nxclass);
}

hdf5::node::Dataset GroupWrapper::create_field(const std::string &name,
                                               const std::string &type_code,
                                               const boost::python::object &shape,
                                               const boost::python::object &chunk,
                                               const boost::python::object &filter) const
{
  using namespace boost::python;

  hdf5::datatype::Datatype type = create_datatype(type_code);

  hdf5::Dimensions current_dimensions{1},chunk_dimensions{1024*1024*100};

  if(!shape.is_none())
  {
    current_dimensions = List2Container<hdf5::Dimensions>(list(shape));
  }

  //
  // setting the maximum dimensions
  //
  hdf5::Dimensions maximum_dimensions(current_dimensions);
  maximum_dimensions.front() = hdf5::dataspace::Simple::UNLIMITED;

  //
  // if the user has provided some chunk dimensions we use them - it is up
  // to the user to set the chunk dimensions correctly
  //
  if(!chunk.is_none())
  {
    chunk_dimensions = List2Container<hdf5::Dimensions>(list(chunk));
  }
  else
  {
    if(current_dimensions.size()>1)
    {
      chunk_dimensions = current_dimensions;
      chunk_dimensions.front() = 1;
    }
  }

  hdf5::dataspace::Simple space{current_dimensions,maximum_dimensions};

  hdf5::property::LinkCreationList lcpl;
  hdf5::property::DatasetCreationList dcpl;
  dcpl.layout(hdf5::property::DatasetLayout::CHUNKED);
  dcpl.chunk(chunk_dimensions);

//  if(!filter.is_none())
//  {
//    //extract filter
//    extract<deflate_type> deflate_object(filter);
//
//    if(!deflate_object.check())
//      throw pni::core::type_error(EXCEPTION_RECORD,
//                                  "Filter is not an instance of filter class!");
//
//    field = create_field(parent,type_id,name,shapes.first,
//                         shapes.second,deflate_object());
//  }

  return hdf5::node::Dataset(group_,name,type,space,lcpl,dcpl);

}

hdf5::node::Node GroupWrapper::open_by_name(const std::string &n) const
{
  return group_.nodes[n];
}

hdf5::node::Node GroupWrapper::open_by_index(size_t i) const
{
  return group_.nodes[i];
}

bool GroupWrapper::exists(const std::string &n) const
{
  return group_.nodes.exists(n);
}

size_t GroupWrapper::nchildren() const
{
  return group_.nodes.size();
}

bool GroupWrapper::is_valid() const
{
  return group_.is_valid();
}


void GroupWrapper::close()
{
  H5Gclose(static_cast<hid_t>(group_));
}


std::string GroupWrapper::filename() const
{
  return group_.link().target().file_path().string();
}


std::string GroupWrapper::name() const
{
  return group_.link().path().name();
}

//--------------------------------------------------------------------
hdf5::node::Node GroupWrapper::parent() const
{
  return group_.link().parent();
}

//--------------------------------------------------------------------
size_t GroupWrapper::__len__() const
{
  return group_.nodes.size();
}

//--------------------------------------------------------------------
std::string GroupWrapper::path() const
{
  using pni::io::nexus::Path;
  using pni::io::nexus::get_path;
  return Path::to_string(get_path(group_));
}

void GroupWrapper::remove(const std::string &name) const
{
  hdf5::node::remove(group_,name);
}

nexus::NodeIteratorWrapper GroupWrapper::__iter__() const
{
  return nexus::NodeIteratorWrapper(group_.nodes.begin(),group_.nodes.end());
}

//--------------------------------------------------------------------
nexus::RecursiveNodeIteratorWrapper GroupWrapper::recursive()
{
  using hdf5::node::RecursiveNodeIterator;
  return nexus::RecursiveNodeIteratorWrapper(RecursiveNodeIterator::begin(group_),
                                             RecursiveNodeIterator::end(group_));
}

PyObject *GroupToPythonObject::convert(const hdf5::node::Group &group)
{
    using namespace boost::python;
    return incref(object(GroupWrapper(group)).ptr());
}


static const char __group_open_docstr[] =
"Opens an existing object \n"
"\n"
"The object is identified by its name (path).\n"
"\n"
":param str n: name (path) to the object to open\n"
":return: the requested child object \n"
":rtype: either an instance of :py:class:`nxfield`, :py:class:`nxgroup`, or "
":py:class:`nxlink`\n"
":raises KeyError: if the child could not be found\n"
;

static const char __group_exists_docstr[] =
"Checks if the object determined by 'name' exists\n"
"\n"
"'name' can be either an absolute or relative path. If the object "
"identified by 'name' exists the method returns true, false otherwise.\n"
"\n"
":param str name: name of the object to check\n"
":return: :py:const:`True` if the object exists, :py:const:`False` otherwise\n"
":rtype: bool"
;

static const char __group_nchilds_docstr[] =
"Read only property providing the number of childs linked below this group.";
static const char __group_childs_docstr[] =
"Read only property providing a sequence object which can be used to iterate\n"
"over all childs linked below this group.";

static const pni::core::string nxgroup_filename_doc =
"Read only property returning the name of the file the group belongs to\n";

static const pni::core::string nxgroup_name_doc =
"Read only property returning the name of the group\n";

static const pni::core::string nxgroup_parent_doc =
"Read only property returning the parent group of this group\n";

static const pni::core::string nxgroup_size_doc =
"Read only property returing the number of links below this group\n";

static const pni::core::string nxgroup_attributes_doc =
"Read only property with the attribute manager for this group\n";

static const pni::core::string nxgroup_recursive_doc =
"Read only property returning a recursive iterator for this group\n";

static const pni::core::string nxgroup_path_doc =
"Read only property returning the NeXus path for this group\n";

static const pni::core::string nxgroup_close_doc =
"Close this group";

static const pni::core::string nxgroup_is_valid_doc =
"Read only property returning :py:const:`True` if this instance is a valid"
" NeXus object";

static const pni::core::string nxgroup__get_item_by_index_doc =
"Get child by index";

static const pni::core::string nxgroup__get_item_by_name_doc =
"Get child by name";

//!
//! \ingroup wrappers
//! \brief create NXGroup wrapper
//!
//! Template function to create a new wrapper for an NXGroup type GType.
//! \param class_name name for the Python class
//!
//!
void wrap_nxgroup(const char *class_name)
{
    using namespace boost::python;

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
#endif
    class_<GroupWrapper>(class_name)
        .def(init<>())
        .def("open",&GroupWrapper::open_by_name,__group_open_docstr)
        .def("__getitem__",&GroupWrapper::open_by_index,nxgroup__get_item_by_index_doc.c_str())
        .def("__getitem__",&GroupWrapper::open_by_name)
        .def("_create_group",&GroupWrapper::create_group,
                ("n",arg("nxclass")=pni::core::string()))
        .def("__create_field",&GroupWrapper::create_field,
                ("name","type",arg("shape")=object(),arg("chunk")=object(),
                 arg("filter")=object()))
        .def("exists",&GroupWrapper::exists,__group_exists_docstr)
        .def("close",&GroupWrapper::close,nxgroup_close_doc.c_str())
        .add_property("is_valid",&GroupWrapper::is_valid,nxgroup_is_valid_doc.c_str())
        .def("__len__",&GroupWrapper::__len__)
        .def("__iter__",&GroupWrapper::__iter__,__group_childs_docstr)
        .def("remove",&GroupWrapper::remove)
        .add_property("filename",&GroupWrapper::filename,nxgroup_filename_doc.c_str())
        .add_property("name",&GroupWrapper::name,nxgroup_name_doc.c_str())
        .add_property("parent",&GroupWrapper::parent,nxgroup_parent_doc.c_str())
        .add_property("size",&GroupWrapper::__len__,nxgroup_size_doc.c_str())
        .def_readonly("attributes",&GroupWrapper::attributes,nxgroup_attributes_doc.c_str())
        .add_property("recursive",&GroupWrapper::recursive,nxgroup_recursive_doc.c_str())
        .add_property("path",&GroupWrapper::path,nxgroup_path_doc.c_str())
        ;
#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif
}


