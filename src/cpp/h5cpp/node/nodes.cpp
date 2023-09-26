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
// Created on: Jan 25, 2018
//     Authors: Eugen Wintersberger <eugen.wintersberger@desy.de>,
//              Jan Kotanski <jan.kotanski@desy.de>
//

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PNI_CORE_USYMBOL
extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>
#include "../errors.hpp"
#include "wrappers.hpp"


#if PY_MAJOR_VERSION >= 3
static void * init_numpy()
{
    import_array();
    return NULL;
}
#else 
static void init_numpy()
{
    import_array();
}
#endif

hdf5::node::Link get_link_by_index(const hdf5::node::LinkView &self,size_t index)
{
  if(index>=self.size())
    throw IndexError();

  return self[index];
}

hdf5::node::Link get_link_by_name(const hdf5::node::LinkView &self,const std::string &name)
{
  return self[name];
}

boost::python::object object_from_node(const hdf5::node::Node &node)
{
  if(node.type()==hdf5::node::Type::Dataset)
    return boost::python::object(hdf5::node::Dataset(node));
  else if(node.type()==hdf5::node::Type::Group)
    return boost::python::object(hdf5::node::Group(node));
  else
    return boost::python::object(node);
}

boost::python::object get_node_by_index(const hdf5::node::NodeView &self,size_t index)
{
  if(index>=self.size())
    throw IndexError();

  return object_from_node(self[index]);
}

boost::python::object get_node_by_name(const hdf5::node::NodeView &self,const std::string &name)
{
  return object_from_node(self[name]);
}

hdf5::node::RecursiveNodeIterator recursive_node_begin(const hdf5::node::Group &self)
{
  return hdf5::node::RecursiveNodeIterator::begin(self);
}

hdf5::node::RecursiveNodeIterator recursive_node_end(const hdf5::node::Group &self)
{
  return hdf5::node::RecursiveNodeIterator::end(self);
}

class RecursiveNodeIteratorWrapper
{
  private:
    hdf5::node::RecursiveNodeIterator begin;
    hdf5::node::RecursiveNodeIterator end;

  public:
    RecursiveNodeIteratorWrapper(const hdf5::node::Group &group):
      begin(hdf5::node::RecursiveNodeIterator::begin(group)),
      end(hdf5::node::RecursiveNodeIterator::end(group))
    {}

    static RecursiveNodeIteratorWrapper create(const hdf5::node::NodeView &self)
    {
      return RecursiveNodeIteratorWrapper(self.group());
    }

    boost::python::object next()
    {
      if(begin == end)
      {
        PyErr_SetString(PyExc_StopIteration,"No more data!");
        boost::python::throw_error_already_set();
      }

      boost::python::object result = object_from_node(*begin);
      begin++;
      return result;

    }

    RecursiveNodeIteratorWrapper __iter__()
    {
      return *this;
    }
};

class RecursiveLinkIteratorWrapper
{
  private:
    hdf5::node::RecursiveLinkIterator begin;
    hdf5::node::RecursiveLinkIterator end;

  public:
    RecursiveLinkIteratorWrapper(const hdf5::node::Group &group):
      begin(hdf5::node::RecursiveLinkIterator::begin(group)),
      end(hdf5::node::RecursiveLinkIterator::end(group))
    {}

    static RecursiveLinkIteratorWrapper create(const hdf5::node::LinkView &self)
    {
      return RecursiveLinkIteratorWrapper(self.group());
    }

    hdf5::node::Link next()
    {
      if(begin == end)
      {
        PyErr_SetString(PyExc_StopIteration,"No more data!");
        boost::python::throw_error_already_set();
      }

      hdf5::node::Link result = *begin;
      begin++;
      return result;

    }

    RecursiveLinkIteratorWrapper __iter__()
    {
      return *this;
    }
};

//
// this function is a hack - it seems that the default exists() method
// of NodeView does not work as expected from Python though it does
// in C++. Further investigation is required. However, for the moment,
// this seems to work.
//
bool custom_node_view_exists(const hdf5::node::NodeView &self,
                             const std::string &node_name,
                             const hdf5::property::LinkAccessList &lapl)
{
  return self.group().links.exists(node_name,lapl) &&
         self.group().links[node_name].is_resolvable();
}



BOOST_PYTHON_MODULE(_node)
{
  using namespace boost::python;
  using namespace hdf5::node;

  init_numpy();

  //
  // setting up the documentation options
  //
  docstring_options doc_opts;
  doc_opts.disable_signatures();
  doc_opts.enable_user_defined();

  // ========================================================================
  // Wrapping enumerations
  // ========================================================================

  enum_<Type>("Type")
      .value("UNKOWN",Type::Unknown)
      .value("GROUP",Type::Group)
      .value("DATASET",Type::Dataset)
      .value("DATATYPE",Type::Datatype)
      ;

  enum_<LinkType>("LinkType")
      .value("HARD",LinkType::Hard)
      .value("SOFT",LinkType::Soft)
      .value("EXTERNAL",LinkType::External)
      .value("ERROR",LinkType::Error)
      ;

  // ========================================================================
  // wrapping classes
  // ========================================================================

  class_<Node>("Node")
      .add_property("type",&Node::type)
      .add_property("is_valid",&Node::is_valid)
      .add_property("link",make_function(&Node::link,return_internal_reference<>()))
      .def_readonly("attributes",&Node::attributes)
      ;

  class_<GroupView>("GroupView",init<Group&>())
      .add_property("size",&GroupView::size)
          ;

  class_<RecursiveNodeIteratorWrapper>("RecursiveNodeIterator",no_init)
#if PY_MAJOR_VERSION >= 3
      .def("__next__",&RecursiveNodeIteratorWrapper::next)
#else
	  .def("next",&RecursiveNodeIteratorWrapper::next)
#endif

      .def("__iter__",&RecursiveNodeIteratorWrapper::__iter__)
      ;

  class_<NodeView,bases<GroupView>>("NodeView",init<Group &>())
      .def("exists",custom_node_view_exists,(arg("name"),arg("lapl")=hdf5::property::LinkAccessList()))
      .def("__getitem__",get_node_by_index)
      .def("__getitem__",get_node_by_name)
      .add_property("recursive",RecursiveNodeIteratorWrapper::create)
      ;

  class_<RecursiveLinkIteratorWrapper>("RecursiveLinkIterator",no_init)
#if PY_MAJOR_VERSION >=3
	  .def("__next__",&RecursiveLinkIteratorWrapper::next)
#else
      .def("next",&RecursiveLinkIteratorWrapper::next)
#endif
      .def("__iter__",&RecursiveLinkIteratorWrapper::__iter__)
      ;

  class_<LinkView,bases<GroupView>>("LinkView",init<Group &>())
      .def("exists",&LinkView::exists,(arg("name"),arg("lapl")=hdf5::property::LinkAccessList()))
      .def("__getitem__",get_link_by_index)
      .def("__getitem__",get_link_by_name)
      .add_property("recursive",RecursiveLinkIteratorWrapper::create)
          ;

  class_<Group,bases<Node>>("Group")
      .def(init<Group,
                std::string,
                hdf5::property::LinkCreationList,
                hdf5::property::GroupCreationList,
                hdf5::property::GroupAccessList>(
                (arg("parent"),arg("name"),
                arg("lcpl")=hdf5::property::LinkCreationList(),
                arg("gcpl")=hdf5::property::GroupCreationList(),
                arg("gapl")=hdf5::property::GroupAccessList())
                ))
      .def(init<const Group &>())
      .def(init<>())
      .def_readonly("links",&Group::links)
      .def_readonly("nodes",&Group::nodes)
      .def("close",&Group::close)
      .def("get_group_", &Group::get_group,
	   (arg("path"),
	    arg("lapl")=hdf5::property::LinkAccessList()))
      .def("get_dataset_", &Group::get_dataset,
	   (arg("path"),
	    arg("lapl")=hdf5::property::LinkAccessList()))
      .def("has_group_", &Group::has_group,
	   (arg("path"),
	    arg("lapl")=hdf5::property::LinkAccessList()))
      .def("has_dataset_", &Group::has_dataset,
	   (arg("path"),
	    arg("lapl")=hdf5::property::LinkAccessList()))
      ;

  class_<LinkTarget>("LinkTarget")
      .add_property("file_path",&LinkTarget::file_path)
      .add_property("object_path",&LinkTarget::object_path)
      ;

  class_<Link>("Link")
      .add_property("path",&Link::path)
      .def("target",&Link::target,(args("lapl")=hdf5::property::LinkAccessList()))
      .def("type",&Link::type,(args("lapl")=hdf5::property::LinkAccessList()))
      .add_property("parent",&Link::parent)
      .add_property("file",make_function(&Link::file,return_internal_reference<>()))
      .add_property("exists",&Link::exists)
      .add_property("is_resolvable",&Link::is_resolvable)
      .add_property("node",&Link::operator*)
      ;


  create_dataset_wrapper();
  create_function_wrapper();


}
