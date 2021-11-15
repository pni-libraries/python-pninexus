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
//  Created on: Feb 8, 2018
//      Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <boost/python.hpp>
#include "list_converters.hpp"

using namespace boost::python;
using namespace pni;

GroupListToTuple::GroupListToTuple()
{
  to_python_converter<nexus::GroupList,GroupListToTuple>();
}

PyObject *GroupListToTuple::convert(const nexus::GroupList &group_list)
{
  if(group_list.empty())
  {
    return incref(tuple().ptr());
  }
  else
  {
    list l;
    size_t index = 0;
    for(auto group: group_list)
      l.insert(index++,group);

    return incref(tuple(l).ptr());
  }
}

DatasetListToTuple::DatasetListToTuple()
{
  to_python_converter<nexus::DatasetList,DatasetListToTuple>();
}

PyObject *DatasetListToTuple::convert(const nexus::DatasetList &dataset_list)
{
  if(dataset_list.empty())
  {
    return incref(tuple().ptr());
  }
  else
  {
    list l;
    size_t index = 0;
    for(auto dataset: dataset_list)
      l.insert(index++,dataset);

    return incref(tuple(l).ptr());
  }
}

AttributeListToTuple::AttributeListToTuple()
{
  to_python_converter<nexus::AttributeList,AttributeListToTuple>();
}

PyObject *AttributeListToTuple::convert(const nexus::AttributeList &attribute_list)
{
  if(attribute_list.empty())
  {
    return incref(tuple().ptr());
  }
  else
  {
    list l;
    size_t index = 0;
    for(auto attribute: attribute_list)
      l.insert(index++,attribute);

    return incref(tuple(l).ptr());
  }
}

NodeListToTuple::NodeListToTuple()
{
  to_python_converter<nexus::NodeList,NodeListToTuple>();
}

PyObject *NodeListToTuple::convert(const nexus::NodeList &node_list)
{
  if(node_list.empty())
  {
    return incref(tuple().ptr());
  }
  else
  {
    list l;
    size_t index = 0;
    for(auto node: node_list)
    {
      if(node.type()==hdf5::node::Type::Dataset)
        l.insert(index++,hdf5::node::Dataset(node));
      else if(node.type()==hdf5::node::Type::Group)
        l.insert(index++,hdf5::node::Group(node));
    }


    return incref(tuple(l).ptr());
  }
}

PathObjectListToTuple::PathObjectListToTuple()
{
  to_python_converter<nexus::PathObjectList,PathObjectListToTuple>();
}

PyObject *PathObjectListToTuple::convert(const nexus::PathObjectList &object_list)
{
  if(object_list.empty())
  {
    return incref(tuple().ptr());
  }
  else
  {
    list l;
    size_t index = 0;
    for(auto object: object_list)
    {
      if(nexus::is_dataset(object))
      {
        hdf5::node::Node node = object;
        l.insert(index++,hdf5::node::Dataset(node));
      }
      else if(nexus::is_group(object))
      {
        hdf5::node::Node node = object;
        l.insert(index++,hdf5::node::Group(node));
      }
      else if(nexus::is_attribute(object))
        l.insert(index++,hdf5::attribute::Attribute(object));
    }


    return incref(tuple(l).ptr());
  }
}
