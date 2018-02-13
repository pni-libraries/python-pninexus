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
#include "../common/io.hpp"
#include "../common/converters.hpp"

namespace {

void write_all(const hdf5::node::Dataset &self,const boost::python::object &data,
               const hdf5::datatype::Datatype &memory_type,
               const hdf5::dataspace::Dataspace &memory_space,
               const hdf5::dataspace::Dataspace &file_space)
{

  boost::python::object temp_object;
  numpy::ArrayAdapter array_adapter(data);

  self.write(array_adapter,memory_type,memory_space,file_space);
}

void write_with_selection(const hdf5::node::Dataset &self,
                          const boost::python::object &data,
                          const hdf5::dataspace::Selection &selection)
{
  using hdf5::datatype::Datatype;
  using hdf5::datatype::String;
  using hdf5::dataspace::Dataspace;
  using hdf5::dataspace::SelectionOperation;

  boost::python::object temp_object;
  numpy::ArrayAdapter array_adapter;

  if(numpy::is_array(data))
    array_adapter = numpy::ArrayAdapter(data);
  else
  {
    temp_object = numpy::ArrayFactory::create(data);
    array_adapter = numpy::ArrayAdapter(temp_object);
  }

  Datatype mem_type = hdf5::datatype::create<numpy::ArrayAdapter>(array_adapter);
  Dataspace mem_space = hdf5::dataspace::create<numpy::ArrayAdapter>(array_adapter);
  Dataspace file_space = self.dataspace();

  if(io::has_variable_length_string_type(self) &&
      (mem_type.get_class() == hdf5::datatype::Class::STRING))
    mem_type = String::variable();

  file_space.selection(SelectionOperation::SET,selection);
  self.write(array_adapter,mem_type,mem_space,file_space);
}

boost::python::object read_all(const hdf5::node::Dataset &self)
{
  return io::read(self);
}

void read_all_inplace(const hdf5::node::Dataset &self,boost::python::object &object)
{
  io::read(self,object);
}

boost::python::object read_with_selection(const hdf5::node::Dataset &self,
                         const hdf5::dataspace::Selection &selection)
{
  return io::read(self,selection);
}

boost::python::object get_datatype(const hdf5::node::Dataset &self)
{
  return convert_datatype(self.datatype());
}

boost::python::object get_dataspace(const hdf5::node::Dataset &self)
{
  return convert_dataspace(self.dataspace());
}

} //anonymous namespace

void create_dataset_wrapper()
{
  using namespace boost::python;
  using namespace hdf5::node;
  using hdf5::datatype::Datatype;
  using hdf5::dataspace::Dataspace;
  using hdf5::property::LinkCreationList;
  using hdf5::property::DatasetCreationList;
  using hdf5::property::DatasetAccessList;


  void (Dataset::*set_full_extent)(const hdf5::Dimensions &) const = &Dataset::extent;
  void (Dataset::*grow_dimension)(size_t,ssize_t) const = &Dataset::extent;
  class_<Dataset,bases<Node>>("Dataset")
      .def(init<Group,hdf5::Path,Datatype,Dataspace,LinkCreationList,DatasetCreationList,DatasetAccessList>(
                (arg("parent"),arg("path"),arg("type"),arg("space"),
                    arg("lcpl")=LinkCreationList(),
                    arg("dcpl")=DatasetCreationList(),
                    arg("dapl")=DatasetAccessList()
                    )
                ))
      .def("close",&Dataset::close)
      .def("_write",write_all)
      .def("_write",write_with_selection)
      .def("read",read_all)
      .def("read",read_all_inplace)
      .def("read",read_with_selection)
      .add_property("creation_list",&Dataset::creation_list)
      .add_property("access_list",&Dataset::access_list)
      .add_property("dataspace",get_dataspace)
      .add_property("datatype",get_datatype)
      .def("extent",set_full_extent)
      .def("extent",grow_dimension)
      ;
}
