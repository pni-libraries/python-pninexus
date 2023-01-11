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
//     Authors:
//             Eugen Wintersberger <eugen.wintersberger@desy.de>
//             Jan Kotanski <jan.kotanski@desy.de>
//
//

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>
#include "../common/io.hpp"
#include "../common/converters.hpp"

namespace {

void dataset_write(const hdf5::node::Dataset &self,
                   const boost::python::object &data,
                   const hdf5::datatype::Datatype &memory_type,
                   const hdf5::dataspace::Dataspace &memory_space,
                   const hdf5::dataspace::Dataspace &file_space)
{
  numpy::ArrayAdapter array_adapter(data);

  self.write(array_adapter,memory_type,memory_space,file_space);
}

void dataset_write_chunk(const hdf5::node::Dataset &self,
			 const boost::python::object &data,
			 boost::python::list offset,
			 std::uint32_t filter_mask = 0,
			 const hdf5::property::DatasetTransferList &dtpl =
			 hdf5::property::DatasetTransferList())
{
  numpy::ArrayAdapter array_adapter(data);
  std::vector<long long unsigned int> voffset;

  for (boost::python::ssize_t i = 0, end = len(offset); i < end; ++i){
    boost::python::object o = offset[i];
    boost::python::extract<long long unsigned int> s(o);
    if (s.check()){
      voffset.push_back(s());
    }
  }

  self.write_chunk(array_adapter,voffset,filter_mask,dtpl);
}

#if H5_VERSION_GE(1,10,2)

std::uint32_t dataset_read_chunk(const hdf5::node::Dataset &self,
				 boost::python::object &data,
				 boost::python::list offset,
				 const hdf5::property::DatasetTransferList &dtpl =
				 hdf5::property::DatasetTransferList())
{
  numpy::ArrayAdapter array_adapter(data);
  std::vector<long long unsigned int> voffset;


  for (boost::python::ssize_t i = 0, end = len(offset); i < end; ++i){
    boost::python::object o = offset[i];
    boost::python::extract<long long unsigned int> s(o);
    if (s.check()){
      voffset.push_back(s());
    }
  }

  return self.read_chunk(array_adapter,voffset,dtpl);
}

long long unsigned int dataset_chunk_storage_size(const hdf5::node::Dataset &self,
						  boost::python::list offset)
{
  std::vector<long long unsigned int> voffset;

  for (boost::python::ssize_t i = 0, end = len(offset); i < end; ++i){
    boost::python::object o = offset[i];
    boost::python::extract<long long unsigned int> s(o);
    if (s.check()){
      voffset.push_back(s());
    }
  }

  return self.chunk_storage_size(voffset);
}

  
#endif

boost::python::object dataset_read(const hdf5::node::Dataset &self,
                  boost::python::object &data,
                  const hdf5::datatype::Datatype &memory_type,
                  const hdf5::dataspace::Dataspace &memory_space,
                  const hdf5::dataspace::Dataspace &file_space)
{
  numpy::ArrayAdapter array_adapter(data);

  self.read(array_adapter,memory_type,memory_space,file_space);

  if(self.datatype().get_class() == hdf5::datatype::Class::String)
  {
    hdf5::datatype::String string_type = self.datatype();
    if(!string_type.is_variable_length())
    {
      // std::cout<<"Saving fixed length string array!"<<std::endl;
      PyObject *ptr = reinterpret_cast<PyObject*>(static_cast<PyArrayObject*>(array_adapter));
      Py_XINCREF(ptr);

      boost::python::handle<> h(ptr);
      data = boost::python::object(h);
    }
  }

  return data;
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
  using hdf5::property::DatasetTransferList;


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
      .def(init<const hdf5::node::Dataset&>())
      .def("close",&Dataset::close)
      .def("_write",dataset_write)
      .def("write_chunk",
	   dataset_write_chunk,
	   (arg("data"),
	    arg("offset"),
	    arg("filter_mask")=0,
	    arg("dtpl")=DatasetTransferList()))
#if H5_VERSION_GE(1,10,2)
      .def("read_chunk",
	   dataset_read_chunk,
	   (arg("data"),
	    arg("offset"),
	    arg("dtpl")=DatasetTransferList()))
      .def("chunk_storage_size",dataset_chunk_storage_size)
#endif    
      .def("_read",dataset_read)
      .add_property("creation_list",&Dataset::creation_list)
      .add_property("access_list",&Dataset::access_list)
      .add_property("dataspace",get_dataspace)
      .add_property("datatype",get_datatype)
      .def("extent",set_full_extent)
      .def("extent",grow_dimension)
#if H5_VERSION_GE(1,10,0)
      .def("refresh",&Dataset::refresh)
#endif
      ;

#if H5_VERSION_GE(1,10,0)
  using hdf5::property::VirtualDataMaps;

  class_<VirtualDataset,bases<Dataset>>("VirtualDataset")
    .def(init<Group,hdf5::Path,Datatype,Dataspace,VirtualDataMaps,
	 LinkCreationList,DatasetCreationList,DatasetAccessList>(
		   (arg("parent"), arg("path"), arg("type"), arg("space"), arg("vds_maps"),
                    arg("lcpl")=LinkCreationList(),
                    arg("dcpl")=DatasetCreationList(),
                    arg("dapl")=DatasetAccessList()
                    )
                ))
      .def(init<const hdf5::node::VirtualDataset&>())
      ;

#endif
}
