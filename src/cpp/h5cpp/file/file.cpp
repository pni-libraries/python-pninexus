//
// (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// Created on: Feb 17, 2012
//     Authors:
//            Eugen Wintersberger <eugen.wintersberger@desy.de>
//            Jan Kotanski <jan.kotanski@desy.de>
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
#include "../common/converters.hpp"
#include "../common/io.hpp"
#include "../numpy/numpy.hpp"

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



using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(root_overloads,root,0,0);

hdf5::file::File create_file(const fs::path &file_path,
                             hdf5::file::AccessFlagsBase flags,
                             const hdf5::property::FileCreationList &fcpl,
                             const hdf5::property::FileAccessList &fapl)
{
  return hdf5::file::create(file_path,flags,fcpl,fapl);
}

hdf5::file::File open_file(const fs::path &file_path,
                           hdf5::file::AccessFlagsBase flags,
                           const hdf5::property::FileAccessList &fapl)
{
  return hdf5::file::open(file_path,flags,fapl);
}

hdf5::file::File from_buffer_(boost::python::object &data,
			      hdf5::file::ImageFlagsBase flags)
{
  numpy::ArrayAdapter array_adapter(data);
  return hdf5::file::from_buffer(array_adapter, flags);
}

size_t file_to_buffer(const hdf5::file::File &self,
		 boost::python::object &data)
{
  numpy::ArrayAdapter array_adapter(data);
  return self.to_buffer(array_adapter);
}


BOOST_PYTHON_MODULE(_file)
{
  using namespace hdf5::file;

  init_numpy();
  //
  // setting up the documentation options
  //
  docstring_options doc_opts;
  doc_opts.disable_signatures();
  doc_opts.enable_user_defined();

  enum_<Scope>("Scope","The scope of a file")
            .value("LOCAL",Scope::Local)
            .value("GLOBAL",Scope::Global);

  enum_<AccessFlags>("AccessFlags","The access flags used to open the file")
            .value("TRUNCATE",AccessFlags::Truncate)
            .value("EXCLUSIVE",AccessFlags::Exclusive)
            .value("READWRITE",AccessFlags::ReadWrite)
#if H5_VERSION_GE(1,10,0)
            .value("SWMRREAD",AccessFlags::SWMRRead)
            .value("SWMRWRITE",AccessFlags::SWMRWrite)
#endif
            .value("READONLY",AccessFlags::ReadOnly);

  enum_<ImageFlags>("ImageFlags","The image flags used to open the image file")
            .value("READONLY",ImageFlags::ReadOnly)
            .value("READWRITE",ImageFlags::ReadWrite)
            .value("DONT_COPY",ImageFlags::DontCopy)
            .value("DONT_RELEASE",ImageFlags::DontRelease)
            .value("ALL",ImageFlags::All);

  enum_<DriverID>("DriverID","The file driver ID")
    .value("CUSTOM", DriverID::Custom)
    .value("POSIX", DriverID::Posix)
    .value("DIRECT", DriverID::Direct)
    .value("MEMORY", DriverID::Memory)
    .value("MPI", DriverID::MPI);

  //hdf5::node::Group (hdf5::file::File::*root)() = &hdf5::file::File::root;
  class_<File>("File")
            .def(init<>())
            .def(init<const File>())
            .add_property("intent",&File::intent)
            .add_property("is_valid",&File::is_valid)
            .add_property("path",&File::path)
            .add_property("size",&File::size)
            .add_property("buffer_size",&File::buffer_size)
            .def("flush",&File::flush,(arg("scope")=Scope::Global))
            .def("close",&File::close)
            .def("root",&File::root,root_overloads())
            .def("to_buffer",file_to_buffer,(arg("data")))
            ;

  class_<Driver,boost::noncopyable>("Driver",no_init)
    .add_property("id",&Driver::id)
    .def("__call__",&Driver::operator(),(args("fapl")))
    ;

  class_<PosixDriver,bases<Driver>>("PosixDriver");

  // usage of DirectDriver in python bindings not supported
  //
  // void (DirectDriver::*set_alignment)(size_t) = &DirectDriver::alignment;
  // size_t(DirectDriver::*get_alignment)() const = &DirectDriver::alignment;
  // void (DirectDriver::*set_block_size)(size_t) = &DirectDriver::block_size;
  // size_t(DirectDriver::*get_block_size)() const = &DirectDriver::block_size;
  // void (DirectDriver::*set_copy_buffer_size)(size_t) = &DirectDriver::copy_buffer_size;
  // size_t(DirectDriver::*get_copy_buffer_size)() const = &DirectDriver::copy_buffer_size;
  // class_<DirectDriver,bases<Driver>>("DirectDriver")
  // 	   .def(init<size_t,size_t,size_t>((arg("alignment")=0,
  // 				    arg("block_size")=0,
  // 				    arg("copy_buffer_size")=0)))
  // 	   .add_property("alignment",get_alignment,set_alignment)
  // 	   .add_property("block_size",get_block_size,set_block_size)
  // 	   .add_property("copy_buffer_size",get_copy_buffer_size,set_copy_buffer_size)
          ;

  void (MemoryDriver::*set_increment)(size_t) = &MemoryDriver::increment;
  size_t(MemoryDriver::*get_increment)() const = &MemoryDriver::increment;
  void (MemoryDriver::*set_backing_store)(bool) = &MemoryDriver::backing_store;
  bool(MemoryDriver::*get_backing_store)() const = &MemoryDriver::backing_store;
  class_<MemoryDriver,bases<Driver>>("MemoryDriver")
	   .def(init<size_t,bool>((arg("increment")=0,
				    arg("backing_store")=0)))
	   .add_property("increment",get_increment,set_increment)
	   .add_property("backing_store",get_backing_store,set_backing_store)
          ;

  // usage of MPI in python bindings not supported
  // class_<MPIDriver,bases<Driver>>("MPIDriver");
	   
  //need some functions
  def("is_hdf5_file",&is_hdf5_file);
  def("create",&create_file,(arg("file"),arg("flags")=AccessFlags::Exclusive,
                             arg("fcpl")=hdf5::property::FileCreationList(),
                             arg("fapl")=hdf5::property::FileAccessList()));
  def("open",&open_file,(arg("file"),arg("flags")=AccessFlags::ReadOnly,
                         arg("fapl")=hdf5::property::FileAccessList()));
  def("from_buffer",&from_buffer_,(arg("data"),arg("flags")=ImageFlags::ReadOnly));
}
