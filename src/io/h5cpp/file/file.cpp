//
// (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Feb 17, 2012
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>

using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(root_overloads,root,0,0);

hdf5::file::File create_file(const boost::filesystem::path &file_path,
                             hdf5::file::AccessFlags flags)
{
  return hdf5::file::create(file_path,flags);
}

hdf5::file::File open_file(const boost::filesystem::path &file_path,
                           hdf5::file::AccessFlags flags)
{
  return hdf5::file::open(file_path,flags);
}

BOOST_PYTHON_MODULE(_file)
{
  using namespace hdf5::file;

  //
  // setting up the documentation options
  //
  docstring_options doc_opts;
  doc_opts.disable_signatures();
  doc_opts.enable_user_defined();

  enum_<Scope>("Scope","The scope of a file")
            .value("LOCAL",Scope::LOCAL)
            .value("GLOBAL",Scope::GLOBAL);

  enum_<AccessFlags>("AccessFlags","The access flags used to open the file")
            .value("TRUNCATE",AccessFlags::TRUNCATE)
            .value("EXCLUSIVE",AccessFlags::EXCLUSIVE)
            .value("READWRITE",AccessFlags::READWRITE)
#ifdef H5_VERSION_GE(10.0.0)
            .value("SWMRREAD",AccessFlags::SWMR_READ)
            .value("SWMRWRITE",AccessFlags::SWMR_WRITE)
#endif
            .value("READONLY",AccessFlags::READONLY);


  //hdf5::node::Group (hdf5::file::File::*root)() = &hdf5::file::File::root;
  class_<File>("File")
            .def(init<>())
            .def(init<const File>())
            .add_property("intent",&File::intent)
            .add_property("is_valid",&File::is_valid)
            .add_property("path",&File::path)
            .add_property("size",&File::size)
            .def("flush",&File::flush,(arg("scope")=Scope::GLOBAL))
            .def("close",&File::close)
            .def("root",&File::root,root_overloads())
            ;

  //need some functions
  def("is_hdf5_file",&is_hdf5_file);
  def("create",&create_file,(arg("file"),arg("flags")=AccessFlags::EXCLUSIVE));
  def("open",&open_file,(arg("file"),arg("flags")=AccessFlags::READONLY));
}

