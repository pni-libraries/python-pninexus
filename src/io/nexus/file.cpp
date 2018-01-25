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
#include <pni/core/types.hpp>
#include <pni/core/error.hpp>
#include <pni/io/nexus.hpp>

static const std::string nxfile_flush_doc_string = 
"Flush the content of the file. \n"
"This method writes all the changes made to the file to disk."
;

static const std::string nxfile_close_doc_string = 
"Closes the file.\n"
"\n"
"Other objects belonging to this file residing within the same scope\n"
"must be closed explicitely if the file should be reopened!\n"
;

static const std::string nxfile_root_doc_string = 
"Return the root group of the file.\n"
"\n"
":return: the root group of the file\n"
":rtype: instance of :py:class:`nxgroup`\n"
;

static const std::string nxfile_readonly_doc = 
"Property for file status\n"
"\n"
"If :py:const:`True` the file is in read only mode. \n";

static const std::string nxfile_is_valid_doc = 
"Property for object status\n"
"\n"
"If :py:const:`True` the object is a valid NeXus object, \n"
":py:const:`False` otherwise.\n";

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(root_overloads,root,0,0);

hdf5::file::File create_file(const boost::filesystem::path &file_path,
                             hdf5::file::AccessFlags flags)
{
  return pni::io::nexus::create_file(file_path,flags);
}

hdf5::file::File open_file(const boost::filesystem::path &file_path,
                           hdf5::file::AccessFlags flags)
{
  return pni::io::nexus::open_file(file_path,flags);
}

void wrap_file()
{
    using namespace boost::python; 

    enum_<hdf5::file::Scope>("Scope","The scope of a file")
        .value("LOCAL",hdf5::file::Scope::LOCAL)
        .value("GLOBAL",hdf5::file::Scope::GLOBAL);

    enum_<hdf5::file::AccessFlags>("AccessFlags","The access flags used to open the file")
        .value("TRUNCATE",hdf5::file::AccessFlags::TRUNCATE)
        .value("EXCLUSIVE",hdf5::file::AccessFlags::EXCLUSIVE)
        .value("READWRITE",hdf5::file::AccessFlags::READWRITE)
        .value("READONLY",hdf5::file::AccessFlags::READONLY);


    //hdf5::node::Group (hdf5::file::File::*root)() = &hdf5::file::File::root;
    class_<hdf5::file::File>("File")
        .def(init<>())
        .add_property("intent",&hdf5::file::File::intent,nxfile_readonly_doc.c_str())
        .add_property("is_valid",&hdf5::file::File::is_valid,nxfile_is_valid_doc.c_str())
        .add_property("path",&hdf5::file::File::path)
        .add_property("size",&hdf5::file::File::size)
        .def("flush",&hdf5::file::File::flush,(arg("scope")=hdf5::file::Scope::GLOBAL),nxfile_flush_doc_string.c_str())
        .def("close",&hdf5::file::File::close,nxfile_close_doc_string.c_str())
        .def("root",&hdf5::file::File::root,root_overloads())
        ;
    //need some functions
    def("is_nexus_file",&pni::io::nexus::is_nexus_file);
    def("create_file",&create_file,(arg("file"),arg("flags")=hdf5::file::AccessFlags::EXCLUSIVE));
    def("open_file",&open_file,(arg("file"),arg("flags")=hdf5::file::AccessFlags::READONLY));
}

