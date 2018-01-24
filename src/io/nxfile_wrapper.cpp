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

#include "nxfile_wrapper.hpp"


FileWrapper::FileWrapper(hdf5::file::File &&file):
             _file(std::move(file))
{}

FileWrapper::FileWrapper(const hdf5::file::File &file):
            _file(file)
{}


bool FileWrapper::is_readonly() const
{
  hdf5::file::AccessFlags flags = _file.intent();
  if(flags == hdf5::file::AccessFlags::READONLY)
    return true;
  else
    return false;
}

void FileWrapper::flush() const
{
  _file.flush(hdf5::file::Scope::GLOBAL);
}

//---------------------------------------------------------------------
//! check if file is valid
bool FileWrapper::is_valid() const
{
  return _file.is_valid();
}

//---------------------------------------------------------------------
//! close the file
void FileWrapper::close()
{
  _file.close();
}

//---------------------------------------------------------------------
hdf5::node::Group FileWrapper::root() const
{
  return _file.root();
}

FileWrapper create_file(const pni::core::string &n,bool ov)
{
  hdf5::file::AccessFlags flags = hdf5::file::AccessFlags::EXCLUSIVE;

  if(ov)
    flags = hdf5::file::AccessFlags::TRUNCATE;

  try
  {
    return FileWrapper(hdf5::file::create(n,flags));
  }
  catch(std::runtime_error &error)
  {
    pni::core::file_error e(EXCEPTION_RECORD,error.what());
    throw e;
  }
}

FileWrapper open_file(const pni::core::string &n, bool ro)
{
  hdf5::file::AccessFlags flags = hdf5::file::AccessFlags::READONLY;

  if(!ro)
    flags = hdf5::file::AccessFlags::READWRITE;

  return FileWrapper(hdf5::file::open(n,flags));
}


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


void wrap_nxfile(const char *class_name)
{
    using namespace boost::python; 

    class_<FileWrapper>(class_name)
        .def(init<>())
        .add_property("readonly",&FileWrapper::is_readonly,nxfile_readonly_doc.c_str())
        .add_property("is_valid",&FileWrapper::is_valid,nxfile_is_valid_doc.c_str())
        .def("flush",&FileWrapper::flush,nxfile_flush_doc_string.c_str())
        .def("close",&FileWrapper::close,nxfile_close_doc_string.c_str())
        .def("root",&FileWrapper::root,nxfile_root_doc_string.c_str())
        ;

    //need some functions
    def("__create_file",&create_file);
    def("__open_file",&open_file);
}

