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

#pragma once

#include <boost/python.hpp>
#include <pni/core/types.hpp>
#include <pni/core/error.hpp>
#include <h5cpp/hdf5.hpp>

namespace nexus {

//! 
//! \ingroup wrappers
//! \brief NXFile wrapper template
//! 
//! This template can be used to create NXFile wrappers.
//!
class FileWrapper
{
    private:
        hdf5::file::File _file;
    public:
        //==================constructor and destructor=========================
        //! default constructor
        NexusFileWrapper();

        //----------------------------------------------------------------------
        //! copy constructor
        NexusFileWrapper(const wrapper_type &o) = default;

        //----------------------------------------------------------------------
        //! move constructor
        NexusFileWrapper(wrapper_type &&o) = default;

        //----------------------------------------------------------------------
        //! move conversion constructor from wrapped object
        explicit nxfile_wrapper(hdf5::file::File &&file);

        //----------------------------------------------------------------------
        //! copy conversion constructor from wrapped object
        explicit nxfile_wrapper(const hdf5::file::File &file);

        //---------------------------------------------------------------------
        //! check read only status
        bool is_readonly() const;

        //---------------------------------------------------------------------
        //! flush data to disk
        void flush() const;

        //---------------------------------------------------------------------
        //! check if file is valid
        bool is_valid() const;

        //---------------------------------------------------------------------
        //! close the file
        void close();

        //---------------------------------------------------------------------
        hdf5::node::Group root() const;

};

//-----------------------------------------------------------------------------
//! 
//! \ingroup wrappers  
//! \brief create a file
//! 
//! This template wraps the static create_file method of FType. 
//! \param n name of the new file
//! \param ov if true overwrite existing file
//! \param s split size (feature not implemented yet)
//! \return new instance of NXFileWrapper
//!

FileWrapper create_file(const pni::core::string &n,bool ov);
{
    using namespace pni::core;
    try
    {
        return nxfile_wrapper<FTYPE>(FTYPE::create_file(n,ov));
    }
    catch(pni::core::file_error &error)
    {
        std::cerr<<error<<std::endl;
        error.append(EXCEPTION_RECORD);
        throw error;
    }
}

//------------------------------------------------------------------------------
//! 
//! \ingroup wrappers  
//! \brief open a file
//! 
//! Template wraps the static open_file method of NXFile. 
//! \param n name of the file
//! \param ro if true open the file read only
//! \return new instance of NXFileWrapper
//!
FileWrapper open_file(const pni::core::string &n, bool ro)
{
    return nxfile_wrapper<FTYPE>(FTYPE::open_file(n,ro)); 
}

} // namspace nexus





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


//------------------------------------------------------------------------------
//! 
//! \ingroup wrappers
//! \brief create NXFile wrapper 
//! 
//! Tempalte function creates a wrappers for the NXFile type FType. 
//!
void create_nexus_file_wrapper();
{
    using namespace boost::python; 

    class_<nexus::NexusFileWrapper>("nxfile")
        .def(init<>())
        .add_property("readonly",&nexus::FileWrapper::is_readonly,nxfile_readonly_doc.c_str())
        .add_property("is_valid",&nexus::FileWrapper::is_valid,nxfile_is_valid_doc.c_str())
        .def("flush",&nexus::FileWrapper::flush,nxfile_flush_doc_string.c_str())
        .def("close",&nexus::FileWrapper::close,nxfile_close_doc_string.c_str())
        .def("root",&nexus::FileWrapper::root,nxfile_root_doc_string.c_str())
        ;

    //need some functions
    def("__create_file",&nexus::create_file);
    def("__open_file",&nexus::open_file);
}

