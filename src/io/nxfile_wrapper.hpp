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
#include <pni/io/nx/nxobject_traits.hpp>

//! 
//! \ingroup wrappers
//! \brief NXFile wrapper template
//! 
//! This template can be used to create NXFile wrappers.
//!
template<typename FTYPE> class nxfile_wrapper
{
    public:
        typedef FTYPE file_type;
        static const pni::io::nx::nximp_code imp_id = 
                     pni::io::nx::nximp_code_map<FTYPE>::icode;
        typedef typename pni::io::nx::nxobject_trait<imp_id>::object_type 
                         object_type;
        typedef nxfile_wrapper<file_type> wrapper_type;
    private:
        file_type _file;
    public:
        //==================constructor and destructor=========================
        //! default constructor
        nxfile_wrapper(){}

        //----------------------------------------------------------------------
        //! copy constructor
        nxfile_wrapper(const wrapper_type &o):_file(o._file){}

        //----------------------------------------------------------------------
        //! move constructor
        nxfile_wrapper(wrapper_type &&o):_file(std::move(o._file)){}

        //----------------------------------------------------------------------
        //! move conversion constructor from wrapped object
        explicit nxfile_wrapper(file_type &&file):_file(std::move(file)){}

        //----------------------------------------------------------------------
        //! copy conversion constructor from wrapped object
        explicit nxfile_wrapper(const file_type &file):_file(file){}

        //---------------------------------------------------------------------
        //! check read only status
        bool is_readonly() const { return _file.is_readonly(); }

        //---------------------------------------------------------------------
        //! flush data to disk
        void flush() const  { _file.flush(); }

        //---------------------------------------------------------------------
        //! check if file is valid
        bool is_valid() const { return _file.is_valid(); }

        //---------------------------------------------------------------------
        //! close the file
        void close() { _file.close(); }

        //---------------------------------------------------------------------
        object_type root() const { return _file.root(); }

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
template<typename FTYPE> 
nxfile_wrapper<FTYPE> create_file(const pni::core::string &n,bool ov)
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
template<typename FTYPE> 
nxfile_wrapper<FTYPE> open_file(const pni::core::string &n, bool ro)
{
    return nxfile_wrapper<FTYPE>(FTYPE::open_file(n,ro)); 
}

//------------------------------------------------------------------------------
//!
//! \ingroup wrappers
//! \brief create split files
//!
//! Create a new file in split mode. 
//! 
//! \throws file_error in case of errors
//! 
//! \param n name of the file
//! \param split_size the size at which to split files in MB
//! \param ow overwrite flag
//! \return new instance of nxfile
//!
template<typename FTYPE>
nxfile_wrapper<FTYPE> create_files(const pni::core::string &n,
                                   ssize_t split_size,
                                   bool ow)
{
    using namespace pni::core;

    try
    {
        return nxfile_wrapper<FTYPE>(FTYPE::create_files(n,split_size,ow));
    }
    catch(pni::core::file_error &error)
    {
        std::cerr<<error<<std::endl;
        error.append(EXCEPTION_RECORD);
        throw error;
    }
}

static const pni::core::string nxfile_flush_doc_string = 
"Flush the content of the file. \n"
"This method writes all the changes made to the file to disk."
;

static const pni::core::string nxfile_close_doc_string = 
"Closes the file.\n"
"\n"
"Other objects belonging to this file residing within the same scope\n"
"must be closed explicitely if the file should be reopened!\n"
;

static const pni::core::string nxfile_root_doc_string = 
"Return the root group of the file.\n"
"\n"
":return: the root group of the file\n"
":rtype: instance of :py:class:`nxgroup`\n"
;

static const pni::core::string nxfile_readonly_doc = 
"Property for file status\n"
"\n"
"If :py:const:`True` the file is in read only mode. \n";

static const pni::core::string nxfile_is_valid_doc = 
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
template<typename FTYPE> void wrap_nxfile()
{
    using namespace boost::python; 

    typedef typename nxfile_wrapper<FTYPE>::wrapper_type wrapper_type;

    class_<wrapper_type>("nxfile")
        .def(init<>())
        .add_property("readonly",&wrapper_type::is_readonly,nxfile_readonly_doc.c_str())
        .add_property("is_valid",&wrapper_type::is_valid,nxfile_is_valid_doc.c_str())
        .def("flush",&wrapper_type::flush,nxfile_flush_doc_string.c_str())
        .def("close",&wrapper_type::close,nxfile_close_doc_string.c_str())
        .def("root",&wrapper_type::root,nxfile_root_doc_string.c_str())
        ;

    //need some functions
    def("__create_file",&create_file<FTYPE>);
    def("__open_file",&open_file<FTYPE>);
    def("__create_files",&create_files<FTYPE>);
}

