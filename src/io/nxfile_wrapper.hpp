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
///

#pragma once

#include <pni/io/nx/nxobject_traits.hpp>

//! 
//! \ingroup wrappers
//! \brief NXFile wrapper template
//! 
//! This template can be used to create NXFile wrappers.
//!
using namespace pni::io::nx;

template<typename FTYPE> class nxfile_wrapper
{
    public:
        typedef FTYPE file_type;
        static const nximp_code imp_id = nximp_code_map<FTYPE>::icode;
        typedef typename nxobject_trait<imp_id>::object_type object_type;
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
nxfile_wrapper<FTYPE> create_file(const string &n,bool ov)
{
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
nxfile_wrapper<FTYPE> open_file(const string &n, bool ro)
{
    return nxfile_wrapper<FTYPE>(FTYPE::open_file(n,ro)); 
}

//------------------------------------------------------------------------------
//! 
//! \ingroup wrappers
//! \brief create NXFile wrapper 
//! 
//! Tempalte function creates a wrappers for the NXFile type FType. 
//!
template<typename FTYPE> void wrap_nxfile()
{
    typedef typename nxfile_wrapper<FTYPE>::wrapper_type wrapper_type;

    class_<wrapper_type>("nxfile")
        .def(init<>())
        .add_property("readonly",&wrapper_type::is_readonly)
        .add_property("is_valid",&wrapper_type::is_valid)
        .def("flush",&wrapper_type::flush)
        .def("close",&wrapper_type::close)
        .def("root",&wrapper_type::root)
        ;

    //need some functions
    def("__create_file",&create_file<FTYPE>);
    def("__open_file",&open_file<FTYPE>);
}

