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
// along with python-pni.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Jan 23, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <h5cpp/hdf5.hpp>

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
    FileWrapper() = default;

    //----------------------------------------------------------------------
    //! copy constructor
    FileWrapper(const FileWrapper &o) = default;

    //----------------------------------------------------------------------
    //! move constructor
    FileWrapper(FileWrapper &&o) = default;

    //----------------------------------------------------------------------
    //! move conversion constructor from wrapped object
    explicit FileWrapper(hdf5::file::File &&file);

    //----------------------------------------------------------------------
    //! copy conversion constructor from wrapped object
    explicit FileWrapper(const hdf5::file::File &file);

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
FileWrapper open_file(const pni::core::string &n, bool ro);

//------------------------------------------------------------------------------
//!
//! \ingroup wrappers
//! \brief create NXFile wrapper
//!
//! Tempalte function creates a wrappers for the NXFile type FType.
//!
void wrap_nxfile(const char *class_name);
