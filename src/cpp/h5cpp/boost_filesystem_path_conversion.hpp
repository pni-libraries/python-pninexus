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
// Created on: Jan 23, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <boost/python.hpp>
#include <h5cpp/core/filesystem.hpp>

//!
//! @brief convert fs::path to a Python object
//!
//! Instances of fs::path are converted to a Python string
//!
struct BoostFilesystemPathToPythonObject
{
    BoostFilesystemPathToPythonObject();
    static PyObject *convert(const fs::path &path);
};

//!
//! @brief convert a Python string to boost::fileystem::path
//!
//! Python strings can be converted to an instance of fs::path
//! if required.
//!
struct PythonObjectToBoostFilesystemPath
{
    using rvalue_type = boost::python::converter::rvalue_from_python_stage1_data;
    using storage_type = boost::python::converter::rvalue_from_python_storage<fs::path>;

    PythonObjectToBoostFilesystemPath();

    static void *convertible(PyObject *ptr);

    static void construct(PyObject *ptr,rvalue_type *data);
};
