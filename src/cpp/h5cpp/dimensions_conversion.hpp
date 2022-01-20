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
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once
#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>
#include <boost/python/stl_iterator.hpp>

//!
//! @brief convert hdf5::Dimensions to a Python tuple
//!
struct DimensionsToTuple
{
    DimensionsToTuple();
    static PyObject *convert(const hdf5::Dimensions &dimensions);
};

//!
//! @brief convert a tuple or list to hdf5::Dimensions
//!
//! Take a list or tuple and convert it to hdf5::Dimensions.
//!
struct PythonToDimensions
{
    using rvalue_type = boost::python::converter::rvalue_from_python_stage1_data;
    using storage_type = boost::python::converter::rvalue_from_python_storage<hdf5::Dimensions>;

    PythonToDimensions();

    static void *convertible(PyObject *ptr);

    static void construct(PyObject *ptr,rvalue_type *data);
};
