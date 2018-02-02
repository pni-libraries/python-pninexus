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
// along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Jan 25, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once
#include <boost/python.hpp>
#include <pni/core/types.hpp>
#include <h5cpp/hdf5.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PNI_CORE_USYMBOL
#define NO_IMPORT_ARRAY
extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

namespace numpy {





//!
//! @brief factory class for arrays
//!
class ArrayFactory
{
  public:
    static boost::python::object create(pni::core::type_id_t tid,
                                        const hdf5::Dimensions &dimensions,
                                        int itemsize=0);
    static boost::python::object create(const boost::python::list &list,
                                        pni::core::type_id_t tid,
                                        const hdf5::Dimensions &dimensions);
};

int to_numpy_type_id(pni::core::type_id_t tid);

pni::core::type_id_t to_pnicore_type_id(int numpy_id);

std::vector<std::string> to_string_vector(const ArrayAdapter &array);

//!
//! \brief convert an arbitrary python object to a numpy array
//!
//! Take an arbitrary Python object and convert it to a numpy array.
//! If the object is already a numpy array we do nothing and just
//! pass the object through. Otherwise the numpy C-API will try
//! to convert the object to a numpy array.
//!
//!
boost::python::object to_numpy_array(const boost::python::object &o);

hdf5::Dimensions get_dimensions(const hdf5::dataspace::Selection &selection);
hdf5::Dimensions::value_type get_size(const hdf5::Dimensions &dimensions);

//!
//! @brief check if object is numpy array
//!
//! Checks if an object is a numpy array.
//!
//! @param o const reference to a
//! @return true if object is a numpy array
//!
bool is_array(const boost::python::object &o);


//!
//! @brief check if object is a numpy scalar
//!
//! Return true if the object is a numpy scalar. False otherwise.
//! This function is quite similar to the general is_scalar function.
//! In fact, is_scalar is calling this function to check whether or not
//! an object is a numpy scalar.
//!
//! @param o const reference to a python object
//! @return result
//!
bool is_scalar(const boost::python::object &o);

} // namespace numpy
