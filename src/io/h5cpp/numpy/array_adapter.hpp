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
// Created on: Feb 1, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once
#include <boost/python.hpp>
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
//! @brief numpy array adapter
//!
//! This stores a reference to a numpy array. It does not take ownership and
//! is considered an Adapter to an existing numpy array. The ownership management
//! is entirely on the original boost::python::object instance.
//!
class ArrayAdapter
{
  private:
   PyArrayObject *pointer_;
  public:
   ArrayAdapter();
   ArrayAdapter(const boost::python::object &object);
   ArrayAdapter(const ArrayAdapter &) = default;

   operator PyArrayObject* () const
   {
     return pointer_;
   }

   int type_number() const;
   npy_intp itemsize() const;
   hdf5::Dimensions dimensions() const;
   size_t size() const;

   void *data();
   const void *data() const;
};

//!
//! @brief check if object is numpy array
//!
//! Checks if an object is a numpy array.
//!
//! @param o const reference to a
//! @return true if object is a numpy array
//!
bool is_array(const boost::python::object &o);

} // namespace numpy
