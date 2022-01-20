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
// Created on: Feb 1, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once
#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>

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
    //!
    //! set to true if we hold the ownership of the array
    //!
    bool          owner_;

    //!
    //! pointer to the numpy array
    //!
    PyArrayObject *pointer_;
  public:
    //!
    //! @brief default constructor
    //!
    ArrayAdapter();

    //!
    //! @brief constructor
    //!
    //! This constructor fetches the pointer from an instance of boost::python::object.
    //! In this case the ownership over the array is managed by the original
    //! instance of boost::python::object.
    //!
    //! @param object reference to a boost::python::object
    //!
    explicit ArrayAdapter(const boost::python::object &object);

    //!
    //! @brief constructor
    //!
    //! This constructor takes ownership over the array.
    //!
    //! @param ptr pointer to the original array instance
    //!
    explicit ArrayAdapter(PyArrayObject *ptr);

    //!
    //! @brief destructor
    //!
    ~ArrayAdapter();

    //!
    //! @brief copy constructor
    //!
    ArrayAdapter(const ArrayAdapter &);

    //!
    //! @brief copy assignment operator
    //!
    ArrayAdapter &operator=(const ArrayAdapter &adapter);

    //!
    //! @brief conversion to a PyArrayObject
    //!
    //! This should allow for easy access to the internal pointer of the array.
    //! In order to access the pointer an explicit call to static_cast is
    //! required.
    //!
    //! Be aware if you are doing this you are on your own with
    //! ownership management. So the best is to use this only in the case
    //! of operations which do not change the reference count of the object.
    //!
    explicit operator PyArrayObject* () const
    {
      return pointer_;
    }

    explicit operator boost::python::object () const
    {
      //
      // we have to pump the reference counter in any case here as someone
      // owns the object (either we do or another instance of boost::python::object.
      //
      Py_XINCREF(pointer_);

      boost::python::handle<> h(reinterpret_cast<PyObject*>(pointer_));
      return boost::python::object(h);
    }

    //!
    //! @brief return the Numpy type number for the elements
    //!
    //!
    //! @return Numpy type number
    //!
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
