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
// Created on: Feb 31, 2018
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

class Dimensions : public std::vector<npy_intp>
{
  private:
    using Base = std::vector<npy_intp>;
  public:
    using std::vector<npy_intp>::vector;
    using std::vector<npy_intp>::operator=;

    explicit Dimensions(const hdf5::Dimensions &dims);
    explicit Dimensions(const hdf5::dataspace::Selection &selection);

    const value_type* dims() const;

    int ndims() const;

    explicit operator hdf5::Dimensions();
};


} // namespace numpy
