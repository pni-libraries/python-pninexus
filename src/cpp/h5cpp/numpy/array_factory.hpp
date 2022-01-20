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

#include "dimensions.hpp"

namespace numpy {

class ArrayFactory

{
  private:
    static boost::python::object create(const hdf5::datatype::Datatype &datatype,
                                        const numpy::Dimensions &dimensions);
  public:

    static PyObject *create_ptr(const hdf5::datatype::Datatype &datatype,
                                        const numpy::Dimensions &dimensions);

    static boost::python::object create(const hdf5::datatype::Datatype &datatype,
                                        const hdf5::dataspace::Dataspace &dataspace);

    static boost::python::object create(const hdf5::datatype::Datatype &datatype,
                                        const hdf5::dataspace::Selection &selection);

    static boost::python::object create(const boost::python::object &object);
};

} // namespace numpy
