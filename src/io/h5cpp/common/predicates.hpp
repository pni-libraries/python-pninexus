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
// Created on: Jan 30, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <boost/python.hpp>


//!
//! @brief return ture if object is a primitive float object
//!
//! @param object reference to the python object
//! @return true if float, false otherwise
bool is_float(const boost::python::object &object);


bool is_string(const boost::python::object &object);

bool is_integer(const boost::python::object &object);

bool is_numpy_array(const boost::python::object &object);

bool is_numpy_scalar(const boost::python::object &object);
