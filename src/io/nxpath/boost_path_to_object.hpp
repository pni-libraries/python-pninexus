//
// (c) Copyright 2018 DESY
//
// This file is part of python-pnicore.
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

#include <boost/python.hpp>
#include <boost/filesystem.hpp>

struct BoostPathToObject
{
    BoostPathToObject();
    static PyObject *convert(const boost::filesystem::path &path);
};

struct ObjectToBoostPath
{
    using rvalue_type = boost::python::converter::rvalue_from_python_stage1_data;
    using storage_type = boost::python::converter::rvalue_from_python_storage<boost::filesystem::path>;

    ObjectToBoostPath();

    static void *convertible(PyObject *ptr);

    static void construct(PyObject *ptr,rvalue_type *data);
};
