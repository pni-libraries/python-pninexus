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

#include "boost_filesystem_path_conversion.hpp"

using namespace boost::python;

BoostFilesystemPathToPythonObject::BoostFilesystemPathToPythonObject()
{
  to_python_converter<fs::path,BoostFilesystemPathToPythonObject>();
}

PyObject *BoostFilesystemPathToPythonObject::convert(const fs::path &path)
{
  return incref(object(path.string()).ptr());
}

PythonObjectToBoostFilesystemPath::PythonObjectToBoostFilesystemPath()
{
  converter::registry::push_back(&convertible,&construct,
                                 type_id<fs::path>());
}

void *PythonObjectToBoostFilesystemPath::convertible(PyObject *ptr)
{
#if PY_MAJOR_VERSION >= 3
  if(!PyUnicode_Check(ptr)) return nullptr;
#else
  if(!PyString_Check(ptr)) return nullptr;
#endif

  return ptr;
}

void PythonObjectToBoostFilesystemPath::construct(PyObject *ptr,rvalue_type *data)
{
  boost::python::str py_string(handle<>(borrowed(ptr)));

  void *storage = ((storage_type*)data)->storage.bytes;
  std::string path = extract<std::string>(py_string);
  new (storage) fs::path(path);
  data->convertible = storage;
}
