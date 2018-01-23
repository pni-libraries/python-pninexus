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

#include "boost_path_to_object.hpp"

using namespace boost::python;

BoostPathToObject::BoostPathToObject()
{
  to_python_converter<boost::filesystem::path,BoostPathToObject>();
}

PyObject *BoostPathToObject::convert(const boost::filesystem::path &path)
{
  return incref(object(path.string()).ptr());
}

ObjectToBoostPath::ObjectToBoostPath()
{
  converter::registry::push_back(&convertible,&construct,
                                 type_id<boost::filesystem::path>());
}

void *ObjectToBoostPath::convertible(PyObject *ptr)
{
#if PY_MAJOR_VERSION >= 3
  if(!PyUnicode_Check(ptr)) return nullptr;
#else
  if(!PyString_Check(ptr)) return nullptr;
#endif
}

void ObjectToBoostPath::construct(PyObject *ptr,rvalue_type *data)
{
  boost::python::str py_string(handle<>(borrowed(ptr)));

  void *storage = ((storage_type*)data)->storage.bytes;
  new (storage) boost::filesystem::path(extract<std::string>(py_string));
  data->convertible = storage;
}
