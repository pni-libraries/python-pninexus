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

#include "dimensions_conversion.hpp"

using namespace boost::python;

DimensionsToTuple::DimensionsToTuple()
{
  to_python_converter<hdf5::Dimensions,DimensionsToTuple>();
}

PyObject *DimensionsToTuple::convert(const hdf5::Dimensions &dimensions)
{
  if(dimensions.empty())
  {
    return incref(tuple().ptr());
  }
  else
  {
    list l;
    size_t index = 0;
    for(auto v: dimensions)
      l.insert(index++,v);

    return incref(tuple(l).ptr());
  }
}


PythonToDimensions::PythonToDimensions()
{
  converter::registry::push_back(&convertible,&construct,
                                 type_id<hdf5::Dimensions>());
}

void *PythonToDimensions::convertible(PyObject *ptr)
{
  if(PyTuple_Check(ptr) || PyList_Check(ptr))
    return ptr;
  else
    return nullptr;
}

void PythonToDimensions::construct(PyObject *ptr,rvalue_type *data)
{
  object sequence(handle<>(borrowed(ptr)));

  stl_input_iterator<hdf5::Dimensions::value_type> begin(sequence),end;

  void *storage = ((storage_type*)data)->storage.bytes;
  new (storage) hdf5::Dimensions(begin,end);
  data->convertible = storage;
}
