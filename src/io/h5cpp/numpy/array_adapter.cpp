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

#include "array_adapter.hpp"

namespace numpy {

//------------------------------------------------------------------------
bool is_array(const boost::python::object &o)
{
    //if the object is not allocated we assume that it is not an array
    if(o.ptr())
        return PyArray_CheckExact(o.ptr());
    else
        return false;
}

ArrayAdapter::ArrayAdapter():
    pointer_(nullptr)
{}

ArrayAdapter::ArrayAdapter(const boost::python::object &object):
    pointer_(nullptr)
{
  if(!is_array(object))
  {
    throw std::runtime_error("Object is not a numpy array");
  }
  pointer_ = (PyArrayObject*)object.ptr();
}

int ArrayAdapter::type_number() const
{
  return PyArray_TYPE(pointer_);
}

npy_intp ArrayAdapter::itemsize() const
{
  return PyArray_ITEMSIZE(pointer_);
}

hdf5::Dimensions ArrayAdapter::dimensions() const
{
  hdf5::Dimensions dims(PyArray_NDIM(pointer_));

  size_t index=0;
  for(auto &dim: dims)
    dim = (size_t)PyArray_DIM(pointer_,index++);

  return dims;
}

void *ArrayAdapter::data()
{
  return (void*)PyArray_DATA(pointer_);
}

const void *ArrayAdapter::data() const
{
  return (const void*)PyArray_DATA(pointer_);
}

size_t ArrayAdapter::size() const
{
  return PyArray_SIZE(pointer_);
}

} // namespace numpy
