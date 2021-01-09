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
    owner_(false),
    pointer_(nullptr)
{}

ArrayAdapter::ArrayAdapter(const boost::python::object &object):
    owner_(false),
    pointer_(nullptr)
{
  if(!is_array(object))
  {
    throw std::runtime_error("Object is not a numpy array");
  }
  pointer_ = (PyArrayObject*)object.ptr();
}

ArrayAdapter::ArrayAdapter(PyArrayObject *ptr):
    owner_(true),
    pointer_(ptr)
{}

ArrayAdapter::~ArrayAdapter()
{
  if(owner_)
    Py_XDECREF(pointer_);
}

ArrayAdapter::ArrayAdapter(const ArrayAdapter &adapter):
      owner_(adapter.owner_),
      pointer_(adapter.pointer_)
{
  //increment the reference counter of the new adapter if we hold ownership
  if(owner_)
    Py_XINCREF(pointer_);
}

ArrayAdapter &ArrayAdapter::operator=(const ArrayAdapter &adapter)
{
  if(this == &adapter) return *this;

  //first decrement the reference count if there is one
  if(owner_)
    Py_XDECREF(pointer_);

  owner_ = adapter.owner_;
  pointer_ = adapter.pointer_;

  //increment the reference counter of the new adapter if we hold ownership
  if(owner_)
    Py_XINCREF(pointer_);

  return *this;
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
  size_t ndims = PyArray_NDIM(pointer_);
  if(ndims==0)
    return hdf5::Dimensions{1};

  hdf5::Dimensions dims(ndims);

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
