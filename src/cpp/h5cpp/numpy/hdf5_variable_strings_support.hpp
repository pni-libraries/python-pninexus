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
//             Jan Kotanski <jan.kotanski@desy.de>
//
#pragma once

#include <boost/python.hpp>
#include "array_adapter.hpp"

namespace numpy {

class VarLengthStringBuffer : public hdf5::VarLengthStringBuffer<char>
{
  public:
    using Base  = hdf5::VarLengthStringBuffer<char>;
    using Cache = std::vector<std::string>;

    using hdf5::VarLengthStringBuffer<char>::VarLengthStringBuffer;

    void push_back(const std::string &data)
    {
      cache_.push_back(data);
    }

    const Cache &cache() const
    {
      return cache_;
    }
    
    void commit_from_cache()
    {
      std::for_each(cache_.begin(),cache_.end(),
      [this](const std::string &str) { Base::push_back(const_cast<char*>(str.c_str()));});
    }

    void commit_to_data_to_cache()
    {

    }

  private:
    Cache cache_;
};

}

namespace hdf5 {

template<>
struct VarLengthStringTrait<numpy::ArrayAdapter>
{
  using BufferType = numpy::VarLengthStringBuffer;
  using DataType = numpy::ArrayAdapter;

  static BufferType to_buffer(const DataType &data)
  {
    BufferType buffer;
    npy_intp itemsize = PyArray_ITEMSIZE(static_cast<PyArrayObject*>(data));

    NpyIter *iter = NpyIter_New(static_cast<PyArrayObject*>(data),
                                NPY_ITER_READONLY | NPY_ITER_C_INDEX,
                                NPY_CORDER , NPY_NO_CASTING,nullptr);
    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter,nullptr);
    char **dataptr = NpyIter_GetDataPtrArray(iter);
    do
    {
      std::string value = std::string(*dataptr,itemsize);
      buffer.push_back(value);
      
    }while(iternext(iter));
    
    buffer.commit_from_cache();
    NpyIter_Deallocate(iter);
    return buffer;
  }

  static void from_buffer(const BufferType &buffer,DataType &data)
  {
    PyArray_Descr *dtype = PyArray_DescrFromType(NPY_OBJECT);
    NpyIter *iter = NpyIter_New(static_cast<PyArrayObject*>(data),
                                NPY_ITER_READWRITE | NPY_ITER_C_INDEX | NPY_ITER_REFS_OK,
                                NPY_CORDER , NPY_NO_CASTING,nullptr);
    if(iter==NULL)
    {
      Py_XDECREF(dtype);
      std::cerr<<"Could not instantiate an iterator for the array!"<<std::endl;
      PyErr_Print();
      return;
    }
    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter,nullptr);
    if(iternext == NULL)
    {
      Py_XDECREF(dtype);
      std::cerr<<"Could not instantiate next iterator function"<<std::endl;
      return;
    }
    PyObject ***dataptr = (PyObject***)NpyIter_GetDataPtrArray(iter);
    for(auto string: buffer)
    {
      if(string == NULL)
	{
	  char empty[] = {'\0'};
	  string = empty;
	}
#if PY_MAJOR_VERSION >= 3
      PyObject *ptr = PyUnicode_FromString(string);
#else
      PyObject *ptr = PyString_FromString(string);
#endif
      if(ptr==NULL)
      {
        std::cerr<<"could not create python string!"<<std::endl;
      }

      dataptr[0][0] = ptr;
      iternext(iter);
    }
    //PyArray_INCREF(static_cast<PyArrayObject*>(data));
    NpyIter_Deallocate(iter);
    Py_XDECREF(dtype);
  }
};


} // namespace hdf5
