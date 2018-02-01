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
#pragma once

#include <boost/python.hpp>
#include "hdf5_numpy.hpp"

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
      Base::push_back(const_cast<char*>(cache_.back().c_str()));
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

    NpyIter *iter = NpyIter_New(static_cast<PyArrayObject*>(data),NPY_ITER_READONLY | NPY_ITER_C_INDEX, NPY_CORDER , NPY_NO_CASTING,nullptr);
    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter,nullptr);
    char **dataptr = NpyIter_GetDataPtrArray(iter);
    do
    {
      std::cout<<*dataptr<<"\t"<<*strideptr<<"\t"<<*innersizeptr<<std::endl;
      buffer.push_back(std::string(*dataptr,itemsize));
    }while(iternext(iter));
    NpyIter_Deallocate(iter);
    return buffer;
  }

  static void from_buffer(const BufferType &buffer,DataType &data)
  {
//    std::transform(buffer.begin(),buffer.end(),data.begin(),
//                   [](const char *ptr)
//                   {
//                     return std::string(ptr,std::strlen(ptr));
//                   });
  }
};


} // namespace hdf5
