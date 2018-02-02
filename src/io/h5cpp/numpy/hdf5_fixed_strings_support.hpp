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
// Created on: Feb 31, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <h5cpp/hdf5.hpp>
#include "array_adapter.hpp"
#include "array_factory.hpp"

namespace numpy {

class FixedLengthStringBuffer
{
  private:
    bool  is_borrowed_;
    char *pointer_;
    size_t size_;
  public:
    FixedLengthStringBuffer():
      is_borrowed_(false),
      pointer_(nullptr),
      size_(0)
    {}
    explicit FixedLengthStringBuffer(size_t size):
        is_borrowed_(false),
        pointer_(nullptr),
        size_(size)
    {
      pointer_=new char[size];
    }

    explicit FixedLengthStringBuffer(char *ptr,size_t size):
        is_borrowed_(true),
        pointer_(ptr),
        size_(size)
    {}

    ~FixedLengthStringBuffer()
    {
      if(pointer_ && !is_borrowed_)
        delete pointer_;

      size_ = 0;
    }

    size_t size() const
    {
      return size_;
    }



    const char *data() const
    {
      return pointer_;
    }

    char *data()
    {
      return pointer_;
    }
};

} // namespace numpy

namespace hdf5 {

template<>
struct FixedLengthStringTrait<numpy::ArrayAdapter>
{
   using DataType = numpy::ArrayAdapter;
   using BufferType = numpy::FixedLengthStringBuffer;


   static BufferType to_buffer(const DataType &data,const datatype::String &file_type)
   {
     BufferType buffer((data.itemsize()+1)*data.size());

     NpyIter *iter = NpyIter_New(static_cast<PyArrayObject*>(data),
                                 NPY_ITER_READONLY | NPY_ITER_C_INDEX,
                                 NPY_CORDER , NPY_NO_CASTING,nullptr);
     NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter,nullptr);
     char **dataptr = NpyIter_GetDataPtrArray(iter);

     char *buffer_ptr = buffer.data();
     do
     {
       std::copy(*dataptr,*dataptr+data.itemsize()+1,buffer_ptr);

       buffer_ptr+=data.itemsize()+1;
     }while(iternext(iter));

     NpyIter_Deallocate(iter);


     return buffer;
   }

   static DataType from_buffer(const BufferType &buffer,const datatype::String &file_type)
   {
     numpy::Dimensions dims{buffer.size()/(file_type.size()+1)};

     numpy::ArrayAdapter adapter(reinterpret_cast<PyArrayObject*>(numpy::ArrayFactory::create_ptr(file_type,dims)));

     NpyIter *iter = NpyIter_New(static_cast<PyArrayObject*>(adapter),
                                 NPY_ITER_READWRITE | NPY_ITER_C_INDEX,
                                 NPY_CORDER , NPY_NO_CASTING,nullptr);
     NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter,nullptr);
     char **dataptr = NpyIter_GetDataPtrArray(iter);

     const char *buffer_ptr = buffer.data();
     do
     {
       std::copy(buffer_ptr,buffer_ptr+adapter.itemsize()+1,*dataptr);
       buffer_ptr+=adapter.itemsize()+1;
     }while(iternext(iter));

     //clear the iterator
     NpyIter_Deallocate(iter);

     return adapter;
   }
};

} // namespace hdf5
