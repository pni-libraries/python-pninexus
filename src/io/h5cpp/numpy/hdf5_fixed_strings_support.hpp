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
  public:
    FixedLengthStringBuffer():
      is_borrowed_(false),
      pointer_(nullptr)
    {}
    explicit FixedLengthStringBuffer(size_t size):
        is_borrowed_(false),
        pointer_(nullptr)
    {
      pointer_=new char[size];
    }

    explicit FixedLengthStringBuffer(char *ptr):
        is_borrowed_(true),
        pointer_(ptr)
    {}

    ~FixedLengthStringBuffer()
    {
      if(pointer_ && !is_borrowed_)
        delete pointer_;
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
     return BufferType(reinterpret_cast<char*>(const_cast<void*>(data.data())));
   }

   static DataType from_buffer(const BufferType &buffer,const datatype::String &file_type)
   {
     return DataType();
   }
};

} // namespace hdf5
