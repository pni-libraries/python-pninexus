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
// Created on: Feb 31, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>
#include "array_adapter.hpp"
#include "array_factory.hpp"

namespace hdf5 {

template<>
struct FixedLengthStringTrait<numpy::ArrayAdapter>
{
   using DataType = numpy::ArrayAdapter;
   using BufferType = FixedLengthStringBuffer<char>;


   static BufferType to_buffer(const DataType &data,
                               const datatype::String &memory_type,
                               const dataspace::Dataspace &memory_space)
   {
     BufferType buffer= BufferType::create(memory_type,memory_space);

     NpyIter *iter = NpyIter_New(static_cast<PyArrayObject*>(data),
                                 NPY_ITER_READONLY | NPY_ITER_C_INDEX,
                                 NPY_CORDER , NPY_NO_CASTING,nullptr);
     NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter,nullptr);
     char **dataptr = NpyIter_GetDataPtrArray(iter);

     char *buffer_ptr = buffer.data();
     do
     {
       std::copy(*dataptr,*dataptr+data.itemsize(),buffer_ptr);

       buffer_ptr+=data.itemsize();
     }while(iternext(iter));

     NpyIter_Deallocate(iter);


     return buffer;
   }

   static DataType from_buffer(const BufferType &buffer,
                               const datatype::String &memory_type,
                               const dataspace::Dataspace &memory_space)
   {
     numpy::Dimensions dims{1};
     if(memory_space.type()==dataspace::Type::Simple)
       dims = numpy::Dimensions(dataspace::Simple(memory_space).current_dimensions());

     numpy::ArrayAdapter adapter(reinterpret_cast<PyArrayObject*>(numpy::ArrayFactory::create_ptr(memory_type,dims)));

     NpyIter *iter = NpyIter_New(static_cast<PyArrayObject*>(adapter),
                                 NPY_ITER_READWRITE | NPY_ITER_C_INDEX,
                                 NPY_CORDER , NPY_NO_CASTING,nullptr);
     NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter,nullptr);
     char **dataptr = NpyIter_GetDataPtrArray(iter);

     const char *buffer_ptr = buffer.data();
     do
     {
       std::copy(buffer_ptr,buffer_ptr+adapter.itemsize(),*dataptr);
       buffer_ptr+=adapter.itemsize();
     }while(iternext(iter));

     //clear the iterator
     NpyIter_Deallocate(iter);

     return adapter;
   }
};

} // namespace hdf5
