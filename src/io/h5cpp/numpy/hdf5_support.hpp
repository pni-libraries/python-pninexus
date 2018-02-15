//
// (c) Copyright 2015 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
//
// This file is part of python-pniio.
//
// python-pniio is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pniio is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Oct 8, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <h5cpp/hdf5.hpp>
#include <cstdint>
#include "array_adapter.hpp"

namespace hdf5  {
namespace dataspace {
    
template<> class TypeTrait<numpy::ArrayAdapter>
{
  public:
    using DataspaceType = Simple;

    static DataspaceType create(const numpy::ArrayAdapter &array)
    {
      return Simple(array.dimensions());
    }

    static void *ptr(numpy::ArrayAdapter &array)
    {
      return array.data();
    }

    static const void *cptr(const numpy::ArrayAdapter &array)
    {
      return array.data();
    }

};
    
} // namespace dataspace

namespace datatype {

template<> class TypeTrait<numpy::ArrayAdapter>
{
  public:
    using TypeClass = Datatype;

    static TypeClass create(const numpy::ArrayAdapter &array)
    {
      switch(array.type_number())
      {
        case NPY_INT8: return hdf5::datatype::create<std::int8_t>();
        case NPY_UINT8: return hdf5::datatype::create<std::uint8_t>();
        case NPY_INT16: return hdf5::datatype::create<std::int16_t>();
        case NPY_UINT16: return hdf5::datatype::create<std::uint16_t>();
        case NPY_INT32: return hdf5::datatype::create<std::int32_t>();
        case NPY_UINT32: return hdf5::datatype::create<std::uint32_t>();
        case NPY_INT64: return hdf5::datatype::create<std::int64_t>();
        case NPY_UINT64: return hdf5::datatype::create<std::uint64_t>();
        case NPY_FLOAT: return hdf5::datatype::create<float>();
        case NPY_DOUBLE: return hdf5::datatype::create<double>();
        case NPY_LONGDOUBLE: return hdf5::datatype::create<long double>();
#if PY_MAJOR_VERSION >= 3
        case NPY_UNICODE:
          return hdf5::datatype::String::fixed(array.itemsize());
#else
        case NPY_STRING:
        {
          String type = String::fixed(array.itemsize());
          type.padding(StringPad::NULLPAD);
          return type;
        }
#endif
        default:
          throw std::runtime_error("Datatype not supported by HDF5!");
      }
    }
};

} // namespace datatype


} // namespace hdf5