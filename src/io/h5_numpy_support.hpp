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
#include <pni/io/nexus.hpp>
#include <pni/core/error.hpp>
#include <boost/python/extract.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PNI_CORE_USYMBOL
#define NO_IMPORT_ARRAY
extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

struct NumpyArray
{
    PyArrayObject *pointer;

    NumpyArray():pointer(nullptr) {}
    NumpyArray(PyArrayObject *ptr):pointer(ptr) {}

    operator PyArrayObject* ()
    {
      return pointer;
    }
};

namespace hdf5  {
namespace dataspace {
    
template<> class TypeTrait<NumpyArray>
{
  public:
    using DataspaceType = Simple;

    static DataspaceType create(const NumpyArray &array)
    {
      using Dimension = hdf5::Dimensions::value_type;
      hdf5::Dimensions current_dimensions(PyArray_NDIM(array.pointer));

      size_t dimension_index = 0;
      for(auto &dimension: current_dimensions)
        dimension = (Dimension)PyArray_DIM(array.pointer,dimension_index++);

      return Simple(current_dimensions);
    }

    static void *ptr(NumpyArray &array)
    {
      return (void*)PyArray_DATA(array.pointer);
    }

    static const void *cptr(const NumpyArray &array)
    {
      return (const void*)PyArray_DATA(const_cast<PyArrayObject*>(array.pointer));
    }

};
    
} // namespace dataspace

namespace datatype {

template<> class TypeTrait<NumpyArray>
{
  public:
    using TypeClass = Datatype;

    static TypeClass create(const NumpyArray &array)
    {
      using pni::io::nexus::DatatypeFactory;
      using namespace pni::core;

      int pytype = PyArray_TYPE(array.pointer);
      //select the data type to use for writing the array data
      switch(pytype)
      {
        case NPY_UINT8:  return DatatypeFactory::create(type_id_t::UINT8);
        case NPY_INT8:   return DatatypeFactory::create(type_id_t::INT8);
        case NPY_UINT16: return DatatypeFactory::create(type_id_t::UINT16);
        case NPY_INT16:  return DatatypeFactory::create(type_id_t::INT16);
        case NPY_UINT32: return DatatypeFactory::create(type_id_t::UINT32);
        case NPY_INT32:  return DatatypeFactory::create(type_id_t::INT32);
        case NPY_UINT64: return DatatypeFactory::create(type_id_t::UINT64);
        case NPY_INT64:  return DatatypeFactory::create(type_id_t::INT64);
#ifndef _MSC_VER
        case NPY_LONGLONG:  return DatatypeFactory::create(type_id_t::INT64);
        case NPY_ULONGLONG: return DatatypeFactory::create(type_id_t::UINT64);
#endif
        case NPY_FLOAT32:    return DatatypeFactory::create(type_id_t::FLOAT32);
        case NPY_FLOAT64:    return DatatypeFactory::create(type_id_t::FLOAT64);
        case NPY_LONGDOUBLE: return DatatypeFactory::create(type_id_t::FLOAT128);
        case NPY_COMPLEX64:  return DatatypeFactory::create(type_id_t::COMPLEX32);
        case NPY_CDOUBLE:    return DatatypeFactory::create(type_id_t::COMPLEX64);
        case NPY_CLONGDOUBLE: return DatatypeFactory::create(type_id_t::COMPLEX128);
        case NPY_BOOL:        return DatatypeFactory::create(type_id_t::BOOL);
#if PY_MAJOR_VERSION >= 3
        case NPY_UNICODE: return DatatypeFactory::create(type_id_t::STRING);
#else
        case NPY_STRING:  return DatatypeFactory::create(type_id_t::STRING);
#endif
        default:
          throw type_error(EXCEPTION_RECORD,
                           "Type of numpy array cannot be handled!");
      };
    }
};

} // namespace datatype

} // namespace hdf5
