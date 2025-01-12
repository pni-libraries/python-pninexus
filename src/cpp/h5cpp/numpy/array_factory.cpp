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
//             Jan Kotanski <jan.kotanski@desy.de>
//

#include "array_factory.hpp"
#include <cstdint>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PNI_CORE_USYMBOL
#define NO_IMPORT_ARRAY
extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}
#include <h5cpp/datatype/datatype.hpp>
#include <h5cpp/datatype/enum.hpp>
#include <h5cpp/contrib/nexus/ebool.hpp>

namespace {

int get_type_number(const hdf5::datatype::Datatype &datatype)
{
  using namespace hdf5::datatype;
  if     (datatype == create<uint8_t>()) return NPY_UINT8;
  else if(datatype == create<int8_t>()) return NPY_INT8;
  else if(datatype == create<uint16_t>()) return NPY_UINT16;
  else if(datatype == create<int16_t>()) return NPY_INT16;
  else if(datatype == create<uint32_t>()) return NPY_UINT32;
  else if(datatype == create<int32_t>())  return NPY_INT32;
  else if(datatype == create<uint64_t>()) return NPY_UINT64;
  else if(datatype == create<int64_t>())  return NPY_INT64;
  else if(datatype == create<float16_t>())    return NPY_FLOAT16;
  else if(datatype == create<float>())    return NPY_FLOAT;
  else if(datatype == create<double>())   return NPY_DOUBLE;
  else if(datatype == create<long double>()) return NPY_LONGDOUBLE;
  // else if(datatype == create<std::complex<float16_t>>())    return NPY_COMPLEX32;
  else if(datatype == create<std::complex<float>>())    return NPY_COMPLEX64;
  else if(datatype == create<std::complex<double>>())   return NPY_COMPLEX128;
  else if(datatype == create<std::complex<long double>>()) return NPY_COMPLEX256;
  else if(datatype.get_class() == Class::String)
  {
    String string_type = datatype;
    if(string_type.is_variable_length())
      return NPY_OBJECT;
    else
    {
#if PY_MAJOR_VERSION >= 3
      //return NPY_UNICODE;
    	return NPY_STRING;
#else
      return NPY_STRING;
#endif
    }
  }
  else if(datatype.get_class() == Class::Enum)
  {
    auto etype = hdf5::datatype::Enum(datatype);
    if(hdf5::datatype::is_bool(etype)){
      return NPY_BOOL;
    }
    else{
      return NPY_INT64;
    }
  }
  else if(datatype == create<bool>()) return NPY_BOOL;
  else
    throw std::runtime_error("HDF5 datatype not supported by numpy!");

}

int get_element_size(const hdf5::datatype::Datatype &datatype)
{
  int element_size = 0;

  if(datatype.get_class() == hdf5::datatype::Class::String)
  {
    hdf5::datatype::String string_type(datatype);
    if(!string_type.is_variable_length())
      element_size = string_type.size();
  }

  return element_size;
}

}

namespace numpy {

boost::python::object
ArrayFactory::create(const hdf5::datatype::Datatype &datatype,
                     const numpy::Dimensions &dimensions)
{
  auto ptr = reinterpret_cast<PyObject*>(create_ptr(datatype,dimensions));
  boost::python::handle<> h(ptr);

  return boost::python::object(h);
}

PyObject *ArrayFactory::create_ptr(const hdf5::datatype::Datatype &datatype,
                                        const numpy::Dimensions &dimensions)
{
  return PyArray_New(&PyArray_Type,
                     dimensions.ndims(),
                     const_cast<npy_intp*>(dimensions.dims()),
                     get_type_number(datatype),
                     nullptr,
                     nullptr,
                     get_element_size(datatype),
                     NPY_CORDER,
                     nullptr);
}

boost::python::object
ArrayFactory::create(const hdf5::datatype::Datatype &datatype,
                     const hdf5::dataspace::Dataspace &dataspace)
{
  Dimensions dims{1};
  if(dataspace.type() == hdf5::dataspace::Type::Simple)
    dims = Dimensions(hdf5::dataspace::Simple(dataspace).current_dimensions());


  return create(datatype,dims);
}

boost::python::object
ArrayFactory::create(const hdf5::datatype::Datatype &datatype,
                     const hdf5::dataspace::Selection &selection)
{
  Dimensions dims(selection);
  return create(datatype,dims);
}

boost::python::object ArrayFactory::create(const boost::python::object &object)
{

  PyObject *ptr = PyArray_FROM_OF(object.ptr(),NPY_ARRAY_C_CONTIGUOUS |
                                               NPY_ARRAY_ENSUREARRAY |
                                               NPY_ARRAY_ENSURECOPY );

  boost::python::handle<> h(ptr);
  return boost::python::object(h);
}

} // namespace numpy
