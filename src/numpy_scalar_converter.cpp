//
// (c) Copyright 2014 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
//
// This file is part of python-pnicore.
//
// python-pnicore is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pnicore is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pnicore.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Oct 21, 2014
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include <pni/core/types.hpp>
#include "numpy_scalar_converter.hpp"

//converter namespace
using namespace pni::core;
using namespace boost::python;
namespace convns = boost::python::converter; 

numpy_scalar_converter::numpy_scalar_converter()
{
    convns::registry::push_back(
    &convertible,&construct<uint8>,boost::python::type_id<uint8>());
    convns::registry::push_back(
    &convertible,&construct<int8>,boost::python::type_id<int8>());
    convns::registry::push_back(
    &convertible,&construct<uint16>,boost::python::type_id<uint16>());
    convns::registry::push_back(
    &convertible,&construct<int16>,boost::python::type_id<int16>());
    convns::registry::push_back(
    &convertible,&construct<uint32>,boost::python::type_id<uint32>());
    convns::registry::push_back(
    &convertible,&construct<int32>,boost::python::type_id<int32>());
    convns::registry::push_back(
    &convertible,&construct<uint64>,boost::python::type_id<uint64>());
    convns::registry::push_back(
    &convertible,&construct<int64>,boost::python::type_id<int64>());

    convns::registry::push_back(
    &convertible,&construct<float32>,boost::python::type_id<float32>());
    convns::registry::push_back(
    &convertible,&construct<float64>,boost::python::type_id<float64>());
    convns::registry::push_back(
    &convertible,&construct<float128>,boost::python::type_id<float128>());
    
    convns::registry::push_back(
    &convertible,&construct<complex32>,boost::python::type_id<complex32>());
    convns::registry::push_back(
    &convertible,&construct<complex64>,boost::python::type_id<complex64>());
    convns::registry::push_back(
    &convertible,&construct<complex128>,boost::python::type_id<complex128>());
    
    convns::registry::push_back(
    &convertible,&construct<bool_t>,boost::python::type_id<bool_t>());
    
}

//----------------------------------------------------------------------------
void* numpy_scalar_converter::convertible(PyObject *obj_ptr)
{
    if(!PyArray_CheckScalar(obj_ptr)) return nullptr;
    return obj_ptr;
}
   
