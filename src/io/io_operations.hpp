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

#include <boost/python.hpp>
#include <boost/python/slice.hpp>
#include <pni/core/types.hpp>
#include <pni/core/arrays.hpp>
#include <pni/io/exceptions.hpp>

#include <core/utils.hpp>
#include <core/numpy_utils.hpp>

boost::python::object get_first_element(const boost::python::object &o);


template<typename RTYPE> 
boost::python::object read_string_data(const RTYPE &readable)
{
    using namespace pni::core;
    using namespace boost::python;

    //
    // first read all data to a pni::core::mdarray array
    //
    auto shape = readable.template shape<shape_t>();
    auto data = dynamic_array<string>::create(shape);
    readable.read(data);

    //
    // copy the content of the mdarray to a Python list 
    //
    list l;
    for(auto s: data)
        l.append(s);

    //
    // create a numpy array from the list of strings
    //
    std::vector<npy_intp> dims(shape.size());
    std::copy(shape.begin(),shape.end(),dims.begin());

    PyArray_Dims d;
    d.ptr = dims.data();
    d.len = dims.size();
    PyObject *orig_ptr = PyArray_ContiguousFromAny(l.ptr(),
                                              numpy::type_id2numpy_id.at(type_id_t::STRING),
                                              1,2);

    //
    // The resulting numpy array has the wrong shape - we fix this here
    //
    PyObject *ptr = PyArray_Newshape(reinterpret_cast<PyArrayObject*>(orig_ptr),&d,NPY_CORDER);
    handle<> h(ptr);
    Py_XDECREF(orig_ptr);
    return object(h);
}
    
//----------------------------------------------------------------------------
template<typename RTYPE>
boost::python::object read_data(const RTYPE &readable)
{
    using namespace pni::core;
    using namespace boost::python;

    type_id_t tid = readable.type_id();
    size_t size = readable.size();

    object np_array;
    if(tid != type_id_t::STRING)
        np_array = numpy::create_array_from_field(readable);


    switch(tid)
    {
        case type_id_t::UINT8:
            readable.read(size,numpy::get_data<uint8>(np_array)); break;
        case type_id_t::INT8:
            readable.read(size,numpy::get_data<int8>(np_array)); break;
        case type_id_t::UINT16:
            readable.read(size,numpy::get_data<uint16>(np_array)); break;
        case type_id_t::INT16:
            readable.read(size,numpy::get_data<int16>(np_array)); break;
        case type_id_t::UINT32:
            readable.read(size,numpy::get_data<uint32>(np_array)); break;
        case type_id_t::INT32:
            readable.read(size,numpy::get_data<int32>(np_array)); break;
        case type_id_t::UINT64:
            readable.read(size,numpy::get_data<uint64>(np_array)); break;
        case type_id_t::INT64:
            readable.read(size,numpy::get_data<int64>(np_array)); break;
        case type_id_t::FLOAT32:
            readable.read(size,numpy::get_data<float32>(np_array)); break;
        case type_id_t::FLOAT64:
            readable.read(size,numpy::get_data<float64>(np_array)); break;
        case type_id_t::FLOAT128:
            readable.read(size,numpy::get_data<float128>(np_array)); break;
        case type_id_t::COMPLEX32:
            readable.read(size,numpy::get_data<complex32>(np_array)); break;
        case type_id_t::COMPLEX64:
            readable.read(size,numpy::get_data<complex64>(np_array)); break;
        case type_id_t::COMPLEX128:
            readable.read(size,numpy::get_data<complex128>(np_array)); break;
        case type_id_t::BOOL:
            readable.read(size,numpy::get_data<bool_t>(np_array)); break;
        case type_id_t::STRING:
            np_array = read_string_data(readable); break;
        default:
            throw value_error(EXCEPTION_RECORD,
                    "Unknow type_id value!");
    }

    return np_array;

    
}
