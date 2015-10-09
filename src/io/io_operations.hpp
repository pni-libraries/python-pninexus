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

template<typename WTYPE>
void write_string_data(const WTYPE &writeable,const boost::python::object &o)
{
    using namespace pni::core;
    using namespace boost::python;

    auto shape = numpy::get_shape<shape_t>(o);
    
    if(shape.empty() && numpy::get_size(o))
        shape = shape_t{1};

    auto data = dynamic_array<string>::create(shape);

    PyArrayObject *array_ptr = reinterpret_cast<PyArrayObject*>(o.ptr());
    PyObject *ptr = PyArray_Flatten(array_ptr,NPY_CORDER);
    handle<> h(PyArray_ToList(reinterpret_cast<PyArrayObject*>(ptr)));
    list l(h);
    
    size_t index=0;
    for(auto &s: data) s = extract<string>(l[index++]);

    //how to make copy more save 
    writeable.write(data);
}

//----------------------------------------------------------------------------
template<
         typename RTYPE,
         typename T
        >
struct reader
{
    static boost::python::object read(const RTYPE &readable)
    {
        using namespace boost::python;
        using namespace pni::core;

        object o = numpy::create_array_from_field(readable);
        readable.read(readable.size(),numpy::get_data<T>(o));
        return o;
    }
};

//----------------------------------------------------------------------------
template<typename RTYPE>
struct reader<RTYPE,pni::core::string>
{
    static pni::core::shape_t get_shape(const RTYPE &readable)
    {
        using namespace pni::core;
        using namespace boost::python;

        auto shape = readable.template shape<shape_t>();
        if(shape.empty() && readable.size()) shape=shape_t{1};

        return shape;
    }

    static boost::python::object read(const RTYPE &readable)
    {
        using namespace pni::core;
        using namespace boost::python;

        //
        // first read all data to a pni::core::mdarray array
        //
        auto shape = get_shape(readable);
        auto data = dynamic_array<string>::create(shape);
        readable.read(data);

        //
        // copy the content of the mdarray to a Python list 
        //
        list l;
        for(auto s: data) l.append(s);

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
};
    
//----------------------------------------------------------------------------
template<typename RTYPE>
boost::python::object read_data(const RTYPE &readable)
{
    using namespace pni::core;
    using namespace boost::python;

    type_id_t tid = readable.type_id();

    switch(tid)
    {
        case type_id_t::UINT8:   return reader<RTYPE,uint8>::read(readable);
        case type_id_t::INT8:    return reader<RTYPE,int8>::read(readable);
        case type_id_t::UINT16:  return reader<RTYPE,uint16>::read(readable); 
        case type_id_t::INT16:   return reader<RTYPE,int16>::read(readable);
        case type_id_t::UINT32:  return reader<RTYPE,uint32>::read(readable); 
        case type_id_t::INT32:   return reader<RTYPE,int32>::read(readable);
        case type_id_t::UINT64:  return reader<RTYPE,uint64>::read(readable); 
        case type_id_t::INT64:   return reader<RTYPE,int64>::read(readable); 
        case type_id_t::FLOAT32: return reader<RTYPE,float32>::read(readable); 
        case type_id_t::FLOAT64: return reader<RTYPE,float64>::read(readable);
        case type_id_t::FLOAT128: 
            return reader<RTYPE,float128>::read(readable);
        case type_id_t::COMPLEX32:
            return reader<RTYPE,complex32>::read(readable); 
        case type_id_t::COMPLEX64:
            return reader<RTYPE,complex64>::read(readable); 
        case type_id_t::COMPLEX128:
            return reader<RTYPE,complex128>::read(readable);
        case type_id_t::BOOL:
            return reader<RTYPE,bool_t>::read(readable); 
        case type_id_t::STRING:
            return reader<RTYPE,string>::read(readable);
        default:
            throw value_error(EXCEPTION_RECORD,"Unknow type_id value!");
            return object();
    }
}

//----------------------------------------------------------------------------
template<
         typename WTYPE,
         typename T
        >
struct full_writer
{
    static void write(const WTYPE &w,const boost::python::object &o)
    {
        w.write(numpy::get_size(o),numpy::get_data<T>(o));
    }
};

template<typename WTYPE> 
struct full_writer<WTYPE,pni::core::string>
{
    static void write(const WTYPE&w,const boost::python::object &o)
    {
        using namespace pni::core;
        using namespace boost::python;

        auto shape = numpy::get_shape<shape_t>(o);
        
        if(shape.empty() && numpy::get_size(o))
            shape = shape_t{1};

        auto data = dynamic_array<string>::create(shape);

        PyArrayObject *array_ptr = reinterpret_cast<PyArrayObject*>(o.ptr());
        PyObject *ptr = PyArray_Flatten(array_ptr,NPY_CORDER);
        handle<> h(PyArray_ToList(reinterpret_cast<PyArrayObject*>(ptr)));
        list l(h);
        
        size_t index=0;
        for(auto &s: data) s = extract<string>(l[index++]);

        //how to make copy more save 
        w.write(data);
    }
};

template<typename WTYPE,typename T>
struct broadcast_writer
{
    static void write(const WTYPE &w,const boost::python::object &o)
    {
        using namespace boost::python;
        using namespace pni::core;

        object v(get_first_element(o));
        extract<T> e(v);

        auto shape = w.template shape<shape_t>();
        auto data = dynamic_array<T>::create(shape);
        std::fill(data.begin(),data.end(),e());

        w.write(data);
    }
};


template<typename WTYPE>
void write_full_data(const WTYPE &writeable,const boost::python::object &data)
{
    using namespace pni::core;
    using namespace boost::python;

    type_id_t tid = numpy::type_id(data);
    
    switch(tid)
    {
        case type_id_t::UINT8:
            full_writer<WTYPE,uint8>::write(writeable,data); break;
        case type_id_t::INT8:
            full_writer<WTYPE,int8>::write(writeable,data); break;
        case type_id_t::UINT16:
            full_writer<WTYPE,uint16>::write(writeable,data); break;
        case type_id_t::INT16:
            full_writer<WTYPE,int16>::write(writeable,data); break;
        case type_id_t::UINT32:
            full_writer<WTYPE,uint32>::write(writeable,data); break;
        case type_id_t::INT32:
            full_writer<WTYPE,int32>::write(writeable,data); break;
        case type_id_t::UINT64:
            full_writer<WTYPE,uint64>::write(writeable,data); break;
        case type_id_t::INT64:
            full_writer<WTYPE,int64>::write(writeable,data); break;
        case type_id_t::FLOAT32:
            full_writer<WTYPE,float32>::write(writeable,data); break;
        case type_id_t::FLOAT64:
            full_writer<WTYPE,float64>::write(writeable,data); break;
        case type_id_t::FLOAT128:
            full_writer<WTYPE,float128>::write(writeable,data); break;
        case type_id_t::COMPLEX32:
            full_writer<WTYPE,complex32>::write(writeable,data); break;
        case type_id_t::COMPLEX64:
            full_writer<WTYPE,complex64>::write(writeable,data); break;
        case type_id_t::COMPLEX128:
            full_writer<WTYPE,complex128>::write(writeable,data); break;
        case type_id_t::BOOL:
            full_writer<WTYPE,bool_t>::write(writeable,data); break;
        case type_id_t::STRING:
            full_writer<WTYPE,string>::write(writeable,data); break;
        default:
            throw value_error(EXCEPTION_RECORD,
                    "Unknow type_id value!");
    }
}

template<typename WTYPE>
void write_broadcast_data(const WTYPE &writeable,const boost::python::object &o)
{
    using namespace pni::core;
    using namespace boost::python;

    type_id_t tid = numpy::type_id(o);
    
    switch(tid)
    {
        case type_id_t::UINT8:
            broadcast_writer<WTYPE,uint8>::write(writeable,o); break;
        case type_id_t::INT8:
            broadcast_writer<WTYPE,int8>::write(writeable,o); break;
        case type_id_t::UINT16:
            broadcast_writer<WTYPE,uint16>::write(writeable,o); break;
        case type_id_t::INT16:
            broadcast_writer<WTYPE,int16>::write(writeable,o); break;
        case type_id_t::UINT32:
            broadcast_writer<WTYPE,uint32>::write(writeable,o); break;
        case type_id_t::INT32:
            broadcast_writer<WTYPE,int32>::write(writeable,o); break;
        case type_id_t::UINT64:
            broadcast_writer<WTYPE,uint64>::write(writeable,o); break;
        case type_id_t::INT64:
            broadcast_writer<WTYPE,int64>::write(writeable,o); break;
        case type_id_t::FLOAT32:
            broadcast_writer<WTYPE,float32>::write(writeable,o); break;
        case type_id_t::FLOAT64:
            broadcast_writer<WTYPE,float64>::write(writeable,o); break;
        case type_id_t::FLOAT128:
            broadcast_writer<WTYPE,float128>::write(writeable,o); break;
        case type_id_t::COMPLEX32:
            broadcast_writer<WTYPE,complex32>::write(writeable,o); break;
        case type_id_t::COMPLEX64:
            broadcast_writer<WTYPE,complex64>::write(writeable,o); break;
        case type_id_t::COMPLEX128:
            broadcast_writer<WTYPE,complex128>::write(writeable,o); break;
        case type_id_t::BOOL:
            broadcast_writer<WTYPE,bool_t>::write(writeable,o); break;
        case type_id_t::STRING:
            broadcast_writer<WTYPE,string>::write(writeable,o); break;
        default:
            throw value_error(EXCEPTION_RECORD,
                    "Unknow type_id value!");
    }
}



//----------------------------------------------------------------------------
//
//! \brief write data to a Nexus object
//!
//! Write data to a 
//!
//! \tparam WTYPE writeable type (field or attribute)
//! \param writable instance of WTYPE where to write data to
//! \param nparray a numpy array with the data to write 
//!
template<typename WTYPE>
void write_data(const WTYPE &writeable,const boost::python::object data)
{
    using namespace pni::core;
    using namespace boost::python;

    size_t size   = numpy::get_size(data);

    if(size == writeable.size())
        write_full_data(writeable,data);
    else if((size == 1) && (writeable.size()!=1))
        write_broadcast_data(writeable,data);
    else
        throw size_mismatch_error(EXCEPTION_RECORD,
                "Sizes of field and input data do not match!");
}
