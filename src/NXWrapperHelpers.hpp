//
// (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// Created on: Feb 17, 2012
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include <vector>
#include <pni/core/types.hpp>
#include <pni/core/arrays.hpp>

#include <pni/io/nx/nx.hpp>

#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>

using namespace pni::core;
using namespace boost::python;
using namespace pni::io::nx::h5;

#define CREATE_PNI2NUMPY_TYPE(type,nptype)\
    template<> class PNI2NumpyType<type>\
    {\
        public:\
               static const int typenum = nptype;\
    };

//! 
//! \ingroup utils  
//! \brief type mape for numpy types
//! 
//! This type-map maps PNI types as defines in pni/utils/Types.hpp to 
//! NumPy type numbers.
//!
template<typename T> class PNI2NumpyType;

CREATE_PNI2NUMPY_TYPE(uint8,NPY_UINT8);
CREATE_PNI2NUMPY_TYPE(int8,NPY_INT8);
CREATE_PNI2NUMPY_TYPE(uint16,NPY_UINT16);
CREATE_PNI2NUMPY_TYPE(int16,NPY_INT16);
CREATE_PNI2NUMPY_TYPE(uint32,NPY_UINT32);
CREATE_PNI2NUMPY_TYPE(int32,NPY_INT32);
CREATE_PNI2NUMPY_TYPE(uint64,NPY_UINT64);
CREATE_PNI2NUMPY_TYPE(int64,NPY_INT64);
CREATE_PNI2NUMPY_TYPE(float32,NPY_FLOAT);
CREATE_PNI2NUMPY_TYPE(float64,NPY_DOUBLE);
CREATE_PNI2NUMPY_TYPE(float128,NPY_LONGDOUBLE);
CREATE_PNI2NUMPY_TYPE(complex32,NPY_CFLOAT);
CREATE_PNI2NUMPY_TYPE(complex64,NPY_CDOUBLE);
CREATE_PNI2NUMPY_TYPE(complex128,NPY_CLONGDOUBLE);
CREATE_PNI2NUMPY_TYPE(string,NPY_STRING);
CREATE_PNI2NUMPY_TYPE(bool_t,NPY_BOOL);

//=============================================================================
//! 
//! \ingroup utils  
//! \brief create string from type id
//! 
//! Helper function that creatds a numpy type code string from a 
//! pni::core::type_id_t.
//! \param tid type id from pniutils
//! \return NumPy typecode
//!
string typeid2str(const type_id_t &tid);

//-----------------------------------------------------------------------------
//! 
//! \ingroup utils  
//! \brief create a python list from a container
//! 
//! Creates a Python list from a C++ container.
//! \tparam CTYPE containerr type
//! \param c instance of CTYPE
//! \return python list with 
//!
template<typename CTYPE> list Container2List(const CTYPE &c)
{
    list l;
    if(c.size()==0) return l;

    for(auto iter=c.begin();iter!=c.end();++iter)
        l.append(*iter);

    return l;

}

//-----------------------------------------------------------------------------
//! 
//! \ingroup utils  
//! \brief create a container from a Python list
//!
//! Convert a Python list to a C++ container. 
//! \tparam CTYPE container type.
//! \param l python list object
//! \return instance of 
//!
template<typename CTYPE> CTYPE List2Container(const list &l)
{
    CTYPE c(len(l));

    size_t index=0;
    for(typename CTYPE::iterator iter=c.begin();iter!=c.end();++iter)
        *iter = extract<typename CTYPE::value_type>(l[index++]);

    return c;
}

//-----------------------------------------------------------------------------
//! 
//! \ingroup utils  
//! \brief tuple to container conversion
//! 
//! Converts a Python tuple to a Shape object. The length of the tuple 
//! determines the rank of the Shape and its elements the number of elements 
//! along each dimension.
//! 
//! \tparam CTYPE container type
//! \param t tuple object
//! \return instance of type CTYPE
//!
template<typename CTYPE> CTYPE Tuple2Container(const tuple &t)
{
    return List2Container<CTYPE>(list(t));
}

//-----------------------------------------------------------------------------
//!
//! \ingroup utils
//! \brief check if object is numpy array
//! 
//! Checks if an object is a numpy array. 
//! \return true if object is a numpy array
//!
bool is_numpy_array(const object &o);

//-----------------------------------------------------------------------------
//!
//! \ingroup utils
//! \brief get the shape of a numpy array
//! 
//! Return the number of elements along each dimension of a numpy array 
//! in use determined container.
//!
//! 
template<typename CTYPE> CTYPE get_numpy_shape(const object &o)
{
    typedef typename CTYPE::value_type value_type;

    if(!is_numpy_array(o))
        throw type_error(EXCEPTION_RECORD,"Object must be a numpy array!");

    const PyArrayObject *py_array = (const PyArrayObject *)o.ptr();

    CTYPE shape(py_array->nd);
    auto iter = shape.begin();
    for(size_t i=0;i<shape.size();++i)
        *iter++ = (value_type)PyArray_DIM(o.ptr(),i);

    return shape;
}

//-----------------------------------------------------------------------------
template<typename T> T *get_numpy_data(const object &o)
{
    return (T*)PyArray_DATA(o.ptr());
}


//-----------------------------------------------------------------------------
//! 
//! \ingroup utils  
//! \brief create a numpy array 
//!
//! This template function creates a new numpy array from shape and type
//! information. This should be a rather simple thing.
//! \tparam T data type to use for the numpy array
//! \tparam CTYPE container type for the shape
//! \param s instance of CTYPE with the required shape
//! \return instance of numpy
//! 
template<
         typename T,
         typename CTYPE
        > 
object create_numpy_array(const CTYPE &s)
{
    PyObject *ptr = nullptr;
    //create the buffer for with the shape information
    std::vector<npy_intp> dims(s.size());
    std::copy(s.begin(),s.end(),dims.begin());

    ptr = PyArray_SimpleNew(s.size(),dims.data(),PNI2NumpyType<T>::typenum);

    handle<> h(ptr);

    return object(h);
}

//-----------------------------------------------------------------------------
//!
//! \ingroup utils
//! \brief get array information from nested lists
//! 
//! Though arrays of data are usually represented by numpy-arrays in Python 
//! the fact that libpninx supports arrays of variable length strings causes 
//! some problems.  Numpy does not support arrays of variable length strings. 
//! Thus we use nested lists to represent such structures. 
//! This function determines the rank (the number of dimensions) for an array 
//! that should represent the data stored in the list. 
//! 
//! \param o object with the nested list
//! \return rank of the array
//!
size_t  nested_list_rank(const object &o);

//-----------------------------------------------------------------------------
template<typename CTYPE> CTYPE nested_list_shape(const object &o)
{
    size_t rank = nested_list_rank(o);
    CTYPE c(rank);
    object lo(o);
    for(auto iter=c.begin();iter!=c.end()-1;++iter)
    {
        list l = extract<list>(lo)();
        *iter = len(l);
        lo = l[0];
    }

    return c;
}

//-----------------------------------------------------------------------------
//! 
//! \ingroup utils  
//! \brief selection from tuple 
//! 
//! Adopts the selection of a field according to a tuple with indices and 
//! slices.  In order to succeed the tuple passed to this function must contain 
//! only indices, slices, and a single ellipsis.
//! 
//! \throws TypeError if one of typle element is from an unsupported type
//! \throws IndexError if more than one ellipsis is contained in the tuple or 
//! if an index exceeds the number of elements along the correpsonding field 
//! dimension.
//! \throws ShapeMissmatchError if the size of the tuple exceeds the rank of 
//! the field from which the selection should be drawn.
//! \param t tuple with indices and slices
//! \param f reference to the field for which to create the selection
//! \return vector with slices
//!
std::vector<pni::core::slice> create_selection(const tuple &t,const nxfield &f);

//-----------------------------------------------------------------------------
//!
//! \ingroup utils
//! \brief check if unicode
//! 
//! Check if the instance of objec represents a unicode object. 
//! \param o instance to check
//! \return true if o is a unicode object, false otherwise
//!
bool is_unicode(const object &o);

//-----------------------------------------------------------------------------
//!
//! \ingroup utils
//! \brief convert unicode to string
//! 
//! Converts a Python unicode object to a common Python String object using 
//! UTF8 encoding.
//! 
//! \param o python unicode object
//! \return python string object
//!
object unicode2str(const object &o);

