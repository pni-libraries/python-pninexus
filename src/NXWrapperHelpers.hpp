/*
 * (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
 * This file is part of python-pniio.
 *
 * python-pniio is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * python-pniio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************
 *
 * Declearation of helper functions and classes for wrappers.
 *
 * Created on: Feb 17, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#pragma once

extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include<pni/core/Types.hpp>
#include<pni/core/Array.hpp>
#include<pni/core/RBuffer.hpp>

#include<pni/io/nx/NX.hpp>

#include<boost/python/extract.hpp>
#include<boost/python/list.hpp>
#include<boost/python/tuple.hpp>

using namespace pni::core;
using namespace boost::python;
using namespace pni::io::nx::h5;

#define CREATE_PNI2NUMPY_TYPE(type,nptype)\
    template<> class PNI2NumpyType<type>\
    {\
        public:\
               static const int typenum = nptype;\
    };

/*! 
\ingroup utils  
\brief type mape for numpy types

This type-map maps PNI types as defines in pni/utils/Types.hpp to 
NumPy type numbers.
*/
template<typename T> class PNI2NumpyType;

CREATE_PNI2NUMPY_TYPE(UInt8,NPY_UBYTE);
CREATE_PNI2NUMPY_TYPE(Int8,NPY_BYTE);
CREATE_PNI2NUMPY_TYPE(UInt16,NPY_USHORT);
CREATE_PNI2NUMPY_TYPE(Int16,NPY_SHORT);
CREATE_PNI2NUMPY_TYPE(UInt32,NPY_UINT);
CREATE_PNI2NUMPY_TYPE(Int32,NPY_INT);
CREATE_PNI2NUMPY_TYPE(UInt64,NPY_ULONG);
CREATE_PNI2NUMPY_TYPE(Int64,NPY_LONG);
CREATE_PNI2NUMPY_TYPE(Float32,NPY_FLOAT);
CREATE_PNI2NUMPY_TYPE(Float64,NPY_DOUBLE);
CREATE_PNI2NUMPY_TYPE(Float128,NPY_LONGDOUBLE);
CREATE_PNI2NUMPY_TYPE(Complex32,NPY_CFLOAT);
CREATE_PNI2NUMPY_TYPE(Complex64,NPY_CDOUBLE);
CREATE_PNI2NUMPY_TYPE(Complex128,NPY_CLONGDOUBLE);
CREATE_PNI2NUMPY_TYPE(String,NPY_STRING);
CREATE_PNI2NUMPY_TYPE(Bool,NPY_BOOL);

//=============================================================================
/*! 
\ingroup utils  
\brief create string from type id

Helper function that creatds a numpy type code string from a pni::utils::TypeID.
\param tid type id from pniutils
\return NumPy typecode
*/
String typeid2str(const TypeID &tid);

//-----------------------------------------------------------------------------
/*! 
\ingroup utils  
\brief create a python list from a container

Creates a Python list from a C++ container.
\tparam CTYPE containerr type
\param c instance of CTYPE
\return python list with 
*/
template<typename CTYPE> list Container2List(const CTYPE &c)
{
    list l;
    if(c.size()==0) return l;
    
    for(auto iter=c.begin();iter!=c.end();++iter)
        l.append(*iter);

    return l;

}

//-----------------------------------------------------------------------------
/*! 
\ingroup utils  
\brief create a container from a Python list

Convert a Python list to a C++ container. 
\tparam CTYPE container type.
\param l python list object
\return instance of 
*/
template<typename CTYPE> CTYPE List2Container(const list &l)
{
    CTYPE c(len(l));

    size_t index=0;
    for(typename CTYPE::iterator iter=c.begin();iter!=c.end();++iter)
        *iter = extract<typename CTYPE::value_type>(l[index++]);

    return c;
}

//-----------------------------------------------------------------------------
/*! 
\ingroup utils  
\brief tuple to container conversion

Converts a Python tuple to a Shape object. The length of the tuple determines
the rank of the Shape and its elements the number of elements along each
dimension.
\tparam CTYPE container type
\param t tuple object
\return instance of type CTYPE
*/
template<typename CTYPE> CTYPE Tuple2Container(const tuple &t)
{
    return List2Container<CTYPE>(list(t));
}

//-----------------------------------------------------------------------------
/*! 
\ingroup utils  
\brief create reference array from numpy array

This template method creates a reference array to the data held by a numpy
array. The method for this purpose assumes that the object passed to it referes
to a numpy array. 
\param o python object
*/
template<typename T> DArray<T,RBuffer<T> > Numpy2RefArray(const object &o)
{
    const PyArrayObject *py_array = (const PyArrayObject *)o.ptr();

    //create a shape object
    std::vector<size_t> shape(py_array->nd);
    for(size_t i=0;i<shape.size();i++) 
        shape[i] = (size_t)PyArray_DIM(o.ptr(),i);

    RBuffer<T> rbuffer(PyArray_SIZE(o.ptr()),(T *)PyArray_DATA(o.ptr()));
    return DArray<T,RBuffer<T> >(shape,rbuffer);
}

//-----------------------------------------------------------------------------
/*! 
\ingroup utils  
\brief create a numpy array 

This template function creates a new numpy array from shape and type
information. This should be a rather simple thing.
*/
template<typename T,typename CTYPE> object CreateNumpyArray(const CTYPE &s)
{
    PyObject *ptr = nullptr;
    //create the buffer for with the shape information
    DBuffer<npy_intp> dims(s.size());
    std::copy(s.begin(),s.end(),dims.begin());

    ptr = PyArray_SimpleNew(s.size(),const_cast<npy_intp*>(dims.ptr()),
                            PNI2NumpyType<T>::typenum);

    handle<> h(ptr);

    return object(h);
}

//-----------------------------------------------------------------------------
/*!
\ingroup utils
\brief get array information from nested lists

Though arrays of data are usually represented by numpy-arrays in Python the fact
that libpninx supports arrays of variable length strings causes some problems. 
Numpy does not support arrays of variable length strings. Thus we use nested
lists to represent such structures. 
This function determines the rank (the number of dimensions) for an array that
should represent the data stored in the list. 
\param o object with the nested list
\return rank of the array
*/
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
/*! 
\ingroup utils  
\brief selection from tuple 

Adopts the selection of a field according to a tuple with indices and slices.
In order to succeed the tuple passed to this function must contain only indices,
slices, and a single ellipsis.
\throws TypeError if one of typle element is from an unsupported type
\throws IndexError if more than one ellipsis is contained in the tuple or if an
Index exceeds the number of elements along the correpsonding field dimension.
\throws ShapeMissmatchError if the size of the tuple exceeds the rank of the
field from which the selection should be drawn.
\param t tuple with indices and slices
\param f reference to the field for which to create the selection
*/
std::vector<Slice> create_selection(const tuple &t,const NXField &f);

//-----------------------------------------------------------------------------
/*!
\ingroup utils
\brief check if unicode

Check if the instance of objec represents a unicode object. 
\param o instance to check
\return true if o is a unicode object, false otherwise
*/
bool is_unicode(const object &o);

//-----------------------------------------------------------------------------
/*!
\ingroup utils
\brief convert unicode to string

Converts a Python unicode object to a common Python String object using 
UTF8 encoding.
\param o python unicode object
\return python string object
*/
object unicode2str(const object &o);

