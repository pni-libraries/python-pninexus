/*
 * (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
 * This file is part of libpninx-python.
 *
 * libpninx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * libpninx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libpninx.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************
 *
 * Declearation of helper functions and classes for wrappers.
 *
 * Created on: Feb 17, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#ifndef __NXWRAPPERHELPERS_HPP__
#define __NXWRAPPERHELPERS_HPP__

extern "C"{
#include<numpy/arrayobject.h>
}

#include<pni/utils/Types.hpp>
#include<pni/utils/Shape.hpp>
#include<pni/utils/ArrayFactory.hpp>
#include<pni/utils/Array.hpp>
#include<pni/utils/RefBuffer.hpp>

#include<pni/nx/NX.hpp>

#include<boost/python/list.hpp>
#include<boost/python/tuple.hpp>
using namespace pni::utils;
using namespace boost::python;
using namespace pni::nx::h5;
/*! 
\ingroup utils  
\brief type mape for numpy types

This type-map maps PNI types as defines in pni/utils/Types.hpp to 
NumPy type numbers.
*/
template<typename T> class PNI2NumpyType;

//-----------------------------------------------------------------------------
//! \cond NO_API_DOC
template<> class PNI2NumpyType<UInt8>{
    public:
        static const int typenum = NPY_UBYTE;
};

//-----------------------------------------------------------------------------
template<> class PNI2NumpyType<Int8>{
    public:
        static const int typenum = NPY_BYTE;
};

//-----------------------------------------------------------------------------
template<> class PNI2NumpyType<UInt16>{
    public:
        static const int typenum = NPY_USHORT;
};

//-----------------------------------------------------------------------------
template<> class PNI2NumpyType<Int16>{
    public:
        static const int typenum = NPY_SHORT;
};

//-----------------------------------------------------------------------------
template<> class PNI2NumpyType<UInt32>{
    public:
        static const int typenum = NPY_UINT;
};

//-----------------------------------------------------------------------------
template<> class PNI2NumpyType<Int32>{
    public:
        static const int typenum = NPY_INT;
};

//-----------------------------------------------------------------------------
template<> class PNI2NumpyType<UInt64>{
    public:
        static const int typenum = NPY_ULONG;
};

//-----------------------------------------------------------------------------
template<> class PNI2NumpyType<Int64>{
    public:
        static const int typenum = NPY_LONG;
};

//-----------------------------------------------------------------------------
template<> class PNI2NumpyType<Float32>{
    public:
        static const int typenum = NPY_FLOAT;
};

//-----------------------------------------------------------------------------
template<> class PNI2NumpyType<Float64>{
    public:
        static const int typenum = NPY_DOUBLE;
};

//-----------------------------------------------------------------------------
template<> class PNI2NumpyType<Float128>{
    public:
        static const int typenum = NPY_LONGDOUBLE;
};

//-----------------------------------------------------------------------------
template<> class PNI2NumpyType<Complex32>{
    public:
        static const int typenum = NPY_CFLOAT;
};

//-----------------------------------------------------------------------------
template<> class PNI2NumpyType<Complex64>{
    public:
        static const int typenum = NPY_CDOUBLE;
};

//-----------------------------------------------------------------------------
template<> class PNI2NumpyType<Complex128>{
    public:
        static const int typenum = NPY_CLONGDOUBLE;
};

//-----------------------------------------------------------------------------
template<> class PNI2NumpyType<String>{
    public:
        static const int typenum = NPY_STRING;
};
//! \endcond

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
\brief create list from shape

Creates a Python list from a Shape object. The length of the list corresponds to
the number of dimension in the Shape object. The lists elements are the numbers
of elements along each dimension.
\param s shape object
\return python list with 
*/
list Shape2List(const Shape &s);

//-----------------------------------------------------------------------------
/*! 
\ingroup utils  
\brief list to Shape conversion

Converts a Python list to a Shape object. The length of the list is interpreted
as the number of dimensions and each element of the list as the number of
elements along a particular dimension.
\param l list object
\return Shape object
*/
Shape List2Shape(const list &l);

//-----------------------------------------------------------------------------
/*! 
\ingroup utils  
\brief tuple to Shape conversion

Converts a Python tuple to a Shape object. The length of the tuple determines
the rank of the Shape and its elements the number of elements along each
dimension.
\param t tuple object
\return instance of class Shape
*/
Shape Tuple2Shape(const tuple &t);

//-----------------------------------------------------------------------------
/*! 
\ingroup utils  
\brief create reference array from numpy array

This template method creates a reference array to the data held by a numpy
array. The method for this purpose assumes that the object passed to it referes
to a numpy array. 
\param o python object
*/
template<typename T> Array<T,RefBuffer> Numpy2RefArray(const object &o)
{
    const PyArrayObject *py_array = (const PyArrayObject *)o.ptr();

    std::vector<size_t> dims(py_array->nd);
    for(size_t i=0;i<dims.size();i++) dims[i] = (size_t)PyArray_DIM(o.ptr(),i);

    Shape s(dims);

    RefBuffer<T> rbuffer(PyArray_SIZE(o.ptr()),(T *)PyArray_DATA(o.ptr()));
    return ArrayFactory<T,RefBuffer>::create(s,(T *)PyArray_DATA(o.ptr()));
}

//-----------------------------------------------------------------------------
/*! 
\ingroup utils  
\brief create a numpy array 

This template function creates a new numpy array from shape and type
information. This should be a rather simple thing.
*/
template<typename T> PyObject *CreateNumpyArray(const Shape &s)
{
    PyObject *ptr = nullptr;
    Buffer<npy_intp> dims(s.rank());
    for(size_t i=0;i<s.rank();i++) dims[i] = s[i];

    ptr = PyArray_SimpleNew(s.rank(),dims.ptr(),PNI2NumpyType<T>::typenum);

    return ptr;
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
NXSelection create_selection(const tuple &t,const NXField &f);


#endif
