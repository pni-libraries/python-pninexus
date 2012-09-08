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
#include<pni/utils/Array.hpp>
#include<pni/utils/RBuffer.hpp>

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
\brief create a python list from a container

Creates a Python list from a C++ container.
\tparam CTYPE containerr type
\param s instance of CTYPE
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
template<typename T,typename CTYPE> PyObject *CreateNumpyArray(const CTYPE &s)
{
    PyObject *ptr = nullptr;
    //create the buffer for with the shape information
    DBuffer<npy_intp> dims(s.size());
    std::copy(s.begin(),s.end(),dims.begin());

    ptr = PyArray_SimpleNew(s.size(),dims.ptr(),PNI2NumpyType<T>::typenum);

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
