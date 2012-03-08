#ifndef __NXWRAPPERHELPERS_HPP__
#define __NXWRAPPERHELPERS_HPP__

extern "C"{
#include<numpy/arrayobject.h>
}

#include<pni/utils/Types.hpp>
#include<pni/utils/Shape.hpp>
#include<pni/utils/Array.hpp>
#include<pni/utils/RefBuffer.hpp>

#include<boost/python/list.hpp>
using namespace pni::utils;
using namespace boost::python;
/*! \brief type mape for numpy types

This type-map maps PNI types as defines in pni/utils/Types.hpp to 
NumPy type numbers.
*/
template<typename T> class PNI2NumpyType;

//-----------------------------------------------------------------------------
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

//=============================================================================
/*! \brief create string from type id

Helper function that creatds a numpy type code string from a pni::utils::TypeID.
\param tid type id from pniutils
\return NumPy typecode
*/
String typeid2str(const TypeID &tid);

/*! \brief create list from shape

Creates a Python list from a Shape object. The length of the list corresponds to
the number of dimension in the Shape object. The lists elements are the numbers
of elements along each dimension.
\param s shape object
\return python list with 
*/
list Shape2List(const Shape &s);

/*! \brief list to Shape conversion

Converts a Python list to a Shape object. The length of the list is interpreted
as the number of dimensions and each element of the list as the number of
elements along a particular dimension.
\param l list object
\return Shape object
*/
Shape List2Shape(const list &l);

/*! \brief create reference array from numpy array

This template method creates a reference array to the data held by a numpy
array. The method for this purpose assumes that the object passed to it referes
to a numpy array. 
\param o python object
*/
template<typename T> Array<T,RefBuffer> Numpy2RefArray(const object &o)
{
    const PyArrayObject *py_array = (const PyArrayObject *)o.ptr();

    Shape s(py_array->nd);
    for(size_t i=0;i<s.rank();i++)
        s.dim(i,(size_t)PyArray_DIM(o.ptr(),i));

    RefBuffer<T> rbuffer(PyArray_SIZE(o.ptr()),(T *)PyArray_DATA(o.ptr()));
    return Array<T,RefBuffer>(s,rbuffer);
}

/*! \brief create a numpy array 

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


#endif
