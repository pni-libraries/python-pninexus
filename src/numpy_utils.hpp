//
// (c) Copyright 2014 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// Created on: May 14, 2014
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include <map>
#include <pni/core/types.hpp>
#include <pni/core/arrays.hpp>

#include <boost/python/extract.hpp>

using namespace pni::core;
using namespace boost::python;

namespace numpy
{

    //! 
    //! \ingroup numpy_utils  
    //! \brief type mape for numpy types
    //! 
    //! This type-map maps PNI types as defines in pni/utils/Types.hpp to 
    //! NumPy type numbers.
    //!
    template<typename T> class pni2numpy_type;


#define CREATE_PNI2NUMPY_TYPE(type,nptype)\
    template<> class pni2numpy_type<type>\
    {\
        public:\
               static const int typenum = nptype;\
    };

    CREATE_PNI2NUMPY_TYPE(uint8,NPY_UINT8)
    CREATE_PNI2NUMPY_TYPE(int8,NPY_INT8)
    CREATE_PNI2NUMPY_TYPE(uint16,NPY_UINT16)
    CREATE_PNI2NUMPY_TYPE(int16,NPY_INT16)
    CREATE_PNI2NUMPY_TYPE(uint32,NPY_UINT32)
    CREATE_PNI2NUMPY_TYPE(int32,NPY_INT32)
    CREATE_PNI2NUMPY_TYPE(uint64,NPY_UINT64)
    CREATE_PNI2NUMPY_TYPE(int64,NPY_INT64)
    CREATE_PNI2NUMPY_TYPE(float32,NPY_FLOAT)
    CREATE_PNI2NUMPY_TYPE(float64,NPY_DOUBLE)
    CREATE_PNI2NUMPY_TYPE(float128,NPY_LONGDOUBLE)
    CREATE_PNI2NUMPY_TYPE(complex32,NPY_CFLOAT)
    CREATE_PNI2NUMPY_TYPE(complex64,NPY_CDOUBLE)
    CREATE_PNI2NUMPY_TYPE(complex128,NPY_CLONGDOUBLE)
    CREATE_PNI2NUMPY_TYPE(string,NPY_STRING)
    CREATE_PNI2NUMPY_TYPE(bool_t,NPY_BOOL)

    static const std::map<type_id_t,int> type_id2numpy_id = {
        {type_id_t::UINT8,NPY_UINT8},
        {type_id_t::INT8,NPY_INT8},
        {type_id_t::UINT16,NPY_UINT16},
        {type_id_t::INT16,NPY_INT16},
        {type_id_t::UINT32,NPY_UINT32},
        {type_id_t::INT32,NPY_INT32},
        {type_id_t::UINT64,NPY_UINT64},
        {type_id_t::INT64,NPY_INT64},
        {type_id_t::FLOAT32,NPY_FLOAT},
        {type_id_t::FLOAT64,NPY_DOUBLE},
        {type_id_t::FLOAT128,NPY_LONGDOUBLE},
        {type_id_t::COMPLEX32,NPY_CFLOAT},
        {type_id_t::COMPLEX64,NPY_CDOUBLE},
        {type_id_t::COMPLEX128,NPY_CLONGDOUBLE},
        {type_id_t::STRING,NPY_STRING},
        {type_id_t::BOOL,NPY_BOOL}
    };

    //-------------------------------------------------------------------------
    //!
    //! \ingroup numpy_utils
    //! \brief check if object is numpy array
    //! 
    //! Checks if an object is a numpy array. 
    //! \return true if object is a numpy array
    //!
    bool is_array(const object &o);

    //------------------------------------------------------------------------
    //!
    //! \ingroup numpy_utils
    //! \brief check if object is a numpy scalar
    //!
    //! Retrun true if the object is a numpy scalar. False otherwise
    //! \param o python object
    //! \return result
    //!
    bool is_scalar(const object &o);

    //------------------------------------------------------------------------
    //!
    //! \ingroup numpy_utils
    //! \brief get type_id of a numpy object
    //! 
    //! Return the type id of an array or scalar numpy object.
    //! 
    type_id_t type_id(const object &o);

    //------------------------------------------------------------------------
    //!
    //! \ingroup numpy_utils
    //! \brief get type string
    //! 
    //! This function is necessary as numpy uses a different string
    //! representation for complex numbers. 
    //!
    //! \param id type id for which to obtain the string rep.
    //! \return string representation of the type
    //!
    string type_str(type_id_t id);

    //-------------------------------------------------------------------------
    //!
    //! \ingroup numpy_utils
    //! \briefa get type string
    //!
    string type_str(const object &o);

    //-------------------------------------------------------------------------
    //!
    //! \ingroup numpy_utils
    //! \brief get the shape of a numpy array
    //! 
    //! Return the number of elements along each dimension of a numpy array 
    //! stored in a user determined container.
    //!
    //! \tparam CTYPE container type for the shape data
    //! \param o object with numpy array
    //! \return instance of CTYPE with shape data
    //! 
    template<typename CTYPE> 
    CTYPE get_shape(const object &o)
    {
        typedef typename CTYPE::value_type value_type;

        if(!is_array(o))
            throw type_error(EXCEPTION_RECORD,"Object must be a numpy array!");

        const PyArrayObject *py_array = (const PyArrayObject *)o.ptr();

        CTYPE shape(py_array->nd);
        auto iter = shape.begin();
        for(size_t i=0;i<shape.size();++i)
            *iter++ = (value_type)PyArray_DIM(o.ptr(),i);

        return shape;
    }

    //-------------------------------------------------------------------------
    //!
    //! \ingroup numpy utils
    //! \brief get pointer to data
    //!
    //! Returns a pointer to the data stored in a numpy array. The pointer is
    //! casted to a type provided by the user as a template parameter. 
    //! 
    //! \tparam T requested data type
    //! \param o numpy array as boost::python object
    //! \return pointer to data
    //! 
    template<typename T> 
    T *get_data(const object &o)
    {
        return (T*)PyArray_DATA(o.ptr());
    }

    //------------------------------------------------------------------------
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
    object create_array(const CTYPE &s)
    {
        PyObject *ptr = nullptr;
        //create the buffer for with the shape information
        std::vector<npy_intp> dims(s.size());
        std::copy(s.begin(),s.end(),dims.begin());

        ptr = reinterpret_cast<PyObject*>(PyArray_SimpleNew(s.size(),
                                          dims.data(),pni2numpy_type<T>::typenum));

        handle<> h(ptr);

        return object(h);
    }
    
    template< typename CTYPE > 
    object create_array(type_id_t tid,const CTYPE &s,int itemsize=0)
    {
        PyObject *ptr = nullptr;
        //create the buffer for with the shape information
        std::vector<npy_intp> dims(s.size());
        std::copy(s.begin(),s.end(),dims.begin());

        ptr = reinterpret_cast<PyObject*>(
                PyArray_New(&PyArray_Type,
                            dims.size(),
                             dims.data(),
                             type_id2numpy_id.at(tid),
                             nullptr,
                             nullptr,
                             itemsize,
                             NPY_CORDER,
                             nullptr));
                             

        /*
        ptr = reinterpret_cast<PyObject*>(PyArray_SimpleNew(s.size(),
                                          dims.data(),type_id2numpy_id.at(tid)));
                                          */

        handle<> h(ptr);

        return object(h);
    }

    //-------------------------------------------------------------------------
    template<typename CTYPE>
    size_t max_string_size(const CTYPE &strings)
    {
        size_t max_size = 0;

        for(auto s: strings)
            if(max_size<s.size()) max_size=s.size();

        return max_size;
    }

    //-------------------------------------------------------------------------
    //!
    //! 
    //! 
    template<typename ATYPE>
    object create_array_from_array(const ATYPE &array)
    {
        typedef typename ATYPE::value_type value_type;

        auto shape = array.template shape<shape_t>();

        return create_array(type_id(array),shape);
    }

    //------------------------------------------------------------------------
    template<typename FTYPE>
    object create_array_from_field(const FTYPE &field)
    {
        type_id_t tid = field.type_id();
        auto shape = field.template shape<shape_t>();
        
        return create_array(tid,shape);

    }

    //-------------------------------------------------------------------------
    //!
    //! \ingroup numpy_utils
    //! \brief copy data from numpy array
    //! 
    //! Copy the data from a numpy array to a container type.  
    //! 
    //! \tparam DTYPE container type
    //! \param o numpy array object
    //! \param container instance of the destination container
    //!
    template<typename DTYPE>
    void copy_string_from_array(const object &o,DTYPE &container)
    {
        typedef typename DTYPE::value_type value_type;
        typedef pni2numpy_type<value_type> pni2numpy;

        PyArrayObject *array = (PyArrayObject *)(o.ptr());
        PyArray_Descr* dtype = PyArray_DescrFromType(pni2numpy::typenum);
        
        NpyIter *aiter = NpyIter_New(array,NPY_ITER_C_INDEX | NPY_ITER_READONLY,
                         NPY_CORDER,
                         NPY_SAME_KIND_CASTING,dtype);


        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(aiter,NULL);
        char **dataptr = NpyIter_GetDataPtrArray(aiter);
        auto citer = container.begin();
        int max_item_size = PyArray_ITEMSIZE(array);
        do
        {
            value_type d = value_type(dataptr[0]);
            if(d.size()>size_t(max_item_size))
                *citer++ = value_type(d,0,max_item_size);
            else
               *citer++ = d;
        }
        while((iternext(aiter))&&(citer!=container.end()));

        //need to destroy the dtype here - decrement the reference counter
        Py_DECREF(dtype);
        NpyIter_Deallocate(aiter);
    }

    //------------------------------------------------------------------------
    template<typename DTYPE>
    void copy_string_to_array(const DTYPE &source,object &dest)
    {
        typedef typename DTYPE::value_type value_type;
        typedef pni2numpy_type<value_type> pni2numpy;

        PyArrayObject *array = (PyArrayObject *)(dest.ptr());
        PyArray_Descr* dtype = PyArray_DescrFromType(NPY_STRING);
        
        NpyIter *aiter = NpyIter_New(array,NPY_ITER_C_INDEX | NPY_ITER_READWRITE,
                         NPY_CORDER,
                         NPY_SAME_KIND_CASTING,dtype);


        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(aiter,NULL);
        char **dataptr = NpyIter_GetDataPtrArray(aiter);
        auto citer = source.begin();
        do
        {
            std::copy(citer->begin(),citer->end(),dataptr[0]);
            citer++;
        }
        while((iternext(aiter))&&(citer!=source.end()));

        //need to destroy the dtype here - decrement the reference counter
        Py_DECREF(dtype);
        NpyIter_Deallocate(aiter);
    }



//end of namespace
}
