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
#pragma once

#include <boost/python/extract.hpp>
#include <h5cpp/hdf5.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PNI_CORE_USYMBOL
#define NO_IMPORT_ARRAY
extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include <map>
#include <pni/core/types.hpp>
#include <pni/core/arrays.hpp>

#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#endif


//!
//! \ingroup pub_api devel_doc
//! \brief numpy utility namespace 
//!
namespace numpy
{

    //! 
    //! \ingroup devel_doc  
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

    CREATE_PNI2NUMPY_TYPE(pni::core::uint8,NPY_UINT8)
    CREATE_PNI2NUMPY_TYPE(pni::core::int8,NPY_INT8)
    CREATE_PNI2NUMPY_TYPE(pni::core::uint16,NPY_UINT16)
    CREATE_PNI2NUMPY_TYPE(pni::core::int16,NPY_INT16)
    CREATE_PNI2NUMPY_TYPE(pni::core::uint32,NPY_UINT32)
    CREATE_PNI2NUMPY_TYPE(pni::core::int32,NPY_INT32)
    CREATE_PNI2NUMPY_TYPE(pni::core::uint64,NPY_UINT64)
    CREATE_PNI2NUMPY_TYPE(pni::core::int64,NPY_INT64)
    CREATE_PNI2NUMPY_TYPE(pni::core::float32,NPY_FLOAT)
    CREATE_PNI2NUMPY_TYPE(pni::core::float64,NPY_DOUBLE)
    CREATE_PNI2NUMPY_TYPE(pni::core::float128,NPY_LONGDOUBLE)
    CREATE_PNI2NUMPY_TYPE(pni::core::complex32,NPY_CFLOAT)
    CREATE_PNI2NUMPY_TYPE(pni::core::complex64,NPY_CDOUBLE)
    CREATE_PNI2NUMPY_TYPE(pni::core::complex128,NPY_CLONGDOUBLE)
#if PY_MAJOR_VERSION >= 3
    CREATE_PNI2NUMPY_TYPE(pni::core::string,NPY_UNICODE)
#else
    CREATE_PNI2NUMPY_TYPE(pni::core::string,NPY_STRING)
#endif
    CREATE_PNI2NUMPY_TYPE(pni::core::bool_t,NPY_BOOL)

    //!
    //! \ingroup devel_doc
    //! \brief map between type_id_t and numpy types
    //!
    //! A static map between `libpnicore` `type_id_t` values and `numpy` 
    //! data types.
    static const std::map<pni::core::type_id_t,int> type_id2numpy_id = {
        {pni::core::type_id_t::UINT8,  NPY_UINT8},
        {pni::core::type_id_t::INT8,   NPY_INT8},
        {pni::core::type_id_t::UINT16, NPY_UINT16},
        {pni::core::type_id_t::INT16,  NPY_INT16},
        {pni::core::type_id_t::UINT32, NPY_UINT32},
        {pni::core::type_id_t::INT32,  NPY_INT32},
        {pni::core::type_id_t::UINT64, NPY_UINT64},
        {pni::core::type_id_t::INT64,  NPY_INT64},
        {pni::core::type_id_t::FLOAT32, NPY_FLOAT},
        {pni::core::type_id_t::FLOAT64, NPY_DOUBLE},
        {pni::core::type_id_t::FLOAT128,NPY_LONGDOUBLE},
        {pni::core::type_id_t::COMPLEX32, NPY_CFLOAT},
        {pni::core::type_id_t::COMPLEX64, NPY_CDOUBLE},
        {pni::core::type_id_t::COMPLEX128,NPY_CLONGDOUBLE},
#if PY_MAJOR_VERSION >= 3
        {pni::core::type_id_t::STRING,NPY_UNICODE},
#else
        {pni::core::type_id_t::STRING,NPY_STRING},
#endif
        {pni::core::type_id_t::BOOL,  NPY_BOOL}
    };

    //-------------------------------------------------------------------------
    //!
    //! \ingroup pub_api
    //! \brief check if object is numpy array
    //! 
    //! Checks if an object is a numpy array. 
    //! \return true if object is a numpy array
    //!
    bool is_array(const boost::python::object &o);

    //------------------------------------------------------------------------
    //!
    //! \ingroup pub_api
    //! \brief check if object is a numpy scalar
    //!
    //! Retrun true if the object is a numpy scalar. False otherwise. 
    //! This function is quite similar to the general is_scalar function. 
    //! In fact, is_scalar is calling this function to check whether or not 
    //! an object is a numpy scalar. 
    //! 
    //! \param o python object
    //! \return result
    //!
    bool is_scalar(const boost::python::object &o);

    //------------------------------------------------------------------------
    //!
    //! \ingroup pub_api
    //! \brief get type_id of a numpy object
    //! 
    //! Return the `libpnicore` type ID from a numpy scalar or array. 
    //! 
    //! \throws type_error if the Python type has no counterpart in 
    //! `libpnicore` or if the object is not an array or scalar 
    //! 
    //! \param o reference to the Python scalar or array
    //! \return type ID of the scalar or array element type
    //! 
    pni::core::type_id_t type_id(const boost::python::object &o);

    //------------------------------------------------------------------------
    //!
    //! \ingroup pub_api
    //! \brief get type string
    //! 
    //! Returns the numpy string representation for a particular `libpnicore` 
    //! data type. This function is necessary as numpy uses a different string
    //! representation for complex numbers (thus we cannot use the
    //! corresponding function provided by `libpnicore`). 
    //!
    //! \param id type id for which to obtain the string rep.
    //! \return string representation of the type
    //!
    pni::core::string type_str(pni::core::type_id_t id);

    //-------------------------------------------------------------------------
    //!
    //! \ingroup pub_api
    //! \briefa get type string
    //! 
    //! Return the string representation 
    //!
    pni::core::string type_str(const boost::python::object &o);
    
    boost::python::object create_array(pni::core::type_id_t tid,
                                       const hdf5::Dimensions &dimensions);

    //-------------------------------------------------------------------------
    //!
    //! \ingroup pub_api
    //! \brief get the shape of a numpy array
    //! 
    //! Return the number of elements along each dimension of a numpy array 
    //! stored in a user determined C++ container type.
    //!
    //! \throws type_error if the object is not a numpy array
    //! \tparam CTYPE container type for the shape data
    //! \param o object with numpy array
    //! \return instance of CTYPE with shape data
    //! 
    template<typename CTYPE> 
    CTYPE get_shape(const boost::python::object &o)
    {
        typedef typename CTYPE::value_type value_type;

        if(!is_array(o))
            throw pni::core::type_error(EXCEPTION_RECORD,"Object must be a numpy array!");

        const PyArrayObject *py_array = (const PyArrayObject *)o.ptr();

        CTYPE shape(PyArray_NDIM(py_array));
        auto iter = shape.begin();
        for(size_t i=0;i<shape.size();++i)
            *iter++ = (value_type)PyArray_DIM(py_array,i);

        return shape;
    }

    //------------------------------------------------------------------------
    //! 
    //! \ingroup pub_api
    //! \brief get number of elements
    //! 
    //! Returns the number of elements stored in the array.
    //!
    //! \param o array object
    //! \return number of elements
    //!
    size_t get_size(const boost::python::object &o);

    //-------------------------------------------------------------------------
    //!
    //! \ingroup pub_api
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
    T *get_data(const boost::python::object &o)
    {
        if(!is_array(o))
            throw pni::core::type_error(EXCEPTION_RECORD,
                    "Argument must be a numpy array!");

        return (T*)PyArray_DATA((PyArrayObject*)o.ptr());
    }

    //------------------------------------------------------------------------
    //! 
    //! \ingroup pub_api
    //! \brief create a numpy array 
    //!
    //! This template function creates a new numpy array from shape and type
    //! information. This should be a rather simple thing.
    //! If T does not have a numpy counterpart a compile time error will occur.
    //!
    //! \tparam T data type to use for the numpy array
    //! \tparam CTYPE container type for the shape
    //! \param s instance of CTYPE with the required shape
    //! \return instance of numpy
    //! 
    template<
             typename T,
             typename CTYPE
            > 
    boost::python::object create_array(const CTYPE &s)
    {
        PyObject *ptr = nullptr;
        //create the buffer for with the shape information
        std::vector<npy_intp> dims(s.size());
        std::copy(s.begin(),s.end(),dims.begin());

        ptr = reinterpret_cast<PyObject*>(PyArray_SimpleNew(s.size(),
                                          dims.data(),pni2numpy_type<T>::typenum));
        
        boost::python::handle<> h(ptr);

        return boost::python::object(h);
    }
   
    //--------------------------------------------------------------------------
    //!
    //! \ingroup pnicore_numpy
    //!
    template< typename CTYPE > 
    boost::python::object 
    create_array(pni::core::type_id_t tid,const CTYPE &s,int itemsize=0)
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
                             
        
        boost::python::handle<> h(ptr);

        return boost::python::object(h);
    }

    //-------------------------------------------------------------------------
    //!
    //! \ingroup pub_api
    //! \brief get maximum string size from a container 
    //! 
    //! Return the maximum size of all strings stored in a container. 
    //! 
    //! \tparam CTYPE container type with strings
    //! \param strings reference to the string container
    //! \return maximum string size
    //! 
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
    boost::python::object create_array_from_array(const ATYPE &array)
    {
        auto shape = array.template shape<pni::core::shape_t>();

        return create_array(type_id(array),shape);
    }

    //------------------------------------------------------------------------
    template<typename FTYPE>
    boost::python::object create_array_from_field(const FTYPE &field)
    {
        pni::core::type_id_t tid = field.type_id();
        auto shape = field.template shape<pni::core::shape_t>();
        
        return create_array(tid,shape);

    }

    //------------------------------------------------------------------------
    //!
    //! \ingroup pub_api
    //! \brief convert to numpy array
    //!
    //! Take an arbitrary Python object and convert it to a numpy array. 
    //! If the object is already a numpy array we do nothing and just 
    //! pass the object through. Otherwise the numpy C-API will try 
    //! to convert the object to a numpy array. 
    //!
    //! 
    boost::python::object to_numpy_array(const boost::python::object &o);


//end of namespace
}
