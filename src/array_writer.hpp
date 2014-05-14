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
// Created on: May 7, 2014
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <pni/core/types.hpp>
#include <pni/core/arrays.hpp>

#include "NXWrapperHelpers.hpp"
#include "numpy_utils.hpp"

using namespace pni::core;

//! 
//! \ingroup ioclasses  
//! \brief write array data
//! 
//! Write array data to a writeable.
//!
class array_writer
{
    private:
        //! 
        //! \brief write a numpy array to the writable objects
        //! 
        //! \throw type_error if the array data type is not supported by pniio
        //! \tparam WTYPE writable type (field or attribute)
        //! \param w instance of WTYPE
        //! \param o object representing a numpy array
        //!
        template<typename WTYPE>
        static void _write_numpy_array(const WTYPE &w,const object &o)
        {
            dynamic_array<string> data;
            shape_t shape;
            //select the data type to use for writing the array data
            switch(PyArray_TYPE(o.ptr()))
            {
                case PyArray_UINT8:
                    w.write(numpy::get_data<const uint8>(o));
                    break;
                case PyArray_INT8:
                    w.write(numpy::get_data<const int8>(o));
                    break;
                case PyArray_UINT16:
                    w.write(numpy::get_data<const uint16>(o));
                    break;
                case PyArray_INT16:
                    w.write(numpy::get_data<const int16>(o));
                    break;
                case PyArray_UINT32:
                    w.write(numpy::get_data<const uint32>(o)); 
                    break;
                case PyArray_INT32:
                    w.write(numpy::get_data<const int32>(o));
                    break;
                case PyArray_UINT64:
                    w.write(numpy::get_data<const uint64>(o)); 
                    break;
                case PyArray_INT64:
                    w.write(numpy::get_data<const int64>(o)); 
                    break;
                case PyArray_FLOAT32:
                    w.write(numpy::get_data<const float32>(o)); 
                    break;
                case PyArray_FLOAT64:
                    w.write(numpy::get_data<const float64>(o)); 
                    break;
                case PyArray_LONGDOUBLE:
                    w.write(numpy::get_data<const float128>(o));
                    break;
                case NPY_CFLOAT:
                    w.write(numpy::get_data<const complex32>(o));
                    break;
                case NPY_CDOUBLE:
                    w.write(numpy::get_data<const complex64>(o)); 
                    break;
                case NPY_CLONGDOUBLE:
                    w.write(numpy::get_data<const complex128>(o));
                    break;
                case NPY_BOOL:
                    w.write(numpy::get_data<const bool_t>(o)); 
                    break;
                case NPY_STRING:
                    shape = numpy::get_shape<shape_t>(o);
                    data = dynamic_array<string>::create(shape);
                    numpy::copy_string_from_array(o,data);
                    w.write(data);
                    break;
                default:
                    throw type_error(EXCEPTION_RECORD,
                    "Type of numpy array cannot be handled!");
            };

        }

        //---------------------------------------------------------------------
        //!
        //! \brief broadcast a scalar to a field
        //!
        //! Broadcast a scalar value to the field.
        //! \tparam T type of the scalar
        //! \tparam WTYPE writable type (field or attribute)
        //! \param w instance of WTYPE
        //! \param o object representing a scalar
        //!
        template<
                 typename T,
                 typename WTYPE
                >
        static void _write_scalar(const WTYPE &w,const object &o)
        {
            typedef dynamic_array<T> array_type;
            //get writable parameters
            auto shape = w.template shape<shape_t>();

            T value = extract<T>(o)();
            auto data = array_type::create(shape);
            std::fill(data.begin(),data.end(),value);
            w.write(data);
        }
    public:
        //! 
        //! \brief write array data
        //!
        //! Writes array data o to writeable w.
        //! \throws type_error if o is not a numpy array or the datatype 
        //! cannot be handled
        //! \tparam WTYPE writeable type
        //! \param w instance of WTYPE where to store data
        //! \param o numpy array object
        //!
        template<
                 typename T,
                 typename WTYPE
                > 
        static void write(const WTYPE &w,const object &o)
        {
            //check if the object from which to read data is an array
            if(!numpy::is_array(o))
                _write_scalar<T>(w,o);
            else
                _write_numpy_array(w,o);
        }
};
