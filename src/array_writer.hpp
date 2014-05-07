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
            //select the data type to use for writing the array data
            switch(PyArray_TYPE(o.ptr()))
            {
                case PyArray_UINT8:
                    w.write((const uint8*)get_numpy_data<uint8>(o));
                    break;
                case PyArray_INT8:
                    w.write((const int8 *)get_numpy_data<int8>(o));
                    break;
                case PyArray_UINT16:
                    w.write((const uint16*)get_numpy_data<uint16>(o));
                    break;
                case PyArray_INT16:
                    w.write((const int16*)get_numpy_data<int16>(o));
                    break;
                case PyArray_UINT32:
                    w.write((const uint32*)get_numpy_data<uint32>(o)); 
                    break;
                case PyArray_INT32:
                    w.write((const int32 *)get_numpy_data<int32>(o));
                    break;
                case PyArray_UINT64:
                    w.write((const uint64*)get_numpy_data<uint64>(o)); 
                    break;
                case PyArray_INT64:
                    w.write((const int64*)get_numpy_data<int64>(o)); 
                    break;
                case PyArray_FLOAT32:
                    w.write((const float32*)get_numpy_data<float32>(o)); 
                    break;
                case PyArray_FLOAT64:
                    w.write((const float64*)get_numpy_data<float64>(o)); 
                    break;
                case PyArray_LONGDOUBLE:
                    w.write((const float128*)get_numpy_data<float128>(o));
                    break;
                case NPY_CFLOAT:
                    w.write((const complex32*)get_numpy_data<complex32>(o));
                    break;
                case NPY_CDOUBLE:
                    w.write((const complex64*)get_numpy_data<complex64>(o)); 
                    break;
                case NPY_CLONGDOUBLE:
                    w.write((const complex128*)get_numpy_data<complex128>(o));
                    break;
                case NPY_BOOL:
                    w.write((const bool_t*)get_numpy_data<bool_t>(o)); 
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
            if(!is_numpy_array(o))
                _write_scalar<T>(w,o);
            else
                _write_numpy_array(w,o);
        }
};
