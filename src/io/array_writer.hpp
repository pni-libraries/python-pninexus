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

#include <boost/python.hpp>
#include <pni/core/types.hpp>
#include <pni/core/arrays.hpp>
#include <core/utils.hpp>
#include <core/numpy_utils.hpp>

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
        static void _write_numpy_array(const WTYPE &w,
                                       const boost::python::object &o)
        {
            using namespace pni::core;
            using namespace boost::python;
            type_id_t tid = numpy::type_id(o);

            //select the data type to use for writing the array data
            if(tid == type_id_t::UINT8)
                w.write(w.size(),numpy::get_data<const uint8>(o));
            else if(tid == type_id_t::INT8)
                w.write(w.size(),numpy::get_data<const int8>(o));
            else if(tid == type_id_t::UINT16)
                w.write(w.size(),numpy::get_data<const uint16>(o));
            else if(tid == type_id_t::INT16)
                w.write(w.size(),numpy::get_data<const int16>(o));
            else if(tid == type_id_t::UINT32)
                w.write(w.size(),numpy::get_data<const uint32>(o)); 
            else if(tid == type_id_t::INT32)
                w.write(w.size(),numpy::get_data<const int32>(o));
            else if(tid == type_id_t::UINT64)
                w.write(w.size(),numpy::get_data<const uint64>(o)); 
            else if(tid == type_id_t::INT64)
                w.write(w.size(),numpy::get_data<const int64>(o)); 
            else if(tid == type_id_t::FLOAT32)
                w.write(w.size(),numpy::get_data<const float32>(o)); 
            else if(tid == type_id_t::FLOAT64)
                w.write(w.size(),numpy::get_data<const float64>(o)); 
            else if(tid == type_id_t::FLOAT128)
                w.write(w.size(),numpy::get_data<const float128>(o));
            else if(tid == type_id_t::COMPLEX32)
                w.write(w.size(),numpy::get_data<const complex32>(o));
            else if(tid == type_id_t::COMPLEX64)
                w.write(w.size(),numpy::get_data<const complex64>(o)); 
            else if(tid == type_id_t::COMPLEX128)
                w.write(w.size(),numpy::get_data<const complex128>(o));
            else if(tid == type_id_t::BOOL)
                w.write(w.size(),numpy::get_data<const bool_t>(o)); 
            else if(tid == type_id_t::STRING)
            {
                auto shape = numpy::get_shape<shape_t>(o);
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
            else
                throw type_error(EXCEPTION_RECORD,
                "Type of numpy array cannot be handled!");

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
        static void write(const WTYPE &w,
                          const boost::python::object &o)
        {
            using namespace pni::core;

            if(w.size() != numpy::get_size(o))
                throw size_mismatch_error(EXCEPTION_RECORD,
                        "Size of fields and numpy array do not match!");

            /*
            if(!((w.size() == 1) && (numpy::get_size(o)==1)))
            {
                auto w_shape = w.template shape<shape_t>();
                auto o_shape = numpy::get_shape<shape_t>(o);
                shape_t o_shape_clean;
                std::copy_if(o_shape.begin(),o_shape.end(),
                             std::back_inserter(o_shape_clean),
                             [](size_t i){ return i!=1;});
                
                if((w_shape.size()!=o_shape_clean.size()) ||
                   !std::equal(w_shape.begin(),w_shape.end(),o_shape_clean.begin()))
                    throw shape_mismatch_error(EXCEPTION_RECORD,
                            "Shapes of field and numpy array do not match!");
            }
            */

            _write_numpy_array(w,o);
        }
};
