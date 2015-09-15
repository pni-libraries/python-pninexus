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
// Created on: March 9, 2012
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

extern "C" {
#include <Python.h>
}

#include "array_writer.hpp"
#include "array_reader.hpp"
#include "scalar_reader.hpp"
#include "scalar_writer.hpp"

//! 
//! \ingroup ioclasses  
//! \brief read all possible scalars from a readable
//! 
//! Function reads data from an readable object and returns a Python object 
//! with the result. The template parameter IOOp must be a type that implements 
//! the readable interface.
//! 
//! \throws type_error if readable is of unknow data type
//! \tparam IOOP reader type to use for the operation (scalar or array reader)
//! \tparam OType readable type (field or attribute)
//! \param readable object from which to read data
//! \return a Python object with the data
//!
template<
         typename IOOP,
         typename OType
        > 
object io_read(const OType &readable)
{
    type_id_t tid = readable.type_id();

    if(tid == type_id_t::UINT8)  return IOOP::template read<uint8>(readable);
    if(tid == type_id_t::INT8)   return IOOP::template read<int8>(readable);
    if(tid == type_id_t::UINT16) return IOOP::template read<uint16>(readable);
    if(tid == type_id_t::INT16)  return IOOP::template read<int16>(readable);
    if(tid == type_id_t::UINT32) return IOOP::template read<uint32>(readable);
    if(tid == type_id_t::INT32)  return IOOP::template read<int32>(readable);
    if(tid == type_id_t::UINT64) return IOOP::template read<uint64>(readable);
    if(tid == type_id_t::INT64)  return IOOP::template read<int64>(readable);

    if(tid == type_id_t::FLOAT32) 
        return IOOP::template read<float32>(readable);
    if(tid == type_id_t::FLOAT64) 
        return IOOP::template read<float64>(readable);
    if(tid == type_id_t::FLOAT128) 
        return IOOP::template read<float128>(readable);

    if(tid == type_id_t::COMPLEX32) 
        return IOOP::template read<complex32>(readable);
    if(tid == type_id_t::COMPLEX64) 
        return IOOP::template read<complex64>(readable);
    if(tid == type_id_t::COMPLEX128) 
        return IOOP::template read<complex128>(readable);

    if(tid == type_id_t::STRING) 
    {
        
        //in case of a scalar string we can use the standard IO operator
        if(readable.size()==1) return IOOP::template read<string>(readable);

        //for arrays we need to do some magic
        auto shape = readable.template shape<shape_t>();
        auto data = dynamic_array<string>::create(shape);

        readable.read(data);
           
        size_t itemsize = numpy::max_string_size(data);
        
        if(!itemsize) itemsize=1;

#if PY_MAJOR_VERSION >= 3
        //On Python 3 strings are UTF8 encoded, thus every character occupies
        //at least 4 Byte of memory
        object array = numpy::create_array(tid,shape,int(itemsize));
#else 
        //On Python 2 strings are just array so char 
        object array = numpy::create_array(tid,shape,int(itemsize));
#endif
        numpy::copy_string_to_array(data,array);
        return array;
    }
    if(tid == type_id_t::BOOL)
        return IOOP::template read<bool_t>(readable);

    throw type_error(EXCEPTION_RECORD,"Cannot handle field datatype!");
   
    return object();
}

//-----------------------------------------------------------------------------
//! 
//! \ingroup ioclasses  
//! \brief write data 
//! 
//! Write data from a Python object to a writable object.
//! 
//! \throws type_error if type of the writable cannot be handled
//! \tparam IOOP writer type to use (scalar or array)
//! \tparam OTYPE writeable type (field or attribute)
//! \param writeable object to which data will be written
//! \param obj Python object form which data will be written
//!
template<
         typename IOOP,
         typename OTYPE
        > 
void io_write(const OTYPE &writeable,const object &obj)
{
    type_id_t tid = writeable.type_id();

    if(tid == type_id_t::UINT8) 
        IOOP::template write<uint8>(writeable,obj);
    else if(tid == type_id_t::INT8) 
        IOOP::template write<int8>(writeable,obj); 
    else if(tid == type_id_t::UINT16)
        IOOP::template write<uint16>(writeable,obj); 
    else if(tid == type_id_t::INT16) 
        IOOP::template write<int16>(writeable,obj); 
    else if(tid == type_id_t::UINT32) 
        IOOP::template write<uint32>(writeable,obj); 
    else if(tid == type_id_t::INT32) 
        IOOP::template write<int32>(writeable,obj); 
    else if(tid == type_id_t::UINT64) 
        IOOP::template write<uint64>(writeable,obj); 
    else if(tid == type_id_t::INT64)
        IOOP::template write<int64>(writeable,obj); 
    else if(tid == type_id_t::FLOAT32) 
        IOOP::template write<float32>(writeable,obj); 
    else if(tid == type_id_t::FLOAT64)
        IOOP::template write<float64>(writeable,obj); 
    else if(tid == type_id_t::FLOAT128) 
        IOOP::template write<float128>(writeable,obj); 
    else if(tid == type_id_t::COMPLEX32) 
        IOOP::template write<complex32>(writeable,obj); 
    else if(tid == type_id_t::COMPLEX64)
        IOOP::template write<complex64>(writeable,obj); 
    else if(tid == type_id_t::COMPLEX128)
        IOOP::template write<complex128>(writeable,obj); 
    else if(tid == type_id_t::STRING)
    {   
        //need to check if the object represents a unicode string
#if PY_MAJOR_VERSION >= 3
        IOOP::template write<string>(writeable,obj);
#else
        //in Python2 we have to take care about unicode. 
        //In this case we convert the unicode data to string first. 
        //This might not be the best approach
        object data;
        if(is_unicode(obj))
            data = unicode2str(obj);
        else    
            data = obj;

        IOOP::template write<string>(writeable,data);
#endif
    }
    else if(tid == type_id_t::BOOL)
        IOOP::template write<bool_t>(writeable,obj); 
    else 
        //raise an exception here if the datatype cannot be managed
        throw type_error(EXCEPTION_RECORD,"Datatype of writabel is unknown!");

}
