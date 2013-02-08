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
 * Definition of IO classes to read and write data.
 *
 * Created on: March 9, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#pragma once

#include "NXWrapperHelpers.hpp"


/*! 
\ingroup ioclasses  
\brief reads a single scalar

ScalarReader reads a single scalar form a readable object. For now only fields,
selections, and attributes expose the appropriate interface. The class provides
a static template method which reads the data and returns the result as a 
Python object.
*/
class ScalarReader
{
    public:
        /*! \brief read scalar data

        Reads scalar data of type T from a readable object and returns a native
        Python object as result.

        \tparam T data type to read
        \tparam OTYPE object type where to read data from
        \param readable instance of OTYPE from which to read data 
        \return native Python object
        */
        template<typename T,typename OTYPE> static 
            object read(const OTYPE &readable)
        {
            T value; //create a new instance where to store the data
            readable.read(value); //read data
            object o(value); //create python object
            return o;
        }
};

//-----------------------------------------------------------------------------
/*! 
\ingroup ioclasses  
\brief reads a single array 

Reads a single array from a readable object which might be either a selection,
a field, or an attribute object. The result is returned as a numpy array.
*/
class ArrayReader
{
    public:
        /*! \brief read array 

        Read a single array from the field.
        \tparam T data type to read
        \tparam OTYPE object type from which to read data
        \param readable instance of OTYPE from which to read data
        \return numpy array as Python object.
        */
        template<typename T,typename OTYPE> static 
            object read(const OTYPE &readable)
        {
            //create the numpy array which will store the data
            object narray = CreateNumpyArray<T>(readable.template shape<shape_t>());

            //create a reference array to the numpy arrays buffer 
            darray<T,rbuffer<T> > rarray = Numpy2RefArray<T>(narray);
            //read data to the numpy buffer
            readable.read(rarray);
            return narray;
        }
};

//-----------------------------------------------------------------------------
/*! 
\ingroup ioclasses
\brief write scalar data

Writes a scalar value from a Python object to a writeable object. 
*/
class ScalarWriter
{
    public:
        /*! \brief write scalar data

        Writes scalar data from object o to writable.
        \throws ShapeMissmatchError if o is not a scalar object
        \throws TypeError if type conversion fails
        \tparam T data type to write
        \tparam WTYPE writeable type
        \param writeable instance of WTYPE where to store data
        \param o object form which to write data
        */
        template<typename T,typename WTYPE> static
            void write(const WTYPE &writeable,const object &o)
        {
            //check if the data object is a numpy array and throw an exception
            //in this case
            /*
            if(PyArray_CheckExact(o.ptr()))
                throw ShapeMissmatchError(EXCEPTION_RECORD,
                        "Object is not a scalar!");
            */ 
            //extract the value to write - this will throw an exception if 
            //the operation fails.
            T value = extract<T>(o);
            writeable.write(value);
        }
};

//-----------------------------------------------------------------------------
/*! 
\ingroup ioclasses  
\brief write array data

Write array data to a writeable.
*/
class ArrayWriter
{
    private:
        /*!
        \brief write a numpy array to the writable object
        \tparam WTYPE writable type
        \param w instance of WTYPE
        \param o object representing a numpy array
        */
        template<typename WTYPE>
        static void _write_numpy_array(const WTYPE &w,const object &o)
        {
            //select the data type to use for writing the array data
            switch(PyArray_TYPE(o.ptr()))
            {
                case PyArray_UINT8:
                    w.write(Numpy2RefArray<uint8>(o));break;
                case PyArray_INT8:
                    w.write(Numpy2RefArray<int8>(o));break;
                case PyArray_UINT16:
                    w.write(Numpy2RefArray<uint16>(o));break;
                case PyArray_INT16:
                    w.write(Numpy2RefArray<int16>(o));break;
                case PyArray_UINT32:
                    w.write(Numpy2RefArray<uint32>(o)); break;
                case PyArray_INT32:
                    w.write(Numpy2RefArray<int32>(o));break;
                case PyArray_UINT64:
                    w.write(Numpy2RefArray<uint64>(o)); break;
                case PyArray_INT64:
                    w.write(Numpy2RefArray<int64>(o)); break;
                case PyArray_FLOAT32:
                    w.write(Numpy2RefArray<float32>(o)); break;
                case PyArray_FLOAT64:
                    w.write(Numpy2RefArray<float64>(o)); break;
                case PyArray_LONGDOUBLE:
                    w.write(Numpy2RefArray<float128>(o));break;
                case NPY_CFLOAT:
                    w.write(Numpy2RefArray<complex32>(o));break;
                case NPY_CDOUBLE:
                    w.write(Numpy2RefArray<complex64>(o)); break;
                case NPY_CLONGDOUBLE:
                    w.write(Numpy2RefArray<complex128>(o));break;
                case NPY_BOOL:
                    w.write(Numpy2RefArray<bool>(o)); break;
                default:
                    throw type_error(EXCEPTION_RECORD,
                    "Type of numpy array cannot be handled!");
            };

        }

        //---------------------------------------------------------------------
        /*!
        \brief broadcast a scalar to a field

        Broadcast a scalar value to the field.
        \tparam T type of the scalar
        \tparam WTYPE writable type
        \param w instance of WTYPE
        \param o object representing a scalar
        */
        template<typename T,typename WTYPE>
        static void _write_scalar(const WTYPE &w,const object &o)
        {
            //get writable parameters
            auto shape = w.template shape<shape_t>();

            T value = extract<T>(o)();
            darray<T> data(shape);
            std::fill(data.begin(),data.end(),value);
            w.write(data);
        }
    public:
        /*! \brief write array data

        Writes array data o to writeable w.
        \throws TypeError if o is not a numpy array or the datatype cannot be
        handled
        \tparam WTYPE writeable type
        \param w instance of WTYPE where to store data
        \param o numpy array object
        */
        template<typename T,typename WTYPE> static
            void write(const WTYPE &w,const object &o)
        {
           
            //check if the object from which to read data is an array
            if(!is_numpy_array(o))
                _write_scalar<T>(w,o);
            else
                _write_numpy_array(w,o);

        }
};



//-----------------------------------------------------------------------------
/*! 
\ingroup ioclasses  
\brief read all possible scalars from a readable

Function reads data from an readable object and returns a Python object with the 
result. The template parameter IOOp must be a type that implements the readable 
interface.
\throws TypeError if readable is of unknow data type
\param readable object from which to read data
\return a Python object with the data
*/
template<typename IOOP,typename OType> object io_read(const OType &readable)
{
    if(readable.type_id() == type_id_t::UINT8) 
        return IOOP::template read<uint8>(readable);
    if(readable.type_id() == type_id_t::INT8)  
        return IOOP::template read<int8>(readable);
    if(readable.type_id() == type_id_t::UINT16) 
        return IOOP::template read<uint16>(readable);
    if(readable.type_id() == type_id_t::INT16)  
        return IOOP::template read<int16>(readable);
    if(readable.type_id() == type_id_t::UINT32) 
        return IOOP::template read<uint32>(readable);
    if(readable.type_id() == type_id_t::INT32)  
        return IOOP::template read<int32>(readable);
    if(readable.type_id() == type_id_t::UINT64) 
        return IOOP::template read<uint64>(readable);
    if(readable.type_id() == type_id_t::INT64)  
        return IOOP::template read<int64>(readable);

    if(readable.type_id() == type_id_t::FLOAT32) 
        return IOOP::template read<float32>(readable);
    if(readable.type_id() == type_id_t::FLOAT64) 
        return IOOP::template read<float64>(readable);
    if(readable.type_id() == type_id_t::FLOAT128) 
        return IOOP::template read<float128>(readable);

    if(readable.type_id() == type_id_t::COMPLEX32) 
        return IOOP::template read<complex32>(readable);
    if(readable.type_id() == type_id_t::COMPLEX64) 
        return IOOP::template read<complex64>(readable);
    if(readable.type_id() == type_id_t::COMPLEX128) 
        return IOOP::template read<complex128>(readable);

    if(readable.type_id() == type_id_t::STRING) 
        return IOOP::template read<string>(readable);
    if(readable.type_id() == type_id_t::BOOL)
        return IOOP::template read<bool>(readable);

    throw type_error(EXCEPTION_RECORD,"Cannot handle field datatype!");
   
    return object();
}

//-----------------------------------------------------------------------------
/*! 
\ingroup ioclasses  
\brief write data 

Write data from a Python object to a writable object.
\throws TypeError if type of the writable cannot be handled
\tparam IOOP IO operation to use
\tparam OTYPE writeable type
\param writeable object to which data will be written
\param obj Python object form which data will be written
*/
template<typename IOOP,typename OTYPE> 
void io_write(const OTYPE &writeable,const object &obj)
{
    if(writeable.type_id() == type_id_t::UINT8)
    {
        IOOP::template write<uint8>(writeable,obj); return;
    }

    if(writeable.type_id() == type_id_t::INT8) 
    {
        IOOP::template write<int8>(writeable,obj); return;
    }

    if(writeable.type_id() == type_id_t::UINT16)
    {
        IOOP::template write<uint16>(writeable,obj); return;
    }

    if(writeable.type_id() == type_id_t::INT16) 
    {
        IOOP::template write<int16>(writeable,obj); return;
    }

    if(writeable.type_id() == type_id_t::UINT32) 
    {
        IOOP::template write<uint32>(writeable,obj); return;
    }

    if(writeable.type_id() == type_id_t::INT32) 
    {
        IOOP::template write<int32>(writeable,obj); return;
    }

    if(writeable.type_id() == type_id_t::UINT64) 
    {
        IOOP::template write<uint64>(writeable,obj); return;
    }

    if(writeable.type_id() == type_id_t::INT64)
    {
        IOOP::template write<int64>(writeable,obj); return;
    }
    
    if(writeable.type_id() == type_id_t::FLOAT32) 
    {
        IOOP::template write<float32>(writeable,obj); return;
    }

    if(writeable.type_id() == type_id_t::FLOAT64)
    {
        IOOP::template write<float64>(writeable,obj); return;
    }

    if(writeable.type_id() == type_id_t::FLOAT128) 
    {
        IOOP::template write<float128>(writeable,obj); return;
    }

    if(writeable.type_id() == type_id_t::COMPLEX32) 
    {
        IOOP::template write<complex32>(writeable,obj); return;
    }

    if(writeable.type_id() == type_id_t::COMPLEX64)
    { 
        IOOP::template write<complex64>(writeable,obj); return;
    }

    if(writeable.type_id() == type_id_t::COMPLEX128)
    {
        IOOP::template write<complex128>(writeable,obj); return;
    }

    if(writeable.type_id() == type_id_t::STRING)
    {   
        //need to check if the object represents a unicode string
        object data;
        if(is_unicode(obj))
            data = unicode2str(obj);
        else    
            data = obj;

        IOOP::template write<string>(writeable,data); return;
    }

    if(writeable.type_id() == type_id_t::BOOL)
    {
        IOOP::template write<bool>(writeable,obj); return;
    }

    //raise an exception here if the datatype cannot be managed
    throw type_error(EXCEPTION_RECORD,"Datatype of writabel is unknown!");

}
