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
 * Definition of IO classes to read and write data.
 *
 * Created on: March 9, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#ifndef __NXIOOPERATIONS_HPP__
#define __NXIOOPERATIONS_HPP__

#include "NXWrapperHelpers.hpp"

//-----------------------------------------------------------------------------
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
            DArray<T,RBuffer<T> > rarray = Numpy2RefArray<T>(narray);
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
                case NPY_UBYTE:
                    w.write(Numpy2RefArray<UInt8>(o));break;
                case NPY_BYTE:
                    w.write(Numpy2RefArray<Int8>(o));break;
                case NPY_USHORT:
                    w.write(Numpy2RefArray<UInt16>(o));break;
                case NPY_SHORT:
                    w.write(Numpy2RefArray<Int16>(o));break;
                case NPY_UINT:
                    w.write(Numpy2RefArray<UInt32>(o)); break;
                case NPY_INT:
                    w.write(Numpy2RefArray<Int32>(o));break;
                case NPY_ULONG:
                    w.write(Numpy2RefArray<UInt64>(o)); break;
                case NPY_LONG:
                    w.write(Numpy2RefArray<Int64>(o)); break;
                case NPY_FLOAT:
                    w.write(Numpy2RefArray<Float32>(o)); break;
                case NPY_DOUBLE:
                    w.write(Numpy2RefArray<Float64>(o)); break;
                case NPY_LONGDOUBLE:
                    w.write(Numpy2RefArray<Float128>(o));break;
                case NPY_CFLOAT:
                    w.write(Numpy2RefArray<Complex32>(o));break;
                case NPY_CDOUBLE:
                    w.write(Numpy2RefArray<Complex64>(o)); break;
                case NPY_CLONGDOUBLE:
                    w.write(Numpy2RefArray<Complex128>(o));break;
                case NPY_BOOL:
                    w.write(Numpy2RefArray<Bool>(o)); break;
                default:
                    throw TypeError(EXCEPTION_RECORD,
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
            DArray<T> data(shape);
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
            if(!PyArray_CheckExact(o.ptr()))
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
    if(readable.type_id() == TypeID::UINT8) 
        return IOOP::template read<UInt8>(readable);
    if(readable.type_id() == TypeID::INT8)  
        return IOOP::template read<Int8>(readable);
    if(readable.type_id() == TypeID::UINT16) 
        return IOOP::template read<UInt16>(readable);
    if(readable.type_id() == TypeID::INT16)  
        return IOOP::template read<Int16>(readable);
    if(readable.type_id() == TypeID::UINT32) 
        return IOOP::template read<UInt32>(readable);
    if(readable.type_id() == TypeID::INT32)  
        return IOOP::template read<Int32>(readable);
    if(readable.type_id() == TypeID::UINT64) 
        return IOOP::template read<UInt64>(readable);
    if(readable.type_id() == TypeID::INT64)  
        return IOOP::template read<Int64>(readable);

    if(readable.type_id() == TypeID::FLOAT32) 
        return IOOP::template read<Float32>(readable);
    if(readable.type_id() == TypeID::FLOAT64) 
        return IOOP::template read<Float64>(readable);
    if(readable.type_id() == TypeID::FLOAT128) 
        return IOOP::template read<Float128>(readable);

    if(readable.type_id() == TypeID::COMPLEX32) 
        return IOOP::template read<Complex32>(readable);
    if(readable.type_id() == TypeID::COMPLEX64) 
        return IOOP::template read<Complex64>(readable);
    if(readable.type_id() == TypeID::COMPLEX128) 
        return IOOP::template read<Complex128>(readable);

    if(readable.type_id() == TypeID::STRING) 
        return IOOP::template read<String>(readable);
    if(readable.type_id() == TypeID::BOOL)
        return IOOP::template read<Bool>(readable);

    throw TypeError(EXCEPTION_RECORD,"Cannot handle field datatype!");
   
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
    if(writeable.type_id() == TypeID::UINT8)
    {
        IOOP::template write<UInt8>(writeable,obj); return;
    }

    if(writeable.type_id() == TypeID::INT8) 
    {
        IOOP::template write<Int8>(writeable,obj); return;
    }

    if(writeable.type_id() == TypeID::UINT16)
    {
        IOOP::template write<UInt16>(writeable,obj); return;
    }

    if(writeable.type_id() == TypeID::INT16) 
    {
        IOOP::template write<Int16>(writeable,obj); return;
    }

    if(writeable.type_id() == TypeID::UINT32) 
    {
        IOOP::template write<UInt32>(writeable,obj); return;
    }

    if(writeable.type_id() == TypeID::INT32) 
    {
        IOOP::template write<Int32>(writeable,obj); return;
    }

    if(writeable.type_id() == TypeID::UINT64) 
    {
        IOOP::template write<UInt64>(writeable,obj); return;
    }

    if(writeable.type_id() == TypeID::INT64)
    {
        IOOP::template write<Int64>(writeable,obj); return;
    }
    
    if(writeable.type_id() == TypeID::FLOAT32) 
    {
        IOOP::template write<Float32>(writeable,obj); return;
    }

    if(writeable.type_id() == TypeID::FLOAT64)
    {
        IOOP::template write<Float64>(writeable,obj); return;
    }

    if(writeable.type_id() == TypeID::FLOAT128) 
    {
        IOOP::template write<Float128>(writeable,obj); return;
    }

    if(writeable.type_id() == TypeID::COMPLEX32) 
    {
        IOOP::template write<Complex32>(writeable,obj); return;
    }

    if(writeable.type_id() == TypeID::COMPLEX64)
    { 
        IOOP::template write<Complex64>(writeable,obj); return;
    }

    if(writeable.type_id() == TypeID::COMPLEX128)
    {
        IOOP::template write<Complex128>(writeable,obj); return;
    }

    if(writeable.type_id() == TypeID::STRING)
    {   
        //need to check if the object represents a unicode string
        object data;
        if(is_unicode(obj))
            data = unicode2str(obj);
        else    
            data = obj;

        IOOP::template write<String>(writeable,data); return;
    }

    if(writeable.type_id() == TypeID::BOOL)
    {
        IOOP::template write<Bool>(writeable,obj); return;
    }

    //raise an exception here if the datatype cannot be managed
    throw TypeError(EXCEPTION_RECORD,"Datatype of writabel is unknown!");

}

#endif
