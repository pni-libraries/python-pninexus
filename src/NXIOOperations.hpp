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
class ScalarReader{
    public:
        /*! \brief read scalar data

        Reads scalar data of type T (first template parameter) and returns a 
        native Python object as result. The type of the readable is determined 
        by the second template parameter.
        \param readable an object of type OType with readable interface
        \return native Python object
        */
        template<typename T,typename OType> object read(const OType &readable)
        {
            T value;
            readable.read(value);
            object o(value);
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
class ArrayReader{
    public:
        /*! \brief read array 

        Read a single array from the field.
        \param readable an object of type OType with readable interface
        \return nyumpy array as Python object.
        */
        template<typename T,typename OType> object read(const OType &readable)
        {
            PyObject *ptr = CreateNumpyArray<T>(readable.shape());
            handle<> h(ptr);
            object o(h);
            Array<T,RefBuffer> rarray = Numpy2RefArray<T>(o);
            readable.read(rarray);
            return o;
        }
};

//-----------------------------------------------------------------------------
/*! 
\ingroup ioclasses
\brief write scalar data

Writes a scalar value from a Python object to a writeable object. 
*/
class ScalarWriter{
    public:
        /*! \brief write scalar data

        Writes scalar data from object o to writable.
        \param writeable object where to store data
        \param o object form which to write data
        */
        template<typename T,typename WType>
            void write(const WType &writeable,const object &o)
        {
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
class ArrayWriter{
    public:
        /*! \brief write array data

        Writes array data o to writeable w.
        \throws TypeError if o is not a numpy array or the datatype cannot be
        handled
        \param w writeable object
        \param o numpy array object
        */
        template<typename WType> static
            void write(const WType &w,const object &o)
        {
            
            if(!PyArray_CheckExact(o.ptr())){
                TypeError error;
                error.issuer("template<typename WType> static void "
                        "ArrayWriter::write(const WType &w,const "
                        "object &o)");
                error.description("Python object is not a numpy array!");
                throw error;
            }

            switch(PyArray_TYPE(o.ptr())){
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
                default:
                    TypeError error;
                    error.issuer("template<typename WType> static void "
                            "ArrayWriter::write(const WType &w,const "
                            "object &o)");
                    error.description("Type of numpy array cannot be "
                            "handled!");
                    throw error;
            };
        }
};

//-----------------------------------------------------------------------------
/*! 
\ingroup ioclasses  
\brief broadcast array writer

Writes a single scalar value to all positions of a multidimensional writeable.
The actual implementation of this writer is not very efficient as it requires 
the allocation of an intermediate Array object. 
This should be changed in future.
*/
class ArrayBroadcastWriter{
    private:
        /*! \brief write data to writeable
    
        Write data to the field. 
        */
        template<typename T,typename WType>  static
            void __write(const WType &w,const object &o)
        {
            Array<T,Buffer> a(w.shape());
            const T &value = extract<T>(o);
            a = value;
            w.write(a);
        }
    public:
        /*! \brief write data

        Write scalar data from o to the writable w.
        throws TypeError if o is of unknown type
        \param w writeable object
        \param o scalar Python object from which to write data
        */
        template<typename WType> static
            void write(const WType &w,const object &o)
        {
            //need to figure out the datatype used for o
            //Python does not support that many types. Thus the check can be
            //short.
            if(PyInt_Check(o.ptr())){
                __write<Int64>(w,o);
                return;
            }
            if(PyLong_Check(o.ptr())){
                __write<Int64>(w,o);
                return;
            }
            if(PyFloat_Check(o.ptr())){
                __write<Float64>(w,o);
                return;
            }
            if(PyComplex_Check(o.ptr())){
                __write<Complex64>(w,o);
                return;
            }

            //need here a string type

            TypeError error;
            error.issuer("template<typename WType> static void "
                    "ArrayBroadcastWriter::write(const WType &w,"
                    "const object &o)");
            error.description("Python object is of unknown type!");
            throw(error);
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
template<typename IOOp,typename OType> object io_read(const OType &readable)
{
    IOOp operation;
    if(readable.type_id() == TypeID::UINT8) 
        return operation.template read<UInt8>(readable);
    if(readable.type_id() == TypeID::INT8)  
        return operation.template read<Int8>(readable);
    if(readable.type_id() == TypeID::UINT16) 
        return operation.template read<UInt16>(readable);
    if(readable.type_id() == TypeID::INT16)  
        return operation.template read<Int16>(readable);
    if(readable.type_id() == TypeID::UINT32) 
        return operation.template read<UInt32>(readable);
    if(readable.type_id() == TypeID::INT32)  
        return operation.template read<Int32>(readable);
    if(readable.type_id() == TypeID::UINT64) 
        return operation.template read<UInt64>(readable);
    if(readable.type_id() == TypeID::INT64)  
        return operation.template read<Int64>(readable);

    if(readable.type_id() == TypeID::FLOAT32) 
        return operation.template read<Float32>(readable);
    if(readable.type_id() == TypeID::FLOAT64) 
        return operation.template read<Float64>(readable);
    if(readable.type_id() == TypeID::FLOAT128) 
        return operation.template read<Float128>(readable);

    if(readable.type_id() == TypeID::COMPLEX32) 
        return operation.template read<Complex32>(readable);
    if(readable.type_id() == TypeID::COMPLEX64) 
        return operation.template read<Complex64>(readable);
    if(readable.type_id() == TypeID::COMPLEX128) 
        return operation.template read<Complex128>(readable);

    if(readable.type_id() == TypeID::STRING) 
        return operation.template read<String>(readable);

    TypeError error;
    error.issuer("template<typename IOOp,typename OType> object "
            "io_read(const OType &readable)");
    error.description("Cannot handle field datatype!");
    throw(error);
   
    return object();
}

//-----------------------------------------------------------------------------
/*! 
\ingroup ioclasses  
\brief write data 

Write data from a Python object to a writable object.
\throws TypeError if type of the writable cannot be handled
\param writeable object to which data will be written
\param obj Python object form which data will be written
*/
template<typename IOOp,typename OType> 
void io_write(const OType &writeable,const object &obj)
{
    IOOp operation;
    if(writeable.type_id() == TypeID::UINT8)
    {
        operation.template write<UInt8>(writeable,obj);
        return;
    }

    if(writeable.type_id() == TypeID::INT8) 
    {
        operation.template write<Int8>(writeable,obj);
        return;
    }

    if(writeable.type_id() == TypeID::UINT16)
    {
        operation.template write<UInt16>(writeable,obj);
        return;
    }

    if(writeable.type_id() == TypeID::INT16) 
    {
        operation.template write<Int16>(writeable,obj);
        return;
    }

    if(writeable.type_id() == TypeID::UINT32) 
    {
        operation.template write<UInt32>(writeable,obj);
        return;
    }

    if(writeable.type_id() == TypeID::INT32) 
    {
        operation.template write<Int32>(writeable,obj);
        return;
    }

    if(writeable.type_id() == TypeID::UINT64) 
    {
        operation.template write<UInt64>(writeable,obj);
        return;
    }

    if(writeable.type_id() == TypeID::INT64)
    {
        operation.template write<Int64>(writeable,obj);
        return;
    }
    
    if(writeable.type_id() == TypeID::FLOAT32) 
    {
        operation.template write<Float32>(writeable,obj);
        return;
    }

    if(writeable.type_id() == TypeID::FLOAT64)
    {
        operation.template write<Float64>(writeable,obj);
        return;
    }

    if(writeable.type_id() == TypeID::FLOAT128) 
    {
        operation.template write<Float128>(writeable,obj);
        return;
    }

    if(writeable.type_id() == TypeID::COMPLEX32) 
    {
        operation.template write<Complex32>(writeable,obj);
        return;
    }

    if(writeable.type_id() == TypeID::COMPLEX64)
    { 
        operation.template write<Complex64>(writeable,obj);
        return;
    }

    if(writeable.type_id() == TypeID::COMPLEX128)
    {
        operation.template write<Complex128>(writeable,obj);
        return;
    }

    if(writeable.type_id() == TypeID::STRING)
    {    
        operation.template write<String>(writeable,obj);
        return;
    }

    //raise an exception here if the datatype cannot be managed
    TypeError error;
    error.issuer("template<typename IOOp,typename OType> void io_write"
            "(const OType &writeable,const object &obj)");
    error.description("Datatype of writabel is unknown!");
    throw(error);

}

#endif
