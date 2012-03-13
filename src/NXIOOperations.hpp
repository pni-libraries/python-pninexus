#ifndef __NXIOOPERATIONS_HPP__
#define __NXIOOPERATIONS_HPP__

#include "NXWrapperHelpers.hpp"


/*! \brief reads a single scalar

This reader reads a single scalar form a readable object (this might be either a
selection or a field object).
*/
class ScalarReader{
    public:
        template<typename T,typename OType> object run(const OType &readable)
        {
            T value;
            readable.read(value);
            object o(value);
            return o;
        }
};

/*! \brief reads a single array 

Reads a single array from a readable object which might be either a selection or
a field object.
*/
class ArrayReader{
    public:
        template<typename T,typename OType> object run(const OType &readable)
        {
            PyObject *ptr = CreateNumpyArray<T>(readable.shape());
            handle<> h(ptr);
            object o(h);
            Array<T,RefBuffer> rarray = Numpy2RefArray<T>(o);
            readable.read(rarray);
            return o;
        }
};

class ScalarWriter{
    public:
        template<typename T,typename WType>
            void write(const WType &writeable,const object &o)
        {
            T value = extract<T>(o);
            writeable.write(value);
        }
};

class ArrayWriter{
    public:
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

class ArrayBroadcastWriter{
    private:
        template<typename T,typename WType>  static
            void __write(const WType &w,const object &o)
        {
            Array<T,Buffer> a(w.shape());
            const T &value = extract<T>(o);
            a = value;
            w.write(a);
        }
    public:
        template<typename WType> static
            void write(const WType &w,const object &o)
        {
            //need to figure out the datatype used for o
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
        }
};


/*! \brief read all possible scalars from a readable

*/
template<typename IOOp,typename OType> object io_read(const OType &readable)
{
    IOOp operation;
    if(readable.type_id() == TypeID::UINT8) 
        return operation.template run<UInt8>(readable);
    if(readable.type_id() == TypeID::INT8)  
        return operation.template run<Int8>(readable);
    if(readable.type_id() == TypeID::UINT16) 
        return operation.template run<UInt16>(readable);
    if(readable.type_id() == TypeID::INT16)  
        return operation.template run<Int16>(readable);
    if(readable.type_id() == TypeID::UINT32) 
        return operation.template run<UInt32>(readable);
    if(readable.type_id() == TypeID::INT32)  
        return operation.template run<Int32>(readable);
    if(readable.type_id() == TypeID::UINT64) 
        return operation.template run<UInt64>(readable);
    if(readable.type_id() == TypeID::INT64)  
        return operation.template run<Int64>(readable);

    if(readable.type_id() == TypeID::FLOAT32) 
        return operation.template run<Float32>(readable);
    if(readable.type_id() == TypeID::FLOAT64) 
        return operation.template run<Float64>(readable);
    if(readable.type_id() == TypeID::FLOAT128) 
        return operation.template run<Float128>(readable);

    if(readable.type_id() == TypeID::COMPLEX32) 
        return operation.template run<Complex32>(readable);
    if(readable.type_id() == TypeID::COMPLEX64) 
        return operation.template run<Complex64>(readable);
    if(readable.type_id() == TypeID::COMPLEX128) 
        return operation.template run<Complex128>(readable);

    if(readable.type_id() == TypeID::STRING) 
        return operation.template run<String>(readable);
   
    //should raise an exception here
    return object();
}

template<typename IOOp,typename OType> void io_write(const OType &writeable,const object &obj)
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

}

#endif
