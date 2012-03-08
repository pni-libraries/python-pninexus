#ifndef __NXATTRIBUTEWRAPPER_HPP__
#define __NXATTRIBUTEWRAPPER_HPP__

extern "C"{
#include<numpy/arrayobject.h>
}

#include <pni/utils/Types.hpp>
#include <pni/utils/Shape.hpp>
#include <pni/utils/Array.hpp>
using namespace pni::utils;

#include "NXWrapperHelpers.hpp"

template<typename AttrType> class NXAttributeWrapper{
    private:
        AttrType _attribute;
    public:
        //--------------constructors and destructor----------------
        //! default constructor
        NXAttributeWrapper():_attribute(){}

        //! copy constructor
        NXAttributeWrapper(const NXAttributeWrapper<AttrType> &a):
            _attribute(a._attribute)
        {}

        //! move constructor
        NXAttributeWrapper(NXAttributeWrapper<AttrType> &&a):
            _attribute(std::move(a._attribute))
        {}

        //! copy constructor from implementation
        explicit NXAttributeWrapper(const AttrType &a):
            _attribute(a)
        {}

        //! move constructor from implementation
        explicit NXAttributeWrapper(AttrType &&a):
            _attribute(std::move(a))
        {}

        //! destructor
        ~NXAttributeWrapper(){}

        //--------------assignment operator------------------------
        //! copy assignment
        NXAttributeWrapper<AttrType> &operator=
            (const NXAttributeWrapper<AttrType> &a)
        {
            if(this != &a) _attribute = a._attribute;
            return *this;
        }

        //! move assignment
        NXAttributeWrapper<AttrType> &operator=
            (NXAttributeWrapper<AttrType> &&a)
        {
            if(this != &a) _attribute = std::move(a._attribute);
            return *this;
        }

        //----------------standard methodes------------------------
        //! get attribute shape
        list shape() const
        {
            return Shape2List(this->_attribute.shape());
        }

        //! get attribute type id
        String type_id() const
        {
            return typeid2str(this->_attribute.type_id()); 
        }

        //! close the attribute
        void close()
        {
            this->_attribute.close();
        }

        bool is_valid() const
        {
            return this->_attribute.is_valid();
        }

#define READ_SCALAR_ATTRIBUTE(typeid,type)\
        if(this->_attribute.type_id() == typeid){\
            type value = this->_attribute.template read<type>();\
            object o(value);\
            return o;\
        }

#define READ_ARRAY_ATTRIBUTE(typeid,type)\
        if(this->_attribute.type_id() == typeid){\
                PyObject *ptr =\
                    CreateNumpyArray<type>(this->_attribute.shape());\
                handle<> h(ptr);\
                object o(h);\
                Array<type,RefBuffer> rarray = Numpy2RefArray<type>(o);\
                this->_attribute.read(rarray);\
                return o;\
        }

        //--------------read methods-------------------------------
        object read() const
        {
            if(this->_attribute.shape().rank() == 0){
                READ_SCALAR_ATTRIBUTE(TypeID::UINT8,UInt8);
                READ_SCALAR_ATTRIBUTE(TypeID::INT8,Int8);
                READ_SCALAR_ATTRIBUTE(TypeID::UINT16,UInt16);
                READ_SCALAR_ATTRIBUTE(TypeID::INT16,Int16);
                READ_SCALAR_ATTRIBUTE(TypeID::UINT32,UInt32);
                READ_SCALAR_ATTRIBUTE(TypeID::INT32,Int32);
                READ_SCALAR_ATTRIBUTE(TypeID::UINT64,UInt64);
                READ_SCALAR_ATTRIBUTE(TypeID::INT64,Int64);

                READ_SCALAR_ATTRIBUTE(TypeID::FLOAT32,Float32);
                READ_SCALAR_ATTRIBUTE(TypeID::FLOAT64,Float64);
                READ_SCALAR_ATTRIBUTE(TypeID::FLOAT128,Float128);
                READ_SCALAR_ATTRIBUTE(TypeID::COMPLEX32,Complex32);
                READ_SCALAR_ATTRIBUTE(TypeID::COMPLEX64,Complex64);
                READ_SCALAR_ATTRIBUTE(TypeID::COMPLEX128,Complex128);
                
                READ_SCALAR_ATTRIBUTE(TypeID::STRING,String);
            }else{
                //we need to read array data
                READ_ARRAY_ATTRIBUTE(TypeID::UINT8,UInt8);
                READ_ARRAY_ATTRIBUTE(TypeID::INT8,Int8);
                READ_ARRAY_ATTRIBUTE(TypeID::UINT16,UInt16);
                READ_ARRAY_ATTRIBUTE(TypeID::INT16,Int16);
                READ_ARRAY_ATTRIBUTE(TypeID::UINT32,UInt32);
                READ_ARRAY_ATTRIBUTE(TypeID::INT32,Int32);
                READ_ARRAY_ATTRIBUTE(TypeID::UINT64,UInt64);
                READ_ARRAY_ATTRIBUTE(TypeID::INT64,Int64);

                READ_ARRAY_ATTRIBUTE(TypeID::FLOAT32,Float32);
                READ_ARRAY_ATTRIBUTE(TypeID::FLOAT64,Float64);
                READ_ARRAY_ATTRIBUTE(TypeID::FLOAT128,Float128);
                READ_ARRAY_ATTRIBUTE(TypeID::COMPLEX32,Complex32);
                READ_ARRAY_ATTRIBUTE(TypeID::COMPLEX64,Complex64);
                READ_ARRAY_ATTRIBUTE(TypeID::COMPLEX128,Complex128);
            }
        }

#define WRITE_SCALAR(typid,type)\
        if(this->_attribute.type_id() == typid){\
            type value = extract<type>(o);\
            this->_attribute.write(value);\
        }

        //-------------write methods-------------------------------
        void write(object o) const
        {
            //before we can write an object we need to find out what 
            //it really i
            if(this->_attribute.shape().rank() == 0){
                //we need to write a scalar
                WRITE_SCALAR(TypeID::STRING,String);
                WRITE_SCALAR(TypeID::UINT8,UInt8);
                WRITE_SCALAR(TypeID::INT8,Int8);
                WRITE_SCALAR(TypeID::UINT16,UInt16);
                WRITE_SCALAR(TypeID::INT16,Int16);
                WRITE_SCALAR(TypeID::UINT32,UInt32);
                WRITE_SCALAR(TypeID::INT32,Int32);
                WRITE_SCALAR(TypeID::UINT64,UInt64);
                WRITE_SCALAR(TypeID::INT64,Int64);
                WRITE_SCALAR(TypeID::FLOAT32,Float32);
                WRITE_SCALAR(TypeID::FLOAT64,Float64);
                WRITE_SCALAR(TypeID::FLOAT128,Float128);
                WRITE_SCALAR(TypeID::COMPLEX32,Complex32);
                WRITE_SCALAR(TypeID::COMPLEX64,Complex64);
                WRITE_SCALAR(TypeID::COMPLEX128,Complex128);
            }else{
                //now things become a bit more difficult
                //check if object is a numpy array
                if(!PyArray_CheckExact(o.ptr())){
                    std::cerr<<"Object is not a numpy array!"<<std::endl;
                    return;
                }
               
                switch(PyArray_TYPE(o.ptr())){
                    case NPY_UBYTE:
                        this->_attribute.write(Numpy2RefArray<UInt8>(o));
                        break;
                    case NPY_BYTE:
                        this->_attribute.write(Numpy2RefArray<Int8>(o));
                        break;
                    case NPY_USHORT:
                        this->_attribute.write(Numpy2RefArray<UInt16>(o));
                        break;
                    case NPY_SHORT:
                        this->_attribute.write(Numpy2RefArray<Int16>(o));
                        break;
                    case NPY_UINT:
                        this->_attribute.write(Numpy2RefArray<UInt32>(o));
                        break;
                    case NPY_INT:
                        this->_attribute.write(Numpy2RefArray<Int32>(o));
                        break;
                    case NPY_ULONG:
                        this->_attribute.write(Numpy2RefArray<UInt64>(o));
                        break;
                    case NPY_LONG:
                        this->_attribute.write(Numpy2RefArray<Int64>(o));
                        break;
                    case NPY_FLOAT:
                        this->_attribute.write(Numpy2RefArray<Float32>(o));
                        break;
                    case NPY_DOUBLE:
                        this->_attribute.write(Numpy2RefArray<Float64>(o));
                        break;
                    case NPY_LONGDOUBLE:
                        this->_attribute.write(Numpy2RefArray<Float128>(o));
                        break;
                    case NPY_CFLOAT:
                        this->_attribute.write(Numpy2RefArray<Complex32>(o));
                        break;
                    case NPY_CDOUBLE:
                        this->_attribute.write(Numpy2RefArray<Complex64>(o));
                        break;
                    case NPY_CLONGDOUBLE:
                        this->_attribute.write(Numpy2RefArray<Complex128>(o));
                        break;
                    default:
                        std::cerr<<"Array is of unkown type!"<<std::endl;

                };

            }

        }

};

template<typename AType> void wrap_nxattribute()
{
    class_<NXAttributeWrapper<AType> >("NXAttribute")
        .add_property("shape",&NXAttributeWrapper<AType>::shape)
        .add_property("type_id",&NXAttributeWrapper<AType>::type_id)
        .add_property("is_valid",&NXAttributeWrapper<AType>::is_valid)
        .def("read",&NXAttributeWrapper<AType>::read)
        .add_property("value",&NXAttributeWrapper<AType>::read,
                              &NXAttributeWrapper<AType>::write)
        .def("write",&NXAttributeWrapper<AType>::write)
        .def("close",&NXAttributeWrapper<AType>::close)
        ;

}

#endif
