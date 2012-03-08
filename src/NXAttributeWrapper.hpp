#ifndef __NXATTRIBUTEWRAPPER_HPP__
#define __NXATTRIBUTEWRAPPER_HPP__

#include <pni/utils/Types.hpp>
#include <pni/utils/Shape.hpp>
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

        //--------------read methods-------------------------------
        object read() const
        {


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
        .def("write",&NXAttributeWrapper<AType>::write)
        .def("close",&NXAttributeWrapper<AType>::close)
        ;

}

#endif
