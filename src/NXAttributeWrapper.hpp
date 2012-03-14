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
#include "NXIOOperations.hpp"

template<typename AttrType> class NXAttributeWrapper{
    private:
        AttrType _attribute;
    public:
        //===============constructors and destructor===========================
        //! default constructor
        NXAttributeWrapper():_attribute(){}

        //---------------------------------------------------------------------
        //! copy constructor
        NXAttributeWrapper(const NXAttributeWrapper<AttrType> &a):
            _attribute(a._attribute)
        {}

        //---------------------------------------------------------------------
        //! move constructor
        NXAttributeWrapper(NXAttributeWrapper<AttrType> &&a):
            _attribute(std::move(a._attribute))
        {}

        //---------------------------------------------------------------------
        //! copy constructor from implementation
        explicit NXAttributeWrapper(const AttrType &a):
            _attribute(a)
        {}

        //---------------------------------------------------------------------
        //! move constructor from implementation
        explicit NXAttributeWrapper(AttrType &&a):
            _attribute(std::move(a))
        {}

        //---------------------------------------------------------------------
        //! destructor
        ~NXAttributeWrapper(){}

        //=====================assignment operator=============================
        //! copy assignment
        NXAttributeWrapper<AttrType> &operator=
            (const NXAttributeWrapper<AttrType> &a)
        {
            if(this != &a) _attribute = a._attribute;
            return *this;
        }

        //---------------------------------------------------------------------
        //! move assignment
        NXAttributeWrapper<AttrType> &operator=
            (NXAttributeWrapper<AttrType> &&a)
        {
            if(this != &a) _attribute = std::move(a._attribute);
            return *this;
        }

        //==========================inquery methodes===========================
        //! get attribute shape
        list shape() const
        {
            return Shape2List(this->_attribute.shape());
        }

        //---------------------------------------------------------------------
        //! get attribute type id
        String type_id() const
        {
            return typeid2str(this->_attribute.type_id()); 
        }

        //---------------------------------------------------------------------
        //! close the attribute
        void close()
        {
            this->_attribute.close();
        }

        //----------------------------------------------------------------------
        bool is_valid() const
        {
            return this->_attribute.is_valid();
        }

        //----------------------------------------------------------------------
        String name() const
        {
            return this->_attribute.name();
        }

        //=========================read methods================================
        object read() const
        {
            if(this->_attribute.shape().rank() == 0){
                return io_read<ScalarReader>(this->_attribute);
            }else{
                return io_read<ArrayReader>(this->_attribute);
            }

            //should raise an exception here
            pni::nx::NXAttributeError error;
            error.issuer("template<typename AttrType> object "
                         "NXAttributeWrapper<AttrType>::read() const");
            error.description("Found no appropriate procedure to read this"
                              "attribute!");

            //this is only to avoid compiler warnings
            return object();
        }

        //=====================write methods===================================
        void write(object o) const
        {
            //before we can write an object we need to find out what 
            //it really i
            if(this->_attribute.shape().rank() == 0){
                io_write<ScalarWriter>(this->_attribute,o);
            }else if(PyArray_CheckExact(o.ptr())){

                ArrayWriter::write(this->_attribute,o);

            }else{
                //throw an exception here
                pni::nx::NXAttributeError error;
                error.issuer("template<typename AttrType> void"
                        "NXAttributeWrapper<AttrType>::write(object o)");
                error.description("Found no procedure to write this data!");
                throw(error);
            }

        }

};

template<typename AType> void wrap_nxattribute()
{
    class_<NXAttributeWrapper<AType> >("NXAttribute")
        .add_property("shape",&NXAttributeWrapper<AType>::shape)
        .add_property("type_id",&NXAttributeWrapper<AType>::type_id)
        .add_property("is_valid",&NXAttributeWrapper<AType>::is_valid)
        .add_property("name",&NXAttributeWrapper<AType>::name)
        .add_property("value",&NXAttributeWrapper<AType>::read,
                              &NXAttributeWrapper<AType>::write)
        .def("close",&NXAttributeWrapper<AType>::close)
        ;

}

#endif
