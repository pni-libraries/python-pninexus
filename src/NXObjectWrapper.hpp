#ifndef __NXOBJECTWRAPPER_HPP__
#define __NXOBJECTWRAPPER_HPP__

#include <pni/utils/Types.hpp>

#include "NXObjectMap.hpp"
#include "NXAttributeWrapper.hpp"



template<typename OType> class NXObjectWrapper
{
    protected:
        OType _object;
    public:
        //================constructors and destructor==========================
        //! default constructor
        NXObjectWrapper():_object(){}

        //---------------------------------------------------------------------
        //! copy constructor
        NXObjectWrapper(const NXObjectWrapper<OType> &o):
            _object(o._object)
        { }

        //---------------------------------------------------------------------
        //! move constructor
        NXObjectWrapper(NXObjectWrapper<OType> &&o):
            _object(std::move(o._object)) 
        {
        }

        //---------------------------------------------------------------------
        //! copy conversion constructor from wrapped object
        explicit NXObjectWrapper(const OType &o):_object(o){}

        //---------------------------------------------------------------------
        //! move conversion constructor from wrapped object
        explicit NXObjectWrapper(OType &&o):_object(std::move(o)){
        }

        //---------------------------------------------------------------------
        //! destructor
        virtual ~NXObjectWrapper()
        {
            //close the object on wrapper destruction
            this->close();
        }

        //==================assignment operators===============================
        //! copy conversion assignment from wrapped type
        NXObjectWrapper<OType> &operator=(const OType &o)
        {
            if(&_object != &o) _object = o;
            return *this;
        }

        //---------------------------------------------------------------------
        //! move conversion assignment from wrapped type
        NXObjectWrapper<OType> &operator=(OType &&o)
        {
            if(&_object != &o) _object = std::move(o);
            return *this;
        }

        //---------------------------------------------------------------------
        //move assignment
        NXObjectWrapper<OType> &operator=(NXObjectWrapper<OType> &&o)
        {
            if(this != &o) _object = std::move(o._object);
            return *this;
        }

        //---------------------------------------------------------------------
        //copy assignment
        NXObjectWrapper<OType> &operator=(const NXObjectWrapper<OType> &o)
        {
            if(this != &o) _object = o._object;
            return *this;
        }


        //======================object methods=================================

        //! obtain base name
        String base() const
        {
            return _object.base();
        }

        //! obtain name
        String name() const
        {
            return _object.name();
        }

        //! obtain path
        String path() const
        {
            return _object.path();
        }

        //! get validity status
        bool is_valid() const
        {
            return _object.is_valid();
        }

        //! close the object
        void close()
        {
            _object.close();
        }

#define ATTRIBUTE_CREATOR(pytype,type,name)\
        if(type_str == pytype)\
            return attr_type(this->_object.template attr<type>(name));

        NXAttributeWrapper<typename NXObjectMap<OType>::AttributeType>  
            scalar_attr(const String &name,const String &type_str)
        {
            typedef NXAttributeWrapper<typename NXObjectMap<OType>::AttributeType > attr_type;

            ATTRIBUTE_CREATOR("string",String,name);
            ATTRIBUTE_CREATOR("int8",Int8,name);
            ATTRIBUTE_CREATOR("uint8",UInt8,name);
            ATTRIBUTE_CREATOR("int16",Int16,name);
            ATTRIBUTE_CREATOR("uint16",UInt16,name);
            ATTRIBUTE_CREATOR("int32",Int32,name);
            ATTRIBUTE_CREATOR("uint32",UInt32,name);
            ATTRIBUTE_CREATOR("int64",Int64,name);
            ATTRIBUTE_CREATOR("uint64",UInt64,name);

            ATTRIBUTE_CREATOR("float32",Float32,name);
            ATTRIBUTE_CREATOR("float64",Float64,name);
            ATTRIBUTE_CREATOR("float128",Float128,name);

            ATTRIBUTE_CREATOR("complex64",Complex32,name);
            ATTRIBUTE_CREATOR("complex128",Complex64,name);
            ATTRIBUTE_CREATOR("complex256",Complex128,name);


            //should raise here an exception if something goes wrong

        }

#define ARRAY_ATTRIBUTE_CREATOR(pytype,type,name)\
        if(type_str == pytype)\
            return attr_type(this->_object.template attr<type>(name,s));

        NXAttributeWrapper<typename NXObjectMap<OType>::AttributeType> 
            array_attr(const String &name,const String &type_str,const list &l)
        {

            typedef NXAttributeWrapper<typename NXObjectMap<OType>::AttributeType > attr_type;
            Shape s= List2Shape(l);
            std::cout<<s<<std::endl;
            ARRAY_ATTRIBUTE_CREATOR("string",String,name);
            ARRAY_ATTRIBUTE_CREATOR("int8",Int8,name);
            ARRAY_ATTRIBUTE_CREATOR("uint8",UInt8,name);
            ARRAY_ATTRIBUTE_CREATOR("int16",Int16,name);
            ARRAY_ATTRIBUTE_CREATOR("uint16",UInt16,name);
            ARRAY_ATTRIBUTE_CREATOR("int32",Int32,name);
            ARRAY_ATTRIBUTE_CREATOR("uint32",UInt32,name);
            ARRAY_ATTRIBUTE_CREATOR("int64",Int64,name);
            ARRAY_ATTRIBUTE_CREATOR("uint64",UInt64,name);

            ARRAY_ATTRIBUTE_CREATOR("float32",Float32,name);
            ARRAY_ATTRIBUTE_CREATOR("float64",Float64,name);
            ARRAY_ATTRIBUTE_CREATOR("float128",Float128,name);

            ARRAY_ATTRIBUTE_CREATOR("complex64",Complex32,name);
            ARRAY_ATTRIBUTE_CREATOR("complex128",Complex64,name);
            ARRAY_ATTRIBUTE_CREATOR("complex256",Complex128,name);


        }

        NXAttributeWrapper<typename NXObjectMap<OType>::AttributeType>
            open_attr(const String &n) const
        {
            typedef NXAttributeWrapper<typename
                NXObjectMap<OType>::AttributeType> attr_type;

            return attr_type(this->_object.attr(n));
        }

};

//template function wrapping a single NXObject 
//type. 

template<typename OType> void wrap_nxobject(const String &class_name)
{
    class_<NXObjectWrapper<OType> >(class_name.c_str())
        .def(init<>())
        .add_property("name",&NXObjectWrapper<OType>::name)
        .add_property("path",&NXObjectWrapper<OType>::path)
        .add_property("base",&NXObjectWrapper<OType>::base)
        .add_property("is_valid",&NXObjectWrapper<OType>::is_valid)
        .def("attr",&NXObjectWrapper<OType>::scalar_attr)
        .def("attr",&NXObjectWrapper<OType>::array_attr)
        .def("attr",&NXObjectWrapper<OType>::open_attr)
        .def("close",&NXObjectWrapper<OType>::close)
        ;
}



#endif
