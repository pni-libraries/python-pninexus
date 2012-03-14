#ifndef __NXOBJECTWRAPPER_HPP__
#define __NXOBJECTWRAPPER_HPP__

#include <pni/utils/Types.hpp>

#include "NXObjectMap.hpp"
#include "NXAttributeWrapper.hpp"
#include "AttributeIterator.hpp"
#include "AttributeCreator.hpp"



template<typename OType> class NXObjectWrapper
{   
    protected:
        //object is not defined private here. The intention of this class is 
        //not encapsulation but rather reducing the writing effort for the 
        //child classes by collecting here all common methods. 
        OType _object; //!< original object that shall be wrapped
    public:
        typedef NXAttributeWrapper<typename NXObjectMap<OType>::AttributeType>
            attribute_type; //!< type for attributes
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
        { }

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

        //----------------------------------------------------------------------
        //! obtain name
        String name() const
        {
            return _object.name();
        }

        //----------------------------------------------------------------------
        //! obtain path
        String path() const
        {
            return _object.path();
        }

        //----------------------------------------------------------------------
        //! get validity status
        bool is_valid() const
        {
            return _object.is_valid();
        }

        //----------------------------------------------------------------------
        //! close the object
        void close()
        {
            _object.close();
        }

        //---------------------------------------------------------------------

        attribute_type create_attribute(const String &name,const String
                &type_code,const object &shape=list())
            const
        {
            //first we need to decide wether we need a scalar or an array 
            //attribute
            list shape_list(shape);
            AttributeCreator<attribute_type>
                creator(name,List2Shape(list(shape)));

            return creator.create(this->_object,type_code);
        }

        //---------------------------------------------------------------------
        attribute_type open_attr(const String &n) const
        {
            return attribute_type(this->_object.attr(n));
        }

        //---------------------------------------------------------------------
        size_t nattrs() const
        {
            return this->_object.nattr();
        }

        //---------------------------------------------------------------------
        attribute_type open_attr_by_id(size_t i) const
        {
            return attribute_type(this->_object.attr(i));
        }

        //---------------------------------------------------------------------
        AttributeIterator<NXObjectWrapper<OType>,attribute_type> 
            get_attribute_iterator() const
        {
            return
                AttributeIterator<NXObjectWrapper<OType>,attribute_type>(*this);
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
        .add_property("nattrs",&NXObjectWrapper<OType>::nattrs)
        .def("attr",&NXObjectWrapper<OType>::create_attribute,("name","type_code",
                    arg("shape")=list()))
        .def("attr",&NXObjectWrapper<OType>::open_attr)
        .def("close",&NXObjectWrapper<OType>::close)
        .add_property("attributes",&NXObjectWrapper<OType>::get_attribute_iterator)
        ;
}



#endif
