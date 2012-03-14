#ifndef __NXOBJECTWRAPPER_HPP__
#define __NXOBJECTWRAPPER_HPP__

#include <pni/utils/Types.hpp>

#include "NXObjectMap.hpp"
#include "NXAttributeWrapper.hpp"
#include "AttributeIterator.hpp"



template<typename OType> class NXObjectWrapper
{   
    private:
        typedef NXAttributeWrapper<typename NXObjectMap<OType>::AttributeType>
            __attribute_type; //!< type for attributes
        /*! \brief create a scalar attribute

        Creates a scalar attribute of type T. The second template parameter is
        the type of the object used to create the attribute. This can be any
        object that exposes the attribute-interface.
        \param o object to create the attribute
        \param n name of the attribute
        \return attribute object
        */
        template<typename T,typename AOType> __attribute_type
            __create_scalar_attr(const AOType &o,const String &n) const
        {
            return attribute_type(o.template attr<T>(n));
        }

        /*! \brief create array attribute
        
        Create an array attribute of type T and shape s. The second template
        parameter determines the type of the object used to create the
        attribute. This can be any object that exposes the attribute-interface.
        \param o attribute creating object
        \param n name of the attribute
        \param s shape of the attribute
        \return attribute object
        */
        template<typename T,typename AOType> __attribute_type
            __create_array_attr(const AOType &o,const String &n,const Shape &s)
            const
        {
            return attribute_type(o.template attr<T>(n,s));
        }

        /*! \brief create scalar attribute

        create a scalar attribute of a datatype determined by typestr. 
        The template parameter determines the type of the object creating the
        attribute. This type must expose the attribute-interface.
        \param o attribute creating object
        \param n name of the attribute
        \param typestr Python type string determining the datatype
        \return attribute object
        */
        template<typename AOType> __attribute_type
            __create_scalar_attr(const AOType &o,const String &n,const String
                    &typestr) const;

        /*! \brief create array attribute

        create a scalar attribute of a datatype determined by typestr and shape
        s. The template parameter determines the type of the object creating the
        attribute. This type must expose the attribute-interface.
        \param o attribute creating object
        \param n name of the attribute
        \param typestr Python type string determining the datatype
        \param s shape of the array attribute
        \return attribute object
        */
        template<typename AOType> __attribute_type
            __create_array_attr(const AOType &,const String &n,const String
                    &typestr,const Shape &s) const;

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
            if(len(shape_list)==0){
                //create a scalar attribute
                return __create_scalar_attr(this->_object,name,type_code);
            }else{
                Shape s = List2Shape(shape_list);
                return __create_array_attr(this->_object,name,type_code,s);
            }
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

//===============implementation of non-inline methods===========================
template<typename OType>
template<typename AOType> 
typename NXObjectWrapper<OType>::__attribute_type NXObjectWrapper<OType>::
__create_scalar_attr(const AOType &o,const String &n,const String &typestr) const
{

    if(typestr == "string") return __create_scalar_attr<String>(o,n); 
    if(typestr == "int8")   return __create_scalar_attr<Int8>(o,n);
    if(typestr == "uint8")  return __create_scalar_attr<UInt8>(o,n);
    if(typestr == "int16")  return __create_scalar_attr<Int16>(o,n);
    if(typestr == "uint16") return __create_scalar_attr<UInt16>(o,n);
    if(typestr == "int32")  return __create_scalar_attr<Int32>(o,n);
    if(typestr == "uint32") return __create_scalar_attr<UInt32>(o,n);
    if(typestr == "int64")  return __create_scalar_attr<Int64>(o,n);
    if(typestr == "uint64") return __create_scalar_attr<UInt64>(o,n);

    if(typestr == "float32") return __create_scalar_attr<Float32>(o,n);
    if(typestr == "float64") return __create_scalar_attr<Float64>(o,n);
    if(typestr == "float128") return __create_scalar_attr<Float128>(o,n); 
    
    if(typestr == "complex64") return __create_scalar_attr<Complex32>(o,n);
    if(typestr == "complex128") return __create_scalar_attr<Complex64>(o,n);
    if(typestr == "complex256") return __create_scalar_attr<Complex128>(o,n);

    //here we should raise an exception
    TypeError error;
    error.issuer("template<typename OType> template<typename AOType> "
                 "typename NXObjectWrapper<OType>::__attribute_type "
                 "NXObjectWrapper<OType>::__create_scalar_attr("
                 "const AOType &o,const String &n,const String &typestr)"
                 " const");
    error.description(
            "Type string ("+typestr+") has no appropriate Nexus type!");
    throw(error);

    return attribute_type();
}

//-----------------------------------------------------------------------------
template<typename OType>
template<typename AOType> 
typename NXObjectWrapper<OType>::__attribute_type NXObjectWrapper<OType>::
__create_array_attr(const AOType &o,const String &n,const String &typestr,
        const Shape &s) const
{
    if(typestr == "string") return __create_array_attr<String>(o,n,s); 
    if(typestr == "int8")   return __create_array_attr<Int8>(o,n,s);
    if(typestr == "uint8")  return __create_array_attr<UInt8>(o,n,s);
    if(typestr == "int16")  return __create_array_attr<Int16>(o,n,s);
    if(typestr == "uint16") return __create_array_attr<UInt16>(o,n,s);
    if(typestr == "int32")  return __create_array_attr<Int32>(o,n,s);
    if(typestr == "uint32") return __create_array_attr<UInt32>(o,n,s);
    if(typestr == "int64")  return __create_array_attr<Int64>(o,n,s);
    if(typestr == "uint64") return __create_array_attr<UInt64>(o,n,s);

    if(typestr == "float32") return __create_array_attr<Float32>(o,n,s);
    if(typestr == "float64") return __create_array_attr<Float64>(o,n,s);
    if(typestr == "float128") return __create_array_attr<Float128>(o,n,s); 
    
    if(typestr == "complex64") return __create_array_attr<Complex32>(o,n,s);
    if(typestr == "complex128") return __create_array_attr<Complex64>(o,n,s);
    if(typestr == "complex256") return __create_array_attr<Complex128>(o,n,s);

    //here we should raise an exception
    TypeError error;
    error.issuer("template<typename OType> template<typename AOType> "
                 "typename NXObjectWrapper<OType>::__attribute_type "
                 "NXObjectWrapper<OType>:: __create_array_attr("
                 "const AOType &o,const String &n,const String &typestr,"
                 "const Shape &s) const");
    error.description(
            "Type string ("+typestr+") has no appropriate Nexus type!");
    throw(error);


    return attribute_type();
}

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
