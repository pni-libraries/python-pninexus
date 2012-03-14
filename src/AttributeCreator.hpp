#ifndef __ATTRIBUTECREATOR_HPP__
#define __ATTRIBUTECREATOR_HPP__

#include<pni/utils/Types.hpp>
#include<pni/utils/Exceptions.hpp>

using namespace pni::utils;

/*! \brief attribute creator class

This class creates attribute objects according to the configuration of the class.
*/
template<typename AttrT> class AttributeCreator{
    private:
        String __n; //!< name of the field
        Shape __s;  //!< shape of the field
    public:
        //---------------------------------------------------------------------
        AttributeCreator(const String &n):
            __n(n),__s(){}

        //---------------------------------------------------------------------
        AttributeCreator(const String &n,const Shape &s):
            __n(n),__s(s){}

        //---------------------------------------------------------------------
        template<typename T,typename OType> 
            AttrT create(const OType &o) const;

        //---------------------------------------------------------------------
        template<typename OType> 
            AttrT create(const OType &o,const String &type_str) const;
};

//-----------------------------------------------------------------------------
template<typename AttrT>
template<typename T,typename OType> 
    AttrT AttributeCreator<AttrT>::create(const OType &o) const
{
    if(__s.rank() == 0){
        //create a scalar attribute
        return AttrT(o.template attr<T>(__n,true));
    }else{
        //create a field with custom chunk 
        return AttrT(o.template attr<T>(__n,__s,true));
    }
}

//------------------------------------------------------------------------------
template<typename AttrT> 
template<typename OType> AttrT 
AttributeCreator<AttrT>::create(const OType &o,const String &type_code) const
{
    if(type_code == "uint8") return this->create<UInt8>(o);
    if(type_code == "int8")  return this->create<Int8>(o);
    if(type_code == "uint16") return this->create<UInt16>(o);
    if(type_code == "int16")  return this->create<Int16>(o);
    if(type_code == "uint32") return this->create<UInt32>(o);
    if(type_code == "int32")  return this->create<Int32>(o);
    if(type_code == "uint64") return this->create<UInt64>(o);
    if(type_code == "int64")  return this->create<Int64>(o);

    if(type_code == "float32") return this->create<Float32>(o);
    if(type_code == "float64") return this->create<Float64>(o);
    if(type_code == "float128") return this->create<Float128>(o);
    
    if(type_code == "complex64") return this->create<Complex32>(o);
    if(type_code == "complex128") return this->create<Complex64>(o);
    if(type_code == "complex256") return this->create<Complex128>(o);

    if(type_code == "string") return this->create<String>(o);

    //raise an exception here
    TypeError error;
    error.issuer("template<typename FieldT> template<typename OType> FieldT "
                 "FieldCreator<FieldT>::create(const OType &o,const String &"
                 "type_code) const");
    error.description("Cannot create field with type-code ("+type_code+")!");
    throw(error);
}

#endif
