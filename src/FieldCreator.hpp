
#include<pni/utils/Types.hpp>
#include<pni/utils/Exceptions.hpp>

using namespace pni::utils;

/*! \brief field creator class

This class creates Field objects according to the configuration of the class.
*/
template<typename FieldT> class FieldCreator{
    private:
        String __n; //!< name of the field
        Shape __s;  //!< shape of the field
        Shape __cs; //!< chunk shape of the field
    public:
        //---------------------------------------------------------------------
        //! constructor 
        FieldCreator(const String &n,const Shape &s,const Shape &cs):
            __n(n),__s(s),__cs(cs){}
       
        //---------------------------------------------------------------------
        FieldCreator(const String &n):
            __n(n),__s(),__cs(){}

        //---------------------------------------------------------------------
        FieldCreator(const String &n,const Shape &s):
            __n(n),__s(s),__cs(){}

        //---------------------------------------------------------------------
        template<typename T,typename OType> 
            FieldT create(const OType &o) const;

        //---------------------------------------------------------------------
        template<typename OType> 
            FieldT create(const OType &o,const String &type_str) const;
};

//-----------------------------------------------------------------------------
template<typename FieldT>
template<typename T,typename OType> 
    FieldT FieldCreator<FieldT>::create(const OType &o) const
{
    if(__s.size() == 0){
        //create a scalar field
        return FieldT(o.template create_field<T>(__n));
    }else{
        //create an array field
        if(__cs.size() == 0){
            //create a field with automatic chunk size
            return FieldT(o.template create_field<T>(__n,__s));
        }else{
            //create a field with custom chunk 
            return FieldT(o.template create_field<T>(__n,__s,__cs));
        }
    }
}

//------------------------------------------------------------------------------
template<typename FieldT> 
template<typename OType> FieldT 
FieldCreator<FieldT>::create(const OType &o,const String &type_code) const
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
