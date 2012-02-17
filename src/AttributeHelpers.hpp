#ifndef __ATTRIBUTEHELPERS_HPP__
#define __ATTRIBUTEHELPERS_HPP__

template<typename OType,typename AType> 
AType create_attribute(const Otype &o,const String n,const TypeID &id)
{
    if(id == TypeID::UINT8)  return o.attr<UInt8>(n);
    if(id == TypeID::INT8)   return o.attr<Int8>(n);
    if(id == TypeID::UINT16) return o.attr<UInt16>(n);
    if(id == TypeID::INT16)  return o.attr<Int16>(n);
    if(id == TypeID::UINT32) return o.attr<UInt32>(n);
    if(id == TypeID::INT32)  return o.attr<Int32>(n);
    if(id == TypeID::UINT64) return o.attr<UInt64>(n);
    if(id == TypeID::INT64)  return o.attr<Int64>(n);

    if(id == TypeID::FLOAT32) return o.attr<Float32>(n);
    if(id == TypeID::FLOAT64) return o.attr<Float64>(n);
    if(id == TypeID::FLOAT128) return o.attr<Float64>(n);

    if(id == TypeID::STRING) return o.attr<String>(n);
}

template<typename OType,typename AType>
AType create_attribute(const OType &o,const String &n,const TypeID &id,
                             const Shape &s)
{
    if(id == TypeID::UINT8)  return o.attr<UInt8>(n,s);
    if(id == TypeID::INT8)   return o.attr<Int8>(n,s);
    if(id == TypeID::UINT16) return o.attr<UInt16>(n,s);
    if(id == TypeID::INT16)  return o.attr<Int16>(n,s);
    if(id == TypeID::UINT32) return o.attr<UInt32>(n,s);
    if(id == TypeID::INT32)  return o.attr<Int32>(n,s);
    if(id == TypeID::UINT64) return o.attr<UInt64>(n,s);
    if(id == TypeID::INT64)  return o.attr<Int64>(n,s);

    if(id == TypeID::FLOAT32) return o.attr<Float32>(n,s);
    if(id == TypeID::FLOAT64) return o.attr<Float64>(n,s);
    if(id == TypeID::FLOAT128) return o.attr<Float64>(n,s);

    if(id == TypeID::STRING) return o.attr<String>(n,s);
}


#endif
