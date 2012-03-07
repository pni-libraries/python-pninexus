//helper functions to create wrappers

#include "NXWrapperHelpers.hpp"

//-----------------------------------------------------------------------------
String typeid2str(const TypeID &tid)
{
    if(tid == TypeID::STRING) return "string";
    if(tid == TypeID::UINT8) return "uint8";
    if(tid == TypeID::INT8)  return "int8";
    if(tid == TypeID::UINT16) return "uint16";
    if(tid == TypeID::INT16)  return "int16";
    if(tid == TypeID::UINT32) return "uint32";
    if(tid == TypeID::INT32)  return "int32";
    if(tid == TypeID::UINT64) return "uint64";
    if(tid == TypeID::INT64) return "int64";

    if(tid == TypeID::FLOAT32) return "float32";
    if(tid == TypeID::FLOAT64) return "float64";
    if(tid == TypeID::FLOAT128) return "float128";

    if(tid == TypeID::COMPLEX32) return "complex64";
    if(tid == TypeID::COMPLEX64) return "complex128";
    if(tid == TypeID::COMPLEX128) return "complex256";

    return "none";
}

//-----------------------------------------------------------------------------
list Shape2List(const Shape &s){
    list l;

    if(s.rank() == 0) return l;

    for(size_t i=0;i<s.rank();i++) l.append(s[i]);

    return l;

}

//-----------------------------------------------------------------------------
Shape List2Shape(const list &l){
    Shape s;

    return s;

}
