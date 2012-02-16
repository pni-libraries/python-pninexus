/*
 * NXGroup.cpp
 *
 *  Created on: Jan 6, 2012
 *      Author: Eugen Wintersberger
 */




#include <boost/python.hpp>

#include <pni/utils/Types.hpp>
#include <pni/utils/Shape.hpp>
#include <pni/utils/ArrayObject.hpp>
#include <pni/utils/ScalarObject.hpp>
#include <pni/utils/NumericObject.hpp>


#include "../src/NXGroup.hpp"
#include "../src/h5/H5DeflateFilter.hpp"
#include "../src/NXFilter.hpp"
#include "../src/NXField.hpp"


using namespace pni::utils;
using namespace pni::nx;
using namespace pni::nx::h5;
using namespace boost::python;


template<typename GImp,typename FImp>
NXField<FImp> create_field(const NXGroup<GImp> &g,const TypeID &id,const String &n)
{
    if(id==TypeID::BINARY) return g.create_field<Binary>(n);
    if(id==TypeID::STRING) return g.create_field<String>(n);

    if(id==TypeID::UINT8)  return g.create_field<UInt8>(n);
    if(id==TypeID::INT8)   return g.create_field<Int8>(n);
    if(id==TypeID::UINT16)  return g.create_field<UInt16>(n);
    if(id==TypeID::INT16)   return g.create_field<Int16>(n);
    if(id==TypeID::UINT32)  return g.create_field<UInt32>(n);
    if(id==TypeID::INT32)   return g.create_field<Int32>(n);
    if(id==TypeID::UINT64)  return g.create_field<UInt64>(n);
    if(id==TypeID::INT64)   return g.create_field<Int64>(n);

    if(id==TypeID::FLOAT32) return g.create_field<Float32>(n);
    if(id==TypeID::FLOAT64) return g.create_field<Float64>(n);
    if(id==TypeID::FLOAT64) return g.create_field<Float64>(n);
    
}


#define GROUP_WRAPPER(wname,imp_type)\
    GROUPTYPE(imp_type) (NXGroup<imp_type>::*(wname ## __create_group))\
            (const String &) const = &NXGroup<imp_type>::create_group;\
    GROUPTYPE(imp_type) (NXGroup<imp_type>::*(wname ## __create_typed_group))\
            (const String &,const String &) const = &NXGroup<imp_type>::create_group;\
    void (NXGroup<imp_type>::*(wname ## __link_1))(const String &) const\
            =&NXGroup<imp_type>::link;\
    void (NXGroup<imp_type>::*(wname ## __link_2))\
            (const NXGroup<MAPTYPE(imp_type,GroupImpl)> &,const String &) const\
            =&NXGroup<imp_type>::link;\
    void (NXGroup<imp_type>::*(wname ## __link_3))(const String &,const String &) const\
            =&NXGroup<imp_type>::link;\
    \
    class_<NXGroup<imp_type>,bases<NXObject<imp_type> > >(#wname)\
        .def(init<>())\
        .def("create_group",(wname ## __create_group))\
        .def("create_group",(wname ## __create_typed_group))\
        .def("create_field",&create_field<imp_type,MAPTYPE(imp_type,FieldImpl)>)\
        .def("open",&NXGroup<imp_type>::open)\
        .def("close",&NXGroup<imp_type>::close)\
        .def("exists",&NXGroup<imp_type>::exists)\
        .def("remove",&NXGroup<imp_type>::remove)\
        .def("link",(wname ## __link_1))\
        .def("link",(wname ## __link_2))\
        .def("link",(wname ## __link_3))\
        ;

#define GROUP_WRAPPERNOCOP(wname,imp_type)\
    pni::nx::NXGroup<typename pni::nx::NXImpMap<imp_type::IMPCODE>::GroupImplementation>\
        (pni::nx::NXGroup<imp_type>::*(wname ## _create_group))\
        (const String &) const = &pni::nx::NXGroup<imp_type>::create_group;\
    pni::nx::NXGroup<typename pni::nx::NXImpMap<imp_type::IMPCODE>::GroupImplementation> \
        (pni::nx::NXGroup<imp_type>::*(wname ## _create_typed_group))\
        (const String &,const String &) const \
        =&pni::nx::NXGroup<imp_type>::create_group;\
    \
    pni::nx::NXNumericField<typename pni::nx::NXImpMap<imp_type::IMPCODE>::NumericFieldImplementation>\
        (pni::nx::NXGroup<imp_type>::*(wname ## _create_numfield1))\
        (const String &,TypeID,const Shape &,const String &,const String &)const \
          = &pni::nx::NXGroup<imp_type>::create_numericfield;\
    \
    pni::nx::NXNumericField<typename pni::nx::NXImpMap<imp_type::IMPCODE>::NumericFieldImplementation>\
        (pni::nx::NXGroup<imp_type>::*(wname ## _create_numfield_deflate1))\
        (const String &,TypeID,const Shape &,const String &,const String\
         &,pni::nx::NXFilter<H5DeflateFilter> &) const\
        =  &pni::nx::NXGroup<imp_type>::create_numericfield;\
    pni::nx::NXNumericField<typename pni::nx::NXImpMap<imp_type::IMPCODE>::NumericFieldImplementation>\
        (pni::nx::NXGroup<imp_type>::*(wname ## _create_numfield_lzf1))\
        (const String &,TypeID,const Shape &,const String &,const String\
         &,pni::nx::NXFilter<H5LZFFilter> &) const \
        = &pni::nx::NXGroup<imp_type>::create_numericfield;\
    \
    pni::nx::NXNumericField<typename pni::nx::NXImpMap<imp_type::IMPCODE>::NumericFieldImplementation>\
        (pni::nx::NXGroup<imp_type>::*(wname ## _create_numfield2))\
        (const ArrayObject &) const = &pni::nx::NXGroup<imp_type>::create_numericfield;\
    pni::nx::NXNumericField<typename pni::nx::NXImpMap<imp_type::IMPCODE>::NumericFieldImplementation>\
            (pni::nx::NXGroup<imp_type>::*(wname ## _create_numfield_deflate2))\
            (const ArrayObject &,pni::nx::NXFilter<H5DeflateFilter> &) const\
            = &pni::nx::NXGroup<imp_type>::create_numericfield;\
    pni::nx::NXNumericField<typename pni::nx::NXImpMap<imp_type::IMPCODE>::NumericFieldImplementation>\
            (pni::nx::NXGroup<imp_type>::*(wname ## _create_numfield_lzf2))\
            (const ArrayObject &,pni::nx::NXFilter<H5LZFFilter> &) const\
            = &pni::nx::NXGroup<imp_type>::create_numericfield;\
    \
    pni::nx::NXNumericField<typename pni::nx::NXImpMap<imp_type::IMPCODE>::NumericFieldImplementation>\
        (pni::nx::NXGroup<imp_type>::*(wname ## _create_numfield3))\
        (const String &,TypeID,const String &,const String &) const\
        = &pni::nx::NXGroup<imp_type>::create_numericfield;\
    pni::nx::NXNumericField<typename pni::nx::NXImpMap<imp_type::IMPCODE>::NumericFieldImplementation>\
        (pni::nx::NXGroup<imp_type>::*(wname ## _create_numfield4))\
        (const ScalarObject &) const =\
        &pni::nx::NXGroup<imp_type>::create_numericfield;\
    \
    class_<pni::nx::NXGroup<imp_type>,bases<pni::nx::NXObject<imp_type> >,boost::noncopyable >(#wname)\
        .def(init<>())\
        .def("create_group",(wname ## _create_group))\
        .def("create_group",(wname ## _create_typed_group))\
        .def("create_numericfield",(wname ## _create_numfield1))\
        .def("create_numericfield",(wname ## _create_numfield_deflate1))\
        .def("create_numericfield",(wname ## _create_numfield_lzf1))\
        .def("create_numericfield",(wname ## _create_numfield2))\
        .def("create_numericfield",(wname ## _create_numfield_deflate2))\
        .def("create_numericfield",(wname ## _create_numfield_lzf2))\
        .def("create_numericfield",(wname ## _create_numfield3))\
        .def("create_numericfield",(wname ## _create_numfield4))\
        .def("create_stringfield",&pni::nx::NXGroup<imp_type>::create_stringfield)\
        .def("create_binaryfield",&pni::nx::NXGroup<imp_type>::create_binaryfield)\
        .def("open",&pni::nx::NXGroup<imp_type>::open)\
        .def("close",&pni::nx::NXGroup<imp_type>::close)\
        .def("exists",&pni::nx::NXGroup<imp_type>::exists)\
        .def("remove",&pni::nx::NXGroup<imp_type>::remove)\
        ;
void wrap_nxgroup(){
    GROUP_WRAPPER(NXGroup_NXGroup,H5Group);
    //GROUP_WRAPPERNOCOP(NXGroup_NXFile,NXFileH5Implementation);
}
