/*
 * NXObject.hpp
 *
 *  Created on: Jan 8, 2012
 *      Author: Eugen Wintersberger
 */


#ifndef __NXOBJECT_HPP__
#define __NXOBJECT_HPP__

using namespace pni::nx;

#define ATTRTYPE(imp_type)\
    NXAttribute<MAPTYPE(imp_type,AttributeImpl)>

#define CREATE_SCALAR_ATTR_PTR(wname,imp_type,atype,ifix)\
    ATTRTYPE(imp_type) \
        (NXObject<imp_type>::*(wname ## __create_ ## ifix ## _scalar_attr))\
        (const String &) const \
        = &NXObject<imp_type>::attr<atype>;\

#define CREATE_ARRAY_ATTR_PTR(wname,imp_type,atype,ifix)\
    ATTRTYPE(imp_type) \
        (NXObject<imp_type>::*(wname ## __create_ ## ifix ## _array_attr))\
        (const String &,const Shape &) const \
        = &NXObject<imp_type>::attr<atype>;\

#define CREATE_SCALAR_ATTR_METHOD(name,wname,ifix)\
    .def(name,(wname ## __create_ ## ifix ## _scalar_attr))

#define CREATE_ARRAY_ATTR_METHOD(name,wname,ifix)\
    .def(name,(wname ## __create_ ## ifix ## _array_attr))


#define NXOBJECT_WRAPPER(wname,imp_type)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,String,str)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,UInt8,ui8)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,Int8,i8)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,UInt16,ui16)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,Int16,i16)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,UInt32,ui32)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,Int32,i32)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,UInt64,ui64)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,Int64,i64)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,Float32,f32)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,Float64,f64)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,Float128,f128)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,String,str)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,UInt8,ui8)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,Int8,i8)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,UInt16,ui16)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,Int16,i16)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,UInt32,ui32)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,Int32,i32)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,UInt64,ui64)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,Int64,i64)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,Float32,f32)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,Float64,f64)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,Float128,f128)\
    ATTRTYPE(imp_type) (NXObject<imp_type>::*(wname ## __get_attribute))\
            (const String &n) const = &NXObject<imp_type>::attr;\
	class_<pni::nx::NXObject<imp_type> >(#wname)\
            .def(init<>())\
            .add_property("is_valid",&pni::nx::NXObject<imp_type>::is_valid)\
            .add_property("path",&pni::nx::NXObject<imp_type>::path)\
            .add_property("name",&pni::nx::NXObject<imp_type>::name)\
            .add_property("base",&pni::nx::NXObject<imp_type>::base)\
            .add_property("nattrs",&pni::nx::NXObject<imp_type>::nattr)\
            .def("del_attr",&pni::nx::NXObject<imp_type>::del_attr)\
            .def("attr_names",&pni::nx::NXObject<imp_type>::attr_names)\
            .def("close",&pni::nx::NXObject<imp_type>::close)\
            .def("has_attr",&pni::nx::NXObject<imp_type>::has_attr)\
            .def("__get_attribute",(wname ## __get_attribute))\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_str_attr",wname,str)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_ui8_attr",wname,ui8)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_i8_attr",wname,i8)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_ui16_attr",wname,ui16)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_i16_attr",wname,i16)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_ui32_attr",wname,ui32)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_i32_attr",wname,i32)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_ui64_attr",wname,ui64)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_i64_attr",wname,i64)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_f32_attr",wname,f32)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_f64_attr",wname,f64)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_f128_attr",wname,f128)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_str_attr",wname,str)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_ui8_attr",wname,ui8)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_i8_attr",wname,i8)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_ui16_attr",wname,ui16)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_i16_attr",wname,i16)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_ui32_attr",wname,ui32)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_i32_attr",wname,i32)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_ui64_attr",wname,ui64)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_i64_attr",wname,i64)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_f32_attr",wname,f32)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_f64_attr",wname,f64)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_f128_attr",wname,f128)\
			;


#define NXOBJECT_WRAPPERNOCOP(wname,imp_type)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,String,str)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,UInt8,ui8)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,Int8,i8)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,UInt16,ui16)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,Int16,i16)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,UInt32,ui32)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,Int32,i32)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,UInt64,ui64)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,Int64,i64)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,Float32,f32)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,Float64,f64)\
    CREATE_SCALAR_ATTR_PTR(wname,imp_type,Float128,f128)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,String,str)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,UInt8,ui8)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,Int8,i8)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,UInt16,ui16)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,Int16,i16)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,UInt32,ui32)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,Int32,i32)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,UInt64,ui64)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,Int64,i64)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,Float32,f32)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,Float64,f64)\
    CREATE_ARRAY_ATTR_PTR(wname,imp_type,Float128,f128)\
    ATTRTYPE(imp_type) (NXObject<imp_type>::*(wname ## __get_attribute))\
            (const String &n) const = &NXObject<imp_type>::attr;\
	class_<pni::nx::NXObject<imp_type>,boost::noncopyable >(#wname)\
            .def(init<>())\
            .add_property("is_valid",&pni::nx::NXObject<imp_type>::is_valid)\
            .add_property("path",&pni::nx::NXObject<imp_type>::path)\
            .add_property("name",&pni::nx::NXObject<imp_type>::name)\
            .add_property("base",&pni::nx::NXObject<imp_type>::base)\
            .add_property("nattrs",&pni::nx::NXObject<imp_type>::nattr)\
            .def("del_attr",&pni::nx::NXObject<imp_type>::del_attr)\
            .def("attr_names",&pni::nx::NXObject<imp_type>::attr_names)\
            .def("close",&pni::nx::NXObject<imp_type>::close)\
            .def("has_attr",&pni::nx::NXObject<imp_type>::has_attr)\
            .def("__get_attribute",(wname ## __get_attribute))\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_str_attr",wname,str)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_ui8_attr",wname,ui8)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_i8_attr",wname,i8)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_ui16_attr",wname,ui16)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_i16_attr",wname,i16)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_ui32_attr",wname,ui32)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_i32_attr",wname,i32)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_ui64_attr",wname,ui64)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_i64_attr",wname,i64)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_f32_attr",wname,f32)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_f64_attr",wname,f64)\
            CREATE_SCALAR_ATTR_METHOD("__create_scalar_f128_attr",wname,f128)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_str_attr",wname,str)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_ui8_attr",wname,ui8)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_i8_attr",wname,i8)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_ui16_attr",wname,ui16)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_i16_attr",wname,i16)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_ui32_attr",wname,ui32)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_i32_attr",wname,i32)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_ui64_attr",wname,ui64)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_i64_attr",wname,i64)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_f32_attr",wname,f32)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_f64_attr",wname,f64)\
            CREATE_ARRAY_ATTR_METHOD("__create_array_f128_attr",wname,f128)\
			;

#endif
