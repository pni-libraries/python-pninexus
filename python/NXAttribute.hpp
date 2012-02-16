#ifndef __NXATTRIBUTEPY_HPP__
#define __NXATTRIBUTEPY_HPP__

#define WRITE_SIMPLE_ATTR_PTR(wname,imp_type,type,ifix)\
    void (NXAttribute<imp_type>::*(wname ## __write_ ## ifix ## _simple))\
            (const type &) const= &NXAttribute<imp_type>::write;\

#define WRITE_SIMPLE_ATTR_METHOD(wname,mname,ifix)\
    .def(#mname,(wname ## __write_ ## ifix ## _simple))

#define WRITE_SCALAR_ATTR_PTR(wname,imp_type,type,ifix)\
    void (NXAttribute<imp_type>::*(wname ## __write_ ## ifix ## _scalar))\
            (const Scalar<type> &) const= &NXAttribute<imp_type>::write;\

#define WRITE_SCALAR_ATTR_METHOD(wname,mname,ifix)\
    .def(#mname,(wname ## __write_ ## ifix ## _scalar))

#define WRITE_ARRAY_ATTR_PTR(wname,imp_type,type,ifix)\
    void (NXAttribute<imp_type>::*(wname ## __write_ ## ifix ## _array))\
            (const Array<type,Buffer> &) const= &NXAttribute<imp_type>::write;\

#define WRITE_ARRAY_ATTR_METHOD(wname,mname,ifix)\
    .def(#mname,(wname ## __write_ ## ifix ## _array))

#define NXATTRIBUTE_WRAPPER(wname,imp_type)\
    WRITE_SIMPLE_ATTR_PTR(wname,imp_type,UInt8,ui8);\
    WRITE_SIMPLE_ATTR_PTR(wname,imp_type,Int8,i8);\
    WRITE_SIMPLE_ATTR_PTR(wname,imp_type,UInt16,ui16);\
    WRITE_SIMPLE_ATTR_PTR(wname,imp_type,Int16,i16);\
    WRITE_SIMPLE_ATTR_PTR(wname,imp_type,UInt32,ui32);\
    WRITE_SIMPLE_ATTR_PTR(wname,imp_type,Int32,i32);\
    WRITE_SIMPLE_ATTR_PTR(wname,imp_type,UInt64,ui64);\
    WRITE_SIMPLE_ATTR_PTR(wname,imp_type,Int64,i64);\
    WRITE_SIMPLE_ATTR_PTR(wname,imp_type,Float32,f32);\
    WRITE_SIMPLE_ATTR_PTR(wname,imp_type,Float64,f64);\
    WRITE_SIMPLE_ATTR_PTR(wname,imp_type,Float128,f128);\
    WRITE_SIMPLE_ATTR_PTR(wname,imp_type,String,str);\
    WRITE_SCALAR_ATTR_PTR(wname,imp_type,UInt8,ui8);\
    WRITE_SCALAR_ATTR_PTR(wname,imp_type,Int8,i8);\
    WRITE_SCALAR_ATTR_PTR(wname,imp_type,UInt16,ui16);\
    WRITE_SCALAR_ATTR_PTR(wname,imp_type,Int16,i16);\
    WRITE_SCALAR_ATTR_PTR(wname,imp_type,UInt32,ui32);\
    WRITE_SCALAR_ATTR_PTR(wname,imp_type,Int32,i32);\
    WRITE_SCALAR_ATTR_PTR(wname,imp_type,UInt64,ui64);\
    WRITE_SCALAR_ATTR_PTR(wname,imp_type,Int64,i64);\
    WRITE_SCALAR_ATTR_PTR(wname,imp_type,Float32,f32);\
    WRITE_SCALAR_ATTR_PTR(wname,imp_type,Float64,f64);\
    WRITE_SCALAR_ATTR_PTR(wname,imp_type,Float128,f128);\
    WRITE_ARRAY_ATTR_PTR(wname,imp_type,UInt8,ui8);\
    WRITE_ARRAY_ATTR_PTR(wname,imp_type,Int8,i8);\
    WRITE_ARRAY_ATTR_PTR(wname,imp_type,UInt16,ui16);\
    WRITE_ARRAY_ATTR_PTR(wname,imp_type,Int16,i16);\
    WRITE_ARRAY_ATTR_PTR(wname,imp_type,UInt32,ui32);\
    WRITE_ARRAY_ATTR_PTR(wname,imp_type,Int32,i32);\
    WRITE_ARRAY_ATTR_PTR(wname,imp_type,UInt64,ui64);\
    WRITE_ARRAY_ATTR_PTR(wname,imp_type,Int64,i64);\
    WRITE_ARRAY_ATTR_PTR(wname,imp_type,Float32,f32);\
    WRITE_ARRAY_ATTR_PTR(wname,imp_type,Float64,f64);\
    WRITE_ARRAY_ATTR_PTR(wname,imp_type,Float128,f128);\
	class_<NXAttribute<imp_type> >(#wname)\
            .def(init<>())\
            .add_property("type_id",&NXAttribute<imp_type>::type_id)\
            .add_property("shape",&NXAttribute<imp_type>::shape)\
            .def("close",&NXAttribute<imp_type>::close)\
            WRITE_SIMPLE_ATTR_METHOD(wname,__write_simple_str,str)\
            WRITE_SIMPLE_ATTR_METHOD(wname,__write_simple_ui8,ui8)\
            WRITE_SIMPLE_ATTR_METHOD(wname,__write_simple_i8,i8)\
            WRITE_SIMPLE_ATTR_METHOD(wname,__write_simple_ui16,ui16)\
            WRITE_SIMPLE_ATTR_METHOD(wname,__write_simple_i16,i16)\
            WRITE_SIMPLE_ATTR_METHOD(wname,__write_simple_ui32,ui32)\
            WRITE_SIMPLE_ATTR_METHOD(wname,__write_simple_i32,i32)\
            WRITE_SIMPLE_ATTR_METHOD(wname,__write_simple_ui64,ui64)\
            WRITE_SIMPLE_ATTR_METHOD(wname,__write_simple_i64,i64)\
            WRITE_SIMPLE_ATTR_METHOD(wname,__write_simple_f32,f32)\
            WRITE_SIMPLE_ATTR_METHOD(wname,__write_simple_f64,f64)\
            WRITE_SIMPLE_ATTR_METHOD(wname,__write_simple_f128,f128)\
            WRITE_SCALAR_ATTR_METHOD(wname,__write_scalar_ui8,ui8)\
            WRITE_SCALAR_ATTR_METHOD(wname,__write_scalar_i8,i8)\
            WRITE_SCALAR_ATTR_METHOD(wname,__write_scalar_ui16,ui16)\
            WRITE_SCALAR_ATTR_METHOD(wname,__write_scalar_i16,i16)\
            WRITE_SCALAR_ATTR_METHOD(wname,__write_scalar_ui32,ui32)\
            WRITE_SCALAR_ATTR_METHOD(wname,__write_scalar_i32,i32)\
            WRITE_SCALAR_ATTR_METHOD(wname,__write_scalar_ui64,ui64)\
            WRITE_SCALAR_ATTR_METHOD(wname,__write_scalar_i64,i64)\
            WRITE_SCALAR_ATTR_METHOD(wname,__write_scalar_f32,f32)\
            WRITE_SCALAR_ATTR_METHOD(wname,__write_scalar_f64,f64)\
            WRITE_SCALAR_ATTR_METHOD(wname,__write_scalar_f128,f128)\
            WRITE_ARRAY_ATTR_METHOD(wname,__write_array_ui8,ui8)\
            WRITE_ARRAY_ATTR_METHOD(wname,__write_array_i8,i8)\
            WRITE_ARRAY_ATTR_METHOD(wname,__write_array_ui16,ui16)\
            WRITE_ARRAY_ATTR_METHOD(wname,__write_array_i16,i16)\
            WRITE_ARRAY_ATTR_METHOD(wname,__write_array_ui32,ui32)\
            WRITE_ARRAY_ATTR_METHOD(wname,__write_array_i32,i32)\
            WRITE_ARRAY_ATTR_METHOD(wname,__write_array_ui64,ui64)\
            WRITE_ARRAY_ATTR_METHOD(wname,__write_array_i64,i64)\
            WRITE_ARRAY_ATTR_METHOD(wname,__write_array_f32,f32)\
            WRITE_ARRAY_ATTR_METHOD(wname,__write_array_f64,f64)\
            WRITE_ARRAY_ATTR_METHOD(wname,__write_array_f128,f128)\
            
            ;

#endif
