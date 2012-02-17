#ifndef __NXATTRIBUTEWRAPPER_HPP__
#define __NXATTRIBUTEWRAPPER_HPP__

#include <pni/utils/Types.hpp>
using namespace pni::utils;


template<typename AType> void wrap_nxattribute()
{
    void (AType::*NXAttribute_write_ui8)(const UInt8 &) const = &AType::write;
    void (AType::*NXAttribute_write_i8)(const Int8 &) const = &AType::write;
    void (AType::*NXAttribute_write_ui16)(const UInt16 &) const = &AType::write;
    void (AType::*NXAttribute_write_i16)(const Int16 &) const = &AType::write;
    void (AType::*NXAttribute_write_ui32)(const UInt32 &) const = &AType::write;
    void (AType::*NXAttribute_write_i32)(const Int32 &) const = &AType::write;
    void (AType::*NXAttribute_write_ui64)(const UInt64 &) const = &AType::write;
    void (AType::*NXAttribute_write_i64)(const Int64 &) const = &AType::write;
    void (AType::*NXAttribute_write_f32)(const Float32 &) const = &AType::write;
    void (AType::*NXAttribute_write_f64)(const Float64 &) const = &AType::write;
    void (AType::*NXAttribute_write_f128)(const Float128 &) const = &AType::write;
    void (AType::*NXAttribute_write_str)(const String &) const = &AType::write;
    void (AType::*NXAttribute_write_scalar_ui8)(const Scalar<UInt8> &) const = &AType::write;
    void (AType::*NXAttribute_write_scalar_i8)(const Scalar<Int8> &) const = &AType::write;
    void (AType::*NXAttribute_write_scalar_ui16)(const Scalar<UInt16> &) const = &AType::write;
    void (AType::*NXAttribute_write_scalar_i16)(const Scalar<Int16> &) const = &AType::write;
    void (AType::*NXAttribute_write_scalar_ui32)(const Scalar<UInt32> &) const = &AType::write;
    void (AType::*NXAttribute_write_scalar_i32)(const Scalar<Int32> &) const = &AType::write;
    void (AType::*NXAttribute_write_scalar_ui64)(const Scalar<UInt64> &) const = &AType::write;
    void (AType::*NXAttribute_write_scalar_i64)(const Scalar<Int64> &) const = &AType::write;
    void (AType::*NXAttribute_write_scalar_f32)(const Scalar<Float32> &) const = &AType::write;
    void (AType::*NXAttribute_write_scalar_f64)(const Scalar<Float64> &) const = &AType::write;
    void (AType::*NXAttribute_write_scalar_f128)(const Scalar<Float128> &) const = &AType::write;
    void (AType::*NXAttribute_write_array_ui8)(const Array<UInt8,Buffer> &) const = &AType::write;
    void (AType::*NXAttribute_write_array_i8)(const Array<Int8,Buffer> &) const = &AType::write;
    void (AType::*NXAttribute_write_array_ui16)(const Array<UInt16,Buffer> &) const = &AType::write;
    void (AType::*NXAttribute_write_array_i16)(const Array<Int16,Buffer> &) const = &AType::write;
    void (AType::*NXAttribute_write_array_ui32)(const Array<UInt32,Buffer> &) const = &AType::write;
    void (AType::*NXAttribute_write_array_i32)(const Array<Int32,Buffer> &) const = &AType::write;
    void (AType::*NXAttribute_write_array_ui64)(const Array<UInt64,Buffer> &) const = &AType::write;
    void (AType::*NXAttribute_write_array_i64)(const Array<Int64,Buffer> &) const = &AType::write;
    void (AType::*NXAttribute_write_array_f32)(const Array<Float32,Buffer> &) const = &AType::write;
    void (AType::*NXAttribute_write_array_f64)(const Array<Float64,Buffer> &) const = &AType::write;
    void (AType::*NXAttribute_write_array_f128)(const Array<Float128,Buffer> &) const = &AType::write;
    void (AType::*NXAttribute_read_ui8)(UInt8 &) const = &AType::read;
    void (AType::*NXAttribute_read_i8)(Int8 &) const = &AType::read;
    void (AType::*NXAttribute_read_ui16)(UInt16 &) const = &AType::read;
    void (AType::*NXAttribute_read_i16)(Int16 &) const = &AType::read;
    void (AType::*NXAttribute_read_ui32)(UInt32 &) const = &AType::read;
    void (AType::*NXAttribute_read_i32)(Int32 &) const = &AType::read;
    void (AType::*NXAttribute_read_ui64)(UInt64 &) const = &AType::read;
    void (AType::*NXAttribute_read_i64)(Int64 &) const = &AType::read;
    void (AType::*NXAttribute_read_f32)(Float32 &) const = &AType::read;
    void (AType::*NXAttribute_read_f64)(Float32 &) const = &AType::read;
    void (AType::*NXAttribute_read_f128)(Float128 &) const = &AType::read;
    void (AType::*NXAttribute_read_str)(String &) const = &AType::read;
    void (AType::*NXAttribute_read_scalar_ui8)(Scalar<UInt8> &) const = &AType::read;
    void (AType::*NXAttribute_read_scalar_i8)(Scalar<Int8> &) const = &AType::read;
    void (AType::*NXAttribute_read_scalar_ui16)(Scalar<UInt16> &) const = &AType::read;
    void (AType::*NXAttribute_read_scalar_i16)(Scalar<Int16> &) const = &AType::read;
    void (AType::*NXAttribute_read_scalar_ui32)(Scalar<UInt32> &) const = &AType::read;
    void (AType::*NXAttribute_read_scalar_i32)(Scalar<Int32> &) const = &AType::read;
    void (AType::*NXAttribute_read_scalar_ui64)(Scalar<UInt64> &) const = &AType::read;
    void (AType::*NXAttribute_read_scalar_i64)(Scalar<Int64> &) const = &AType::read;
    void (AType::*NXAttribute_read_scalar_f32)(Scalar<Float32> &) const = &AType::read;
    void (AType::*NXAttribute_read_scalar_f64)(Scalar<Float64> &) const = &AType::read;
    void (AType::*NXAttribute_read_scalar_f128)(Scalar<Float128> &) const = &AType::read;
    void (AType::*NXAttribute_read_array_ui8)(Array<UInt8,Buffer> &) const = &AType::read;
    void (AType::*NXAttribute_read_array_i8)(Array<Int8,Buffer> &) const = &AType::read;
    void (AType::*NXAttribute_read_array_ui16)(Array<UInt16,Buffer> &) const = &AType::read;
    void (AType::*NXAttribute_read_array_i16)(Array<Int16,Buffer> &) const = &AType::read;
    void (AType::*NXAttribute_read_array_ui32)(Array<UInt32,Buffer> &) const = &AType::read;
    void (AType::*NXAttribute_read_array_i32)(Array<Int32,Buffer> &) const = &AType::read;
    void (AType::*NXAttribute_read_array_ui64)(Array<UInt64,Buffer> &) const = &AType::read;
    void (AType::*NXAttribute_read_array_i64)(Array<Int64,Buffer> &) const = &AType::read;
    void (AType::*NXAttribute_read_array_f32)(Array<Float32,Buffer> &) const = &AType::read;
    void (AType::*NXAttribute_read_array_f64)(Array<Float64,Buffer> &) const = &AType::read;
    void (AType::*NXAttribute_read_array_f128)(Array<Float128,Buffer> &) const = &AType::read;
    class_<AType>("NXAttribute")
        .def(init<>())
        .def("write",NXAttribute_write_ui8)
        .def("write",NXAttribute_write_i8)
        .def("write",NXAttribute_write_ui16)
        .def("write",NXAttribute_write_i16)
        .def("write",NXAttribute_write_ui32)
        .def("write",NXAttribute_write_i32)
        .def("write",NXAttribute_write_ui64)
        .def("write",NXAttribute_write_i64)
        .def("write",NXAttribute_write_f32)
        .def("write",NXAttribute_write_f64)
        .def("write",NXAttribute_write_f128)
        .def("write",NXAttribute_write_str)
        .def("write",NXAttribute_write_scalar_ui8)
        .def("write",NXAttribute_write_scalar_i8)
        .def("write",NXAttribute_write_scalar_ui16)
        .def("write",NXAttribute_write_scalar_i16)
        .def("write",NXAttribute_write_scalar_ui32)
        .def("write",NXAttribute_write_scalar_i32)
        .def("write",NXAttribute_write_scalar_ui64)
        .def("write",NXAttribute_write_scalar_i64)
        .def("write",NXAttribute_write_scalar_f32)
        .def("write",NXAttribute_write_scalar_f64)
        .def("write",NXAttribute_write_scalar_f128)
        .def("write",NXAttribute_write_array_ui8)
        .def("write",NXAttribute_write_array_i8)
        .def("write",NXAttribute_write_array_ui16)
        .def("write",NXAttribute_write_array_i16)
        .def("write",NXAttribute_write_array_ui32)
        .def("write",NXAttribute_write_array_i32)
        .def("write",NXAttribute_write_array_ui64)
        .def("write",NXAttribute_write_array_i64)
        .def("write",NXAttribute_write_array_f32)
        .def("write",NXAttribute_write_array_f64)
        .def("write",NXAttribute_write_array_f128)
        .def("read",NXAttribute_read_ui8)
        .def("read",NXAttribute_read_i8)
        .def("read",NXAttribute_read_ui16)
        .def("read",NXAttribute_read_i16)
        .def("read",NXAttribute_read_ui32)
        .def("read",NXAttribute_read_i32)
        .def("read",NXAttribute_read_ui64)
        .def("read",NXAttribute_read_i64)
        .def("read",NXAttribute_read_f32)
        .def("read",NXAttribute_read_f64)
        .def("read",NXAttribute_read_f128)
        .def("read",NXAttribute_read_str)
        .def("read",NXAttribute_read_scalar_ui8)
        .def("read",NXAttribute_read_scalar_i8)
        .def("read",NXAttribute_read_scalar_ui16)
        .def("read",NXAttribute_read_scalar_i16)
        .def("read",NXAttribute_read_scalar_ui32)
        .def("read",NXAttribute_read_scalar_i32)
        .def("read",NXAttribute_read_scalar_ui64)
        .def("read",NXAttribute_read_scalar_i64)
        .def("read",NXAttribute_read_scalar_f32)
        .def("read",NXAttribute_read_scalar_f64)
        .def("read",NXAttribute_read_scalar_f128)
        .def("read",NXAttribute_read_array_ui8)
        .def("read",NXAttribute_read_array_i8)
        .def("read",NXAttribute_read_array_ui16)
        .def("read",NXAttribute_read_array_i16)
        .def("read",NXAttribute_read_array_ui32)
        .def("read",NXAttribute_read_array_i32)
        .def("read",NXAttribute_read_array_ui64)
        .def("read",NXAttribute_read_array_i64)
        .def("read",NXAttribute_read_array_f32)
        .def("read",NXAttribute_read_array_f64)
        .def("read",NXAttribute_read_array_f128)
        .add_property("shape",&AType::shape)
        .add_property("type_id",&AType::type_id)
        .def("close",&AType::close)
        ;

}

#endif
