//
// (c) Copyright 2015 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
//
// This file is part of python-pnicore.
//
// python-pnicore is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pnicore is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pnicore.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
//  Created on: Mon 27, 2015
//      Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <boost/python.hpp>


#define CHECK_TYPE_ID_FROM_OBJECT_NAME(type) check_type_id ## type ## _from_object

#define CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(type) bool check_type_id_ ## type ## _from_object(const boost::python::object &o)

CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(uint8);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(int8);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(uint16);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(int16);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(uint32);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(int32);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(uint64);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(int64);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(float32);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(float64);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(float128);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(complex32);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(complex64);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(complex128);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(string);
CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(bool_t);


#define CHECK_TYPE_ID_FROM_OBJECT_EXPOSE() \
    def("check_type_id_uint8_from_object",check_type_id_uint8_from_object); \
    def("check_type_id_int8_from_object",check_type_id_int8_from_object); \
    def("check_type_id_uint16_from_object",check_type_id_uint16_from_object); \
    def("check_type_id_int16_from_object",check_type_id_int16_from_object); \
    def("check_type_id_uint32_from_object",check_type_id_uint32_from_object); \
    def("check_type_id_int32_from_object",check_type_id_int32_from_object); \
    def("check_type_id_uint64_from_object",check_type_id_uint64_from_object); \
    def("check_type_id_int64_from_object",check_type_id_int64_from_object); \
    def("check_type_id_float32_from_object",check_type_id_float32_from_object); \
    def("check_type_id_float64_from_object",check_type_id_float64_from_object); \
    def("check_type_id_float128_from_object",check_type_id_float128_from_object);\
    def("check_type_id_complex32_from_object",check_type_id_complex32_from_object); \
    def("check_type_id_complex64_from_object",check_type_id_complex64_from_object); \
    def("check_type_id_complex128_from_object",check_type_id_complex128_from_object);\
    def("check_type_id_string_from_object",check_type_id_string_from_object); \
    def("check_type_id_bool_from_object",check_type_id_bool_t_from_object)






