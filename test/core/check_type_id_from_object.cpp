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

#include <core/numpy_utils.hpp>
#include <pni/core/types.hpp>
#include "check_type_id_from_object.hpp"

using namespace boost::python; 
using namespace pni::core;

#define CHECK_TYPE_ID_FROM_OBJECT_DEF(type,id)\
    CHECK_TYPE_ID_FROM_OBJECT_PROTOTYPE(type)\
    {\
        return numpy::type_id(o)== id ;\
    }

CHECK_TYPE_ID_FROM_OBJECT_DEF(uint8,type_id_t::UINT8)
CHECK_TYPE_ID_FROM_OBJECT_DEF(int8,type_id_t::INT8)
CHECK_TYPE_ID_FROM_OBJECT_DEF(uint16,type_id_t::UINT16)
CHECK_TYPE_ID_FROM_OBJECT_DEF(int16,type_id_t::INT16)
CHECK_TYPE_ID_FROM_OBJECT_DEF(uint32,type_id_t::UINT32)
CHECK_TYPE_ID_FROM_OBJECT_DEF(int32,type_id_t::INT32)
CHECK_TYPE_ID_FROM_OBJECT_DEF(uint64,type_id_t::UINT64)
CHECK_TYPE_ID_FROM_OBJECT_DEF(int64,type_id_t::INT64)
CHECK_TYPE_ID_FROM_OBJECT_DEF(float32,type_id_t::FLOAT32)
CHECK_TYPE_ID_FROM_OBJECT_DEF(float64,type_id_t::FLOAT64)
CHECK_TYPE_ID_FROM_OBJECT_DEF(float128,type_id_t::FLOAT128)
CHECK_TYPE_ID_FROM_OBJECT_DEF(complex32,type_id_t::COMPLEX32)
CHECK_TYPE_ID_FROM_OBJECT_DEF(complex64,type_id_t::COMPLEX64)
CHECK_TYPE_ID_FROM_OBJECT_DEF(complex128,type_id_t::COMPLEX128)
CHECK_TYPE_ID_FROM_OBJECT_DEF(string,type_id_t::STRING)
CHECK_TYPE_ID_FROM_OBJECT_DEF(bool_t,type_id_t::BOOL)




