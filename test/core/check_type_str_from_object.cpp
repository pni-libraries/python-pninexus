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
#include "check_type_str_from_object.hpp"

using namespace boost::python; 
using namespace pni::core;

#define CHECK_TYPE_STR_FROM_OBJECT_DEF(type,name)\
    CHECK_TYPE_STR_FROM_OBJECT_PROTOTYPE(type)\
    {\
        return numpy::type_str(o)== name;\
    }

CHECK_TYPE_STR_FROM_OBJECT_DEF(uint8,"uint8")
CHECK_TYPE_STR_FROM_OBJECT_DEF(int8,"int8")
CHECK_TYPE_STR_FROM_OBJECT_DEF(uint16,"uint16")
CHECK_TYPE_STR_FROM_OBJECT_DEF(int16,"int16")
CHECK_TYPE_STR_FROM_OBJECT_DEF(uint32,"uint32")
CHECK_TYPE_STR_FROM_OBJECT_DEF(int32,"int32")
CHECK_TYPE_STR_FROM_OBJECT_DEF(uint64,"uint64")
CHECK_TYPE_STR_FROM_OBJECT_DEF(int64,"int64")
CHECK_TYPE_STR_FROM_OBJECT_DEF(float32,"float32")
CHECK_TYPE_STR_FROM_OBJECT_DEF(float64,"float64")
CHECK_TYPE_STR_FROM_OBJECT_DEF(float128,"float128")
CHECK_TYPE_STR_FROM_OBJECT_DEF(complex32,"complex32")
CHECK_TYPE_STR_FROM_OBJECT_DEF(complex64,"complex64")
CHECK_TYPE_STR_FROM_OBJECT_DEF(complex128,"complex128")
CHECK_TYPE_STR_FROM_OBJECT_DEF(string,"string")
CHECK_TYPE_STR_FROM_OBJECT_DEF(bool_t,"bool")




