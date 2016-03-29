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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PNI_CORE_USYMBOL
extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include <boost/python.hpp>
#include <vector>
#include <list>

#include <pni/core/types.hpp>
#include <core/numpy_utils.hpp>
#include <core/utils.hpp>
#include "check_type_id_from_object.hpp"
#include "check_type_str_from_object.hpp"


using namespace boost::python; 
using namespace pni::core;

#if PY_MAJOR_VERSION >= 3
int
#else 
void
#endif
init_numpy()
{
    import_array();
}

list get_shape(const object &o)
{
    auto s =  numpy::get_shape<shape_t>(o);

    return Container2List(s);
}


BOOST_PYTHON_MODULE(numpy_utils_test)
{
    init_numpy();
    def("is_array",numpy::is_array);
    def("is_scalar",numpy::is_scalar);

    CHECK_TYPE_ID_FROM_OBJECT_EXPOSE();
    CHECK_TYPE_STR_FROM_OBJECT_EXPOSE();

    def("get_shape",get_shape);
    def("get_size",numpy::get_size);
}


