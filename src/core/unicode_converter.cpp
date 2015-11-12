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
// Created on: Nov 11, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include "unicode_converter.hpp"
#include "utils.hpp"

using namespace pni::core;
using namespace boost::python;

//converter namespace
namespace convns = boost::python::converter; 

//----------------------------------------------------------------------------
unicode_to_string_converter::unicode_to_string_converter()
{
    convns::registry::push_back(
    &convertible,
    &construct,
    type_id<string>()
            );
}

//----------------------------------------------------------------------------
void* unicode_to_string_converter::convertible(PyObject *obj_ptr)
{
    handle<> h(borrowed(obj_ptr));
    object o(h);
    if(!is_unicode(o)) return nullptr;
    return obj_ptr;
}
   
//----------------------------------------------------------------------------
void unicode_to_string_converter::construct(PyObject *obj_ptr,
                                           rvalue_type *data)
{
    handle<> h(PyUnicode_AsUTF8String(obj_ptr));
    object str_obj(h);
    string value = extract<string>(str_obj);

    void *storage = ((storage_type*)data)->storage.bytes;
    
    new (storage) string(value);
    data->convertible = storage;
}
