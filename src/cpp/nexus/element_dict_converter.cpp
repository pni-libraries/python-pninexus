//
// (c) Copyright 2015 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
//
// This file is part of python-pninexus.
//
// python-pninexus is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pninexus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pninexus.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Aug 12, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

extern "C"{
#include <Python.h>
}

#include "element_dict_converter.hpp"

using namespace pni;
using namespace boost::python;
using namespace pni;

//converter namespace
namespace convns = boost::python::converter; 

//----------------------------------------------------------------------------
nxpath_element_to_dict_converter::nxpath_element_to_dict_converter()
{
    to_python_converter<nexus::Path::Element,nxpath_element_to_dict_converter>();
}

//----------------------------------------------------------------------------
PyObject *nxpath_element_to_dict_converter::convert(const nexus::Path::Element &e)
{
    dict d;
    d["name"] = e.first;
    d["base_class"] = e.second;
    return incref(object(d).ptr());
}

//----------------------------------------------------------------------------
dict_to_nxpath_element_converter::dict_to_nxpath_element_converter()
{
    convns::registry::push_back(&convertible, 
                                &construct, 
                                type_id<element_type>());
}

//----------------------------------------------------------------------------
void* dict_to_nxpath_element_converter::convertible(PyObject *obj_ptr)
{
    //if the object is no dictonary we can already stop here
    if(!PyDict_Check(obj_ptr)) return nullptr;

    //check the elements of the dictionary
    handle<> h(borrowed(obj_ptr));
    dict d(h);
    if((!d.has_key("name")) || (!d.has_key("base_class")))
        return nullptr;


    //check value types - must both be strings
#if PY_MAJOR_VERSION >= 3
    if(!PyUnicode_Check(object(d["name"]).ptr())) 
#else
    if(!PyString_Check(object(d["name"]).ptr())) 
#endif
        return nullptr;

#if PY_MAJOR_VERSION >= 3
    if(!PyUnicode_Check(object(d["base_class"]).ptr())) 
#else
    if(!PyString_Check(object(d["base_class"]).ptr())) 
#endif
        return nullptr;

    return obj_ptr;
}

//----------------------------------------------------------------------------
string get_string_from_pyobject(PyObject *ptr)
{
#if PY_MAJOR_VERSION >= 3
    Py_ssize_t str_size = PyUnicode_GET_DATA_SIZE(ptr);
    if(str_size)
    {
        PyObject *utf8_str = PyUnicode_AsUTF8String(ptr);
        string result = PyBytes_AsString(utf8_str);
        Py_XDECREF(utf8_str);
        return result;
    }
#else
    if(PyString_Size(ptr))
        return string(PyString_AsString(ptr));
#endif
    else
        return string();
}

//----------------------------------------------------------------------------
PyObject *get_pyobject_from_string(const string &s)
{
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_FromString(s.c_str());
#else 
    return PyString_FromString(s.c_str());
#endif
}
   
//----------------------------------------------------------------------------
void dict_to_nxpath_element_converter::construct(PyObject *obj_ptr,
                                                 rvalue_type *data)
{
    dict d(handle<>(borrowed(obj_ptr)));
    //build the key objects 

    string name = extract<string>(d["name"]);
    string base_class = extract<string>(d["base_class"]);

    void *storage = ((storage_type*)data)->storage.bytes;
    new (storage) element_type(name,base_class);
    data->convertible = storage;
}
