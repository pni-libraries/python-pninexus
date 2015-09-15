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
// Created on: Aug 12, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

extern "C"{
#include <Python.h>
}

#include "element_dict_converter.hpp"

using namespace pni::core;
using namespace boost::python;
using namespace pni::io::nx;

//converter namespace
namespace convns = boost::python::converter; 

//----------------------------------------------------------------------------
nxpath_element_to_dict_converter::nxpath_element_to_dict_converter()
{
    to_python_converter<nxpath::element_type,nxpath_element_to_dict_converter>();
}

//----------------------------------------------------------------------------
PyObject *nxpath_element_to_dict_converter::convert(const nxpath::element_type &e)
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
    Py_XINCREF(obj_ptr);
    handle<> h(obj_ptr);
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
void dict_to_nxpath_element_converter::construct(PyObject *obj_ptr,
                                                 rvalue_type *data)
{
    using namespace pni::core;

    //build the key objects 
#if PY_MAJOR_VERSION >= 3
    PyObject *name_key = PyUnicode_FromString("name");
    PyObject *base_key = PyUnicode_FromString("base_class");
#else
    PyObject *name_key = PyString_FromString("name");
    PyObject *base_key = PyString_FromString("base_class");
#endif

    //retrieve the content of the dictionary 
    PyObject *name_item = PyDict_GetItem(obj_ptr,name_key);
    PyObject *base_item = PyDict_GetItem(obj_ptr,base_key);

#if PY_MAJOR_VERSION >= 3
    string name(PyUnicode_AS_DATA(name_item));
    string base_class(PyUnicode_AS_DATA(base_item));
#else
    string name;
    string base_class; 
    
    if(PyString_Size(name_item))
        name = string(PyString_AsString(name_item));
    else
        name = string();

    if(PyString_Size(base_item))
        base_class = string(PyString_AsString(base_item));
    else
        base_class = string();
#endif

    void *storage = ((storage_type*)data)->storage.bytes;
    new (storage) element_type(name,base_class);
    data->convertible = storage;

    //remove all temporary python objects
    Py_DECREF(name_key);
    Py_DECREF(base_key);
}
