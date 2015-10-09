//
// (c) Copyright 2015 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
//
// This file is part of python-pniio.
//
// python-pniio is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pniio is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Oct 8, 2015
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include "io_operations.hpp"


boost::python::object get_first_element(const boost::python::object &o)
{
    using namespace pni::core;
    using namespace boost::python;

    typedef std::vector<npy_intp> shape_type;
    auto index = numpy::get_shape<shape_type>(o);

    //set index to first element
    std::fill(index.begin(),index.end(),0);
   
    PyArrayObject *arr_ptr = reinterpret_cast<PyArrayObject*>(o.ptr());
    const char *data = reinterpret_cast<const char*>(PyArray_GetPtr(arr_ptr,index.data()));
    PyObject* ptr = PyArray_GETITEM(arr_ptr,data);
    Py_INCREF(ptr);
    handle<> h(ptr);
    return object(h);
}
