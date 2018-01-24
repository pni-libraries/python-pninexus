//
// (c) Copyright 2014 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
// Created on: Oct 31, 2014
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//

#include "utils.hpp"
#include <pni/io/nexus.hpp>

using namespace boost::python;
using namespace pni::core;


hdf5::datatype::Datatype create_datatype(const std::string &type_code)
{
  pni::core::type_id_t type_id;
  try
  {
    type_id = pni::core::type_id_from_str(type_code);
  }
  catch(pni::core::key_error &error)
  {
    //forward exception
    error.append(EXCEPTION_RECORD);
    throw error;
  }
  return pni::io::nexus::DatatypeFactory::create(type_id);
}

tuple get_tuple_from_args(const object &args)
{
    if(PyTuple_Check(args.ptr())) 
        return tuple(args);
    else
        return make_tuple(args);
}

//----------------------------------------------------------------------------
bool is_ellipsis(const object &o)
{
    return o.ptr()==(PyObject*)Py_Ellipsis;
}

//----------------------------------------------------------------------------
bool is_slice(const object &o)
{
    return PySlice_Check(o.ptr());
}

//----------------------------------------------------------------------------
ssize_t get_index(ssize_t python_index,ssize_t n_elements)
{
    if(python_index<0)
        return n_elements+python_index;
    else
        return python_index;
}

//----------------------------------------------------------------------------
shapes_type get_shapes(const object &s,const object &c)
{
    shape_t shape,chunk;

    if(s.is_none())
    {
        shape = shape_t{1};
        chunk = shape;
    }
    else
    {
        shape = List2Container<shape_t>(list(s));
        chunk = shape;
        
        if(c.is_none()) 
            chunk.front() = 1;
        else
            chunk = List2Container<shape_t>(list(c));
    }
    
    return shapes_type{shape,chunk};
}
