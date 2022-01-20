//
// (c) Copyright 2018 DESY
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
// Created on: Jan 31, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>
#include "../numpy/numpy.hpp"


//!
//! @brief returns true if an IO object has a variable length string type
//!
//! This predicate function template works with attributes and datasets.
//!
//! @param object reference to the object for which to determine the type
//! @return true if variable length string type, false otherwise
//!
template<typename IoType>
bool has_variable_length_string_type(const IoType &object)
{
  if(object.datatype().get_class() != hdf5::datatype::Class::String)
    return false;

  hdf5::datatype::String string_type = object.datatype();

  return string_type.is_variable_length();
}

template<typename IoType>
void write(const IoType &instance,const boost::python::object &object)
{
  boost::python::object temp_array;
  numpy::ArrayAdapter array_adapter;

  if(numpy::is_array(object))
  {
    array_adapter = numpy::ArrayAdapter(object);
  }
  else
  {
    boost::python::object temp_array = numpy::ArrayFactory::create(object);
    array_adapter = numpy::ArrayAdapter(temp_array);
  }

  hdf5::datatype::Datatype memory_type = hdf5::datatype::create<numpy::ArrayAdapter>(array_adapter);
  hdf5::dataspace::Dataspace memory_space = hdf5::dataspace::create<numpy::ArrayAdapter>(array_adapter);
  if(has_variable_length_string_type(instance))
  {
    memory_type = hdf5::datatype::String::variable();
  }

  instance.write(array_adapter,memory_type,memory_space,instance.dataspace());
}

template<typename IoType>
void write(const IoType &instance,const boost::python::object &object,
           const hdf5::dataspace::Selection &selection)
{
  using namespace boost::python;

  if(numpy::is_array(object))
  {
    instance.write(numpy::ArrayAdapter(object),selection);
  }
  else
  {
    boost::python::object temp_array = numpy::ArrayFactory::create(object);
    instance.write(numpy::ArrayAdapter(temp_array),selection);
  }
}



template<typename IoType>
boost::python::object read(const IoType &instance)
{
  using namespace boost::python;

  object array = numpy::ArrayFactory::create(instance.datatype(),
                                             instance.dataspace());
  numpy::ArrayAdapter adapter(array);
  instance.read(adapter);

  //
  // if we read string data we have to fix the shape of the resulting
  // numpy array.
  //
  if(instance.datatype().get_class()==hdf5::datatype::Class::String)
  {
    PyArrayObject *array_ptr = static_cast<PyArrayObject*>(adapter);
    numpy::Dimensions dims{1};

    if(instance.dataspace().type()==hdf5::dataspace::Type::Simple)
      dims = numpy::Dimensions(hdf5::dataspace::Simple(instance.dataspace()).current_dimensions());

    PyArray_Dims py_dims{dims.data(),dims.size()};

    array = object(handle<>(PyArray_Newshape(array_ptr,&py_dims,NPY_CORDER)));
  }
  return array;
}

//!
//! @brief perform an inplace read operation
//!
//! Currently inplace read operations are only supported for numpy arrays.
//! The arrays datatype and dataspace must match the dataset or attribute to
//! read from.
//!
//! The general benefit of inplace read operations is the reduced overhad
//! for creating a new numpy array during every read operation.
//!
//! @param instance reference to the dataset or attribute to read from
//! @param object reference to the python object to read to
//!
template<typename IoType>
void read(const IoType &instance,const boost::python::object &object)
{
  if(!numpy::is_array(object))
    throw std::runtime_error("Inplace reading is only supported for numpy arrays!");

  numpy::ArrayAdapter array_adapter(object);
  instance.read(array_adapter);
}

//!
//! @brief read with selection
//!
//! Read data from a dataset using a selection and return a new numpy array
//! with the data. The major disadvantage of this function is that a new
//! numpy array must be created any time we read data. This can be an expensive
//! operation in the case of larger data like image frames.
//!
//! @param instance reference to the dataset from which to read
//! @param selection reference to the selection
//! @return new numpy array with the obtained data
//!
template<typename IoType>
boost::python::object read(const IoType &instance,const hdf5::dataspace::Selection &selection)
{
  using namespace boost::python;

  object array = numpy::ArrayFactory::create(instance.datatype(),selection);

  if(instance.datatype().get_class() == hdf5::datatype::Class::String)
  {
//    std::vector<std::string> buffer(instance.dataspace().size());
//    instance.read(buffer,selection);
//
//    //copy the content to a list
//    list l;
//    std::for_each(buffer.begin(),buffer.end(),
//                  [&l](const std::string &s) { l.append(s);});
//
//    return numpy::ArrayFactory::create(l,pni::type_id_t::String,output_dims);
  }
  else
  {
    numpy::ArrayAdapter adapter(array);
    instance.read(adapter,selection);
    return array;
  }
}
