//
// (c) Copyright 2018 DESY
//
// This file is part of python-pni.
//
// python-pni is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// python-pni is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with python-pniio.  If not, see <http://www.gnu.org/licenses/>.
// ===========================================================================
//
// Created on: Jan 31, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#pragma once

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include "numpy.hpp"
#include "hdf5_numpy.hpp"
#include "hdf5_numpy_variable_strings.hpp"



namespace io {

template<typename IoType>
void write(const IoType &instance,const numpy::ArrayAdapter &array)
{
//  if(array.type_id() == pni::core::type_id_t::STRING)
//    instance.write(numpy::to_string_vector(array));
//  else
    instance.write(array);
}

template<typename IoType>
void write(const IoType &instance,const boost::python::object &object)
{
  using namespace boost::python;

  if(numpy::is_array(object))
  {
    write(instance,numpy::ArrayAdapter(object));
  }
  else
  {
    boost::python::object temp_array = numpy::to_numpy_array(object);
    write(instance,numpy::ArrayAdapter(temp_array));
  }
}

template<typename IoType>
void write(const IoType &instance,const numpy::ArrayAdapter &array,
           const hdf5::dataspace::Selection &selection)
{
  if(array.type_id() == pni::core::type_id_t::STRING)
    instance.write(numpy::to_string_vector(array),selection);
  else
    instance.write(array,selection);

}

template<typename IoType>
void write(const IoType &instance,const boost::python::object &object,
           const hdf5::dataspace::Selection &selection)
{
  using namespace boost::python;

  if(numpy::is_array(object))
  {
    write(instance,numpy::ArrayAdapter(object),selection);
  }
  else
  {
    boost::python::object temp_array = numpy::to_numpy_array(object);
    write(instance,numpy::ArrayAdapter(temp_array),selection);
  }
}



template<typename IoType>
boost::python::object read(const IoType &instance)
{
  using namespace boost::python;
  using namespace pni::io;

  if(instance.datatype().get_class() == hdf5::datatype::Class::STRING)
  {
    std::vector<std::string> buffer(instance.dataspace().size());
    instance.read(buffer);

    //copy the content to a list
    list l;
    std::for_each(buffer.begin(),buffer.end(),
                  [&l](const std::string &s) { l.append(s);});

    return numpy::ArrayFactory::create(l,pni::core::type_id_t::STRING,
                                       nexus::get_dimensions(instance));
  }
  else
  {
    object array = numpy::ArrayFactory::create(nexus::get_type_id(instance),
                                       nexus::get_dimensions(instance));
    numpy::ArrayAdapter adapter(array);
    instance.read(adapter);
    return array;
  }
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
  using namespace pni::io;

  hdf5::Dimensions output_dims = numpy::get_dimensions(selection);

  if(nexus::get_type_id(instance) == pni::core::type_id_t::STRING)
  {
    std::vector<std::string> buffer(instance.dataspace().size());
    instance.read(buffer,selection);

    //copy the content to a list
    list l;
    std::for_each(buffer.begin(),buffer.end(),
                  [&l](const std::string &s) { l.append(s);});

    return numpy::ArrayFactory::create(l,pni::core::type_id_t::STRING,output_dims);
  }
  else
  {
    object array = numpy::ArrayFactory::create(nexus::get_type_id(instance),output_dims);
    numpy::ArrayAdapter adapter(array);
    instance.read(adapter,selection);
    return array;
  }
}





} // namespace io
