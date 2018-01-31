//
// (c) Copyright 2011 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
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
//  Created on: Jan 5, 2012
//      Author: Eugen Wintersberger
//


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PNI_CORE_USYMBOL
extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include <boost/python.hpp>
#include <boost/python/docstring_options.hpp>
#include <h5cpp/hdf5.hpp>


//import here the namespace for the nxh5 module
using namespace boost::python;

#include "boost_filesystem_path_conversion.hpp"
#include "dimensions_conversion.hpp"
#include "errors.hpp"


#if PY_MAJOR_VERSION >= 3
int
#else 
void
#endif
init_numpy()
{
    import_array();
}

std::string path_to_string(const hdf5::Path &self)
{
  return std::string(self);
}


//=================implementation of the python extension======================
BOOST_PYTHON_MODULE(_h5cpp)
{
  init_numpy();

  //
  // setting up the documentation options
  //
  docstring_options doc_opts;
  doc_opts.disable_signatures();
  doc_opts.enable_user_defined();

  // ======================================================================
  // Register object converters
  // ======================================================================
  BoostFilesystemPathToPythonObject();
  PythonObjectToBoostFilesystemPath();
  DimensionsToTuple();
  PythonToDimensions();

  enum_<hdf5::IterationOrder>("IterationOrder")
      .value("INCREASING",hdf5::IterationOrder::INCREASING)
      .value("DECREASING",hdf5::IterationOrder::DECREASING)
      .value("NATIVE",hdf5::IterationOrder::NATIVE);

  enum_<hdf5::IterationIndex>("IterationIndex")
      .value("NAME",hdf5::IterationIndex::NAME)
      .value("CREATION_ORDER",hdf5::IterationIndex::CREATION_ORDER);


  hdf5::IterationOrder (hdf5::IteratorConfig::*get_iteration_order)() const =  &hdf5::IteratorConfig::order;
  void (hdf5::IteratorConfig::*set_iteration_order)(hdf5::IterationOrder) = &hdf5::IteratorConfig::order;
  hdf5::IterationIndex (hdf5::IteratorConfig::*get_iteration_index)() const = &hdf5::IteratorConfig::index;
  void (hdf5::IteratorConfig::*set_iteration_index)(hdf5::IterationIndex) = &hdf5::IteratorConfig::index;
  const hdf5::property::LinkAccessList &(hdf5::IteratorConfig::*get_link_access_list)() const =
      &hdf5::IteratorConfig::link_access_list;
  void (hdf5::IteratorConfig::*set_link_access_list)(const hdf5::property::LinkAccessList &) =
      &hdf5::IteratorConfig::link_access_list;
  class_<hdf5::IteratorConfig>("IteratorConfig")
      .add_property("order",get_iteration_order,set_iteration_order)
      .add_property("index",get_iteration_index,set_iteration_index)
      .add_property("link_access_list",make_function(get_link_access_list,return_internal_reference<>()),set_link_access_list)
      ;

  bool (hdf5::Path::*get_absolute)() const = &hdf5::Path::absolute;
  void (hdf5::Path::*set_absolute)(bool) = &hdf5::Path::absolute;
  class_<hdf5::Path>("Path")
      .def(init<std::string>())
      .add_property("name",&hdf5::Path::name)
      .add_property("size",&hdf5::Path::size)
      .add_property("parent",&hdf5::Path::parent)
      .add_property("absolute",set_absolute,get_absolute)
      .def("__str__",path_to_string)
      .def("__repr__",path_to_string)
      ;



  exception_registration();
}