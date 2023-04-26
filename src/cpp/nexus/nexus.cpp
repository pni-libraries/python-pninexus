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
//  Created on: Feb 8, 2018
//      Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
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
#include <h5cpp/contrib/stl/stl.hpp>
#include <pni/nexus.hpp>

#include "wrappers.hpp"
#include "list_converters.hpp"


//import here the namespace for the nxh5 module
using namespace boost::python;
using namespace pni;

#if PY_MAJOR_VERSION >= 3
static void * init_numpy()
{
    import_array();
    return NULL;
}
#else 
static void init_numpy()
{
    import_array();
}
#endif



//=================implementation of the python extension======================
BOOST_PYTHON_MODULE(_nexus)
{
  init_numpy();

  //
  // setting up the documentation options
  //
  docstring_options doc_opts;
  doc_opts.disable_signatures();
  doc_opts.enable_user_defined();

  //
  //
  //
  def("is_nexus_file",&pni::nexus::is_nexus_file);

  hdf5::file::File (*create_file_flag)(const fs::path &,
                                       hdf5::file::AccessFlagsBase,
                                       const hdf5::property::FileCreationList &,
                                       const hdf5::property::FileAccessList &) = &pni::nexus::create_file;
  def("create_file",create_file_flag,(arg("path"),
                                      arg("flags")=hdf5::file::AccessFlags::Exclusive,
                                      arg("fcpl")=hdf5::property::FileCreationList(),
                                      arg("fapl")=hdf5::property::FileAccessList()));

  hdf5::file::File (*open_file_flag)(const fs::path &,
                                     hdf5::file::AccessFlagsBase,
                                     const hdf5::property::FileAccessList &) = &pni::nexus::open_file;
  def("open_file",open_file_flag,(arg("path"),
                                  arg("flags")=hdf5::file::AccessFlags::ReadOnly,
                                  arg("fapl")=hdf5::property::FileAccessList()));


  create_factory_wrappers();
  create_predicate_wrappers();
  create_path_wrappers();

  def("create_from_file",&nexus::xml::create_from_file,(arg("parent"),arg("path")));
  def("create_from_string",&nexus::xml::create_from_string,(arg("parent"),arg("xml")));

  GroupListToTuple();
  DatasetListToTuple();
  AttributeListToTuple();
  NodeListToTuple();
  PathObjectListToTuple();

}
