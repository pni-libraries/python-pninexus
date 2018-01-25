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
#include <iostream>
#include <sstream>

#include <pni/io/nexus.hpp>
#include <pni/io/exceptions.hpp>


//import here the namespace for the nxh5 module
using namespace boost::python;

#include "nexus/boost_filesystem_path_conversion.hpp"
#include "nexus/iterator_wrapper.hpp"
#include "nexus/dimensions_conversion.hpp"
#include "nexus/wrappers.hpp"


#if PY_MAJOR_VERSION >= 3
int
#else 
void
#endif
init_numpy()
{
    import_array();
}


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

  // ======================================================================
  // Register object converters
  // ======================================================================
  BoostFilesystemPathToPythonObject();
  PythonObjectToBoostFilesystemPath();
  DimensionsToTuple();
  PythonToDimensions();

  wrap_iterator<NodeIteratorWrapper>("NodeIterator");
  wrap_iterator<RecursiveNodeIteratorWrapper>("RecursiveNodeIterator");
  wrap_iterator<LinkIteratorWrapper>("LinkIterator");
  wrap_iterator<RecursiveLinkIteratorWrapper>("RecursiveLinkIterator");
  wrap_iterator<AttributeIteratorWrapper>("AttributeIterator");

  wrap_attribute();
  wrap_nodes();
  wrap_dataspace();
  wrap_file();
  wrap_link();

  exception_registration();


}
