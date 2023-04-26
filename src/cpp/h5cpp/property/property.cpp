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
// Created on: Jan 26, 2018
//     Authors:
//             Eugen Wintersberger <eugen.wintersberger@desy.de>
//             Jan Kotanski <jan.kotanski@desy.de>
//
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>

#include "wrapper_generators.hpp"


#if H5_VERSION_GE(1, 10, 0)
void add_map(hdf5::property::VirtualDataMaps &self,
	     hdf5::property::VirtualDataMap vdm)
    {
        self.push_back(vdm);
    }

#endif

BOOST_PYTHON_MODULE(_property)
{
  using namespace boost::python;
  using namespace hdf5;
  using namespace hdf5::property;
  using namespace hdf5::dataspace;
  //
  // setting up the documentation options
  //
  docstring_options doc_opts;
  doc_opts.disable_signatures();
  doc_opts.enable_user_defined();

  create_enumeration_wrappers();
  create_class_wrappers();
  create_copyflag_wrapper();
  create_chunk_cache_parameters_wrapper();
  create_creation_order_wrapper();

#if H5_VERSION_GE(1, 10, 0)
  class_<VirtualDataMap>("VirtualDataMap")
    .def(init<View,fs::path,Path,View>((arg("target_view"),arg("source_file"),arg("source_dataset"),arg("source_view"))))
    ;
  class_<VirtualDataMaps>("VirtualDataMaps") 
    .def("add", add_map)
   ;
  
#endif
}
