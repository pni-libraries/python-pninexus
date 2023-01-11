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

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>
#include <pni/nexus.hpp>

using namespace boost::python;
using namespace pni;

namespace {

hdf5::node::Dataset create_without_chunk(const hdf5::node::Group &parent,
                                         const hdf5::Path &path,
                                         const hdf5::datatype::Datatype &datatype,
                                         const hdf5::dataspace::Dataspace &dataspace,
                                         const hdf5::property::LinkCreationList &lcpl,
                                         const hdf5::property::DatasetCreationList &dcpl,
                                         const hdf5::property::DatasetAccessList &dapl)
{
  return nexus::FieldFactory::create(parent,path,datatype,dataspace,lcpl,dcpl,dapl);
}

hdf5::node::Dataset create_with_chunk( const hdf5::node::Group &parent,
                                       const hdf5::Path &path,
                                       const hdf5::datatype::Datatype &datatype,
                                       const hdf5::dataspace::Simple &dataspace,
                                       const hdf5::Dimensions &chunk,
                                       const hdf5::property::LinkCreationList &lcpl,
                                       const hdf5::property::DatasetCreationList &dcpl,
                                       const hdf5::property::DatasetAccessList &dapl)
{
  return nexus::FieldFactory::create(parent,path,datatype,dataspace,chunk,lcpl,dcpl,dapl);
}

}




void create_factory_wrappers()
{
  //
  // wrapping factory classes
  //
  class_<nexus::BaseClassFactory>("BaseClassFactory")
      .def("create_",&nexus::BaseClassFactory::create,
           (arg("parent"),
            arg("path"),
            arg("base_class"),
            arg("lcpl")=hdf5::property::LinkCreationList(),
            arg("gcpl")=hdf5::property::GroupCreationList(),
            arg("gapl")=hdf5::property::GroupAccessList()))
      .staticmethod("create_")
      ;

  class_<pni::nexus::FieldFactory>("FieldFactory")
      .def("create_",create_without_chunk,
           (arg("parent"),arg("path"),arg("type"),arg("space"),
            arg("lcpl") = hdf5::property::LinkCreationList(),
            arg("dcpl") = hdf5::property::DatasetCreationList(),
            arg("dapl") = hdf5::property::DatasetAccessList())
           )
      .staticmethod("create_")
      .def("create_chunked_",create_with_chunk,
           (arg("parent"),arg("path"),arg("type"),arg("space"),arg("chunk"),
            arg("lcpl") = hdf5::property::LinkCreationList(),
            arg("dcpl") = hdf5::property::DatasetCreationList(),
            arg("dapl") = hdf5::property::DatasetAccessList())
      )
      .staticmethod("create_chunked_")
      ;
}
