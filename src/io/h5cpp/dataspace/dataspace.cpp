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
// Created on: Jan 25, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>

static const hsize_t UNLIMITED = H5S_UNLIMITED;


BOOST_PYTHON_MODULE(_dataspace)
{
  using namespace boost::python;
  using namespace hdf5::dataspace;

  enum_<Type>("Type")
      .value("SCALAR",hdf5::dataspace::Type::SCALAR)
      .value("SIMPLE",hdf5::dataspace::Type::SIMPLE)
      ;


  class_<Dataspace>("Dataspace")
      .add_property("is_valid",&Dataspace::is_valid)
      .add_property("size",&Dataspace::size)
      .add_property("type",&Dataspace::type)
      ;

  class_<Simple,bases<Dataspace>>("Simple")
      .def(init<const Dataspace&>())
      .def(init<hdf5::Dimensions>())
      .def(init<hdf5::Dimensions,hdf5::Dimensions>())
      .add_property("rank",&Simple::rank)
      .def("dimensions",&Simple::dimensions)
      .add_property("current_dimensions",&Simple::current_dimensions)
      .add_property("maximum_dimensions",&Simple::maximum_dimensions)
      ;

  class_<Scalar,bases<Dataspace>>("Scalar")
      .def(init<Dataspace>())
      ;

  scope current;
  current.attr("UNLIMITED") = UNLIMITED;
}
