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
// Created on: Jan 25, 2018
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>
#include "wrappers.hpp"

static const hsize_t unlimited = H5S_UNLIMITED;


BOOST_PYTHON_MODULE(_dataspace)
{
  using namespace boost::python;
  using namespace hdf5::dataspace;

  //
  // setting up the documentation options
  //
  docstring_options doc_opts;
  doc_opts.disable_signatures();
  doc_opts.enable_user_defined();

  enum_<Type>("Type")
      .value("SCALAR",hdf5::dataspace::Type::Scalar)
      .value("SIMPLE",hdf5::dataspace::Type::Simple)
      ;

  class_<SelectionManager,boost::noncopyable>("SelectionManager",no_init)
      .add_property("size",&SelectionManager::size)
      .add_property("type",&SelectionManager::type)
      .def("all",&SelectionManager::all)
      .def("none",&SelectionManager::none)
      .def("__call__",&SelectionManager::operator())
      ;


  class_<Dataspace>("Dataspace")
      .add_property("is_valid",&Dataspace::is_valid)
      .add_property("size",&Dataspace::size)
      .add_property("type",&Dataspace::type)
      .def_readonly("selection",&Dataspace::selection)
      ;

  class_<Simple,bases<Dataspace>>("Simple")
      .def(init<const Dataspace&>())
      .def(init<hdf5::Dimensions>())
      .def(init<hdf5::Dimensions,hdf5::Dimensions>())
      .add_property("rank",&Simple::rank)
      .def("dimensions",&Simple::dimensions,(arg("current"),arg("maximum")))
      .add_property("current_dimensions",&Simple::current_dimensions)
      .add_property("maximum_dimensions",&Simple::maximum_dimensions)
      ;

  class_<Scalar,bases<Dataspace>>("Scalar")
      .def(init<Dataspace>())
      ;

  scope current;
  current.attr("UNLIMITED") = unlimited;

  create_selections();
}
