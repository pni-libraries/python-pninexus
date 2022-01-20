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
//     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
//
#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/contrib/stl/stl.hpp>

namespace {

void set_tracked(hdf5::property::CreationOrder &order,bool value)
{
  if(value)
    order.enable_tracked();
  else
    order.disable_tracked();
}

bool get_tracked(const hdf5::property::CreationOrder &order)
{
  return order.tracked();
}

void set_indexed(hdf5::property::CreationOrder &order,bool value)
{
  if(value)
    order.enable_indexed();
  else
    order.disable_indexed();
}

bool get_indexed(const hdf5::property::CreationOrder &order)
{
  return order.indexed();
}

}

void create_creation_order_wrapper()
{
  using namespace boost::python;
  using namespace hdf5::property;

  class_<CreationOrder>("CreationOrder")
      .add_property("tracked",&get_tracked,&set_tracked)
      .add_property("indexed",&get_indexed,&set_indexed)
      ;
}
