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

#include <boost/python.hpp>
#include <h5cpp/hdf5.hpp>


BOOST_PYTHON_MODULE(_filter)
{
  using namespace boost::python;
  using namespace hdf5::filter;

  enum_<Availability>("Availability")
      .value("MANDATORY",Availability::MANDATORY)
      .value("OPTIONAL",Availability::OPTIONAL);

  class_<Filter,boost::noncopyable>("Filter",no_init)
      .add_property("id",&Filter::id)
      .def("__call__",&Filter::operator(),(args("dcpl"),args("availability")=Availability::MANDATORY))
          ;

  class_<Fletcher32,bases<Filter>>("Fletcher32");

  void (Deflate::*set_level)(unsigned int) = &Deflate::level;
  unsigned int(Deflate::*get_level)() const = &Deflate::level;
  class_<Deflate,bases<Filter>>("Deflate")
      .def(init<unsigned int>((arg("level")=0)))
      .add_property("level",get_level,set_level)
          ;

  class_<Shuffle,bases<Filter>>("Shuffle");
}
