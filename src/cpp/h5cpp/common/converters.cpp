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
// Created on: Jan 31, 2018
//     Authors:
//             Eugen Wintersberger <eugen.wintersberger@desy.de>
//             Jan Kotanski <jan.kotanski@desy.de>
//

#include "converters.hpp"

boost::python::object convert_datatype(const hdf5::datatype::Datatype &datatype)
{
  using namespace hdf5::datatype;

  switch(datatype.get_class())
  {
    case Class::Integer:
      return boost::python::object(Integer(datatype));
    case Class::Float:
      return boost::python::object(Float(datatype));
    case Class::Compound:
      return boost::python::object(Compound(datatype));
    case Class::String:
      return boost::python::object(String(datatype));
    default:
      return boost::python::object(datatype);
  }
}

boost::python::object convert_dataspace(const hdf5::dataspace::Dataspace &dataspace)
{
  using namespace hdf5::dataspace;

  switch(dataspace.type())
  {
    case Type::Scalar:
      return boost::python::object(Scalar(dataspace));
    case Type::Simple:
      return boost::python::object(Simple(dataspace));
    default:
      return boost::python::object(dataspace);
  }
}
